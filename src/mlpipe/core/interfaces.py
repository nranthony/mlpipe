"""Authoritative ports for mlpipe. Cycle 0 implements against these.

Design rules (see DESIGN.md §3):
- Steps talk only to RunContext.
- Port signatures mention only our types: polars, numpy, Path, and the small
  dataclasses below. No torch/lightgbm/mlflow types may appear here.
- Keep this file dependency-light: stdlib + polars + numpy only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl

# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Artifact:
    """A persisted, immutable, content-addressed output."""

    key: str                 # logical name, e.g. "clean_table"
    content_hash: str        # sha256 of serialized bytes
    path: Path               # store/<hash[:2]>/<hash>.<ext>
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FoldPlan:
    """Grouped CV plan. An artifact in its own right — comparisons must cite the
    same plan hash. Domain semantics (time order, purge, embargo) belong to the step
    that builds the plan and travel in `meta`; consumers read only folds."""

    folds: list[tuple[np.ndarray, np.ndarray]]  # (train_idx, valid_idx)
    plan_hash: str
    group_column: str | None = None  # rows sharing a group never straddle a split
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    outputs: dict[str, Artifact]
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "success"


# --------------------------------------------------------------------------- ports


@runtime_checkable
class ArtifactStore(Protocol):
    def save(self, obj: Any, key: str, *, ext: str) -> Artifact: ...
    def load(self, content_hash: str) -> Any: ...
    def exists(self, content_hash: str) -> bool: ...


@runtime_checkable
class Tracker(Protocol):
    """Thin view over our manifests. No provenance logic lives here — losing the
    tracker must lose zero provenance."""

    def start_run(self, manifest: dict[str, Any]) -> str: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def register_model(self, artifact: Artifact, name: str) -> str | None: ...
    def set_alias(self, name: str, alias: str, version: str) -> None: ...
    def end_run(self, status: str) -> None: ...
    # Views only (ADR-0010): the store holds the canonical copy of both. Trackers
    # predating them may omit them; the pipeline and RunContext then fall back.
    def log_curve(self, name: str, values: list[float]) -> None: ...
    def log_artifact(self, artifact: Artifact) -> None: ...


@runtime_checkable
class ModelBackend(Protocol):
    """SageMaker-Estimator-equivalent. One adapter per model family. No save/load:
    the train step stores the fitted backend object itself (joblib, via the
    store), so an adapter must be picklable and needs no persistence code."""

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> None: ...
    def predict(self, X: pl.DataFrame) -> np.ndarray: ...


@runtime_checkable
class CurveBackend(ModelBackend, Protocol):
    """Optional richer port (ADR-0010). A backend with `history` also accepts an
    eval set in `fit`; the train step passes each fold's validation rows and
    stores the per-iteration history and the importances as artifacts."""

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        params: dict[str, Any],
        eval_set: tuple[pl.DataFrame, np.ndarray] | None = None,
    ) -> None: ...
    def history(self) -> dict[str, list[float]]: ...  # "<split>_<metric>" -> per iteration
    def feature_importance(self) -> dict[str, float] | None: ...


# --------------------------------------------------------------------------- core


class RunContext(ABC):
    """The only door a Step has to the outside world. This class declares the
    whole step-facing surface: `config` plus the methods below. Steps may use
    nothing else (tests/test_work0006_phase2.py enforces it); the pipeline's
    bookkeeping lives on the implementation in `mlpipe.core.context`."""

    def __init__(self, config: Any, store: ArtifactStore, tracker: Tracker) -> None:
        self.config = config  # the full, validated config; steps read their own subtree
        self._store = store
        self._tracker = tracker

    # inputs: only keys in the running step's `inputs` (they alone are hashed)
    @abstractmethod
    def get(self, key: str) -> Any:
        """Load a declared input artifact into memory."""

    @abstractmethod
    def get_lazy(self, key: str) -> pl.LazyFrame:
        """Scan a declared parquet input without materializing it."""

    @abstractmethod
    def input_hashes(self, keys: list[str]) -> dict[str, str]:
        """Content hashes of available artifacts, by key."""

    @abstractmethod
    def previous(self, step_name: str) -> dict[str, Any] | None:
        """Latest completed manifest for a step, or None on the first run."""

    # outputs
    @abstractmethod
    def put(self, obj: Any, key: str, *, ext: str = "parquet") -> Artifact:
        """Persist an output artifact in the store."""

    @abstractmethod
    def put_alias(self, key: str, source_key: str) -> Artifact:
        """Re-emit a declared input under a new key, with no new bytes."""

    # records
    @abstractmethod
    def log_metric(self, name: str, value: float) -> None:
        """Numeric metric: into the manifest and to the tracker."""

    @abstractmethod
    def log_curve(self, name: str, values: list[float]) -> None:
        """Per-iteration series, to the tracker only: the step also puts it as an
        artifact, which is the record (ADR-0010)."""

    @abstractmethod
    def log_meta(self, key: str, value: Any) -> None:
        """Non-numeric fact for the manifest."""


class Step(ABC):
    """A pure function from named input artifacts to named output artifacts."""

    name: str
    inputs: list[str]
    outputs: list[str]
    # Extra module prefixes whose source counts as this step's behaviour, beyond
    # mlpipe's steps/backends and the step's own top-level package: e.g.
    # ("shrec.",) for a sibling library a feature step calls. Data files never
    # reach a signature this way; pass them as seeded artifacts or config values.
    code_roots: tuple[str, ...] = ()

    def code_deps(self, cfg: dict[str, Any]) -> list[str]:
        """Module names this step resolves at runtime rather than importing, e.g.
        a model backend chosen by config. Statically imported modules are found
        automatically; override only for lazy lookups the parser cannot see."""
        return []

    def signature(self, ctx: RunContext) -> str:
        """hash(step name, code version, resolved config subtree, input hashes).
        Concrete steps normally do not override."""
        from mlpipe.core.signature import step_signature

        return step_signature(self, ctx.input_hashes(self.inputs), ctx.config)

    @abstractmethod
    def run(self, ctx: RunContext) -> StepResult: ...
