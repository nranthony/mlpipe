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
    """Era-aware CV plan. An artifact in its own right — comparisons must cite
    the same plan hash."""

    folds: list[tuple[np.ndarray, np.ndarray]]  # (train_idx, valid_idx)
    era_column: str
    purge_eras: int
    embargo_eras: int
    plan_hash: str


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
    def register_model(self, artifact: Artifact, name: str) -> None: ...
    def end_run(self, status: str) -> None: ...


@runtime_checkable
class ModelBackend(Protocol):
    """SageMaker-Estimator-equivalent. One adapter per model family."""

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        params: dict[str, Any],
        folds: FoldPlan | None = None,
    ) -> None: ...
    def predict(self, X: pl.DataFrame) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...


# --------------------------------------------------------------------------- core


class RunContext:
    """The only door a Step has to the outside world. Cycle 0 implements."""

    def __init__(self, config: Any, store: ArtifactStore, tracker: Tracker) -> None:
        self.config = config
        self._store = store
        self._tracker = tracker

    def get(self, key: str) -> Any:
        raise NotImplementedError("cycle 0")

    def put(self, obj: Any, key: str, *, ext: str = "parquet") -> Artifact:
        raise NotImplementedError("cycle 0")

    def log_metric(self, name: str, value: float) -> None:
        raise NotImplementedError("cycle 0")


class Step(ABC):
    """A pure function from named input artifacts to named output artifacts."""

    name: str
    inputs: list[str]
    outputs: list[str]

    def signature(self, ctx: RunContext) -> str:
        """hash(step name, code version, resolved config subtree, input hashes).
        Cycle 0 implements; concrete steps normally do not override."""
        raise NotImplementedError("cycle 0")

    @abstractmethod
    def run(self, ctx: RunContext) -> StepResult: ...
