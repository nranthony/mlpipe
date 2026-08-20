"""RunContext implementation — the only door a Step has to the outside world —
plus the NoopTracker satisfying the Tracker port until MLflow arrives (cycle 7).
"""

from __future__ import annotations

from typing import Any

import polars as pl

from mlpipe.core import interfaces
from mlpipe.core.interfaces import Artifact


class NoopTracker:
    def start_run(self, manifest: dict[str, Any]) -> str:
        return "noop"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def register_model(self, artifact: Artifact, name: str) -> None:
        pass

    def end_run(self, status: str) -> None:
        pass


class RunContext(interfaces.RunContext):
    """Implements the port; the pipeline reads the per-step bookkeeping."""

    def __init__(self, config: Any, store: Any, tracker: Any) -> None:
        super().__init__(config, store, tracker)
        self.registry: dict[str, Artifact] = {}  # this run's artifacts, by key
        self.manifest_log: Any = None  # set by Pipeline; enables previous()
        self.reset_step()

    def reset_step(self) -> None:
        self.step_inputs: dict[str, str] = {}
        self.step_outputs: dict[str, Artifact] = {}
        self.step_metrics: dict[str, float] = {}
        self.step_meta: dict[str, Any] = {}

    def log_meta(self, key: str, value: Any) -> None:
        """Non-numeric facts destined for the manifest (dataset name, sizes...)."""
        self.step_meta[key] = value

    def register(self, artifact: Artifact) -> None:
        self.registry[artifact.key] = artifact

    def input_hashes(self, keys: list[str]) -> dict[str, str]:
        missing = [k for k in keys if k not in self.registry]
        if missing:
            raise KeyError(f"inputs not available: {missing}")
        return {k: self.registry[k].content_hash for k in keys}

    def get(self, key: str) -> Any:
        artifact = self.registry[key]
        self.step_inputs[key] = artifact.content_hash
        return self._store.load(artifact.content_hash)

    def get_lazy(self, key: str) -> pl.LazyFrame:
        """Streaming access to a parquet artifact — for steps that must not
        materialize multi-GB tables (e.g. validation scans)."""
        artifact = self.registry[key]
        self.step_inputs[key] = artifact.content_hash
        return pl.scan_parquet(artifact.path)

    def put(self, obj: Any, key: str, *, ext: str = "parquet") -> Artifact:
        artifact = self._store.save(obj, key, ext=ext)
        self.step_outputs[key] = artifact
        self.register(artifact)
        return artifact

    def put_alias(self, key: str, source_key: str) -> Artifact:
        """Pass-through output: new key, same content hash, no new bytes."""
        src = self.registry[source_key]
        artifact = Artifact(key=key, content_hash=src.content_hash, path=src.path, meta=src.meta)
        self.step_outputs[key] = artifact
        self.register(artifact)
        return artifact

    def previous(self, step_name: str) -> dict | None:
        """Latest completed manifest for a step — e.g. row counts of the
        previous snapshot. None on the first ever run."""
        if self.manifest_log is None:
            return None
        for m in reversed(self.manifest_log.entries()):
            if m["step"] == step_name and m["status"] in ("success", "cached"):
                return m
        return None

    def log_metric(self, name: str, value: float) -> None:
        self.step_metrics[name] = float(value)
        self._tracker.log_metrics({name: float(value)})
