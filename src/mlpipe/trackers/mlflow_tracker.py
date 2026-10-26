"""MlflowTracker — thin view over our manifests (DESIGN.md §3). Index and
viewing layer ONLY: losing mlruns/ must lose zero provenance. Curves and
charts arrive here as copies of store artifacts (ADR-0010)."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from mlpipe.core.interfaces import Artifact

BATCH = 1000  # MLflow's per-request metric limit


def flat_params(node: Any, prefix: str = "") -> dict[str, str]:
    """A step's config subtree as dotted MLflow params, values cut to MLflow's limit."""
    if isinstance(node, dict) and node:
        out: dict[str, str] = {}
        for k, v in node.items():
            out |= flat_params(v, f"{prefix}{k}.")
        return out
    return {prefix.rstrip("."): str(node)[:500]} if prefix else {}


class MlflowTracker:
    """Each pipeline run is a parent run; each step, executed or cached, a child
    run beneath it. Every run carries `mlpipe.*` tags, so the UI can filter to one
    pipeline run (`mlpipe.run_id`, as in the manifests) or one step signature."""

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlruns/mlflow.db",
        experiment: str = "mlpipe",
        pipeline: str | None = None,
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        self._experiment_id = mlflow.set_experiment(experiment).experiment_id
        self._pipeline = pipeline
        self._parent: str | None = None  # the open pipeline run, if any
        self._run_tags: dict[str, str] = {}

    def start_pipeline(self, run_id: str, tag: str | None) -> None:
        self._run_tags = {"mlpipe.run_id": run_id, "mlpipe.pipeline": self._pipeline or "",
                          "mlpipe.tag": tag or ""}
        run = self._mlflow.start_run(
            run_name=f"{self._pipeline or 'pipeline'}-{run_id[:8]}",
            tags=self._tags({"mlpipe.kind": "pipeline"}),
        )
        self._parent = run.info.run_id

    def end_pipeline(self, status: str) -> None:
        active = self._mlflow.active_run()
        if self._parent is not None and active is not None and active.info.run_id == self._parent:
            self._mlflow.end_run("FINISHED" if status == "success" else "FAILED")
        self._parent, self._run_tags = None, {}

    def _tags(self, extra: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in (self._run_tags | extra).items() if v}  # MLflow refuses ""

    def _step_tags(self, manifest: dict[str, Any], cached: bool) -> dict[str, str]:
        return self._tags({"mlpipe.kind": "step", "mlpipe.step": manifest["step"],
                           "mlpipe.signature": manifest["signature"],
                           "mlpipe.cached": str(cached).lower()})

    def start_run(self, manifest: dict[str, Any]) -> str:
        run = self._mlflow.start_run(
            run_name=f"{manifest['step']}-{manifest['signature'][:8]}",
            nested=self._parent is not None,
            tags=self._step_tags(manifest, cached=False),
        )
        self._mlflow.log_params(self._params(manifest))
        return run.info.run_id

    def log_cached(self, manifest: dict[str, Any]) -> None:
        """A small run for a cache hit: params, the manifest's scalar metrics, and a
        link to the run that executed this signature, where its curves and charts are."""
        tags = self._step_tags(manifest, cached=True)
        note = "Cache hit: outputs reused from an earlier execution; nothing re-ran."
        source = self._source_run(manifest["signature"])
        if source is not None:
            tags["mlpipe.source_run_id"] = source.info.run_id
            note += (f" Curves and charts are on [{source.info.run_name}]"
                     f"(#/experiments/{self._experiment_id}/runs/{source.info.run_id}).")
        tags["mlflow.note.content"] = note
        self._mlflow.start_run(
            run_name=f"{manifest['step']}-{manifest['signature'][:8]} (cached)",
            nested=self._parent is not None,
            tags=tags,
        )
        self._mlflow.log_params(self._params(manifest))
        metrics = {k: float(v) for k, v in (manifest.get("metrics") or {}).items()
                   if isinstance(v, int | float) and not isinstance(v, bool)}
        if metrics:
            self._mlflow.log_metrics(metrics)
        self._mlflow.end_run("FINISHED")

    def _source_run(self, signature: str) -> Any:
        """The newest finished run that executed this signature. Matched on the
        `signature` param, which runs logged before the tags also carry."""
        runs = self._mlflow.MlflowClient().search_runs(
            [self._experiment_id],
            filter_string=f"params.signature = '{signature[:16]}' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=50,
        )
        return next((r for r in runs if r.data.tags.get("mlpipe.cached") != "true"), None)

    def _params(self, manifest: dict[str, Any]) -> dict[str, str]:
        params = {
            "step": manifest["step"],
            "signature": manifest["signature"][:16],
            "config_hash": manifest["config_hash"][:16],
            "git_commit": (manifest.get("git_commit") or "")[:12],
            "tag": manifest.get("tag") or "",
        }
        # the step's own config, so the UI can compare runs by hyperparameter
        params |= {f"cfg.{k}": v for k, v in flat_params(manifest.get("config") or {}).items()}
        return params

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow.active_run() is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def log_curve(self, name: str, values: list[float]) -> None:
        """One request per thousand points; a step index per iteration, so the UI
        draws a line chart."""
        run = self._mlflow.active_run()
        if run is None:
            return
        from mlflow.entities import Metric

        now = int(time.time() * 1000)
        points = [Metric(name, v, now, i) for i, v in enumerate(values)]
        client = self._mlflow.MlflowClient()
        for start in range(0, len(points), BATCH):
            client.log_batch(run.info.run_id, metrics=points[start : start + BATCH])

    def log_artifact(self, artifact: Artifact) -> None:
        """Copied under its key (`eval_chart.html`), not its hash, so the UI's file
        list is readable."""
        if self._mlflow.active_run() is None:
            return
        with tempfile.TemporaryDirectory() as tmp:
            named = Path(tmp) / f"{artifact.key}{artifact.path.suffix}"
            shutil.copyfile(artifact.path, named)
            self._mlflow.log_artifact(str(named))

    def register_model(self, artifact: Artifact, name: str) -> str:
        """Version for this bundle hash; reused if already registered (promote is
        re-runnable without minting duplicate versions)."""
        client = self._mlflow.MlflowClient()
        if not any(m.name == name for m in client.search_registered_models()):
            client.create_registered_model(name)
        for v in client.search_model_versions(f"name='{name}'"):
            if v.tags.get("content_hash") == artifact.content_hash:
                return v.version
        version = client.create_model_version(
            name=name,
            source=str(artifact.path),
            tags={"content_hash": artifact.content_hash, **{
                k: str(v)[:250] for k, v in artifact.meta.items()
            }},
        )
        return version.version

    def set_alias(self, name: str, alias: str, version: str) -> None:
        self._mlflow.MlflowClient().set_registered_model_alias(name, alias, version)

    def end_run(self, status: str) -> None:
        if self._mlflow.active_run() is not None:
            self._mlflow.end_run("FINISHED" if status == "success" else "FAILED")
