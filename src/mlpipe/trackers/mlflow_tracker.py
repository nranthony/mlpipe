"""MlflowTracker — thin view over our manifests (DESIGN.md §3). Index and
viewing layer ONLY: losing mlruns/ must lose zero provenance."""

from __future__ import annotations

from typing import Any

from mlpipe.core.interfaces import Artifact


class MlflowTracker:
    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlruns/mlflow.db",
        experiment: str = "mlpipe",
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def start_run(self, manifest: dict[str, Any]) -> str:
        run = self._mlflow.start_run(
            run_name=f"{manifest['step']}-{manifest['signature'][:8]}"
        )
        params = {
            "step": manifest["step"],
            "signature": manifest["signature"][:16],
            "config_hash": manifest["config_hash"][:16],
            "git_commit": (manifest.get("git_commit") or "")[:12],
            "tag": manifest.get("tag") or "",
        }
        self._mlflow.log_params(params)
        return run.info.run_id

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow.active_run() is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def register_model(self, artifact: Artifact, name: str) -> None:
        client = self._mlflow.MlflowClient()
        if not any(m.name == name for m in client.search_registered_models()):
            client.create_registered_model(name)
        client.create_model_version(
            name=name,
            source=str(artifact.path),
            tags={"content_hash": artifact.content_hash, **{
                k: str(v)[:250] for k, v in artifact.meta.items()
            }},
        )

    def end_run(self, status: str) -> None:
        if self._mlflow.active_run() is not None:
            self._mlflow.end_run("FINISHED" if status == "success" else "FAILED")
