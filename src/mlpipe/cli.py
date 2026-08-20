"""mlpipe CLI — the single write path. Every execution lands in manifests/.

    mlpipe run --pipeline demo --from make_data --to summarize --set k.v=1 --tag exp
"""

from __future__ import annotations

import importlib
from pathlib import Path

import typer
import yaml

from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    pipeline: str = typer.Option("demo", help="pipeline module in mlpipe.steps"),
    from_step: str = typer.Option(None, "--from", help="first step to (re)run"),
    to_step: str = typer.Option(None, "--to", help="last step to run"),
    set_: list[str] = typer.Option(None, "--set", help="config override key.path=value"),
    tag: str = typer.Option(None, help="experiment tag recorded in manifests"),
    config: Path = typer.Option(None, help="YAML config file (else module default)"),
    store_dir: Path = typer.Option(Path("store"), help="artifact store root"),
    manifests_dir: Path = typer.Option(Path("manifests"), help="manifest log root"),
) -> None:
    module, cfg = _load_module_and_config(pipeline, config)
    pipe = Pipeline(
        module.build_steps(), LocalCasStore(store_dir), manifests_dir, cfg, _make_tracker(cfg)
    )
    result = pipe.run(from_step, to_step, overrides=set_, tag=tag)
    for m in result["manifests"]:
        typer.echo(f"{m['step']:<20} {m['status']:<8} sig={m['signature'][:8]} "
                   f"outputs={ {k: h[:8] for k, h in m['outputs'].items()} }")


def _load_module_and_config(pipeline: str, config: Path | None):
    module = importlib.import_module(f"mlpipe.steps.{pipeline}")
    cfg = yaml.safe_load(config.read_text()) if config else getattr(module, "DEFAULT_CONFIG", {})
    return module, cfg


def _make_tracker(cfg: dict):
    if cfg.get("tracker", {}).get("kind") == "mlflow":
        from mlpipe.core.mlflow_tracker import MlflowTracker

        return MlflowTracker(
            cfg["tracker"].get("uri", "sqlite:///mlruns/mlflow.db"),
            cfg["tracker"].get("experiment", "mlpipe"),
        )
    return None


@app.command()
def tune(
    pipeline: str = typer.Option("numerai"),
    n_trials: int = typer.Option(None, help="trials this invocation (study resumes)"),
    set_: list[str] = typer.Option(None, "--set", help="fixed overrides applied to every trial"),
    config: Path = typer.Option(None),
    store_dir: Path = typer.Option(Path("store")),
    manifests_dir: Path = typer.Option(Path("manifests")),
    baselines: Path = typer.Option(Path("baselines.yaml")),
) -> None:
    """Optuna study over the model config subtree. Each trial is a full
    pipeline run from train — the cache makes upstream free. Search space is
    declared in config (tune.space), never in code."""
    import optuna

    module, cfg = _load_module_and_config(pipeline, config)
    tcfg = cfg["tune"]
    pipe = Pipeline(
        module.build_steps(), LocalCasStore(store_dir), manifests_dir, cfg, _make_tracker(cfg)
    )
    study = optuna.create_study(
        study_name=tcfg["study"],
        storage=tcfg["storage"],
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        overrides = list(set_ or [])
        for key, spec in tcfg["space"].items():
            if spec["type"] == "int":
                value = trial.suggest_int(key, spec["low"], spec["high"])
            else:
                value = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
            overrides.append(f"{key}={value}")
        result = pipe.run(
            from_step="train", to_step="evaluate", overrides=overrides,
            tag=f"{tcfg['study']}-trial{trial.number}",
        )
        evaluated = [m for m in result["manifests"] if m["step"] == "evaluate"][-1]
        return evaluated["metrics"]["mean_corr"]

    study.optimize(objective, n_trials=n_trials or tcfg.get("n_trials", 5))
    best = study.best_trial
    promoted = {
        "promoted": {
            "study": tcfg["study"],
            "trial": best.number,
            "mean_corr": best.value,
            "params": best.params,
        }
    }
    baselines.write_text(yaml.safe_dump(promoted, sort_keys=True))
    typer.echo(f"best trial {best.number}: mean_corr={best.value:.5f} -> {baselines}")


@app.command()
def predict(
    name: str = typer.Option("numerai-model"),
    version: str = typer.Option(None, help="registry version (default: latest)"),
    pipeline: str = typer.Option("numerai"),
    config: Path = typer.Option(None),
    store_dir: Path = typer.Option(Path("store")),
    manifests_dir: Path = typer.Option(Path("manifests")),
    out: Path = typer.Option(Path("predictions.parquet")),
) -> None:
    """Serve: load a registered bundle by name/version and predict on the
    latest raw_live snapshot — no access to training internals."""
    import polars as pl

    _, cfg = _load_module_and_config(pipeline, config)
    tracker_cfg = cfg.get("tracker", {})
    import mlflow

    mlflow.set_tracking_uri(tracker_cfg.get("uri", "sqlite:///mlruns/mlflow.db"))
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    chosen = next(
        v for v in sorted(versions, key=lambda v: int(v.version), reverse=True)
        if version is None or v.version == str(version)
    )
    store = LocalCasStore(store_dir)
    bundle = store.load(chosen.tags["content_hash"])

    from mlpipe.core.manifest import ManifestLog

    live_hash = ManifestLog(manifests_dir).latest_outputs()["raw_live"]
    live = store.load(live_hash)
    feats = live.select(bundle["features"])
    if bundle["cleaner"].get("kind") == "baseline":
        feats = feats.with_columns(pl.all().fill_null(bundle["cleaner"].get("fill_value", 2)))
    transformer = store.load(bundle["transformer_hash"])
    matrix = transformer.transform(feats.to_numpy())
    model = store.load(bundle["model_hash"])["full"]
    preds = model.predict(pl.from_numpy(matrix, schema=bundle["features"]))
    frame = pl.DataFrame({"era": live["era"], "prediction": preds})
    frame.write_parquet(out)
    typer.echo(
        f"{name} v{chosen.version} (bundle {chosen.tags['content_hash'][:8]}, "
        f"plan {bundle['plan_hash'][:8]}) -> {out} ({frame.height} rows, live {live_hash[:8]})"
    )


@app.command()
def lineage(
    content_hash: str,
    manifests_dir: Path = typer.Option(Path("manifests")),
) -> None:
    from mlpipe.core.manifest import ManifestLog

    for m in ManifestLog(manifests_dir).lineage(content_hash):
        typer.echo(f"{m['step']:<20} sig={m['signature'][:8]} "
                   f"commit={(m['git_commit'] or '?')[:8]} inputs={list(m['inputs'])}")


if __name__ == "__main__":
    app()
