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
    from_step: str = typer.Option(
        None, "--from", help="first step to run; upstream must be cached under this config"
    ),
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
        from mlpipe.trackers.mlflow_tracker import MlflowTracker

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
) -> None:
    """Optuna study over the config-declared search space (tune.space). Each
    trial is a full pipeline run to evaluate: upstream steps whose config the
    trial leaves alone are cache hits, so a space reaching below train (e.g. a
    cleaner knob) recomputes honestly instead of being inert."""
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
            to_step="evaluate", overrides=overrides,
            tag=f"{tcfg['study']}-trial{trial.number}",
        )
        evaluated = [m for m in result["manifests"] if m["step"] == "evaluate"][-1]
        return evaluated["metrics"]["mean_corr"]

    study.optimize(objective, n_trials=n_trials or tcfg.get("n_trials", 5))
    best = study.best_trial
    sets = " ".join(f"--set {k}={v}" for k, v in best.params.items())
    # tune never promotes (ADR-0006): it names the candidate; a human/agent decides.
    typer.echo(
        f"best trial {best.number}: mean_corr={best.value:.5f}\n"
        f"to promote: mlpipe run --pipeline {pipeline} {sets}  then  mlpipe promote <bundle>"
    )


def _resolve_bundle(log, prefix: str) -> str:
    hashes = {
        m["outputs"]["model_bundle"] for m in log.entries()
        if m["status"] in ("success", "cached") and "model_bundle" in m["outputs"]
    }
    found = [h for h in hashes if h.startswith(prefix)]
    if len(found) != 1:
        raise typer.BadParameter(f"{prefix!r} matches {len(found)} model bundles in the manifests")
    return found[0]


@app.command()
def promote(
    bundle: str = typer.Argument(help="model_bundle hash (or unique prefix)"),
    alias: str = typer.Option("champion"),
    pipeline: str = typer.Option("numerai"),
    config: Path = typer.Option(None),
    store_dir: Path = typer.Option(Path("store")),
    manifests_dir: Path = typer.Option(Path("manifests")),
    baselines: Path = typer.Option(Path("baselines.yaml")),
) -> None:
    """The only way to promote (ADR-0006). Pins the bundle in baselines.yaml with
    the config every lineage step actually ran with, moves the MLflow alias, and
    leaves a manifest. baselines.yaml is the source of truth; the alias is a view."""
    from datetime import UTC, datetime

    from mlpipe.core.interfaces import Artifact
    from mlpipe.core.manifest import ManifestLog, git_commit

    log, store = ManifestLog(manifests_dir), LocalCasStore(store_dir)
    full = _resolve_bundle(log, bundle)
    if not store.exists(full):
        raise typer.BadParameter(f"bundle {full[:12]} is not in the store")
    chain = log.lineage(full)
    content = store.load(full)
    name = content.get("model_name", "mlpipe-model")
    steps = {
        m["step"]: {k: m.get(k) for k in ("config", "config_hash", "signature", "git_commit")}
        for m in reversed(chain)
    }
    _, cfg = _load_module_and_config(pipeline, config)
    tracker = _make_tracker(cfg)
    version = None
    if tracker is not None:
        version = tracker.register_model(
            Artifact("model_bundle", full, store.path_of(full), {"plan_hash": content["plan_hash"]}),
            name,
        )
        tracker.set_alias(name, alias, version)
    now = datetime.now(UTC).isoformat()
    pinned = {
        "model_name": name, "alias": alias, "registry_version": version,
        "bundle_hash": full, "plan_hash": content["plan_hash"], "target": content["target"],
        "metrics": content["metrics"], "steps": steps,
        "promoted_at": now, "git_commit": git_commit(),
    }
    baselines.write_text(yaml.safe_dump({"promoted": pinned}, sort_keys=True))
    log.write({
        "step": "promote", "status": "promoted", "signature": f"promote:{alias}:{full}",
        "inputs": {"model_bundle": full}, "outputs": {}, "metrics": {},
        "meta": {"name": name, "alias": alias, "registry_version": version},
        "git_commit": pinned["git_commit"], "started_at": now, "ended_at": now,
    })
    typer.echo(f"promoted {name}@{alias} -> bundle {full[:8]} (plan {content['plan_hash'][:8]}, "
               f"registry v{version}) -> {baselines}")


@app.command()
def predict(
    bundle: str = typer.Option(None, help="model_bundle hash/prefix (default: baselines.yaml)"),
    pipeline: str = typer.Option("numerai"),
    store_dir: Path = typer.Option(Path("store")),
    manifests_dir: Path = typer.Option(Path("manifests")),
    baselines: Path = typer.Option(Path("baselines.yaml")),
    out: Path = typer.Option(Path("predictions.parquet")),
) -> None:
    """Serve: load the promoted bundle (or an explicit one) and predict on the
    latest raw_live snapshot — no access to training internals, no MLflow."""
    import polars as pl

    from mlpipe.core.manifest import ManifestLog

    log, store = ManifestLog(manifests_dir), LocalCasStore(store_dir)
    if bundle:
        bundle_hash = _resolve_bundle(log, bundle)
    else:
        pinned = (yaml.safe_load(baselines.read_text()) or {}) if baselines.exists() else {}
        bundle_hash = pinned.get("promoted", {}).get("bundle_hash")
        if not bundle_hash:
            raise typer.BadParameter(
                f"no --bundle and no promoted bundle in {baselines} "
                "(missing, or written before ADR-0006); run `mlpipe promote` first"
            )
    content = store.load(bundle_hash)

    live_hash = log.latest_outputs()["raw_live"]
    live = store.load(live_hash)
    feats = live.select(content["features"])
    if content["cleaner"].get("kind") == "baseline":
        feats = feats.with_columns(pl.all().fill_null(content["cleaner"].get("fill_value", 2)))
    transformer = store.load(content["transformer_hash"])
    matrix = transformer.transform(feats.to_numpy())
    model = store.load(content["model_hash"])["full"]
    preds = model.predict(pl.from_numpy(matrix, schema=content["features"]))
    frame = pl.DataFrame({"era": live["era"], "prediction": preds})
    frame.write_parquet(out)
    typer.echo(
        f"{content.get('model_name', '?')} (bundle {bundle_hash[:8]}, "
        f"plan {content['plan_hash'][:8]}) -> {out} ({frame.height} rows, live {live_hash[:8]})"
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
