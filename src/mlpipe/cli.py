"""mlpipe CLI — the single write path. Every execution lands in manifests/.

    mlpipe run --pipeline demo --from make_data --to summarize --set k.v=1 --tag exp

`gc --dry-run` only reports store files nothing reaches; deletion is a human step.
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from mlpipe.core.config import PipelineConfig, validate_config
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.plugins import pipeline_module

app = typer.Typer(add_completion=False, no_args_is_help=True)

# One manifest log per pipeline (work/0004 G): a demo run must not land in the
# real pipeline's history. The store stays shared — content addressing dedupes it.
MANIFESTS = typer.Option(None, help="manifest log root (default: manifests/<pipeline>)")
# No default: the CLI is domain-free and names no pipeline of its own.
PIPELINE = typer.Option(..., help="registered pipeline (builtin or mlpipe.pipelines entry point)")


def _log_dir(pipeline: str, manifests_dir: Path | None) -> Path:
    return manifests_dir or Path("manifests") / pipeline


@app.command()
def run(
    pipeline: str = PIPELINE,
    from_step: str = typer.Option(
        None, "--from", help="first step to run; upstream must be cached under this config"
    ),
    to_step: str = typer.Option(None, "--to", help="last step to run"),
    set_: list[str] = typer.Option(None, "--set", help="config override key.path=value"),
    tag: str = typer.Option(None, help="experiment tag recorded in manifests"),
    config: Path = typer.Option(None, help="YAML config file (else module default)"),
    store_dir: Path = typer.Option(Path("store"), help="artifact store root"),
    manifests_dir: Path = MANIFESTS,
) -> None:
    steps, cfg, top = _load(pipeline, config)
    log_dir = _log_dir(pipeline, manifests_dir)
    pipe = Pipeline(steps, LocalCasStore(store_dir), log_dir, cfg, _make_tracker(top))
    result = pipe.run(from_step, to_step, overrides=set_, tag=tag)
    for m in result["manifests"]:
        typer.echo(f"{m['step']:<20} {m['status']:<8} sig={m['signature'][:8]} "
                   f"outputs={ {k: h[:8] for k, h in m['outputs'].items()} }")


def _load(pipeline: str, config: Path | None) -> tuple[list, dict, PipelineConfig]:
    """Steps, raw config, and the validated top-level keys (tracker, tune, ...)."""
    module = _module(pipeline)
    cfg = yaml.safe_load(config.read_text()) if config else getattr(module, "DEFAULT_CONFIG", {})
    steps = module.build_steps()
    return steps, cfg, validate_config(cfg, [s.name for s in steps])


def _module(pipeline: str):
    try:
        return pipeline_module(pipeline)
    except KeyError as exc:  # a usage error, not a traceback
        raise typer.BadParameter(exc.args[0], param_hint="--pipeline") from None


def _make_tracker(top: PipelineConfig):
    if top.tracker.kind == "mlflow":
        from mlpipe.trackers.mlflow_tracker import MlflowTracker

        return MlflowTracker(top.tracker.uri, top.tracker.experiment)
    return None


@app.command()
def tune(
    pipeline: str = PIPELINE,
    n_trials: int = typer.Option(None, help="trials this invocation (study resumes)"),
    set_: list[str] = typer.Option(None, "--set", help="fixed overrides applied to every trial"),
    config: Path = typer.Option(None),
    store_dir: Path = typer.Option(Path("store")),
    manifests_dir: Path = MANIFESTS,
) -> None:
    """Optuna study over the config-declared search space (tune.space). Each
    trial is a full pipeline run to evaluate: upstream steps whose config the
    trial leaves alone are cache hits, so a space reaching below train (e.g. a
    cleaner knob) recomputes honestly instead of being inert."""
    import optuna

    steps, cfg, top = _load(pipeline, config)
    tcfg = top.tune
    if tcfg is None:
        raise typer.BadParameter(f"pipeline {pipeline!r} config has no `tune` section")
    log_dir = _log_dir(pipeline, manifests_dir)
    pipe = Pipeline(steps, LocalCasStore(store_dir), log_dir, cfg, _make_tracker(top))
    study =optuna.create_study(
        study_name=tcfg.study,
        storage=tcfg.storage,
        direction=tcfg.direction,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        overrides = list(set_ or [])
        for key, spec in tcfg.space.items():
            if spec.type == "int":
                value = trial.suggest_int(key, spec.low, spec.high)
            else:
                value = trial.suggest_float(key, spec.low, spec.high, log=spec.log)
            overrides.append(f"{key}={value}")
        result = pipe.run(
            to_step=tcfg.step, overrides=overrides, tag=f"{tcfg.study}-trial{trial.number}"
        )
        metrics = [m for m in result["manifests"] if m["step"] == tcfg.step][-1]["metrics"]
        if tcfg.metric not in metrics:
            raise KeyError(f"tune.metric {tcfg.metric!r} not logged by {tcfg.step}: {sorted(metrics)}")
        return metrics[tcfg.metric]

    study.optimize(objective, n_trials=n_trials or tcfg.n_trials)
    best = study.best_trial
    sets = " ".join(f"--set {k}={v}" for k, v in best.params.items())
    # tune never promotes (ADR-0006): it names the candidate; a human/agent decides.
    typer.echo(
        f"best trial {best.number}: {tcfg.metric}={best.value:.5f} ({tcfg.direction})\n"
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
    pipeline: str = PIPELINE,
    config: Path = typer.Option(None),
    store_dir: Path = typer.Option(Path("store")),
    manifests_dir: Path = MANIFESTS,
    baselines: Path = typer.Option(Path("baselines.yaml")),
) -> None:
    """The only way to promote (ADR-0006). Pins the bundle in baselines.yaml with
    the config every lineage step actually ran with, moves the MLflow alias, and
    leaves a manifest. baselines.yaml is the source of truth; the alias is a view.
    The pin goes in the (pipeline, alias) slot; other slots are left as they are."""
    from datetime import UTC, datetime

    from mlpipe.baselines import write_pin
    from mlpipe.core.interfaces import Artifact
    from mlpipe.core.manifest import ManifestLog, git_commit

    log, store = ManifestLog(_log_dir(pipeline, manifests_dir)), LocalCasStore(store_dir)
    full =_resolve_bundle(log, bundle)
    if not store.exists(full):
        raise typer.BadParameter(f"bundle {full[:12]} is not in the store")
    chain = log.lineage(full)
    content = store.load(full)
    name = content.get("model_name", "mlpipe-model")
    steps = {
        m["step"]: {k: m.get(k) for k in ("config", "config_hash", "signature", "git_commit")}
        for m in reversed(chain)
    }
    # plan_hash/target are optional: a bundle from a pipeline without CV or a
    # single named target still promotes; the pin records what the bundle has.
    plan_hash, target = content.get("plan_hash"), content.get("target")
    tracker = _make_tracker(_load(pipeline, config)[2])
    version = None
    if tracker is not None:
        meta = {"plan_hash": plan_hash} if plan_hash else {}
        version = tracker.register_model(Artifact("model_bundle", full, store.path_of(full), meta), name)
        tracker.set_alias(name, alias, version)
    now = datetime.now(UTC).isoformat()
    pinned = {
        "model_name": name, "alias": alias, "registry_version": version,
        "bundle_hash": full, "plan_hash": plan_hash, "target": target,
        "metrics": content.get("metrics", {}), "steps": steps,
        "promoted_at": now, "git_commit": git_commit(),
    }
    write_pin(baselines, pipeline, alias, pinned)
    log.write({
        "step": "promote", "status": "promoted", "signature": f"promote:{alias}:{full}",
        "inputs": {"model_bundle": full}, "outputs": {}, "metrics": {},
        "meta": {"name": name, "alias": alias, "registry_version": version},
        "git_commit": pinned["git_commit"], "started_at": now, "ended_at": now,
    })
    typer.echo(f"promoted {name}@{alias} -> bundle {full[:8]} (plan {(plan_hash or '-')[:8]}, "
               f"registry v{version}) -> {baselines} [{pipeline}.{alias}]")


def _default_prepare(frame, bundle: dict):
    """(features in bundle order, id columns to carry into the output)."""
    return frame.select(bundle["features"]), frame.select(bundle.get("id_columns", []))


@app.command()
def predict(
    bundle: str = typer.Option(None, help="model_bundle hash/prefix (default: baselines.yaml)"),
    alias: str = typer.Option("champion", help="baselines.yaml slot to serve when no --bundle"),
    pipeline: str = PIPELINE,
    input_: Path = typer.Option(
        None, "--input", help="parquet to score (default: the pipeline's serve_input hook)"
    ),
    store_dir: Path = typer.Option(Path("store")),
    manifests_dir: Path = MANIFESTS,
    baselines: Path = typer.Option(Path("baselines.yaml")),
    out: Path = typer.Option(Path("predictions.parquet")),
) -> None:
    """Serve: load the promoted bundle (or an explicit one) and predict on --input
    or the pipeline's default serving input — no training internals, no MLflow.

    Pipeline hooks (both optional): `serve_input(log, store) -> (frame, label)`
    and `prepare(frame, bundle) -> (features, ids)`; the output is ids + prediction."""
    import polars as pl

    from mlpipe.baselines import pinned
    from mlpipe.core.manifest import ManifestLog

    module = _module(pipeline)
    log, store = ManifestLog(_log_dir(pipeline, manifests_dir)), LocalCasStore(store_dir)
    if bundle:
        bundle_hash = _resolve_bundle(log, bundle)
    else:
        pin = pinned(baselines, pipeline, alias)
        if not pin:
            raise typer.BadParameter(
                f"no --bundle and no {pipeline}.{alias} pin in {baselines}; "
                "run `mlpipe promote` first"
            )
        bundle_hash = pin["bundle_hash"]
    content = store.load(bundle_hash)

    if input_ is not None:
        frame, source = pl.read_parquet(input_), str(input_)
    elif hasattr(module, "serve_input"):
        frame, source = module.serve_input(log, store)
    else:
        raise typer.BadParameter(f"pipeline {pipeline!r} has no serve_input hook; pass --input")
    feats, ids = getattr(module, "prepare", _default_prepare)(frame, content)
    transformer = store.load(content["transformer_hash"])
    matrix = transformer.transform(feats.to_numpy())
    model = store.load(content["model_hash"])["full"]
    preds = model.predict(pl.from_numpy(matrix, schema=content["features"]))
    result = ids.with_columns(pl.Series("prediction", preds))
    result.write_parquet(out)
    typer.echo(
        f"{content.get('model_name', '?')} (bundle {bundle_hash[:8]}, "
        f"plan {(content.get('plan_hash') or '-')[:8]}) -> {out} ({result.height} rows, {source})"
    )


@app.command()
def lineage(
    content_hash: str,
    pipeline: str = PIPELINE,
    manifests_dir: Path = MANIFESTS,
) -> None:
    from mlpipe.core.manifest import ManifestLog

    log = ManifestLog(_log_dir(pipeline, manifests_dir))
    produced = {h for m in log.entries() for h in m["outputs"].values()}
    found = [h for h in produced if h.startswith(content_hash)]
    if len(found) != 1:  # an unknown hash must not print an empty, "successful" lineage
        raise typer.BadParameter(f"{content_hash!r} matches {len(found)} artifacts in the manifests")
    for m in log.lineage(found[0]):
        typer.echo(f"{m['step']:<20} sig={m['signature'][:8]} "
                   f"commit={(m['git_commit'] or '?')[:8]} inputs={list(m['inputs'])}")


def _gb(paths) -> str:
    return f"{sum(p.stat().st_size for p in paths) / 1e9:.2f} GB"


@app.command()
def gc(
    dry_run: bool = typer.Option(False, "--dry-run", help="required: gc only reports"),
    keep_days: int = typer.Option(
        30, help="keep everything a manifest from the last N days read or wrote"
    ),
    store_dir: Path = typer.Option(Path("store")),
    manifests_root: Path = typer.Option(
        Path("manifests"), help="every manifest log under here counts (the store is shared)"
    ),
    baselines: Path = typer.Option(Path("baselines.yaml")),
    root_file: list[Path] = typer.Option(
        None, "--root-file", help="file naming bundle hashes to keep (e.g. a host's champion record)"
    ),
) -> None:
    """Report store files that no promoted bundle, pinned baseline, root file or
    recent manifest reaches. Never deletes: removing the listed files is a human step."""
    from mlpipe.gc import collect

    if not dry_run:
        raise typer.BadParameter(
            "gc only reports; deleting is a human step. Pass --dry-run", param_hint="--dry-run"
        )
    r = collect(store_dir, manifests_root, baselines, keep_days, root_files=root_file or [])
    gone = r.unreachable
    pinned = ", ".join(sorted(h[:8] for h in r.bundles)) or "none"
    typer.echo(f"store:       {len(r.files)} artifacts, {_gb(r.files.values())}")
    typer.echo(f"roots:       bundles {pinned}; {r.window_manifests} manifests "
               f"in the last {keep_days} days")
    typer.echo(f"kept:        {len(r.files) - len(gone)} artifacts, "
               f"{_gb(p for h, p in r.files.items() if h not in gone)}")
    typer.echo(f"unreachable: {len(gone)} artifacts, {_gb(r.files[h] for h in gone)}")
    for h in sorted(gone, key=lambda h: -r.files[h].stat().st_size):
        typer.echo(f"  {r.files[h].stat().st_size / 1e9:6.2f} GB  {r.files[h]}  "
                   f"({r.last_written.get(h, 'in no manifest')})")
    if r.other_files:
        typer.echo(f"not artifacts, left alone: {len(r.other_files)}")
        for p in r.other_files:
            typer.echo(f"  {p}")
    if r.missing:
        typer.echo(f"named by a root but absent from the store: {len(r.missing)}")
    for note in r.notes:
        typer.echo(f"note: {note}")
    typer.echo("dry run: nothing deleted. Removing the unreachable files is a human step.")


if __name__ == "__main__":
    app()
