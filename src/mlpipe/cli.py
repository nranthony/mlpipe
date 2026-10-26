"""mlpipe CLI — the single write path. Every execution lands in manifests/.

    mlpipe run --pipeline demo --from make_data --to summarize --set k.v=1 --tag exp

`gc --dry-run` only reports store files nothing reaches; deletion is a human step.
"""

from __future__ import annotations

import uuid
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
    steps, serve, cfg, top = _load(pipeline, config)
    log_dir = _log_dir(pipeline, manifests_dir)
    pipe = Pipeline(steps, LocalCasStore(store_dir), log_dir, cfg, _make_tracker(top),
                    sibling_steps=_siblings(steps + serve, steps))
    result = pipe.run(from_step, to_step, overrides=set_, tag=tag)
    for m in result["manifests"]:
        _echo_status(m)


def _echo_status(m: dict) -> None:
    typer.echo(f"{m['step']:<20} {m['status']:<8} sig={m['signature'][:8]} "
               f"outputs={ {k: h[:8] for k, h in m['outputs'].items()} }")


def _load(pipeline: str, config: Path | None) -> tuple[list, list, dict, PipelineConfig]:
    """Training steps, serve steps, raw config, and the validated top-level keys
    (tracker, tune, ...). Both lists share the config, so the top level is
    validated against both lists' names."""
    module = _module(pipeline)
    cfg = yaml.safe_load(config.read_text()) if config else getattr(module, "DEFAULT_CONFIG", {})
    steps, serve = module.build_steps(), _serve_steps(module)
    return steps, serve, cfg, validate_config(cfg, [s.name for s in steps + serve])


def _siblings(every: list, ran: list) -> list[str]:
    """The sibling rule (work/0009 A2): every step name of the module's training
    and serve lists that this Pipeline does not run."""
    own = {s.name for s in ran}
    return [s.name for s in every if s.name not in own]


def _serve_steps(module) -> list:
    """`build_serve_steps()` (optional): a source step (no inputs, one output)
    first, a PredictStep last. Without it, serving is `[PredictStep()]` on --input.
    Called by run/tune for names too, so a serve module must import cheaply."""
    from mlpipe.steps.predict import PredictStep

    if not hasattr(module, "build_serve_steps"):
        return [PredictStep()]
    serve = module.build_serve_steps()
    if not serve or serve[0].inputs or len(serve[0].outputs) != 1:
        raise typer.BadParameter(
            f"{module.__name__}.build_serve_steps(): the first step must be a source "
            "(no inputs, exactly one output)", param_hint="--pipeline"
        )
    if not isinstance(serve[-1], PredictStep):
        raise typer.BadParameter(
            f"{module.__name__}.build_serve_steps(): the last step must be a PredictStep",
            param_hint="--pipeline",
        )
    return serve


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

    steps, serve, cfg, top = _load(pipeline, config)
    tcfg = top.tune
    if tcfg is None:
        raise typer.BadParameter(f"pipeline {pipeline!r} config has no `tune` section")
    log_dir = _log_dir(pipeline, manifests_dir)
    pipe = Pipeline(steps, LocalCasStore(store_dir), log_dir, cfg, _make_tracker(top),
                    sibling_steps=_siblings(steps + serve, steps))
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
    tracker = _make_tracker(_load(pipeline, config)[3])
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
        "run_id": uuid.uuid4().hex,
    })
    typer.echo(f"promoted {name}@{alias} -> bundle {full[:8]} (plan {(plan_hash or '-')[:8]}, "
               f"registry v{version}) -> {baselines} [{pipeline}.{alias}]")


@app.command()
def predict(
    bundle: str = typer.Option(None, help="model_bundle hash/prefix (default: baselines.yaml)"),
    alias: str = typer.Option("champion", help="baselines.yaml slot to serve when no --bundle"),
    pipeline: str = PIPELINE,
    input_: Path = typer.Option(
        None, "--input", exists=True, dir_okay=False,
        help="parquet to score in place of the serve list's source step "
             "(required when the pipeline has no build_serve_steps)",
    ),
    set_: list[str] = typer.Option(None, "--set", help="config override key.path=value"),
    tag: str = typer.Option(None, help="tag recorded in the serve manifests"),
    config: Path = typer.Option(None, help="YAML config file (else module default)"),
    store_dir: Path = typer.Option(Path("store")),
    manifests_dir: Path = MANIFESTS,
    baselines: Path = typer.Option(Path("baselines.yaml")),
    out: Path = typer.Option(Path("predictions.parquet")),
) -> None:
    """Serve (ADR-0007): run the pipeline's serve steps with the promoted bundle
    (or an explicit one) seeded, logged to the same manifest log as training, so
    `mlpipe lineage <predictions>` reaches the training run. No tracker.

    A module's optional `build_serve_steps()` lists a source step (no inputs, one
    output; name what it reads in config, e.g. `--set <source>.<key>=N`) through a
    PredictStep. `--input` replaces the source with a snapshot of the file; without
    serve steps, `--input` is required and is scored by `PredictStep()` alone."""
    import shutil
    from datetime import UTC, datetime

    from pydantic import ValidationError

    from mlpipe.baselines import pinned
    from mlpipe.core.interfaces import Artifact
    from mlpipe.core.manifest import ManifestLog, git_commit

    # Order: the name, the bundle, the module contract, then --input (work/0009 A4).
    module = _module(pipeline)
    log_dir = _log_dir(pipeline, manifests_dir)
    log = ManifestLog(log_dir)
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

    stale = [h for h in ("serve_input", "prepare") if hasattr(module, h)]
    if stale:  # a silently ignored prepare would serve unfilled features: wrong, no error
        raise typer.BadParameter(
            f"{module.__name__} defines {' and '.join(stale)}; these hooks are no longer read "
            "(ADR-0007): declare build_serve_steps() ending in PredictStep(prepare=...)",
            param_hint="--pipeline",
        )
    steps, serve, cfg, _ = _load(pipeline, config)
    has_source = hasattr(module, "build_serve_steps")
    if input_ is None and not has_source:
        raise typer.BadParameter(
            f"pipeline {pipeline!r} has no build_serve_steps(); pass --input", param_hint="--input"
        )

    store = LocalCasStore(store_dir)
    if not store.exists(bundle_hash):
        raise typer.BadParameter(f"bundle {bundle_hash[:12]} is not in the store")
    content = store.load(bundle_hash)
    seeds = [Artifact("model_bundle", bundle_hash, store.path_of(bundle_hash), {})]
    for key, field in (("model", "model_hash"), ("fitted_transformer", "transformer_hash")):
        h = content.get(field)
        if not h or not store.exists(h):
            raise typer.BadParameter(f"bundle {bundle_hash[:12]}: {key} {h} is not in the store")
        seeds.append(Artifact(key, h, store.path_of(h), {}))

    # --input replaces the source step (or, without one, feeds PredictStep directly)
    run_steps = serve[1:] if input_ is not None and has_source else serve
    pipe = Pipeline(run_steps, store, log_dir, cfg, tracker=None,
                    sibling_steps=_siblings(steps + serve, run_steps))
    try:  # before anything is written
        pipe.resolved_config(set_)
    except (TypeError, ValidationError) as exc:  # config errors are usage errors
        raise typer.BadParameter(str(exc), param_hint="--set/--config") from None

    if input_ is not None:
        key = serve[0].outputs[0] if has_source else serve[-1].frame_key
        frame = store.save(input_, key, ext="parquet")  # streamed snapshot
        seeds.append(frame)
        now = datetime.now(UTC).isoformat()
        # lineage would otherwise stop at a bare hash
        record = {
            "step": "input", "status": "success", "signature": f"input:{frame.content_hash}",
            "inputs": {}, "outputs": {key: frame.content_hash}, "metrics": {},
            "meta": {"file": str(input_), "bytes": input_.stat().st_size},
            "git_commit": git_commit(), "tag": tag, "started_at": now, "ended_at": now,
            "run_id": uuid.uuid4().hex,
        }
        log.write(record)
        _echo_status(record)

    result = pipe.run(overrides=set_, tag=tag, seeds=seeds)
    for m in result["manifests"]:
        _echo_status(m)

    preds = result["artifacts"]["predictions"]
    shutil.copyfile(preds.path, out)  # same bytes: the file's sha256 is the artifact hash
    rows = result["manifests"][-1]["meta"].get("rows")
    typer.echo(
        f"{content.get('model_name', '?')} (bundle {bundle_hash[:8]}, "
        f"plan {(content.get('plan_hash') or '-')[:8]}) -> {out} ({rows} rows, "
        f"predictions {preds.content_hash[:8]})"
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


def latest_by_step(entries: list[dict], order: list[str]) -> list[dict]:
    """The newest completed manifest per step, in pipeline order (steps the module
    no longer lists come last)."""
    latest = {m["step"]: m for m in entries if m["status"] in ("success", "cached")}
    rank = {name: i for i, name in enumerate(order)}
    return sorted(latest.values(), key=lambda m: rank.get(m["step"], len(rank)))


@app.command()
def show(
    pipeline: str = PIPELINE,
    step: list[str] = typer.Option(None, "--step", help="only these steps (repeatable)"),
    as_json: bool = typer.Option(False, "--json", help="full records, for agents and scripts"),
    store_dir: Path = typer.Option(Path("store")),
    manifests_dir: Path = MANIFESTS,
) -> None:
    """Read-only map from each step's latest completed run to its outputs' store
    files: what a report (ADR-0010) reads. Prints no data and writes nothing."""
    import json

    from mlpipe.core.manifest import ManifestLog

    module = _module(pipeline)
    order = [s.name for s in module.build_steps() + _serve_steps(module)]
    store = LocalCasStore(store_dir)
    records = [m for m in latest_by_step(ManifestLog(_log_dir(pipeline, manifests_dir)).entries(), order)
               if not step or m["step"] in step]
    if not records:
        raise typer.BadParameter(f"no completed manifests for {pipeline!r}"
                                 + (f" steps {step}" if step else ""))
    out = []
    for m in records:
        outputs = {}
        for key, h in m["outputs"].items():
            path = store.path_of(h) if store.exists(h) else None
            outputs[key] = {"hash": h, "path": str(path) if path else None,
                            "format": path.suffix.lstrip(".") if path else None}
        out.append({k: m.get(k) for k in ("step", "status", "signature", "run_id", "ended_at",
                                          "tag", "git_commit", "config", "metrics", "meta")}
                   | {"outputs": outputs})
    if as_json:
        typer.echo(json.dumps(out, indent=2, default=str))
        return
    for r in out:
        typer.echo(f"{r['step']:<14} {r['status']:<7} sig={r['signature'][:8]} "
                   f"at={(r['ended_at'] or '')[:19]}")
        for key, o in r["outputs"].items():
            typer.echo(f"    {key:<20} {o['hash'][:8]}  {o['path'] or '(not in store)'}")
        if r["metrics"]:
            shown = ", ".join(f"{k}={v:.4g}" for k, v in sorted(r["metrics"].items()))
            typer.echo(f"    metrics: {shown}")


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
