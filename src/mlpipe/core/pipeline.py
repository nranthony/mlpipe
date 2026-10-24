"""Pipeline: ordered steps, cache hit/miss orchestration, manifest per execution.

Cache rule (DESIGN.md §1): a step is skipped iff a completed manifest exists for
its signature AND every output hash in that manifest is present in the store.
Cache hits still write a manifest (status "cached") — every execution leaves one.
"""

from __future__ import annotations

import copy
import json
import platform
import traceback
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from mlpipe.core.config import forbids_extras, validate_config
from mlpipe.core.context import NoopTracker, RunContext
from mlpipe.core.interfaces import Artifact, Step
from mlpipe.core.manifest import ManifestLog, git_commit, git_dirty, mlpipe_info
from mlpipe.core.signature import config_hash, config_subtree, sha256_bytes


def parse_value(raw: str) -> Any:
    """JSON first, then YAML: YAML 1.1 reads `1e-05` (how Python prints small
    floats) as a string, and a string param would also enter the signature."""
    try:
        return json.loads(raw)
    except ValueError:
        return yaml.safe_load(raw)


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    """Apply --set key.path=value pairs (JSON/YAML-typed values) to a config copy."""
    config = copy.deepcopy(config)
    for item in overrides or []:
        keypath, _, raw = item.partition("=")
        node = config
        *parents, leaf = keypath.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = parse_value(raw)
    return config


LOCKFILES = ("uv.lock", "poetry.lock", "pdm.lock", "pixi.lock", "requirements.lock")
# Versions recorded in every manifest when installed (names only: nothing imported).
PACKAGES = ("polars", "numpy", "pydantic", "scikit-learn", "lightgbm", "torch")


def _versions(names: tuple[str, ...]) -> dict[str, str]:
    found = {}
    for name in dict.fromkeys(names):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def environment_info(project_root: Path | None = None, packages: tuple[str, ...] = ()) -> dict:
    """What ran it, beyond the signature. Computed once per run. No CUDA probe:
    that would import torch into core (ADR-0004); backends record devices."""
    root = Path(project_root or ".")
    lock = next((root / n for n in LOCKFILES if (root / n).exists()), None)
    return {
        "lockfile": lock.name if lock else None,
        "lockfile_hash": sha256_bytes(lock.read_bytes()) if lock else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_dirty": git_dirty(project_root),
        "mlpipe": mlpipe_info(),
        "packages": _versions((*PACKAGES, *packages)),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Pipeline:
    def __init__(
        self,
        steps: list[Step],
        store: Any,
        manifests_dir: Path | str,
        config: dict | None = None,
        tracker: Any = None,
        project_root: Path | str | None = None,
        packages: tuple[str, ...] = (),
        sibling_steps: Iterable[str] = (),
    ) -> None:
        self.steps = list(steps)
        # Steps sharing this config that this pipeline doesn't run (a module's serve
        # list beside its training list): their subtrees pass top-level validation
        # untouched and enter no signature here.
        self.sibling_steps = list(sibling_steps)
        self.store = store
        self.log = ManifestLog(manifests_dir)
        self.config = config or {}
        self.tracker = tracker or NoopTracker()
        # The host project: its git commit and lockfile go in every manifest
        # (default: the cwd). `packages` adds names to the recorded versions.
        self.project_root = Path(project_root) if project_root else None
        self.packages = tuple(packages)

    def run(
        self,
        from_step: str | None = None,
        to_step: str | None = None,
        overrides: list[str] | None = None,
        tag: str | None = None,
        seeds: list[Artifact] | None = None,
    ) -> dict[str, Any]:
        config = self.resolved_config(overrides)
        names = [s.name for s in self.steps]
        ctx = RunContext(config, self.store, self.tracker)
        ctx.manifest_log = self.log
        for artifact in seeds or []:
            ctx.register(artifact)

        lo = names.index(from_step) if from_step else 0
        hi = names.index(to_step) + 1 if to_step else len(self.steps)
        env = environment_info(self.project_root, self.packages)
        run_id = uuid.uuid4().hex  # groups this run's manifests; in no signature

        # Upstream of --from is resolved by signature under the CURRENT config and
        # must be a cache hit: --from asserts "upstream is cached", it never serves
        # outputs some other config produced (work/0004 C).
        upstream = [
            self._run_step(s, ctx, config, tag, env, run_id, from_step=from_step)
            for s in self.steps[:lo]
        ]
        written = [self._run_step(s, ctx, config, tag, env, run_id) for s in self.steps[lo:hi]]
        return {"artifacts": dict(ctx.registry), "manifests": upstream + written}

    def resolved_config(self, overrides: list[str] | None = None) -> dict:
        """Overrides applied and every subtree validated: what `run` executes with.
        Raises before any step runs (TypeError, pydantic ValidationError)."""
        config = apply_overrides(self.config, overrides)
        names = [s.name for s in self.steps]
        validate_config(config, names + self.sibling_steps)  # top-level typos die here too
        for step in self.steps:  # config typos die here, before any step runs
            model = getattr(step, "config_model", None)
            if model is not None:
                if not forbids_extras(model):  # else a typo vanishes, signature included
                    raise TypeError(
                        f"step {step.name!r}: {model.__name__} must forbid extra keys "
                        "(subclass mlpipe.core.config.StrictConfig)"
                    )
                config[step.name] = model.model_validate(config.get(step.name, {})).model_dump()
            elif config.get(step.name):  # nothing could validate it: a typo would pass
                raise TypeError(f"step {step.name!r} has config but no config_model")
        return config

    def _run_step(
        self,
        step: Step,
        ctx: RunContext,
        config: dict,
        tag: str | None,
        env: dict,
        run_id: str,
        from_step: str | None = None,
    ) -> dict:
        ctx.reset_step(step.inputs)
        started = _now()
        input_hashes = ctx.input_hashes(step.inputs)
        sig = step.signature(ctx)
        base = {
            "step": step.name,
            "signature": sig,
            "git_commit": git_commit(self.project_root),
            "config_hash": config_hash(config, step.name),
            "config": config_subtree(config, step.name),  # what promote pins (ADR-0006)
            "environment": env,
            "inputs": input_hashes,
            "tag": tag,
            "run_id": run_id,
            "started_at": started,
        }

        cached = self.log.find_cached(sig)
        if cached and all(self.store.exists(h) for h in cached["outputs"].values()):
            for key, h in cached["outputs"].items():
                ctx.register(Artifact(key, h, self.store.path_of(h), {}))
            record = base | {
                "status": "cached",
                "outputs": cached["outputs"],
                "metrics": cached.get("metrics", {}),
                "meta": cached.get("meta", {}),
                "ended_at": _now(),
            }
            self.log.write(record)
            return record
        if from_step is not None:
            raise RuntimeError(
                f"--from {from_step}: upstream step {step.name!r} is not cached for the "
                "current config; run without --from (cached steps are skipped anyway)"
            )

        self.tracker.start_run(base)
        try:
            step.run(ctx)
            missing = [k for k in step.outputs if k not in ctx.step_outputs]
            if missing:
                raise RuntimeError(f"step {step.name!r} did not put outputs: {missing}")
            record = base | {
                "status": "success",
                "outputs": {k: a.content_hash for k, a in ctx.step_outputs.items()},
                "metrics": ctx.step_metrics,
                "meta": ctx.step_meta,
                "ended_at": _now(),
            }
        except Exception as exc:
            # an agent debugging from manifests must see why, not just that (0007 A1)
            self.log.write(base | {
                "status": "failed", "outputs": {}, "metrics": {}, "ended_at": _now(),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            })
            self.tracker.end_run("failed")
            raise
        self.log.write(record)
        self.tracker.end_run("success")
        return record

    def lineage(self, content_hash: str) -> list[dict]:
        return self.log.lineage(content_hash)
