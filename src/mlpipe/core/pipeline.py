"""Pipeline: ordered steps, cache hit/miss orchestration, manifest per execution.

Cache rule (DESIGN.md §1): a step is skipped iff a completed manifest exists for
its signature AND every output hash in that manifest is present in the store.
Cache hits still write a manifest (status "cached") — every execution leaves one.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mlpipe.core.context import NoopTracker, RunContext
from mlpipe.core.interfaces import Artifact, Step
from mlpipe.core.manifest import ManifestLog, git_commit
from mlpipe.core.signature import config_hash, sha256_bytes


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    """Apply --set key.path=value pairs (YAML-typed values) to a config copy."""
    config = copy.deepcopy(config)
    for item in overrides or []:
        keypath, _, raw = item.partition("=")
        node = config
        *parents, leaf = keypath.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return config


def environment_info(config: dict) -> dict:
    lock = Path("uv.lock")
    return {
        "lockfile_hash": sha256_bytes(lock.read_bytes()) if lock.exists() else None,
        "seed": config.get("seed"),
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
    ) -> None:
        self.steps = list(steps)
        self.store = store
        self.log = ManifestLog(manifests_dir)
        self.config = config or {}
        self.tracker = tracker or NoopTracker()

    def run(
        self,
        from_step: str | None = None,
        to_step: str | None = None,
        overrides: list[str] | None = None,
        tag: str | None = None,
        seeds: list[Artifact] | None = None,
    ) -> dict[str, Any]:
        config = apply_overrides(self.config, overrides)
        ctx = RunContext(config, self.store, self.tracker)
        for artifact in seeds or []:
            ctx.register(artifact)

        names = [s.name for s in self.steps]
        lo = names.index(from_step) if from_step else 0
        hi = names.index(to_step) + 1 if to_step else len(self.steps)

        # Upstream of --from is served from the latest manifests + store.
        latest = self.log.latest_outputs()
        for step in self.steps[:lo]:
            for key in step.outputs:
                if key not in latest:
                    raise RuntimeError(f"no cached artifact for upstream {key!r}")
                h = latest[key]
                ctx.register(Artifact(key, h, self.store.path_of(h), {}))

        written = [self._run_step(s, ctx, config, tag) for s in self.steps[lo:hi]]
        return {"artifacts": dict(ctx.registry), "manifests": written}

    def _run_step(self, step: Step, ctx: RunContext, config: dict, tag: str | None) -> dict:
        ctx.reset_step()
        started = _now()
        input_hashes = ctx.input_hashes(step.inputs)
        sig = step.signature(ctx)
        base = {
            "step": step.name,
            "signature": sig,
            "git_commit": git_commit(),
            "config_hash": config_hash(config, step.name),
            "environment": environment_info(config),
            "inputs": input_hashes,
            "tag": tag,
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
        except Exception:
            self.log.write(base | {"status": "failed", "outputs": {}, "metrics": {}, "ended_at": _now()})
            self.tracker.end_run("failed")
            raise
        self.log.write(record)
        self.tracker.end_run("success")
        return record

    def lineage(self, content_hash: str) -> list[dict]:
        return self.log.lineage(content_hash)
