"""Cycle 5 — split / CV plan (DESIGN.md block 5).

Era-aware, purged, embargoed CV. The FoldPlan is an artifact with its own
plan_hash over (era list, cv config) — model comparisons must cite the same
plan hash, and assert_same_plan() is the guard that enforces it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel

from mlpipe.core.interfaces import FoldPlan, Step, StepResult
from mlpipe.core.signature import canonical_json, sha256_bytes


class CVConfig(BaseModel):
    n_folds: int = 4
    purge_eras: int = 4    # eras dropped from train on both sides of the valid block
    embargo_eras: int = 4  # extra eras dropped after the valid block (target overlap)
    era_column: str = "era"


def build_plan(eras: pl.Series, cfg: CVConfig) -> FoldPlan:
    unique = eras.unique(maintain_order=True).to_list()
    order = {era: i for i, era in enumerate(unique)}
    era_idx = eras.replace_strict(order, return_dtype=pl.Int32).to_numpy()

    blocks = np.array_split(np.arange(len(unique)), cfg.n_folds)
    folds = []
    for block in blocks:
        lo, hi = block[0], block[-1]
        valid = np.flatnonzero((era_idx >= lo) & (era_idx <= hi)).astype(np.int32)
        train_mask = (era_idx < lo - cfg.purge_eras) | (
            era_idx > hi + cfg.purge_eras + cfg.embargo_eras
        )
        folds.append((np.flatnonzero(train_mask).astype(np.int32), valid))

    plan_hash = sha256_bytes(
        canonical_json({"eras": unique, "cv": cfg.model_dump()}).encode()
    )
    return FoldPlan(
        folds=folds,
        era_column=cfg.era_column,
        purge_eras=cfg.purge_eras,
        embargo_eras=cfg.embargo_eras,
        plan_hash=plan_hash,
    )


def assert_same_plan(manifest_a: dict, manifest_b: dict) -> str:
    """Comparisons are only valid on identical fold plans — fail loud otherwise."""
    a = manifest_a.get("meta", {}).get("plan_hash")
    b = manifest_b.get("meta", {}).get("plan_hash")
    if not a or a != b:
        raise ValueError(
            f"comparison invalid: fold-plan hashes differ or are missing ({a} vs {b})"
        )
    return a


class CVPlanStep(Step):
    name = "cvplan"
    inputs = ["feature_table"]
    outputs = ["fold_plan"]
    config_model = CVConfig

    def run(self, ctx):
        cfg = CVConfig.model_validate(ctx.config.get(self.name, {}))
        eras = ctx.get_lazy("feature_table").select(cfg.era_column).collect()[cfg.era_column]
        plan = build_plan(eras, cfg)
        art = ctx.put(plan, "fold_plan", ext="joblib")
        ctx.log_meta("plan_hash", plan.plan_hash)
        ctx.log_meta("cv", {"n_eras": eras.n_unique(), **cfg.model_dump()})
        return StepResult(outputs={"fold_plan": art})
