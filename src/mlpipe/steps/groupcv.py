"""Group k-fold CV plan — the unordered counterpart of cvplan's time-ordered plan.

Rows sharing a group (a customer, a patient, a site) always land in the same
fold, so no group straddles a train/valid split. With no group column every row
is its own group: plain shuffled k-fold. Same slot as cvplan (step name
"cvplan", output `fold_plan`), so a pipeline swaps one class for the other.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import Field

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import FoldPlan, Step, StepResult
from mlpipe.core.signature import canonical_json, sha256_bytes


class GroupKFoldConfig(StrictConfig):
    n_folds: int = Field(5, ge=2)
    group_column: str | None = None  # None: every row is its own group
    seed: int = 0  # shuffles groups across folds


def build_group_plan(groups: pl.Series | None, n_rows: int, cfg: GroupKFoldConfig) -> FoldPlan:
    if groups is None:
        unique, group_idx = list(range(n_rows)), np.arange(n_rows)
    else:
        unique = groups.unique(maintain_order=True).to_list()
        order = {g: i for i, g in enumerate(unique)}
        group_idx = groups.replace_strict(order, return_dtype=pl.Int64).to_numpy()
    if len(unique) < cfg.n_folds:
        raise ValueError(f"{len(unique)} groups cannot fill {cfg.n_folds} folds")

    fold_of_group = np.empty(len(unique), dtype=np.int64)
    fold_of_group[np.random.default_rng(cfg.seed).permutation(len(unique))] = (
        np.arange(len(unique)) % cfg.n_folds
    )
    row_fold = fold_of_group[group_idx]
    folds = [
        (np.flatnonzero(row_fold != k).astype(np.int32), np.flatnonzero(row_fold == k).astype(np.int32))
        for k in range(cfg.n_folds)
    ]
    plan_hash = sha256_bytes(canonical_json(
        {"kind": "group_kfold", "groups": unique if groups is not None else n_rows,
         "cv": cfg.model_dump()}
    ).encode())
    return FoldPlan(
        folds=folds,
        plan_hash=plan_hash,
        group_column=cfg.group_column,
        meta={"kind": "group_kfold", "n_groups": len(unique), "seed": cfg.seed},
    )


class GroupKFoldStep(Step):
    name = "cvplan"
    inputs = ["feature_table"]
    outputs = ["fold_plan"]
    config_model = GroupKFoldConfig

    def run(self, ctx):
        cfg = GroupKFoldConfig.model_validate(ctx.config.get(self.name, {}))
        lf = ctx.get_lazy("feature_table")
        n_rows = lf.select(pl.len()).collect().item()
        groups = lf.select(cfg.group_column).collect()[cfg.group_column] if cfg.group_column else None
        plan = build_group_plan(groups, n_rows, cfg)
        art = ctx.put(plan, "fold_plan", ext="joblib")
        ctx.log_meta("plan_hash", plan.plan_hash)
        ctx.log_meta("cv", {"n_rows": n_rows, **plan.meta, **cfg.model_dump()})
        return StepResult(outputs={"fold_plan": art})
