"""Evaluate for plain tabular tasks — row-wise, where a domain pipeline's evaluate
may score per group. Same slot (step name "evaluate", output `eval_report`), same
two scores side by side: out-of-fold (`rmse`, `roc_auc`, ...) and the full-data
model on the validation holdout (`val_rmse`, `val_roc_auc`, ...).
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult
from mlpipe.steps.tabular_metrics import TASKS


class TabularEvalConfig(StrictConfig):
    task: Literal["regression", "binary"] = "regression"
    target: str = "target"


class TabularEvalStep(Step):
    name = "evaluate"
    inputs = ["model", "oof_preds", "fold_plan", "feature_validation"]
    outputs = ["eval_report"]
    config_model = TabularEvalConfig

    def run(self, ctx):
        cfg = TabularEvalConfig.model_validate(ctx.config.get(self.name, {}))
        score = TASKS[cfg.task]
        oof = ctx.get("oof_preds")
        plan = ctx.get("fold_plan")
        summary = score(oof[cfg.target].to_numpy(), oof["prediction"].to_numpy())

        validation = ctx.get("feature_validation").filter(pl.col(cfg.target).is_not_null())
        feats = [c for c in validation.columns if c.startswith("feature_")]
        val_pred = ctx.get("model")["full"].predict(validation.select(feats))
        holdout = score(validation[cfg.target].to_numpy(), val_pred)
        summary |= {f"val_{k}": v for k, v in holdout.items()}

        for name, value in summary.items():
            ctx.log_metric(name, float(value))
        report = {
            "metrics": summary,
            "plan_hash": plan.plan_hash,
            "target": cfg.target,
            "task": cfg.task,
        }
        ctx.log_meta("plan_hash", plan.plan_hash)
        return StepResult(outputs={"eval_report": ctx.put(report, "eval_report", ext="json")})
