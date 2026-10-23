"""Cycle 7 — evaluate (DESIGN.md block 7). Era-wise metrics via plain
functions, standalone plotly HTML report, metrics logged to the tracker.

Two scores, side by side (work/0004 F): out-of-fold CV on train (`mean_corr`,
...) and the full-data model on the validation holdout (`val_mean_corr`, ...).
OOF keeps the unprefixed names so existing comparisons and `tune.metric`
mean what they meant before.

The per-era scorer is configurable by import path (`evaluate.scorer:
"pkg.mod:fn"`, signature `fn(df, pred=, target=, era=) -> [era, corr]`). Its
module is declared through `code_deps`, so a scorer edit re-runs evaluate.

The full model trains on every train era, and the last `purge_eras` of those
carry labels that overlap the first `purge_eras` validation eras (a 60-day
target spans 12 weekly eras). Those validation eras are left out of `val_*`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import polars as pl
from pydantic import field_validator

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult
from mlpipe.examples.numerai.eval_metrics import feature_exposure, summarize
from mlpipe.steps.cvplan import era_order

RANK_PEARSON = "mlpipe.examples.numerai.eval_metrics:era_scores"


class EvalConfig(StrictConfig):
    target: str = "target"
    scorer: str = RANK_PEARSON  # "module:function"

    @field_validator("scorer")
    @classmethod
    def _import_path(cls, value: str) -> str:
        module, _, attr = value.partition(":")
        if not module or not attr:
            raise ValueError(f"scorer must be 'module:function', got {value!r}")
        return value


def load_scorer(path: str) -> Callable[..., pl.DataFrame]:
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)


def era_chart(series: dict[str, pl.DataFrame], title: str) -> str:
    import plotly.graph_objects as go

    traces = []
    for label, scores in series.items():
        corrs = scores["corr"].fill_null(0.0).to_list()
        eras = scores["era"].to_list()
        cumulative = [sum(corrs[: i + 1]) for i in range(len(corrs))]
        traces += [
            go.Bar(x=eras, y=corrs, name=f"{label} era corr"),
            go.Scatter(x=eras, y=cumulative, name=f"{label} cumulative", yaxis="y2"),
        ]
    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        yaxis2={"overlaying": "y", "side": "right"},
        template="plotly_white",
    )
    return fig.to_html(include_plotlyjs=True, full_html=True)


class EvaluateStep(Step):
    name = "evaluate"
    inputs = ["model", "oof_preds", "fold_plan", "feature_table", "feature_validation"]
    outputs = ["eval_report", "eval_chart"]
    config_model = EvalConfig

    def code_deps(self, cfg):
        """The scorer is imported by name at run time: declare its module."""
        return [EvalConfig.model_validate(cfg or {}).scorer.partition(":")[0]]

    def run(self, ctx):
        cfg = EvalConfig.model_validate(ctx.config.get(self.name, {}))
        era_scores = load_scorer(cfg.scorer)
        oof = ctx.get("oof_preds")
        plan = ctx.get("fold_plan")
        features_lf = ctx.get_lazy("feature_table")
        feature_cols = [
            c for c in features_lf.collect_schema().names() if c.startswith("feature_")
        ]

        scores = era_scores(oof, target=cfg.target)
        summary = summarize(scores)
        summary["feature_exposure"] = feature_exposure(
            features_lf, oof["prediction"], feature_cols
        )

        # holdout: the full-data model on validation rows it never saw, minus the
        # first purge_eras eras, whose labels overlap the last train eras'
        validation = ctx.get("feature_validation")
        skip = int(plan.meta.get("purge_eras", 0))
        val_eras = era_order(validation["era"].unique().to_list())
        validation = validation.filter(pl.col("era").is_in(val_eras[skip:]))
        val_preds = pl.Series(
            "prediction", ctx.get("model")["full"].predict(validation.select(feature_cols))
        )
        val_scores = era_scores(
            validation.select("era", cfg.target).with_columns(val_preds), target=cfg.target
        )
        holdout = summarize(val_scores)
        holdout["feature_exposure"] = feature_exposure(
            validation.lazy(), val_preds, feature_cols
        )
        summary |= {f"val_{k}": v for k, v in holdout.items()}

        for name, value in summary.items():
            ctx.log_metric(name, float(value))

        report = {
            "metrics": summary,
            "plan_hash": plan.plan_hash,
            "target": cfg.target,
            "scorer": cfg.scorer,
            "val_skipped_eras": val_eras[:skip],
            "era_scores": dict(zip(scores["era"].to_list(), scores["corr"].to_list())),
            "val_era_scores": dict(
                zip(val_scores["era"].to_list(), val_scores["corr"].to_list())
            ),
        }
        outs = {
            "eval_report": ctx.put(report, "eval_report", ext="json"),
            "eval_chart": ctx.put(
                era_chart(
                    {"oof": scores, "validation": val_scores},
                    f"era-wise corr (plan {plan.plan_hash[:8]})",
                ),
                "eval_chart",
                ext="html",
            ),
        }
        ctx.log_meta("plan_hash", plan.plan_hash)
        ctx.log_meta("scorer", cfg.scorer)
        return StepResult(outputs=outs)
