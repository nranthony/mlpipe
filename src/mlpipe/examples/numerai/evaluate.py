"""Cycle 7 — evaluate (DESIGN.md block 7). Era-wise metrics via plain
functions, standalone plotly HTML report, metrics logged to the tracker.

Two scores, side by side (work/0004 F): out-of-fold CV on train (`mean_corr`,
...) and the full-data model on the validation holdout (`val_mean_corr`, ...).
OOF keeps the unprefixed names so existing comparisons and `tune.metric`
mean what they meant before.
"""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel

from mlpipe.core.interfaces import Step, StepResult
from mlpipe.examples.numerai.eval_metrics import era_scores, feature_exposure, summarize


class EvalConfig(BaseModel):
    target: str = "target"


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

    def run(self, ctx):
        cfg = EvalConfig.model_validate(ctx.config.get(self.name, {}))
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

        # holdout: the full-data model on validation rows it never saw
        validation = ctx.get("feature_validation")
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
        return StepResult(outputs=outs)
