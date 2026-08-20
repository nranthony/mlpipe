"""Cycle 7 — evaluate (DESIGN.md block 7). Era-wise metrics via plain
functions, standalone plotly HTML report, metrics logged to the tracker."""

from __future__ import annotations

from pydantic import BaseModel

from mlpipe.core.interfaces import Step, StepResult
from mlpipe.steps.eval_metrics import era_scores, feature_exposure, summarize


class EvalConfig(BaseModel):
    target: str = "target"


def era_chart(scores, title: str) -> str:
    import plotly.graph_objects as go

    corrs = scores["corr"].to_list()
    eras = scores["era"].to_list()
    cumulative = [sum(corrs[: i + 1]) for i in range(len(corrs))]
    fig = go.Figure(
        [
            go.Bar(x=eras, y=corrs, name="era corr"),
            go.Scatter(x=eras, y=cumulative, name="cumulative", yaxis="y2"),
        ]
    )
    fig.update_layout(
        title=title,
        yaxis2={"overlaying": "y", "side": "right"},
        template="plotly_white",
    )
    return fig.to_html(include_plotlyjs=True, full_html=True)


class EvaluateStep(Step):
    name = "evaluate"
    inputs = ["model", "oof_preds", "fold_plan", "feature_table"]
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
        for name, value in summary.items():
            ctx.log_metric(name, float(value))

        report = {
            "metrics": summary,
            "plan_hash": plan.plan_hash,
            "target": cfg.target,
            "era_scores": dict(zip(scores["era"].to_list(), scores["corr"].to_list())),
        }
        outs = {
            "eval_report": ctx.put(report, "eval_report", ext="json"),
            "eval_chart": ctx.put(
                era_chart(scores, f"era-wise corr (plan {plan.plan_hash[:8]})"),
                "eval_chart",
                ext="html",
            ),
        }
        ctx.log_meta("plan_hash", plan.plan_hash)
        return StepResult(outputs=outs)
