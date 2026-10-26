"""Evaluate for plain tabular tasks — row-wise, where a domain pipeline's evaluate
may score per group. Same slot (step name "evaluate", output `eval_report`), same
two scores side by side: out-of-fold (`rmse`, `roc_auc`, ...) and the full-data
model on the validation holdout (`val_rmse`, `val_roc_auc`, ...).

It also writes `eval_chart` (ADR-0010): OOF and holdout charts for the task, the
training curves and the feature importances. The report carries the curve summary
(best vs final iteration per metric), so an agent sees overfitting without the chart.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult
from mlpipe.steps import charts
from mlpipe.steps.tabular_metrics import TASKS


class TabularEvalConfig(StrictConfig):
    task: Literal["regression", "binary"] = "regression"
    target: str = "target"


def _binary_sections(scored: dict[str, tuple[np.ndarray, np.ndarray]]) -> list:
    import plotly.graph_objects as go
    from sklearn import metrics as skm

    roc, pr, calib = go.Figure(), go.Figure(), go.Figure()
    for (name, (y, p)), color in zip(scored.items(), charts.PALETTE, strict=False):
        fpr, tpr, _ = skm.roc_curve(y, p)
        roc.add_trace(go.Scatter(x=fpr, y=tpr, name=name, line={"color": color}))
        prec, rec, _ = skm.precision_recall_curve(y, p)
        pr.add_trace(go.Scatter(x=rec, y=prec, name=name, line={"color": color}))
        # 10 quantile bins: mean prediction vs observed rate
        order = np.argsort(p)
        bins = [b for b in np.array_split(order, 10) if len(b)]
        calib.add_trace(go.Scatter(
            x=[p[b].mean() for b in bins], y=[y[b].mean() for b in bins], name=name,
            mode="lines+markers", line={"color": color},
        ))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], showlegend=False, line={"dash": "dot", "color": "#9aa5b1"}))
    calib.add_trace(go.Scatter(x=[0, 1], y=[0, 1], showlegend=False, line={"dash": "dot", "color": "#9aa5b1"}))
    roc.update_layout(xaxis_title="false positive rate", yaxis_title="true positive rate", height=400)
    pr.update_layout(xaxis_title="recall", yaxis_title="precision", height=400)
    calib.update_layout(xaxis_title="mean predicted probability", yaxis_title="observed rate", height=400)

    y, p = scored["holdout"]
    cm = skm.confusion_matrix(y, p >= 0.5, labels=[0, 1])
    confusion = go.Figure(go.Heatmap(
        z=cm, x=["pred 0", "pred 1"], y=["true 0", "true 1"], text=cm, texttemplate="%{text}",
        colorscale="Blues", showscale=False,
    ))
    confusion.update_layout(height=320, width=420, yaxis_autorange="reversed")
    spread = go.Figure([
        go.Histogram(x=p[y == label], name=f"true {label:g}", opacity=0.6, nbinsx=30,
                     marker_color=charts.PALETTE[i])
        for i, label in enumerate((0.0, 1.0))
    ])
    spread.update_layout(barmode="overlay", xaxis_title="predicted probability", height=350)
    return [("ROC curve", roc), ("Precision-recall", pr), ("Calibration", calib),
            ("Holdout confusion matrix (threshold 0.5)", confusion),
            ("Holdout predictions by true class", spread)]


def _regression_sections(scored: dict[str, tuple[np.ndarray, np.ndarray]]) -> list:
    import plotly.graph_objects as go

    fit, residuals = go.Figure(), go.Figure()
    for (name, (y, p)), color in zip(scored.items(), charts.PALETTE, strict=False):
        fit.add_trace(go.Scattergl(x=y, y=p, mode="markers", name=name,
                                   marker={"color": color, "size": 4, "opacity": 0.5}))
        residuals.add_trace(go.Histogram(x=p - y, name=name, opacity=0.6, marker_color=color))
    lo = min(float(np.min(y)) for y, _ in scored.values())
    hi = max(float(np.max(y)) for y, _ in scored.values())
    fit.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], showlegend=False, line={"dash": "dot", "color": "#9aa5b1"}))
    fit.update_layout(xaxis_title="actual", yaxis_title="predicted", height=420)
    residuals.update_layout(barmode="overlay", xaxis_title="prediction − actual", height=350)
    return [("Predicted vs actual", fit), ("Residuals", residuals)]


def _resolved(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = ~(np.isnan(y) | np.isnan(p))
    return y[keep].astype(np.float64), p[keep].astype(np.float64)


class TabularEvalStep(Step):
    name = "evaluate"
    inputs = ["model", "oof_preds", "fold_plan", "feature_validation", "train_history",
              "feature_importance"]
    outputs = ["eval_report", "eval_chart"]
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

        history = ctx.get("train_history")
        curves = charts.curve_summary(history) if history.height else {}
        for name, value in summary.items():
            ctx.log_metric(name, float(value))
        for metric, s in curves.items():
            ctx.log_metric(f"best_iteration_{metric}", s["best_iteration"])
        report = {
            "metrics": summary,
            "plan_hash": plan.plan_hash,
            "target": cfg.target,
            "task": cfg.task,
            "curves": curves,
        }
        ctx.log_meta("plan_hash", plan.plan_hash)

        scored = {
            "oof": _resolved(oof[cfg.target].to_numpy(), oof["prediction"].to_numpy()),
            "holdout": _resolved(validation[cfg.target].to_numpy(), np.asarray(val_pred)),
        }
        sections = (_binary_sections if cfg.task == "binary" else _regression_sections)(scored)
        if history.height:
            heading = ("Training curves (dotted: train, solid: valid, dashed line: best "
                       "mean-valid iteration)")
            sections.append((heading, charts.curves_figure(history)))
        importance = ctx.get("feature_importance")
        if importance.height:
            sections.append(("Feature importance", charts.importance_figure(importance)))
        facts = {"task": cfg.task, "target": cfg.target, "fold plan": plan.plan_hash[:12]}
        facts |= {k: f"{summary[k]:.4f}  (holdout {summary['val_' + k]:.4f})"
                  for k in summary if not k.startswith("val_") and k != "n_rows"}
        facts |= {f"best iteration ({m})": f"{s['best_iteration']} of {s['final_iteration']}"
                  for m, s in curves.items()}
        chart = charts.page(f"Evaluation — {cfg.target} ({cfg.task})", sections, facts)
        return StepResult(outputs={
            "eval_report": ctx.put(report, "eval_report", ext="json"),
            "eval_chart": ctx.put(chart, "eval_chart", ext="html"),
        })
