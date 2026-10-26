"""Plotly building blocks shared by the profile and evaluate steps (ADR-0010).

Each chart is computed from artifacts the step already declared, and the page
is a store artifact (`ext="html"`) that the pipeline also hands to the tracker.
Pages are self-contained (plotly.js inline once), so they open offline. plotly
is imported inside functions: it ships in the `track` extra, not the base deps.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import polars as pl

# metrics where larger is better; everything else (losses, errors) is minimized
HIGHER_IS_BETTER = ("auc", "ndcg", "map", "average_precision", "accuracy", "corr", "r2")

PALETTE = ["#2a6fdb", "#e0663a", "#2e9e6a", "#9b59b6", "#c49a00", "#5a6b7d", "#d6457a"]


def higher_is_better(metric: str) -> bool:
    return any(token in metric for token in HIGHER_IS_BETTER)


def page(title: str, sections: list[tuple[str, Any]], facts: dict[str, Any] | None = None) -> str:
    """One HTML page: a title, a facts table, then (heading, figure) sections."""
    parts, first = [], True
    for i, (heading, fig) in enumerate(sections):
        fig.update_layout(template="plotly_white", margin={"l": 50, "r": 20, "t": 40, "b": 40})
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        # fixed div ids: plotly's default is random, which would give the same
        # chart a new content hash on every run
        parts.append(fig.to_html(include_plotlyjs=first, full_html=False, div_id=f"chart{i}"))
        first = False
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in (facts or {}).items()
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;"
        "color:#1d2733;background:#fff}h1{font-size:1.5rem}h2{font-size:1.1rem;margin-top:2rem}"
        "table{border-collapse:collapse;font-size:.9rem}th,td{padding:4px 12px;text-align:left;"
        "border-bottom:1px solid #e3e7ec}th{color:#5a6b7d;font-weight:500}"
        f"</style></head><body><h1>{html.escape(title)}</h1><table>{rows}</table>"
        + "".join(parts) + "</body></html>"
    )


def curve_summary(history: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Per metric, the fold-mean validation curve's best iteration and value,
    and its final value: how far past the best the configured rounds ran."""
    out: dict[str, dict[str, float]] = {}
    valid = history.filter(pl.col("split") == "valid")
    for metric in valid["metric"].unique(maintain_order=True).to_list():
        mean = (valid.filter(pl.col("metric") == metric).group_by("iteration")
                .agg(pl.col("value").mean()).sort("iteration"))
        values = mean["value"].to_numpy()
        best = int(np.argmax(values) if higher_is_better(metric) else np.argmin(values))
        out[metric] = {
            "best_iteration": int(mean["iteration"][best]),
            "best_value": float(values[best]),
            "final_iteration": int(mean["iteration"][-1]),
            "final_value": float(values[-1]),
        }
    return out


def curves_figure(history: pl.DataFrame):
    """One panel per metric: each fold's train (dotted) and valid (solid) curves,
    plus the fold-mean valid curve in black with its best iteration marked."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metrics = history["metric"].unique(maintain_order=True).to_list()
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)
    summary = curve_summary(history)
    for col, metric in enumerate(metrics, start=1):
        sub = history.filter(pl.col("metric") == metric)
        for fold in sorted(sub["fold"].unique().to_list()):
            color = PALETTE[fold % len(PALETTE)]
            for split, dash in (("train", "dot"), ("valid", "solid")):
                s = sub.filter((pl.col("fold") == fold) & (pl.col("split") == split))
                fig.add_trace(go.Scatter(
                    x=s["iteration"], y=s["value"], mode="lines", name=f"fold {fold} {split}",
                    legendgroup=f"{fold}{split}", showlegend=col == 1,
                    line={"color": color, "dash": dash, "width": 1}, opacity=0.7,
                ), row=1, col=col)
        mean = (sub.filter(pl.col("split") == "valid").group_by("iteration")
                .agg(pl.col("value").mean()).sort("iteration"))
        fig.add_trace(go.Scatter(x=mean["iteration"], y=mean["value"], mode="lines",
                                 name="valid mean", legendgroup="mean", showlegend=col == 1,
                                 line={"color": "#1d2733", "width": 2.5}), row=1, col=col)
        best = summary[metric]
        fig.add_vline(x=best["best_iteration"], line_dash="dash", line_color="#5a6b7d",
                      row=1, col=col)
        fig.update_xaxes(title_text="iteration", row=1, col=col)
    fig.update_layout(height=420)
    return fig


def importance_figure(importance: pl.DataFrame, top: int = 30):
    """Mean gain over the fold models, with the spread across folds as error bars."""
    import plotly.graph_objects as go

    folds = importance.filter(pl.col("model") != "full")
    stats = (folds.group_by("feature")
             .agg(pl.col("gain").mean().alias("mean"), pl.col("gain").std().alias("std"))
             .sort("mean", descending=True).head(top).sort("mean"))
    fig = go.Figure(go.Bar(
        x=stats["mean"], y=stats["feature"], orientation="h", marker_color=PALETTE[0],
        error_x={"type": "data", "array": stats["std"].fill_null(0.0).to_list()},
    ))
    fig.update_layout(height=max(300, 22 * stats.height + 80), xaxis_title="gain (mean over folds)")
    return fig
