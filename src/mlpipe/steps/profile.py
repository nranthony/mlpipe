"""Generic data profile of the cleaned training table (ADR-0010): per-column facts as
JSON (`data_profile`) and a chart page (`profile_chart`). Nothing consumes its
outputs, so it never invalidates training. It reads only `feature_*` columns and
the target, capped at `max_features` columns for wide tables.

Charts: distributions split by target class (binary) or plain (otherwise), target
rate per binned feature, and a correlation heatmap.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import Field

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult
from mlpipe.steps import charts


class ProfileConfig(StrictConfig):
    target: str | None = None
    max_features: int = Field(40, ge=1)
    bins: int = Field(10, ge=2)  # quantile bins for the target-rate chart


def column_facts(s: pl.Series) -> dict:
    facts = {"dtype": str(s.dtype), "null_rate": round(s.null_count() / max(len(s), 1), 4),
             "n_unique": s.n_unique()}
    if s.dtype.is_numeric():
        v = s.drop_nulls().cast(pl.Float64)
        if len(v):
            facts |= {"min": v.min(), "max": v.max(), "mean": round(float(v.mean()), 6),
                      "std": round(float(v.std() or 0.0), 6)}
    return facts


def binned(s: pl.Series, bins: int) -> pl.DataFrame:
    """Columns `order` (sort key) and `bin` (label). Few distinct values: the values
    themselves. Otherwise: quantile bins labelled by range. Nulls bin last."""
    if s.n_unique() <= bins:
        order = s.rank("dense").cast(pl.Float64)
        label = s.cast(pl.Utf8)
    else:
        cut = s.qcut(bins, allow_duplicates=True)
        order = cut.to_physical().cast(pl.Float64)
        label = cut.cast(pl.Utf8)
    return pl.DataFrame({"order": order.fill_null(float("inf")), "bin": label.fill_null("null")})


def correlations(df: pl.DataFrame) -> np.ndarray:
    """Pairwise Pearson over rows where both columns are present."""
    m = df.cast(pl.Float64).to_numpy()
    n = m.shape[1]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            keep = ~(np.isnan(m[:, i]) | np.isnan(m[:, j]))
            a, b = m[keep, i], m[keep, j]
            if keep.sum() > 2 and a.std() > 0 and b.std() > 0:
                out[i, j] = out[j, i] = float(np.corrcoef(a, b)[0, 1])
    return out


class ProfileStep(Step):
    name = "profile"
    inputs = ["clean_table"]
    outputs = ["data_profile", "profile_chart"]
    config_model = ProfileConfig

    def run(self, ctx):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        cfg = ProfileConfig.model_validate(ctx.config.get(self.name, {}))
        df = ctx.get("clean_table")
        feats = [c for c in df.columns if c.startswith("feature_")][: cfg.max_features]
        target = df[cfg.target] if cfg.target else None
        binary = target is not None and set(target.drop_nulls().unique().to_list()) <= {0, 1}

        profile = {
            "rows": df.height,
            "columns": {c: column_facts(df[c]) for c in [*feats, *([cfg.target] if cfg.target else [])]},
            "features_shown": len(feats),
            "features_total": sum(c.startswith("feature_") for c in df.columns),
        }
        sections = []

        cols = 3
        rows = -(-len(feats) // cols)
        dist = make_subplots(rows=rows, cols=cols, subplot_titles=feats)
        for k, f in enumerate(feats):
            r, c = divmod(k, cols)
            groups = ([(f"{cfg.target}={v:g}", df.filter(pl.col(cfg.target) == v)[f]) for v in (0, 1)]
                      if binary else [("all", df[f])])
            for g, (label, values) in enumerate(groups):
                dist.add_trace(go.Histogram(
                    x=values.drop_nulls(), name=label, legendgroup=label, showlegend=k == 0,
                    marker_color=charts.PALETTE[g], opacity=0.6, nbinsx=30,
                ), row=r + 1, col=c + 1)
        dist.update_layout(barmode="overlay", height=230 * rows)
        sections.append(("Distributions" + (f" by {cfg.target}" if binary else ""), dist))

        if target is not None:
            rate = make_subplots(rows=rows, cols=cols, subplot_titles=feats)
            rates = {}
            for k, f in enumerate(feats):
                r, c = divmod(k, cols)
                agg = (binned(df[f], cfg.bins).with_columns(y=target)
                       .group_by("order", "bin")
                       .agg(pl.col("y").mean().alias("rate"), pl.len().alias("n")).sort("order"))
                rates[f] = {b: {"rate": round(float(t), 4), "n": n}
                            for b, t, n in agg.select("bin", "rate", "n").iter_rows()}
                rate.add_trace(go.Bar(x=agg["bin"], y=agg["rate"], text=agg["n"],
                                      marker_color=charts.PALETTE[0], showlegend=False,
                                      hovertemplate="%{x}<br>rate %{y:.3f}<br>n %{text}<extra></extra>"),
                               row=r + 1, col=c + 1)
            rate.update_layout(height=230 * rows)
            profile["target_rate_by_bin"] = rates
            sections.append((f"Mean {cfg.target} per bin (hover for row counts)", rate))

        names = [*feats, *([cfg.target] if cfg.target else [])]
        corr = correlations(df.select(names))
        heat = go.Figure(go.Heatmap(z=corr, x=names, y=names, zmin=-1, zmax=1,
                                    colorscale="RdBu", reversescale=True))
        heat.update_layout(height=max(400, 22 * len(names) + 150), yaxis_autorange="reversed")
        sections.append(("Correlation (pairwise complete rows)", heat))

        nulls = {c: f["null_rate"] for c, f in profile["columns"].items() if f["null_rate"]}
        facts = {"rows": df.height, "features": f"{len(feats)} shown of {profile['features_total']}",
                 "target": cfg.target or "none",
                 "columns with nulls": ", ".join(f"{c} ({r:.1%})" for c, r in nulls.items()) or "none"}
        ctx.log_meta("profile", {"rows": df.height, "features_shown": len(feats)})
        return StepResult(outputs={
            "data_profile": ctx.put(profile, "data_profile", ext="json"),
            "profile_chart": ctx.put(charts.page("Data profile", sections, facts), "profile_chart",
                                     ext="html"),
        })
