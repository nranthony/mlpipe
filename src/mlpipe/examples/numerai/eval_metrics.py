"""Era-wise evaluation metrics as plain functions (DESIGN.md §4: trivial tier —
no classes, swap by replacing the function)."""

from __future__ import annotations

import numpy as np
import polars as pl


def era_scores(
    df: pl.DataFrame, pred: str = "prediction", target: str = "target", era: str = "era"
) -> pl.DataFrame:
    """Per-era Pearson correlation of rank-percentiled predictions vs target."""
    return (
        df.lazy()
        .filter(pl.col(pred).is_not_null() & pl.col(target).is_not_null())
        .with_columns((pl.col(pred).rank() / pl.len()).over(era).alias("_ranked"))
        .group_by(era, maintain_order=True)
        .agg(pl.corr("_ranked", target).alias("corr"))
        .collect()
    )


def summarize(scores: pl.DataFrame) -> dict[str, float]:
    corrs = scores["corr"].drop_nulls().to_numpy()
    cumulative = np.cumsum(corrs)
    drawdown = float(np.max(np.maximum.accumulate(cumulative) - cumulative)) if len(corrs) else 0.0
    std = float(np.std(corrs, ddof=1)) if len(corrs) > 1 else float("nan")
    return {
        "mean_corr": float(np.mean(corrs)) if len(corrs) else float("nan"),
        "sharpe": float(np.mean(corrs) / std) if std and std > 0 else float("nan"),
        "max_drawdown": drawdown,
        "n_eras": len(corrs),
    }


def feature_exposure(
    features: pl.LazyFrame, preds: pl.Series, feature_cols: list[str]
) -> float:
    """Max absolute correlation between predictions and any single feature."""
    lf = features.with_columns(preds.alias("_pred")).filter(pl.col("_pred").is_not_null())
    row = lf.select(
        [pl.corr("_pred", f).abs().alias(f) for f in feature_cols]
    ).collect(engine="streaming").row(0)
    return float(np.nanmax(np.array(row, dtype=np.float64))) if row else 0.0
