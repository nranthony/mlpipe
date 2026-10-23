"""Era-wise evaluation metrics as plain functions (DESIGN.md §4: trivial tier —
no classes, swap by replacing the function)."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import polars as pl

_INV_NORM = np.frompyfunc(NormalDist().inv_cdf, 1, 1)  # stdlib: scipy is no dependency


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


def _gaussianize(s: pl.Series) -> pl.Series:
    return pl.Series(_INV_NORM(s.to_numpy()).astype(np.float64))


def _pow_1_5(col: str) -> pl.Expr:
    return pl.col(col).sign() * pl.col(col).abs().pow(1.5)


def numerai_corr_scores(
    df: pl.DataFrame, pred: str = "prediction", target: str = "target", era: str = "era"
) -> pl.DataFrame:
    """Per-era `numerai_corr`, ported from numerai-tools: predictions tie-kept
    ranked to (rank - 0.5) / n and gaussianised; the target centred; both raised
    to a signed power 1.5, then Pearson. The tails weigh more, as in the payout.
    Payout-accurate scoring (NCORR, MMC) is the host's job, not this example's."""
    return (
        df.lazy()
        .filter(pl.col(pred).is_not_null() & pl.col(target).is_not_null())
        .with_columns(
            ((pl.col(pred).rank("average") - 0.5) / pl.len()).over(era).alias("_u"),
            (pl.col(target) - pl.col(target).mean()).over(era).alias("_t"),
        )
        .with_columns(pl.col("_u").map_batches(_gaussianize, return_dtype=pl.Float64).alias("_g"))
        .with_columns(_pow_1_5("_g").alias("_g"), _pow_1_5("_t").alias("_t"))
        .group_by(era, maintain_order=True)
        .agg(pl.corr("_g", "_t").alias("corr"))
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
