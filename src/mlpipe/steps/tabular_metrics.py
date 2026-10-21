"""Row-wise metrics for plain tabular tasks, as plain functions (DESIGN.md §4:
trivial tier). The era-wise Numerai metrics live in eval_metrics."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def _resolved(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    keep = ~(np.isnan(y) | np.isnan(pred))  # unresolved targets, rows in no valid fold
    return y[keep], pred[keep]


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y, pred = _resolved(y, pred)
    return {
        "rmse": float(np.sqrt(skm.mean_squared_error(y, pred))),
        "mae": float(skm.mean_absolute_error(y, pred)),
        "r2": float(skm.r2_score(y, pred)),
        "corr": float(np.corrcoef(pred, y)[0, 1]),
        "n_rows": len(y),
    }


def binary_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """`proba` must be P(y == 1). A regressor's raw scores are rejected rather
    than silently producing a meaningless log-loss."""
    y, proba = _resolved(y, proba)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("binary metrics need a 0/1 target")
    if proba.min() < 0.0 or proba.max() > 1.0:
        raise ValueError(
            "binary metrics need probabilities in [0, 1]; use a classifier backend "
            "(e.g. train.model.kind = lightgbm_binary)"
        )
    return {
        "roc_auc": float(skm.roc_auc_score(y, proba)),
        "pr_auc": float(skm.average_precision_score(y, proba)),
        "log_loss": float(skm.log_loss(y, proba, labels=[0.0, 1.0])),
        "brier": float(skm.brier_score_loss(y, proba)),
        "accuracy": float(skm.accuracy_score(y, proba >= 0.5)),
        "n_rows": len(y),
    }


TASKS = {"regression": regression_metrics, "binary": binary_metrics}
