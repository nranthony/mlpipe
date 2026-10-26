"""LightGBM adapter for the ModelBackend port (DESIGN.md §3).

The only module (besides tests) allowed to import lightgbm. Port signatures
mention only our types: polars, numpy. Implements the optional CurveBackend
members (ADR-0010): per-round history on an eval set, and gain importances.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

DEFAULTS = {
    "objective": "regression",
    "random_state": 42,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}


def _matrix(X: pl.DataFrame) -> np.ndarray:
    return X.to_numpy().astype(np.float32)


class LightGBMBackend:
    def __init__(self) -> None:
        self._model = None
        self._history: dict[str, list[float]] = {}

    def _estimator(self, params: dict[str, Any]):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**{**DEFAULTS, **params})

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        params: dict[str, Any],
        eval_set: tuple[pl.DataFrame, np.ndarray] | None = None,
    ) -> None:
        from lightgbm import record_evaluation

        self._model = self._estimator(params)
        self._columns = X.columns
        train = _matrix(X)
        if eval_set is None:
            self._model.fit(train, y)
            self._history = {}
            return
        # Passing the training arrays themselves lets LightGBM reuse its own
        # scores for the train curve rather than predicting the rows again.
        recorded: dict[str, dict[str, list[float]]] = {}
        self._model.fit(
            train, y,
            eval_X=(train, _matrix(eval_set[0])),
            eval_y=(y, eval_set[1]),
            eval_names=["train", "valid"],
            callbacks=[record_evaluation(recorded)],
        )
        self._history = {
            f"{split}_{metric}": [float(v) for v in values]
            for split, metrics in recorded.items()
            for metric, values in metrics.items()
        }

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self._model.predict(_matrix(X))

    def history(self) -> dict[str, list[float]]:
        return dict(self._history)

    def feature_importance(self) -> dict[str, float] | None:
        gain = self._model.booster_.feature_importance(importance_type="gain")
        return {c: float(g) for c, g in zip(self._columns, gain, strict=True)}


class LightGBMBinaryBackend(LightGBMBackend):
    """Binary classifier. `predict` returns P(y == 1), so the port stays one
    np.ndarray per row and probability metrics (log-loss, Brier) are valid."""

    def _estimator(self, params: dict[str, Any]):
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**{**DEFAULTS, "objective": "binary", **params})

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self._model.predict_proba(_matrix(X))[:, 1]
