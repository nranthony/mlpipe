"""LightGBM adapter for the ModelBackend port (DESIGN.md §3).

The only module (besides tests) allowed to import lightgbm. Port signatures
mention only our types: polars, numpy, Path, FoldPlan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from mlpipe.core.interfaces import FoldPlan

DEFAULTS = {
    "objective": "regression",
    "random_state": 42,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}


class LightGBMBackend:
    def __init__(self) -> None:
        self._model = None

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        params: dict[str, Any],
        folds: FoldPlan | None = None,
    ) -> None:
        from lightgbm import LGBMRegressor

        self._model = LGBMRegressor(**{**DEFAULTS, **params})
        self._model.fit(X.to_numpy().astype(np.float32), y)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self._model.predict(X.to_numpy().astype(np.float32))

    def save(self, path: Path) -> None:
        joblib.dump(self._model, path)

    def load(self, path: Path) -> None:
        self._model = joblib.load(path)
