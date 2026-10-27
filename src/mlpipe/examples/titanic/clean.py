"""Titanic cleaning: the raw Kaggle columns become numeric `feature_*` columns,
and a seeded, stratified holdout is split off (Kaggle's `test.csv` has no labels,
so it cannot be the validation set). Nulls stay null where LightGBM can use them.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import Field

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult

TITLES = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3}  # anything else: 4 (rare)
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
EMBARKED = {"S": 0, "C": 1, "Q": 2}
DECKS = {d: i for i, d in enumerate("ABCDEFGT")}


class TitanicCleanConfig(StrictConfig):
    holdout_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    seed: int = 0


def features(raw: pl.DataFrame) -> pl.DataFrame:
    """The `feature_*` columns, keyed by PassengerId. Training and serving
    (`serve.prepare`) both call this; `Survived` is carried only when present,
    since Kaggle's `test.csv` has none."""
    label = [pl.col("Survived").cast(pl.Float64)] if "Survived" in raw.columns else []
    title = (
        pl.col("Name").str.extract(r",\s*([^.]+)\.", 1).str.strip_chars()
        .replace(TITLE_ALIASES)
    )
    family = pl.col("SibSp") + pl.col("Parch") + 1
    return raw.select(
        pl.col("PassengerId").cast(pl.Int64),
        *label,
        (pl.col("Sex") == "female").cast(pl.Int8).alias("feature_sex_female"),
        pl.col("Pclass").cast(pl.Int8).alias("feature_pclass"),
        # a "NaN" cell is missing too, the same as Kaggle's empty one
        pl.col("Age").cast(pl.Float64).fill_nan(None).alias("feature_age"),
        pl.col("SibSp").cast(pl.Int8).alias("feature_sibsp"),
        pl.col("Parch").cast(pl.Int8).alias("feature_parch"),
        pl.col("Fare").cast(pl.Float64).fill_nan(None).alias("feature_fare"),
        pl.col("Embarked").replace_strict(EMBARKED, default=None, return_dtype=pl.Int8)
        .alias("feature_embarked"),
        title.replace_strict(TITLES, default=4, return_dtype=pl.Int8).alias("feature_title"),
        family.cast(pl.Int8).alias("feature_family_size"),
        (family == 1).cast(pl.Int8).alias("feature_is_alone"),
        pl.col("Cabin").is_not_null().cast(pl.Int8).alias("feature_cabin_known"),
        pl.col("Cabin").str.slice(0, 1).replace_strict(DECKS, default=None, return_dtype=pl.Int8)
        .alias("feature_deck"),
    )


def stratified_holdout(target: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Boolean mask of holdout rows: `fraction` of each class, chosen by `seed`."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(target), dtype=bool)
    for value in np.unique(target):
        rows = np.flatnonzero(target == value)
        mask[rng.permutation(rows)[: round(len(rows) * fraction)]] = True
    return mask


class TitanicCleanStep(Step):
    name = "clean"
    inputs = ["raw_train"]
    outputs = ["clean_table", "clean_validation"]
    config_model = TitanicCleanConfig

    def run(self, ctx):
        cfg = TitanicCleanConfig.model_validate(ctx.config.get(self.name, {}))
        df = features(ctx.get("raw_train"))
        holdout = stratified_holdout(df["Survived"].to_numpy(), cfg.holdout_fraction, cfg.seed)
        train, validation = df.filter(~holdout), df.filter(holdout)
        ctx.log_meta("split", {
            "train_rows": train.height, "holdout_rows": validation.height,
            "train_survival": round(float(train["Survived"].mean()), 4),
            "holdout_survival": round(float(validation["Survived"].mean()), 4),
        })
        return StepResult(outputs={
            "clean_table": ctx.put(train, "clean_table"),
            "clean_validation": ctx.put(validation, "clean_validation"),
        })
