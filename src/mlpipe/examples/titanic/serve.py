"""Titanic serving for `mlpipe predict` (ADR-0007): ingest_test -> predict
(see pipeline.build_serve_steps).

Kaggle's `test.csv` has no `Survived`. `prepare` runs it through clean's own
`features`, so serving derives exactly what training did (and clean.py is in the
predict step's code closure). `finish` turns P(Survived=1) into Kaggle's
submission shape, `PassengerId, Survived` (0/1).
"""

from __future__ import annotations

from typing import Any

import polars as pl

from mlpipe.examples.titanic.clean import features

# P(Survived=1) above this is a 1. A tie is a 0, as in a classifier's predict().
THRESHOLD = 0.5


def prepare(frame: pl.DataFrame, bundle: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(features in bundle order, PassengerId)."""
    derived = features(frame)
    return derived.select(bundle["features"]), derived.select("PassengerId")


def finish(scored: pl.DataFrame, bundle: dict[str, Any]) -> pl.DataFrame:
    """PassengerId + prediction -> the submission's PassengerId, Survived (0/1)."""
    if bundle.get("target") != "Survived":
        raise ValueError(f"titanic finish thresholds a Survived model, got target {bundle.get('target')!r}")
    return scored.select(
        pl.col("PassengerId").cast(pl.Int64),
        (pl.col("prediction") > THRESHOLD).cast(pl.Int64).alias("Survived"),
    )
