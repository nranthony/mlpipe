"""The real Numerai pipeline, plus its serving hooks for `mlpipe predict`.

    mlpipe run --pipeline numerai
    mlpipe predict --pipeline numerai            # latest live snapshot
"""

from __future__ import annotations

from typing import Any

import polars as pl

from mlpipe.core.interfaces import Step
from mlpipe.examples.numerai.clean import CleanStep
from mlpipe.examples.numerai.evaluate import EvaluateStep
from mlpipe.examples.numerai.ingest import IngestStep
from mlpipe.examples.numerai.validate import ValidateStep
from mlpipe.steps.cvplan import CVPlanStep
from mlpipe.steps.features import FeatureStep
from mlpipe.steps.register import RegisterStep
from mlpipe.steps.train import TrainStep

DEFAULT_CONFIG = {
    "download_dir": "/workspace/raw_data/numerai",  # shared profile-wide cache
    # v6.0 (2026-10): v5.x training files are no longer served. The era column is
    # `date`, and `medium` = 350 Int8 + the String `feature_country`.
    "ingest": {"version": "v6.0"},
    "validate": {
        "row_tolerance": 0.2,
        "max_feature_null_frac": 0.0,  # Int8 (and String): dense, so strict
        "max_feature_null_frac_by_dtype": {"Float32": 0.1},  # factor features: ≤7.1% seen
        "era_column": "date",
        "feature_dtypes": ["Int8", "Float32", "String"],
    },
    "clean": {
        "cleaner": {"kind": "baseline", "feature_set": "medium"},
        "era_column": "date",
        "exclude_features": ["feature_country"],  # String; encoding it is model work
    },
    "features": {"transformer": {"kind": "passthrough"}},
    # `target` is target_jupiter_60 in v6.0: 60 business days overlap 12 weekly eras.
    "cvplan": {"n_folds": 4, "purge_eras": 12, "embargo_eras": 4},
    "train": {
        "model": {
            "kind": "lightgbm",
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 6,
                "colsample_bytree": 0.1,
                "n_jobs": 16,
            },
        },
        "target": "target",
    },
    # numerai_corr, not rank-Pearson: ranks models the way the payout does
    "evaluate": {
        "target": "target",
        "scorer": "mlpipe.examples.numerai.eval_metrics:numerai_corr_scores",
    },
    "register": {"model_name": "numerai-lgbm"},
    "tracker": {"kind": "mlflow", "uri": "sqlite:///mlruns/mlflow.db", "experiment": "numerai"},
    "tune": {
        "study": "lgbm-medium",
        "storage": "sqlite:///mlruns/optuna.db",
        "metric": "mean_corr",
        "direction": "maximize",
        "n_trials": 3,
        "space": {
            "train.model.params.num_leaves": {"type": "int", "low": 16, "high": 64},
            "train.model.params.learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
            "train.model.params.colsample_bytree": {"type": "float", "low": 0.05, "high": 0.3},
        },
    },
}


def build_steps() -> list[Step]:
    return [
        IngestStep(),
        ValidateStep(),
        CleanStep(),
        FeatureStep(),
        CVPlanStep(),
        TrainStep(),
        EvaluateStep(),
        RegisterStep(recipe="serve_recipe"),
    ]


# --------------------------------------------------------------- serving hooks


def serve_input(log: Any, store: Any) -> tuple[pl.DataFrame, str]:
    """Default input when `mlpipe predict` gets no --input: the latest ingested
    live snapshot."""
    live_hash = log.latest_outputs()["raw_live"]
    return store.load(live_hash), f"live {live_hash[:8]}"


def prepare(frame: pl.DataFrame, bundle: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(features in bundle order, id columns for the output). Replays the
    cleaner's stateless fill, and keys rows by `numerai_ticker` (what a v6.0
    submission is keyed by) plus the era under its canonical name."""
    recipe = bundle["recipe"]  # clean's serve_recipe
    feats = frame.select(bundle["features"])
    if recipe["cleaner"]["kind"] == "baseline":
        feats = feats.with_columns(pl.all().fill_null(recipe["cleaner"]["fill_value"]))
    ids = [c for c in ("numerai_ticker",) if c in frame.columns]
    return feats, frame.select(*ids, pl.col(recipe["era_column"]).alias("era"))
