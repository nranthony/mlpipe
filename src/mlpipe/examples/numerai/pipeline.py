"""The Numerai harness pipeline (ADR-0009), plus its serve steps for `mlpipe predict`.

    mlpipe run --pipeline numerai-example
    mlpipe predict --pipeline numerai-example --set ingest_live.round=N
"""

from __future__ import annotations

from mlpipe.core.interfaces import Step
from mlpipe.examples.numerai import serve
from mlpipe.examples.numerai.clean import CleanStep
from mlpipe.examples.numerai.evaluate import EvaluateStep
from mlpipe.examples.numerai.ingest import IngestStep
from mlpipe.examples.numerai.serve import IngestLiveStep, ValidateLiveStep
from mlpipe.examples.numerai.validate import ValidateStep
from mlpipe.steps.cvplan import CVPlanStep
from mlpipe.steps.features import FeatureStep
from mlpipe.steps.predict import PredictStep
from mlpipe.steps.register import RegisterStep
from mlpipe.steps.train import TrainStep

DEFAULT_CONFIG = {
    # shared cache beside the checkout: relative to the cwd, which is the repo root
    # (like store/ and manifests/). Not in any signature, so moving it busts nothing.
    "download_dir": "../raw_data/numerai",
    # v6.0 (2026-10): v5.x training files are no longer served. The era column is
    # `date`, and `medium` = 350 Int8 + the String `feature_country`.
    # snapshot: the folder <version>/<snapshot>/ (the files' download date); a missing
    # past-dated folder is an error, only today's is downloaded
    "ingest": {"version": "v6.0", "snapshot": "2026-10-20"},
    # no round: `mlpipe predict --set ingest_live.round=N` names it
    "ingest_live": {"version": "v6.0"},
    "validate": {
        "row_tolerance": 0.2,
        "max_feature_null_frac": 0.0,  # Int8 (and String): dense, so strict
        "max_feature_null_frac_by_dtype": {"Float32": 0.1},  # factor features: ≤7.1% seen
        "era_column": "date",
        "feature_dtypes": ["Int8", "Float32", "String"],
    },
    "validate_live": {
        "max_feature_null_frac": 0.0,
        "max_feature_null_frac_by_dtype": {"Float32": 0.1},
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
    # harness state only (ADR-0009): no tournament decision is taken on these
    "tracker": {"kind": "mlflow", "uri": "sqlite:///mlruns/mlflow.db", "experiment": "numerai-harness"},
    "tune": {
        "study": "harness-lgbm-medium",
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


def build_serve_steps() -> list[Step]:
    """`mlpipe predict` runs these with the promoted bundle seeded (ADR-0007)."""
    return [
        IngestLiveStep(),
        ValidateLiveStep(),
        PredictStep(frame_key="validated_live", prepare=serve.prepare),
    ]
