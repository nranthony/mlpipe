"""The real Numerai pipeline, assembled cycle by cycle.

    mlpipe run --pipeline numerai
"""

from __future__ import annotations

from mlpipe.core.interfaces import Step
from mlpipe.steps.clean import CleanStep
from mlpipe.steps.cvplan import CVPlanStep
from mlpipe.steps.evaluate import EvaluateStep
from mlpipe.steps.features import FeatureStep
from mlpipe.steps.ingest import IngestStep
from mlpipe.steps.register import RegisterStep
from mlpipe.steps.train import TrainStep
from mlpipe.steps.validate import ValidateStep

DEFAULT_CONFIG = {
    "download_dir": "/workspace/raw_data/numerai",  # shared profile-wide cache
    "ingest": {"version": "v5.2"},
    "validate": {"row_tolerance": 0.2, "max_feature_null_frac": 0.0},
    "clean": {"cleaner": {"kind": "baseline", "feature_set": "medium"}},
    "features": {"transformer": {"kind": "passthrough"}},
    "cvplan": {"n_folds": 4, "purge_eras": 4, "embargo_eras": 4},
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
    "evaluate": {"target": "target"},
    "register": {"model_name": "numerai-lgbm"},
    "tracker": {"kind": "mlflow", "uri": "sqlite:///mlruns/mlflow.db", "experiment": "numerai"},
    "tune": {
        "study": "lgbm-medium",
        "storage": "sqlite:///mlruns/optuna.db",
        "n_trials": 3,
        "space": {
            "train.model.params.num_leaves": {"type": "int", "low": 16, "high": 64},
            "train.model.params.learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
            "train.model.params.colsample_bytree": {"type": "float", "low": 0.05, "high": 0.3},
        },
    },
    "seed": 42,
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
        RegisterStep(),
    ]
