"""The Titanic example pipeline (work/0013): generic steps around one cleaning step.

    mlpipe run --pipeline titanic-example
    mlpipe predict --pipeline titanic-example --bundle <hash> --out submission.parquet

Trains on `<download_dir>/titanic/train.csv` (Kaggle, 891 rows) and serves
`<download_dir>/titanic/test.csv` (418 rows, no labels), each pinned by sha256.
"""

from __future__ import annotations

from mlpipe.core.interfaces import Step
from mlpipe.examples.titanic import serve
from mlpipe.examples.titanic.clean import TitanicCleanStep
from mlpipe.steps.features import FeatureStep
from mlpipe.steps.groupcv import GroupKFoldStep
from mlpipe.steps.ingest_file import IngestFileStep
from mlpipe.steps.predict import PredictStep
from mlpipe.steps.profile import ProfileStep
from mlpipe.steps.register import RegisterStep
from mlpipe.steps.tabular_eval import TabularEvalStep
from mlpipe.steps.train import TrainStep

DEFAULT_CONFIG = {
    "download_dir": "../raw_data/kaggle",
    "tracker": {"kind": "mlflow", "experiment": "titanic-example"},
    "ingest": {
        "files": {
            "raw_train": {
                "path": "titanic/train.csv",
                "sha256": "7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f",
            },
        },
    },
    # the serve source (`mlpipe predict`): Kaggle's unlabeled test set
    "ingest_test": {
        "files": {
            "raw_test": {
                "path": "titanic/test.csv",
                "sha256": "56023b9948236f3c7a1c9448fcf418b283e109ef177fa8c7e069158dd7dd52b2",
            },
        },
    },
    "clean": {"holdout_fraction": 0.2, "seed": 0},
    "profile": {"target": "Survived"},
    "features": {"transformer": {"kind": "passthrough"}},
    "cvplan": {"n_folds": 5, "seed": 0},
    "train": {
        "model": {
            "kind": "lightgbm_binary",
            "params": {
                "n_estimators": 400,
                "learning_rate": 0.03,
                "num_leaves": 15,
                "min_child_samples": 10,
                "metric": ["binary_logloss", "auc"],
                "n_jobs": 1,
            },
        },
        "target": "Survived",
    },
    "evaluate": {"task": "binary", "target": "Survived"},
    "register": {"model_name": "titanic-example", "id_columns": ["PassengerId"]},
}


def build_steps() -> list[Step]:
    return [IngestFileStep(outputs=["raw_train"]), TitanicCleanStep(), ProfileStep(),
            FeatureStep(), GroupKFoldStep(), TrainStep(), TabularEvalStep(), RegisterStep()]


def build_serve_steps() -> list[Step]:
    """`mlpipe predict`: test.csv through clean's features, out as PassengerId, Survived."""
    return [IngestFileStep(outputs=["raw_test"], name="ingest_test"),
            PredictStep(frame_key="raw_test", prepare=serve.prepare, finish=serve.finish)]
