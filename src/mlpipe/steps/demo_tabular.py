"""A second, non-Numerai pipeline built only from the generic steps (ADR-0005
decision 5, ADR-0009): synthetic customers with several rows each, a regression
target (`spend`) and a binary one (`churned`). No eras anywhere. It is what the
CLI tests for train, register, promote, tune, predict and lineage run against.

    mlpipe run --pipeline demo-tabular
"""

from __future__ import annotations

import numpy as np
import polars as pl

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult
from mlpipe.steps.features import FeatureStep
from mlpipe.steps.groupcv import GroupKFoldStep
from mlpipe.steps.register import RegisterStep
from mlpipe.steps.tabular_eval import TabularEvalStep
from mlpipe.steps.train import TrainStep

DEFAULT_CONFIG = {
    "make": {"seed": 0, "customers": 120, "validation_customers": 40},
    "features": {"transformer": {"kind": "passthrough"}},
    "cvplan": {"n_folds": 4, "group_column": "customer_id"},
    "train": {
        "model": {
            "kind": "lightgbm",
            "params": {"n_estimators": 60, "num_leaves": 8, "min_child_samples": 5, "n_jobs": 1},
        },
        "target": "spend",
    },
    "evaluate": {"task": "regression", "target": "spend"},
    "register": {"model_name": "demo-tabular", "id_columns": ["customer_id"]},
}


def customers(n_customers: int, offset: int, rng: np.random.Generator) -> pl.DataFrame:
    rows = 6
    ids = np.repeat(np.arange(offset, offset + n_customers), rows)
    effect = np.repeat(rng.normal(scale=0.5, size=n_customers), rows)  # shared per customer
    x1, x2, noise = (rng.normal(size=len(ids)) for _ in range(3))
    signal = 1.5 * x1 - x2 + effect
    return pl.DataFrame({
        "customer_id": [f"c{i:04d}" for i in ids],
        "feature_x1": x1,
        "feature_x2": x2,
        "feature_noise": noise,
        "spend": signal + rng.normal(scale=0.3, size=len(ids)),
        "churned": (signal + rng.logistic(scale=0.3, size=len(ids)) > 0).astype(np.float64),
    })


class MakeConfig(StrictConfig):
    seed: int = 0
    customers: int = 120
    validation_customers: int = 40


class MakeCustomers(Step):
    name = "make"
    inputs: list[str] = []
    outputs = ["clean_table", "clean_validation"]
    config_model = MakeConfig

    def run(self, ctx):
        cfg = MakeConfig.model_validate(ctx.config.get(self.name, {}))
        rng = np.random.default_rng(cfg.seed)
        return StepResult(outputs={
            "clean_table": ctx.put(customers(cfg.customers, 0, rng), "clean_table"),
            "clean_validation": ctx.put(customers(cfg.validation_customers, 1000, rng), "clean_validation"),
        })


def build_steps() -> list[Step]:
    return [MakeCustomers(), FeatureStep(), GroupKFoldStep(), TrainStep(), TabularEvalStep(),
            RegisterStep()]
