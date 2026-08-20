"""Cycle 4 — feature engineering (DESIGN.md block 4).

Fitted transforms use the borrowed sklearn fit/transform protocol and learn
from train ONLY. The fitted transformer is a versioned joblib artifact that
travels with the model — the train/serve-skew guard.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from mlpipe.core.interfaces import Step, StepResult


class PassthroughConfig(BaseModel):
    kind: Literal["passthrough"] = "passthrough"  # int8 features unchanged (GBM path)


class StandardScalerConfig(BaseModel):
    kind: Literal["standard_scaler"] = "standard_scaler"  # float32 z-scores (torch path)


class FeatureConfig(BaseModel):
    transformer: PassthroughConfig | StandardScalerConfig = Field(
        default_factory=PassthroughConfig, discriminator="kind"
    )


def make_transformer(kind: str):
    return {"passthrough": FunctionTransformer, "standard_scaler": StandardScaler}[kind]()


class FeatureStep(Step):
    name = "features"
    inputs = ["clean_table", "clean_validation"]
    outputs = ["feature_table", "feature_validation", "fitted_transformer"]
    config_model = FeatureConfig

    def run(self, ctx):
        cfg = FeatureConfig.model_validate(ctx.config.get(self.name, {})).transformer
        train = ctx.get("clean_table")
        validation = ctx.get("clean_validation")
        feats = [c for c in train.columns if c.startswith("feature_")]

        def matrix(df: pl.DataFrame) -> np.ndarray:
            arr = df.select(feats).to_numpy()
            return arr.astype(np.float32) if cfg.kind == "standard_scaler" else arr

        transformer = make_transformer(cfg.kind)
        transformer.fit(matrix(train))  # statistics from the train partition ONLY

        outs = {}
        for out_key, df in [("feature_table", train), ("feature_validation", validation)]:
            transformed = pl.from_numpy(transformer.transform(matrix(df)), schema=feats)
            outs[out_key] = ctx.put(df.drop(feats).hstack(transformed), out_key)
        outs["fitted_transformer"] = ctx.put(transformer, "fitted_transformer", ext="joblib")
        ctx.log_meta("features", {"transformer": cfg.kind, "n_features": len(feats)})
        return StepResult(outputs=outs)
