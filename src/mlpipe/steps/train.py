"""Cycle 6 — train (DESIGN.md block 6). Written ONCE against the ModelBackend
port: this module must never import a model library. Swapping model families
is a config change (train.model.kind) resolved through the backend registry.

The model artifact bundles the full-data backend AND the per-fold backends,
so stored state alone reproduces the OOF predictions.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from mlpipe.backends.registry import module_for, resolve
from mlpipe.core.interfaces import Step, StepResult


class LightGBMConfig(BaseModel):
    kind: Literal["lightgbm"] = "lightgbm"
    params: dict[str, Any] = {}


class TorchMLPConfig(BaseModel):
    kind: Literal["torch_mlp"] = "torch_mlp"  # backend arrives in cycle 8
    params: dict[str, Any] = {}


class TrainConfig(BaseModel):
    model: LightGBMConfig | TorchMLPConfig = Field(
        default_factory=LightGBMConfig, discriminator="kind"
    )
    target: str = "target"


def era_corr(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(pred, y)[0, 1])


class TrainStep(Step):
    name = "train"
    inputs = ["feature_table", "fold_plan"]
    outputs = ["model", "oof_preds"]
    config_model = TrainConfig

    def code_deps(self, cfg):
        """The backend is resolved lazily through the registry, so the static
        parser cannot see it — declare the configured kind's module explicitly."""
        return [module_for(TrainConfig.model_validate(cfg or {}).model.kind)]

    def run(self, ctx):
        cfg = TrainConfig.model_validate(ctx.config.get(self.name, {}))
        df = ctx.get("feature_table")
        plan = ctx.get("fold_plan")
        feats = [c for c in df.columns if c.startswith("feature_")]
        X = df.select(feats)
        y = df[cfg.target].to_numpy()

        backend_cls = resolve(cfg.model.kind)
        oof = np.full(df.height, np.nan, dtype=np.float32)
        fold_backends = []
        for i, (tr, va) in enumerate(plan.folds):
            backend = backend_cls()
            backend.fit(X[tr], y[tr], cfg.model.params)
            oof[va] = backend.predict(X[va])
            fold_backends.append(backend)
            ctx.log_metric(f"fold{i}_corr", era_corr(oof[va], y[va]))

        full = backend_cls()
        full.fit(X, y, cfg.model.params, plan)

        outs = {
            "model": ctx.put({"full": full, "fold_models": fold_backends}, "model", ext="joblib"),
            "oof_preds": ctx.put(
                pl.DataFrame({"era": df["era"], "prediction": oof, cfg.target: y}), "oof_preds"
            ),
        }
        ctx.log_meta("plan_hash", plan.plan_hash)
        ctx.log_meta("model", cfg.model.model_dump())
        return StepResult(outputs=outs)
