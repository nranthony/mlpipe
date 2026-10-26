"""Cycle 6 — train (DESIGN.md block 6). Written ONCE against the ModelBackend
port: this module must never import a model library. Swapping model families
is a config change (train.model.kind) resolved through the backend registry.

The model artifact bundles the full-data backend AND the per-fold backends,
so stored state alone reproduces the OOF predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from pydantic import Field, field_validator

from mlpipe.backends.registry import kinds, module_for, resolve
from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult
from mlpipe.core.signature import module_code_hash


class ModelConfig(StrictConfig):
    kind: str = "lightgbm"  # a registered backend: builtin or `mlpipe.backends` entry point
    params: dict[str, Any] = {}

    @field_validator("kind")
    @classmethod
    def _registered(cls, kind: str) -> str:
        if kind not in kinds():
            raise ValueError(f"unknown model kind {kind!r}; registered: {sorted(kinds())}")
        return kind


class TrainConfig(StrictConfig):
    model: ModelConfig = Field(default_factory=ModelConfig)
    target: str = "target"


def pearson(pred: np.ndarray, y: np.ndarray) -> float:
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
            ctx.log_metric(f"fold{i}_corr", pearson(oof[va], y[va]))

        full = backend_cls()
        full.fit(X, y, cfg.model.params)

        # A pickled backend names its class, not its source. Hashing the backend's
        # code into the artifact makes a predict-only edit change the model's bytes,
        # so every consumer that calls predict (evaluate, register, `mlpipe predict`)
        # re-runs instead of serving a stale cache hit.
        code = module_code_hash(module_for(cfg.model.kind))
        outs = {
            "model": ctx.put(
                {"full": full, "fold_models": fold_backends, "backend_code": code},
                "model", ext="joblib",
            ),
            # the plan's group column travels with the OOF rows, if any
            "oof_preds": ctx.put(
                pl.DataFrame([
                    *([df[plan.group_column]] if plan.group_column else []),
                    pl.Series("prediction", oof),
                    pl.Series(cfg.target, y),
                ]),
                "oof_preds",
            ),
        }
        ctx.log_meta("plan_hash", plan.plan_hash)
        ctx.log_meta("model", cfg.model.model_dump())
        return StepResult(outputs=outs)
