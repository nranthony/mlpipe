"""Cycle 6 — train (DESIGN.md block 6). Written ONCE against the ModelBackend
port: this module must never import a model library. Swapping model families
is a config change (train.model.kind) resolved through the backend registry.

The model artifact bundles the full-data backend AND the per-fold backends,
so stored state alone reproduces the OOF predictions.

A backend with the optional CurveBackend members (ADR-0010) gets each fold's
validation rows as its eval set. Its per-iteration history and importances become
artifacts (`train_history`, `feature_importance`), and the curves also go to the
tracker. A backend without them yields empty frames, so the outputs are always there.
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


HISTORY_SCHEMA = {"fold": pl.Int32, "iteration": pl.Int32, "split": pl.Utf8,
                  "metric": pl.Utf8, "value": pl.Float64}
IMPORTANCE_SCHEMA = {"model": pl.Utf8, "feature": pl.Utf8, "gain": pl.Float64}


def history_rows(fold: int, history: dict[str, list[float]]) -> list[dict]:
    rows = []
    for name, values in history.items():
        split, _, metric = name.partition("_")
        rows += [{"fold": fold, "iteration": i + 1, "split": split, "metric": metric,
                  "value": v} for i, v in enumerate(values)]
    return rows


def importance_rows(model: str, backend) -> list[dict]:
    gains = backend.feature_importance() if hasattr(backend, "feature_importance") else None
    return [{"model": model, "feature": f, "gain": g} for f, g in (gains or {}).items()]


class TrainStep(Step):
    name = "train"
    inputs = ["feature_table", "fold_plan"]
    outputs = ["model", "oof_preds", "train_history", "feature_importance"]
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
        curves = hasattr(backend_cls, "history")  # accepts an eval set (CurveBackend)
        oof = np.full(df.height, np.nan, dtype=np.float32)
        fold_backends, history, importance = [], [], []
        for i, (tr, va) in enumerate(plan.folds):
            backend = backend_cls()
            if curves:
                backend.fit(X[tr], y[tr], cfg.model.params, eval_set=(X[va], y[va]))
                for name, values in backend.history().items():
                    ctx.log_curve(f"fold{i}_{name}", values)
                history += history_rows(i, backend.history())
            else:
                backend.fit(X[tr], y[tr], cfg.model.params)
            oof[va] = backend.predict(X[va])
            fold_backends.append(backend)
            importance += importance_rows(f"fold{i}", backend)
            ctx.log_metric(f"fold{i}_corr", pearson(oof[va], y[va]))

        full = backend_cls()
        full.fit(X, y, cfg.model.params)
        importance += importance_rows("full", full)

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
            "train_history": ctx.put(pl.DataFrame(history, schema=HISTORY_SCHEMA), "train_history"),
            "feature_importance": ctx.put(
                pl.DataFrame(importance, schema=IMPORTANCE_SCHEMA), "feature_importance"
            ),
        }
        ctx.log_meta("plan_hash", plan.plan_hash)
        ctx.log_meta("model", cfg.model.model_dump())
        return StepResult(outputs=outs)
