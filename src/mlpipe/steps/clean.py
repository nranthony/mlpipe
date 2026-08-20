"""Cycle 3 — clean & transform (DESIGN.md block 3). Stateless only.

Cleaners are pure polars expressions selected by a pydantic discriminated
union: no fitted state, no data-derived constants — same input bytes must
always produce the same output bytes (what makes content caching valid).
"""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from mlpipe.core.interfaces import Step, StepResult


class BaselineCleanConfig(BaseModel):
    kind: Literal["baseline"] = "baseline"
    feature_set: str = "medium"
    fill_value: int = 2          # constant from config, never computed from data
    targets: list[str] = ["target"]


class MinimalCleanConfig(BaseModel):
    kind: Literal["minimal"] = "minimal"
    feature_set: str = "medium"
    targets: list[str] = ["target"]


class CleanConfig(BaseModel):
    cleaner: BaselineCleanConfig | MinimalCleanConfig = Field(
        default_factory=BaselineCleanConfig, discriminator="kind"
    )


def _baseline(lf: pl.LazyFrame, cfg, features: list[str], targets: list[str]) -> pl.LazyFrame:
    return (
        lf.select("era", *features, *targets)
        .with_columns(pl.col(features).fill_null(cfg.fill_value))
        .drop_nulls(subset=targets)
    )


def _minimal(lf: pl.LazyFrame, cfg, features: list[str], targets: list[str]) -> pl.LazyFrame:
    return lf.select("era", *features, *targets)


CLEANERS = {"baseline": _baseline, "minimal": _minimal}  # kind -> implementation


class CleanStep(Step):
    name = "clean"
    inputs = ["validated_train", "features_meta"]
    outputs = ["clean_table"]
    config_model = CleanConfig

    def run(self, ctx):
        cfg = CleanConfig.model_validate(ctx.config.get(self.name, {})).cleaner
        features = sorted(ctx.get("features_meta")["feature_sets"][cfg.feature_set])
        lf = CLEANERS[cfg.kind](ctx.get_lazy("validated_train"), cfg, features, cfg.targets)
        df = lf.collect(engine="streaming")
        art = ctx.put(df, "clean_table")
        ctx.log_meta("clean", {"rows": df.height, "features": len(features), "cleaner": cfg.kind})
        return StepResult(outputs={"clean_table": art})
