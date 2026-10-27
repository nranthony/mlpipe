"""Cycle 3 — clean & transform (DESIGN.md block 3). Stateless only.

Cleaners are pure polars expressions selected by a pydantic discriminated
union: no fitted state, no data-derived constants — same input bytes must
always produce the same output bytes (what makes content caching valid).
"""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import Field

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult
from mlpipe.steps.cvplan import era_order


class BaselineCleanConfig(StrictConfig):
    kind: Literal["baseline"] = "baseline"
    feature_set: str = "medium"
    fill_value: int = 2          # constant from config, never computed from data
    targets: list[str] = ["target"]


class MinimalCleanConfig(StrictConfig):
    kind: Literal["minimal"] = "minimal"
    feature_set: str = "medium"
    targets: list[str] = ["target"]


class EraSlice(StrictConfig):
    """Smoke-run row cut (work/0003). Whole eras only: row sampling would keep the
    plan hash and look comparable to a full run. `tail`/`head` keep n contiguous
    eras (purge/embargo stay meaningful); `stride` keeps every n-th era, which
    breaks contiguity, so purge_eras then spans n× more real eras. The slice
    applies to validation too: its newest eras have no target yet and the cleaner
    drops them (tail 100 on v6.0 leaves 75 validation eras)."""

    mode: Literal["tail", "head", "stride"] = "tail"
    n: int = Field(gt=0)


class CleanConfig(StrictConfig):
    cleaner: BaselineCleanConfig | MinimalCleanConfig = Field(
        default_factory=BaselineCleanConfig, discriminator="kind"
    )
    era_slice: EraSlice | None = None
    # The source's era column (v6.0: `date`); renamed to the canonical `era`
    # here, so nothing downstream knows the source's name.
    era_column: str = "era"
    # Features dropped from the chosen set, explicitly (e.g. v6.0's String
    # `feature_country`, which no numeric backend can take as is).
    exclude_features: list[str] = []


def slice_eras(lf: pl.LazyFrame, cut: EraSlice | None) -> pl.LazyFrame:
    if cut is None:
        return lf
    eras = era_order(lf.select(pl.col("era").unique()).collect()["era"].to_list())
    keep = {"tail": eras[-cut.n:], "head": eras[: cut.n], "stride": eras[:: cut.n]}[cut.mode]
    return lf.filter(pl.col("era").is_in(keep))


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
    inputs = ["validated_train", "validated_validation", "features_meta"]
    # serve_recipe: what serving must replay on live rows (register bundles it)
    outputs = ["clean_table", "clean_validation", "serve_recipe"]
    config_model = CleanConfig

    def run(self, ctx):
        step_cfg = CleanConfig.model_validate(ctx.config.get(self.name, {}))
        cfg = step_cfg.cleaner
        chosen = ctx.get("features_meta")["feature_sets"][cfg.feature_set]
        unknown = set(step_cfg.exclude_features) - set(chosen)
        if unknown:
            raise ValueError(f"exclude_features not in feature set {cfg.feature_set!r}: {sorted(unknown)}")
        features = sorted(set(chosen) - set(step_cfg.exclude_features))
        outs = {}
        for in_key, out_key in [
            ("validated_train", "clean_table"),
            ("validated_validation", "clean_validation"),
        ]:
            lf = ctx.get_lazy(in_key)
            if step_cfg.era_column != "era":
                lf = lf.rename({step_cfg.era_column: "era"})
            lf = slice_eras(lf, step_cfg.era_slice)
            df = CLEANERS[cfg.kind](lf, cfg, features, cfg.targets).collect(engine="streaming")
            outs[out_key] = ctx.put(df, out_key)
            ctx.log_meta(out_key, {
                "rows": df.height, "eras": df["era"].n_unique(), "features": len(features),
                "excluded": step_cfg.exclude_features,
                "cleaner": cfg.kind, "era_slice": step_cfg.era_slice and step_cfg.era_slice.model_dump(),
            })
        outs["serve_recipe"] = ctx.put(
            {"cleaner": cfg.model_dump(), "era_column": step_cfg.era_column},
            "serve_recipe", ext="json",
        )
        return StepResult(outputs=outs)
