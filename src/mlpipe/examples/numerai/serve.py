"""Serving steps for `mlpipe predict` (ADR-0007): ingest_live -> validate_live ->
predict (see pipeline.build_serve_steps).

The round is named in config (`--set ingest_live.round=N`), so the live file is
the immutable `<version>/live/<round>/live.parquet`. This module is imported by
`run`/`tune` just for step names: numerapi is imported lazily inside functions.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import Field

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult
from mlpipe.examples.numerai.validate import ValidateConfig, check_table


def round_open_time(round_num: int) -> datetime:
    """When tournament round `round_num` opened (numerapi `get_competitions`)."""
    from numerapi import NumerAPI

    for rnd in NumerAPI().get_competitions():
        if rnd["number"] == round_num:
            return rnd["openTime"]
    raise ValueError(f"round {round_num} is not in numerapi get_competitions()")


class IngestLiveConfig(StrictConfig):
    version: str = "v6.0"
    round: int = Field(gt=0)
    era_column: str = "date"
    max_age_days: int = Field(default=7, gt=0)


def check_live_date(path: Path, cfg: IngestLiveConfig, open_time: datetime) -> str:
    """ADR-0007 decision 6, two-sided: the file must hold exactly one ISO date, no
    more than `max_age_days` before the round's open date and not after it (the
    likeliest wrong file in a backfill is the current round's). Returns the date."""
    era = cfg.era_column
    where = f"{path} (round {cfg.round})"
    schema = pl.scan_parquet(path).collect_schema()
    if era not in schema.names():
        raise ValueError(f"live file {where} has no era column {era!r}")
    values = pl.scan_parquet(path).select(pl.col(era).unique()).collect()[era].to_list()
    if len(values) != 1:
        raise ValueError(f"live file {where}: expected one {era!r} value, got {len(values)}: {values[:5]}")
    try:
        live_date = date.fromisoformat(values[0])
    except (TypeError, ValueError):
        raise ValueError(f"live file {where}: {era!r} value {values[0]!r} is not an ISO date") from None
    opened = open_time.date()
    if live_date > opened:
        raise ValueError(
            f"live file {where}: data date {live_date} is after round open date {opened} "
            "(a later round's file?)"
        )
    if (opened - live_date).days > cfg.max_age_days:
        raise ValueError(
            f"live file {where}: data date {live_date} is {(opened - live_date).days} days before "
            f"round open date {opened} (max_age_days={cfg.max_age_days})"
        )
    return live_date.isoformat()


class IngestLiveStep(Step):
    name = "ingest_live"
    inputs: list[str] = []
    outputs = ["raw_live"]
    config_model = IngestLiveConfig

    def run(self, ctx):
        cfg = IngestLiveConfig.model_validate(ctx.config.get(self.name, {}))
        download_dir = Path(ctx.config.get("download_dir", "../raw_data/numerai"))
        final = download_dir / cfg.version / "live" / str(cfg.round) / "live.parquet"
        open_time = round_open_time(cfg.round)  # every run, never on a cache hit

        if final.exists():  # another tool's download is checked the same way; never deleted here
            live_date = check_live_date(final, cfg, open_time)
        else:
            final.parent.mkdir(parents=True, exist_ok=True)
            part = final.with_name(final.name + ".part")
            try:
                from numerapi import NumerAPI

                NumerAPI().download_dataset(f"{cfg.version}/live.parquet", str(part), round_num=cfg.round)
                live_date = check_live_date(part, cfg, open_time)
                os.replace(part, final)
            except BaseException:
                for leftover in (part, part.with_name(part.name + ".temp")):
                    leftover.unlink(missing_ok=True)  # files this step created
                raise

        art = ctx.put(final, "raw_live", ext="parquet")
        rows = pl.scan_parquet(final).select(pl.len()).collect().item()
        ctx.log_meta("raw_live", {
            "file": f"{cfg.version}/live/{cfg.round}/live.parquet", "bytes": final.stat().st_size,
            "rows": rows, "round": cfg.round, "open_time": open_time.isoformat(),
            "live_date": live_date,
        })
        return StepResult(outputs={"raw_live": art})


class ValidateLiveConfig(StrictConfig):
    # as ValidateConfig, minus row_tolerance: one live file has no previous snapshot
    max_feature_null_frac: float = 0.0
    max_feature_null_frac_by_dtype: dict[str, float] = {}
    era_column: str = "era"
    feature_dtypes: list[str] = ["Int8"]


class ValidateLiveStep(Step):
    name = "validate_live"
    inputs = ["raw_live", "model_bundle"]
    outputs = ["validated_live", "live_validation_report"]
    config_model = ValidateLiveConfig

    def run(self, ctx):
        live_cfg = ValidateLiveConfig.model_validate(ctx.config.get(self.name, {}))
        bundle = ctx.get("model_bundle")
        # check_table reads row_tolerance unconditionally: give it the default
        report, failures = check_table(
            "raw_live", ctx.get_lazy("raw_live"), list(bundle["features"]),
            ValidateConfig(**live_cfg.model_dump()), prev_rows=None,
        )
        if failures:
            raise ValueError("live validation failed:\n- " + "\n- ".join(failures))
        outs = {
            "validated_live": ctx.put_alias("validated_live", "raw_live"),
            "live_validation_report": ctx.put({"raw_live": report}, "live_validation_report", ext="json"),
        }
        ctx.log_meta("rows", report["rows"])
        return StepResult(outputs=outs)


def prepare(frame: pl.DataFrame, bundle: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(features in bundle order, id columns for the output). Replays the
    cleaner's stateless fill, and keys rows by `numerai_ticker` (what a v6.0
    submission is keyed by) plus the era under its canonical name."""
    recipe = bundle["recipe"]  # clean's serve_recipe
    feats = frame.select(bundle["features"])
    if recipe["cleaner"]["kind"] == "baseline":
        feats = feats.with_columns(pl.all().fill_null(recipe["cleaner"]["fill_value"]))
    ids = [c for c in ("numerai_ticker",) if c in frame.columns]
    return feats, frame.select(*ids, pl.col(recipe["era_column"]).alias("era"))
