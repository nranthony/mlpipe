"""Cycle 2 — validate (DESIGN.md block 2). Hard gate: fail loud, or pass through.

pandera schemas gate structure (columns, era presence; feature dtypes against
the configured allow-list) on lazy
frames; quantitative checks (null bounds, era monotonicity, row counts vs the
previous snapshot) run as streaming polars aggregations so multi-GB tables are
never materialized. Outputs alias the raw hashes — validation adds no bytes.
"""

from __future__ import annotations

from typing import Any

import pandera.polars as pa
import polars as pl

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult

PASS_THROUGH = {
    "raw_train": "validated_train",
    "raw_validation": "validated_validation",
    "raw_live": "validated_live",
}


class ValidateConfig(StrictConfig):
    row_tolerance: float = 0.2
    max_feature_null_frac: float = 0.0
    # Per-dtype overrides of the bound above. v6.0's Float32 factor features are
    # legitimately sparse (≤7.1% null in train) while every Int8 feature is dense,
    # so loosening one global bound would blind the gate to Int8 drift.
    max_feature_null_frac_by_dtype: dict[str, float] = {}
    # The source's name for the era column: v5.x `era`, v6.0 `date` (weekly ISO
    # dates, so string order is time order). Clean renames it to `era`.
    era_column: str = "era"
    # Allowed feature dtypes. v5.x was all Int8; v6.0 adds Float32 factor
    # features and a String `feature_country`. Anything else is schema drift.
    feature_dtypes: list[str] = ["Int8"]


def build_schema(era_column: str) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {era_column: pa.Column(pl.String, nullable=False)},
        strict=False,
    )


def check_table(
    key: str,
    lf: pl.LazyFrame,
    expected_features: list[str],
    cfg: ValidateConfig,
    prev_rows: int | None,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    schema = lf.collect_schema()
    era = cfg.era_column

    missing = sorted(set(expected_features) - set(schema.names()))
    if missing:
        failures.append(
            f"{key}: missing {len(missing)} expected feature column(s), "
            f"e.g. {missing[:3]} — upstream schema changed or download corrupt"
        )
    try:
        build_schema(era).validate(lf, lazy=True)
    except pa.errors.SchemaErrors as err:
        failures.append(f"{key}: pandera schema gate failed ({era!r} must be a non-null String): {err}")
        return {"passed": False}, failures  # no era column: the checks below cannot run
    bad = {
        c: str(t) for c, t in schema.items()
        if c.startswith("feature_") and str(t) not in cfg.feature_dtypes
    }
    if bad:
        failures.append(
            f"{key}: features must be one of {cfg.feature_dtypes}, got e.g. "
            f"{dict(list(bad.items())[:3])} ({len(bad)} column(s))"
        )

    by_dtype: dict[str, list[str]] = {}
    for f in expected_features:
        if f in schema.names():
            by_dtype.setdefault(str(schema[f]), []).append(f)
    stats = (
        lf.select(
            pl.len().alias("rows"),
            (pl.col(era) >= pl.col(era).shift(1)).fill_null(True).all().alias("era_monotonic"),
            *[
                pl.max_horizontal(pl.col(cols).null_count()).alias(f"nulls_{dt}")
                for dt, cols in by_dtype.items()
            ],
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )

    if not stats["era_monotonic"]:
        failures.append(f"{key}: era column is not monotonically non-decreasing")
    null_fracs = {dt: stats[f"nulls_{dt}"] / max(stats["rows"], 1) for dt in by_dtype}
    for dt, frac in null_fracs.items():
        bound = cfg.max_feature_null_frac_by_dtype.get(dt, cfg.max_feature_null_frac)
        if frac > bound:
            failures.append(
                f"{key}: worst {dt} feature null fraction {frac:.4f} exceeds bound {bound}"
            )
    null_frac = max(null_fracs.values(), default=0.0)
    tolerance = cfg.row_tolerance
    if prev_rows and abs(stats["rows"] - prev_rows) / prev_rows > tolerance:
        failures.append(
            f"{key}: row count {stats['rows']:,} deviates more than {tolerance:.0%} "
            f"from previous snapshot ({prev_rows:,})"
        )

    report = {
        "rows": stats["rows"],
        "previous_rows": prev_rows,
        "era_monotonic": bool(stats["era_monotonic"]),
        "max_feature_null_frac": null_frac,
        "max_feature_null_frac_by_dtype": null_fracs,
        "missing_features": missing[:20],
        "passed": not failures,
    }
    return report, failures


class ValidateStep(Step):
    name = "validate"
    inputs = ["raw_train", "raw_validation", "raw_live", "features_meta"]
    outputs = ["validated_train", "validated_validation", "validated_live", "validation_report"]
    config_model = ValidateConfig

    def run(self, ctx):
        cfg = ValidateConfig.model_validate(ctx.config.get(self.name, {}))
        expected = sorted(ctx.get("features_meta")["feature_sets"]["all"])
        prev = ctx.previous(self.name)
        prev_rows = (prev or {}).get("meta", {}).get("rows", {})

        report: dict[str, Any] = {}
        failures: list[str] = []
        for raw_key in PASS_THROUGH:
            table_report, table_failures = check_table(
                raw_key, ctx.get_lazy(raw_key), expected, cfg, prev_rows.get(raw_key)
            )
            report[raw_key] = table_report
            failures.extend(table_failures)

        if failures:
            raise ValueError("validation failed:\n- " + "\n- ".join(failures))

        outs = {out: ctx.put_alias(out, raw) for raw, out in PASS_THROUGH.items()}
        outs["validation_report"] = ctx.put(report, "validation_report", ext="json")
        ctx.log_meta("rows", {k: report[k]["rows"] for k in PASS_THROUGH})
        return StepResult(outputs=outs)
