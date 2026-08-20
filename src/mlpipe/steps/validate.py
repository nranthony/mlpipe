"""Cycle 2 — validate (DESIGN.md block 2). Hard gate: fail loud, or pass through.

pandera schemas gate structure (columns, Int8 features, era presence) on lazy
frames; quantitative checks (null bounds, era monotonicity, row counts vs the
previous snapshot) run as streaming polars aggregations so multi-GB tables are
never materialized. Outputs alias the raw hashes — validation adds no bytes.
"""

from __future__ import annotations

from typing import Any

import pandera.polars as pa
import polars as pl

from mlpipe.core.interfaces import Step, StepResult

PASS_THROUGH = {
    "raw_train": "validated_train",
    "raw_validation": "validated_validation",
    "raw_live": "validated_live",
}


def build_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "era": pa.Column(pl.String, nullable=False),
            "^feature_.*$": pa.Column(pl.Int8, regex=True),
        },
        strict=False,
    )


def check_table(
    key: str, lf: pl.LazyFrame, expected_features: list[str], cfg: dict, prev_rows: int | None
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    schema = lf.collect_schema()

    missing = sorted(set(expected_features) - set(schema.names()))
    if missing:
        failures.append(
            f"{key}: missing {len(missing)} expected feature column(s), "
            f"e.g. {missing[:3]} — upstream schema changed or download corrupt"
        )
    try:
        build_schema().validate(lf, lazy=True)
    except pa.errors.SchemaErrors as err:
        failures.append(f"{key}: pandera schema gate failed (features must be Int8): {err}")

    present = [f for f in expected_features if f in schema.names()]
    stats = (
        lf.select(
            pl.len().alias("rows"),
            (pl.col("era") >= pl.col("era").shift(1)).fill_null(True).all().alias("era_monotonic"),
            pl.max_horizontal(pl.col(present).null_count()).alias("max_feature_nulls")
            if present
            else pl.lit(0).alias("max_feature_nulls"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )

    if not stats["era_monotonic"]:
        failures.append(f"{key}: era column is not monotonically non-decreasing")
    null_frac = stats["max_feature_nulls"] / max(stats["rows"], 1)
    if null_frac > cfg.get("max_feature_null_frac", 0.0):
        failures.append(
            f"{key}: worst feature null fraction {null_frac:.4f} exceeds "
            f"bound {cfg.get('max_feature_null_frac', 0.0)}"
        )
    tolerance = cfg.get("row_tolerance", 0.2)
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
        "missing_features": missing[:20],
        "passed": not failures,
    }
    return report, failures


class ValidateStep(Step):
    name = "validate"
    inputs = ["raw_train", "raw_validation", "raw_live", "features_meta"]
    outputs = ["validated_train", "validated_validation", "validated_live", "validation_report"]

    def run(self, ctx):
        cfg = ctx.config.get(self.name, {})
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
