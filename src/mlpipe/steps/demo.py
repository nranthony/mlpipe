"""Toy two-step pipeline proving the cycle-0 spine end to end via the CLI.
Replaced by real blocks from cycle 1 on.
"""

from __future__ import annotations

import polars as pl

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult

DEFAULT_CONFIG = {
    "make_data": {"rows": 100, "seed": 1},
    "summarize": {"factor": 2},
}


class MakeDataConfig(StrictConfig):
    rows: int = 100
    seed: int = 1


class SummarizeConfig(StrictConfig):
    factor: int = 1


class MakeData(Step):
    name = "make_data"
    inputs: list[str] = []
    outputs = ["raw_table"]
    config_model = MakeDataConfig

    def run(self, ctx):
        cfg = MakeDataConfig.model_validate(ctx.config.get(self.name, {}))
        n, seed = cfg.rows, cfg.seed
        df = pl.DataFrame(
            {
                "era": pl.Series([f"e{i % 10}" for i in range(n)]),
                "x": pl.Series([(i * seed) % 127 for i in range(n)], dtype=pl.Int8),
            }
        )
        art = ctx.put(df, "raw_table")
        return StepResult(outputs={"raw_table": art})


class Summarize(Step):
    name = "summarize"
    inputs = ["raw_table"]
    outputs = ["summary"]
    config_model = SummarizeConfig

    def run(self, ctx):
        df = ctx.get("raw_table")
        factor = SummarizeConfig.model_validate(ctx.config.get(self.name, {})).factor
        total = int(df["x"].sum()) * factor
        art = ctx.put({"total": total, "rows": df.height}, "summary", ext="json")
        ctx.log_metric("total", float(total))
        return StepResult(outputs={"summary": art})


def build_steps() -> list[Step]:
    return [MakeData(), Summarize()]
