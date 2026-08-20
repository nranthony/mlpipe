"""Cycle 2 acceptance tests — validate (goals/cycle2_validate.md)."""

from __future__ import annotations

import polars as pl
import pytest

from mlpipe.core.interfaces import Step, StepResult
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.validate import ValidateStep

FEATURES_META = {
    "feature_sets": {"all": ["feature_alpha", "feature_beta"], "small": ["feature_alpha"]},
    "targets": ["target"],
}


def clean_frame(rows: int = 6) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "era": [f"e{i // 2}" for i in range(rows)],  # monotonic
            "feature_alpha": pl.Series(range(rows), dtype=pl.Int8),
            "feature_beta": pl.Series([i % 5 for i in range(rows)], dtype=pl.Int8),
            "target": pl.Series([i / rows for i in range(rows)], dtype=pl.Float32),
        }
    )


@pytest.fixture()
def env(tmp_path):
    store = LocalCasStore(tmp_path / "store")
    return store, tmp_path / "manifests"


def run_validate(store, manifests, frames: dict[str, pl.DataFrame], config=None):
    seeds = [store.save(df, key, ext="parquet") for key, df in frames.items()]
    seeds.append(store.save(FEATURES_META, "features_meta", ext="json"))
    pipe = Pipeline([ValidateStep()], store, manifests, config or {})
    return pipe.run(seeds=seeds)


def all_clean():
    return {
        "raw_train": clean_frame(),
        "raw_validation": clean_frame(),
        "raw_live": clean_frame(),
    }


def test_cycle2_clean_data_passes_and_aliases(env):
    store, manifests = env
    result = run_validate(store, manifests, all_clean())
    (m,) = result["manifests"]
    assert m["status"] == "success"
    # pass-through: validated_* share the raw tables' content hashes
    assert m["outputs"]["validated_train"] == m["inputs"]["raw_train"]
    report = store.load(m["outputs"]["validation_report"])
    assert report["raw_train"]["passed"] is True
    assert report["raw_train"]["rows"] == 6
    assert report["raw_train"]["era_monotonic"] is True


def test_cycle2_clean_data_caches(env):
    store, manifests = env
    run_validate(store, manifests, all_clean())
    result = run_validate(store, manifests, all_clean())
    assert result["manifests"][0]["status"] == "cached"


def test_cycle2_missing_column_fails_loud(env):
    store, manifests = env
    frames = all_clean()
    frames["raw_train"] = frames["raw_train"].drop("feature_beta")
    with pytest.raises(ValueError, match="raw_train.*feature_beta"):
        run_validate(store, manifests, frames)


def test_cycle2_poisoned_dtype_fails_loud(env):
    store, manifests = env
    frames = all_clean()
    frames["raw_validation"] = frames["raw_validation"].with_columns(
        pl.col("feature_alpha").cast(pl.Int16)
    )
    with pytest.raises(ValueError, match="raw_validation.*[Ii]nt8"):
        run_validate(store, manifests, frames)


def test_cycle2_non_monotonic_era_fails_loud(env):
    store, manifests = env
    frames = all_clean()
    frames["raw_train"] = frames["raw_train"].reverse()
    with pytest.raises(ValueError, match="raw_train.*monotonic"):
        run_validate(store, manifests, frames)


def test_cycle2_feature_nulls_fail_loud(env):
    store, manifests = env
    frames = all_clean()
    frames["raw_live"] = frames["raw_live"].with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(None)
        .otherwise(pl.col("feature_alpha"))
        .cast(pl.Int8)
        .alias("feature_alpha")
    )
    with pytest.raises(ValueError, match="raw_live.*null"):
        run_validate(store, manifests, frames)


def test_cycle2_row_count_tolerance_vs_previous(env):
    store, manifests = env
    run_validate(store, manifests, all_clean())
    frames = all_clean()
    frames["raw_train"] = clean_frame(rows=2)  # 66% shrink vs previous snapshot
    with pytest.raises(ValueError, match="raw_train.*row"):
        run_validate(store, manifests, frames, config={"validate": {"row_tolerance": 0.2}})


class _Alias(Step):
    name = "alias"
    inputs = ["raw_table"]
    outputs = ["out_table"]

    def run(self, ctx):
        ctx.get("raw_table")
        return StepResult(outputs={"out_table": ctx.put_alias("out_table", "raw_table")})


def test_cycle2_lineage_walks_through_aliases(env):
    store, manifests = env
    from test_cycle0_spine import MakeData, make_frame

    seed = store.save(make_frame(), "seed", ext="parquet")
    pipe = Pipeline([MakeData(), _Alias()], store, manifests, {"make_data": {"rows": 2}})
    result = pipe.run(seeds=[seed])
    final = result["artifacts"]["out_table"].content_hash
    chain = pipe.lineage(final)
    assert [m["step"] for m in chain] == ["alias", "make_data"]
