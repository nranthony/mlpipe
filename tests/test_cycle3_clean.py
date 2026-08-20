"""Cycle 3 acceptance tests — clean & transform (goals/cycle3_clean.md)."""

from __future__ import annotations

import io

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.clean import CLEANERS, BaselineCleanConfig, CleanStep

FEATURES_META = {
    "feature_sets": {"all": ["feature_alpha", "feature_beta"], "small": ["feature_alpha"]},
    "targets": ["target"],
}


def input_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "era": ["e0", "e0", "e1", "e1"],
            "data_type": ["train"] * 4,
            "feature_alpha": pl.Series([0, None, 2, 3], dtype=pl.Int8),
            "feature_beta": pl.Series([4, 3, 2, 1], dtype=pl.Int8),
            "target": pl.Series([0.25, 0.5, None, 1.0], dtype=pl.Float32),
        }
    )


@pytest.fixture()
def env(tmp_path):
    return LocalCasStore(tmp_path / "store"), tmp_path


def base_config(cleaner=None):
    return {"clean": {"cleaner": cleaner or {"kind": "baseline", "feature_set": "all"}}}


def run_clean(store, manifests, config):
    seeds = [
        store.save(input_frame(), "validated_train", ext="parquet"),
        store.save(FEATURES_META, "features_meta", ext="json"),
    ]
    return Pipeline([CleanStep()], store, manifests, config).run(seeds=seeds)


def test_cycle3_two_fresh_runs_byte_identical(env):
    store, tmp = env
    r1 = run_clean(store, tmp / "m1", base_config())
    r2 = run_clean(store, tmp / "m2", base_config())  # separate manifests: no cache
    assert r1["manifests"][0]["status"] == r2["manifests"][0]["status"] == "success"
    assert (
        r1["artifacts"]["clean_table"].content_hash
        == r2["artifacts"]["clean_table"].content_hash
    )  # same sha256 == byte-identical parquet


@settings(max_examples=25, deadline=None)
@given(
    values=st.lists(
        st.one_of(st.none(), st.integers(min_value=-5, max_value=4)), min_size=1, max_size=8
    ),
    fill=st.integers(min_value=0, max_value=4),
)
def test_cycle3_property_cleaners_are_pure(values, fill):
    df = pl.DataFrame(
        {
            "era": [f"e{i // 2}" for i in range(len(values))],
            "feature_alpha": pl.Series(values, dtype=pl.Int8),
            "target": pl.Series([0.5] * len(values), dtype=pl.Float32),
        }
    )
    cfg = BaselineCleanConfig(feature_set="all", fill_value=fill)
    outs = []
    for _ in range(2):
        out = CLEANERS["baseline"](df.lazy(), cfg, ["feature_alpha"], ["target"]).collect()
        buf = io.BytesIO()
        out.write_parquet(buf)
        outs.append(buf.getvalue())
    assert outs[0] == outs[1]
    assert out["feature_alpha"].null_count() == 0


def test_cycle3_cleaner_swap_changes_hash(env):
    store, tmp = env
    r1 = run_clean(store, tmp / "m", base_config())
    r2 = run_clean(
        store, tmp / "m", base_config(cleaner={"kind": "minimal", "feature_set": "all"})
    )
    assert r2["manifests"][0]["status"] == "success"  # config change: no cache hit
    assert (
        r1["artifacts"]["clean_table"].content_hash
        != r2["artifacts"]["clean_table"].content_hash
    )
    # baseline drops the null-target row and fills feature nulls; minimal keeps both
    minimal = store.load(r2["artifacts"]["clean_table"].content_hash)
    baseline = store.load(r1["artifacts"]["clean_table"].content_hash)
    assert minimal.height == 4 and baseline.height == 3
    assert baseline["feature_alpha"].null_count() == 0
    assert minimal["feature_alpha"].null_count() == 1


def test_cycle3_feature_set_selects_columns(env):
    store, tmp = env
    r = run_clean(
        store, tmp / "m", base_config(cleaner={"kind": "baseline", "feature_set": "small"})
    )
    df = store.load(r["artifacts"]["clean_table"].content_hash)
    assert "feature_alpha" in df.columns and "feature_beta" not in df.columns


def test_cycle3_bad_config_dies_before_any_step(env):
    store, tmp = env
    with pytest.raises(ValidationError):
        run_clean(store, tmp / "m", base_config(cleaner={"kind": "no_such_cleaner"}))
    assert not (tmp / "m" / "index.jsonl").exists()  # nothing ran, nothing logged


def test_cycle3_rerun_is_cache_hit(env):
    store, tmp = env
    run_clean(store, tmp / "m", base_config())
    r = run_clean(store, tmp / "m", base_config())
    assert r["manifests"][0]["status"] == "cached"
