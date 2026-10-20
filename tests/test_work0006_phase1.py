"""work/0006 phase 1: era_slice smoke runs (step 2) and --from resolving
upstream by current-config signature (step 3)."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError
from test_cycle0_spine import MakeData, Summarize, seed_artifact
from test_cycle3_clean import run_clean

from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.clean import CleanConfig, CleanStep, EraSlice, slice_eras

ERAS = [f"e{i:02d}" for i in range(10)]


def era_frame() -> pl.LazyFrame:
    return pl.LazyFrame({"era": [e for e in ERAS for _ in range(3)], "x": list(range(30))})


@pytest.fixture()
def env(tmp_path):
    return LocalCasStore(tmp_path / "store"), tmp_path


# ---------------------------------------------------------------- era_slice


@pytest.mark.parametrize(
    ("mode", "n", "expected"),
    [("tail", 3, ERAS[-3:]), ("head", 2, ERAS[:2]), ("stride", 4, ERAS[::4])],
)
def test_era_slice_keeps_whole_eras(mode, n, expected):
    out = slice_eras(era_frame(), EraSlice(mode=mode, n=n)).collect()
    assert out["era"].unique(maintain_order=True).to_list() == expected
    assert out.group_by("era").len()["len"].to_list() == [3] * len(expected)


def test_era_slice_none_is_identity():
    assert slice_eras(era_frame(), None).collect().equals(era_frame().collect())


def test_era_slice_rejects_bad_config_before_running():
    with pytest.raises(ValidationError):
        CleanConfig.model_validate({"era_slice": {"mode": "tail", "n": 0}})
    with pytest.raises(ValidationError):
        CleanConfig.model_validate({"era_slice": {"mode": "random", "n": 3}})


def test_era_slice_changes_output_hash_and_records_meta(env):
    store, tmp = env
    cleaner = {"kind": "baseline", "feature_set": "all"}
    full = run_clean(store, tmp / "m1", {"clean": {"cleaner": cleaner}})
    cut = run_clean(
        store, tmp / "m2", {"clean": {"cleaner": cleaner, "era_slice": {"mode": "tail", "n": 1}}}
    )
    assert (
        full["artifacts"]["clean_table"].content_hash
        != cut["artifacts"]["clean_table"].content_hash
    )
    (m,) = cut["manifests"]
    assert m["meta"]["clean_table"]["eras"] == 1
    assert m["meta"]["clean_table"]["era_slice"] == {"mode": "tail", "n": 1}
    assert CleanStep.name == m["step"]


# ---------------------------------------------------------------- --from


def two_step(store, manifests, factor=1, rows=2):
    cfg = {"make_data": {"rows": rows}, "summarize": {"factor": factor}}
    return Pipeline([MakeData(), Summarize()], store, manifests, cfg)


def test_from_serves_upstream_cached_under_current_config(env):
    store, tmp = env
    seeds = [seed_artifact(store)]
    two_step(store, tmp / "m").run(seeds=seeds)
    result = two_step(store, tmp / "m", factor=3).run(from_step="summarize", seeds=seeds)
    statuses = {m["step"]: m["status"] for m in result["manifests"]}
    assert statuses == {"make_data": "cached", "summarize": "success"}


def test_from_refuses_upstream_produced_by_another_config(env):
    """The old trapdoor: --from served the latest raw_table whatever config made it."""
    store, tmp = env
    seeds = [seed_artifact(store)]
    two_step(store, tmp / "m", rows=2).run(seeds=seeds)
    with pytest.raises(RuntimeError, match="upstream step 'make_data' is not cached"):
        two_step(store, tmp / "m", rows=1).run(from_step="summarize", seeds=seeds)


def test_full_run_after_upstream_change_recomputes_only_what_changed(env):
    store, tmp = env
    seeds = [seed_artifact(store)]
    two_step(store, tmp / "m").run(seeds=seeds)
    result = two_step(store, tmp / "m", factor=5).run(seeds=seeds)
    assert [m["status"] for m in result["manifests"]] == ["cached", "success"]
