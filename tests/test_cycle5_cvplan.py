"""Cycle 5 acceptance tests — split / CV plan (goals/cycle5_cvplan.md)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.cvplan import CVConfig, CVPlanStep, assert_same_plan, build_plan


def era_series(n_eras: int = 20, rows_per_era: int = 5) -> pl.Series:
    return pl.Series("era", [f"e{i:03d}" for i in range(n_eras) for _ in range(rows_per_era)])


def frame(n_eras: int = 20, rows_per_era: int = 5) -> pl.DataFrame:
    eras = era_series(n_eras, rows_per_era)
    return pl.DataFrame({"era": eras, "feature_x": pl.Series(range(len(eras)), dtype=pl.Int8)})


CFG = CVConfig(n_folds=4, purge_eras=2, embargo_eras=1)


def era_of(eras: pl.Series, idx: np.ndarray) -> set[str]:
    return set(eras.gather(idx.tolist()).to_list())


def test_cycle5_no_era_in_both_train_and_valid():
    eras = era_series()
    plan = build_plan(eras, CFG)
    assert len(plan.folds) == 4
    for train_idx, valid_idx in plan.folds:
        assert not era_of(eras, train_idx) & era_of(eras, valid_idx)


def test_cycle5_purge_and_embargo_honored():
    eras = era_series()
    order = {e: i for i, e in enumerate(eras.unique(maintain_order=True).to_list())}
    plan = build_plan(eras, CFG)
    for train_idx, valid_idx in plan.folds:
        v = [order[e] for e in era_of(eras, valid_idx)]
        lo, hi = min(v), max(v)
        for e in era_of(eras, train_idx):
            i = order[e]
            assert i < lo - CFG.purge_eras or i > hi + CFG.purge_eras + CFG.embargo_eras


def test_cycle5_plan_hash_changes_iff_config_or_eras_change():
    base = build_plan(era_series(), CFG).plan_hash
    assert build_plan(era_series(rows_per_era=9), CFG).plan_hash == base  # rows: no effect
    assert build_plan(era_series(n_eras=21), CFG).plan_hash != base      # eras: effect
    assert build_plan(era_series(), CVConfig(n_folds=4, purge_eras=3, embargo_eras=1)).plan_hash != base


def test_cycle5_assert_same_plan_guard():
    a = {"meta": {"plan_hash": "abc"}}
    assert assert_same_plan(a, {"meta": {"plan_hash": "abc"}}) == "abc"
    with pytest.raises(ValueError, match="comparison invalid"):
        assert_same_plan(a, {"meta": {"plan_hash": "def"}})
    with pytest.raises(ValueError, match="comparison invalid"):
        assert_same_plan(a, {"meta": {}})


def test_cycle5_step_roundtrip_and_cache(tmp_path):
    store = LocalCasStore(tmp_path / "store")
    seeds = [store.save(frame(), "feature_table", ext="parquet")]
    config = {"cvplan": CFG.model_dump()}
    pipe = Pipeline([CVPlanStep()], store, tmp_path / "m", config)
    r1 = pipe.run(seeds=seeds)
    plan = store.load(r1["artifacts"]["fold_plan"].content_hash)
    assert plan.plan_hash == r1["manifests"][0]["meta"]["plan_hash"]
    assert all(v.dtype == np.int32 for _, v in plan.folds)
    r2 = Pipeline([CVPlanStep()], store, tmp_path / "m", config).run(seeds=seeds)
    assert r2["manifests"][0]["status"] == "cached"
