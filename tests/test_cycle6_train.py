"""Cycle 6 acceptance tests — train via the ModelBackend port (goals/cycle6_train.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mlpipe.steps.train as train_module
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.cvplan import CVConfig, build_plan
from mlpipe.steps.train import TrainStep


def training_frame(rows: int = 240) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    f1 = rng.integers(0, 5, rows).astype(np.int8)
    f2 = rng.integers(0, 5, rows).astype(np.int8)
    target = (0.2 * f1 - 0.1 * f2 + rng.normal(0, 0.1, rows)).astype(np.float32)
    return pl.DataFrame(
        {
            "era": [f"e{i // 20:03d}" for i in range(rows)],
            "feature_one": f1,
            "feature_two": f2,
            "target": target,
        }
    )


@pytest.fixture()
def env(tmp_path):
    store = LocalCasStore(tmp_path / "store")
    df = training_frame()
    plan = build_plan(df["era"], CVConfig(n_folds=3, purge_eras=1, embargo_eras=0))
    seeds = [
        store.save(df, "feature_table", ext="parquet"),
        store.save(plan, "fold_plan", ext="joblib"),
    ]
    config = {"train": {"model": {"kind": "lightgbm", "params": {"n_estimators": 20}}}}
    return store, tmp_path / "m", seeds, config, df, plan


def test_cycle6_trainstep_source_has_no_lightgbm_import():
    import re

    source = Path(train_module.__file__).read_text()
    assert not re.search(r"^\s*(import|from)\s+lightgbm", source, re.MULTILINE)
    assert "lgb." not in source  # no aliased usage either


def test_cycle6_per_fold_training_respects_plan(env):
    store, manifests, seeds, config, df, plan = env
    result = Pipeline([TrainStep()], store, manifests, config).run(seeds=seeds)
    oof = store.load(result["artifacts"]["oof_preds"].content_hash)
    preds = oof["prediction"].to_numpy()
    valid_union = np.concatenate([va for _, va in plan.folds])
    assert not np.isnan(preds[valid_union]).any()
    uncovered = np.setdiff1d(np.arange(df.height), valid_union)
    assert np.isnan(preds[uncovered]).all()  # only fold-valid rows get OOF preds
    m = result["manifests"][0]
    assert m["meta"]["plan_hash"] == plan.plan_hash
    assert {f"fold{i}_corr" for i in range(3)} <= set(m["metrics"])


def test_cycle6_model_artifact_reproduces_oof(env):
    store, manifests, seeds, config, df, plan = env
    result = Pipeline([TrainStep()], store, manifests, config).run(seeds=seeds)
    bundle = store.load(result["artifacts"]["model"].content_hash)
    oof = store.load(result["artifacts"]["oof_preds"].content_hash)["prediction"].to_numpy()
    X = df.select("feature_one", "feature_two")
    for backend, (_, va) in zip(bundle["fold_models"], plan.folds):
        assert np.allclose(backend.predict(X[va]), oof[va], atol=1e-6)
    assert bundle["full"].predict(X).shape == (df.height,)


def test_cycle6_rerun_is_cache_hit(env):
    store, manifests, seeds, config, *_ = env
    Pipeline([TrainStep()], store, manifests, config).run(seeds=seeds)
    result = Pipeline([TrainStep()], store, manifests, config).run(seeds=seeds)
    assert result["manifests"][0]["status"] == "cached"
