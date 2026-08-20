"""Cycle 4 acceptance tests — feature engineering (goals/cycle4_features.md)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.features import FeatureStep

FEATS = ["feature_alpha", "feature_beta"]


def frame(shift: float = 0.0, rows: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    return pl.DataFrame(
        {
            "era": [f"e{i // 10}" for i in range(rows)],
            "feature_alpha": pl.Series(rng.normal(shift, 1, rows).round(2)),
            "feature_beta": pl.Series(rng.normal(-shift, 2, rows).round(2)),
            "target": pl.Series(rng.uniform(0, 1, rows), dtype=pl.Float32),
        }
    )


@pytest.fixture()
def env(tmp_path):
    return LocalCasStore(tmp_path / "store"), tmp_path / "manifests"


def run_features(store, manifests, train, validation, kind="standard_scaler"):
    seeds = [
        store.save(train, "clean_table", ext="parquet"),
        store.save(validation, "clean_validation", ext="parquet"),
    ]
    config = {"features": {"transformer": {"kind": kind}}}
    return Pipeline([FeatureStep()], store, manifests, config).run(seeds=seeds)


def test_cycle4_transformer_roundtrips_via_store(env):
    store, manifests = env
    train, validation = frame(0.0), frame(3.0)
    result = run_features(store, manifests, train, validation)
    loaded = store.load(result["artifacts"]["fitted_transformer"].content_hash)
    expected_means = train.select(FEATS).to_numpy().astype(np.float32).mean(axis=0)
    assert np.allclose(loaded.mean_, expected_means, atol=1e-5)


def test_cycle4_loaded_transformer_reproduces_validation_features(env):
    store, manifests = env
    train, validation = frame(0.0), frame(3.0)
    result = run_features(store, manifests, train, validation)
    loaded = store.load(result["artifacts"]["fitted_transformer"].content_hash)
    pipeline_features = store.load(result["artifacts"]["feature_validation"].content_hash)
    reapplied = loaded.transform(validation.select(FEATS).to_numpy().astype(np.float32))
    assert np.allclose(pipeline_features.select(FEATS).to_numpy(), reapplied, atol=1e-6)


def test_cycle4_no_leakage_statistics_from_train_only(env):
    store, manifests = env
    train, validation = frame(0.0), frame(3.0)  # deliberately shifted validation
    result = run_features(store, manifests, train, validation)
    loaded = store.load(result["artifacts"]["fitted_transformer"].content_hash)
    train_means = train.select(FEATS).to_numpy().mean(axis=0)
    combined_means = np.vstack(
        [train.select(FEATS).to_numpy(), validation.select(FEATS).to_numpy()]
    ).mean(axis=0)
    assert np.allclose(loaded.mean_, train_means, atol=1e-5)
    assert not np.allclose(loaded.mean_, combined_means, atol=1e-2)


def test_cycle4_passthrough_preserves_int8(env):
    store, manifests = env
    train = pl.DataFrame(
        {
            "era": ["e0", "e1"],
            "feature_alpha": pl.Series([1, 2], dtype=pl.Int8),
            "feature_beta": pl.Series([3, 4], dtype=pl.Int8),
            "target": pl.Series([0.1, 0.9], dtype=pl.Float32),
        }
    )
    result = run_features(store, manifests, train, train, kind="passthrough")
    out = store.load(result["artifacts"]["feature_table"].content_hash)
    assert out["feature_alpha"].dtype == pl.Int8


def test_cycle4_rerun_is_cache_hit(env):
    store, manifests = env
    train, validation = frame(0.0), frame(3.0)
    run_features(store, manifests, train, validation)
    result = run_features(store, manifests, train, validation)
    assert result["manifests"][0]["status"] == "cached"
