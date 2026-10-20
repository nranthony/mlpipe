"""work/0006 phase 2 — generic core (plan steps 9–12)."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

import mlpipe.steps.features as features_mod
from mlpipe.core.interfaces import FoldPlan
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.cvplan import CVConfig, build_plan
from mlpipe.steps.features import FeatureStep

# --------------------------------------------------------------------------- step 9

PINNED_PLAN_HASH = "d3e6be73bfd7d2ecd355ec352ab5684cbe89b559d3d05eab9646926dc1401985"


def test_work0006_foldplan_is_domain_free():
    plan = FoldPlan(folds=[(np.array([0]), np.array([1]))], plan_hash="h")
    assert plan.group_column is None and plan.meta == {}
    assert not any("era" in f or "purge" in f or "embargo" in f for f in plan.__dataclass_fields__)


def test_work0006_cvplan_carries_era_semantics_in_meta():
    eras = pl.Series("era", [f"e{i:02d}" for i in range(12) for _ in range(3)])
    plan = build_plan(eras, CVConfig(n_folds=3, purge_eras=1, embargo_eras=2))
    assert plan.group_column == "era"
    assert plan.meta == {"purge_eras": 1, "embargo_eras": 2}


def test_work0006_cvplan_hash_unchanged_by_refactor():
    # Pinned before the FoldPlan refactor: the hash covers (eras, CVConfig) only,
    # so plans cited in STATUS.md stay comparable across it.
    eras = pl.Series("era", [f"e{i:02d}" for i in range(12) for _ in range(3)])
    assert build_plan(eras, CVConfig(n_folds=3)).plan_hash == PINNED_PLAN_HASH


class SpyScaler(StandardScaler):
    calls: list = []

    def fit(self, X, y=None, **kw):
        SpyScaler.calls.append((X.shape, y))
        return super().fit(X, y, **kw)


def test_work0006_feature_fits_never_see_the_target(tmp_path, monkeypatch):
    """D1: FeatureStep fits are unsupervised. Target-aware transforms belong in the
    backend, fitted per fold, or they leak the target across CV folds."""
    rng = np.random.default_rng(0)
    train = pl.DataFrame({
        "era": [f"e{i // 10}" for i in range(40)],
        "feature_a": rng.normal(size=40),
        "feature_b": rng.normal(size=40),
        "target": rng.uniform(size=40),
    })
    SpyScaler.calls = []
    monkeypatch.setattr(features_mod, "make_transformer", lambda kind: SpyScaler())
    store = LocalCasStore(tmp_path / "store")
    seeds = [store.save(train, "clean_table", ext="parquet"),
             store.save(train, "clean_validation", ext="parquet")]
    Pipeline([FeatureStep()], store, tmp_path / "m", {}).run(seeds=seeds)
    assert SpyScaler.calls == [((40, 2), None)]
