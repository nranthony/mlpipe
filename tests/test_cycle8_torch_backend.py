"""Cycle 8 acceptance tests — Torch backend, the port proof (goals/cycle8_torch_backend.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_cycle6_train import training_frame

import mlpipe.steps.train as train_module
from mlpipe.backends.torch_backend import TorchMLPBackend, estimate_vram_gb
from mlpipe.core.interfaces import ModelBackend
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.cvplan import CVConfig, build_plan
from mlpipe.steps.train import TrainStep

TORCH_PARAMS = {"hidden": [16], "epochs": 2, "batch_size": 64}


def test_cycle8_port_conformance():
    from mlpipe.backends.lightgbm_backend import LightGBMBackend

    assert isinstance(TorchMLPBackend(), ModelBackend)
    assert isinstance(LightGBMBackend(), ModelBackend)


def test_cycle8_trainstep_source_has_no_torch_import():
    import re

    source = Path(train_module.__file__).read_text()
    assert not re.search(r"^\s*(import|from)\s+torch", source, re.MULTILINE)


def test_cycle8_swap_via_config_only(tmp_path):
    """--set train.model.kind=torch_mlp: same TrainStep, same seeds, new backend."""
    store = LocalCasStore(tmp_path / "store")
    df = training_frame()
    plan = build_plan(df["era"], CVConfig(n_folds=2, purge_eras=1, embargo_eras=0))
    seeds = [
        store.save(df, "feature_table", ext="parquet"),
        store.save(plan, "fold_plan", ext="joblib"),
    ]
    config = {"train": {"model": {"kind": "lightgbm", "params": {"n_estimators": 10}}}}
    pipe = Pipeline([TrainStep()], store, tmp_path / "m", config)
    pipe.run(seeds=seeds)
    result = pipe.run(
        seeds=seeds,
        overrides=[
            "train.model.kind=torch_mlp",
            f"train.model.params={TORCH_PARAMS}",
        ],
    )
    (m,) = result["manifests"]
    assert m["status"] == "success"  # config change re-trains, no code edits
    assert m["meta"]["model"]["kind"] == "torch_mlp"
    assert m["meta"]["plan_hash"] == plan.plan_hash  # comparable: same plan
    bundle = store.load(result["artifacts"]["model"].content_hash)
    assert isinstance(bundle["full"], TorchMLPBackend)
    oof = store.load(result["artifacts"]["oof_preds"].content_hash)
    valid_union = np.concatenate([va for _, va in plan.folds])
    assert not np.isnan(oof["prediction"].to_numpy()[valid_union]).any()


def test_cycle8_vram_guard_blocks_oversized_config(tmp_path):
    df = training_frame(rows=64)
    backend = TorchMLPBackend()
    with pytest.raises(MemoryError, match="VRAM"):
        backend.fit(
            df.select("feature_one", "feature_two"),
            df["target"].to_numpy(),
            {"hidden": [4096] * 64, "batch_size": 2_000_000, "vram_budget_gb": 12.0},
        )
    assert estimate_vram_gb(780, [256, 128], 4096) < 1.0  # our real config is tiny


def test_cycle8_backend_roundtrip(tmp_path):
    df = training_frame()
    X = df.select("feature_one", "feature_two")
    y = df["target"].to_numpy()
    backend = TorchMLPBackend()
    backend.fit(X, y, TORCH_PARAMS)
    preds = backend.predict(X)
    path = tmp_path / "model.joblib"
    backend.save(path)
    fresh = TorchMLPBackend()
    fresh.load(path)
    assert np.allclose(fresh.predict(X), preds, atol=1e-6)
