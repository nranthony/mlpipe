"""Cycle 1 acceptance tests — ingest & snapshot (goals/cycle1_ingest.md).

Fixtures pre-place files in the download cache, so `fetch` never hits the
network: the download is skipped whenever the file already exists.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.ingest import DATASETS, IngestStep


def fixture_frame(rows: int = 6) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "era": [f"e{i % 3}" for i in range(rows)],
            "feature_alpha": pl.Series(range(rows), dtype=pl.Int8),
            "feature_beta": pl.Series([i % 5 for i in range(rows)], dtype=pl.Int8),
            "target": pl.Series([i / rows for i in range(rows)], dtype=pl.Float32),
        }
    )


@pytest.fixture()
def env(tmp_path):
    raw = tmp_path / "raw" / "v9.9-test"
    raw.mkdir(parents=True)
    for fname in DATASETS.values():
        fixture_frame().write_parquet(raw / fname)
    (raw / "features.json").write_text(
        json.dumps({"feature_sets": {"small": ["feature_alpha", "feature_beta"]}})
    )
    config = {
        "download_dir": str(tmp_path / "raw"),
        "ingest": {"version": "v9.9-test"},
    }
    store = LocalCasStore(tmp_path / "store")
    return store, tmp_path / "manifests", config


def build_pipeline(store, manifests, config):
    return Pipeline([IngestStep()], store, manifests, config)


def test_cycle1_snapshot_roundtrip(env):
    store, manifests, config = env
    result = build_pipeline(store, manifests, config).run()

    for key in ("raw_train", "raw_validation", "raw_live"):
        df = store.load(result["artifacts"][key].content_hash)
        assert isinstance(df, pl.DataFrame)
        assert df["feature_alpha"].dtype == pl.Int8  # int8 intact through snapshot
        assert df.equals(fixture_frame())

    meta = store.load(result["artifacts"]["features_meta"].content_hash)
    assert meta["feature_sets"]["small"] == ["feature_alpha", "feature_beta"]


def test_cycle1_rerun_is_cache_hit(env):
    store, manifests, config = env
    build_pipeline(store, manifests, config).run()
    result = build_pipeline(store, manifests, config).run()
    (manifest,) = result["manifests"]
    assert manifest["status"] == "cached"
    assert manifest["meta"]["dataset"]["version"] == "v9.9-test"  # meta survives caching


def test_cycle1_manifest_records_dataset(env):
    store, manifests, config = env
    result = build_pipeline(store, manifests, config).run()
    (manifest,) = result["manifests"]
    meta = manifest["meta"]
    assert meta["dataset"] == {"name": "numerai", "version": "v9.9-test"}
    for key, fname in DATASETS.items():
        assert meta[key]["file"] == f"v9.9-test/{fname}"
        assert meta[key]["bytes"] > 0
        assert meta[key]["rows"] == 6


def test_cycle1_version_change_busts_cache_but_download_dir_does_not(env, tmp_path):
    store, manifests, config = env
    build_pipeline(store, manifests, config).run()

    moved = dict(config, download_dir=str(tmp_path / "raw2"))
    (tmp_path / "raw2").mkdir()
    (tmp_path / "raw" / "v9.9-test").rename(tmp_path / "raw2" / "v9.9-test")
    result = build_pipeline(store, manifests, moved).run()
    assert result["manifests"][0]["status"] == "cached"  # infra move: same identity

    result = build_pipeline(
        store, manifests, dict(moved, ingest={"version": "v9.9-test", "round": 2})
    ).run()
    assert result["manifests"][0]["status"] == "success"  # identity change: re-runs
