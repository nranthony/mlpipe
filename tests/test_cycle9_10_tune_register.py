"""Cycles 9 & 10 acceptance tests — tune (Optuna) and register/package,
end-to-end on a toy snapshot through the real CLI."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from mlpipe.cli import app
from mlpipe.core.manifest import ManifestLog
from mlpipe.core.store import LocalCasStore

runner = CliRunner()


def toy_snapshot(root):
    raw = root / "raw" / "v0-toy"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(5)
    rows = 120
    fa = rng.integers(0, 5, rows).astype(np.int8)
    fb = rng.integers(0, 5, rows).astype(np.int8)
    target = (0.25 * fa - 0.15 * fb + rng.normal(0, 0.2, rows)).astype(np.float32)
    df = pl.DataFrame(
        {
            "era": [f"e{i // 15:03d}" for i in range(rows)],
            "feature_fa": fa,
            "feature_fb": fb,
            "target": target,
        }
    )
    for name in ("train.parquet", "validation.parquet", "live.parquet"):
        df.write_parquet(raw / name)
    (raw / "features.json").write_text(
        json.dumps({"feature_sets": {"all": ["feature_fa", "feature_fb"]}, "targets": ["target"]})
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    toy_snapshot(tmp)
    config = {
        "download_dir": str(tmp / "raw"),
        "ingest": {"version": "v0-toy"},
        "validate": {"row_tolerance": 0.5},
        "clean": {"cleaner": {"kind": "baseline", "feature_set": "all"}},
        "features": {"transformer": {"kind": "passthrough"}},
        "cvplan": {"n_folds": 2, "purge_eras": 0, "embargo_eras": 0},
        "train": {"model": {"kind": "lightgbm", "params": {"n_estimators": 5}}},
        "evaluate": {"target": "target"},
        "register": {"model_name": "toy-model"},
        "tracker": {"kind": "mlflow", "uri": f"sqlite:///{tmp}/mlflow.db", "experiment": "toy"},
        "tune": {
            "study": "toy-study",
            "storage": f"sqlite:///{tmp}/optuna.db",
            "n_trials": 2,
            "space": {"train.model.params.num_leaves": {"type": "int", "low": 4, "high": 16}},
        },
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    dirs = [
        "--config", str(cfg_path),
        "--store-dir", str(tmp / "store"),
        "--manifests-dir", str(tmp / "manifests"),
    ]
    result = runner.invoke(app, ["run", "--pipeline", "numerai", *dirs])
    assert result.exit_code == 0, result.output
    return tmp, cfg_path, dirs


def test_cycle10_register_creates_loadable_version(env):
    tmp, _, _ = env
    log = ManifestLog(tmp / "manifests")
    reg = [m for m in log.entries() if m["step"] == "register"][-1]
    bundle_hash = reg["outputs"]["model_bundle"]
    bundle = LocalCasStore(tmp / "store").load(bundle_hash)
    assert bundle["features"] == ["feature_fa", "feature_fb"]
    assert bundle["plan_hash"]

    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    versions = mlflow.MlflowClient().search_model_versions("name='toy-model'")
    assert versions and versions[0].tags["content_hash"] == bundle_hash


def test_cycle10_fresh_process_predicts_from_raw_live(env):
    tmp, _, dirs = env
    out = tmp / "preds.parquet"
    proc = subprocess.run(
        [sys.executable, "-m", "mlpipe.cli", "predict", "--name", "toy-model", *dirs,
         "--out", str(out)],
        capture_output=True, text=True, cwd="/workspace/mlpipe", check=False,
    )
    assert proc.returncode == 0, proc.stderr
    preds = pl.read_parquet(out)
    assert preds.height == 120 and "prediction" in preds.columns


def test_cycle10_lineage_from_registered_bundle_reaches_raw_snapshot(env):
    tmp, _, _ = env
    log = ManifestLog(tmp / "manifests")
    reg = [m for m in log.entries() if m["step"] == "register"][-1]
    chain = log.lineage(reg["outputs"]["model_bundle"])
    steps = [m["step"] for m in chain]
    assert steps[0] == "register" and "ingest" in steps
    ingest = next(m for m in chain if m["step"] == "ingest")
    assert "raw_train" in ingest["outputs"]  # raw snapshot hash reachable


def test_cycle9_tune_runs_resume_and_promote(env):
    tmp, _, dirs = env
    baselines = tmp / "baselines.yaml"
    for _ in range(2):  # second invocation resumes the same study
        result = runner.invoke(
            app, ["tune", "--pipeline", "numerai", "--n-trials", "2", *dirs,
                  "--baselines", str(baselines)],
        )
        assert result.exit_code == 0, result.output

    import optuna

    study = optuna.load_study(study_name="toy-study", storage=f"sqlite:///{tmp}/optuna.db")
    assert len(study.trials) == 4  # interrupted/re-invoked study resumed, not restarted

    promoted = yaml.safe_load(baselines.read_text())["promoted"]
    assert promoted["study"] == "toy-study"
    assert "train.model.params.num_leaves" in promoted["params"]

    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    runs = mlflow.search_runs(experiment_names=["toy"])
    tags = [t for t in runs["params.tag"].dropna() if t.startswith("toy-study-trial")]
    assert len(tags) >= 4  # every trial appears in MLflow, tagged study+trial
