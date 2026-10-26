"""Cycles 9 & 10 acceptance tests — tune (Optuna) and register/package,
end-to-end through the real CLI on the generic `demo-tabular` pipeline (ADR-0009:
library behaviour is tested without the Numerai harness, whose serve path is in
test_work0009_numerai.py)."""

from __future__ import annotations

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
from mlpipe.steps.demo_tabular import customers

runner = CliRunner()
PIPELINE = "demo-tabular"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    config = {
        "make": {"seed": 0, "customers": 40, "validation_customers": 20},
        "features": {"transformer": {"kind": "passthrough"}},
        "cvplan": {"n_folds": 2, "group_column": "customer_id"},
        "train": {"model": {"kind": "lightgbm", "params": {"n_estimators": 5}}, "target": "spend"},
        "evaluate": {"task": "regression", "target": "spend"},
        "register": {"model_name": "toy-model", "id_columns": ["customer_id"]},
        "tracker": {"kind": "mlflow", "uri": f"sqlite:///{tmp}/mlflow.db", "experiment": "toy"},
        "tune": {
            "study": "toy-study",
            "storage": f"sqlite:///{tmp}/optuna.db",
            "metric": "r2",
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
        "--pipeline", PIPELINE,
    ]
    result = runner.invoke(app, ["run", *dirs])
    assert result.exit_code == 0, result.output
    return tmp, cfg_path, dirs


def bundle_hash_of(tmp) -> str:
    log = ManifestLog(tmp / "manifests")
    return [m for m in log.entries() if m["step"] == "register"][-1]["outputs"]["model_bundle"]


def promote(tmp, dirs, bundle):
    result = runner.invoke(
        app, ["promote", bundle, *dirs, "--baselines", str(tmp / "baselines.yaml")]
    )
    assert result.exit_code == 0, result.output
    return yaml.safe_load((tmp / "baselines.yaml").read_text())[PIPELINE]["champion"]


def test_cycle10_register_builds_bundle_without_registering(env):
    tmp, _, _ = env
    bundle = LocalCasStore(tmp / "store").load(bundle_hash_of(tmp))
    assert bundle["features"] == ["feature_x1", "feature_x2", "feature_noise"]
    assert bundle["plan_hash"] and bundle["model_name"] == "toy-model"

    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    names = [m.name for m in mlflow.MlflowClient().search_registered_models()]
    assert "toy-model" not in names  # ADR-0006: runs never register on their own


def test_adr0006_promote_pins_baseline_and_moves_alias(env):
    tmp, _, dirs = env
    full = bundle_hash_of(tmp)
    pinned = promote(tmp, dirs, full[:10])  # unique prefix resolves
    assert pinned["bundle_hash"] == full and pinned["alias"] == "champion"
    assert pinned["plan_hash"] and {"r2", "val_r2"} <= set(pinned["metrics"])
    assert pinned["steps"]["train"]["config"]["model"]["params"]["n_estimators"] == 5
    assert set(pinned["steps"]) >= {"make", "cvplan", "train", "register"}

    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    client = mlflow.MlflowClient()
    champion = client.get_model_version_by_alias("toy-model", "champion")
    assert champion.tags["content_hash"] == full
    assert promote(tmp, dirs, full)["registry_version"] == champion.version  # no duplicate
    assert len(client.search_model_versions("name='toy-model'")) == 1
    log = ManifestLog(tmp / "manifests")
    assert [m for m in log.entries() if m["step"] == "promote"][-1]["inputs"] == {
        "model_bundle": full
    }


def test_adr0006_promote_rejects_unknown_bundle(env):
    _, _, dirs = env
    result = runner.invoke(app, ["promote", "ffffffff", *dirs])
    assert result.exit_code != 0


def test_cycle10_fresh_process_predicts_from_stored_state(env):
    """A new process serves from the store, manifests and pin alone."""
    tmp, _, dirs = env
    promote(tmp, dirs, bundle_hash_of(tmp))
    src, out = tmp / "to_score.parquet", tmp / "preds.parquet"
    customers(5, 2000, np.random.default_rng(3)).write_parquet(src)
    proc = subprocess.run(
        [sys.executable, "-m", "mlpipe.cli", "predict", *dirs, "--input", str(src),
         "--baselines", str(tmp / "baselines.yaml"), "--out", str(out)],
        capture_output=True, text=True, cwd="/workspace/mlpipe", check=False,
    )
    assert proc.returncode == 0, proc.stderr
    preds = pl.read_parquet(out)
    assert preds.height == 30 and preds.columns == ["customer_id", "prediction"]


def test_cycle10_lineage_from_registered_bundle_reaches_raw_snapshot(env):
    tmp, _, _ = env
    log = ManifestLog(tmp / "manifests")
    reg = [m for m in log.entries() if m["step"] == "register"][-1]
    chain = log.lineage(reg["outputs"]["model_bundle"])
    steps = [m["step"] for m in chain]
    assert steps[0] == "register" and "make" in steps
    source = next(m for m in chain if m["step"] == "make")
    assert "clean_table" in source["outputs"]  # the source snapshot hash is reachable


def test_cycle9_tune_runs_resume_and_never_promotes(env):
    tmp, _, dirs = env
    before = (tmp / "baselines.yaml").read_text() if (tmp / "baselines.yaml").exists() else None
    for _ in range(2):  # second invocation resumes the same study
        result = runner.invoke(app, ["tune", "--n-trials", "2", *dirs])
        assert result.exit_code == 0, result.output
    assert "train.model.params.num_leaves=" in result.output  # names the candidate

    import optuna

    study = optuna.load_study(study_name="toy-study", storage=f"sqlite:///{tmp}/optuna.db")
    assert len(study.trials) == 4  # interrupted/re-invoked study resumed, not restarted

    after = (tmp / "baselines.yaml").read_text() if (tmp / "baselines.yaml").exists() else None
    assert after == before  # ADR-0006: tune does not write baselines

    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    runs = mlflow.search_runs(experiment_names=["toy"])
    tags = [t for t in runs["params.tag"].dropna() if t.startswith("toy-study-trial")]
    assert len(tags) >= 4  # every trial appears in MLflow, tagged study+trial


def test_work0006_lineage_resolves_a_prefix_and_rejects_unknown_hashes(env):
    tmp, _, _ = env
    lineage_dirs = ["--manifests-dir", str(tmp / "manifests"), "--pipeline", PIPELINE]
    result = runner.invoke(app, ["lineage", bundle_hash_of(tmp)[:8], *lineage_dirs])
    assert result.exit_code == 0 and result.output.splitlines()[0].startswith("register")
    assert runner.invoke(app, ["lineage", "ffffffff", *lineage_dirs]).exit_code != 0
