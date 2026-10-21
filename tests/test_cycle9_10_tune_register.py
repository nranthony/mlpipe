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
            "metric": "mean_corr",
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
        "--pipeline", "numerai",
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
    return yaml.safe_load((tmp / "baselines.yaml").read_text())["promoted"]


def test_cycle10_register_builds_bundle_without_registering(env):
    tmp, _, _ = env
    bundle = LocalCasStore(tmp / "store").load(bundle_hash_of(tmp))
    assert bundle["features"] == ["feature_fa", "feature_fb"]
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
    assert pinned["plan_hash"] and {"mean_corr", "val_mean_corr"} <= set(pinned["metrics"])
    assert pinned["steps"]["train"]["config"]["model"]["params"]["n_estimators"] == 5
    assert set(pinned["steps"]) >= {"ingest", "clean", "cvplan", "train", "register"}

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


def test_cycle10_fresh_process_predicts_from_raw_live(env):
    tmp, _, dirs = env
    promote(tmp, dirs, bundle_hash_of(tmp))
    out = tmp / "preds.parquet"
    proc = subprocess.run(
        [sys.executable, "-m", "mlpipe.cli", "predict", *dirs[2:],
         "--baselines", str(tmp / "baselines.yaml"), "--out", str(out)],
        capture_output=True, text=True, cwd="/workspace/mlpipe", check=False,
    )
    assert proc.returncode == 0, proc.stderr
    preds = pl.read_parquet(out)
    assert preds.height == 120 and preds.columns == ["era", "prediction"]  # toy live: no ticker


def test_work0006_predict_scores_an_explicit_input_keyed_by_ticker(env):
    tmp, _, dirs = env
    promote(tmp, dirs, bundle_hash_of(tmp))
    live = pl.read_parquet(tmp / "raw" / "v0-toy" / "live.parquet").head(7)
    src = tmp / "to_score.parquet"
    live.with_columns(pl.Series("numerai_ticker", [f"T{i}" for i in range(7)])).write_parquet(src)
    out = tmp / "explicit.parquet"
    result = runner.invoke(app, ["predict", *dirs[2:], "--input", str(src),
                                 "--baselines", str(tmp / "baselines.yaml"), "--out", str(out)])
    assert result.exit_code == 0, result.output
    preds = pl.read_parquet(out)
    assert preds.columns == ["numerai_ticker", "era", "prediction"] and preds.height == 7


def test_cycle10_lineage_from_registered_bundle_reaches_raw_snapshot(env):
    tmp, _, _ = env
    log = ManifestLog(tmp / "manifests")
    reg = [m for m in log.entries() if m["step"] == "register"][-1]
    chain = log.lineage(reg["outputs"]["model_bundle"])
    steps = [m["step"] for m in chain]
    assert steps[0] == "register" and "ingest" in steps
    ingest = next(m for m in chain if m["step"] == "ingest")
    assert "raw_train" in ingest["outputs"]  # raw snapshot hash reachable


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
    lineage_dirs = ["--manifests-dir", str(tmp / "manifests"), "--pipeline", "numerai"]
    result = runner.invoke(app, ["lineage", bundle_hash_of(tmp)[:8], *lineage_dirs])
    assert result.exit_code == 0 and result.output.splitlines()[0].startswith("register")
    assert runner.invoke(app, ["lineage", "ffffffff", *lineage_dirs]).exit_code != 0
