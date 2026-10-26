"""work/0006 step 12 — a second, non-Numerai domain through the shared steps.

Synthetic customer data (`mlpipe.steps.demo_tabular`): several rows per customer,
a regression target and a binary one. No eras anywhere. The pipeline reuses
FeatureStep, TrainStep and RegisterStep unchanged, and swaps in the generic CV and
evaluate steps.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from mlpipe.cli import app
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.demo_tabular import build_steps as steps
from mlpipe.steps.demo_tabular import customers
from mlpipe.steps.groupcv import GroupKFoldConfig, build_group_plan
from mlpipe.steps.tabular_metrics import binary_metrics, regression_metrics


def config(task: str) -> dict:
    target, kind = {"regression": ("spend", "lightgbm"), "binary": ("churned", "lightgbm_binary")}[task]
    params = {"n_estimators": 60, "num_leaves": 8, "min_child_samples": 5, "n_jobs": 1}
    return {
        "cvplan": {"n_folds": 4, "group_column": "customer_id"},
        "train": {"model": {"kind": kind, "params": params}, "target": target},
        "evaluate": {"task": task, "target": target},
        "register": {"model_name": f"churn-{task}", "id_columns": ["customer_id"]},
    }


@pytest.fixture()
def env(tmp_path):
    return LocalCasStore(tmp_path / "store"), tmp_path / "manifests"


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_work0006_second_domain_runs_end_to_end_and_caches(env, task):
    store, manifests = env
    result = Pipeline(steps(), store, manifests, config(task)).run()
    report = store.load(result["artifacts"]["eval_report"].content_hash)
    m = report["metrics"]
    if task == "regression":
        assert m["val_r2"] > 0.6 and m["r2"] > 0.6
    else:
        assert m["val_roc_auc"] > 0.85 and 0.0 < m["val_log_loss"] < 0.6
    assert m["n_rows"] == 720 and m["val_n_rows"] == 240

    bundle = store.load(result["artifacts"]["model_bundle"].content_hash)
    assert bundle["plan_hash"] == report["plan_hash"] and "era_column" not in bundle
    for key in ("clean_table", "feature_table", "oof_preds"):
        assert "era" not in store.load(result["artifacts"][key].content_hash).columns

    again = Pipeline(steps(), store, manifests, config(task)).run()
    assert {m["status"] for m in again["manifests"]} == {"cached"}


def test_work0006_default_prepare_carries_the_bundle_id_columns(env, tmp_path):
    """`demo` has no serve steps, so predict scores `--input` with the default
    prepare: bundle features in, the register step's `id_columns` carried through."""
    store, manifests = env
    result = Pipeline(steps(), store, manifests, config("regression")).run()
    bundle = result["artifacts"]["model_bundle"].content_hash
    src, out = tmp_path / "score.parquet", tmp_path / "preds.parquet"
    customers(5, 2000, np.random.default_rng(3)).write_parquet(src)
    cli = CliRunner().invoke(app, [
        "predict", "--pipeline", "demo", "--bundle", bundle[:10], "--input", str(src),
        "--store-dir", str(tmp_path / "store"), "--manifests-dir", str(manifests), "--out", str(out),
    ])
    assert cli.exit_code == 0, cli.output
    preds = pl.read_parquet(out)
    assert preds.columns == ["customer_id", "prediction"] and preds.height == 30


def test_work0006_group_kfold_never_splits_a_customer():
    rng = np.random.default_rng(1)
    df = customers(30, 0, rng)
    plan = build_group_plan(df["customer_id"], df.height, GroupKFoldConfig(n_folds=5,
                                                                          group_column="customer_id"))
    ids = df["customer_id"].to_numpy()
    valid_union = np.concatenate([va for _, va in plan.folds])
    assert sorted(valid_union) == list(range(df.height))  # each row validated exactly once
    for tr, va in plan.folds:
        assert not set(ids[tr]) & set(ids[va])
    assert plan.group_column == "customer_id" and plan.meta["n_groups"] == 30


def test_work0006_group_kfold_hash_covers_groups_and_config():
    df = customers(30, 0, np.random.default_rng(1))
    cfg = GroupKFoldConfig(n_folds=5, group_column="customer_id")
    base = build_group_plan(df["customer_id"], df.height, cfg).plan_hash
    assert build_group_plan(df["customer_id"], df.height, cfg).plan_hash == base
    reseeded = cfg.model_copy(update={"seed": 1})
    assert build_group_plan(df["customer_id"], df.height, reseeded).plan_hash != base
    assert build_group_plan(df["customer_id"].head(60), 60, cfg).plan_hash != base


def test_work0006_group_kfold_without_groups_is_row_kfold():
    plan = build_group_plan(None, 50, GroupKFoldConfig(n_folds=5))
    assert plan.group_column is None and [len(va) for _, va in plan.folds] == [10] * 5
    with pytest.raises(ValueError, match="3 groups cannot fill 5 folds"):
        build_group_plan(pl.Series(["a", "b", "c"]), 3, GroupKFoldConfig(n_folds=5))


def test_work0006_tabular_metrics():
    y = np.array([0.0, 0.0, 1.0, 1.0, np.nan])
    assert binary_metrics(y, np.array([0.1, 0.4, 0.35, 0.8, 0.5]))["roc_auc"] == 0.75
    with pytest.raises(ValueError, match="probabilities in"):
        binary_metrics(y, np.array([-1.0, 0.2, 0.3, 2.0, 0.0]))  # a regressor's raw scores
    with pytest.raises(ValueError, match="0/1 target"):
        binary_metrics(np.array([0.0, 2.0]), np.array([0.1, 0.9]))
    r = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert r["rmse"] == 0.0 and r["r2"] == 1.0 and r["n_rows"] == 3
