"""Cycle 7 acceptance tests — evaluate + MLflow wiring (goals/cycle7_evaluate.md)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlpipe.core.mlflow_tracker import MlflowTracker
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.cvplan import CVConfig, build_plan
from mlpipe.steps.eval_metrics import era_scores, summarize
from mlpipe.steps.evaluate import EvaluateStep


def oof_frame() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    rows = 120
    target = rng.uniform(0, 1, rows).astype(np.float32)
    noisy = (target + rng.normal(0, 0.4, rows)).astype(np.float32)
    return pl.DataFrame(
        {
            "era": [f"e{i // 20:03d}" for i in range(rows)],
            "prediction": noisy,
            "target": target,
        }
    )


def test_cycle7_metric_functions():
    df = oof_frame()
    scores = era_scores(df)
    assert scores.height == 6
    summary = summarize(scores)
    assert 0 < summary["mean_corr"] <= 1
    assert summary["n_eras"] == 6
    assert summary["max_drawdown"] >= 0
    perfect = df.with_columns(pl.col("target").alias("prediction"))
    # predictions are rank-percentiled before correlating, so perfect raw
    # predictions correlate ~0.98 with the raw target, not exactly 1.0
    assert summarize(era_scores(perfect))["mean_corr"] > 0.95


@pytest.fixture()
def env(tmp_path):
    store = LocalCasStore(tmp_path / "store")
    oof = oof_frame()
    features = oof.select("era", "target").with_columns(
        pl.col("target").rank().cast(pl.Int8).alias("feature_probe")
    )
    plan = build_plan(oof["era"], CVConfig(n_folds=2, purge_eras=0, embargo_eras=0))
    seeds = [
        store.save({"full": "stub"}, "model", ext="joblib"),
        store.save(oof, "oof_preds", ext="parquet"),
        store.save(plan, "fold_plan", ext="joblib"),
        store.save(features, "feature_table", ext="parquet"),
    ]
    return store, tmp_path, seeds, plan


def test_cycle7_evaluate_step_report_and_chart(env):
    store, tmp, seeds, plan = env
    result = Pipeline([EvaluateStep()], store, tmp / "m", {}).run(seeds=seeds)
    (m,) = result["manifests"]
    assert {"mean_corr", "sharpe", "max_drawdown", "feature_exposure"} <= set(m["metrics"])
    report = store.load(result["artifacts"]["eval_report"].content_hash)
    assert report["plan_hash"] == plan.plan_hash
    html = store.load(result["artifacts"]["eval_chart"].content_hash)
    assert html.startswith("<!DOCTYPE html") or "<html" in html[:200]
    assert "plotly" in html  # standalone: plotly.js embedded, renders offline


def test_cycle7_mlflow_two_runs_comparable_and_disposable(env, tmp_path):
    store, tmp, seeds, _ = env
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    for tag in ("a", "b"):
        tracker = MlflowTracker(uri, experiment="test-exp")
        Pipeline([EvaluateStep()], store, tmp / f"m{tag}", {}, tracker).run(
            seeds=seeds, tag=tag
        )
    import mlflow

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert len(runs) == 2  # side-by-side comparable
    assert set(runs["params.tag"]) == {"a", "b"}
    assert runs["metrics.mean_corr"].notna().all()
    # deleting the tracker DB loses zero provenance: manifests still complete
    (tmp_path / "mlflow.db").unlink()
    assert (tmp / "ma" / "index.jsonl").exists() and (tmp / "mb" / "index.jsonl").exists()


def test_cycle7_rerun_is_cache_hit(env):
    store, tmp, seeds, _ = env
    Pipeline([EvaluateStep()], store, tmp / "m", {}).run(seeds=seeds)
    result = Pipeline([EvaluateStep()], store, tmp / "m", {}).run(seeds=seeds)
    assert result["manifests"][0]["status"] == "cached"
