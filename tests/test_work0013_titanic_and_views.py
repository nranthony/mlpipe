"""work/0013 — the Titanic example and the viewing surfaces (ADR-0010).

Tests never read `raw_data`: a synthetic, Titanic-shaped CSV stands in for Kaggle's
`train.csv`, pinned by its own sha256 through a config override.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from mlpipe.backends.lightgbm_backend import LightGBMBinaryBackend
from mlpipe.backends.torch_backend import TorchMLPBackend
from mlpipe.cli import app
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.examples.titanic.clean import features, stratified_holdout
from mlpipe.examples.titanic.pipeline import DEFAULT_CONFIG, build_steps
from mlpipe.steps import charts
from mlpipe.steps.ingest_file import sha256_file
from mlpipe.steps.train import history_rows, importance_rows
from mlpipe.trackers.mlflow_tracker import flat_params

ROUNDS = 30
TITLES = ["Mr", "Mrs", "Miss", "Master", "Dr", "Mlle"]


def titanic_like(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], n)
    pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.2, 0.55])
    age = np.where(rng.random(n) < 0.2, np.nan, rng.uniform(1, 70, n).round())
    logit = 2.2 * (sex == "female") - 0.9 * (pclass - 2) + rng.normal(0, 0.8, n) - 0.6
    title = [rng.choice(["Mrs", "Miss", "Mlle"]) if s == "female" else rng.choice(["Mr", "Master", "Dr"])
             for s in sex]
    return pl.DataFrame({
        "PassengerId": np.arange(1, n + 1),
        "Survived": (logit > 0).astype(int),
        "Pclass": pclass,
        "Name": [f"Family{i}, {t}. Person" for i, t in enumerate(title)],
        "Sex": sex,
        "Age": age,
        "SibSp": rng.integers(0, 3, n),
        "Parch": rng.integers(0, 3, n),
        "Ticket": [f"T{i}" for i in range(n)],
        "Fare": rng.uniform(5, 200, n).round(2),
        "Cabin": [None if rng.random() < 0.7 else f"{rng.choice(list('ABCDE'))}{i}" for i in range(n)],
        "Embarked": rng.choice(["S", "C", "Q", None], n, p=[0.7, 0.18, 0.1, 0.02]),
    })


@pytest.fixture()
def titanic(tmp_path):
    """(config, store, manifests dir) for a run on a synthetic train.csv."""
    src = tmp_path / "raw" / "titanic" / "train.csv"
    src.parent.mkdir(parents=True)
    titanic_like(300).write_csv(src)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["download_dir"] = str(tmp_path / "raw")
    cfg["tracker"] = {"kind": "none"}
    cfg["ingest"]["files"]["raw_train"]["sha256"] = sha256_file(src)
    cfg["train"]["model"]["params"]["n_estimators"] = ROUNDS
    return cfg, LocalCasStore(tmp_path / "store"), tmp_path / "manifests"


class RecordingTracker:
    """Records what reaches the tracker port."""

    def __init__(self):
        self.curves, self.artifacts, self.metrics = {}, [], []

    def start_run(self, manifest):
        return "rec"

    def log_metrics(self, metrics, step=None):
        self.metrics.append((metrics, step))

    def log_curve(self, name, values):
        self.curves[name] = values

    def log_artifact(self, artifact):
        self.artifacts.append(artifact.key)

    def register_model(self, artifact, name):
        return None

    def set_alias(self, name, alias, version):
        pass

    def end_run(self, status):
        pass


class OldTracker(RecordingTracker):
    """A host tracker written before ADR-0010: no log_curve, no log_artifact."""

    log_curve = None
    log_artifact = None

    def __getattribute__(self, name):
        if name in ("log_curve", "log_artifact"):
            raise AttributeError(name)
        return super().__getattribute__(name)


# ------------------------------------------------------------------ phase A


def test_work0013_titanic_runs_end_to_end_and_caches(titanic):
    cfg, store, manifests = titanic
    result = Pipeline(build_steps(), store, manifests, cfg).run()
    arts = result["artifacts"]
    report = store.load(arts["eval_report"].content_hash)
    assert report["task"] == "binary" and report["metrics"]["val_roc_auc"] > 0.75
    assert report["metrics"]["n_rows"] + report["metrics"]["val_n_rows"] == 300
    clean = store.load(arts["clean_table"].content_hash)
    assert {"feature_sex_female", "feature_title", "feature_deck"} <= set(clean.columns)

    again = Pipeline(build_steps(), store, manifests, cfg).run()
    assert {m["status"] for m in again["manifests"]} == {"cached"}


def test_work0013_ingest_refuses_a_file_that_moved_from_its_pin(titanic):
    cfg, store, manifests = titanic
    cfg["ingest"]["files"]["raw_train"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 is [0-9a-f]{64}, config pins 0000"):
        Pipeline(build_steps(), store, manifests, cfg).run(to_step="ingest")


def test_work0013_ingest_files_must_match_the_declared_outputs(titanic):
    cfg, store, manifests = titanic
    cfg["ingest"]["files"]["extra"] = dict(cfg["ingest"]["files"]["raw_train"])
    with pytest.raises(ValueError, match="must name exactly"):
        Pipeline(build_steps(), store, manifests, cfg).run(to_step="ingest")


def test_work0013_titles_and_holdout():
    raw = titanic_like(200, seed=3)
    f = features(raw)
    title_of = dict(zip(raw["Name"], f["feature_title"], strict=True))
    assert {title_of[n] for n in raw["Name"] if ", Dr." in n} == {4}  # rare
    assert {title_of[n] for n in raw["Name"] if ", Mlle." in n} == {2}  # alias of Miss
    y = f["Survived"].to_numpy()
    mask = stratified_holdout(y, 0.2, seed=0)
    assert abs(y[mask].mean() - y.mean()) < 0.03  # stratified
    assert (mask == stratified_holdout(y, 0.2, seed=0)).all()


# ------------------------------------------------------------------ phase B


def test_work0013_train_stores_curves_and_importance(titanic):
    cfg, store, manifests = titanic
    tracker = RecordingTracker()
    arts = Pipeline(build_steps(), store, manifests, cfg, tracker=tracker).run()["artifacts"]
    history = store.load(arts["train_history"].content_hash)
    # 5 folds x (train, valid) x (binary_logloss, auc) x ROUNDS
    assert history.height == 5 * 2 * 2 * ROUNDS
    assert set(history["metric"]) == {"binary_logloss", "auc"}
    assert len(tracker.curves["fold0_valid_binary_logloss"]) == ROUNDS
    # curves reach the tracker only, never the manifest's scalar metrics
    train = next(m for m in Pipeline(build_steps(), store, manifests, cfg).run()["manifests"]
                 if m["step"] == "train")
    assert not any(k.startswith("fold0_valid") for k in train["metrics"])

    importance = store.load(arts["feature_importance"].content_hash)
    assert set(importance["model"]) == {"fold0", "fold1", "fold2", "fold3", "fold4", "full"}
    top = importance.filter(pl.col("model") == "full").sort("gain", descending=True)["feature"][0]
    assert top in ("feature_sex_female", "feature_title", "feature_pclass")


def test_work0013_backend_without_curves_gives_empty_frames():
    assert history_rows(0, {}) == []
    assert importance_rows("full", object()) == []


def test_work0013_lightgbm_history_matches_rounds():
    rng = np.random.default_rng(0)
    X = pl.DataFrame({"feature_a": rng.normal(size=200), "feature_b": rng.normal(size=200)})
    y = (X["feature_a"].to_numpy() > 0).astype(int)
    b = LightGBMBinaryBackend()
    b.fit(X[:150], y[:150], {"n_estimators": 12}, eval_set=(X[150:], y[150:]))
    assert set(b.history()) == {"train_binary_logloss", "valid_binary_logloss"}
    assert all(len(v) == 12 for v in b.history().values())
    assert set(b.feature_importance()) == {"feature_a", "feature_b"}
    plain = LightGBMBinaryBackend()
    plain.fit(X[:150], y[:150], {"n_estimators": 12})  # no eval set: no history, same model
    assert plain.history() == {}
    np.testing.assert_allclose(plain.predict(X[150:]), b.predict(X[150:]))


def test_work0013_torch_history_per_epoch():
    rng = np.random.default_rng(0)
    X = pl.DataFrame({"feature_a": rng.normal(size=128).astype(np.float32)})
    y = X["feature_a"].to_numpy() * 2
    b = TorchMLPBackend()
    b.fit(X[:96], y[:96], {"hidden": [8], "epochs": 3, "batch_size": 32}, eval_set=(X[96:], y[96:]))
    h = b.history()
    assert set(h) == {"train_mse", "valid_mse"} and all(len(v) == 3 for v in h.values())
    assert b.feature_importance() is None


def test_work0013_old_tracker_still_gets_curves_as_stepped_metrics(titanic):
    cfg, store, manifests = titanic
    tracker = OldTracker()
    Pipeline(build_steps(), store, manifests, cfg, tracker=tracker).run()
    stepped = [(m, s) for m, s in tracker.metrics if "fold0_valid_auc" in m]
    assert [s for _, s in stepped] == list(range(ROUNDS))


# ------------------------------------------------------------------ phase C


def test_work0013_charts_and_tracker_artifacts(titanic):
    cfg, store, manifests = titanic
    tracker = RecordingTracker()
    arts = Pipeline(build_steps(), store, manifests, cfg, tracker=tracker).run()["artifacts"]
    chart = store.load(arts["eval_chart"].content_hash)
    for heading in ("ROC curve", "Calibration", "Training curves", "Feature importance"):
        assert heading in chart
    profile = store.load(arts["data_profile"].content_hash)
    assert profile["rows"] == 240 and "feature_age" in profile["target_rate_by_bin"]
    assert profile["columns"]["feature_age"]["null_rate"] > 0
    assert set(tracker.artifacts) == {"data_profile", "profile_chart", "eval_report", "eval_chart",
                                      "model_bundle"}
    report = store.load(arts["eval_report"].content_hash)
    assert set(report["curves"]) == {"binary_logloss", "auc"}
    assert 1 <= report["curves"]["auc"]["best_iteration"] <= ROUNDS


def test_work0013_chart_pages_are_byte_stable():
    import plotly.graph_objects as go

    def render():
        return charts.page("t", [("a", go.Figure(go.Bar(x=[1, 2], y=[3, 4])))], {"k": 1})

    assert render() == render()


def test_work0013_curve_summary_direction():
    h = pl.DataFrame({
        "fold": [0] * 6, "iteration": [1, 2, 3] * 2, "split": ["valid"] * 6,
        "metric": ["auc"] * 3 + ["binary_logloss"] * 3, "value": [0.7, 0.9, 0.8, 0.6, 0.4, 0.5],
    })
    s = charts.curve_summary(h)
    assert s["auc"]["best_iteration"] == 2 and s["binary_logloss"]["best_iteration"] == 2
    assert s["auc"]["final_iteration"] == 3


def test_work0013_mlflow_groups_steps_and_links_cache_hits(titanic, tmp_path):
    import mlflow

    from mlpipe.trackers.mlflow_tracker import MlflowTracker

    cfg, store, manifests = titanic
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    ids = []
    for _ in range(2):  # the second run is all cache hits
        tracker = MlflowTracker(uri, experiment="t", pipeline="titanic-example")
        result = Pipeline(build_steps(), store, manifests, cfg, tracker=tracker).run()
        ids.append(result["manifests"][0]["run_id"])

    client = mlflow.MlflowClient(uri)
    exp = client.get_experiment_by_name("t").experiment_id
    runs = client.search_runs([exp], max_results=100)
    parents = [r for r in runs if r.data.tags.get("mlpipe.kind") == "pipeline"]
    assert {p.data.tags["mlpipe.run_id"] for p in parents} == set(ids)
    assert all(p.info.status == "FINISHED" for p in parents)
    names = [s.name for s in build_steps()]
    for parent in parents:
        kids = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == parent.info.run_id]
        assert sorted(k.data.tags["mlpipe.step"] for k in kids) == sorted(names)
    (again,) = [p for p in parents if p.data.tags["mlpipe.run_id"] == ids[1]]
    cached = {r.data.tags["mlpipe.step"]: r for r in runs
              if r.data.tags.get("mlflow.parentRunId") == again.info.run_id}
    assert {r.data.tags["mlpipe.cached"] for r in cached.values()} == {"true"}
    source = client.get_run(cached["train"].data.tags["mlpipe.source_run_id"])
    assert source.data.tags["mlpipe.run_id"] == ids[0] and source.data.tags["mlpipe.cached"] == "false"
    assert len(client.get_metric_history(source.info.run_id, "fold0_valid_auc")) == ROUNDS
    assert "val_roc_auc" in cached["evaluate"].data.metrics  # scalars copied from the manifest


def test_work0013_mlflow_params_flatten_the_step_config():
    p = flat_params({"model": {"kind": "lightgbm", "params": {"num_leaves": 15}}, "target": "y"})
    assert p == {"model.kind": "lightgbm", "model.params.num_leaves": "15", "target": "y"}
    assert flat_params({"x": "a" * 900})["x"] == "a" * 500


# ------------------------------------------------------------------ phase D


def test_work0013_show_maps_steps_to_store_files(titanic, tmp_path):
    cfg, store, manifests = titanic
    Pipeline(build_steps(), store, manifests, cfg).run()
    cli = CliRunner().invoke(app, ["show", "--pipeline", "titanic-example", "--json",
                                   "--store-dir", str(tmp_path / "store"),
                                   "--manifests-dir", str(manifests)])
    assert cli.exit_code == 0, cli.output
    records = json.loads(cli.output)
    assert [r["step"] for r in records] == [s.name for s in build_steps()]
    train = next(r for r in records if r["step"] == "train")
    hist = train["outputs"]["train_history"]
    assert hist["format"] == "parquet" and pl.read_parquet(hist["path"]).height > 0

    only = CliRunner().invoke(app, ["show", "--pipeline", "titanic-example", "--step", "evaluate",
                                    "--store-dir", str(tmp_path / "store"),
                                    "--manifests-dir", str(manifests)])
    assert only.exit_code == 0 and "eval_chart" in only.output and "train_history" not in only.output


def test_work0013_show_without_runs_is_an_error(tmp_path):
    cli = CliRunner().invoke(app, ["show", "--pipeline", "titanic-example",
                                   "--manifests-dir", str(tmp_path / "none")])
    assert cli.exit_code != 0 and "no completed manifests" in cli.output


def test_work0013_report_page_lists_its_sources(tmp_path):
    import plotly.graph_objects as go

    from mlpipe.report import table, write_report

    out = write_report(tmp_path / "r.html", "Test Report",
                       [{"heading": "h", "note": "n", "figure": go.Figure(go.Bar(x=[1], y=[2])),
                         "html": table([{"a": 1.23456}])}],
                       sources={"oof_preds": "ab" * 32})
    page = out.read_text()
    assert page.startswith("<title>Test Report</title>") and "ab" * 32 in page
    assert "<html" not in page and "toImage" in page  # fragment; PNG button removed
