"""work/0010 — store, registry and Numerai-example hygiene."""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
import yaml
from test_cycle7_evaluate import env, oof_frame  # noqa: F401  (fixture)
from typer.testing import CliRunner

from mlpipe import baselines
from mlpipe.cli import app
from mlpipe.core.manifest import ManifestLog
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.signature import code_modules, sha256_bytes, step_signature
from mlpipe.core.store import CHUNK, LocalCasStore
from mlpipe.examples.numerai.eval_metrics import numerai_corr_scores
from mlpipe.examples.numerai.evaluate import EvalConfig, EvaluateStep
from mlpipe.examples.numerai.validate import ValidateConfig, check_table
from mlpipe.gc import collect
from mlpipe.steps.cvplan import CVConfig, build_plan, era_order

runner = CliRunner()
NUMERAI_CORR = "mlpipe.examples.numerai.eval_metrics:numerai_corr_scores"


# ------------------------------------------------------------ 1. streamed snapshots


def test_work0010_path_snapshot_streams_and_hashes_like_the_bytes(tmp_path):
    src = tmp_path / "download.parquet"
    data = np.random.default_rng(0).bytes(2 * CHUNK + 123)  # spans several chunks
    src.write_bytes(data)
    store = LocalCasStore(tmp_path / "store")
    art = store.save(src, "raw_train", ext="parquet")
    assert art.content_hash == sha256_bytes(data)
    assert art.path.read_bytes() == data
    assert art.path.stat().st_ino != src.stat().st_ino  # copied, never linked
    again = store.save(src, "raw_train", ext="parquet")  # dedup: same file, no temp left
    assert again.path == art.path
    assert [p for p in store.root.iterdir() if p.is_file()] == []


# ------------------------------------------------------- 2. baselines per pipeline/alias


def test_work0010_baselines_slots_do_not_overwrite_each_other(tmp_path):
    path = tmp_path / "baselines.yaml"
    path.write_text("promoted:\n  mean_corr: 0.02\n")  # the pre-ADR-0006 block
    baselines.write_pin(path, "numerai", "champion", {"bundle_hash": "a" * 64})
    baselines.write_pin(path, "numerai", "staging", {"bundle_hash": "b" * 64})
    baselines.write_pin(path, "demo", "champion", {"bundle_hash": "c" * 64})
    data = yaml.safe_load(path.read_text())
    assert data["promoted"] == {"mean_corr": 0.02}  # kept as written, pins nothing
    assert baselines.pinned(path, "numerai", "champion")["bundle_hash"] == "a" * 64
    assert set(baselines.pins(data)) == {
        ("numerai", "champion"), ("numerai", "staging"), ("demo", "champion")
    }


def _bundle_log(tmp_path):
    store, log = LocalCasStore(tmp_path / "store"), ManifestLog(tmp_path / "m")
    hashes = []
    for name in ("one", "two"):
        bundle = store.save({"model_name": name, "metrics": {}}, "model_bundle", ext="json")
        log.write({"step": "register", "status": "success", "signature": name, "inputs": {},
                   "outputs": {"model_bundle": bundle.content_hash}, "metrics": {}, "meta": {},
                   "config": {}, "git_commit": None})
        hashes.append(bundle.content_hash)
    return store, log, hashes


def test_work0010_promote_writes_its_pipeline_and_alias_slot(tmp_path):
    _, _, (one, two) = _bundle_log(tmp_path)
    path = tmp_path / "baselines.yaml"
    common = ["--pipeline", "demo", "--store-dir", str(tmp_path / "store"),
              "--manifests-dir", str(tmp_path / "m"), "--baselines", str(path)]
    for bundle, alias in ((one, "champion"), (two, "staging")):
        result = runner.invoke(app, ["promote", bundle[:10], "--alias", alias, *common])
        assert result.exit_code == 0, result.output
    assert baselines.pinned(path, "demo", "champion")["bundle_hash"] == one  # not overwritten
    assert baselines.pinned(path, "demo", "staging")["bundle_hash"] == two


def test_work0010_predict_names_the_missing_slot(tmp_path):
    path = tmp_path / "baselines.yaml"
    baselines.write_pin(path, "numerai", "champion", {"bundle_hash": "a" * 64})
    result = runner.invoke(app, ["predict", "--pipeline", "demo", "--baselines", str(path),
                                 "--manifests-dir", str(tmp_path / "m")])
    assert result.exit_code != 0 and "demo.champion" in result.output


# --------------------------------------------------------------- 4. gc roots


def test_work0010_gc_roots_every_pin_and_root_files(tmp_path):
    store, _, (one, two) = _bundle_log(tmp_path)
    stray = store.save({"orphan": True}, "x", ext="json").content_hash
    path = tmp_path / "baselines.yaml"
    baselines.write_pin(path, "demo", "staging", {"bundle_hash": one})
    host_record = tmp_path / "champion.json"  # a host's own promotion record, in git
    host_record.write_text(f'{{"bundle": "{two}"}}')
    later = datetime(2100, 1, 1, tzinfo=UTC)

    r = collect(store.root, tmp_path, path, keep_days=0, now=later, root_files=[host_record])
    assert r.bundles == {one, two}
    assert r.unreachable == [stray]

    empty = tmp_path / "empty.txt"
    empty.write_text("nothing here")
    r = collect(store.root, tmp_path, path, keep_days=0, now=later, root_files=[empty])
    assert any("names no 64-hex hash" in n for n in r.notes)


def test_work0010_gc_cli_takes_root_files(tmp_path):
    store, _, (one, _) = _bundle_log(tmp_path)
    record = tmp_path / "keep.txt"
    record.write_text(one + "\n")
    result = runner.invoke(app, ["gc", "--dry-run", "--keep-days", "0", "--store-dir",
                                 str(store.root), "--manifests-root", str(tmp_path),
                                 "--baselines", str(tmp_path / "none.yaml"),
                                 "--root-file", str(record)])
    assert result.exit_code == 0, result.output
    assert one[:8] in result.output


# ------------------------------------------------- 5. validate report is content-only


def test_work0010_validation_report_does_not_carry_history():
    lf = pl.LazyFrame({"era": ["a", "a", "b"], "feature_x": pl.Series([1, 2, 3], dtype=pl.Int8)})
    first, _ = check_table("raw_train", lf, ["feature_x"], ValidateConfig(), None)
    second, _ = check_table("raw_train", lf, ["feature_x"], ValidateConfig(), 3)
    assert first == second and "previous_rows" not in first


# ------------------------------------------------------------- 6. era time order


def test_work0010_cvplan_folds_follow_time_not_row_order():
    rows = [f"{d:04d}" for d in range(12) for _ in range(3)]
    ordered = pl.Series("era", rows)
    shuffled = ordered.sample(fraction=1.0, shuffle=True, seed=3)
    cfg = CVConfig(n_folds=3, purge_eras=1, embargo_eras=0)
    a, b = build_plan(ordered, cfg), build_plan(shuffled, cfg)
    assert a.plan_hash == b.plan_hash
    for _, valid in b.folds:  # each validation block is a contiguous run of eras
        eras = sorted({int(e) for e in shuffled.gather(valid).to_list()})
        assert eras == list(range(eras[0], eras[-1] + 1))


def test_work0010_era_order_handles_unpadded_ids_and_dates():
    assert era_order(["10", "2", "1"]) == ["1", "2", "10"]
    assert era_order(["2026-01-09", "2025-12-26", "2026-01-02"]) == [
        "2025-12-26", "2026-01-02", "2026-01-09"
    ]
    assert era_order([3, 1, 2]) == [1, 2, 3]


# ------------------------------------------------------------ 7. scorer by path


def test_work0010_numerai_corr_matches_an_independent_reference():
    stats = pytest.importorskip("scipy.stats")
    df = oof_frame().with_columns(pl.col("prediction").round(1))  # force ties
    ours = dict(numerai_corr_scores(df).iter_rows())
    for era, group in df.group_by("era"):
        p = group["prediction"].to_numpy().astype(float)
        t = group["target"].to_numpy().astype(float)
        g = stats.norm.ppf((stats.rankdata(p, method="average") - 0.5) / len(p))
        tc = t - t.mean()
        ref = np.corrcoef(np.sign(g) * np.abs(g) ** 1.5, np.sign(tc) * np.abs(tc) ** 1.5)[0, 1]
        assert ours[era[0]] == pytest.approx(ref, abs=1e-6)


def test_work0010_scorer_config_changes_metrics_and_signature(env):  # noqa: F811
    store, tmp, seeds, _ = env
    ranked = Pipeline([EvaluateStep()], store, tmp / "a", {}).run(seeds=seeds)
    config = {"evaluate": {"scorer": NUMERAI_CORR}}
    gauss = Pipeline([EvaluateStep()], store, tmp / "b", config).run(seeds=seeds)
    (a,), (b,) = ranked["manifests"], gauss["manifests"]
    assert a["signature"] != b["signature"]
    assert a["metrics"]["mean_corr"] != b["metrics"]["mean_corr"]
    assert b["meta"]["scorer"] == NUMERAI_CORR
    with pytest.raises(ValueError, match="module:function"):
        EvalConfig(scorer="no_colon_here")


def test_work0010_a_host_scorer_edit_reruns_evaluate(tmp_path, monkeypatch):
    pkg = tmp_path / "scorer10"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "metric.py").write_text("from scorer10.util import k\n\ndef score(df, **kw):\n    return k\n")
    (pkg / "util.py").write_text("k = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.startswith("scorer10")]:
        monkeypatch.delitem(sys.modules, name)
    config = {"evaluate": {"scorer": "scorer10.metric:score"}}
    assert {"scorer10.metric", "scorer10.util"} <= set(code_modules(EvaluateStep(), config))
    before = step_signature(EvaluateStep(), {}, config)
    (pkg / "util.py").write_text("k = 2\n")
    assert step_signature(EvaluateStep(), {}, config) != before


# ------------------------------------------------- 8. validation boundary eras


def test_work0010_validation_skips_the_eras_whose_labels_overlap_training(env):  # noqa: F811
    store, tmp, seeds, _ = env
    plan = build_plan(oof_frame()["era"], CVConfig(n_folds=2, purge_eras=2, embargo_eras=0))
    seeds = [*seeds[:2], store.save(plan, "fold_plan", ext="joblib"), *seeds[3:]]
    result = Pipeline([EvaluateStep()], store, tmp / "m", {}).run(seeds=seeds)
    (m,) = result["manifests"]
    assert m["metrics"]["val_n_eras"] == 3  # v000-v005, minus 2 skipped and 1 unresolved
    report = store.load(result["artifacts"]["eval_report"].content_hash)
    assert report["val_skipped_eras"] == ["v000", "v001"]
    assert set(report["val_era_scores"]) == {"v002", "v003", "v004"}
