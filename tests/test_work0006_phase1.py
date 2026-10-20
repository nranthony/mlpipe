"""work/0006 phase 1: era_slice smoke runs (step 2), --from resolving
upstream by current-config signature (step 3), and the top-level
PipelineConfig schema (step 5)."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError
from test_cycle0_spine import MakeData, Summarize, seed_artifact
from test_cycle3_clean import run_clean

from mlpipe.core.config import validate_config
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.clean import CleanConfig, CleanStep, EraSlice, slice_eras

ERAS = [f"e{i:02d}" for i in range(10)]


def era_frame() -> pl.LazyFrame:
    return pl.LazyFrame({"era": [e for e in ERAS for _ in range(3)], "x": list(range(30))})


@pytest.fixture()
def env(tmp_path):
    return LocalCasStore(tmp_path / "store"), tmp_path


# ---------------------------------------------------------------- era_slice


@pytest.mark.parametrize(
    ("mode", "n", "expected"),
    [("tail", 3, ERAS[-3:]), ("head", 2, ERAS[:2]), ("stride", 4, ERAS[::4])],
)
def test_era_slice_keeps_whole_eras(mode, n, expected):
    out = slice_eras(era_frame(), EraSlice(mode=mode, n=n)).collect()
    assert out["era"].unique(maintain_order=True).to_list() == expected
    assert out.group_by("era").len()["len"].to_list() == [3] * len(expected)


def test_era_slice_none_is_identity():
    assert slice_eras(era_frame(), None).collect().equals(era_frame().collect())


def test_era_slice_rejects_bad_config_before_running():
    with pytest.raises(ValidationError):
        CleanConfig.model_validate({"era_slice": {"mode": "tail", "n": 0}})
    with pytest.raises(ValidationError):
        CleanConfig.model_validate({"era_slice": {"mode": "random", "n": 3}})


def test_era_slice_changes_output_hash_and_records_meta(env):
    store, tmp = env
    cleaner = {"kind": "baseline", "feature_set": "all"}
    full = run_clean(store, tmp / "m1", {"clean": {"cleaner": cleaner}})
    cut = run_clean(
        store, tmp / "m2", {"clean": {"cleaner": cleaner, "era_slice": {"mode": "tail", "n": 1}}}
    )
    assert (
        full["artifacts"]["clean_table"].content_hash
        != cut["artifacts"]["clean_table"].content_hash
    )
    (m,) = cut["manifests"]
    assert m["meta"]["clean_table"]["eras"] == 1
    assert m["meta"]["clean_table"]["era_slice"] == {"mode": "tail", "n": 1}
    assert CleanStep.name == m["step"]


# ---------------------------------------------------------------- --from


def two_step(store, manifests, factor=1, rows=2):
    cfg = {"make_data": {"rows": rows}, "summarize": {"factor": factor}}
    return Pipeline([MakeData(), Summarize()], store, manifests, cfg)


def test_from_serves_upstream_cached_under_current_config(env):
    store, tmp = env
    seeds = [seed_artifact(store)]
    two_step(store, tmp / "m").run(seeds=seeds)
    result = two_step(store, tmp / "m", factor=3).run(from_step="summarize", seeds=seeds)
    statuses = {m["step"]: m["status"] for m in result["manifests"]}
    assert statuses == {"make_data": "cached", "summarize": "success"}


def test_from_refuses_upstream_produced_by_another_config(env):
    """The old trapdoor: --from served the latest raw_table whatever config made it."""
    store, tmp = env
    seeds = [seed_artifact(store)]
    two_step(store, tmp / "m", rows=2).run(seeds=seeds)
    with pytest.raises(RuntimeError, match="upstream step 'make_data' is not cached"):
        two_step(store, tmp / "m", rows=1).run(from_step="summarize", seeds=seeds)


def test_full_run_after_upstream_change_recomputes_only_what_changed(env):
    store, tmp = env
    seeds = [seed_artifact(store)]
    two_step(store, tmp / "m").run(seeds=seeds)
    result = two_step(store, tmp / "m", factor=5).run(seeds=seeds)
    assert [m["status"] for m in result["manifests"]] == ["cached", "success"]


# ---------------------------------------------------------------- PipelineConfig (step 5)


@pytest.mark.parametrize(
    "bad",
    [
        {"trakcer": {"kind": "mlflow"}},  # top-level typo
        {"seed": 42},  # dropped: a top-level seed can never reach a signature
        {"tracker": {"kind": "wandb"}},  # local only (DESIGN.md §0)
        {"tune": {"study": "s", "storage": "sqlite://", "space": {}}},  # no metric
    ],
)
def test_top_level_config_errors_die_before_any_step_runs(env, bad):
    store, tmp = env
    with pytest.raises(ValidationError):
        Pipeline(
            [MakeData(), Summarize()], store, tmp / "m", {"make_data": {"rows": 2}, **bad}
        ).run()
    assert not (tmp / "m" / "index.jsonl").exists()  # nothing ran, nothing written


def test_override_typo_at_top_level_dies_too(env):
    store, tmp = env
    with pytest.raises(ValidationError):
        two_step(store, tmp / "m").run(overrides=["trakcer.kind=mlflow"])


def test_tune_space_is_typed_and_direction_defaults_to_maximize():
    cfg = validate_config(
        {
            "tune": {
                "study": "s", "storage": "sqlite://", "metric": "rmse", "direction": "minimize",
                "space": {"a.b": {"type": "float", "low": 0.1, "high": 1.0, "log": True}},
            },
            "make_data": {"anything": "is the step's own business"},
        },
        ["make_data"],
    )
    assert cfg.tune.direction == "minimize" and cfg.tune.space["a.b"].log is True
    assert cfg.tracker.kind == "none"
    with pytest.raises(ValidationError):
        validate_config(
            {"tune": {"study": "s", "storage": "x", "metric": "m",
                      "space": {"a": {"type": "categorical"}}}}, []
        )


@pytest.mark.parametrize("name", ["demo", "numerai"])
def test_shipped_default_configs_validate(name):
    import importlib

    module = importlib.import_module(f"mlpipe.steps.{name}")
    validate_config(module.DEFAULT_CONFIG, [s.name for s in module.build_steps()])


# ---------------------------------------------------------------- hygiene (step 7)


def test_default_manifest_log_is_per_pipeline(tmp_path, monkeypatch):
    """work/0004 G: demo runs must not land in the real pipeline's history."""
    from typer.testing import CliRunner

    from mlpipe.cli import app

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["run", "--pipeline", "demo"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "manifests" / "demo" / "index.jsonl").exists()
    assert not (tmp_path / "manifests" / "index.jsonl").exists()
    assert (tmp_path / "store").is_dir()  # the store stays shared


# ---------------------------------------------------------------- manifest hardening (7b)


class Boom(Summarize):
    def run(self, ctx):
        raise ValueError("bad input shape (3, 0)")


def test_failed_manifest_carries_error_and_traceback(env):
    store, tmp = env
    pipe = Pipeline([MakeData(), Boom()], store, tmp / "m", {"make_data": {"rows": 2}})
    with pytest.raises(ValueError):
        pipe.run(seeds=[seed_artifact(store)])
    failed = [m for m in pipe.log.entries() if m["status"] == "failed"]
    assert failed[-1]["error"] == "ValueError: bad input shape (3, 0)"
    assert "raise ValueError" in failed[-1]["traceback"]


def test_environment_records_runtime_and_dirty_state(env):
    store, tmp = env
    (m, _) = two_step(store, tmp / "m").run(seeds=[seed_artifact(store)])["manifests"]
    envinfo = m["environment"]
    assert {"lockfile", "lockfile_hash", "python_version", "platform", "git_dirty"} <= set(envinfo)
    assert "seed" not in envinfo  # dropped in step 5


def _append_many(root, n):
    from mlpipe.core.manifest import ManifestLog

    log = ManifestLog(root)
    for i in range(n):  # records larger than a pipe/page buffer, to provoke tearing
        log.write({"step": "s", "signature": f"{i}", "status": "success", "pad": "x" * 70_000})


def test_concurrent_index_appends_never_interleave(tmp_path):
    import json
    import multiprocessing as mp

    procs = [mp.get_context("fork").Process(target=_append_many, args=(tmp_path, 25))
             for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    lines = (tmp_path / "index.jsonl").read_text().splitlines()
    assert len(lines) == 100
    assert all(json.loads(line)["pad"] == "x" * 70_000 for line in lines)


# ---------------------------------------------------------------- Numerai v6.0 shape (step 8)

V6_META = {
    "feature_sets": {
        "medium": ["feature_a", "feature_country"],
        "all": ["feature_a", "feature_country", "feature_f"],
    },
    "targets": ["target"],
}
V6_VALIDATE = {"era_column": "date", "feature_dtypes": ["Int8", "Float32", "String"]}
V6_CLEAN = {
    "cleaner": {"kind": "baseline", "feature_set": "medium"},
    "era_column": "date",
    "exclude_features": ["feature_country"],
}


def v6_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "numerai_ticker": ["A", "B", "A", "B"],
        "date": ["2019-01-04", "2019-01-04", "2019-01-11", "2019-01-11"],
        "feature_a": pl.Series([1, 2, 3, 4], dtype=pl.Int8),
        "feature_country": ["KR", "US", "KR", "US"],
        "feature_f": pl.Series([0.1, 0.2, 0.3, 0.4], dtype=pl.Float32),
        "target": pl.Series([0.25, 0.5, None, 1.0], dtype=pl.Float32),
    })


def test_v6_validate_accepts_date_era_and_mixed_feature_dtypes(env):
    from mlpipe.steps.validate import ValidateStep

    store, tmp = env
    seeds = [store.save(v6_frame(), k, ext="parquet") for k in ("raw_train", "raw_validation", "raw_live")]
    seeds.append(store.save(V6_META, "features_meta", ext="json"))
    pipe = Pipeline([ValidateStep()], store, tmp / "m", {"validate": V6_VALIDATE})
    (m,) = pipe.run(seeds=seeds)["manifests"]
    assert m["status"] == "success"
    # the v5 defaults (era column `era`, Int8 only) reject the same bytes, loudly
    with pytest.raises(ValueError, match="'era' must be a non-null String"):
        Pipeline([ValidateStep()], store, tmp / "m2", {}).run(seeds=seeds)
    with pytest.raises(ValueError, match="features must be one of"):
        Pipeline([ValidateStep()], store, tmp / "m3", {"validate": {"era_column": "date"}}).run(
            seeds=seeds
        )


def test_v6_clean_renames_era_and_drops_excluded_features(env):
    store, tmp = env
    seeds = [
        store.save(v6_frame(), "validated_train", ext="parquet"),
        store.save(v6_frame(), "validated_validation", ext="parquet"),
        store.save(V6_META, "features_meta", ext="json"),
    ]
    result = Pipeline([CleanStep()], store, tmp / "m", {"clean": V6_CLEAN}).run(seeds=seeds)
    out = store.load(result["artifacts"]["clean_table"].content_hash)
    assert out.columns == ["era", "feature_a", "target"]  # canonical era; country gone
    assert out.height == 3  # null-target row dropped
    bad = {"clean": {**V6_CLEAN, "exclude_features": ["feature_typo"]}}
    with pytest.raises(ValueError, match="feature_typo"):
        Pipeline([CleanStep()], store, tmp / "m2", bad).run(seeds=seeds)


def test_v6_null_bound_is_per_dtype(env):
    """Sparse Float32 factors pass under their own bound; an Int8 null still fails."""
    from mlpipe.steps.validate import ValidateStep

    store, tmp = env
    sparse = v6_frame().with_columns(pl.Series("feature_f", [None, 0.2, 0.3, 0.4], pl.Float32))
    cfg = {"validate": {**V6_VALIDATE, "max_feature_null_frac_by_dtype": {"Float32": 0.3}}}

    def run(frame, manifests):
        seeds = [store.save(frame, k, ext="parquet") for k in ("raw_train", "raw_validation", "raw_live")]
        seeds.append(store.save(V6_META, "features_meta", ext="json"))
        return Pipeline([ValidateStep()], store, tmp / manifests, cfg).run(seeds=seeds)

    report = store.load(run(sparse, "m1")["manifests"][0]["outputs"]["validation_report"])
    assert report["raw_train"]["max_feature_null_frac_by_dtype"]["Float32"] == 0.25
    holed = sparse.with_columns(pl.Series("feature_a", [None, 2, 3, 4], pl.Int8))
    with pytest.raises(ValueError, match="worst Int8 feature null fraction 0.2500"):
        run(holed, "m2")
