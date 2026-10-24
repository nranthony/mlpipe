"""work/0009 part B — the Numerai example's snapshot ingest and serve steps
(ADR-0007). Toy data; every numerapi call is monkeypatched, no network."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml
from pydantic import ValidationError
from test_cycle9_10_tune_register import (
    LIVE_ROUND,
    SNAPSHOT,
    bundle_hash_of,
    promote,
    toy_snapshot,
)
from typer.testing import CliRunner

from mlpipe.cli import app
from mlpipe.core.manifest import ManifestLog
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.examples.numerai import serve
from mlpipe.examples.numerai.ingest import IngestConfig, IngestStep, fetch
from mlpipe.examples.numerai.serve import IngestLiveStep, ValidateLiveStep

runner = CliRunner()
OPEN = datetime(2026, 1, 12, 14, 0, tzinfo=UTC)


# ------------------------------------------------------------------ helpers


def live_frame(dates, rows: int | None = None) -> pl.DataFrame:
    n = rows or len(dates)
    return pl.DataFrame({
        "era": dates if rows is None else (list(dates) * n)[:n],
        "feature_fa": pl.Series([i % 5 for i in range(n)], dtype=pl.Int8),
        "feature_fb": pl.Series([(i + 1) % 5 for i in range(n)], dtype=pl.Int8),
    })


def fake_numerapi(monkeypatch, writer):
    """Replace numerapi.NumerAPI; `writer(dest: Path)` plays the download."""
    calls = []

    class Fake:
        def download_dataset(self, filename, dest_path=None, round_num=None):
            calls.append({"filename": filename, "dest": Path(dest_path), "round_num": round_num})
            writer(Path(dest_path))

    monkeypatch.setattr("numerapi.NumerAPI", Fake)
    return calls


def forbid_numerapi(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network touched")

    monkeypatch.setattr("numerapi.NumerAPI", boom)


def live_cfg(tmp_path, round_=7, **extra):
    return {
        "download_dir": str(tmp_path / "raw"),
        "ingest_live": {"version": "v0-toy", "era_column": "era", "round": round_, **extra},
    }


def run_live(tmp_path, cfg):
    return Pipeline([IngestLiveStep()], LocalCasStore(tmp_path / "store"), tmp_path / "m", cfg).run()


def live_path(tmp_path, round_=7) -> Path:
    return tmp_path / "raw" / "v0-toy" / "live" / str(round_) / "live.parquet"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ B1


def test_work0009_snapshot_is_required_and_a_date():
    with pytest.raises(ValidationError, match="snapshot"):
        IngestConfig.model_validate({"version": "v6.0"})
    for bad in ("2026-13-01", "20261020", "latest", "2026-1-5"):
        with pytest.raises(ValidationError, match="snapshot"):
            IngestConfig.model_validate({"snapshot": bad})
    assert IngestConfig.model_validate({"snapshot": "2026-10-20"}).version == "v6.0"


def test_work0009_ingest_without_snapshot_fails_validation(tmp_path):
    cfg = {"download_dir": str(tmp_path), "ingest": {"version": "v0-toy"}}
    with pytest.raises(ValueError, match="snapshot"):
        Pipeline([IngestStep()], LocalCasStore(tmp_path / "s"), tmp_path / "m", cfg).run()


def test_work0009_fetch_reads_the_snapshot_folder(tmp_path, monkeypatch):
    forbid_numerapi(monkeypatch)
    folder = tmp_path / "v6.0" / "2026-10-20"
    folder.mkdir(parents=True)
    (folder / "train.parquet").write_bytes(b"x")
    assert fetch("v6.0", "2026-10-20", "train.parquet", tmp_path) == folder / "train.parquet"


def test_work0009_missing_past_snapshot_raises_without_network(tmp_path, monkeypatch):
    forbid_numerapi(monkeypatch)
    with pytest.raises(FileNotFoundError, match="2026-10-20.*not today"):
        fetch("v6.0", "2026-10-20", "train.parquet", tmp_path)
    assert not (tmp_path / "v6.0").exists()  # nothing created


def test_work0009_todays_snapshot_downloads_to_the_final_path(tmp_path, monkeypatch):
    today = datetime.now(UTC).date().isoformat()
    calls = fake_numerapi(monkeypatch, lambda dest: dest.write_bytes(b"data"))
    path = fetch("v6.0", today, "train.parquet", tmp_path)
    assert path == tmp_path / "v6.0" / today / "train.parquet" and path.read_bytes() == b"data"
    assert calls == [{"filename": "v6.0/train.parquet", "dest": path, "round_num": None}]
    forbid_numerapi(monkeypatch)
    assert fetch("v6.0", today, "train.parquet", tmp_path) == path  # now present: no call


# ------------------------------------------------------------------ B3 / B4: ingest_live


def test_work0009_two_rounds_two_paths_two_signatures(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(serve, "round_open_time", lambda r: asked.append(r) or OPEN)
    forbid_numerapi(monkeypatch)
    for r in (7, 8):
        live_path(tmp_path, r).parent.mkdir(parents=True)
        live_frame(["2026-01-09"], rows=3 + r).write_parquet(live_path(tmp_path, r))
    (m7,) = run_live(tmp_path, live_cfg(tmp_path, 7))["manifests"]
    (m8,) = run_live(tmp_path, live_cfg(tmp_path, 8))["manifests"]
    assert m7["status"] == "success" and m8["status"] == "success"  # round 8: a cache miss
    assert m7["signature"] != m8["signature"]
    assert m7["meta"]["raw_live"]["file"] == "v0-toy/live/7/live.parquet"
    assert m8["meta"]["raw_live"]["file"] == "v0-toy/live/8/live.parquet"
    meta = m7["meta"]["raw_live"]
    assert meta["round"] == 7 and meta["live_date"] == "2026-01-09"
    assert meta["open_time"] == OPEN.isoformat() and meta["rows"] == 10

    n = len(asked)
    (again,) = run_live(tmp_path, live_cfg(tmp_path, 7))["manifests"]
    assert again["status"] == "cached" and len(asked) == n  # no open-time lookup on a hit


def test_work0009_absent_round_file_is_downloaded_with_round_num(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "round_open_time", lambda r: OPEN)
    calls = fake_numerapi(monkeypatch, lambda dest: live_frame(["2026-01-09"] * 3).write_parquet(dest))
    result = run_live(tmp_path, live_cfg(tmp_path, 7))
    final = live_path(tmp_path)
    assert final.exists() and not list(final.parent.glob("*.part*"))
    assert calls == [{"filename": "v0-toy/live.parquet", "dest": Path(f"{final}.part"), "round_num": 7}]
    assert result["manifests"][0]["status"] == "success"


@pytest.mark.parametrize(
    ("dates", "error"),
    [
        (["2026-01-04"] * 3, "8 days before"),
        (["2026-01-13"] * 3, "after round open date"),
        (["2026-01-09", "2026-01-10"], "expected one"),
        (["not-a-date"] * 3, "not an ISO date"),
    ],
)
def test_work0009_freshness_failures_leave_nothing_behind(tmp_path, monkeypatch, dates, error):
    monkeypatch.setattr(serve, "round_open_time", lambda r: OPEN)

    def write(dest):
        live_frame(dates).write_parquet(dest)
        Path(f"{dest}.temp").write_bytes(b"partial")

    fake_numerapi(monkeypatch, write)
    with pytest.raises(ValueError, match=error):
        run_live(tmp_path, live_cfg(tmp_path, 7))
    assert list(live_path(tmp_path).parent.iterdir()) == []  # no final, no .part, no .part.temp


def test_work0009_freshness_six_days_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "round_open_time", lambda r: OPEN)
    fake_numerapi(monkeypatch, lambda dest: live_frame(["2026-01-06"] * 3).write_parquet(dest))
    (m,) = run_live(tmp_path, live_cfg(tmp_path, 7))["manifests"]
    assert m["status"] == "success" and live_path(tmp_path).exists()


def test_work0009_max_age_days_is_configurable(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "round_open_time", lambda r: OPEN)
    fake_numerapi(monkeypatch, lambda dest: live_frame(["2026-01-04"] * 3).write_parquet(dest))
    (m,) = run_live(tmp_path, live_cfg(tmp_path, 7, max_age_days=8))["manifests"]
    assert m["status"] == "success"


def test_work0009_failed_existing_round_file_is_named_and_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "round_open_time", lambda r: OPEN)
    forbid_numerapi(monkeypatch)
    final = live_path(tmp_path)
    final.parent.mkdir(parents=True)
    live_frame(["2026-01-13"] * 3).write_parquet(final)
    before = sha(final)
    with pytest.raises(ValueError, match="live.parquet"):
        run_live(tmp_path, live_cfg(tmp_path, 7))
    assert final.exists() and sha(final) == before  # removing it is a human step


def test_work0009_a_raising_downloader_leaves_no_file_or_part(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "round_open_time", lambda r: OPEN)

    def write(dest):
        Path(f"{dest}.temp").write_bytes(b"half")
        dest.write_bytes(b"half")
        raise ConnectionError("dropped")

    fake_numerapi(monkeypatch, write)
    with pytest.raises(ConnectionError):
        run_live(tmp_path, live_cfg(tmp_path, 7))
    assert list(live_path(tmp_path).parent.iterdir()) == []


def test_work0009_ingest_live_without_round_fails_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "round_open_time", lambda r: OPEN)
    cfg = {"download_dir": str(tmp_path / "raw"), "ingest_live": {"version": "v0-toy"}}
    with pytest.raises(ValueError, match="round"):
        run_live(tmp_path, cfg)


# ------------------------------------------------------------------ B3: validate_live


def run_validate_live(tmp_path, frame, bundle, cfg=None):
    store = LocalCasStore(tmp_path / "store")
    seeds = [store.save(frame, "raw_live", ext="parquet"), store.save(bundle, "model_bundle", ext="json")]
    pipe = Pipeline([ValidateLiveStep()], store, tmp_path / "m",
                    {"validate_live": {"era_column": "era", **(cfg or {})}})
    return store, pipe.run(seeds=seeds)


BUNDLE = {"features": ["feature_fa", "feature_fb"]}


def test_work0009_validate_live_passes_and_aliases(tmp_path):
    frame = live_frame(["2026-01-09"] * 4)
    store, result = run_validate_live(tmp_path, frame, BUNDLE)
    (m,) = result["manifests"]
    assert m["status"] == "success" and m["outputs"]["validated_live"] == m["inputs"]["raw_live"]
    assert store.load(m["outputs"]["live_validation_report"])["raw_live"]["passed"] is True


def test_work0009_validate_live_checks_the_bundles_features(tmp_path):
    bundle = {"features": ["feature_fa", "feature_zz"]}
    with pytest.raises(ValueError, match="raw_live.*feature_zz"):
        run_validate_live(tmp_path, live_frame(["2026-01-09"] * 4), bundle)


def test_work0009_validate_live_null_bound(tmp_path):
    holed = live_frame(["2026-01-09"] * 4).with_columns(
        pl.Series("feature_fa", [None, 1, 2, 3], dtype=pl.Int8)
    )
    with pytest.raises(ValueError, match="raw_live.*null"):
        run_validate_live(tmp_path, holed, BUNDLE)
    _, ok = run_validate_live(tmp_path, holed, BUNDLE, {"max_feature_null_frac": 0.5})
    assert ok["manifests"][0]["status"] == "success"


# ------------------------------------------------------------------ B5 / B6: CLI end to end


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("serve_e2e")
    toy_snapshot(tmp)
    v = tmp / "raw" / "v0-toy"
    for name in ("train.parquet", "validation.parquet", "features.json"):  # legacy flat files
        (v / name).write_bytes((v / SNAPSHOT / name).read_bytes())
    flat = {p: (sha(p), p.stat().st_mtime_ns) for p in sorted(v.glob("*.*"))}
    config = {
        "download_dir": str(tmp / "raw"),
        "ingest": {"version": "v0-toy", "snapshot": SNAPSHOT},
        "ingest_live": {"version": "v0-toy", "era_column": "era"},
        "validate": {"row_tolerance": 0.5},
        "validate_live": {"era_column": "era"},
        "clean": {"cleaner": {"kind": "baseline", "feature_set": "all"}},
        "features": {"transformer": {"kind": "passthrough"}},
        "cvplan": {"n_folds": 2, "purge_eras": 0, "embargo_eras": 0},
        "train": {"model": {"kind": "lightgbm", "params": {"n_estimators": 5}}},
        "evaluate": {"target": "target"},
        "register": {"model_name": "toy-model"},
        "tracker": {"kind": "mlflow", "uri": f"sqlite:///{tmp}/mlflow.db", "experiment": "toy"},
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    dirs = ["--config", str(cfg_path), "--store-dir", str(tmp / "store"),
            "--manifests-dir", str(tmp / "manifests"), "--pipeline", "numerai"]
    result = runner.invoke(app, ["run", *dirs])
    assert result.exit_code == 0, result.output
    return tmp, dirs, flat


def steps_of(tmp):
    return [m["step"] for m in ManifestLog(tmp / "manifests").entries()]


def test_work0009_predict_runs_the_three_serve_steps(env, monkeypatch):
    tmp, dirs, _ = env
    monkeypatch.setattr(serve, "round_open_time", lambda r: datetime(2026, 1, 5, tzinfo=UTC))
    forbid_numerapi(monkeypatch)  # the round file is already in the folder
    promote(tmp, dirs, bundle_hash_of(tmp))
    before = len(steps_of(tmp))
    out = tmp / "round.parquet"
    result = runner.invoke(app, ["predict", *dirs, "--set", f"ingest_live.round={LIVE_ROUND}",
                                 "--baselines", str(tmp / "baselines.yaml"), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert steps_of(tmp)[before:] == ["ingest_live", "validate_live", "predict"]
    preds = pl.read_parquet(out)
    assert preds.columns == ["era", "prediction"] and preds.height == 120  # toy live: no ticker


def test_work0009_predict_without_a_round_is_a_usage_error(env):
    tmp, dirs, _ = env
    promote(tmp, dirs, bundle_hash_of(tmp))
    result = runner.invoke(app, ["predict", *dirs, "--baselines", str(tmp / "baselines.yaml"),
                                 "--out", str(tmp / "none.parquet")])
    assert result.exit_code == 2 and "round" in result.output
    assert not (tmp / "none.parquet").exists()


def test_work0009_predict_input_skips_ingest_live_but_validates(env, monkeypatch):
    tmp, dirs, _ = env
    forbid_numerapi(monkeypatch)
    monkeypatch.setattr(serve, "round_open_time", lambda r: pytest.fail("ingest_live must not run"))
    promote(tmp, dirs, bundle_hash_of(tmp))
    src = tmp / "raw" / "v0-toy" / "live" / str(LIVE_ROUND) / "live.parquet"
    scored = pl.read_parquet(src).head(9).with_columns(
        pl.Series("numerai_ticker", [f"T{i}" for i in range(9)])
    )
    file = tmp / "nine.parquet"
    scored.write_parquet(file)
    before = steps_of(tmp)
    out = tmp / "nine_out.parquet"
    result = runner.invoke(app, ["predict", *dirs, "--input", str(file),
                                 "--baselines", str(tmp / "baselines.yaml"), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert steps_of(tmp)[len(before):] == ["input", "validate_live", "predict"]
    assert pl.read_parquet(out).columns == ["numerai_ticker", "era", "prediction"]


def test_work0009_validate_live_gates_an_input_file(env):
    tmp, dirs, _ = env
    promote(tmp, dirs, bundle_hash_of(tmp))
    src = tmp / "raw" / "v0-toy" / "live" / str(LIVE_ROUND) / "live.parquet"
    bad = tmp / "bad.parquet"
    pl.read_parquet(src).drop("feature_fb").write_parquet(bad)
    result = runner.invoke(app, ["predict", *dirs, "--input", str(bad),
                                 "--baselines", str(tmp / "baselines.yaml"),
                                 "--out", str(tmp / "bad_out.parquet")])
    assert result.exit_code != 0 and not (tmp / "bad_out.parquet").exists()


def test_work0009_flat_download_files_are_untouched(env):
    _, _, flat = env  # recorded before the run; predict tests above ran since
    assert flat
    for path, (digest, mtime) in flat.items():
        assert sha(path) == digest and path.stat().st_mtime_ns == mtime
