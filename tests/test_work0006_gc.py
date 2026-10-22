"""work/0006 step 16 — `mlpipe gc --dry-run` reports unreachable store files."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
from typer.testing import CliRunner

from mlpipe.cli import app
from mlpipe.core.manifest import ManifestLog
from mlpipe.core.store import LocalCasStore
from mlpipe.gc import collect

runner = CliRunner()
OLD = "2020-01-01T00:00:00+00:00"


def _record(step, inputs, outputs, status="success"):
    return {"step": step, "status": status, "signature": step, "inputs": inputs,
            "outputs": outputs, "started_at": OLD, "ended_at": OLD}


def test_work0006_gc_keeps_promoted_lineage_and_what_a_bundle_names(tmp_path):
    store = LocalCasStore(tmp_path / "store")
    log = ManifestLog(tmp_path / "manifests" / "toy")
    raw = store.save(pl.DataFrame({"x": [1, 2]}), "raw")
    model = store.save({"weights": [1]}, "model", ext="json")
    side = store.save({"named only by the bundle": True}, "side", ext="json")
    stale = store.save(pl.DataFrame({"x": [3]}), "raw")  # an old run's output
    plan_hash = "e" * 64  # 64 hex but no store file: must not count as missing
    bundle = store.save(
        {"model_hash": model.content_hash, "side": [side.content_hash], "plan_hash": plan_hash},
        "model_bundle", ext="json",
    )
    log.write(_record("make", {}, {"raw": stale.content_hash}))
    log.write(_record("make", {}, {"raw": raw.content_hash}))
    log.write(_record("train", {"raw": raw.content_hash}, {"model": model.content_hash}))
    log.write(_record("register", {"model": model.content_hash},
                      {"model_bundle": bundle.content_hash}))
    log.write(_record("promote", {"model_bundle": bundle.content_hash}, {}, status="promoted"))
    tmp = store.root / "ab" / ".abc.123.tmp"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text("half-written")
    (tmp_path / "baselines.yaml").write_text("promoted:\n  mean_corr: 0.02\n")

    r = collect(store.root, tmp_path / "manifests", tmp_path / "baselines.yaml", keep_days=30)

    assert r.bundles == {bundle.content_hash}
    assert r.unreachable == [stale.content_hash]
    assert {a.content_hash for a in (raw, model, side, bundle)} <= r.kept
    assert r.other_files == [tmp] and r.missing == set()
    assert r.last_written[stale.content_hash] == "make 2020-01-01"
    assert any("before ADR-0006" in n for n in r.notes)


def test_work0006_gc_retention_window_keeps_recent_manifests(tmp_path):
    store = LocalCasStore(tmp_path / "store")
    log = ManifestLog(tmp_path / "manifests" / "toy")
    art = store.save(pl.DataFrame({"x": [1]}), "raw")
    log.write(_record("make", {}, {"raw": art.content_hash}))
    after = datetime(2020, 1, 20, tzinfo=UTC)
    baselines = tmp_path / "none.yaml"

    assert collect(store.root, log.root.parent, baselines, 30, now=after).unreachable == []
    assert collect(store.root, log.root.parent, baselines, 10, now=after).unreachable == [
        art.content_hash
    ]


def test_work0006_gc_cli_reports_and_never_deletes(tmp_path):
    dirs = ["--store-dir", str(tmp_path / "store")]
    run = ["run", "--pipeline", "demo", *dirs, "--manifests-dir", str(tmp_path / "m" / "demo")]
    assert runner.invoke(app, run).exit_code == 0
    assert runner.invoke(app, [*run, "--set", "summarize.factor=3"]).exit_code == 0
    gc = ["gc", *dirs, "--manifests-root", str(tmp_path / "m"),
          "--baselines", str(tmp_path / "baselines.yaml")]
    before = sorted((tmp_path / "store").rglob("*"))

    refused = runner.invoke(app, gc)
    assert refused.exit_code != 0 and "--dry-run" in refused.output

    recent = runner.invoke(app, [*gc, "--dry-run"])
    assert recent.exit_code == 0, recent.output
    assert "store:       3 artifacts" in recent.output
    assert "unreachable: 0 artifacts" in recent.output

    aged = runner.invoke(app, [*gc, "--dry-run", "--keep-days", "0"])
    assert aged.exit_code == 0, aged.output
    assert "unreachable: 3 artifacts" in aged.output and "nothing deleted" in aged.output
    assert sorted((tmp_path / "store").rglob("*")) == before
