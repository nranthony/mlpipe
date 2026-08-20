"""Cycle 0 acceptance tests — the provenance spine (goals/cycle0_spine.md)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

from mlpipe.core.interfaces import Step, StepResult
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore


def make_frame(offset: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "era": ["e1", "e1", "e2", "e2"],
            "x": pl.Series([1 + offset, 2, 3, 4], dtype=pl.Int8),
        }
    )


class MakeData(Step):
    name = "make_data"
    inputs = ["seed"]
    outputs = ["raw_table"]
    calls = 0

    def run(self, ctx):
        type(self).calls += 1
        df = ctx.get("seed")
        rows = ctx.config.get(self.name, {}).get("rows", 2)
        art = ctx.put(df.head(rows), "raw_table")
        return StepResult(outputs={"raw_table": art})


class Summarize(Step):
    name = "summarize"
    inputs = ["raw_table"]
    outputs = ["summary"]
    calls = 0

    def run(self, ctx):
        type(self).calls += 1
        df = ctx.get("raw_table")
        factor = ctx.config.get(self.name, {}).get("factor", 1)
        total = int(df["x"].sum()) * factor
        art = ctx.put({"total": total}, "summary", ext="json")
        ctx.log_metric("total", float(total))
        return StepResult(outputs={"summary": art})


@pytest.fixture()
def dirs(tmp_path):
    return LocalCasStore(tmp_path / "store"), tmp_path / "manifests"


@pytest.fixture(autouse=True)
def reset_counters():
    MakeData.calls = 0
    Summarize.calls = 0


def build_pipeline(store, manifests, config=None):
    cfg = {"make_data": {"rows": 2}, "summarize": {"factor": 1}, "seed": 42}
    if config:
        cfg.update(config)
    return Pipeline([MakeData(), Summarize()], store, manifests, cfg)


def seed_artifact(store, offset=0):
    return store.save(make_frame(offset), "seed", ext="parquet")


def test_cycle0_roundtrip(dirs):
    store, _ = dirs
    df = make_frame()
    art = store.save(df, "raw", ext="parquet")
    back = store.load(art.content_hash)
    assert back.equals(df)
    assert back.schema == df.schema
    assert back["x"].dtype == pl.Int8


def test_cycle0_dedup(dirs):
    store, _ = dirs
    a = store.save(make_frame(), "k1", ext="parquet")
    b = store.save(make_frame(), "k2", ext="parquet")
    assert a.content_hash == b.content_hash
    files = [p for p in Path(store.root).rglob("*") if p.is_file()]
    assert len(files) == 1


def test_cycle0_cache_hit(dirs):
    store, manifests = dirs
    seeds = [seed_artifact(store)]
    build_pipeline(store, manifests).run(seeds=seeds)
    result = build_pipeline(store, manifests).run(seeds=seeds)
    assert [m["status"] for m in result["manifests"]] == ["cached", "cached"]
    assert MakeData.calls == 1 and Summarize.calls == 1


def test_cycle0_cache_miss_on_config(dirs):
    store, manifests = dirs
    seeds = [seed_artifact(store)]
    build_pipeline(store, manifests).run(seeds=seeds)
    result = build_pipeline(
        store, manifests, {"summarize": {"factor": 3}}
    ).run(seeds=seeds)
    statuses = {m["step"]: m["status"] for m in result["manifests"]}
    assert statuses == {"make_data": "cached", "summarize": "success"}
    assert MakeData.calls == 1 and Summarize.calls == 2


DYN_SRC = """
import polars as pl
from mlpipe.core.interfaces import Step, StepResult

CONSTANT = __CONST__

class Scale(Step):
    name = "scale"
    inputs = ["raw_table"]
    outputs = ["scaled"]
    calls = 0

    def run(self, ctx):
        type(self).calls += 1
        df = ctx.get("raw_table")
        out = df.with_columns((pl.col("x") * CONSTANT).cast(pl.Int8))
        art = ctx.put(out, "scaled")
        return StepResult(outputs={"scaled": art})
"""


def load_scale_step(path: Path, const: int, tag: str):
    path.write_text(DYN_SRC.replace("__CONST__", str(const)))
    spec = importlib.util.spec_from_file_location(f"dynstep_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Scale


def test_cycle0_cache_miss_on_code(dirs, tmp_path):
    store, manifests = dirs
    seeds = [seed_artifact(store)]
    src = tmp_path / "dynstep.py"

    step_v1 = load_scale_step(src, const=2, tag="v1")
    cfg = {"make_data": {"rows": 2}}
    Pipeline([MakeData(), step_v1()], store, manifests, cfg).run(seeds=seeds)

    step_v2 = load_scale_step(src, const=3, tag="v2")
    result = Pipeline([MakeData(), step_v2()], store, manifests, cfg).run(seeds=seeds)
    statuses = {m["step"]: m["status"] for m in result["manifests"]}
    assert statuses == {"make_data": "cached", "scale": "success"}


def test_cycle0_cache_miss_on_data(dirs):
    store, manifests = dirs
    build_pipeline(store, manifests).run(seeds=[seed_artifact(store)])
    result = build_pipeline(store, manifests).run(seeds=[seed_artifact(store, offset=5)])
    assert [m["status"] for m in result["manifests"]] == ["success", "success"]
    assert MakeData.calls == 2 and Summarize.calls == 2


def test_cycle0_lineage(dirs):
    store, manifests = dirs
    seed = seed_artifact(store)
    pipe = build_pipeline(store, manifests)
    result = pipe.run(seeds=[seed])
    final_hash = result["artifacts"]["summary"].content_hash

    chain = pipe.lineage(final_hash)
    assert [m["step"] for m in chain] == ["summarize", "make_data"]
    assert seed.content_hash in chain[-1]["inputs"].values()
    assert chain[-1]["config_hash"]


def test_cycle0_manifest_completeness(dirs):
    store, manifests = dirs
    seeds = [seed_artifact(store)]
    build_pipeline(store, manifests).run(seeds=seeds)
    build_pipeline(store, manifests).run(seeds=seeds)  # cached pass

    import json

    index = (manifests / "index.jsonl").read_text().strip().splitlines()
    assert len(index) == 4  # 2 fresh + 2 cached executions, one manifest each
    for line in index:
        m = json.loads(line)
        assert m["git_commit"]
        assert m["config_hash"]
        assert isinstance(m["inputs"], dict) and m["inputs"]
        assert isinstance(m["outputs"], dict) and m["outputs"]
        assert m["started_at"] and m["ended_at"]
        assert m["status"] in ("success", "cached")
        assert m["environment"]["lockfile_hash"]
