"""work/0009 part A — run_id, sibling steps, PredictStep and the serve flow of
`mlpipe predict` (ADR-0007). Toy data only: a small host package written to a
tmp dir, registered as builtin pipelines, with a numpy least-squares backend."""

from __future__ import annotations

import functools
import hashlib
import importlib
import sys

import polars as pl
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from mlpipe import plugins
from mlpipe.backends import registry
from mlpipe.cli import app
from mlpipe.core.interfaces import Step
from mlpipe.core.manifest import ManifestLog
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.signature import step_signature
from mlpipe.core.store import LocalCasStore
from mlpipe.steps import demo
from mlpipe.steps.predict import PredictStep, default_prepare

runner = CliRunner()

PKG = "serve9"
FILES = {
    "__init__.py": "",
    "backend.py": '''\
import numpy as np


class LeastSquares:
    def fit(self, X, y, params):
        self.w = np.linalg.lstsq(np.c_[X.to_numpy(), np.ones(X.height)], y, rcond=None)[0]

    def predict(self, X):
        return np.c_[X.to_numpy(), np.ones(X.height)] @ self.w
''',
    "pipeline.py": '''\
import numpy as np
import polars as pl

from mlpipe.core.interfaces import Step, StepResult
from mlpipe.steps.features import FeatureStep
from mlpipe.steps.groupcv import GroupKFoldStep
from mlpipe.steps.register import RegisterStep
from mlpipe.steps.tabular_eval import TabularEvalStep
from mlpipe.steps.train import TrainStep

DEFAULT_CONFIG = {
    "cvplan": {"n_folds": 3},
    "train": {"model": {"kind": "lstsq9"}, "target": "y"},
    "evaluate": {"target": "y"},
    "register": {"model_name": "toy9", "id_columns": ["row_id"]},
}


def frame(n, offset):
    rng = np.random.default_rng(offset)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pl.DataFrame({
        "row_id": [f"r{offset + i}" for i in range(n)],
        "feature_x1": x1, "feature_x2": x2,
        "y": 2.0 * x1 - x2 + rng.normal(scale=0.1, size=n),
    })


class IngestStep(Step):
    name = "ingest"
    inputs = []
    outputs = ["clean_table", "clean_validation"]

    def run(self, ctx):
        return StepResult(outputs={
            "clean_table": ctx.put(frame(60, 0), "clean_table"),
            "clean_validation": ctx.put(frame(20, 1000), "clean_validation"),
        })


def build_steps():
    return [IngestStep(), FeatureStep(), GroupKFoldStep(), TrainStep(), TabularEvalStep(),
            RegisterStep()]
''',
    "served.py": '''\
from pydantic import Field

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult
from mlpipe.steps.predict import PredictStep
from serve9.pipeline import *  # noqa: F403


class LoadConfig(StrictConfig):
    round: int = Field(gt=0)


class LoadStep(Step):
    name = "load"
    inputs = []
    outputs = ["raw_rows"]
    config_model = LoadConfig

    def run(self, ctx):
        cfg = LoadConfig.model_validate(ctx.config.get(self.name, {}))
        return StepResult(outputs={"raw_rows": ctx.put(frame(5, 5000 + cfg.round), "raw_rows")})


def build_serve_steps():
    return [LoadStep(), PredictStep(frame_key="raw_rows")]
''',
    "hooked.py": "from serve9.pipeline import *  # noqa: F403\n\n\n"
                 "def serve_input(log, store):\n    return None, 'latest'\n",
    "prepared.py": "from serve9.pipeline import *  # noqa: F403\n\n\n"
                   "def prepare(frame, bundle):\n    return frame, frame\n",
    "badsource.py": "from mlpipe.steps.predict import PredictStep\n"
                    "from serve9.pipeline import *  # noqa: F403\n\n\n"
                    "def build_serve_steps():\n    return [PredictStep()]\n",
    "badlast.py": "from serve9.served import *  # noqa: F403\n\n\n"
                  "def build_serve_steps():\n    return [LoadStep()]\n",
}
VARIANTS = ("pipeline", "served", "hooked", "prepared", "badsource", "badlast")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """One trained toy bundle under pipeline `serve9`; the variants share its log."""
    root = tmp_path_factory.mktemp("work0009")
    for name, text in FILES.items():
        (root / PKG).mkdir(exist_ok=True)
        (root / PKG / name).write_text(text)
    mp = pytest.MonkeyPatch()
    mp.syspath_prepend(str(root))
    mp.setitem(registry.REGISTRY, "lstsq9", f"{PKG}.backend:LeastSquares")
    for v in VARIANTS:
        mp.setitem(plugins.BUILTIN_PIPELINES, f"{PKG}{v}", f"{PKG}.{v}")
    dirs = ["--store-dir", str(root / "store"), "--manifests-dir", str(root / "manifests")]
    result = runner.invoke(app, ["run", "--pipeline", f"{PKG}pipeline", *dirs])
    assert result.exit_code == 0, result.output
    log = ManifestLog(root / "manifests")
    bundle = [m for m in log.entries() if m["step"] == "register"][-1]["outputs"]["model_bundle"]
    yield root, dirs, bundle
    mp.undo()
    for name in [m for m in sys.modules if m == PKG or m.startswith(f"{PKG}.")]:
        del sys.modules[name]


def score_file(root, name: str, offset: int, n: int = 5):
    path = root / f"{name}.parquet"
    importlib.import_module(f"{PKG}.pipeline").frame(n, offset).write_parquet(path)
    return path


def predict(dirs, bundle, pipeline="pipeline", *args):
    return runner.invoke(app, ["predict", "--pipeline", f"{PKG}{pipeline}", "--bundle",
                               bundle[:10], *dirs, *args])


# ------------------------------------------------------------------- A1. run_id


class Boom(Step):
    name = "boom"
    inputs = ["summary"]
    outputs = ["nothing"]

    def run(self, ctx):
        raise RuntimeError("boom")


def test_work0009_run_id_groups_one_run(tmp_path):
    store, cfg = LocalCasStore(tmp_path / "store"), demo.DEFAULT_CONFIG
    first = Pipeline(demo.build_steps(), store, tmp_path / "m", cfg).run()
    ids = {m["run_id"] for m in first["manifests"]}
    assert len(ids) == 1 and len(ids.pop()) == 32
    second = Pipeline(demo.build_steps(), store, tmp_path / "m", cfg).run()
    assert {m["status"] for m in second["manifests"]} == {"cached"}
    assert len({m["run_id"] for m in second["manifests"]}) == 1
    assert second["manifests"][0]["run_id"] != first["manifests"][0]["run_id"]

    with pytest.raises(RuntimeError, match="boom"):
        Pipeline([*demo.build_steps(), Boom()], store, tmp_path / "m", cfg).run()
    last = ManifestLog(tmp_path / "m").entries()[-3:]
    assert [m["status"] for m in last] == ["cached", "cached", "failed"]
    assert len({m["run_id"] for m in last}) == 1


# -------------------------------------------------------------- A2. sibling steps


def test_work0009_sibling_subtrees_pass_top_level_validation(tmp_path):
    store = LocalCasStore(tmp_path / "store")
    cfg = {**demo.DEFAULT_CONFIG, "serve_source": {"round": 7}}
    with pytest.raises(ValidationError, match="serve_source"):
        Pipeline(demo.build_steps(), store, tmp_path / "m", cfg).run()
    result = Pipeline(demo.build_steps(), store, tmp_path / "m", cfg,
                      sibling_steps=["serve_source"]).run()
    assert all("serve_source" not in m["config"] for m in result["manifests"])

    bogus = {**cfg, "bogus": 1}  # an unknown top-level key still dies
    with pytest.raises(ValidationError, match="bogus"):
        Pipeline(demo.build_steps(), store, tmp_path / "m", bogus,
                 sibling_steps=["serve_source"]).run()


def test_work0009_a_name_in_both_lists_is_validated_as_an_own_step(tmp_path):
    """The review dropped the overlap error (spec A2): an overlapping name is
    simply one of this pipeline's steps, so its subtree is still validated."""
    store = LocalCasStore(tmp_path / "store")
    pipe = Pipeline(demo.build_steps(), store, tmp_path / "m", demo.DEFAULT_CONFIG,
                    sibling_steps=["summarize"])
    assert pipe.run()["manifests"][-1]["step"] == "summarize"
    with pytest.raises(ValidationError, match="factr"):
        pipe.run(overrides=["summarize.factr=3"])


def test_work0009_cli_run_accepts_the_serve_steps_subtrees(env):
    root, dirs, _ = env
    cfg = root / "with_serve.yaml"
    cfg.write_text("cvplan: {n_folds: 3}\ntrain: {model: {kind: lstsq9}, target: y}\n"
                   "evaluate: {target: y}\nregister: {model_name: toy9, id_columns: [row_id]}\n"
                   "load: {round: 3}\n")
    result = runner.invoke(app, ["run", "--pipeline", f"{PKG}served", "--config", str(cfg), *dirs])
    assert result.exit_code == 0, result.output
    assert "cached" in result.output and "load" not in result.output  # training only


# ------------------------------------------------------------------ A3. PredictStep


def seeds_for(store, bundle_hash, frame_key="input_frame", frame=None):
    from mlpipe.core.interfaces import Artifact

    content = store.load(bundle_hash)
    arts = [Artifact("model_bundle", bundle_hash, store.path_of(bundle_hash), {})]
    for key, field in (("model", "model_hash"), ("fitted_transformer", "transformer_hash")):
        arts.append(Artifact(key, content[field], store.path_of(content[field]), {}))
    return [*arts, store.save(frame, frame_key)] if frame is not None else arts


def test_work0009_predict_step_matches_the_old_cli_arithmetic(env, tmp_path):
    root, _, bundle_hash = env
    store = LocalCasStore(root / "store")
    frame = importlib.import_module(f"{PKG}.pipeline").frame(8, 3000)
    result = Pipeline([PredictStep()], store, tmp_path / "m", {}).run(
        seeds=seeds_for(store, bundle_hash, frame=frame))
    got = store.load(result["artifacts"]["predictions"].content_hash)

    content = store.load(bundle_hash)  # what cli.predict computed before work/0009
    feats, ids = default_prepare(frame, content)
    matrix = store.load(content["transformer_hash"]).transform(feats.to_numpy())
    model = store.load(content["model_hash"])["full"]
    preds = model.predict(pl.from_numpy(matrix, schema=content["features"]))
    assert got.equals(ids.with_columns(pl.Series("prediction", preds)))
    assert got.columns == ["row_id", "prediction"]
    (m,) = result["manifests"]
    assert m["meta"]["rows"] == 8 and m["meta"]["model_name"] == "toy9"
    assert m["meta"]["bundle"] == bundle_hash and m["meta"]["plan_hash"] == content["plan_hash"]


def test_work0009_a_prepare_edit_changes_the_predict_signature(tmp_path, monkeypatch):
    pkg = tmp_path / "prep9"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "serve.py").write_text(
        "from prep9.util import k\n\n\ndef prepare(frame, bundle):\n    return frame, frame * k\n")
    (pkg / "util.py").write_text("k = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.startswith("prep9")]:
        monkeypatch.delitem(sys.modules, name)
    step = PredictStep(prepare=importlib.import_module("prep9.serve").prepare)
    assert step.code_deps({}) == ["prep9.serve"]
    before = step_signature(step, {}, {})
    assert before != step_signature(PredictStep(), {}, {})
    (pkg / "util.py").write_text("k = 2\n")
    assert step_signature(step, {}, {}) != before


def test_work0009_prepare_must_be_a_module_level_function():
    for bad in (lambda f, b: (f, f), functools.partial(default_prepare), "default_prepare"):
        with pytest.raises(TypeError, match="module-level function"):
            PredictStep(prepare=bad)

    def nested(frame, bundle):
        return frame, frame

    with pytest.raises(TypeError):
        PredictStep(prepare=nested)
    assert PredictStep(prepare=default_prepare).code_deps({}) == ["mlpipe.steps.predict"]


def test_work0009_a_seeded_model_that_is_not_the_bundles_raises(env, tmp_path):
    root, _, bundle_hash = env
    store = LocalCasStore(root / "store")
    frame = importlib.import_module(f"{PKG}.pipeline").frame(4, 3100)
    seeds = seeds_for(store, bundle_hash, frame=frame)
    seeds[1] = store.save({"full": None, "other": True}, "model", ext="joblib")
    with pytest.raises(ValueError, match="not the bundle's model_hash"):
        Pipeline([PredictStep()], store, tmp_path / "m", {}).run(seeds=seeds)


# ----------------------------------------------------------- A4. mlpipe predict


def test_work0009_predict_input_writes_manifests_and_caches(env, tmp_path):
    root, dirs, bundle = env
    src, out = score_file(tmp_path, "score", 4000, n=6), tmp_path / "preds.parquet"
    result = predict(dirs, bundle, "pipeline", "--input", str(src), "--out", str(out))
    assert result.exit_code == 0, result.output

    entries = ManifestLog(root / "manifests").entries()
    record, pred = entries[-2], entries[-1]
    assert record["step"] == "input" and record["status"] == "success"
    assert record["meta"] == {"file": str(src), "bytes": src.stat().st_size}
    frame_hash = record["outputs"]["input_frame"]
    assert record["signature"] == f"input:{frame_hash}" and record["run_id"]
    assert pred["step"] == "predict" and pred["status"] == "success"
    assert pred["inputs"]["input_frame"] == frame_hash and pred["inputs"]["model_bundle"] == bundle
    preds_hash = pred["outputs"]["predictions"]
    assert hashlib.sha256(out.read_bytes()).hexdigest() == preds_hash
    assert pl.read_parquet(out).columns == ["row_id", "prediction"]
    assert preds_hash[:8] in result.output and "6 rows" in result.output

    again = predict(dirs, bundle, "pipeline", "--input", str(src), "--out", str(out))
    assert again.exit_code == 0, again.output
    last = ManifestLog(root / "manifests").entries()[-1]
    assert last["step"] == "predict" and last["status"] == "cached"
    assert last["outputs"]["predictions"] == preds_hash

    lineage = runner.invoke(app, ["lineage", preds_hash[:10], "--pipeline", f"{PKG}pipeline",
                                  "--manifests-dir", str(root / "manifests")])
    assert lineage.exit_code == 0, lineage.output
    steps = [line.split()[0] for line in lineage.output.splitlines()]
    assert steps[0] == "predict" and {"input", "register", "train", "ingest"} <= set(steps)


def test_work0009_predict_runs_the_serve_steps(env, tmp_path):
    root, dirs, bundle = env
    out = tmp_path / "served.parquet"
    result = predict(dirs, bundle, "served", "--set", "load.round=3", "--out", str(out))
    assert result.exit_code == 0, result.output
    load, pred = ManifestLog(root / "manifests").entries()[-2:]
    assert (load["step"], pred["step"]) == ("load", "predict")
    assert load["run_id"] == pred["run_id"] and load["config"] == {"round": 3}
    assert pred["inputs"]["raw_rows"] == load["outputs"]["raw_rows"]
    assert pl.read_parquet(out).height == 5

    # --input drops the source step; its config subtree is then a sibling
    src = score_file(tmp_path, "adhoc", 4100)
    result = predict(dirs, bundle, "served", "--input", str(src), "--set", "load.round=3",
                     "--out", str(out))
    assert result.exit_code == 0, result.output
    record, pred = ManifestLog(root / "manifests").entries()[-2:]
    assert record["step"] == "input" and list(record["outputs"]) == ["raw_rows"]
    assert pred["step"] == "predict" and pred["inputs"]["raw_rows"] == record["outputs"]["raw_rows"]


def test_work0009_predict_rejects_the_old_hooks(env, tmp_path):
    _, dirs, bundle = env
    src = score_file(tmp_path, "hooks", 4200)
    for variant, hook in (("hooked", "serve_input"), ("prepared", "prepare")):
        result = predict(dirs, bundle, variant, "--input", str(src))
        assert result.exit_code == 2 and "ADR-0007" in result.output and hook in result.output


def test_work0009_predict_without_serve_steps_needs_input(env):
    _, dirs, bundle = env
    result = predict(dirs, bundle, "pipeline")
    assert result.exit_code == 2 and "--input" in result.output


def test_work0009_bad_serve_lists_are_usage_errors(env, tmp_path):
    _, dirs, bundle = env
    for variant, words in (("badsource", "source"), ("badlast", "PredictStep")):
        result = predict(dirs, bundle, variant, "--set", "load.round=1")
        assert result.exit_code == 2 and words in result.output, result.output


def test_work0009_config_errors_are_usage_errors(env, tmp_path):
    root, dirs, bundle = env
    src = score_file(tmp_path, "cfg", 4300)
    before = len(ManifestLog(root / "manifests").entries())
    result = predict(dirs, bundle, "pipeline", "--input", str(src), "--set", "predict.x=1")
    assert result.exit_code == 2 and "config_model" in result.output, result.output
    assert len(ManifestLog(root / "manifests").entries()) == before  # no stray input record
    result = predict(dirs, bundle, "served")  # load.round is required
    assert result.exit_code == 2 and "round" in result.output, result.output


def test_work0009_a_backend_edit_after_training_fails_a_cache_miss(env, tmp_path):
    root, dirs, bundle = env
    backend = root / PKG / "backend.py"
    original = backend.read_text()
    try:
        backend.write_text(original + "\n# edited after training\n")
        src = score_file(tmp_path, "edited", 4400)
        result = predict(dirs, bundle, "pipeline", "--input", str(src),
                         "--out", str(tmp_path / "p.parquet"))
        assert result.exit_code != 0
        assert "backend code changed" in str(result.exception)
        assert ManifestLog(root / "manifests").entries()[-1]["status"] == "failed"
    finally:
        backend.write_text(original)


def test_work0009_a_bundle_whose_model_is_missing_is_a_usage_error(env, tmp_path):
    root, _, bundle = env
    store, log = LocalCasStore(tmp_path / "store"), ManifestLog(tmp_path / "m")
    content = LocalCasStore(root / "store").load(bundle) | {"model_hash": "f" * 64}
    fake = store.save(content, "model_bundle", ext="json").content_hash
    log.write({"step": "register", "status": "success", "signature": "s", "inputs": {},
               "outputs": {"model_bundle": fake}, "metrics": {}, "meta": {}, "config": {},
               "git_commit": None})
    src = score_file(tmp_path, "missing", 4500)
    dirs = ["--store-dir", str(tmp_path / "store"), "--manifests-dir", str(tmp_path / "m")]
    result = predict(dirs, fake, "pipeline", "--input", str(src))
    assert result.exit_code == 2 and "f" * 12 in result.output, result.output
