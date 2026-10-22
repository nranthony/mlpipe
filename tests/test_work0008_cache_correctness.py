"""work/0008 — cache and provenance correctness before a host adopts the core.

Each test pins one way a step's output could change without its signature
changing (a stale cache hit), or one way a manifest misreported what ran.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import BaseModel, ValidationError

import mlpipe.examples
import mlpipe.steps
from mlpipe.core import signature
from mlpipe.core.config import forbids_extras
from mlpipe.core.interfaces import FoldPlan, Step, StepResult
from mlpipe.core.pipeline import Pipeline, apply_overrides
from mlpipe.core.signature import code_modules, step_signature
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.demo import MakeData, Summarize
from mlpipe.steps.register import RegisterStep
from mlpipe.steps.train import TrainStep

SRC = Path(mlpipe.steps.__file__).parents[1]  # src/mlpipe

# --------------------------------------------------------------------------- item 1


class Sneaky(Step):
    """Reads `raw_table` without declaring it."""

    name = "sneak"
    inputs: list[str] = []
    outputs = ["copy"]

    def run(self, ctx):
        return StepResult(outputs={"copy": ctx.put(ctx.get("raw_table"), "copy")})


class SneakyAlias(Sneaky):
    def run(self, ctx):
        return StepResult(outputs={"copy": ctx.put_alias("copy", "raw_table")})


@pytest.mark.parametrize("step", [Sneaky(), SneakyAlias()])
def test_work0008_undeclared_inputs_cannot_be_read(tmp_path, step):
    pipe = Pipeline([MakeData(), step], LocalCasStore(tmp_path / "s"), tmp_path / "m")
    with pytest.raises(KeyError, match="undeclared input 'raw_table'"):
        pipe.run()


def test_work0008_declared_inputs_still_read(tmp_path):
    pipe = Pipeline([MakeData(), Summarize()], LocalCasStore(tmp_path / "s"), tmp_path / "m")
    assert [m["status"] for m in pipe.run()["manifests"]] == ["success", "success"]


# --------------------------------------------------------------------------- item 2


def config_reads(path: Path) -> list[tuple[int, str]]:
    """Literal keys a module reads from `ctx.config` (`.get("k")` or `["k"]`)."""
    reads = []
    for node in ast.walk(ast.parse(path.read_text())):
        target = key = None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args):
            target, key = node.func.value, node.args[0]
        elif isinstance(node, ast.Subscript):
            target, key = node.value, node.slice
        if (isinstance(target, ast.Attribute) and target.attr == "config"
                and isinstance(key, ast.Constant) and isinstance(key.value, str)):
            reads.append((node.lineno, key.value))
    return reads


def test_work0008_steps_read_only_their_own_config():
    """A step's signature hashes config[self.name] only. Reading another step's
    section (or any literal key) is invisible to it; `download_dir` is the one
    top-level location key, outside signatures by design."""
    stray = [
        f"{p.relative_to(SRC)}:{line} config[{key!r}]"
        for root in (SRC / "steps", SRC / "examples")
        for p in sorted(root.rglob("*.py"))
        for line, key in config_reads(p)
        if key != "download_dir"
    ]
    assert not stray, stray


def test_work0008_config_read_check_catches_a_cross_read(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def run(self, ctx):\n    ctx.config.get('clean', {})\n    ctx.config['train']\n")
    assert config_reads(bad) == [(2, "clean"), (3, "train")]


def _register_seeds(store: LocalCasStore, recipe: dict) -> list:
    plan = FoldPlan(folds=[(np.array([0]), np.array([1]))], plan_hash="p" * 64)
    return [
        store.save({"full": None}, "model", ext="joblib"),
        store.save({"kind": "passthrough"}, "fitted_transformer", ext="joblib"),
        store.save(pl.DataFrame({"era": ["a"], "feature_x": [1]}), "feature_table"),
        store.save(plan, "fold_plan", ext="joblib"),
        store.save({"plan_hash": plan.plan_hash, "target": "target", "metrics": {}},
                   "eval_report", ext="json"),
        store.save(recipe, "serve_recipe", ext="json"),
    ]


def test_work0008_register_recipe_is_an_input_not_a_config_read(tmp_path):
    store = LocalCasStore(tmp_path / "s")
    pipe = Pipeline([RegisterStep(recipe="serve_recipe")], store, tmp_path / "m")
    fill2 = {"cleaner": {"kind": "baseline", "fill_value": 2}, "era_column": "date"}

    first = pipe.run(seeds=_register_seeds(store, fill2))["manifests"][0]
    bundle = store.load(first["outputs"]["model_bundle"])
    assert bundle["recipe"] == fill2
    assert "cleaner" not in bundle and "model_config" not in bundle
    assert pipe.run(seeds=_register_seeds(store, fill2))["manifests"][0]["status"] == "cached"

    # The flaw this fixes: a recipe change with byte-identical tables used to
    # leave register cached, so the bundle kept the old fill.
    fill3 = {**fill2, "cleaner": {"kind": "baseline", "fill_value": 3}}
    rerun = pipe.run(seeds=_register_seeds(store, fill3))["manifests"][0]
    assert rerun["status"] == "success"
    assert store.load(rerun["outputs"]["model_bundle"])["recipe"] == fill3


def test_work0008_register_without_a_recipe_keeps_its_inputs():
    assert RegisterStep().inputs == RegisterStep.inputs
    assert RegisterStep(recipe="r").inputs == [*RegisterStep.inputs, "r"]


# --------------------------------------------------------------------------- item 3


def test_work0008_a_typo_in_a_step_config_fails_before_any_step_runs(tmp_path):
    pipe = Pipeline([TrainStep()], LocalCasStore(tmp_path / "s"), tmp_path / "m",
                    {"train": {"modle": {"kind": "torch_mlp"}}})
    with pytest.raises(ValidationError, match="modle"):
        pipe.run()
    with pytest.raises(ValidationError, match="kindd"):  # nested models forbid too
        Pipeline([TrainStep()], LocalCasStore(tmp_path / "s"), tmp_path / "m").run(
            overrides=["train.model.kindd=lightgbm"])
    assert not (tmp_path / "m" / "index.jsonl").exists()


class LooseConfig(BaseModel):
    x: int = 1


class LooseStep(Summarize):
    config_model = LooseConfig


def test_work0008_pipeline_refuses_a_config_model_that_allows_extras(tmp_path):
    pipe = Pipeline([MakeData(), LooseStep()], LocalCasStore(tmp_path / "s"), tmp_path / "m")
    with pytest.raises(TypeError, match="must forbid extra keys"):
        pipe.run()


class Unvalidated(Step):
    """Reads its config with no config_model: nothing could reject a typo."""

    name = "loose"
    inputs: list[str] = []
    outputs = ["out"]

    def run(self, ctx):
        n = ctx.config.get(self.name, {}).get("n", 1)
        return StepResult(outputs={"out": ctx.put({"n": n}, "out", ext="json")})


def test_work0008_config_for_a_step_without_a_config_model_is_refused(tmp_path):
    store, logs = LocalCasStore(tmp_path / "s"), tmp_path / "m"
    assert Pipeline([Unvalidated()], store, logs).run()["manifests"][0]["status"] == "success"
    with pytest.raises(TypeError, match="has config but no config_model"):
        Pipeline([Unvalidated()], store, logs, {"loose": {"nn": 2}}).run()


def _shipped_config_models() -> list[type[BaseModel]]:
    found = []
    for pkg in (mlpipe.steps, mlpipe.examples):
        for info in pkgutil.walk_packages(pkg.__path__, f"{pkg.__name__}."):
            module = importlib.import_module(info.name)
            found += [
                obj for obj in vars(module).values()
                if isinstance(obj, type) and issubclass(obj, BaseModel)
                and obj.__module__ == module.__name__
            ]
    return found


def test_work0008_every_shipped_config_model_forbids_extras():
    models = _shipped_config_models()
    assert len(models) >= 15  # the walk really found them
    assert [m.__qualname__ for m in models if not forbids_extras(m)] == []


# --------------------------------------------------------------------------- items 4, 5

HOST = {
    "hostpkg8/__init__.py": "",
    "hostpkg8/helpers.py": "def slope(x, y):\n    return float((x * y).sum() / (x * x).sum())\n",
    "hostpkg8/backend.py": (
        "from hostpkg8.helpers import slope\n\n\n"
        "class Backend:\n"
        "    def fit(self, X, y, params):\n"
        "        self.w = slope(X.to_numpy()[:, 0], y)\n\n"
        "    def predict(self, X):\n"
        "        return X.to_numpy()[:, 0] * self.w\n"
    ),
    "extlib8/__init__.py": "",
    "extlib8/core.py": "K = 1\n",
    "hostpkg8/feature_step.py": (
        "from extlib8.core import K\n"
        "from mlpipe.core.interfaces import Step\n\n\n"
        "class UsesLib(Step):\n"
        "    name = 'uses_lib'\n"
        "    inputs = []\n"
        "    outputs = []\n\n"
        "    def run(self, ctx):\n"
        "        return K\n"
    ),
    "hostpkg8-0.1.dist-info/METADATA": "Metadata-Version: 2.1\nName: hostpkg8\nVersion: 0.1\n",
    "hostpkg8-0.1.dist-info/entry_points.txt": "[mlpipe.backends]\nslope8 = hostpkg8.backend:Backend\n",
}


@pytest.fixture()
def host(tmp_path, monkeypatch):
    for rel, text in HOST.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [m for m in sys.modules if m.split(".")[0] in ("hostpkg8", "extlib8")]:
        del sys.modules[name]


def test_work0008_host_backend_helpers_are_in_trains_closure(host):
    config = {"train": {"model": {"kind": "slope8"}}}
    mods = code_modules(TrainStep(), config)
    assert {"hostpkg8.backend", "hostpkg8.helpers"} <= set(mods)
    before = signature.code_hash(TrainStep(), config)
    (host / "hostpkg8/helpers.py").write_text("def slope(x, y):\n    return 2.0\n")
    assert signature.code_hash(TrainStep(), config) != before


def test_work0008_code_roots_bring_a_sibling_library_into_the_closure(host):
    from hostpkg8.feature_step import UsesLib

    assert "extlib8.core" not in code_modules(UsesLib())

    class Rooted(UsesLib):
        code_roots = ("extlib8.",)

    Rooted.__module__ = UsesLib.__module__  # same source file, roots added
    assert "extlib8.core" in code_modules(Rooted())


# --------------------------------------------------------------------------- item 6


def test_work0008_small_floats_in_overrides_stay_floats():
    out = apply_overrides({}, [
        "a.small=1e-05", "a.bare=1e-5", f"a.tune={1e-05}", "a.dotted=1.5e-05",
        "a.flag=true", "a.word=abc", "a.yaml_flag=yes", "a.list=[1, 2]",
    ])["a"]
    assert out == {"small": 1e-05, "bare": 1e-05, "tune": 1e-05, "dotted": 1.5e-05,
                   "flag": True, "word": "abc", "yaml_flag": True, "list": [1, 2]}
    assert all(isinstance(out[k], float) for k in ("small", "bare", "tune"))


# --------------------------------------------------------------------------- item 7


def _git_head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_work0008_manifest_records_which_mlpipe_ran(tmp_path):
    from importlib import metadata

    # project_root is a plain directory (no git): the host commit is unknown,
    # while mlpipe's own version and commit are still recorded.
    pipe = Pipeline([MakeData()], LocalCasStore(tmp_path / "s"), tmp_path / "m",
                    project_root=tmp_path, packages=("pyyaml",))
    record = pipe.run()["manifests"][0]
    env = record["environment"]
    assert record["git_commit"] is None and env["git_dirty"] is None
    assert env["mlpipe"]["version"] == metadata.version("mlpipe")
    assert env["mlpipe"]["commit"] == _git_head(SRC.parents[1])
    assert {"polars", "numpy", "pyyaml"} <= set(env["packages"])


def test_work0008_core_signature_version_is_in_every_signature(monkeypatch):
    step, inputs = Summarize(), {"raw_table": "h" * 64}
    before = step_signature(step, inputs, {})
    monkeypatch.setattr(signature, "CORE_SIGNATURE_VERSION", signature.CORE_SIGNATURE_VERSION + 1)
    assert step_signature(step, inputs, {}) != before
