"""work/0006 phase 2 — generic core (plan steps 9–12)."""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from sklearn.preprocessing import StandardScaler
from typer.testing import CliRunner

import mlpipe.steps.features as features_mod
from mlpipe.backends import registry
from mlpipe.backends.lightgbm_backend import LightGBMBackend
from mlpipe.backends.torch_backend import TorchMLPBackend
from mlpipe.cli import app
from mlpipe.core import context, interfaces
from mlpipe.core.interfaces import FoldPlan
from mlpipe.core.manifest import ManifestLog
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.signature import code_hash, code_modules, module_code_hash
from mlpipe.core.store import LocalCasStore
from mlpipe.plugins import pipeline_module
from mlpipe.steps.cvplan import CVConfig, build_plan
from mlpipe.steps.features import FeatureStep
from mlpipe.steps.train import TrainConfig

# --------------------------------------------------------------------------- step 9

PINNED_PLAN_HASH = "d3e6be73bfd7d2ecd355ec352ab5684cbe89b559d3d05eab9646926dc1401985"


def test_work0006_foldplan_is_domain_free():
    plan = FoldPlan(folds=[(np.array([0]), np.array([1]))], plan_hash="h")
    assert plan.group_column is None and plan.meta == {}
    assert not any("era" in f or "purge" in f or "embargo" in f for f in plan.__dataclass_fields__)


def test_work0006_cvplan_carries_era_semantics_in_meta():
    eras = pl.Series("era", [f"e{i:02d}" for i in range(12) for _ in range(3)])
    plan = build_plan(eras, CVConfig(n_folds=3, purge_eras=1, embargo_eras=2))
    assert plan.group_column == "era"
    assert plan.meta == {"purge_eras": 1, "embargo_eras": 2}


def test_work0006_cvplan_hash_unchanged_by_refactor():
    # Pinned before the FoldPlan refactor: the hash covers (eras, CVConfig) only,
    # so plans cited in STATUS.md stay comparable across it.
    eras = pl.Series("era", [f"e{i:02d}" for i in range(12) for _ in range(3)])
    assert build_plan(eras, CVConfig(n_folds=3)).plan_hash == PINNED_PLAN_HASH


class SpyScaler(StandardScaler):
    calls: list = []

    def fit(self, X, y=None, **kw):
        SpyScaler.calls.append((X.shape, y))
        return super().fit(X, y, **kw)


def test_work0006_feature_fits_never_see_the_target(tmp_path, monkeypatch):
    """D1: FeatureStep fits are unsupervised. Target-aware transforms belong in the
    backend, fitted per fold, or they leak the target across CV folds."""
    rng = np.random.default_rng(0)
    train = pl.DataFrame({
        "era": [f"e{i // 10}" for i in range(40)],
        "feature_a": rng.normal(size=40),
        "feature_b": rng.normal(size=40),
        "target": rng.uniform(size=40),
    })
    SpyScaler.calls = []
    monkeypatch.setattr(features_mod, "make_transformer", lambda kind: SpyScaler())
    store = LocalCasStore(tmp_path / "store")
    seeds = [store.save(train, "clean_table", ext="parquet"),
             store.save(train, "clean_validation", ext="parquet")]
    Pipeline([FeatureStep()], store, tmp_path / "m", {}).run(seeds=seeds)
    assert SpyScaler.calls == [((40, 2), None)]


# --------------------------------------------------------------------------- step 10

STEPS_DIR = Path(features_mod.__file__).parent


def declared_surface() -> set[str]:
    return {"config"} | set(interfaces.RunContext.__abstractmethods__)


def ctx_uses(path: Path) -> list[tuple[int, str]]:
    """Every `ctx.<attr>` (and `getattr(ctx, ...)`) in a module."""
    uses = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "ctx":
            uses.append((node.lineno, node.attr))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"getattr", "setattr", "vars"}
                and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "ctx"):
            uses.append((node.lineno, f"{node.func.id}()"))
    return uses


def test_work0006_steps_use_only_the_declared_run_context_surface():
    surface = declared_surface()
    stray = [
        f"{p.name}:{line} ctx.{attr}"
        for p in sorted(STEPS_DIR.glob("*.py"))
        for line, attr in ctx_uses(p)
        if attr not in surface
    ]
    assert not stray, f"steps reach past the RunContext port: {stray}"


def test_work0006_run_context_surface_is_declared_and_implemented():
    assert declared_surface() == {
        "config", "get", "get_lazy", "input_hashes", "previous",
        "put", "put_alias", "log_metric", "log_meta",
    }
    assert not context.RunContext.__abstractmethods__  # the implementation is complete


def test_work0006_model_backend_fit_takes_no_fold_plan():
    for cls in (interfaces.ModelBackend, LightGBMBackend, TorchMLPBackend):
        assert list(inspect.signature(cls.fit).parameters) == ["self", "X", "y", "params"]


def test_work0006_surface_check_catches_a_bypass(tmp_path):
    bad = tmp_path / "bad_step.py"
    bad.write_text("def run(ctx):\n    ctx._store.load('h')\n    getattr(ctx, 'registry')\n")
    assert sorted(ctx_uses(bad)) == [(2, "_store"), (3, "getattr()")]


# --------------------------------------------------------------------------- step 11

HOST_FILES = {
    "hostpkg/__init__.py": "",
    "hostpkg/helpers.py": "def scale(x):\n    return x * 2.0\n",
    "hostpkg/steps.py": '''\
import numpy as np
import polars as pl

from hostpkg.helpers import scale
from mlpipe.core.interfaces import Step, StepResult


class MakeTable(Step):
    name = "make"
    inputs = []
    outputs = ["feature_table"]

    def run(self, ctx):
        rng = np.random.default_rng(ctx.config.get("make", {}).get("seed", 0))
        a = rng.normal(size=80)
        df = pl.DataFrame({
            "era": [f"e{i // 10}" for i in range(80)],
            "feature_a": scale(a),
            "target": a + rng.normal(scale=0.5, size=80),
        })
        return StepResult(outputs={"feature_table": ctx.put(df, "feature_table")})
''',
    "hostpkg/backend.py": '''\
import numpy as np


class SlopeBackend:
    def fit(self, X, y, params):
        x = X.to_numpy()[:, 0]
        self.w = float(np.dot(x, y) / np.dot(x, x))

    def predict(self, X):
        return X.to_numpy()[:, 0] * self.w

    def save(self, path):
        path.write_text(str(self.w))

    def load(self, path):
        self.w = float(path.read_text())
''',
    "hostpkg/pipeline.py": '''\
from hostpkg.steps import MakeTable
from mlpipe.steps.cvplan import CVPlanStep
from mlpipe.steps.train import TrainStep

DEFAULT_CONFIG = {
    "cvplan": {"n_folds": 2, "purge_eras": 0, "embargo_eras": 0},
    "train": {"model": {"kind": "hostslope"}},
}


def build_steps():
    return [MakeTable(), CVPlanStep(), TrainStep()]
''',
    "hostpkg-0.1.dist-info/METADATA": "Metadata-Version: 2.1\nName: hostpkg\nVersion: 0.1\n",
    "hostpkg-0.1.dist-info/entry_points.txt": (
        "[mlpipe.pipelines]\nhostpipe = hostpkg.pipeline\n\n"
        "[mlpipe.backends]\nhostslope = hostpkg.backend:SlopeBackend\n"
    ),
}


@pytest.fixture()
def host_site(tmp_path, monkeypatch):
    """A host package 'installed' the way an editable install would appear:
    importable from sys.path, with entry points in its dist-info."""
    site = tmp_path / "site"
    for rel, text in HOST_FILES.items():
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_text(text)
    monkeypatch.syspath_prepend(str(site))
    yield site
    for name in [m for m in sys.modules if m == "hostpkg" or m.startswith("hostpkg.")]:
        del sys.modules[name]


def test_work0006_host_pipeline_and_backend_found_by_entry_point(host_site, tmp_path):
    assert pipeline_module("hostpipe").__name__ == "hostpkg.pipeline"
    assert registry.module_for("hostslope") == "hostpkg.backend"
    assert TrainConfig.model_validate({"model": {"kind": "hostslope"}}).model.kind == "hostslope"

    dirs = ["--store-dir", str(tmp_path / "store"), "--manifests-dir", str(tmp_path / "m")]
    result = CliRunner().invoke(app, ["run", "--pipeline", "hostpipe", *dirs])
    assert result.exit_code == 0, result.output
    assert [line.split()[:2] for line in result.output.splitlines()] == [
        ["make", "success"], ["cvplan", "success"], ["train", "success"],
    ]


def test_work0006_unknown_names_are_rejected():
    with pytest.raises(KeyError, match="unknown pipeline 'nope'"):
        pipeline_module("nope")
    with pytest.raises(ValueError, match="unknown model kind 'nope'"):
        TrainConfig.model_validate({"model": {"kind": "nope"}})


def test_work0006_a_name_claimed_twice_is_an_error(host_site):
    ep = host_site / "hostpkg-0.1.dist-info/entry_points.txt"
    ep.write_text("[mlpipe.backends]\nlightgbm = hostpkg.backend:SlopeBackend\n")
    with pytest.raises(ValueError, match="'lightgbm' is registered twice"):
        registry.kinds()


def test_work0006_host_helper_edit_changes_the_step_signature(host_site):
    from hostpkg.steps import MakeTable

    mods = code_modules(MakeTable())
    assert "hostpkg.helpers" in mods                      # host package is first-party
    assert not any(m.startswith("mlpipe.core") for m in mods)  # core never is
    before = code_hash(MakeTable())
    (host_site / "hostpkg/helpers.py").write_text("def scale(x):\n    return x * 3.0\n")
    assert code_hash(MakeTable()) != before


def test_work0006_model_artifact_carries_backend_code(host_site, tmp_path):
    before = module_code_hash("hostpkg.backend")
    pipe = Pipeline(pipeline_module("hostpipe").build_steps(), LocalCasStore(tmp_path / "s"),
                    tmp_path / "m", pipeline_module("hostpipe").DEFAULT_CONFIG)
    first = pipe.run()["artifacts"]["model"].content_hash
    assert LocalCasStore(tmp_path / "s").load(first)["backend_code"] == before

    # A predict-only edit leaves the fitted state identical, yet the model's bytes
    # move, so anything downstream that calls predict is invalidated too.
    path = host_site / "hostpkg/backend.py"
    path.write_text(path.read_text().replace("* self.w", "* self.w * 1.0"))
    assert module_code_hash("hostpkg.backend") != before
    assert pipe.run()["artifacts"]["model"].content_hash != first
    with pytest.raises(ModuleNotFoundError):
        module_code_hash("hostpkg.nope")


# --------------------------------------------------------------------------- step 13

SRC = Path(interfaces.__file__).parents[1]  # src/mlpipe


def test_work0006_cli_names_no_numerai_artifacts():
    text = (SRC / "cli.py").read_text()
    for name in (r"\bera\b", "era_column", "raw_live", "mean_corr", "(?i)numerai", "ticker"):
        assert not re.search(name, text), f"cli.py names {name!r}"


def test_work0006_nothing_generic_imports_the_examples():
    """mlpipe.examples is a leaf: core, steps, backends, trackers and the CLI
    never import it (plugins names the builtin pipelines as strings only)."""
    leaks = []
    for path in SRC.rglob("*.py"):
        if "examples" in path.relative_to(SRC).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            leaks += [f"{path.relative_to(SRC)}: {n}" for n in names if n.startswith("mlpipe.examples")]
    assert not leaks


def test_work0006_example_steps_also_use_only_the_run_context_surface():
    surface = declared_surface()
    stray = [f"{p.name}:{line} ctx.{attr}" for p in sorted((SRC / "examples").rglob("*.py"))
             for line, attr in ctx_uses(p) if attr not in surface]
    assert not stray


def test_work0006_promote_accepts_a_bundle_without_plan_or_target(tmp_path):
    store, log = LocalCasStore(tmp_path / "store"), ManifestLog(tmp_path / "m")
    bundle = store.save({"model_name": "plain", "metrics": {"rmse": 1.0}}, "model_bundle", ext="json")
    log.write({"step": "register", "status": "success", "signature": "s", "inputs": {},
               "outputs": {"model_bundle": bundle.content_hash}, "metrics": {}, "meta": {},
               "config": {}, "git_commit": None})
    baselines = tmp_path / "baselines.yaml"
    result = CliRunner().invoke(app, [
        "promote", bundle.content_hash[:10], "--pipeline", "demo", "--store-dir",
        str(tmp_path / "store"), "--manifests-dir", str(tmp_path / "m"), "--baselines", str(baselines),
    ])
    assert result.exit_code == 0, result.output
    pinned = yaml.safe_load(baselines.read_text())["promoted"]
    assert pinned["plan_hash"] is None and pinned["target"] is None
    assert pinned["metrics"] == {"rmse": 1.0}


def test_work0006_every_command_requires_a_pipeline():
    for cmd in (["run"], ["tune"], ["promote", "abc"], ["predict"], ["lineage", "abc"]):
        result = CliRunner().invoke(app, cmd)
        assert result.exit_code != 0 and "--pipeline" in result.output, cmd
