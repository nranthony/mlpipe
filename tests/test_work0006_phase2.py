"""work/0006 phase 2 — generic core (plan steps 9–12)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

import mlpipe.steps.features as features_mod
from mlpipe.backends.lightgbm_backend import LightGBMBackend
from mlpipe.backends.torch_backend import TorchMLPBackend
from mlpipe.core import context, interfaces
from mlpipe.core.interfaces import FoldPlan
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore
from mlpipe.steps.cvplan import CVConfig, build_plan
from mlpipe.steps.features import FeatureStep

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
