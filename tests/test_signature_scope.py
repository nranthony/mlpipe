"""Signature scope: a step's code hash must cover every source file its
behaviour depends on, not just the module defining the step class.

The gap this closes (work/0002): editing src/mlpipe/backends/*.py left
TrainStep's signature unchanged, so the pipeline served a stale model for
changed code. Commit e1c56c5 (the torch OOM fix) was exactly that shape.
"""

from __future__ import annotations

import sys

import pytest

from mlpipe.core import signature as sig
from mlpipe.core.interfaces import Step, StepResult
from mlpipe.steps.evaluate import EvaluateStep
from mlpipe.steps.train import TrainStep

# --------------------------------------------------------------- real modules


def test_train_closure_covers_the_configured_backend():
    mods = sig.code_modules(TrainStep(), {"train": {"model": {"kind": "lightgbm"}}})
    assert "mlpipe.steps.train" in mods                      # own module
    assert "mlpipe.backends.registry" in mods                # statically imported
    assert "mlpipe.backends.lightgbm_backend" in mods        # lazily resolved
    assert "mlpipe.backends.torch_backend" not in mods       # the other kind


def test_switching_backend_switches_the_hashed_files():
    step = TrainStep()
    torch_mods = sig.code_modules(step, {"train": {"model": {"kind": "torch_mlp"}}})
    assert "mlpipe.backends.torch_backend" in torch_mods
    assert "mlpipe.backends.lightgbm_backend" not in torch_mods


def test_evaluate_closure_covers_its_helper_module():
    """evaluate.py imports eval_metrics.py — the metric functions are behaviour."""
    assert "mlpipe.steps.eval_metrics" in sig.code_modules(EvaluateStep(), {})


def test_core_is_excluded_from_the_closure():
    """Documented trade: core edits do not invalidate 11 GB of artifacts."""
    mods = sig.code_modules(TrainStep(), {"train": {"model": {"kind": "lightgbm"}}})
    assert not any(m.startswith("mlpipe.core") for m in mods)


def test_nothing_in_the_closure_gets_imported():
    """Signatures are computed on cache hits too — they must not pull in torch."""
    heavy = ("lightgbm", "torch")
    before = {n for n in heavy if n in sys.modules}
    sig.code_modules(TrainStep(), {"train": {"model": {"kind": "torch_mlp"}}})
    assert {n for n in heavy if n in sys.modules} == before


# ------------------------------------------------------- sensitivity to edits


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    """A throwaway package whose files we can actually mutate."""
    root = tmp_path / "sigpkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "helper.py").write_text("def scale(x):\n    return x * 2\n")
    (root / "unrelated.py").write_text("VALUE = 1\n")
    (root / "stepmod.py").write_text(
        "from mlpipe.core.interfaces import Step, StepResult\n"
        "from sigpkg.helper import scale\n"
        "\n"
        "class TmpStep(Step):\n"
        "    name = 'tmp'\n"
        "    inputs: list[str] = []\n"
        "    outputs: list[str] = []\n"
        "    def run(self, ctx):\n"
        "        return StepResult(outputs={})\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sig, "FIRST_PARTY", ("sigpkg.",))
    for name in [m for m in sys.modules if m.startswith("sigpkg")]:
        del sys.modules[name]
    from sigpkg.stepmod import TmpStep

    yield root, TmpStep()
    for name in [m for m in sys.modules if m.startswith("sigpkg")]:
        del sys.modules[name]


def test_editing_an_imported_helper_changes_the_signature(pkg):
    root, step = pkg
    before = sig.step_signature(step, {}, {})
    (root / "helper.py").write_text("def scale(x):\n    return x * 3\n")  # behaviour change
    assert sig.step_signature(step, {}, {}) != before


def test_editing_an_unimported_module_leaves_the_signature_alone(pkg):
    root, step = pkg
    before = sig.step_signature(step, {}, {})
    (root / "unrelated.py").write_text("VALUE = 2\n")
    assert sig.step_signature(step, {}, {}) == before


def test_declared_lazy_dep_is_hashed(pkg):
    """A module reachable only through code_deps still counts."""
    root, step = pkg

    class LazyStep(type(step)):
        def code_deps(self, cfg):
            return ["sigpkg.unrelated"]

    lazy = LazyStep()
    before = sig.step_signature(lazy, {}, {})
    (root / "unrelated.py").write_text("VALUE = 3\n")
    assert sig.step_signature(lazy, {}, {}) != before


def test_editing_the_step_module_still_changes_the_signature(pkg):
    """The original guarantee must survive the widening."""
    root, step = pkg
    before = sig.step_signature(step, {}, {})
    (root / "stepmod.py").write_text((root / "stepmod.py").read_text() + "# tweak\n")
    assert sig.step_signature(step, {}, {}) != before


def test_step_without_lazy_deps_declares_none():
    class Bare(Step):
        name = "bare"
        inputs: list[str] = []
        outputs: list[str] = []

        def run(self, ctx):
            return StepResult(outputs={})

    assert Bare().code_deps({}) == []
