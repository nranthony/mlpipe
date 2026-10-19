"""Core boundary (ADR-0004): mlpipe.core imports only the stdlib, the base
dependencies, and itself — never steps, backends, trackers, model libraries,
tracker SDKs or domain packages. Checked by parsing, so lazy imports inside
functions count too.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import mlpipe.core

CORE_DIR = Path(mlpipe.core.__file__).parent
BASE_DEPS = {"polars", "numpy", "pydantic", "yaml", "joblib", "typer"}


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module)
    return roots


def _allowed(name: str) -> bool:
    top = name.split(".")[0]
    if name == "mlpipe.core" or name.startswith("mlpipe.core."):
        return True
    return top != "mlpipe" and (top in sys.stdlib_module_names or top in BASE_DEPS)


def test_core_imports_stay_inside_the_boundary():
    violations = {
        f"{p.name}: {name}"
        for p in sorted(CORE_DIR.glob("*.py"))
        for name in _imported_roots(p)
        if not _allowed(name)
    }
    assert not violations, sorted(violations)


def test_boundary_check_catches_a_leak(tmp_path):
    leaky = tmp_path / "leaky.py"
    leaky.write_text("def f():\n    import lightgbm\n    from mlpipe.steps import train\n")
    assert {n for n in _imported_roots(leaky) if not _allowed(n)} == {
        "lightgbm",
        "mlpipe.steps",
    }
