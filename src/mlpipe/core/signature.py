"""Signature hashing: what makes a step execution unique (DESIGN.md §1).

signature = hash(step name, code hash over the step's source file *closure*,
canonical JSON of the step's resolved config subtree, sorted input hashes).

The closure is the step's own module plus every first-party module it can reach
by import, plus any module it resolves lazily and declares via `code_deps`
(model backends). Modules are located with `find_spec` and parsed with `ast` —
nothing in the closure is imported, so computing a signature never pulls in
lightgbm or torch.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

# Module prefixes whose source counts as step behaviour. mlpipe.core is
# deliberately excluded: it is the orchestrator, not the computation, and
# including it would invalidate every cached artifact on any core edit. Core
# changes are covered by tests and by the git_commit in every manifest.
FIRST_PARTY = ("mlpipe.steps.", "mlpipe.backends.")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_subtree(config: Any, step_name: str) -> dict[str, Any]:
    """The slice of config a step sees: config[step_name], {} if absent."""
    if hasattr(config, "model_dump"):
        config = config.model_dump()
    if isinstance(config, dict):
        return config.get(step_name, {})
    return {}


def config_hash(config: Any, step_name: str) -> str:
    return sha256_bytes(canonical_json(config_subtree(config, step_name)).encode())


def _module_path(name: str) -> Path | None:
    """Source file for a module name, without importing the module itself."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    return Path(spec.origin) if spec and spec.origin else None


def _imports(path: Path) -> set[str]:
    """First-party module names a source file imports (static parse, no import)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)  # `from pkg.mod import name`
            found.update(f"{node.module}.{a.name}" for a in node.names)  # `from pkg import mod`
    return {m for m in found if m.startswith(FIRST_PARTY)}


def code_modules(step: Any, config: Any = None) -> dict[str, Path]:
    """Every source file the step's behaviour depends on: own module + closure."""
    own = type(step).__module__
    found = {own: Path(inspect.getfile(type(step)))}
    queue = list(_imports(found[own])) + list(step.code_deps(config_subtree(config, step.name)))
    while queue:
        name = queue.pop()
        if name in found:
            continue
        path = _module_path(name)
        if path is None or not path.exists():
            continue  # a symbol, not a module
        found[name] = path
        queue.extend(_imports(path) - found.keys())
    return found


def code_hash(step: Any, config: Any = None) -> str:
    """Hash over the whole closure, keyed by module name so moves are visible."""
    files = {n: sha256_bytes(p.read_bytes()) for n, p in code_modules(step, config).items()}
    return sha256_bytes(canonical_json(files).encode())


def step_signature(step: Any, input_hashes: dict[str, str], config: Any) -> str:
    payload = {
        "step": step.name,
        "code": code_hash(step, config),
        "config": config_subtree(config, step.name),
        "inputs": {k: input_hashes[k] for k in sorted(input_hashes)},
    }
    return sha256_bytes(canonical_json(payload).encode())
