"""Signature hashing: what makes a step execution unique (DESIGN.md §1).

signature = hash(step name, code hash over the step's source file *closure*,
canonical JSON of the step's resolved config subtree, sorted input hashes,
CORE_SIGNATURE_VERSION).

The closure is the step's own module plus every first-party module it can reach
by import, plus any module it resolves lazily and declares via `code_deps`
(model backends) with that module's own first-party imports. Modules are
located with `find_spec` and parsed with `ast` — nothing in the closure is
imported, so computing a signature never pulls in lightgbm or torch.
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
# including it would invalidate every cached artifact on any core edit.
FIRST_PARTY = ("mlpipe.steps.", "mlpipe.backends.", "mlpipe.examples.")

# Bump by hand when a core edit changes what a step computes or how its config
# reads (store codecs, override parsing, config normalisation): every signature
# then moves once. Which mlpipe ran is recorded in each manifest's environment.
CORE_SIGNATURE_VERSION = 1


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
    """Source file for a module name, without importing the module itself.
    `pkg.mod.Symbol` (from `from pkg.mod import Symbol`) is no module: find_spec on
    it would import `pkg.mod` to look for a `__path__`, so a non-package parent
    answers None first (numerai work/0013: shrec imported on every cache hit)."""
    parent = name.rpartition(".")[0]
    try:
        if parent:
            outer = importlib.util.find_spec(parent)  # imports only parent's packages
            if outer is None or outer.submodule_search_locations is None:
                return None
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    return Path(spec.origin) if spec and spec.origin else None


def _first_party(own: str) -> tuple[str, ...]:
    """Prefixes whose source counts as behaviour for code rooted at module `own`:
    mlpipe's steps and backends, plus a host step's own top-level package, so a
    host helper edit changes the signature (work/0007 M1). Never mlpipe.core."""
    root = own.split(".")[0]
    return FIRST_PARTY if root in ("mlpipe", "__main__") else (*FIRST_PARTY, f"{root}.")


def _from_base(node: ast.ImportFrom, name: str, path: Path) -> str | None:
    """Absolute module a `from X import ...` reads from; relative forms
    (`from .helpers import f`, `from .. import mod`) resolve against the
    importing module's package. None if unresolvable (e.g. a __main__ script)."""
    if not node.level:
        return node.module
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    try:
        return importlib.util.resolve_name("." * node.level + (node.module or ""), package)
    except (ImportError, ValueError):
        return None


def _imports(path: Path, name: str, prefixes: tuple[str, ...]) -> set[str]:
    """First-party module names a source file imports (static parse, no import)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and (base := _from_base(node, name, path)):
            found.add(base)  # `from pkg.mod import name`
            found.update(f"{base}.{a.name}" for a in node.names)  # `from pkg import mod`
    host_roots = {p[:-1] for p in prefixes[len(FIRST_PARTY):]}  # `import hostpkg` itself
    return {m for m in found if m.startswith(prefixes) or m in host_roots}


def _closure(found: dict[str, Path], queue: list[str], prefixes: tuple[str, ...]) -> dict[str, Path]:
    while queue:
        name = queue.pop()
        if name in found:
            continue
        path = _module_path(name)
        if path is None or not path.exists():
            continue  # a symbol, not a module
        found[name] = path
        queue.extend(_imports(path, name, prefixes) - found.keys())
    return found


def _hash_files(found: dict[str, Path]) -> str:
    """Keyed by module name, so moves are visible."""
    files = {n: sha256_bytes(p.read_bytes()) for n, p in found.items()}
    return sha256_bytes(canonical_json(files).encode())


def code_modules(step: Any, config: Any = None) -> dict[str, Path]:
    """Every source file the step's behaviour depends on: own module + closure."""
    own = type(step).__module__
    roots = tuple(getattr(step, "code_roots", ()))
    prefixes = (*_first_party(own), *roots)
    found = {own: Path(inspect.getfile(type(step)))}
    _closure(found, list(_imports(found[own], own, prefixes)), prefixes)
    # A lazily resolved dep (a host's model backend) is closed over with its OWN
    # package as first-party, so the host helpers it imports are hashed too.
    for dep in step.code_deps(config_subtree(config, step.name)):
        _closure(found, [dep], (*_first_party(dep), *roots))
    return found


def code_hash(step: Any, config: Any = None) -> str:
    """Hash over the whole closure."""
    return _hash_files(code_modules(step, config))


def module_code_hash(name: str) -> str:
    """Hash of a module and its first-party closure, for artifacts whose bytes do
    not carry the code that runs them (a pickled model names its class only)."""
    found = _closure({}, [name], _first_party(name))
    if name not in found:
        raise ModuleNotFoundError(f"no source for module {name!r}")
    return _hash_files(found)


def step_signature(step: Any, input_hashes: dict[str, str], config: Any) -> str:
    payload = {
        "core": CORE_SIGNATURE_VERSION,
        "step": step.name,
        "code": code_hash(step, config),
        "config": config_subtree(config, step.name),
        "inputs": {k: input_hashes[k] for k in sorted(input_hashes)},
    }
    return sha256_bytes(canonical_json(payload).encode())
