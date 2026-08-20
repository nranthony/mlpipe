"""Signature hashing: what makes a step execution unique (DESIGN.md §1).

signature = hash(step name, code hash of the step's module file,
canonical JSON of the step's resolved config subtree, sorted input hashes).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any


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


def code_hash(step: Any) -> str:
    """Hash of the source file defining the step's class."""
    return sha256_bytes(Path(inspect.getfile(type(step))).read_bytes())


def step_signature(step: Any, input_hashes: dict[str, str], config: Any) -> str:
    payload = {
        "step": step.name,
        "code": code_hash(step),
        "config": config_subtree(config, step.name),
        "inputs": {k: input_hashes[k] for k in sorted(input_hashes)},
    }
    return sha256_bytes(canonical_json(payload).encode())
