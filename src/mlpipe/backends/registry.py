"""Registry: model config kind -> ModelBackend implementation class.

Values are lazy import strings so importing the registry (or steps that use
it) never pulls a model library — that happens only when a kind is resolved.
Host packages add kinds through the `mlpipe.backends` entry-point group
(mlpipe.plugins).
"""

from __future__ import annotations

import importlib

from mlpipe.plugins import BACKENDS, registered

REGISTRY = {
    "lightgbm": "mlpipe.backends.lightgbm_backend:LightGBMBackend",
    "torch_mlp": "mlpipe.backends.torch_backend:TorchMLPBackend",
}


def kinds() -> dict[str, str]:
    return registered(BACKENDS, REGISTRY)


def module_for(kind: str) -> str:
    """Module name backing a kind, without importing it (used by signatures)."""
    return kinds()[kind].split(":")[0]


def resolve(kind: str) -> type:
    module, cls = kinds()[kind].split(":")
    return getattr(importlib.import_module(module), cls)
