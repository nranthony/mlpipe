"""Registry: model config kind -> ModelBackend implementation class.

Values are lazy import strings so importing the registry (or steps that use
it) never pulls a model library — that happens only when a kind is resolved.
"""

from __future__ import annotations

import importlib

REGISTRY = {
    "lightgbm": "mlpipe.backends.lightgbm_backend:LightGBMBackend",
    "torch_mlp": "mlpipe.backends.torch_backend:TorchMLPBackend",
}


def resolve(kind: str) -> type:
    target = REGISTRY[kind]
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)
