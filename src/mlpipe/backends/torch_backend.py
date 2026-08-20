"""TorchMLPBackend — the port proof (DESIGN.md block 8, cycle 8).

Tensors, DataLoaders, and device placement live ENTIRELY inside this adapter;
the port surface is polars/numpy/Path/FoldPlan only. Centralized VRAM guard:
any config whose estimated footprint exceeds the budget fails before training.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from mlpipe.core.interfaces import FoldPlan

DEFAULTS = {
    "hidden": [256, 128],
    "dropout": 0.1,
    "lr": 1e-3,
    "epochs": 3,
    "batch_size": 4096,
    "seed": 42,
    "vram_budget_gb": 12.0,
}


def estimate_vram_gb(n_features: int, hidden: list[int], batch_size: int) -> float:
    widths = [n_features, *hidden, 1]
    params = sum(a * b + b for a, b in itertools.pairwise(widths))
    activations = batch_size * sum(widths)
    # fp32 weights + grads + Adam moments (4x params); activations x2 (fwd+bwd)
    return (params * 4 * 4 + activations * 4 * 2) / 2**30


class TorchMLPBackend:
    def __init__(self) -> None:
        self._model = None
        self._params: dict[str, Any] = {}

    def _build(self, n_features: int, p: dict[str, Any]):
        import torch

        layers: list[Any] = []
        widths = [n_features, *p["hidden"]]
        for a, b in itertools.pairwise(widths):
            layers += [torch.nn.Linear(a, b), torch.nn.ReLU(), torch.nn.Dropout(p["dropout"])]
        layers.append(torch.nn.Linear(widths[-1], 1))
        return torch.nn.Sequential(*layers)

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        params: dict[str, Any],
        folds: FoldPlan | None = None,
    ) -> None:
        import torch

        p = {**DEFAULTS, **params}
        self._params = p
        estimate = estimate_vram_gb(X.width, p["hidden"], p["batch_size"])
        if estimate > p["vram_budget_gb"]:
            raise MemoryError(
                f"config would need ~{estimate:.1f} GiB VRAM, budget is "
                f"{p['vram_budget_gb']} GiB — shrink hidden/batch_size"
            )
        torch.manual_seed(p["seed"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self._build(X.width, p).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
        # keep the CPU tensor in the source dtype (int8 for numerai features);
        # cast per batch on the device so RAM never holds a float32 full matrix
        features = torch.from_numpy(np.ascontiguousarray(X.to_numpy()))
        target = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
        n = len(target)
        for _ in range(p["epochs"]):
            perm = torch.randperm(n)
            for start in range(0, n, p["batch_size"]):
                idx = perm[start : start + p["batch_size"]]
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(
                    model(features[idx].to(device).float()), target[idx].to(device)
                )
                loss.backward()
                optimizer.step()
        self._model = model.eval().cpu()  # stored on CPU: picklable, portable

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self._model.to(device)
        features = torch.from_numpy(np.ascontiguousarray(X.to_numpy()))
        outs = []
        with torch.no_grad():
            for start in range(0, len(features), 65536):
                outs.append(model(features[start : start + 65536].to(device).float()).cpu())
        self._model = model.cpu()
        return torch.cat(outs).squeeze(1).numpy()

    def save(self, path: Path) -> None:
        joblib.dump({"model": self._model, "params": self._params}, path)

    def load(self, path: Path) -> None:
        state = joblib.load(path)
        self._model, self._params = state["model"], state["params"]
