"""Serving — predict (ADR-0007). The last step of every serve list: scores a frame
with a pinned bundle's model and fitted transformer. `mlpipe predict` seeds the
bundle and both artifacts it names, so `predictions` gets a manifest whose
lineage reaches the bundle and, through it, the training run.

A pickled backend names its class, not its source, and serving retrains nothing:
a backend edit after training would not move this step's signature. So `run`
refuses a model whose recorded backend code differs from the code now on disk.
A cache hit returns what the training-time code produced.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import polars as pl

from mlpipe.core.interfaces import Step, StepResult
from mlpipe.core.signature import module_code_hash


def default_prepare(frame: pl.DataFrame, bundle: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(features in bundle order, id columns to carry into the output)."""
    return frame.select(bundle["features"]), frame.select(bundle.get("id_columns", []))


class PredictStep(Step):
    name = "predict"
    inputs = ["model_bundle", "model", "fitted_transformer", "input_frame"]
    outputs = ["predictions"]

    def __init__(self, frame_key: str = "input_frame", prepare: Callable | None = None) -> None:
        # `prepare(frame, bundle) -> (features, ids)`: a host's serving fill. Its
        # module enters the signature (code_deps), so it must name its own code: a
        # lambda, a nested function, a partial (module `functools`) or a __main__
        # script would hash the wrong file or none.
        if prepare is not None and not (
            inspect.isfunction(prepare)
            and prepare.__qualname__ == prepare.__name__ != "<lambda>"
            and prepare.__module__ != "__main__"
        ):
            raise TypeError(f"prepare must be a plain module-level function, got {prepare!r}")
        self.frame_key = frame_key
        self.prepare = prepare
        self.inputs = ["model_bundle", "model", "fitted_transformer", frame_key]

    def code_deps(self, cfg):
        return [self.prepare.__module__] if self.prepare else []

    def run(self, ctx):
        bundle = ctx.get("model_bundle")
        hashes = ctx.input_hashes(self.inputs)
        for key, field in (("model", "model_hash"), ("fitted_transformer", "transformer_hash")):
            if hashes[key] != bundle[field]:
                raise ValueError(
                    f"seeded {key} {hashes[key][:12]} is not the bundle's {field} {bundle[field][:12]}"
                )
        transformer = ctx.get("fitted_transformer")
        stored = ctx.get("model")
        model = stored["full"]
        if stored.get("backend_code") != module_code_hash(type(model).__module__):
            raise RuntimeError("backend code changed since this bundle was trained; retrain")

        feats, ids = (self.prepare or default_prepare)(ctx.get(self.frame_key), bundle)
        matrix = transformer.transform(feats.to_numpy())
        preds = model.predict(pl.from_numpy(matrix, schema=bundle["features"]))
        result = ids.with_columns(pl.Series("prediction", preds))
        art = ctx.put(result, "predictions")
        ctx.log_meta("rows", result.height)
        ctx.log_meta("bundle", hashes["model_bundle"])
        ctx.log_meta("model_name", bundle.get("model_name"))
        ctx.log_meta("plan_hash", bundle.get("plan_hash"))
        return StepResult(outputs={"predictions": art})
