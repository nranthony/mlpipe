"""Cycle 10 — register & package (DESIGN.md block 8/10).

The registered unit is a BUNDLE: model + fitted transformer + feature list +
cleaner recipe + plan hash — everything serving needs, nothing it doesn't.
The bundle is a store artifact; MLflow registry is a view pointing at it.
"""

from __future__ import annotations

from pydantic import BaseModel

from mlpipe.core.interfaces import Step, StepResult


class RegisterConfig(BaseModel):
    model_name: str = "numerai-model"


class RegisterStep(Step):
    name = "register"
    inputs = ["model", "fitted_transformer", "feature_table", "fold_plan", "eval_report"]
    outputs = ["model_bundle"]
    config_model = RegisterConfig

    def run(self, ctx):
        cfg = RegisterConfig.model_validate(ctx.config.get(self.name, {}))
        hashes = ctx.input_hashes(self.inputs)
        features = [
            c
            for c in ctx.get_lazy("feature_table").collect_schema().names()
            if c.startswith("feature_")
        ]
        report = ctx.get("eval_report")

        bundle = {
            "model_hash": hashes["model"],
            "transformer_hash": hashes["fitted_transformer"],
            "features": features,
            "plan_hash": report["plan_hash"],
            "target": report["target"],
            "metrics": report["metrics"],
            "cleaner": ctx.config.get("clean", {}).get("cleaner", {}),
            "model_config": ctx.config.get("train", {}).get("model", {}),
        }
        art = ctx.put(bundle, "model_bundle", ext="json")
        art.meta.update({"plan_hash": report["plan_hash"], "model_name": cfg.model_name})
        ctx.register_model(art, cfg.model_name)
        ctx.log_meta("registered", {"name": cfg.model_name, "bundle_hash": art.content_hash})
        return StepResult(outputs={"model_bundle": art})
