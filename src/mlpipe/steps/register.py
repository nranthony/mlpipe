"""Cycle 10 — register & package (DESIGN.md block 8/10).

The registered unit is a BUNDLE: model + fitted transformer + feature list +
plan hash + the serving recipe an upstream step emits — everything serving
needs, nothing it doesn't. The bundle is a store artifact. Registering it in
MLflow happens only on `mlpipe promote` (ADR-0006), never on every run.
"""

from __future__ import annotations

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult


class RegisterConfig(StrictConfig):
    model_name: str = "model"
    # Columns `predict` carries from the scored frame into its output (the
    # default PredictStep prepare reads them from the bundle), e.g. a row key.
    id_columns: list[str] = []


class RegisterStep(Step):
    name = "register"
    inputs = ["model", "fitted_transformer", "feature_table", "fold_plan", "eval_report"]
    outputs = ["model_bundle"]
    config_model = RegisterConfig

    def __init__(self, recipe: str | None = None) -> None:
        # The key of a JSON artifact holding serving instructions (e.g. a
        # cleaner's fill and the source group column). It is a declared input, so
        # a recipe change moves this step's signature; reading another step's
        # config would leave register cached with a stale bundle.
        self.recipe = recipe
        if recipe:
            self.inputs = [*type(self).inputs, recipe]

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
            "model_name": cfg.model_name,
            "model_hash": hashes["model"],
            "transformer_hash": hashes["fitted_transformer"],
            "features": features,
            "plan_hash": report["plan_hash"],
            "target": report["target"],
            "metrics": report["metrics"],
        }
        if cfg.id_columns:
            bundle["id_columns"] = cfg.id_columns
        if self.recipe:
            bundle["recipe"] = ctx.get(self.recipe)
        art = ctx.put(bundle, "model_bundle", ext="json")
        ctx.log_meta("bundle", {"name": cfg.model_name, "plan_hash": report["plan_hash"]})
        return StepResult(outputs={"model_bundle": art})
