"""Top-level pipeline config (work/0004 B): the keys that belong to no step.

A config file holds one subtree per step (validated by that step's
`config_model`) plus the orchestrator-level keys below. Anything else at the
top level is a typo and dies here, before any step runs (DESIGN.md §5).

Top-level keys are deliberately outside every step signature, so nothing that
changes a step's output may live here. That is why there is no top-level
`seed`: a seed must sit in the subtree of the step it affects (e.g.
`train.model.params`), where it is hashed.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrackerConfig(_Strict):
    kind: Literal["none", "mlflow"] = "none"
    uri: str = "sqlite:///mlruns/mlflow.db"
    experiment: str = "mlpipe"


class IntSpace(_Strict):
    type: Literal["int"]
    low: int
    high: int


class FloatSpace(_Strict):
    type: Literal["float"]
    low: float
    high: float
    log: bool = False


class TuneConfig(_Strict):
    study: str
    storage: str
    metric: str  # a metric the `step` below logs, e.g. evaluate's mean_corr
    direction: Literal["maximize", "minimize"] = "maximize"
    step: str = "evaluate"  # trials run up to this step and read its metrics
    n_trials: int = Field(default=5, gt=0)
    space: dict[str, Annotated[IntSpace | FloatSpace, Field(discriminator="type")]]


class PipelineConfig(_Strict):
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    tune: TuneConfig | None = None
    # Machine-local input cache: a location, not content, so it is outside every
    # signature by design (moving it must not bust the ingest cache).
    download_dir: str | None = None


def validate_config(config: dict, step_names: list[str]) -> PipelineConfig:
    """Validate the non-step keys; unknown top-level keys are rejected."""
    return PipelineConfig.model_validate(
        {k: v for k, v in config.items() if k not in step_names}
    )
