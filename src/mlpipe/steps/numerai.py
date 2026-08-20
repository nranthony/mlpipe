"""The real Numerai pipeline, assembled cycle by cycle.

    mlpipe run --pipeline numerai
"""

from __future__ import annotations

from mlpipe.core.interfaces import Step
from mlpipe.steps.clean import CleanStep
from mlpipe.steps.features import FeatureStep
from mlpipe.steps.ingest import IngestStep
from mlpipe.steps.validate import ValidateStep

DEFAULT_CONFIG = {
    "download_dir": "/workspace/raw_data/numerai",  # shared profile-wide cache
    "ingest": {"version": "v5.2"},
    "validate": {"row_tolerance": 0.2, "max_feature_null_frac": 0.0},
    "clean": {"cleaner": {"kind": "baseline", "feature_set": "medium"}},
    "features": {"transformer": {"kind": "passthrough"}},
    "seed": 42,
}


def build_steps() -> list[Step]:
    return [IngestStep(), ValidateStep(), CleanStep(), FeatureStep()]
