"""The real Numerai pipeline, assembled cycle by cycle.

    mlpipe run --pipeline numerai
"""

from __future__ import annotations

from mlpipe.core.interfaces import Step
from mlpipe.steps.ingest import IngestStep

DEFAULT_CONFIG = {
    "download_dir": "/workspace/raw_data/numerai",  # shared profile-wide cache
    "ingest": {"version": "v5.2"},
    "seed": 42,
}


def build_steps() -> list[Step]:
    return [IngestStep()]
