"""Cycle 1 — ingest & snapshot (DESIGN.md block 1).

The pipeline's scoped training-data pull: files live in the shared download
folder under `<version>/<snapshot>/`, and the exact bytes are frozen into the
CAS store. A snapshot folder is named for the day its files were downloaded, so
a missing folder is only downloaded when it is today's: a past-dated folder must
never hold bytes from a later day (ADR-0007). The live file is `serve.py`'s.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
from pydantic import field_validator

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult

DATASETS = {
    "raw_train": "train.parquet",
    "raw_validation": "validation.parquet",
}


def fetch(version: str, snapshot: str, filename: str, download_dir: Path) -> Path:
    """Return the local path of <version>/<snapshot>/<filename>. Downloads only
    when the snapshot is today's UTC date; numerapi writes `<path>.temp` and
    renames at the end, so the final path is never partial."""
    dest = download_dir / version / snapshot / filename
    if dest.exists():
        return dest
    today = datetime.now(UTC).date().isoformat()
    if snapshot != today:
        raise FileNotFoundError(
            f"{dest} does not exist and snapshot {snapshot} is not today ({today}): copy the "
            f"files into {dest.parent}/ or set ingest.snapshot to today's date to download"
        )
    from numerapi import NumerAPI

    dest.parent.mkdir(parents=True, exist_ok=True)
    NumerAPI().download_dataset(f"{version}/{filename}", str(dest))
    return dest


class IngestConfig(StrictConfig):
    version: str = "v6.0"  # v5.x training files retired (2026-10)
    snapshot: str  # YYYY-MM-DD: the folder <version>/<snapshot>/ (the files' download date)

    @field_validator("snapshot")
    @classmethod
    def _is_date(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError(f"snapshot must be YYYY-MM-DD, got {v!r}")
        date.fromisoformat(v)
        return v


class IngestStep(Step):
    name = "ingest"
    inputs: list[str] = []
    outputs = ["raw_train", "raw_validation", "features_meta"]
    config_model = IngestConfig

    def run(self, ctx):
        cfg = IngestConfig.model_validate(ctx.config.get(self.name, {}))
        version, snapshot = cfg.version, cfg.snapshot
        download_dir = Path(ctx.config.get("download_dir", "/workspace/raw_data/numerai"))

        outs = {}
        for key, fname in DATASETS.items():
            path = fetch(version, snapshot, fname, download_dir)
            outs[key] = ctx.put(path, key, ext="parquet")  # exact bytes, no re-encode
            rows = pl.scan_parquet(path).select(pl.len()).collect().item()
            ctx.log_meta(
                key,
                {"file": f"{version}/{snapshot}/{fname}", "bytes": path.stat().st_size, "rows": rows},
            )

        features_path = fetch(version, snapshot, "features.json", download_dir)
        outs["features_meta"] = ctx.put(json.loads(features_path.read_text()), "features_meta", ext="json")
        ctx.log_meta("dataset", {"name": "numerai", "version": version, "snapshot": snapshot})
        return StepResult(outputs=outs)
