"""Cycle 1 — ingest & snapshot (DESIGN.md block 1).

The pipeline's one scoped network pull: numerapi fetches into the shared
download cache (skipped when the file is already there), and the exact
downloaded bytes are frozen into the CAS store. Nothing else touches the API.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from mlpipe.core.interfaces import Step, StepResult

DATASETS = {
    "raw_train": "train.parquet",
    "raw_validation": "validation.parquet",
    "raw_live": "live.parquet",
}


def fetch(version: str, filename: str, download_dir: Path) -> Path:
    """Return the local path of <version>/<filename>, downloading if absent."""
    dest = download_dir / version / filename
    if not dest.exists():
        from numerapi import NumerAPI

        NumerAPI().download_dataset(f"{version}/{filename}", str(dest))
    return dest


class IngestStep(Step):
    name = "ingest"
    inputs: list[str] = []
    outputs = ["raw_train", "raw_validation", "raw_live", "features_meta"]

    def run(self, ctx):
        cfg = ctx.config.get(self.name, {})
        version = cfg.get("version", "v5.2")
        download_dir = Path(ctx.config.get("download_dir", "/workspace/raw_data/numerai"))

        outs = {}
        for key, fname in DATASETS.items():
            path = fetch(version, fname, download_dir)
            outs[key] = ctx.put(path, key, ext="parquet")  # exact bytes, no re-encode
            rows = pl.scan_parquet(path).select(pl.len()).collect().item()
            ctx.log_meta(key, {"file": f"{version}/{fname}", "bytes": path.stat().st_size, "rows": rows})

        features_path = fetch(version, "features.json", download_dir)
        outs["features_meta"] = ctx.put(json.loads(features_path.read_text()), "features_meta", ext="json")
        ctx.log_meta("dataset", {"name": "numerai", "version": version})
        return StepResult(outputs=outs)
