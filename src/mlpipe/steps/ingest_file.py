"""Generic ingest of local files (CSV or parquet) under the shared download folder.

An ingest step has no inputs, so its signature is its config alone: a file edited
in place would be served from a stale cache. Each file therefore pins its sha256 in
config. A mismatch refuses to run and prints the actual hash, and a new file means
a new hash, so a new signature. CSV is parsed with every row read for type
inference (deterministic given the bytes) and stored as parquet.

    ingest:
      files:
        raw_train: {path: titanic/train.csv, sha256: 7d118fef...}
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
from pydantic import field_validator

from mlpipe.core.config import StrictConfig
from mlpipe.core.interfaces import Step, StepResult


class FileSpec(StrictConfig):
    path: str  # relative to the top-level `download_dir`
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _is_hex(cls, v: str) -> str:
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return v


class IngestFileConfig(StrictConfig):
    files: dict[str, FileSpec] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pl.DataFrame:
    if path.suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=None)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"{path}: only .csv and .parquet are ingested")


class IngestFileStep(Step):
    name = "ingest"
    inputs: list[str] = []
    config_model = IngestFileConfig

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)  # one per configured file

    def run(self, ctx):
        cfg = IngestFileConfig.model_validate(ctx.config.get(self.name, {}))
        if set(cfg.files) != set(self.outputs):
            raise ValueError(f"ingest.files must name exactly {sorted(self.outputs)}, "
                             f"got {sorted(cfg.files)}")
        download_dir = Path(ctx.config.get("download_dir", "../raw_data"))
        outs = {}
        for key, spec in cfg.files.items():
            path = download_dir / spec.path
            actual = sha256_file(path)
            if actual != spec.sha256:
                raise ValueError(f"{path}: sha256 is {actual}, config pins {spec.sha256}; "
                                 "if the new file is intended, pin its hash")
            df = read_table(path)
            outs[key] = ctx.put(df, key)
            ctx.log_meta(key, {"file": spec.path, "sha256": actual, "rows": df.height,
                               "columns": df.width})
        return StepResult(outputs=outs)
