"""LocalCasStore: content-addressed artifact store (DESIGN.md §1).

sha256 addressing, store/<hash[:2]>/<hash>.<ext>, atomic writes (temp + rename),
dedup on save. Codecs: polars⇄parquet, joblib, canonical JSON.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import joblib
import polars as pl

from mlpipe.core.interfaces import Artifact
from mlpipe.core.signature import canonical_json, sha256_bytes


class LocalCasStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _serialize(self, obj: Any, ext: str) -> bytes:
        buf = io.BytesIO()
        if ext == "parquet":
            obj.write_parquet(buf)
        elif ext == "joblib":
            joblib.dump(obj, buf)
        elif ext == "json":
            buf.write(canonical_json(obj).encode())
        else:
            raise ValueError(f"unknown artifact ext: {ext!r}")
        return buf.getvalue()

    def save(self, obj: Any, key: str, *, ext: str = "parquet") -> Artifact:
        data = self._serialize(obj, ext)
        h = sha256_bytes(data)
        path = self.root / h[:2] / f"{h}.{ext}"
        if not path.exists():  # dedup: identical bytes are stored once
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
            tmp.write_bytes(data)
            tmp.replace(path)
        return Artifact(key=key, content_hash=h, path=path, meta={"ext": ext})

    def path_of(self, content_hash: str) -> Path:
        shard = self.root / content_hash[:2]
        matches = list(shard.glob(f"{content_hash}.*")) if shard.exists() else []
        if not matches:
            raise FileNotFoundError(f"artifact not in store: {content_hash}")
        return matches[0]

    def exists(self, content_hash: str) -> bool:
        try:
            self.path_of(content_hash)
            return True
        except FileNotFoundError:
            return False

    def load(self, content_hash: str) -> Any:
        path = self.path_of(content_hash)
        ext = path.suffix.lstrip(".")
        if ext == "parquet":
            return pl.read_parquet(path)
        if ext == "joblib":
            return joblib.load(path)
        if ext == "json":
            return json.loads(path.read_text())
        raise ValueError(f"unknown artifact ext: {ext!r}")
