"""LocalCasStore: content-addressed artifact store (DESIGN.md §1).

sha256 addressing, store/<hash[:2]>/<hash>.<ext>, atomic writes (temp + rename),
dedup on save. Codecs: polars⇄parquet, joblib, canonical JSON. A `Path` is
snapshotted by streaming: one pass copies and hashes it, so a multi-GB download
never sits in memory. It is copied, not hard-linked: a link would share the
inode with a source that may be rewritten in place, and the store is immutable.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import joblib
import polars as pl

from mlpipe.core.interfaces import Artifact
from mlpipe.core.signature import canonical_json, sha256_bytes

CHUNK = 8 << 20  # 8 MiB


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
        elif ext == "html":
            buf.write(obj.encode())
        else:
            raise ValueError(f"unknown artifact ext: {ext!r}")
        return buf.getvalue()

    def save(self, obj: Any, key: str, *, ext: str = "parquet") -> Artifact:
        if isinstance(obj, Path):  # snapshot a file's exact bytes (e.g. a download)
            return self._save_file(obj, key, ext)
        data = self._serialize(obj, ext)
        h = sha256_bytes(data)
        path = self.root / h[:2] / f"{h}.{ext}"
        if not path.exists():  # dedup: identical bytes are stored once
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
            tmp.write_bytes(data)
            tmp.replace(path)
        return Artifact(key=key, content_hash=h, path=path, meta={"ext": ext})

    def _save_file(self, src: Path, key: str, ext: str) -> Artifact:
        digest = hashlib.sha256()
        tmp = self.root / f".{key}.{os.getpid()}.tmp"  # same filesystem: rename is atomic
        try:
            with src.open("rb") as fin, tmp.open("wb") as fout:
                while chunk := fin.read(CHUNK):
                    digest.update(chunk)
                    fout.write(chunk)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        h = digest.hexdigest()
        path = self.root / h[:2] / f"{h}.{ext}"
        if path.exists():  # dedup
            tmp.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
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
        if ext == "html":
            return path.read_text()
        raise ValueError(f"unknown artifact ext: {ext!r}")
