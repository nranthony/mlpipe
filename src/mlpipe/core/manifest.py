"""Run manifests: one append-only JSON file per step execution (DESIGN.md §1),
plus manifests/index.jsonl for fast signature lookup and lineage walking.
Losing the tracker must lose zero provenance — these files are the source of truth.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class ManifestLog:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"

    def write(self, record: dict) -> Path:
        record = dict(record)
        record["manifest_id"] = uuid.uuid4().hex
        path = self.root / f"{record['step']}_{record['manifest_id']}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True))
        with self.index.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return path

    def entries(self) -> list[dict]:
        if not self.index.exists():
            return []
        return [json.loads(line) for line in self.index.read_text().splitlines() if line]

    def find_cached(self, signature: str) -> dict | None:
        """Latest completed execution with this signature, if any."""
        for m in reversed(self.entries()):
            if m["signature"] == signature and m["status"] in ("success", "cached"):
                return m
        return None

    def latest_outputs(self) -> dict[str, str]:
        """Most recent artifact hash for every output key ever produced."""
        latest: dict[str, str] = {}
        for m in self.entries():
            if m["status"] in ("success", "cached"):
                latest.update(m["outputs"])
        return latest

    def lineage(self, content_hash: str) -> list[dict]:
        """Manifests from the producer of content_hash back to raw inputs."""
        producer: dict[str, dict] = {}
        for m in self.entries():
            if m["status"] in ("success", "cached"):
                for h in m["outputs"].values():
                    producer[h] = m
        chain: list[dict] = []
        frontier = [content_hash]
        while frontier:
            m = producer.get(frontier.pop(0))
            if m is None or any(c["manifest_id"] == m["manifest_id"] for c in chain):
                continue
            chain.append(m)
            frontier.extend(m["inputs"].values())
        return chain
