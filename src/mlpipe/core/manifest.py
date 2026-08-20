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
        """Manifests from the producer of content_hash back to raw inputs.
        Pass-through steps re-emit an input hash, so a hash can have several
        producers: for a manifest's inputs, the producer must come earlier."""
        done = [m for m in self.entries() if m["status"] in ("success", "cached")]
        producers: dict[str, list[int]] = {}
        for i, m in enumerate(done):
            for h in m["outputs"].values():
                producers.setdefault(h, []).append(i)
        chain: list[dict] = []
        seen: set[str] = set()
        frontier = [(content_hash, len(done))]
        while frontier:
            h, before = frontier.pop(0)
            earlier = [i for i in producers.get(h, []) if i < before]
            if not earlier:
                continue
            i = max(earlier)
            m = done[i]
            if m["manifest_id"] in seen:
                continue
            seen.add(m["manifest_id"])
            chain.append(m)
            frontier.extend((ih, i) for ih in m["inputs"].values())
        return chain
