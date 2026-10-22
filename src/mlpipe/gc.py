"""Store garbage report (work/0006 step 16): which store files no root reaches.

Reports only; deleting is a human step. Roots:
- the bundle pinned in baselines.yaml and every bundle a `promote` manifest names
  (ADR-0006), each with its full lineage;
- every artifact a manifest from the last `keep_days` days read or wrote, so the
  recent cache stays warm.
Kept JSON artifacts are scanned for store hashes they name (a bundle's model_hash
and transformer_hash). The store is shared, so every manifest log under the
manifests root counts, not just one pipeline's.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from mlpipe.core.manifest import ManifestLog

HASH = re.compile(r"[0-9a-f]{64}")


@dataclass
class GcReport:
    files: dict[str, Path]  # hash -> path, every artifact in the store
    kept: set[str]
    bundles: set[str]  # pinned + promoted model bundles
    window_manifests: int
    last_written: dict[str, str]  # hash -> "step date" of its latest producer
    other_files: list[Path] = field(default_factory=list)  # temp writes, foreign files
    notes: list[str] = field(default_factory=list)

    @property
    def unreachable(self) -> list[str]:
        return sorted(h for h in self.files if h not in self.kept)

    @property
    def missing(self) -> set[str]:
        """Hashes a root names that are absent from the store."""
        return self.kept - self.files.keys()


def _scan_store(root: Path) -> tuple[dict[str, Path], list[Path]]:
    files: dict[str, Path] = {}
    other: list[Path] = []
    for path in sorted(root.glob("*/*")):
        stem = path.name.split(".")[0]
        if path.is_file() and HASH.fullmatch(stem) and path.parent.name == stem[:2]:
            files[stem] = path
        elif path.is_file():
            other.append(path)
    return files, other


def _logs(root: Path) -> list[ManifestLog]:
    dirs = [root, *sorted(p for p in root.iterdir() if p.is_dir())] if root.exists() else []
    return [ManifestLog(d) for d in dirs if (d / "index.jsonl").exists()]


def _pinned_bundle(baselines: Path, notes: list[str]) -> str | None:
    if not baselines.exists():
        notes.append(f"no {baselines}: no pinned bundle")
        return None
    pinned = (yaml.safe_load(baselines.read_text()) or {}).get("promoted") or {}
    if not pinned.get("bundle_hash"):
        notes.append(f"{baselines} pins no bundle_hash (written before ADR-0006); it roots nothing")
    return pinned.get("bundle_hash")


def _when(m: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(m.get("ended_at") or m["started_at"])
    except (KeyError, TypeError, ValueError):
        return None


def _hashes_in(obj) -> set[str]:
    if isinstance(obj, dict):
        return set().union(*map(_hashes_in, obj.values()))
    if isinstance(obj, list):
        return set().union(*map(_hashes_in, obj))
    return {obj} if isinstance(obj, str) and HASH.fullmatch(obj) else set()


def collect(
    store_root: Path,
    manifests_root: Path,
    baselines: Path,
    keep_days: int,
    now: datetime | None = None,
) -> GcReport:
    files, other = _scan_store(store_root)
    notes: list[str] = []
    cutoff = (now or datetime.now(UTC)) - timedelta(days=keep_days)
    logs = _logs(manifests_root)
    if not logs:
        notes.append(f"no manifest logs under {manifests_root}")

    kept: set[str] = set()
    bundles: set[str] = set()
    last_written: dict[str, str] = {}
    window = 0
    pinned = _pinned_bundle(baselines, notes)
    if pinned:
        bundles.add(pinned)
    for log in logs:
        for m in log.entries():
            if m["step"] == "promote" and m["status"] == "promoted":
                bundles.update(m["inputs"].values())
            when = _when(m)
            for h in m["outputs"].values():
                last_written[h] = f"{m['step']} {when.date() if when else '?'}"
            if when is None or when >= cutoff:  # undated counts as recent: keep
                window += 1
                kept.update(m["inputs"].values(), m["outputs"].values())

    for bundle in bundles:
        kept.add(bundle)
        for log in logs:
            for m in log.lineage(bundle):
                kept.update(m["inputs"].values(), m["outputs"].values())

    frontier = [h for h in kept if h in files and files[h].suffix == ".json"]
    while frontier:
        # only store files count: a bundle's plan_hash is 64 hex but no artifact
        named = {h for h in _hashes_in(json.loads(files[frontier.pop()].read_text()))
                 if h in files} - kept
        kept |= named
        frontier += [h for h in named if h in files and files[h].suffix == ".json"]

    return GcReport(files, kept, bundles, window, last_written, other, notes)
