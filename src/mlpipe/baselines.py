"""baselines.yaml: the promoted pins (ADR-0006), keyed `{pipeline: {alias: pin}}`.

Only `mlpipe promote` writes it. One slot per (pipeline, alias), so promoting
`staging`, or a second pipeline, leaves `champion` alone. A pin is a dict with a
`bundle_hash`; anything else in the file (the pre-ADR-0006 `promoted:` block) is
kept as written and pins nothing.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load(path: Path) -> dict:
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def pins(data: dict) -> dict[tuple[str, str], dict]:
    """(pipeline, alias) -> pin, for every well-formed pin in the file."""
    return {
        (pipeline, alias): pin
        for pipeline, aliases in data.items() if isinstance(aliases, dict)
        for alias, pin in aliases.items() if isinstance(pin, dict) and pin.get("bundle_hash")
    }


def pinned(path: Path, pipeline: str, alias: str) -> dict | None:
    return pins(load(path)).get((pipeline, alias))


def write_pin(path: Path, pipeline: str, alias: str, pin: dict) -> None:
    data = load(path)
    aliases = data.get(pipeline)
    data[pipeline] = {**(aliases if isinstance(aliases, dict) else {}), alias: pin}
    path.write_text(yaml.safe_dump(data, sort_keys=True))
