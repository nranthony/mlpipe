"""Entry-point discovery: host packages register pipelines and model backends
without mlpipe knowing their names (stdlib `importlib.metadata`).

    [project.entry-points."mlpipe.pipelines"]
    myproj = "myproj.pipeline"             # module with build_steps() and DEFAULT_CONFIG

    [project.entry-points."mlpipe.backends"]
    xgboost = "myproj.backends:XGBBackend"  # a ModelBackend class

A pipeline module may also define `build_serve_steps() -> list[Step]` (ADR-0007),
run by `mlpipe predict` with the pinned bundle seeded. Its first step is the
source (no inputs, exactly one output; what it reads is named in its config, e.g.
a round), its last a `mlpipe.steps.predict.PredictStep` (a host's feature fill
goes in as `PredictStep(prepare=...)`). Both lists share DEFAULT_CONFIG. `run`
and `tune` call it for step names too, so it must import cheaply (heavy clients
imported inside functions). Without it, predict scores `--input` with
`PredictStep()`. The old `serve_input` and `prepare` hooks are rejected.

Entry points live in installed package metadata, so a new one appears only after
the host package is (re)installed. mlpipe's own pipelines and backends are
builtins here, which keeps them working in an environment synced before this
existed.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from types import ModuleType

PIPELINES = "mlpipe.pipelines"
BACKENDS = "mlpipe.backends"

BUILTIN_PIPELINES = {
    "demo": "mlpipe.steps.demo",
    "demo-tabular": "mlpipe.steps.demo_tabular",
    # "-example": a host's own Numerai pipeline keeps the plain name (ADR-0009)
    "numerai-example": "mlpipe.examples.numerai.pipeline",
    "titanic-example": "mlpipe.examples.titanic.pipeline",
}


def registered(group: str, builtins: dict[str, str]) -> dict[str, str]:
    """name -> import target (`module` or `module:attr`): builtins plus installed
    entry points. One name claimed by two different targets is an error, never a
    silent override."""
    found = dict(builtins)
    for ep in entry_points(group=group):
        if found.setdefault(ep.name, ep.value) != ep.value:
            raise ValueError(
                f"{group}: {ep.name!r} is registered twice ({found[ep.name]} and {ep.value})"
            )
    return found


def pipeline_module(name: str) -> ModuleType:
    targets = registered(PIPELINES, BUILTIN_PIPELINES)
    if name not in targets:
        raise KeyError(f"unknown pipeline {name!r}; registered: {sorted(targets)}")
    return importlib.import_module(targets[name])
