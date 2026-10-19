# Ownership map (library + future scaffolding template)

Revised by ADR-0005: the engine is a **library** that host repos install. A Copier
template, if built, carries scaffolding only. See DESIGN.md §7.

## library  (ships in the `mlpipe` distribution; host repos never edit it)
- src/mlpipe/core/**
- src/mlpipe/cli.py
- src/mlpipe/trackers/**, generic src/mlpipe/backends/**
- tests for the above (tests/test_cycle0_spine.py, tests/test_core_boundary.py,
  tests/test_signature_scope.py)

## template-owned  (future scaffolding template; updates flow in, repos should not edit)
- the managed sections of AGENTS.md; CLAUDE.md (thin @AGENTS.md stub)
- .claude/skills/** (repo procedures, e.g. /template-sync once it exists)
- .claude/settings.json

## skeleton  (template provides once at generation; repo owns thereafter)
- the host package's steps/ and backends/ (here: examples/numerai/ once moved)
- configs/**
- goals/**, DESIGN.md (project sections)
- work/README.md, docs/adr/0000-template.md

## repo-owned  (template and library never touch)
- data/**, store/**, manifests/**, mlruns/**
- baselines.yaml
- experiments/**, notebooks/**
- docs/adr/** (except the template), work/** (except README), AGENTS.local.md

Conflict policy for agents: library → change it here, release a tag, and host repos
bump the pin. template-owned → take the template's side unless tests break.
skeleton → preserve local, and port patterns manually if the template changelog says
they matter. repo-owned → never modified by sync.
