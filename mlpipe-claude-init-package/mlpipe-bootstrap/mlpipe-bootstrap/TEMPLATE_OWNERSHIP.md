# Ownership map (governs template extraction and future /template-sync)

Zones (see DESIGN.md §7). Declared now so extraction later is a git mv, not a refactor.

## template-owned  (updates flow in from the template; repos should not edit)
- src/mlpipe/core/**
- src/mlpipe/cli.py
- CLAUDE.md, DESIGN.md, TEMPLATE_OWNERSHIP.md
- tests/test_cycle0_spine.py
- .claude/commands/**  (including template-sync, once it exists)

## skeleton  (template provides once at generation; repo owns thereafter)
- src/mlpipe/steps/**
- src/mlpipe/backends/**
- configs/**
- goals/**

## repo-owned  (template never touches)
- data/**, store/**, manifests/**, mlruns/**
- baselines.yaml
- experiments/**, notebooks/**

Conflict policy for agents: template-owned → take the template's side unless tests
break; skeleton → preserve local, port patterns manually if the template changelog
says they matter; repo-owned → never modified by sync.
