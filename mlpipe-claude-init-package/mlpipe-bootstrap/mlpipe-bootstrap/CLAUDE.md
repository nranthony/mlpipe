# CLAUDE.md — operating instructions for agents in this repo

## What this repo is
A local, modular MLOps pipeline (see DESIGN.md — read it first, it is authoritative).
Work proceeds in cycles defined by `goals/cycleN_*.md`. One cycle at a time.

## Session-start ritual
1. Read DESIGN.md if not already in context.
2. Check `goals/STATUS.md` for the current cycle and its state.
3. If `.copier-answers.yml` exists, compare its pinned template version against the
   template repo's latest tag; if behind, propose `/template-sync` before new work.

## Hard rules
- Steps interact with the world ONLY through `RunContext`. No direct file I/O, no
  tracker SDK calls, no model-library imports in step modules outside the backends.
- Polars DataFrames cross step boundaries; Parquet at rest; DuckDB allowed inside a
  step only. Never let pandas or DuckDB types leak across a boundary.
- `src/mlpipe/core/` stays free of Numerai-specific imports. Domain code lives in
  `src/mlpipe/steps/` and `src/mlpipe/backends/`.
- Core line budget: 400 target, 700 ceiling. If a change would exceed it, stop and
  write an ADR proposing the trade instead of committing the change.
- ModelBackend method signatures may only mention our types (polars, numpy, Path,
  FoldPlan). A `torch.Tensor` in a port signature is a bug.
- Config changes go through pydantic schemas. If a new option isn't representable in
  the schema, extend the schema first; never bypass validation.
- Every run goes through the CLI (`mlpipe run ...`), never by importing internals —
  this is what guarantees a manifest exists for every execution.
- GPU budget is 12 GB VRAM. Flag any model config that would exceed it before running.
- No hosted/cloud storage of run data. Local only.

## Definition of done for a cycle
- Acceptance tests in the goal file pass (`pytest tests/ -k cycleN`).
- The block runs via CLI with upstream served from cache (verify a cache hit in the
  manifest log).
- `goals/STATUS.md` updated; a short work-log entry appended there (what changed, what
  was measured, what's pinned in baselines.yaml if anything was promoted).

## When comparing experiments
Cite the fold-plan hash for both runs. If plan hashes differ, the comparison is
invalid — say so instead of reporting a winner.

## When blocked
Sandbox blocks (installs, env config, credentials) are human-only boundaries: stop and
prompt, never route around. New dependencies require a short ADR (status: Proposed).
