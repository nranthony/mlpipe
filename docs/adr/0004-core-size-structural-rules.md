# ADR-0004: Core size — structural rules replace the hard line ceiling

- Status: Accepted
- Date: 2026-10-19
- Deciders: nanthony + Claude (work/0004 finding D, work/0006)

## Context

DESIGN.md §0 set "custom minimal core (~400 lines target, hard ceiling 700)", and
AGENTS.md repeats it as a hard rule. No record says why. No test enforces it. What it
counts has also drifted: cycle 0 reported 400 with `interfaces.py` excluded, and later
entries counted all of `core/` (697 of 700). Since 2026-08-24 the rule has blocked
core edits on an accounting question (work/0004 D).

The goals behind the number:

1. An agent can read the whole engine in one pass.
2. The core stays small enough to copy into other repos.
3. Domain logic doesn't creep into the core.
4. A future orchestrator shim stays cheap.

Goal 2 lapses under ADR-0005, because the core is now installed, not copied. Goals 3
and 4 depend on what the core imports and exposes, not on its length.

## Decision

1. **Hard rules, enforced by tests:**
   - `mlpipe.core.*` imports only the stdlib, the base dependencies (polars, numpy,
     pydantic, pyyaml, joblib, typer) and other `mlpipe.core` modules. It never
     imports `mlpipe.steps`, `mlpipe.backends`, `mlpipe.trackers`, a model library,
     a tracker SDK or domain packages.
   - Steps reach the world only through the declared `RunContext` surface. The test
     for this lands with work/0004 finding E (work/0006 plan, step 10).
2. **Soft signal:** line count over the *orchestration* modules only (`context`,
   `manifest`, `pipeline`, `signature`, `store`). Port declarations
   (`interfaces.py`) and adapters don't count. Crossing ~600 lines needs one sentence
   of justification in the `goals/STATUS.md` entry for that change. It doesn't need
   an ADR.
3. Adapters such as `MlflowTracker` live outside `core/`, in `mlpipe.trackers`.

## Consequences

- Core edits are unblocked. Orchestration code is 511 lines today, and `core/` is 642
  after the tracker moves out.
- The protection that matters is now checked mechanically, not by counting.
- The new risk is quiet growth below the review threshold. The single-pass
  legibility goal is still the reviewer's call.

## Alternatives considered

- **Keep 700 and count only orchestration.** Recommended in work/0004. It fixes the
  accounting but still treats an arbitrary number as law.
- **Raise the ceiling.** It moves the same problem further out.
- **Drop all size guidance.** It loses the review prompt that keeps the engine
  readable.
