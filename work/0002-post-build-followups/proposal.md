# Proposal: post-build follow-ups and the choice of next direction

- Status: In review — **gap 1 fixed 2026-08-24** (see "Resolution" below); gaps 2 and 3
  still open, direction for the rest still awaiting nanthony
- Author: Claude (agent, windows-ai-sandbox)
- Date: 2026-08-22 (session `numerai-mlpipe-testing`)

## Summary

The build is complete (all cycles 0-10, `goals/STATUS.md`) and the repo is at rest:
working tree clean, `pytest tests/ -q` → **53 passed in 15s** (re-verified 2026-08-22),
store 11 GB, `work/` empty. Three known gaps were carried out of the build; each was
re-confirmed against the source in this session rather than taken from the work log.
This item records those gaps and the four candidate directions, so the decision
survives the session.

## Motivation — the three confirmed gaps

1. **Step signatures hash only the step's own module.**
   `src/mlpipe/core/signature.py:39` computes `code_hash` as
   `sha256(inspect.getfile(type(step)).read_bytes())` — the file defining the step
   class, nothing else. Editing `src/mlpipe/backends/*.py` or any helper a step
   imports does **not** change the signature, so the pipeline serves a stale cached
   artifact for changed code. The cycle-8 torch OOM fix (commit `e1c56c5`, a
   backend-only edit) is exactly the shape of change this misses. This is the gap
   that undercuts the reproducibility guarantee the whole CAS + manifest design
   rests on.

2. **`mlpipe tune` overrides below `train` are inert.**
   `src/mlpipe/cli.py:94` hardcodes `from_step="train", to_step="evaluate"` for every
   trial. A study whose search space touches `clean.*` or `features.*` accepts those
   overrides and silently ignores them — the trial reports a score for a config it
   did not actually run.

3. **Registration has no promotion gate.**
   `src/mlpipe/steps/register.py:47` calls `ctx.register_model(...)` on every fresh
   run of the block, so any model that reaches `register` becomes a registry version.
   There is no explicit promote/approve step separating "trained" from "registered".

## Proposal — four candidate directions

1. **Fix signature scope (recommended first).** Widen step code hashing beyond the
   step's own module — hash the step module plus the backend/helper modules it
   declares. Small and contained; restores the core reproducibility claim. Touches
   `core/signature.py`, the DESIGN.md §2 contract wording, and needs a cache-bust
   story for the existing 11 GB store (existing hashes change → mass re-run, or an
   explicit version bump on the signature payload). **Rationale for going first:**
   every experiment run before this lands inherits a cache that cannot be fully
   trusted.
2. **Modelling iteration.** The actual Numerai goal: larger Optuna study, full
   feature set instead of medium, more trees; try to beat `mean_corr` 0.02793 on
   plan `48de731b`. Long runs belong in background shells (see `AGENTS.local.md`).
3. **Template extraction.** Begin the Copier template this repo was built to become,
   working from the zones in `TEMPLATE_OWNERSHIP.md`.
4. **Clear the small follow-ups.** Gaps 2 and 3 above: wire `tune`'s `--from` out of
   config rather than hardcoding it, and add an explicit promotion gate in place of
   auto-register.

A fifth item was deferred during the build and is not proposed here, only recorded:
the **ZenML orchestrator-exit-door spike** was skipped because the install is a
sandbox human boundary. It stays skipped until the exit door actually needs proving.

## Resolution — gap 1 (2026-08-24)

Fixed by widening the code hash from one file to a **source-file closure**: the step's
own module, plus everything under `mlpipe.steps.*` / `mlpipe.backends.*` it can reach by
import, plus lazily-resolved modules a step declares through the new `Step.code_deps`
hook. `TrainStep.code_deps` returns the registry module for the configured `model.kind`,
which is what the static parser cannot see. Modules are located with `find_spec` and read
with `ast` — nothing is imported, so signatures stay computable on a cache hit without
loading lightgbm or torch (test-enforced).

`mlpipe.core.*` is excluded by design, recorded in DESIGN.md §1: core is the
orchestrator, not the computation, and including it would invalidate every artifact on
any core edit. The residual risk is a core serialization change altering output bytes
without changing a signature — bounded by tests and by `git_commit` in every manifest.
This exclusion is the one part worth revisiting if it ever bites.

Verified on the demo pipeline: signatures changed (`make_data` `e94c3661` → `ab93bc8a`),
so the first run recomputed rather than cache-hitting, but the output content hashes were
byte-identical (`5109abe5`, `b3cee3df`) and the store stayed at 37 files. Second run
cache-hit under the new signature. **The invalidation costs compute, not disk.**

Answering this item's own open question about the 11 GB store: no re-run was triggered.
The next real `mlpipe run` recomputes ingest→register (~1.5-2h, dominated by the ~64-min
LightGBM fit) and will re-write byte-identical artifacts to the same hashes for every
step whose behaviour did not actually change.

**Left tight:** core is now 697/700 lines. The change fits under the ceiling, so no ADR
was required, but the next core edit breaches it. The obvious trade to propose then is
moving `core/mlflow_tracker.py` (55 lines) out of core — it is an adapter, not core
orchestration, and core's own rules already push tracker specifics to the edge.

## Open questions

- Which direction (or ordering) does nanthony want? Asked at the end of the
  2026-08-22 session; not answered.
- If gap 1 is fixed: re-run the 11 GB store from scratch, or bump a signature
  version and accept a one-time full invalidation? Both mean the same recompute;
  the difference is whether old manifests stay interpretable.
- Do gaps 1 and 3 change a DESIGN.md §0 decision (→ ADR) or only correct a §2
  contract (→ edit DESIGN.md directly)? ADR-0002's rejection sets the calibration:
  consequences of existing acceptance criteria go in DESIGN.md; changes of
  direction get an ADR. Gap 1 reads as a **correction of the stated signature
  contract**, gap 3 as a **change of direction** on when registration happens.

## Alternatives

- **Leave gap 1 and rely on discipline** (always `--from` the edited step): rejected
  as a standing posture — it makes correctness depend on the operator remembering,
  which is what the signature mechanism exists to remove. Acceptable only as a
  stopgap while modelling work proceeds, and only if recorded in `AGENTS.local.md`.
- **Fix all three gaps in one pass before any modelling:** viable, but gaps 2 and 3
  are cheap and independent, so they need not block; gap 1 is the only one whose
  cost grows with every run made before it lands.
