# Proposal: Numerai example as a real-data harness — cheap separation fixes

- Status: **Done 2026-10-26**; `uv lock` run host-side, `baselines.yaml` deleted
  (approved). Built 2026-10-26 (see goals/STATUS.md). Decision is in
  [ADR-0009](../../../docs/adr/0009-numerai-example-is-a-real-data-harness.md) (Accepted
  2026-10-26, task 2 option (a)).
  - Open before archiving: `uv lock` (human), and task 7's deletion (proposed).
  - Task 5 differed from the plan: `demo` has no train or register step, so the
    tests moved to a new builtin, `demo-tabular`.
  - Open question 2 was answered yes: cycles 5/7 got notes, not rewrites.
- Author: Claude, from [work/0011](../../0011-numerai-separation-audit/handoff.md)
  findings A, B (Low), C and D6

## Summary

These are the fixes the work/0011 audit rated cheap, which ADR-0009 keeps. None of them
changes what the example computes. The "shrink the example" fix waits for ADR-0009's
trigger.

## Tasks

1. **numerapi gets its own extra.**
   - The edit: `data = ["pandera[polars]", "duckdb"]` plus a new
     `numerai = ["numerapi"]`. Only `examples/numerai/` (`ingest.py`, `serve.py`)
     imports numerapi.
   - It is a manifest edit, so `uv lock` is a human step, and so is the host sandbox's
     `uv sync --frozen --extra numerai`.
   - Instructions that say `--extra data` for Numerai change to `--extra numerai`.
   - No new package: numerapi is already locked.
2. **The example's pipeline name stops colliding with hosts.** Today `"numerai"` is in
   `plugins.BUILTIN_PIPELINES`, so every host install lists it. Options:
   - (a) Rename it to `numerai-example`. Cost: the manifest log lives at
     `manifests/<pipeline>/`, so the cache history stays under `manifests/numerai/`
     and the first run under the new name recomputes everything (~30 min on v6.0)
     unless the folder is copied. Tests that pass `--pipeline numerai` change too.
   - (b) Leave the wheel's builtin out unless the module is importable from a source
     checkout. Fragile.
   - (c) Leave the example out of the wheel (`[tool.hatch.build.targets.wheel]`
     `exclude`) and register it only in this repo. Needs a check that the editable
     install still finds it.

   Leaning (a) plus copying the manifests folder (a human step, since it is
   gitignored run data). This is the one decision this item needs from you.
3. **Harness defaults say what they are.** The MLflow `experiment: "numerai"` becomes
   `"numerai-harness"`, and the Optuna `study: "lgbm-medium"` becomes
   `"harness-lgbm-medium"`. Tracker and tune settings are in no step signature, so no
   cache is lost (check before changing).
4. **Scrub Numerai and era wording from generic modules.** This is work/0011 finding B,
   Low row:
   - `core/interfaces.py`, `core/config.py`;
   - `steps/train.py`, `steps/register.py`, `steps/groupcv.py`,
     `steps/tabular_eval.py`, `steps/tabular_metrics.py`, `steps/demo.py`;
   - `backends/torch_backend.py`.

   Comment-only edits still change code hashes, so expect a one-time re-run of the
   steps whose closures include them. A real-data rerun gets a log line in STATUS.
5. **Port the generic tests to the demo pipeline.** `tests/test_cycle9_10_tune_register.py`
   exercises generic register, predict and lineage via `--pipeline numerai` and
   `numerai_ticker`. Those parts move to `demo`; the Numerai-specific parts stay.
6. **Fix `TEMPLATE_OWNERSHIP.md`** (found by the wrap-up):
   - its library zone omits `src/mlpipe/steps/**` (generic steps, now including
     `predict.py`);
   - "examples/numerai/ once moved" is done, so drop "once moved".
7. **The legacy `baselines.yaml` `promoted:` block.** It predates ADR-0006 and pins a
   Numerai run the registry no longer holds. Only `mlpipe promote` writes that file,
   and removing the block is a deletion, so it is proposed to you, not done.
8. **List the known divergences in `examples/numerai/README` or a module docstring**:
   corr implementation, sharpe `ddof`, target name and CV fold maths, each with a
   pointer to the numerai repo's version. Per ADR-0009 decision 3 they are recorded,
   not fixed.

## Out of scope

- Shrinking the example, or replacing ADR-0007 decisions 4-7: that waits for ADR-0009's
  trigger.
- Feature-selection config and moving `cvplan` (work/0011 "one spec" item). That is a
  library API change and gets its own item.
- Anything in the numerai repo.

## Open questions

- Task 2: which option, and should the manifests folder be copied to keep the cache?
- Task 4: should the scrub also touch `goals/cycle5/7` acceptance wording (work/0011
  D6), or leave closed cycles alone?
