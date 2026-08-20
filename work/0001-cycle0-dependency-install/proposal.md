# Proposal: provision the cycle-0 dev environment (human step)

- Status: In review (awaiting human action)
- Author: Claude (agent, windows-ai-sandbox)

## Summary

Cycle 0 (`goals/cycle0_spine.md`) cannot start: the declared dependencies cannot be
materialized from inside the sandbox. No **new** dependencies are requested — only
those already declared in `pyproject.toml` (`polars`, `numpy`, `pydantic`, `pyyaml`,
`joblib`, `typer` + `dev` extra `pytest`, `ruff`).

## Motivation

Verified 2026-08-20 inside the sandbox:

- Package registries are closed and install commands are denied (policy, not retried).
- `uv run --offline` was permitted but resolution fails: the uv cache at
  `/root/.cache/uv` has **wheels** for numpy/pydantic/pyyaml/joblib/typer/pytest/
  hatchling, but **no polars wheel at all**, and its simple-index metadata cache is
  nearly empty (~12 packages), so offline resolution fails on the first lookup.
- `.venv/` exists but is empty (created by the failed attempt; safe to delete or reuse).

## Proposal

On the host (any one of these unblocks the cycle):

1. **Preferred:** with network available, in the repo root run
   `uv lock` then `uv sync --extra dev`, and commit `uv.lock`. Future agent runs can
   then use `uv sync --frozen` / `uv run pytest` offline.
2. Or temporarily open PyPI to the sandbox and tell the agent to sync.
3. Or seed `/root/.cache/uv` with current index metadata plus the missing wheels
   (at minimum polars) so `uv lock --offline` can resolve.

After any of these, the agent resumes cycle 0: implement `core/store.py`,
`core/manifest.py`, `core/context.py`, `core/signature.py`, `core/pipeline.py`,
`cli.py`, and `tests/test_cycle0_spine.py` per the goal file.

## Open questions

- Should `uv.lock` be committed now (recommended — makes every later cycle's env
  reproducible and offline-friendly)?

## Alternatives

- Implementing the spine dependency-free (pickle instead of joblib, no polars):
  rejected — polars round-tripping is acceptance test 1, and it would diverge from
  DESIGN.md §0 for no lasting benefit.
