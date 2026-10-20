# Plan: mlpipe as a generic library (work/0006)

- Decisions: ADR-0004 (core size), ADR-0005 (library, Numerai as example), ADR-0006
  (promotion). Answers from nanthony, 2026-10-19:
  - mlpipe is non-Numerai work; Numerai stays as a test case;
  - this repo is the engine, with no new repo;
  - structural rules replace the line ceiling;
  - an explicit `promote` verb;
  - no external servers.
- Absorbs work/0002 (gaps 2, 3), work/0003 (era_slice), and work/0004 findings A–G.
  Those items archive when their steps here land.
- Every step ends with `pytest tests/ -q` green and `ruff check` clean, plus a
  STATUS.md log line.

## Phase 1 — unblock and correct

| # | Step | Source | State |
|---|---|---|---|
| 1 | ADR-0004/0005/0006; move `MlflowTracker` out of core; core boundary test | 0004 D | **done** 2026-10-19 (`3eded0f`); pytest green 2026-10-20 |
| 2 | `era_slice` smoke runs (clean step) + `--tag smoke` convention | 0003 | **done** 2026-10-20 |
| 3 | `--from` resolves upstream by current-config signatures, not "latest ever"; `tune` no longer uses `--from` | 0004 C, 0002 gap 2 | **done** 2026-10-20 |
| 4 | `mlpipe promote <bundle-hash>`; `register` stops auto-registering; `tune` stops writing baselines | ADR-0006, 0004 A, 0002 gap 3 | **done** 2026-10-20 |
| 5 | Top-level `PipelineConfig` pydantic schema (tracker, tune, seed, download_dir); seed reaches backends or is dropped | 0004 B | todo |
| 6 | `evaluate` scores the validation holdout too | 0004 F | todo |
| 7 | Hygiene: demo vs real manifests apart, STATUS header, bootstrap twin (deletion = human step) | 0004 G | todo |

## Phase 2 — generic core

| # | Step | State |
|---|---|---|
| 8 | Real-data smoke run (era_slice) through the whole chain; verify cache hit on rerun | todo |
| 9 | `FoldPlan.era_column` → `group_column`; era semantics into the CV step | todo |
| 10 | Declare the full `RunContext` surface in `interfaces.py`; test that steps use nothing else; drop dead `ModelBackend.fit(folds)` | todo (0004 E) |
| 11 | Entry-point discovery: `mlpipe.pipelines`, `mlpipe.backends` (stdlib `importlib.metadata`) | todo |
| 12 | Second-domain smoke pipeline (plain tabular, group k-fold) in tests | todo |

## Phase 3 — package and adopt

| # | Step | State |
|---|---|---|
| 13 | Move Numerai steps to `examples/numerai/` (registered via entry point); full real-data run (~2 h, all Numerai signatures go cold once); tag `v0.1.0` | todo; **human: push + tag** |
| 14 | numerai repo adoption: its work/0010 decisions as ADRs there; wrap one stage; prove cache hit | todo; **human: dependency install there** |
| 15 | Optional: Copier scaffolding template + `/template-sync` skill | after 14; **human: copier install + ADR** |

## Blockers / human steps

- *Resolved 2026-10-20:* the venv was repopulated host-side, and 65 tests passed at
  step 1.
- **`.venv-sandbox` is empty again** (2026-10-19). `uv run` recreates a bare Python
  3.13 venv with no pytest, and an offline `uv sync --frozen --all-extras` was denied
  in-sandbox. This is the recurrence work/0005 Q4 asked about. **Ask:** run
  `uv sync --frozen --all-extras` for this repo host-side (or in an egress window),
  then `uv run --no-sync pytest tests/ -q`. Step 1's code changes (tracker move,
  boundary test) are verified only by a stdlib AST check: 0 violations, orchestration
  511 lines, core 642.
