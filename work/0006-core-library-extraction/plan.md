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
- Also absorbs the independent review in work/0007 (2026-10-20). Its finding IDs
  (M1–M5, D1–D4, A1–A3) are cited per step below, and its "Disposition" table records
  the verdicts and the two corrections. Execution is tracked here only.
- Every step ends with `pytest tests/ -q` green and `ruff check` clean, plus a
  STATUS.md log line.

## Phase 1 — unblock and correct

| # | Step | Source | State |
|---|---|---|---|
| 1 | ADR-0004/0005/0006; move `MlflowTracker` out of core; core boundary test | 0004 D | **done** 2026-10-19 (`3eded0f`); pytest green 2026-10-20 |
| 2 | `era_slice` smoke runs (clean step) + `--tag smoke` convention | 0003 | **done** 2026-10-20 |
| 3 | `--from` resolves upstream by current-config signatures, not "latest ever"; `tune` no longer uses `--from` | 0004 C, 0002 gap 2 | **done** 2026-10-20 |
| 4 | `mlpipe promote <bundle-hash>`; `register` stops auto-registering; `tune` stops writing baselines | ADR-0006, 0004 A, 0002 gap 3 | **done** 2026-10-20 |
| 5 | Top-level `PipelineConfig` pydantic schema (tracker, tune, seed, download_dir); seed reaches backends or is dropped; `tune.metric` + `tune.direction` replace the hardcoded `mean_corr`/maximize | 0004 B, 0007 A3 (tune half) | **done** 2026-10-20 (seed dropped) |
| 6 | `evaluate` scores the validation holdout too, side by side with OOF | 0004 F, 0007 D3 | **done** 2026-10-20 |
| 7 | Hygiene: demo vs real manifests apart, STATUS header, bootstrap twin (deletion = human step) | 0004 G | **done** 2026-10-20 (`manifests/<pipeline>`, ruff exclude); twin deleted by the human 2026-10-21 |
| 7b | Manifest hardening: failed records carry `error` + `traceback`; environment adds `git_dirty`, `python_version`, `platform`, lockfile candidates (no CUDA probe in core); `fcntl.flock` around `index.jsonl` appends | 0007 A1, M5, M4, M3 | **done** 2026-10-20 |

## Phase 2 — generic core

| # | Step | State |
|---|---|---|
| 8 | Real-data smoke run (era_slice) through the whole chain; verify cache hit on rerun | **done** 2026-10-20 on **v6.0** (v5.x training files retired); plan `7c4df3df`, bundle `484a5c65` |
| 9 | `FoldPlan.era_column` → `group_column` + `meta` (defaulted fields last); era/purge/embargo semantics into the CV step. DESIGN.md block 4 rule: FeatureStep fits are unsupervised, target-aware transforms fit per fold inside the backend (0007 D1, D2) | **done** 2026-10-20; plan hash unchanged (`7c4df3df` on v6.0) |
| 10 | Declare the full `RunContext` surface in `interfaces.py`; test that steps use nothing else; drop dead `ModelBackend.fit(folds)` | **done** 2026-10-21; `FoldPlan` stays in core (decision 2026-10-21) |
| 11 | Entry-point discovery: `mlpipe.pipelines`, `mlpipe.backends` (stdlib `importlib.metadata`). **Same step:** the signature closure treats the step's root package as first-party (plus `mlpipe.steps.`/`mlpipe.backends.`, never `mlpipe.core.`), with a test that a host-helper edit changes the signature (0007 M1). **Also:** steps that call a model's `predict` (evaluate, register, and the predict path) must carry the backend module in their closure; today only train declares it (found at step 10) | **done** 2026-10-21; the backend fix is `backend_code` in the model artifact |
| 12 | Second-domain smoke pipeline (plain tabular, group k-fold) in tests, with generic regression/classification metrics (0007 D4) | **done** 2026-10-21; binary only, multiclass out of scope for v0.1.0 |

## Phase 3 — package and adopt

| # | Step | State |
|---|---|---|
| 13 | Move Numerai steps to `examples/numerai/` (registered via entry point). `cli.py` goes domain-free: `predict --input` + pipeline-provided prepare hook, `promote` with optional `plan_hash`/`target`, and a test that `cli.py` names no Numerai artifacts (0007 A2, A3 promote half). Full real-data run (~2 h, all Numerai signatures go cold once); tag `v0.1.0` | **done** 2026-10-21 (`mlpipe.examples.numerai`, decision 2026-10-21). Full v6.0 run: plan `8b0a1167`, bundle `f7cc4688`, 26 min. Human steps done: version 0.1.0 + `uv lock`, tag `v0.1.0` (`4827571`) |
| 14 | numerai repo adoption: its work/0010 decisions as ADRs there; wrap one stage; prove cache hit | todo; **blocked on the v0.2.0 tag** (work/0008, built 2026-10-22: the false-cache-hit paths numerai's work/0012 review found). Driven from numerai: host steps over mlpipe's core, not the example steps (review §6). **Human: dependency install there, pinned to v0.2.0** |
| 15 | Optional: Copier scaffolding template + `/template-sync` skill (blueprint: 0007 "Template Architecture") | after 14; **human: copier install + ADR** |
| 16 | `mlpipe gc --dry-run`: report unreachable store shards (roots: promoted lineages, `baselines.yaml`, a retention window). The actual purge is run by a human | **done** 2026-10-22 (`mlpipe/gc.py`, outside core; refuses without `--dry-run`) |

## Blockers / human steps

- *Resolved 2026-10-20:* `.venv-sandbox` was found empty on 2026-10-19 (the work/0005 Q4
  recurrence). It was repopulated host-side, and 65 tests passed at step 1. If it
  empties again, the ask is the same: `uv sync --frozen --all-extras` host-side, then
  `uv run --no-sync pytest tests/ -q`.
- Steps 13–15 have human steps (push and tag, installs in numerai, Copier).
