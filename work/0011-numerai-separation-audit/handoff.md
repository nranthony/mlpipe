# Handoff — mlpipe ↔ numerai separation audit

- For: any agent picking up mlpipe/numerai boundary cleanup
- Written: 2026-10-23, from a read-only audit session (2× Haiku keyword sweeps,
  2× Sonnet semantic reviews; high-severity claims spot-checked by hand)
- Repos audited: `mlpipe` @ `3d9ee29`, `numerai` @ `7f47f05` (pins mlpipe `v0.2.0`)
- Status: **findings only — nothing changed.** Priority is mlpipe cleanliness;
  reverse-direction notes are lower priority.
- Since the audit (2026-10-26): resolved, D4 (ADR-0007 split, `3d9ee29`) and the A
  stale-`live.parquet` bullet (work/0009: round-keyed `ingest_live`). Line numbers
  below are as of `3d9ee29`; the raw-data default is now `../raw_data/numerai`
  (`59dac43`). The A "shrink the example" fix is still open, and work/0009 grew the
  example's real-data serving since.

## Verdict

No code coupling: the numerai repo never imports `mlpipe.examples` (it uses only
`mlpipe.core.*`), and `core/` is clean. The bleed is (1) the Numerai example has
become a second production tournament pipeline, (2) generic steps hard-wire Numerai
conventions without naming Numerai, and (3) docs/STATUS carry tournament operations.

## Priority 1 — clean mlpipe

### A. Example is a de facto production pipeline (high)

- `src/mlpipe/examples/numerai/pipeline.py:1` — "The real Numerai pipeline"; defaults
  to real data `/workspace/raw_data/numerai` (:24), MLflow `experiment: numerai` (:63),
  Optuna study `lgbm-medium` (:65), `purge_eras: 12` (:42).
- `ingest.py:28-31` — live `numerapi` download, skipped when the file exists: the
  stale-`live.parquet` defect the host fixed with round-keyed paths
  (numerai `work/0012-ml-core-review/review.md:124`).
- Duplicates host concepts with **divergent behaviour**:
  - `eval_metrics.py:39` hand-ports `numerai_corr`; host calls `numerai_tools`
    directly (`numerai_client/scoring.py:20,129,138`).
  - Sharpe: `ddof=1` (`eval_metrics.py:62`) vs `ddof=0` (`scoring.py:83`) — same
    per-era series, different sharpe.
  - Target `target` (`clean.py:24`) vs `target_jupiter_60` (`spine.py:50`).
  - CV: mlpipe `cvplan` (purge 12 / embargo 4) vs host `cv.py:16-52` (embargo 4,
    different fold arithmetic) — fold-plan hashes can never match.
  - Live data: snapshotted into CAS (mlpipe `validated_live`) vs fetched per round,
    never snapshotted (host `steps/snapshot.py:8`).
- `src/mlpipe/plugins.py:27` hard-codes `"numerai"` in `BUILTIN_PIPELINES`, so every
  host install sees it alongside the host's own `numerai-client` entry point.

**Proposed fix:** shrink the example to a synthetic Numerai-shaped fixture (`era`,
`feature_*`, `target`); drop numerapi, raw_data path, v6/`feature_country` specifics,
MLflow/Optuna defaults and the hand-rolled metric; register via entry point (or
rename, e.g. `demo-tabular`). Real ingest/validate/v6-clean logic lives host-side
(already in `snapshot.py`, `prep.py`). Needs an ADR — partly reverses ADR-0005/0007.

### B. Generic steps are Numerai-shaped

| Sev | Where | Issue | Fix |
|---|---|---|---|
| High | `steps/features.py:49`, `train.py:59`, `tabular_eval.py:37`, `register.py:43` | `feature_` prefix is the only feature-selection contract; other column names → empty matrix | `features: {prefix\|columns\|exclude}` in schema (default = prefix), or features step emits a feature-list artifact the rest consume |
| High | `steps/cvplan.py:1-87` | Era-purged/embargoed CV (`era_column="era"`, `purge_eras`, `embargo_eras`, `era_order`, `n_eras`) in generic `steps/` | Move into `examples/numerai/`, or rename to group terms; `groupcv` is the generic default |
| Med | `features.py:41-42`, `train.py:46`, `tabular_eval.py:25` | Fixed slot names (`clean_table`/`clean_validation`/`feature_validation`) assume ingest→validate→clean and a separate validation split | Configurable input keys, or optional validation |
| Med | `train.py:41,71,90`, `tabular_eval.py:34`, `cli.py:254` | Logs Pearson `fold{i}_corr` (regression only); hard-coded `prediction` column; no multiclass/proba | Per-task metric via `tabular_metrics.TASKS`; name the output column in config |
| Low | `core/interfaces.py:36`, `core/config.py:54`, `train.py:86`, `register.py:29-30`, `torch_backend.py:70`, `tabular_metrics.py:2`, `groupcv.py:1`, `tabular_eval.py:1-2`, `steps/demo.py:38` | Numerai/era wording in generic modules | Scrub comments; demo uses a neutral group column |

`tests/test_work0006_second_domain.py:35-37` uses `feature_x1/x2/noise`, so it proves
"no eras" but not "no Numerai column conventions" — add a case with other column
names (fails today).

### C. Packaging, pins, tests

- `pyproject.toml:4` description names Numerai; `:16` puts `numerapi` in the generic
  `data` extra (only `examples/numerai/ingest.py` uses it) → own `numerai` extra.
- `baselines.yaml:1-9` legacy `promoted:` block pins a Numerai run (`mean_corr
  0.0254`, `lgbm-medium` trial 2) though STATUS says nothing is pinned → remove or
  move host-side (deletion is a human step).
- `tests/test_cycle9_10_tune_register.py:75,92,202` exercises generic
  register/predict/lineage via `--pipeline numerai` and `numerai_ticker` → port to
  demo. (`test_work0006_phase1.py:152` demo+numerai parametrisation and
  `test_work0010_hygiene.py` "numerai" labels are fine.)

### D. Docs and planning

1. `DESIGN.md` §8 (:231-238) — all Numerai data facts, still says v5.2 → move to
   example docs / host.
2. `DESIGN.md` §2 table (:111, ~:172) — Numerai pipeline (numerapi, `raw_live`,
   `numerai_ticker`) written as the library contract → rewrite generically.
3. `goals/STATUS.md:12-26` current state and log `:459-489` — host data paths,
   live-serving issue, pins → keep a pointer, move the state.
4. ADR-0007 decisions 4-7 (round keys, numerapi, 7-day freshness, download folder)
   are example operations; 1-3 are library → split via a new ADR (ADRs append-only).
5. `work/0009` motivation (`proposal.md:14-24,37-39`) and `work/0003` real-data
   counts are tournament operations.
6. Stale: `DESIGN.md:12` ("moving to"), `TEMPLATE_OWNERSHIP.md:19` ("once moved"),
   `work/0004/proposal.md:54` (nonexistent `steps/numerai.py`),
   `work/0007/proposal.md:205` (resolved — mark it), `goals/cycle1_ingest.md:2`
   (v5.2); cycles 5/7 phrase generic acceptance in era/Numerai-metric terms.

## Priority 2 — reverse direction (numerai repo, side notes)

- `numerai_client/cv.py` reimplements purged era CV → use mlpipe's fold plan + hash.
- `tracking.py` / `provenance.py` overlap mlpipe trackers and manifests.
- `optimize.py` overlaps `mlpipe tune` (Wilcoxon pruning / holdout rules may stay host-side).
- `steps/snapshot.py` (sha256-verified snapshot) is a candidate to upstream as a
  generic step.
- `submit.py`, `rounds.py` correctly host-only.

## Suggested sequencing

1. **Cheap, no API change:** drop legacy `baselines.yaml` block (human-confirmed
   delete), `numerapi` → own extra, scrub comments, `plugins.py` entry → entry point,
   port the register test to demo.
2. **One spec:** configurable feature selection + relocate `cvplan` to the example.
3. **ADR (Proposed):** shrink the example to a synthetic fixture; split ADR-0007
   library vs example decisions; rewrite DESIGN §2/§8.
