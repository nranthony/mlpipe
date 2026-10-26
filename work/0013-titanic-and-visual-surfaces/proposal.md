# Proposal: Titanic example and the visual surfaces (MLflow + agent reports)

- Status: Draft
- Author: Claude, from the user's ask (2026-10-26): use Titanic to see the features,
  training curves and other interfaces mlpipe should expose to a data scientist
- Related: [ADR-0005](../../docs/adr/0005-generic-library-numerai-as-example.md)
  (examples hold domain code), [ADR-0009](../../docs/adr/0009-numerai-example-is-a-real-data-harness.md)
  (examples are harnesses), [ADR-0002](../../docs/adr/0002-pipeline-contract-amendments.md)
  (precedent for port changes)

## Summary

Add a small, fast, real-data example (`titanic-example`, Kaggle train set, 891 rows,
runs in seconds) and use it to build the viewing surfaces the library lacks. Two
surfaces, split by the user's rule:

- **MLflow** gets everything that fits it cleanly: per-step params, scalar metrics,
  step-indexed training curves, and the HTML charts as run artifacts.
- **Agent reports** cover the rest: an agent reads the manifests and stored artifacts
  through a read-only CLI and writes ad-hoc HTML reports, each naming the hashes it used.

Generic pieces go in `steps/`, `backends/`, `trackers/` and core, so every pipeline
(numerai-example and host repos included) gets them. Only Titanic's cleaning lives in
`examples/titanic/`.

## Motivation

What a data scientist can see today:

- **MLflow:** one run per executed step, with signature, config hash and commit as
  params, and scalar metrics (`fold{i}_corr`, `rmse`/`roc_auc`, ...).
- **One HTML chart**, and only in the numerai example (`eval_chart`).

There are no training curves (backends fit without an eval set and keep no history),
no feature views, no tabular evaluation charts, and MLflow carries no artifacts and no
step config. Numerai is too slow (~30 min to retrain) and too domain-shaped to iterate
on views. Titanic is the opposite: seconds per run, mixed dtypes, missing values and a
binary target. The binary backend (`lightgbm_binary`) and binary metrics already exist.

## Data (in place)

`/workspace/raw_data/kaggle/titanic/` (the human downloaded it, 2026-10-26; the
`kaggle/` level leaves room for other Kaggle sets):

| File | Rows | sha256 |
|---|---|---|
| `train.csv` | 891 | `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f` |
| `test.csv` | 418 (no labels) | `56023b9948236f3c7a1c9448fcf418b283e109ef177fa8c7e069158dd7dd52b2` |
| `gender_submission.csv` | 418 | `7a262a7f4cb807f53911dccc84257ed3976b21b7b50ffc468b58446bff8bb6a4` |

## Proposal

### Phase A: `titanic-example` runs end to end (no new views yet)

1. **Generic `ingest_file` step** (`steps/ingest_file.py`), reusable for any local
   CSV/parquet under `download_dir` (`../raw_data/kaggle` here).
   - Config: `files: {raw_train: titanic/train.csv}` and a pinned `sha256` per file.
   - An ingest step has no inputs, so its cache key is config only. If the file
     changed on disk, a cache hit would be stale. The pinned hash closes that hole:
     the step refuses a mismatch, and a new file means a new hash, which means a new
     signature.
   - CSV is read with an explicit schema and stored as parquet. The source hash goes
     in the manifest meta.
2. **`examples/titanic/`**:
   - `clean.py` derives `feature_*` columns:
     - sex, pclass, age (nulls kept, since LightGBM handles them), sibsp, parch, fare;
     - embarked (encoded);
     - title from name (rare titles bucketed);
     - family size, is-alone, cabin known, cabin deck.
   - The clean step also makes a seeded, stratified 20% holdout. Kaggle's
     `test.csv` has no labels, so it cannot be the validation set.
   - `pipeline.py`: `ingest → clean → features (passthrough) → cvplan (GroupKFoldStep,
     no group, 5 folds) → train (lightgbm_binary) → evaluate (TabularEvalStep, binary)
     → register`.
   - Builtin name `titanic-example` (the `-example` suffix, per ADR-0009).
3. **Tests**:
   - a synthetic Titanic-shaped CSV fixture, because tests must not read `raw_data`;
   - the default config validates (add it to the phase-1 parametrize list);
   - a full run, then a second run fully cached.

### Phase B: training curves and feature importance

4. **The ModelBackend port** gains two optional, backward-compatible members. Both
   mention only our types:
   - `fit(X, y, params, eval_set: tuple[pl.DataFrame, np.ndarray] | None = None)`;
   - `history() -> dict[str, list[float]]`, where keys are `"<split>_<metric>"`:
     - LightGBM uses its built-in `record_evaluation` callback: train and valid
       per boosting round;
     - torch records per-epoch loss.
   - `feature_importance() -> dict[str, float] | None`: LightGBM gain, torch None.
5. **`TrainStep`** passes each fold's validation rows as `eval_set`. It writes two new
   artifacts:
   - `train_history`: parquet with fold, iteration, split, metric and value;
   - `feature_importance`: per-fold and full-model gain.

   Each fold's curves go to the tracker step-indexed. That needs one core addition:
   `RunContext.log_curve(name, values)`, which logs to the tracker with `step=i` and
   keeps the values out of the manifest's scalar metrics, since the artifact holds them.
6. **Cost:** this edits `train.py` and the backends, so every pipeline's train,
   evaluate and register re-run once. For numerai-example that is ~30 min of LightGBM
   on the next real run. The model bytes change only because the backend code hash
   changes; predictions do not, since there is no early stopping. Log it in STATUS.

### Phase C: charts, and MLflow gets them

7. **A generic `profile` step** (`steps/profile.py`, input `clean_table`): writes a
   `data_profile` JSON and a `profile_chart` HTML:
   - dtype, null rate and cardinality per column;
   - distributions, split by target;
   - target rate per binned feature;
   - a correlation heatmap.

   It sits beside the main chain (nothing consumes it), so it never invalidates
   training.
8. **`TabularEvalStep` gains `eval_chart`**:
   - binary: ROC and PR curves, calibration, confusion matrix at 0.5;
   - regression: residuals;
   - OOF and holdout side by side;
   - feature importance and per-fold curves, read from the new train artifacts.
9. **The Tracker port** gains `log_artifact(artifact)`. After a step succeeds, the
   pipeline hands the tracker each `html`/`json` output.
   - `MlflowTracker` also logs the step's config subtree as params, so the MLflow UI can
     compare runs by hyperparameter.
   - MLflow stays a view: the store copy is canonical.
   - Check before building: where MLflow 3 puts artifacts for a `sqlite:///mlruns/...`
     tracking URI. It must land inside the gitignored `mlruns/`.

### Phase D: agent reports

10. **`mlpipe show --pipeline P [--step S] [--json]`**: the latest manifest per step,
    with config, metrics, meta, and each output's hash, store path and format.
    - It is read-only and prints no data.
    - It gives an agent (or a human) a sanctioned map from "the latest titanic
      evaluate" to files, without importing internals.
11. **`reports/<pipeline>/<YYYY-MM-DD>-<slug>.html`** (gitignored, like `mlruns/`):
    - Agents write these by reading artifacts from the paths `mlpipe show` prints.
    - Each report states the artifact hashes and fold-plan hash it used, so the
      AGENTS.md comparison rule applies to reports too.
    - Reports never write to `store/` or `manifests/`.
12. **A repo skill `.claude/skills/report/SKILL.md`**: the procedure (run `show`, load,
    chart with plotly, cite hashes, save under `reports/`). This is the
    "repo-specific procedure" whose arrival AGENTS.md says brings the skills directory
    back.

## Acceptance

- `uv run --no-sync pytest tests/ -k work0013` passes, and the full suite stays green.
- `mlpipe run --pipeline titanic-example` runs on the real file, and a second run is
  fully cached (checked in the manifest log).
- The human opens the MLflow UI on the host
  (`uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`) and sees on
  the train run: per-fold train/valid log-loss curves, step params, and the eval
  and profile HTML as artifacts.
- One agent report is produced from `mlpipe show`: for example, survival by
  title × class, plus the folds where OOF and holdout disagree.
- STATUS entry. Core stays at about +10 lines: `log_curve`, the port members, and
  the artifact hand-off. The size note is justified there, as core is already past 600.

## Decisions this needs (an ADR on acceptance)

Phases B and C change two ports in `core/interfaces.py`, so they get one short ADR
(ADR-0010) recording:

- `eval_set`, `history`, `feature_importance` and `log_artifact`;
- that curves and charts are artifacts first and tracker views second.

No new dependency: plotly and mlflow are already in the `track` extra, and lightgbm's
`record_evaluation` is built in.

## Open questions

1. **Order:** A then B first, so you see curves soon, and C/D next? (Recommended.)
2. **Kaggle `test.csv` as a serve path** (`mlpipe predict` producing a
   submission-shaped file). It would exercise the generic predict on a second domain.
   In this item, or a later one? (Leaning later.)
3. **Agent reports as a local `reports/` dir only, or also published as private
   claude.ai pages** when you want to view one away from the host?

## Alternatives

- **Synthetic data in `demo-tabular` instead of Titanic.** Cheaper, but the views
  would only ever show made-up structure. Real data is the point here.
- **A dashboard app (Streamlit or similar).** That would be a new dependency and a
  second UI to maintain next to MLflow. Agent-written HTML covers the ad-hoc need
  without either.
- **Curves only in MLflow, no artifact.** That breaks "losing mlruns/ loses zero
  provenance".
