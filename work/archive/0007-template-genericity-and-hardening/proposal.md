# Proposal: generic MLOps template & library hardening (cross-domain readiness assessment)

- Status: Reviewed 2026-10-20 — findings verified against the code and folded into the
  [work/0006 plan](../0006-core-library-extraction/plan.md); see "Disposition" at the end.
  This file stays as the independent-review record; execution is tracked in the 0006 plan only.
- Archived 2026-10-27: every finding landed in work/0006 — M1 step 11
  (`core/signature.py` `_first_party`), M2 step 16 (`mlpipe gc --dry-run`), M3/M4/M5/A1
  step 7b, D1/D2 step 9, D3 step 6, D4 step 12, A2/A3 steps 5 and 13. The template
  blueprint is ADR-0008 / work/0006 step 15, which stays open there (waits on step 14).
- Author: Antigravity (agent, pair programming with nanthony)
- Date: 2026-10-20
- Related: [work/0004](../0004-consistency-review-and-orchestrator-question/proposal.md) (consistency review),
  [work/0006](../0006-core-library-extraction/proposal.md) (library extraction),
  [work/0006 plan](../0006-core-library-extraction/plan.md),
  [DESIGN.md](../../DESIGN.md) §1, §3, §7,
  [ADR-0004](../../docs/adr/0004-core-size-structural-rules.md),
  [ADR-0005](../../docs/adr/0005-generic-library-numerai-as-example.md),
  [ADR-0006](../../docs/adr/0006-explicit-promotion.md)

## Summary

This work item establishes the cross-domain architectural assessment, gap analysis, and implementation roadmap
for turning `mlpipe` into an effective generic library and scaffolding template for seeding MLOps in other
locations and repositories.

ADR-0005 correctly established that **the core engine must be a versioned pip library, while templates carry
scaffolding only**. However, an audit of the current codebase reveals multiple critical hidden couplings,
infrastructure omissions, data science integrity traps, and AI agent observability blind spots. If unaddressed,
any second domain or host repository adopting `mlpipe` will suffer from silent cache poisoning, fragile CLI
commands, and un-diagnosable failures.

This document serves as the primary evaluation report and cross-checking baseline for future agent sessions
and models implementing these hardening improvements.

---

## Architecture Ratings Across Disciplines

| Lens | Current Rating | Target Rating | Core Assessment |
|---|:---:|:---:|---|
| **MLOps Engineer** | **7.5 / 10** | **9.0 / 10** | **+** Robust CAS store, signature cache over AST closures, zero-copy aliasing, append-only manifests.<br>**−** P0 bug: `FIRST_PARTY` in closure hasher is hardcoded to `mlpipe.*`, store lacks retention/pruning, `index.jsonl` lacks file locking, git dirty state ignored. |
| **Data Scientist** | **6.5 / 10** | **8.5 / 10** | **+** FoldPlan hash enforcement, explicit stateful/stateless boundary, `@champion` versioning.<br>**−** Data leakage pattern in reference features/CV step, validation holdout un-scored, metric primitives domain-locked, `FoldPlan` coupled to era/purge concepts. |
| **AI Engineer** | **6.0 / 10** | **8.5 / 10** | **+** Agent-native CLI single write path, grep-able JSON provenance, low cognitive overhead.<br>**−** `predict`, `tune`, and `promote` in `cli.py` hardcoded to Numerai artifacts, failed manifests drop error tracebacks, model bundle lacks unified serving interface. |

---

## Detailed Findings & Gap Analysis

### Part I: MLOps Engineering Findings

#### Finding M1: Hardcoded `FIRST_PARTY` in `signature.py` (P0 Critical Blocker)
- **Location**: `src/mlpipe/core/signature.py:27-28`
  ```python
  FIRST_PARTY = ("mlpipe.steps.", "mlpipe.backends.")
  ```
- **The Issue**: When `mlpipe` is installed as a library in a host repo (e.g. `/workspace/numerai` or a new project),
  domain steps live in `host_project.steps.*` or `my_pipeline.steps.*`. In `_imports(path)`:
  ```python
  return {m for m in found if m.startswith(FIRST_PARTY)}
  ```
  Every import from the host project is discarded!
- **Consequence**: When an engineer or agent modifies helper functions, preprocessing logic, or feature definitions
  in the host repository, **the step signature remains identical**. The engine reports a false cache hit and serves
  stale outputs, completely breaking the reproducibility contract that work/0002 was created to establish.
- **Recommended Fix**:
  1. Dynamically infer the first-party prefix from the step's module root:
     `root_pkg = type(step).__module__.split(".")[0]`
  2. Allow `PipelineConfig` or `RunContext` to declare additional first-party prefixes:
     `FIRST_PARTY_PREFIXES = ("mlpipe.", root_pkg + ".")`

#### Finding M2: Unbounded CAS Store Growth (No Retention / Garbage Collection)
- **Location**: `src/mlpipe/core/store.py`
- **The Issue**: `LocalCasStore` is strictly append-only and deduplicated. Intermediate tables (clean, features, OOF)
  and model artifacts accumulate indefinitely. With Optuna tuning studies (`mlpipe tune`) running dozens of trials,
  disk usage grows unbounded (current test store is already ~11 GB).
- **Consequence**: Long-running agent workflows or automated tuning will exhaust local disk space.
- **Recommended Fix**: Implement an `mlpipe gc` CLI verb that:
  - Traverses `manifests/index.jsonl` to collect all reachable artifact hashes from promoted bundles and recent runs.
  - Flags unreferenced artifact shards in `store/`.
  - Supports `--dry-run` and `--retention-days / --keep-promoted-only` flags before purging orphaned files.

#### Finding M3: Missing Concurrency Control on `manifests/index.jsonl`
- **Location**: `src/mlpipe/core/manifest.py:35-37`
  ```python
  with self.index.open("a") as f:
      f.write(json.dumps(record, sort_keys=True) + "\n")
  ```
- **The Issue**: Writing to `index.jsonl` relies on non-locked file appends.
- **Consequence**: When multiple processes run concurrently (e.g., parallel Optuna study workers, parallel pipelines,
  or concurrent agent sessions), JSON lines can interleave or partially overwrite, corrupting the manifest index.
- **Recommended Fix**: Introduce advisory file locking (via `fcntl.flock` or a minimal standard locking mechanism)
  around writes to `index.jsonl`.

#### Finding M4: Incomplete Environment & Runtime Tracking
- **Location**: `src/mlpipe/core/pipeline.py:36-42`
  ```python
  def environment_info(config: dict) -> dict:
      lock = Path("uv.lock")
      return {
          "lockfile_hash": sha256_bytes(lock.read_bytes()) if lock.exists() else None,
          "seed": config.get("seed"),
      }
  ```
- **The Issue**: It only checks `uv.lock` in the current working directory.
- **Consequence**: In host repositories using Poetry, Pipenv, or conda, `lockfile_hash` is silently `None`. Furthermore,
  Python minor version, OS platform, and CUDA runtime availability are missing.
- **Recommended Fix**: Expand `environment_info` to check multiple candidate lockfiles (`uv.lock`, `poetry.lock`,
  `requirements.lock`), and record `python_version`, `platform`, and `cuda_available`.

#### Finding M5: Git Dirty State Blind Spot
- **Location**: `src/mlpipe/core/manifest.py:14-21`
- **The Issue**: Manifests record `git_commit: git_commit()` which executes `git rev-parse HEAD`.
- **Consequence**: If an agent or engineer runs experiments in a dirty working tree, the run manifest records a clean
  commit hash that does not reflect what actually executed. Combined with core's exclusion from signatures, uncommitted
  changes become provenance traps.
- **Recommended Fix**: Record `git_dirty: bool` (e.g. checking `git status --porcelain`) and optionally a short diff
  patch hash in `environment_info`.

---

### Part II: Data Science Findings

#### Finding D1: Cross-Validation Feature Leakage Pattern in Reference Steps
- **Location**: `src/mlpipe/steps/features.py:55` vs `src/mlpipe/steps/train.py:64-69`
- **The Issue**:
  - `FeatureStep` fits stateful transformers (e.g., `StandardScaler`, and potentially target encoders or imputers)
    across the **entire** `train` dataset:
    ```python
    transformer.fit(matrix(train))
    outs["feature_table"] = transformer.transform(train)
    ```
  - `TrainStep` then slices folds from `feature_table` for cross-validation:
    ```python
    for tr, va in plan.folds:
        backend.fit(X[tr], y[tr])
        oof[va] = backend.predict(X[va])
    ```
- **Consequence**: The validation fold `va` was already used to fit the `transformer` in `FeatureStep`! For any stateful
  transformation (scaling, centering, imputation, PCA, target encoding), this is textbook data leakage into CV folds,
  artificially inflating out-of-fold validation metrics.
- **Recommended Fix**:
  - In template documentation and reference pipelines, establish a clear guideline:
    1. Dataset-level transformations (`FeatureStep`) must be strictly stateless (or fit on historical splits).
    2. Stateful transformers evaluated under CV must be encapsulated within fold-level fitting pipelines.

#### Finding D2: Domain Coupling in `FoldPlan` (Core Interfaces)
- **Location**: `src/mlpipe/core/interfaces.py:34-43`
  ```python
  @dataclass(frozen=True)
  class FoldPlan:
      folds: list[tuple[np.ndarray, np.ndarray]]
      era_column: str
      purge_eras: int
      embargo_eras: int
      plan_hash: str
  ```
- **The Issue**: `FoldPlan` in core explicitly defines `era_column`, `purge_eras`, and `embargo_eras`.
- **Consequence**: Non-financial, non-time-series tabular datasets (e.g., standard classification, customer churn,
  imaging metadata) do not have "eras", "purge", or "embargo". They use K-Fold, Stratified K-Fold, or Group K-Fold.
- **Recommended Fix** (Alongsides work/0006 Phase 2 Step 9):
  Generalize `FoldPlan` to:
  ```python
  @dataclass(frozen=True)
  class FoldPlan:
      folds: list[tuple[np.ndarray, np.ndarray]]
      group_column: str | None = None
      plan_hash: str
      meta: dict[str, Any] = field(default_factory=dict)
  ```
  Move era, purge, and embargo semantics exclusively into domain CV steps.

#### Finding D3: Un-Scored Validation Holdout (Work 0004 Finding F / Work 0006 Step 6)
- **Location**: `src/mlpipe/steps/evaluate.py:36-74`
- **The Issue**: `clean_validation` and `feature_validation` are generated by upstream steps, but `EvaluateStep` only
  evaluates `oof_preds`.
- **Consequence**: The headline evaluation metrics and `@champion` promotions reflect only in-sample cross-validation,
  leaving models blind to distribution shift on frozen holdout data.
- **Recommended Fix**: `EvaluateStep` must evaluate and log both OOF cross-validation scores and held-out validation
  scores side-by-side.

#### Finding D4: Missing Classification Primitives & `predict_proba`
- **Location**: `src/mlpipe/core/interfaces.py:75-88` (`ModelBackend`) and `src/mlpipe/steps/eval_metrics.py`
- **The Issue**:
  - `ModelBackend.predict(X: pl.DataFrame) -> np.ndarray` has no provision for classification probabilities (`predict_proba`).
  - `eval_metrics.py` only implements Numerai correlation metrics (`era_scores`, `sharpe`, `drawdown`).
- **Consequence**: A data scientist attempting to use `mlpipe` for binary or multi-class classification cannot compute
  ROC-AUC, PR-AUC, Brier score, or log-loss.
- **Recommended Fix**: Add standard classification and regression metric functions to reference steps and allow
  `ModelBackend` adapters to support probability outputs.

---

### Part III: AI Engineering & Agent Operability Findings

#### Finding A1: Missing Failure Diagnostics in Manifests (Agent Blind Spot)
- **Location**: `src/mlpipe/core/pipeline.py:150-153`
  ```python
  except Exception:
      self.log.write(base | {"status": "failed", "outputs": {}, "metrics": {}, "ended_at": _now()})
      self.tracker.end_run("failed")
      raise
  ```
- **The Issue**: When a pipeline run fails, the manifest records `"status": "failed"`, but records **no error message
  or traceback**.
- **Consequence**: An LLM agent inspecting manifests to debug a broken experiment only sees that the step failed,
  forcing it to parse unstructured stderr or guess the failure cause.
- **Recommended Fix**: Store `error: str(exc)` and `traceback: traceback.format_exc()` in failed manifest records.

#### Finding A2: Hardcoded Numerai Serving Logic in `cli.py`
- **Location**: `src/mlpipe/cli.py:180-222` (`predict` command)
  ```python
  live_hash = log.latest_outputs()["raw_live"]
  live = store.load(live_hash)
  feats = live.select(content["features"])
  if content["cleaner"].get("kind") == "baseline":
      feats = feats.with_columns(pl.all().fill_null(content["cleaner"].get("fill_value", 2)))
  transformer = store.load(content["transformer_hash"])
  matrix = transformer.transform(feats.to_numpy())
  model = store.load(content["model_hash"])["full"]
  preds = model.predict(pl.from_numpy(matrix, schema=content["features"]))
  frame = pl.DataFrame({"era": live["era"], "prediction": preds})
  ```
- **The Issue**: The `predict` command assumes:
  - Input artifact is named `"raw_live"`.
  - Feature cleaner has `fill_value=2`.
  - Input has an `"era"` column.
  - Transformer takes numpy and returns numpy.
- **Consequence**: `mlpipe predict` cannot serve predictions for any other dataset or domain.
- **Recommended Fix**: Decouple serving from `cli.py`. The registered `model_bundle` should encapsulate its inference
  pipeline, or `predict` should take an input file path (`--input <path>`) and delegate to a standard bundle predictor
  method: `bundle.predict(df) -> pl.DataFrame`.

#### Finding A3: Coupled Tuning and Promotion CLI Verbs
- **Location**: `src/mlpipe/cli.py:100, 164`
  - In `tune`: hardcodes `return evaluated["metrics"]["mean_corr"]` and `direction="maximize"`.
  - In `promote`: hardcodes `"target": content["target"]` and `"plan_hash": content["plan_hash"]`.
- **Consequence**: Host repos with tasks minimizing loss or with multiple targets will fail during tuning or promotion.
- **Recommended Fix**:
  - `tune`: read metric name and direction from `cfg["tune"]["metric"]` and `cfg["tune"]["direction"]`.
  - `promote`: allow `plan_hash` and `target` to be optional in the bundle metadata.

---

## Template Architecture & Scaffolding Blueprint

Under ADR-0005, the boundary between library code, template scaffolding, and host repository ownership is
rigorously defined.

```
┌────────────────────────────────────────────────────────┐
│                   mlpipe (Library)                     │
│  - core/ (store, context, pipeline, signature, etc.)   │
│  - cli.py (generic run, tune, promote, lineage, gc)    │
│  - trackers/ (MLflow adapter)                          │
│  - generic backends/ (LightGBM, Torch, Sklearn)        │
└──────────────────────────┬─────────────────────────────┘
                           │ pinned git dependency
                           ▼
┌────────────────────────────────────────────────────────┐
│             Copier Scaffolding Template                │
│  - pyproject.toml (with entry-point skeleton)          │
│  - configs/pipeline.yaml skeleton                      │
│  - src/<domain>/steps/ (domain-specific Step classes)   │
│  - tests/ (smoke tests, boundary checks)               │
│  - AGENTS.md & DESIGN.md (agent operating context)     │
│  - docs/adr/ (decision records) & work/ (in-flight)    │
└──────────────────────────┬─────────────────────────────┘
                           │ instantiated into
                           ▼
┌────────────────────────────────────────────────────────┐
│                 Host Project Repo                      │
│  - data/, store/, manifests/, mlruns/ (local storage)  │
│  - baselines.yaml (promoted champion models)           │
│  - Domain-specific pipelines & experiment logs         │
└────────────────────────────────────────────────────────┘
```

### What Belongs in the Future Copier Template
1. **Repository Structure**:
   - `src/<domain>/steps/` (with minimal sample Ingest, Clean, Train, Evaluate steps).
   - `configs/default.yaml` (specifying tracker, store_dir, tune space).
   - `tests/test_smoke.py` (running a small slice to verify cache hits).
2. **Configuration & Tooling**:
   - `pyproject.toml` pre-wired with:
     ```toml
     [project.entry-points."mlpipe.pipelines"]
     default = "<domain>.pipeline:build_steps"
     ```
   - Pre-commit / Ruff configuration.
   - `uv.lock` pinning `mlpipe @ git+https://.../mlpipe.git@v0.1.0`.
3. **Agent Governance**:
   - Managed `AGENTS.md` specifying the CLI-only write path and fold plan hash rules.
   - Staged `DESIGN.md` explaining the domain contracts.

---

## Action Plan: Hardening Phasing

This section outlines the ordered execution sequence for implementing these recommendations. It integrates
directly into the remaining steps of **work/0006**:

### Phase A: Engine Hardening & Bug Fixes (Immediate)
- [ ] **A1 (P0)**: Update `src/mlpipe/core/signature.py` to resolve first-party modules dynamically from `type(step).__module__`.
- [ ] **A2**: Add `error` and `traceback` logging to failed manifest records in `src/mlpipe/core/pipeline.py`.
- [ ] **A3**: Add advisory file locking around `manifests/index.jsonl` writes in `src/mlpipe/core/manifest.py`.
- [ ] **A4**: Enhance `environment_info` with Python runtime, platform, CUDA status, and `git_dirty` flag.

### Phase B: CLI & Core Decoupling (Aligning with work/0006 Phase 2)
- [ ] **B1**: Generalize `FoldPlan` in `src/mlpipe/core/interfaces.py` (`era_column` → `group_column`, generic `meta` dict).
- [ ] **B2**: Decouple `mlpipe tune` in `src/mlpipe/cli.py` to read objective metric and direction from config.
- [ ] **B3**: Decouple `mlpipe predict` to take `--input` and execute a self-contained bundle predictor interface.
- [ ] **B4**: Decouple `mlpipe promote` to handle optional plan hashes and targets.
- [ ] **B5**: Implement `mlpipe gc` for store retention and pruning.

### Phase C: Data Science & Configuration Integrity (work/0006 Steps 5–6)
- [ ] **C1**: Implement top-level `PipelineConfig` Pydantic model for core orchestrator settings.
- [ ] **C2**: Extend `EvaluateStep` to evaluate and log holdout validation metrics alongside OOF CV metrics.
- [ ] **C3**: Add standard evaluation metrics (regression + classification) to reference steps.

### Phase D: Entry Points & Second-Domain Smoke Test (work/0006 Steps 11–12)
- [ ] **D1**: Implement `importlib.metadata.entry_points` discovery for `mlpipe.pipelines` and `mlpipe.backends`.
- [ ] **D2**: Create an end-to-end non-Numerai smoke pipeline in `tests/` (plain tabular classification/regression with K-Fold).
- [ ] **D3**: Move Numerai steps to `examples/numerai/` and tag release `v0.1.0`.

---

## Open Questions

1. **Storage Backends**: Should `ArtifactStore` remain strictly local, or should an S3/GCS CAS adapter be added as an optional extra in the future?
   *(Recommendation: Keep strictly local for now; design ports so cloud CAS is a simple alternative implementation).*
2. **DAG vs. Linear Orchestration**: Should `Pipeline` support full DAG resolution (branching steps) before `v0.1.0`, or remain a linear sequence?
   *(Recommendation: Keep linear for `v0.1.0`. Most tabular MLOps pipelines are strictly linear or easily flattened. Revisit topological sorting if multi-branching becomes a hard requirement).*
3. **Serving Representation**: Should the model bundle artifact export an ONNX representation, or continue bundling fitted scikit-learn/PyTorch objects?
   *(Recommendation: Continue bundling native objects for `v0.1.0`, with a standard callable `predict(df)` wrapper).*

---

## Disposition (2026-10-20, Claude Code cross-check)

Each finding was checked against the code at `8b8842a`. "Lands in" refers to the
[work/0006 plan](../0006-core-library-extraction/plan.md) step that now owns it.

| Finding | Verdict | Lands in | Notes |
|---|---|---|---|
| M1 first-party prefix | **Valid, latent** | step 11 (must land with it) | No host steps exist until entry points do, so nothing is mis-cached today; at step 14 it would be silent cache poisoning. **Correction:** the suggested `("mlpipe.", root_pkg + ".")` would pull `mlpipe.core` into every signature, which DESIGN.md §1 excludes on purpose. Use: the step's root package + `mlpipe.steps.`/`mlpipe.backends.`, with `mlpipe.core.` always excluded. |
| M2 store GC | Valid, deferred | step 16 (after v0.1.0) | Deleting store files is a human step in the sandbox, so `mlpipe gc` ships dry-run-first; promoted lineages and `baselines.yaml` are roots. |
| M3 index lock | Valid, low | step 7b | `tune` is sequential today; matters once parallel trials or concurrent agents exist. `fcntl.flock` is stdlib, so it stays inside the core boundary. |
| M4 environment info | Partly | step 7b | Add lockfile candidates, `python_version`, `platform`. **Not** `cuda_available`: probing it from core would mean importing torch (ADR-0004); backends already record device info in their own meta. |
| M5 git dirty | Valid | step 7b | Record `git_dirty`; it matters more because `mlpipe.core` is outside signatures. |
| D1 CV leakage | **Partly** | step 9 (rule + doc) | Today's transformers fit on feature columns only (`transformer.fit(X)`, no target) and the default is passthrough, so no target leaks; `standard_scaler` leaks only unsupervised mean/var across folds (mild). The hazard is real for target-aware transforms → DESIGN.md block 4 rule: FeatureStep fits are unsupervised; anything target-aware fits per fold inside the backend. |
| D2 FoldPlan coupling | Valid (already planned) | step 9 | **Correction:** the sketched dataclass puts `plan_hash` (no default) after `group_column = None`, which is a `TypeError` at class creation; order the fields with the defaults last. |
| D3 holdout un-scored | Valid (already planned) | step 6 | = work/0004 F. |
| D4 classification metrics / proba | Valid | step 12 | Generic metrics arrive with the second domain; whether `predict` returns probabilities is a backend choice. The port changes only if step 12 needs it. |
| A1 failed-manifest diagnostics | Valid, cheap | step 7b | `error` + `traceback` in the failed record. |
| A2 Numerai serving in `cli.py` | Valid | step 13 | `predict --input <parquet>` plus a pipeline-provided prepare hook; `cli.py` gets a test that it names no Numerai artifacts (`raw_live`, `era`, `mean_corr`). |
| A3 tune metric / promote fields | Valid | tune → step 5; promote → step 13 | `tune.metric` / `tune.direction` belong in the step-5 `PipelineConfig`. `promote` treats `plan_hash`/`target` as optional. |
| Ratings table | Not adopted | — | Subjective scores; the findings carry the substance. |
| Template blueprint | Consistent | step 15 | Matches ADR-0005 and TEMPLATE_OWNERSHIP.md (sample steps = skeleton zone). Minor: `uv.lock` is generated by the host, not a template file. |
| Open Q1 cloud CAS | Closed | — | Local only (decision 5, 2026-10-19). A remote store would need its own ADR. |
| Open Q2 linear vs DAG | Agreed | — | Linear for v0.1.0. |
| Open Q3 ONNX | Agreed | step 13 | Native objects behind a `predict(df)` wrapper. |
