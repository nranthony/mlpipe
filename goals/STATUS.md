# Status

**Current state (update in place; the log below is append-only):**
- Build cycles 0–10: code complete since 2026-08-20.
- Active work: work/0006 plan ([plan.md](../work/0006-core-library-extraction/plan.md)).
  Steps 1–13 and 16 done; `v0.1.0` tagged. Next: step 14 (numerai adoption; numerai's
  work/0011 v6.0 migration is done, so it is unblocked), then optional step 15.
- Data: Numerai **v6.0** (v5.x training files are no longer served). Raw files are in
  `/workspace/raw_data/numerai/v6.0` (downloaded host-side 2026-10-20).
- Real data: one full v6.0 run (step 13: plan `8b0a1167`, bundle `f7cc4688`), but at
  purge 4. The example now purges 12 eras, so the next run's cvplan → register go cold
  and its scores are not comparable with `8b0a1167`.
- Promoted: nothing valid. `baselines.yaml` is in the pre-ADR-0006 shape (no
  `bundle_hash`), and the registry it referred to is gone. Only `mlpipe promote` may
  rewrite it; `gc --dry-run` reports that it roots nothing. Smoke runs are never promoted.
- Store: 36 artifacts, 5.36 GB, all kept today by the 30-day window alone.
- Known issue (found 2026-10-22, not fixed): the Numerai example's `ingest` signature
  covers only `version` + code, and `fetch` skips any file already on disk. A new
  round's `live.parquet` at the same path is never re-downloaded, and ingest stays
  cached, so `predict` scores stale live rows. A fix needs a way for a file's identity
  (size + mtime, or a re-download policy) to reach the signature.

Log:
- 2026-08-20: Bootstrap skeleton delivered to repo root. Cycle 0 attempted; blocked
  at environment provisioning (registries closed in sandbox, polars absent from the
  offline uv cache). Ask recorded in work/0001-cycle0-dependency-install/.
- 2026-08-20: Adopted agent-native conventions (ADR-0001): AGENTS.md entry point,
  docs/adr/, work/, next-cycle command migrated to a skill.
- 2026-08-20: Environment provisioned host-side (uv lock + sync --all-extras, uv.lock committed). All cycle deps verified in-sandbox; torch sees the 12 GiB RTX 3080 Ti. work/0001 archived. Cycle 0 unblocked.
- 2026-08-20: Cycle 0 complete. Spine implemented: LocalCasStore (CAS, atomic writes, parquet/joblib/json codecs), ManifestLog (append-only + index.jsonl + lineage), RunContext, signature hashing (name+code+config subtree+input hashes), Pipeline cache hit/miss, typer CLI, NoopTracker. All 8 acceptance tests pass; CLI demo shows fresh->cached->iterate-on-block with upstream from cache. Core budget: 400/400 lines exactly (interfaces.py excluded: 526 total, under 700 ceiling). Nothing promoted to baselines.yaml (no models yet).
- 2026-08-20: Cycle 1 complete. IngestStep pulls Numerai v5.2 via numerapi into the shared /workspace/raw_data/numerai cache and freezes the exact downloaded bytes into the CAS (raw_train 2,746,268 rows / 2.6 GB; raw_validation 4,121,080 rows / 4.4 GB; raw_live 6,997 rows; features_meta). Manifest records dataset name/version/sizes via new ctx.log_meta. CLI rerun with unchanged upstream: cache hit (sig 98c3e4d0). Store artifacts load as polars, all 2748 features Int8. Core at 409 lines (target 400, ceiling 700 — slight overage from Path snapshot + log_meta). 4 new acceptance tests; 12 total pass.
- 2026-08-20: Cycle 2 complete. ValidateStep: pandera schema gate (era String non-null, ^feature_.*$ Int8, regex) on lazy frames + streaming polars aggs for null bounds, era monotonicity, and row-count tolerance vs previous snapshot (via new ctx.previous). validated_* are zero-copy aliases of the raw hashes (new ctx.put_alias); lineage walker now resolves pass-through hashes to the earlier producer (validated_train -> validate -> ingest). Real v5.2 gate passes (all eras monotonic, zero feature nulls) and caches on rerun. Core at 447 lines (ceiling 700). 8 new tests; 20 total pass.
- 2026-08-20: Cycle 3 complete (closed after checkpoint): CleanStep with pydantic discriminated-union cleaner config (baseline|minimal), pure polars-expression cleaners in a kind->impl registry, pipeline now validates/resolves step config_model subtrees before any step runs. All 26 unit/property tests pass (byte-identical reruns, cleaner swap changes hash, bad config dies pre-run). REMAINING for DoD: real-data CLI run (clean on v5.2 medium set), rerun cache-hit proof, --set cleaner swap via CLI, then close the cycle.
- 2026-08-20: Cycle 3 closed. Real v5.2 run: clean success on medium set (sig b06fd957), rerun cached, CLI swap --from clean --set clean.cleaner.kind=minimal re-ran only clean with upstream from cache and changed the output hash (1bc1eb83). 47 GB RAM / 16 cores available for the training cycles.
- 2026-08-20: Cycle 4 complete. FeatureStep: sklearn-protocol transformer (passthrough | standard_scaler discriminated union), fit on train ONLY, joblib artifact travels with the model. CleanStep extended with clean_validation (same stateless exprs) so validation features exist for the skew-guard acceptance test. Tests prove: transformer round-trips via store, loaded transformer reproduces pipeline validation features, no leakage (stats = train-only means, != combined). Real run: features success on medium set (passthrough, int8 preserved); clean_table re-produced identical hash ea77b6cf after code change = byte-determinism on real data. 5 new tests; 31 total.
- 2026-08-20: Cycle 5 complete. CVPlanStep: era-aware purged+embargoed folds over contiguous era blocks; FoldPlan artifact (int32 indices, joblib) with plan_hash over (era list, cv config) only — proven to change iff eras or config change, invariant to row counts. assert_same_plan() guard for comparisons. Real run: 4 folds / purge 4 / embargo 4 over v5.2 train eras (sig 7708cb23, plan artifact 2fff1bbe); rerun cached. 5 new tests; 36 total.
- 2026-08-20: Cycle 6 complete. TrainStep written once against the ModelBackend port (no lightgbm import — test-enforced); LightGBMBackend adapter ~55 lines; lazy registry kind->class. Model artifact bundles full + per-fold backends so stored state reproduces OOF. Real run: 4 folds + full on medium set (~64 min, RAM within budget); fold OOF corrs 0.0245-0.0308; rerun served train from cache (sig 714c3d8a).
- 2026-08-20: Cycle 7 complete. Era-wise metrics as plain functions; standalone plotly HTML report (4.7 MB, plotly.js embedded); MlflowTracker (sqlite:///mlruns/mlflow.db) wired via CLI config. Real eval on plan 48de731b: mean_corr 0.02793, sharpe 1.711, max_drawdown 0.0357, feature_exposure 0.2549 over 574 eras. numerai-lgbm v1 registered. MLflow holds comparable runs; deleting mlruns loses zero provenance (manifests independent).
- 2026-08-20: Cycle 8 complete. TorchMLPBackend: tensors/device handling fully inside the adapter, int8 CPU tensor with per-batch GPU float cast (first full run OOM-killed at the float32 full-matrix cast — fixed), centralized VRAM estimate guard. Real proof: --set train.model.kind=torch_mlp (config-only, zero TrainStep edits) trained on the SAME plan 48de731b; assert_same_plan validates; MLflow comparison: lightgbm mean_corr 0.02793/sharpe 1.71 vs torch_mlp 0.01406/1.10. numerai-torch v1 registered. Optional ZenML spike SKIPPED: zenml install is a sandbox human boundary; revisit if the orchestrator exit door needs proving.
- 2026-08-20: Cycle 10 complete. RegisterStep bundles model+transformer+feature list+cleaner recipe+plan hash as a store artifact; MLflow registry version tags its content hash. Real proof: fresh process `mlpipe predict --name numerai-lgbm` loaded v1 by name, predicted 6,997 live rows from the raw_live snapshot (no training internals); `mlpipe lineage <bundle>` walks register->evaluate->train->features->clean->validate->ingest to the raw snapshot hashes.
- 2026-08-20: Cycle 9 complete. `mlpipe tune`: Optuna study over the config-declared space (num_leaves, learning_rate, colsample_bytree), each trial a pipeline run from train with upstream cached; persistent storage sqlite:///mlruns/optuna.db (resume proven by tests); trials tagged lgbm-medium-trialN in MLflow. Real study: 3 trials at 50 trees, best trial 2 (num_leaves 54, lr 0.0161, colsample 0.191) mean_corr 0.02542 -> promoted to baselines.yaml.
- 2026-08-20: BUILD COMPLETE. All 11 cycles done in one day. Final state: 53 tests green, ruff clean, core 638 lines (ceiling 700 — interfaces+tracker included), store 11 GB, full provenance from any registered model version back to the raw snapshot. Registered: numerai-lgbm v1 (mean_corr 0.02793, sharpe 1.711, plan 48de731b), numerai-torch v1 (0.01406). Pinned baseline: lgbm-medium trial 2. Known follow-ups: backend/helper code not in step signatures (only the step's own module is hashed); tune --from train means cleaner overrides in tune are inert; ZenML spike skipped (install = human boundary).
- 2026-08-24: Gap 1 from work/0002 fixed — step code hashing widened from the step's own
  module to its source-file closure (own module + reachable `mlpipe.steps.*`/
  `mlpipe.backends.*` + lazily-resolved modules declared via the new `Step.code_deps`;
  `TrainStep` declares the configured backend). Closure found via `find_spec`+`ast`, never
  imported, so cache-hit signatures don't load torch. `mlpipe.core.*` excluded by design
  (DESIGN.md §1 updated). 10 new tests, 63 total green, ruff clean. Demo proof: signature
  changed (`e94c3661`->`ab93bc8a`) so the step recomputed, output hashes byte-identical,
  store still 37 files, rerun cached — invalidation costs compute, not disk. NOT yet
  re-run on real data: the next `mlpipe run` recomputes ingest->register (~1.5-2h).
  Nothing re-promoted; baselines.yaml untouched. Core now 697/700 lines: the next core
  change needs the mlflow_tracker-out-of-core trade (or an ADR raising the ceiling).
- 2026-10-06: Dependency-resolution hardening (ADR-0003), from a `depaudit posture` review.
  hatchling pinned to 1.32.0 with its five requirements under `build-constraint-dependencies`
  (build deps are not in uv.lock); `no-build = false` removed — the project runs under the
  sandbox's wheels-only default, since uv exempts the project's own editable build;
  `exclude-newer` pin and explicit PyPI default index added. Re-locked via with-egress: 147
  packages, no version changes. `.venv-sandbox` (found empty) repopulated with
  `uv sync --frozen --all-extras` on Python 3.13; 63 tests green. Use `uv run --no-sync`
  from here on. Also resolves the dangling "ADR-0004" reference flagged in work/0004. NOT
  done: CVE scan of uv.lock; depaudit re-run.
- 2026-10-19: Direction set (work/0006 accepted): mlpipe is a generic library, not a
  Numerai project; Numerai becomes `examples/numerai/` (a real-data test case);
  tournament work lives in the numerai repo. ADR-0004 replaces the 400/700 line ceiling
  with a test-enforced core import boundary (`tests/test_core_boundary.py`) plus a soft
  ~600-line signal over the orchestration modules (511 today). ADR-0005: library-first,
  Copier template for scaffolding only. ADR-0006: explicit `mlpipe promote` (not yet
  built). `MlflowTracker` moved from `core/` to `mlpipe.trackers` (core 697 -> 642).
  DESIGN.md §0/§1/§4/§7, AGENTS.md, TEMPLATE_OWNERSHIP.md updated. NOT verified by
  pytest: `.venv-sandbox` was empty again (work/0005 Q4 recurred), and only a stdlib AST
  check ran (0 boundary violations). Human ask: host-side `uv sync --frozen --all-extras`.
  Plan: work/0006/plan.md.
- 2026-10-20: work/0006 steps 1-3. Venv repopulated host-side; step 1 verified (65
  green). Step 2: `clean.era_slice` {mode: tail|head|stride, n}, whole eras only (no row
  sampling, so the plan hash changes honestly); clean meta records eras + slice;
  `--tag smoke` convention in DESIGN.md §5. Step 3: `--from X` now resolves upstream by
  signature under the current config and fails if any upstream step is not a cache
  hit. It no longer serves the latest outputs regardless of config (closes work/0004 C
  and work/0002 gap 2). `tune` runs trials without `--from`, so a space below train
  recomputes instead of being inert. CleanConfig now dumps `era_slice: null`, so the
  clean signature changes (Numerai signatures were already cold). 74 tests green, ruff
  clean on src/tests. Orchestration modules 520 lines.
- 2026-10-20: work/0006 step 4: `mlpipe promote <bundle|prefix>` (ADR-0006). Writes
  `baselines.yaml` with bundle/plan hashes, metrics, and per-step config, config hash,
  signature and commit gathered along the bundle's lineage (manifests now record the
  step's resolved `config` subtree). Registers the bundle idempotently and moves
  `@champion` when a tracker is configured. Writes a `status: promoted` manifest.
  `register` no longer registers; the bundle carries `model_name`. `tune` prints the
  best trial and the command to reproduce it, and never writes baselines. `predict`
  serves from `baselines.yaml` or `--bundle`, without MLflow. Tracker port: `+set_alias`;
  `RunContext.register_model` removed. 76 tests green, ruff clean. Orchestration modules
  521 lines. NOTE: the committed `baselines.yaml` (tune trial 2, 2026-08-20) is the
  pre-ADR-0006 shape with no bundle hash, so `predict` refuses it until a real bundle is
  promoted. Left as-is, as the historical record (repo-owned).
- 2026-10-20: work/0007 (an independent review by another agent) was cross-checked
  against the code and folded into the work/0006 plan. 12 of its 14 findings hold.
  Two needed correcting: M1's fix would have pulled `mlpipe.core` into signatures,
  and D2's FoldPlan sketch puts a non-default field after a defaulted one. D1 (CV
  leakage) holds only in part: today's fits use features only, so no target leaks.
  Where the findings land:
  - new step 7b, manifest hardening: A1, M3, M4, M5;
  - M1, the host-package signature closure, ships with step 11 (latent until then);
  - tune metric/direction, step 5;
  - domain-free `predict`/`promote`, step 13;
  - classification metrics, step 12;
  - new step 16, `gc --dry-run`.
  No code changed.
- 2026-10-20: work/0006 step 5. A top-level `PipelineConfig` (`core/config.py`)
  validates `tracker`, `tune` and `download_dir`; any other non-step key, including
  one set with `--set`, fails before any step runs. `tune` adds `metric`,
  `direction` and `step`, plus a typed `space`; the hardcoded `mean_corr`/maximize
  is gone (0007 A3, tune half). The top-level `seed` is **dropped**, not wired in:
  as a top-level key it can never enter a signature, so wiring it to the backends
  would let runs with different seeds share a cache entry. The backends keep their
  seeds in hashed params and code defaults. Manifests no longer record
  `environment.seed`. `download_dir` stays outside signatures on purpose, as a
  location rather than content (existing ingest test). The Numerai config gains
  `tune.metric: mean_corr`. Numerai step signatures are unchanged by this. The demo
  step signatures change, because `demo.py` was edited. 84 tests green,
  ruff clean. Orchestration modules 520 lines; config.py adds 65, outside that set.
- 2026-10-20: work/0006 step 6 (closes work/0004 F). `evaluate` now takes
  `feature_validation` and scores the full-data model on the holdout. The OOF and
  holdout scores sit side by side: `mean_corr`, `sharpe` and so on stay OOF, and the
  holdout gets `val_*` names. Eras whose targets are unresolved (null) are skipped,
  not scored as zero. The report gains `val_era_scores`, the chart shows both series,
  and promote pins both sets of metrics through the bundle. The evaluate signature
  changes because its inputs changed; Numerai was already cold. Previously reported
  numbers, such as 0.02793 on plan 48de731b, were OOF only. 85 tests green, ruff clean.
- 2026-10-20: work/0006 step 7 (closes work/0004 G except the twin's deletion).
  - The CLI's default manifest log is now `manifests/<pipeline>`, so demo runs no
    longer land in the Numerai history. The store stays shared.
  - `lineage` gains `--pipeline`.
  - ruff excludes the bootstrap twin, so `ruff check .` is clean repo-wide.
  - The pyproject description is generic.
  - The STATUS header now carries the current state.
  - Found: `store/`, `manifests/`, `mlruns/` and `/workspace/raw_data/numerai` are
    absent from this checkout (they are gitignored, so not recoverable from git), and
    step 8 needs a fresh download.
  - The pyproject ADR citation flagged by 0004 G was already fixed under ADR-0003.
  - Human ask: delete `mlpipe-claude-init-package/` (25 tracked files).
  - 86 tests green.
- 2026-10-20: work/0006 step 7b (work/0007 A1, M3, M4, M5). This completes phase 1.
  - Failed manifests record `error` and `traceback`.
  - The environment record (computed once per run) adds `lockfile` (the first of uv,
    poetry, pdm, pixi or requirements found), `python_version`, `platform` and
    `git_dirty` (tracked files only). It has no CUDA probe, per ADR-0004.
  - `index.jsonl` appends are `fcntl.flock`ed. The 4-process test guards against
    regression, but may not tell lock from no-lock, since Linux `O_APPEND` writes are
    usually atomic already.
  - Fixed a `lo =names` whitespace slip from step 5.
  - 89 tests green, ruff clean repo-wide. Orchestration modules 557 lines (still under the ~600 signal).
- 2026-10-20: work/0006 step 8, the real-data smoke run, on **Numerai v6.0**.
  - **Why v6.0:** v5.2 `train.parquet` is no longer served (only v5.x `live` and
    `features.json` remain), so the files were downloaded host-side into
    `/workspace/raw_data/numerai/v6.0`.
  - **How v6.0 differs:**
    - the era column is `date`;
    - the features are 1017 Int8, 22 sparse Float32 factors and the String
      `feature_country`;
    - each row carries `numerai_ticker`.
  - **Changes, all in the Numerai steps' config with core untouched:**
    - validate has `era_column`, `feature_dtypes` and
      `max_feature_null_frac_by_dtype` (`Float32: 0.1`; Int8 stays at 0);
    - clean has `era_column`, renamed to the canonical `era`, and `exclude_features`
      (`feature_country`, a String, dropped for now; encoding it is model work);
    - the bundle carries `era_column` for `predict`;
    - the ingest default is v6.0.
  - **Run:** `mlpipe run --pipeline numerai --set clean.era_slice.n=100 --set
    train.model.params.n_estimators=50 --tag smoke`. The first attempt failed at
    validate on Float32 nulls (0.0714). The failed manifest carried the error and
    traceback (7b working), which led to the per-dtype bound.
  - **Second attempt:** ingest→register succeeded in 77 s. Clean: 100 train eras /
    652,973 rows, and 87 validation eras / 597,251 rows (the last 13 dates have no
    resolved target), 350 features. Plan `7c4df3df` (4 folds, purge/embargo 4),
    bundle `484a5c65`.
  - **Rerun:** all 8 steps cached. `lineage` walks from the bundle to the ingest
    snapshot, and `predict --bundle` served 7,154 live rows.
  - **Smoke metrics (plumbing only, not model quality):**
    - OOF `mean_corr` 0.0242 (sharpe 1.21);
    - holdout `val_mean_corr` 0.0016 (sharpe 0.11), with 2015–18 training eras
      against 2024–26 holdout eras and 50 trees.
  - **Disk:** store 3.6 GB.
  - **Provenance:** the manifests record `git_dirty: true` at `b571c5a`, accurately:
    this code was uncommitted when it ran and lands in the next commit, and the step
    signatures are unaffected.
  - **Not done:** `predict` doesn't emit `numerai_ticker` (needed for a v6 submission;
    step 13 makes `predict` domain-free). 92 tests green, ruff clean.
- 2026-10-20: work/0006 step 9, a domain-free `FoldPlan` (0007 D1, D2).
  - `FoldPlan` is now `folds, plan_hash, group_column=None, meta={}`, with the
    defaulted fields last (D2's correction). Era/purge/embargo are the CV step's
    semantics: `cvplan` sets `group_column` to its era column and puts purge/embargo
    in `meta`. Nothing downstream read the old fields.
  - The plan hash still covers (era list, `CVConfig`) only, so it is unchanged. A
    pinned-hash test guards this. On real v6.0 data the smoke rerun kept plan
    `7c4df3df`: ingest→features were cached, and cvplan onward re-ran because their
    code changed. The model, OOF predictions, eval report and bundle `484a5c65` came
    out byte-identical. Only the plotly chart HTML differs.
  - D1 rule, now in DESIGN.md block 4: FeatureStep fits are unsupervised (X only).
    A target-aware transform fits per fold inside the backend. A spy test asserts
    `fit` gets the feature matrix and `y=None`.
  - Numerai repo: opened `work/0011-v6-data-migration` there (`d46b931`). v5.x
    training data is retired, `target_ender_20` is gone, and the frozen holdout's era
    ids don't exist in v6.0.
  - 96 tests green, ruff clean.
- 2026-10-21: work/0006 step 10, the declared `RunContext` surface (0004 E).
  - `interfaces.RunContext` is now an ABC that declares the whole step-facing
    surface: `config`; `get`, `get_lazy`, `input_hashes`, `previous`; `put`,
    `put_alias`; `log_metric`, `log_meta`. The pipeline bookkeeping (`registry`,
    `reset_step`, `step_*`, `manifest_log`) stays on the core implementation.
  - An AST test fails any `ctx.<attr>` in `mlpipe/steps/` outside that surface,
    including `getattr(ctx, …)`. A self-test proves the check catches `ctx._store`.
  - Dropped the dead `folds` argument from `ModelBackend.fit` and both backends. The
    train step owns the fold loop.
  - `FoldPlan` stays in core (decision 2026-10-21). It is the CV→train→evaluate
    contract and carries the plan-hash rule, which is generic to any CV pipeline.
  - Real-data rerun (v6.0 smoke): only train re-ran, since its backend code changed.
    It produced the byte-identical model `3bc9b4c2`, so evaluate and register were
    served from cache by content.
  - **Gap found:** evaluate calls the model's `predict`, but its signature doesn't
    include the backend module; only train declares it. A predict-only backend change
    would re-run train with identical bytes and leave evaluate stale. Folded into
    step 11.
  - 100 tests green, ruff clean. The orchestration modules are unchanged at 557 lines.
- 2026-10-21: the bootstrap twin `mlpipe-claude-init-package/` was deleted by the human
  (work/0004 G, plan step 7). This commit records the deletion and drops its ruff
  exclude.
- 2026-10-21: work/0006 step 11, entry-point discovery and signature-closure fixes (0007 M1).
  - New `mlpipe.plugins`: the `mlpipe.pipelines` and `mlpipe.backends` entry-point
    groups (stdlib `importlib.metadata`).
    - mlpipe's own pipelines and backends stay builtins, so no resync is needed.
    - A name registered twice is an error.
    - `cli._load` and the backend registry go through it.
    - `train.model` is one `ModelConfig(kind, params)` whose `kind` is validated
      against the registry. It replaces the closed two-member union.
    - Signatures are unchanged, because they hash the raw config subtree.
  - Signature closure: for a host step, the host's own top-level package is
    first-party; `mlpipe.core` never is. mlpipe steps keep their exact prefixes, and
    on real data ingest→cvplan were cache hits.
  - Step-10 gap fixed: train stores `backend_code = module_code_hash(backend)` in the
    model artifact. A predict-only backend edit now moves the model's bytes, so
    evaluate, register and `mlpipe predict` invalidate.
  - Tests use a fake installed host package (`dist-info` with `entry_points.txt` on
    `sys.path`) and cover:
    - `mlpipe run --pipeline hostpipe` with a host backend, end to end;
    - unknown names;
    - a duplicate claim;
    - a host-helper edit changing the signature;
    - a predict-only edit changing the model hash.
  - Real data (v6.0 smoke): train onward re-ran, plan `7c4df3df`.
    - OOF predictions and eval report `f48e7200` are byte-identical.
    - New model `a450b937` and bundle `e6e0f558`.
    - `predict` served 7,154 rows.
  - 105 tests green, ruff clean. Orchestration modules: 586 lines, up 29 for the
    closure refactor plus `module_code_hash`, which needs core's module-finding to
    hash an artifact's code. Still under the ~600 signal.
- 2026-10-21: work/0006 step 12, a second, non-Numerai domain (0007 D4). Phase 2 is done.
  - New generic steps, each filling an existing slot:
    - `GroupKFoldStep` (`steps/groupcv.py`, slot `cvplan`): a group never straddles
      a split; with no group column it is shuffled row k-fold. The plan hash covers
      (groups, config).
    - `TabularEvalStep` (`steps/tabular_eval.py`, slot `evaluate`), with OOF and
      `val_*` side by side:
      - `task: regression` → rmse, mae, r2, corr;
      - `task: binary` → roc_auc, pr_auc, log_loss, brier, accuracy.
    - The metric functions are in `steps/tabular_metrics.py` (sklearn, already in the
      `ml` extra). Binary metrics reject non-probability scores.
    - Backend `lightgbm_binary`: its `predict` returns P(y=1), so the port is
      unchanged. Multiclass is out of scope for v0.1.0.
  - Shared steps made domain-free:
    - train writes the plan's `group_column` into `oof_preds`, where it used to
      hard-code `era`;
    - the fold metric helper `era_corr` is renamed `pearson`;
    - register's default `model_name` is `model`, and `era_column` enters the bundle
      only when the clean config sets it.
  - `tests/test_work0006_second_domain.py`: synthetic customers (6 rows each, 120
    train / 40 validation, per-customer effect) through make → features (shared) →
    group k-fold → train (shared) → tabular evaluate → register (shared). No `era`
    column anywhere; the rerun is all cached.
    - regression: `val_r2` 0.91, OOF `r2` 0.87;
    - binary: `val_roc_auc` 0.95, `val_log_loss` 0.30, OOF `roc_auc` 0.94.
  - Numerai regression check (v6.0 smoke): ingest→cvplan cached; train onward re-ran,
    because the backend module and train code changed.
    - OOF predictions `33167a49` and eval report `f48e7200` are byte-identical;
      plan `7c4df3df`.
    - New bundle `cbf55165`.
  - 111 tests green, ruff clean. Orchestration modules unchanged at 586 lines.
- 2026-10-21: work/0006 step 13, Numerai moves into `mlpipe.examples.numerai` and the CLI
  goes domain-free (0007 A2, A3 promote half), plus the full v6.0 run.
  - Layout (decision 2026-10-21, option 1): ingest, validate, clean, evaluate,
    eval_metrics and the pipeline (`pipeline.py`) moved via `git mv` into
    `mlpipe.examples.numerai`. It stays a builtin pipeline, so no reinstall is needed.
    - This refines ADR-0005's `examples/numerai/` wording, which pictured a separate
      directory.
    - `mlpipe.examples.` joined the first-party signature prefixes, so imports between
      example modules stay in the closure.
  - CLI:
    - `--pipeline` is required on every verb.
    - `predict` takes `--input <parquet>`, or the pipeline's `serve_input(log, store)`
      hook. An optional `prepare(frame, bundle)` hook returns `(features, ids)`;
      Numerai's does the cleaner fill and keys rows by `numerai_ticker` + `era`.
    - `promote` takes bundles without `plan_hash`/`target`.
    - `lineage` resolves a hash prefix and rejects unknown hashes, where before it
      printed an empty "success".
  - New tests: `cli.py` names no Numerai artifact; nothing outside `mlpipe.examples`
    imports it; example steps use only the `RunContext` surface; promote without a
    plan; `--pipeline` required; predict `--input` with ticker; the lineage prefix.
  - Smoke after the move: the moved steps re-ran under their new module names, but
    with byte-identical outputs, so features, cvplan, train and register were cache
    hits (bundle `cbf55165` unchanged). `predict` output is `numerai_ticker`, `era`,
    `prediction`: 7,154 unique tickers, none null.
  - **Full v6.0 run** (`--tag full-v6`, default config):
    - Data: 835 train eras, medium feature set (350 Int8; `feature_country`
      excluded), LightGBM with 200 trees.
    - Run time: 26 min ingest→register (ingest/validate cached), well under the ~2 h
      estimate.
    - Fold plan `8b0a1167` (4 folds, purge/embargo 4). Bundle `f7cc4688`.
    - OOF: `mean_corr` 0.0316, sharpe 2.00, max drawdown 0.090, feature exposure 0.255.
    - Holdout (394 resolved validation eras, 2019→2026): `val_mean_corr` 0.0216,
      sharpe 1.24, max drawdown 0.291, exposure 0.302.
    - Not comparable with any v5.2 number (`baselines.yaml` still holds the pre-ADR-0006
      v5.2 tune pin) or with the smoke plan `7c4df3df`.
  - Rerun: all 8 steps cached. `lineage` walks register→ingest; `predict` served
    7,154 rows. Store 5.0 GB.
  - **Not promoted.** Promotion is a human or agent decision via `mlpipe promote f7cc4688`
    (ADR-0006).
  - 118 tests green, ruff clean.
  - Human steps left for v0.1.0: bump `version` to 0.1.0 in `pyproject.toml` and run
    `uv lock` host-side (`uv lock` is denied in the sandbox), commit, `git tag v0.1.0`,
    push with tags.
- 2026-10-21 — **Review fixes (Gemini review of step 13), before the v0.1.0 tag.**
  All four notes checked against the code and found valid:
  - Signature closure skipped relative imports (`from .helpers import f`, `from . import
    mod`), so a host package using them could edit a helper without moving the step's
    signature. They now resolve against the importing module's package. Nothing in
    `src/mlpipe` uses relative imports, so no existing mlpipe signature moves. Test
    added.
  - An unknown `--pipeline` raised a raw `KeyError` traceback. The CLI now reports it
    as a usage error (exit 2) on run, tune, promote and predict. Test added.
  - The default `prepare` hook read `bundle["id_columns"]`, but nothing wrote it, so
    a hook-less pipeline's predictions had no key column. `register.id_columns`
    (default empty, written only when set) now fills it. Tested end to end through
    `predict --pipeline demo --input` on the second domain.
  - `ModelBackend.save/load` were never called: train pickles the fitted backend
    object through the store. Dropped from the port, both backends and the host test
    backend before v0.1.0 freezes the API. The torch round-trip test now uses the
    store path. Editing the backend modules moves `backend_code`, so Numerai's
    train (and everything after it) goes cold once.
  - Orchestration modules at 599 lines (signature.py +12 for relative imports); at
    the ~600 signal, justified as a correctness fix with no new concept.
  - 120 tests green, ruff clean.
- 2026-10-21 — **v0.1.0 released** (human: commit 4827571, tag `v0.1.0`).
- 2026-10-21 — **Numerai example: purge 4 → 12 eras.** v6.0's `target` is
  `target_jupiter_60` (verified: corr 1.0000; docs.numer.ai agrees), a 60-business-day
  horizon overlapping ~12 weekly eras. The purge of 4 was sized for v5.2's 20-day target,
  so step 13's v6.0 OOF scores (plan `8b0a1167`: mean_corr 0.0316) carry some leakage and
  are not comparable with the next run's plan. Found while moving the numerai repo to
  v6.0 (its work/0011, ADR-0007). Config only; 120 tests green.
- 2026-10-22 — **work/0006 step 16: `mlpipe gc --dry-run`.** Reports store files that no
  root reaches; never deletes (without `--dry-run` it refuses, exit 2).
  - Roots:
    - the bundle pinned in `baselines.yaml`;
    - every bundle a `promote` manifest names, each with its full lineage;
    - every artifact a manifest from the last `--keep-days` days (default 30) read or
      wrote.
  - Hashes named inside kept JSON artifacts (a bundle's `model_hash`/`transformer_hash`)
    are kept too.
  - Every manifest log under `--manifests-root` counts, because the store is shared
    across pipelines.
  - Temp and foreign files in the store are listed, never counted as unreachable.
  - Lives in `mlpipe/gc.py`, outside core (it only reads), so the orchestration line
    count is unchanged at 599.
  - Real store: 36 artifacts, 5.36 GB, 0 unreachable at 30 days. At `--keep-days 0`
    all 36 would go: nothing is promoted under ADR-0006, so only the window keeps the
    cache. The largest are the raw v6 train (2.18 GB) and validation (1.40 GB) snapshots.
  - 3 new tests; 123 green, ruff clean.
