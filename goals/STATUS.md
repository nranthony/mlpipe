# Status

**Current state (update in place; the log below is append-only):**
- Build cycles 0–10: code complete since 2026-08-20.
- work/0006 (library extraction, [plan](../work/archive/0006-core-library-extraction/plan.md))
  is complete and archived; it shipped in `v0.3.0`. `v0.2.0` tagged (work/0008, cache and provenance
  correctness; archived). Step 14 (numerai adoption, pinned to `v0.2.0`) done
  2026-10-23 on the numerai side, confirmed here 2026-10-27. work/0010 (hygiene) built
  and archived. ADR-0007 / work/0009 (serve pipeline, inputs named in config) built
  2026-10-24, real-data cache hit 2026-10-26, archived. Real serve runs 2026-10-26
  (rounds 1384 and 1378, bundle `1d3c948c`, not promoted) pass. ADR-0008 (step 15, Copier
  template) built 2026-10-27 (`copier.yml` + `template/`); work/0006 is complete. work/0013 (Titanic example,
  viewing surfaces, ADR-0010) built 2026-10-26; MLflow run grouping and the Kaggle
  test.csv serve path added 2026-10-27; it archives once the human has looked at the
  MLflow UI and the first report.
- Release: `v0.3.0` tagged and pushed (2026-10-27). numerai pins it (`bf4a4c7` there,
  121 tests pass). A host adopting mlpipe into existing code starts from
  [docs/adopting.md](../docs/adopting.md).
- Due next: the ADR-0009 re-decision (shrink the Numerai example; its trigger, step
  14, fired) and work/0011's "one spec" (feature selection, cvplan relocation).
- numerai-example's next full run retrains once (~30 min): the work/0013 train
  changes and a clean docstring edit moved its signatures.
- Data: Numerai **v6.0** (v5.x training files are no longer served). Raw files are in
  `/workspace/raw_data/numerai/v6.0` (downloaded host-side 2026-10-20). Since work/0009
  the example reads `v6.0/2026-10-20/` (copied there 2026-10-26, not moved: the folder
  is shared) and live files from `v6.0/live/<round>/`.
- Real data: the example's comparable baseline is the 2026-10-23 v6.0 run (work/0010):
  plan `8b92bc2c`, bundle `1d3c948c`, purge 12, `numerai_corr` scorer, `val_*` skipping
  12 boundary eras. OOF mean_corr 0.0305 (sharpe 1.92), val_mean_corr 0.0193 (sharpe
  1.11). The step-13 run (plan `8b0a1167`, purge 4, rank-Pearson) is not comparable.
- Promoted: nothing. `baselines.yaml` is absent (the legacy pre-ADR-0006 block was
  deleted 2026-10-26, human-approved); the first `mlpipe promote` creates it. Smoke runs
  are never promoted.
- Store: 44 artifacts, 5.45 GB, all kept today by the 30-day window alone.
- Fixed 2026-10-24 (work/0009): the example's stale `live.parquet`. Serving now runs
  `ingest_live` → `validate_live` → `predict` with the round named in config
  (`mlpipe predict --pipeline numerai-example --set ingest_live.round=N`), and leaves
  manifests.

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
- 2026-10-22 — **Folded in the numerai repo's ML-core review** (`numerai/work/0012`).
  Its mlpipe findings were re-checked here by reading and, for `--set` floats, a probe.
  - All hold, and they become three items:
    - **work/0008, cache and provenance correctness.** Undeclared input reads; register
      reading other steps' config; typos dropped from step config and signatures; host
      backend helpers outside the closure; invisible sibling code and data files;
      `1e-05` parsing as a string; manifests recording the host's commit instead of
      mlpipe's. Release blocker for v0.2.0, which numerai pins instead of v0.1.0.
    - **work/0009, live identity and serving.** Round-keyed live snapshots, a serve
      pipeline whose predictions have manifests, `run_id`.
    - **work/0010, hygiene.** Streamed snapshots, per-pipeline `baselines.yaml`, a
      gc root for host promotions, sorted cvplan eras, a pluggable evaluate scorer, the
      fit/validation boundary.
  - Corrected this file's stale-live note: the review rightly rejects size + mtime.
  - The review's adoption design (numerai writes host steps over mlpipe's core; it
    doesn't reuse the example steps, `tune` or `promote`) supersedes the step-14 drafts
    from earlier today, which never left the session scratchpad. Step 14 is now driven
    from numerai.
- 2026-10-22 — **work/0008 built: cache and provenance correctness (v0.2.0).**
  - Step reads:
    - `RunContext` serves only a step's declared `inputs`: `get`, `get_lazy` and
      `put_alias` raise on anything else.
    - Generic `register` no longer reads `clean`/`train` config. Numerai's `clean`
      emits a `serve_recipe` artifact, and `RegisterStep(recipe="serve_recipe")`
      declares it. The bundle's unread `cleaner`/`model_config` fields are gone; the
      lineage already pins each step's config.
    - An AST test holds steps to their own config section (plus `download_dir`).
  - Strict config:
    - `StrictConfig` (`extra="forbid"`) underlies every shipped config model.
    - `Pipeline.run` refuses a `config_model` that allows extras, and config for a step
      with no `config_model`. The demo steps and Numerai `ingest` gained models.
    - A cycle-1 test that changed ingest's signature with an unread `round` key now
      uses a real version change.
  - Signatures and provenance:
    - A lazily resolved backend is closed over with its own package as first-party,
      so host helpers count.
    - New `Step.code_roots` hook for sibling libraries.
    - `--set` values parse as JSON, then YAML (`1e-05` was a string).
    - Manifests record `environment.mlpipe {version, commit, dirty}` and key
      package versions. `Pipeline(project_root=…)` points git and lockfile lookups at
      the host project.
    - `CORE_SIGNATURE_VERSION = 1` is in every signature, so all cached artifacts go
      cold once.
  - Demo proof: fresh run, then fully cached; output bytes identical to before
    (`5109abe5`, `b3cee3df`); `--set summarize.factr=3` is now rejected.
  - Orchestration modules at 672 lines (past the ~600 signal). Justification: closing
    false-cache-hit paths and recording which mlpipe ran is correctness the core
    exists to provide, and it adds no new concept.
  - The sandbox venv's metadata still says mlpipe 0.0.1 (a stale editable install), so
    sandbox manifests record that until a host-side sync.
  - 16 new tests; 139 green, ruff clean.
- 2026-10-23 — **v0.2.0 released.** Human ran `uv lock` (`51d0b94`), tagged `v0.2.0`
  and pushed. work/0008 archived; work/0006 step 14 unblocked (numerai pins `v0.2.0`).
- 2026-10-23 — **work/0010 built: store, registry and example hygiene.**
  - Store: a `Path` snapshot is copied and hashed in one streamed pass (8 MiB chunks),
    not read whole into RAM. It is not hard-linked: the store is immutable and a
    download can be rewritten in place. Same bytes, same hash, so no signature moves.
  - `baselines.yaml` is keyed `{pipeline: {alias: pin}}` (`mlpipe/baselines.py`).
    `promote` writes only its slot; `predict --alias` reads one; `gc` roots every pin.
    The legacy `promoted:` block stays as written and pins nothing.
  - `gc --root-file` takes any text naming bundle hashes (a host's champion record).
    DESIGN.md says a host promoting by its own gate should leave a `promote` manifest.
  - Index lookup (item 3) dropped after measuring: 0.55 ms on the real log, ~100 ms at
    10,000 entries.
  - Numerai example:
    - validate's `previous_rows` moved from the report to manifest meta;
    - `cvplan` and clean's `era_slice` sort eras into time order (pinned plan hash
      unchanged for already-sorted input);
    - `evaluate.scorer` takes a `module:function` path, declared via `code_deps`; the
      example config uses a stdlib `numerai_corr` port (matches scipy to 1e-6);
    - `val_*` metrics skip the first `purge_eras` validation eras (label overlap with
      the full fit), listed as `val_skipped_eras` in the report.
  - Numerai signatures from validate down move (code edits); the next real run
    recomputes from there.
  - Orchestration modules at 698 lines; the store's streamed path is the addition, and
    it is the store's own job.
  - 13 new tests; 152 green, ruff clean.
- 2026-10-23 — **work/0010 baseline run (full v6.0, CLI).** Plan `8b92bc2c`, bundle
  `1d3c948c`, 28 min (train 27 min). Every step cold, as expected after v0.2.0's
  `CORE_SIGNATURE_VERSION`; rerun fully `cached` in 2.7 s.
  - Raw snapshots kept their 2026-10-21 hashes (`bbfd61b4`, `a91d5f0e`): the streamed
    store hashes real 2 GB files exactly as the in-memory path did.
  - Metrics (`numerai_corr`): OOF mean_corr 0.0305, sharpe 1.92, max drawdown 0.107,
    exposure 0.250 over 835 eras. Validation (12 boundary eras skipped) mean_corr
    0.0193, sharpe 1.11, max drawdown 0.284, exposure 0.302 over 382 eras.
  - Not comparable with step 13's `8b0a1167` (purge 4, rank-Pearson): no winner.
  - Manifests record `environment.mlpipe.dirty: true` (the human's uncommitted
    `.vscode/settings.json`) and version 0.0.1 (stale sandbox metadata).
  - work/0010 archived.
- 2026-10-23 — **ADR-0007 and ADR-0008 drafted (Proposed).** ADR-0007 decides work/0009:
  round-keyed immutable live paths, `ingest_live.round` as required config, serving as
  a pipeline with manifests, dated training snapshots, a fetch-time freshness check and
  `run_id`. numerapi's client was checked: `download_dataset(..., round_num=)` passes the
  round to the server; whether old live rounds are still served is a host-side check.
  ADR-0008 pins `copier==9.18.2` (newest before the 2026-09-21 cutoff; copier-org, MIT),
  keeps the template in this repo under `template/`, forbids tasks, migrations and Jinja
  extensions, and builds it after numerai's step 14 shows its first cache hit.
- 2026-10-23 — **ADR-0008 accepted; ADR-0007 reworked.**
  - ADR-0008: the human installed Copier as a host tool, locked the `template` extra
    (`f5f0ee0`) and synced `.venv-sandbox` (now mlpipe 0.2.0, copier 9.18.2).
  - ADR-0007 is renamed `0007-serve-pipeline-and-named-inputs.md` and split into
    library decisions (inputs that change over time are named in config; serving is a
    pipeline; `run_id`) and Numerai-example decisions (round, snapshot date, freshness).
  - The draft's move of the v6.0 files into `v6.0/2026-10-20/` is dropped. The download
    folder is shared with the numerai repo, which reads the flat paths. The example now
    only adds paths there.
  - The past-round check failed on the host with `No module named 'numerapi'`: the
    host `.venv` has base deps only, and numerapi is in the `data` extra.
- 2026-10-23 — **ADR-0007 accepted.**
  - Past-round check, on the host with `--extra data`: during round 1383,
    `round_num=1378` returned a 4.44 MB live file dated `2026-10-15`. Past rounds are
    served, so backfills can be rebuilt.
  - That result changed decision 6. A freshness check against the fetch time would
    fail every backfill (8 days old here). It now compares against the round's
    `openTime`, which `ingest_live` records when it fetches.
- 2026-10-24 — **work/0009 built (ADR-0007).**
  - Spec written, then cross-checked by a fresh-context Opus review. All 17 findings
    were folded in, including two blockers: `predict --input` would have rejected the
    dropped source step's config, and `check_table` reads `row_tolerance`.
  - Part A, the library (`6aec62f`, built by an Opus agent):
    - `run_id` on every manifest;
    - `Pipeline(sibling_steps=…)` plus `resolved_config()`;
    - `steps/predict.py` `PredictStep`, which refuses a model whose backend code
      changed since training;
    - `mlpipe predict` runs a module's `build_serve_steps()`. `--input` replaces the
      source step and writes an `input` record. The old hooks are refused.
  - Part B, the Numerai example (built by a Sonnet agent):
    - `ingest: {version, snapshot}` reads `v6.0/<snapshot>/`; a missing folder is
      downloaded only when its date is today;
    - `serve.py`: `ingest_live` (round in config; two-sided freshness against the
      round's `openTime`, checked on the downloaded file before it is renamed into
      place), `validate_live` (the bundle's features), `prepare`;
    - training validate drops `raw_live`.
  - `ManifestLog.latest_outputs` removed (no users).
  - Measured: 192 tests pass (was 152), ruff clean.
  - Size: orchestration modules at 707 lines (was 698). A run id and sibling-step
    validation are orchestration and belong in `pipeline.py`. Removing
    `latest_outputs` offsets most of what they added.
  - Not yet run on real data. Next: the human copies the v6.0 files into
    `v6.0/2026-10-20/`. Then `mlpipe run --pipeline numerai` should re-run only
    ingest and validate, with bundle `1d3c948c` cached (the cache-hit proof). A real
    `predict --set ingest_live.round=<current>` needs the Numerai API, probably from
    the host.
- 2026-10-26 — **work/0009 real-data cache hit; archived.**
  - The human copied train, validation and features.json into `v6.0/2026-10-20/`
    (`cp -p`; the flat files stay).
  - `mlpipe run --pipeline numerai`:
    - ingest (sig `9c32caf4`) and validate (`65cc2727`) re-ran, as expected (new
      config and code). The raw hashes are unchanged (`bbfd61b4`, `c1a8bbf2`,
      `9152ccfc`).
    - clean, features, cvplan (plan `973fa231`), train and evaluate were all cached.
    - register re-ran (`d0e7d32c`), not predicted: `dbfedd2` edited a comment in
      `steps/register.py`, which changed its code hash. It wrote the same bytes:
      bundle `1d3c948c`, unchanged.
  - Nothing promoted. A real serve run is still untried: it needs a promoted bundle
    and the Numerai API.
- 2026-10-26 — Example `download_dir` default is now `../raw_data/numerai`, relative to
  the cwd (the repo root, as for `store/` and `manifests/`), so a host checkout finds
  the same folder without `--set`. It is in no signature. Rerun: ingest re-ran on its
  own code edit (sig `3e8f2675`, same raw hashes); everything else was cached, bundle
  `1d3c948c`.
- 2026-10-26 — **First real serve runs** (human opened egress to the Numerai API; run
  in the sandbox with `--bundle 1d3c948c`, plan `8b92bc2c`; nothing promoted).
  - Round 1384 (current): a 4.43 MB live file was downloaded to
    `v6.0/live/1384/live.parquet`. Its `live_date` is 2026-10-23 and the round's
    `open_time` 2026-10-24T12:12Z, so the check passes. Predictions `34ccf6f8`, 7148
    rows.
  - A rerun was cached for all three steps, with no download.
  - `mlpipe lineage 34ccf6f8` reaches ingest through `validate_live`, `ingest_live`
    and the bundle's training chain.
  - Round 1378 (a backfill): `live_date` 2026-10-15, `open_time` 2026-10-16T12:00Z,
    predictions `baa4d9f1`, 7155 rows.
  - In both rounds the live data is dated the day before the round opens, so the
    7-day window has slack. A wrong round's file would be off by a week or more, and
    the check would catch it.
- 2026-10-26 — **ADR-0009 accepted; work/0012 built.** The Numerai example is a
  real-data test harness, not a tournament pipeline.
  - **Rename:** the pipeline is now `numerai-example`. The MLflow experiment is
    `numerai-harness` and the Optuna study `harness-lgbm-medium`. `manifests/numerai/`
    was copied to `manifests/numerai-example/`; the old folder is kept.
  - **Extras:** `numerapi` moved from `data` to a new `numerai` extra. `uv.lock` is
    stale until a human runs `uv lock`; the sandbox venv already has numerapi.
  - **`demo-tabular`:** a new builtin pipeline (`steps/demo_tabular.py`), lifted from
    the second-domain test. The cycle 9/10 CLI tests (register, promote, tune,
    predict, lineage) now run on it instead of the Numerai harness. The Numerai
    serve tests keep their toy helpers in `test_work0009_numerai.py`. A duplicate
    ticker test was dropped (already covered there).
  - **Wording:** Numerai/era wording removed from 9 generic modules.
    - The example's docstring lists its divergences from the numerai repo.
    - TEMPLATE_OWNERSHIP now lists `steps/**` and `plugins.py` as library code.
    - Notes added to the cycle 1/5/7 goals, AGENTS.md, DESIGN.md §0/§7, and the
      work/0003 command.
  - **Measured:** 192 tests pass, ruff clean, core 707 lines (unchanged).
  - **Real-data rerun:** `train` re-ran on a comment edit (sig `ccb430a6`) and
    produced the same model `00d5befb`. `register` (sig `198903c4`) produced the same
    bundle `1d3c948c`, plan `973fa231`. Everything else was cached from the copied
    log. A round-1384 predict was cached on all three steps, predictions `34ccf6f8`.
  - **Open:** `uv lock` (human), and deleting the legacy `baselines.yaml` block
    (whole file, proposed).
  - **Closed the same day:**
    - `uv lock` was run host-side; the only change is that numerapi moved to the
      `numerai` extra.
    - `baselines.yaml` was deleted (approved).
    - The human approved deleting `manifests/numerai/` (gitignored run data).
    - work/0012 archived.
- 2026-10-26 — **work/0013 built: Titanic example and viewing surfaces (ADR-0010,
  accepted).** All four phases are in; the Kaggle test.csv serve path is deferred to a
  later item.
  - **`titanic-example`** (`examples/titanic/`): the generic `IngestFileStep`
    (`steps/ingest_file.py`; CSV/parquet under `download_dir`, each file pinned by
    sha256, since an ingest signature is config-only) reads
    `../raw_data/kaggle/titanic/train.csv`. A Titanic clean step derives 13
    `feature_*` columns and a seeded, stratified 20% holdout. The rest is the generic
    chain with `lightgbm_binary`. A full run takes 4 s.
  - **Real run:**
    - plan `9bb6921d`;
    - OOF accuracy 0.823, ROC-AUC 0.860;
    - holdout accuracy 0.843, ROC-AUC 0.857, log-loss 0.482;
    - bundle `5ce94ca3` (not promoted).

    Curves: mean validation log-loss is best at round 88 of 400 (AUC at 117), so the
    default config overfits. Left as is, as a visible example. A rerun was fully
    cached.
  - **Curves and importance:** the optional `CurveBackend` protocol (LightGBM per
    round, torch per epoch). Train writes `train_history` and `feature_importance`,
    and logs step-indexed curves via the new `RunContext.log_curve`.
  - **Charts:**
    - `ProfileStep` writes `data_profile` and `profile_chart`;
    - `TabularEvalStep` writes `eval_chart` and a `curves` summary;
    - chart pages are byte-stable (fixed plotly div ids).
  - **MLflow:**
    - the Tracker port gains `log_curve` (batched) and `log_artifact`;
    - the pipeline hands every `.html`/`.json` output to `log_artifact`;
    - `cfg.*` params come from each step's config.

    Checked in `mlruns/mlflow.db`: the train run holds 8,005 metric points and 13
    params; the profile and eval runs carry their HTML/JSON as named artifacts.

    Grouping (follow-up, ADR-0010 addendum): in the host UI the curves seemed missing,
    because train ran once and every later run was a cache hit, which logged nothing.
    Now each pipeline run is a parent MLflow run with every step nested under it;
    cached steps get a small run (scalar metrics, `mlpipe.source_run_id`, a note
    linking the run that executed them). All runs carry `mlpipe.*` tags. Checked on
    the real DB: a cached rerun gave parent `titanic-example-d9d5c80e` with 8 cached
    children, `train-3ae69f86 (cached)` linking the original `train-3ae69f86`.

    Host UI (confirmed by the human):
    `uv run --extra track mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
  - **Agent reports:**
    - `mlpipe show [--step] [--json]` (read-only);
    - `mlpipe.report.write_report` (theme-aware page that lists the hashes it read);
    - the `report` skill;
    - gitignored `reports/`.

    First report: `reports/titanic-example/2026-10-26-survival-and-errors.html`
    (survival by title × class, OOF accuracy per cell, the 12 most confident misses,
    rounds needed), published privately at
    https://claude.ai/artifact/2KJb5X7WQqxesQgmhUkK2f.
  - **Measured:** 210 tests pass (17 new in `test_work0013_titanic_and_views.py`),
    ruff clean.
  - **Core size:** orchestration 707 → 746 lines. The +18 lines are the
    `log_curve`/`log_artifact` fallbacks and the artifact hand-off; the +21 are the
    optional grouping hooks (`start_pipeline`/`end_pipeline` around a run,
    `log_cached` on a hit). Each is the minimum that keeps trackers predating it working.
  - **Pending:**
    - numerai-example's train, evaluate and register re-run once on the next real
      run (~30 min). Its own evaluate is unchanged, but train now also builds a
      validation dataset per fold, so watch memory.
    - MLflow experiment artifact paths are absolute (`/workspace/mlpipe/mlruns/<id>`),
      so a host UI whose checkout lives at a different path will not open artifacts
      logged from the sandbox.
- 2026-10-27 — **work/0006 step 14 confirmed done; two host follow-ups fixed.**
  - **Step 14:** the numerai repo's own records show it finished 2026-10-23:
    - pins `v0.2.0`, with ADR-0008 there and entry point `numerai-client`;
    - stage A (`snapshot → prep`) is host-proved: the second session was `cached`
      with the same prep hash `410367210bbb`, and lineage reaches the snapshot;
    - stage B (`regime`) was host-proved 2026-10-25;
    - it uses only `mlpipe.core`, never the example.

    This unblocks step 15 (Copier) and the ADR-0009 re-decision.
  - **Follow-ups numerai raised:**
    - `_module_path` no longer imports `pkg.mod` to resolve a `from pkg.mod import
      Symbol` name, so a signature on a cache hit no longer imports shrec.
    - `lineage` names the execution that computed an artifact, not a later cache hit,
      so its commit is the provenance.
    - Their third ask, a streaming `save` for snapshotted files, landed in `e09dec8`
      but is in no tag yet.

    No signature changes: a titanic-example rerun was 8/8 cached.
  - **Measured:** 212 tests pass, ruff clean.
  - **Core size:** orchestration 746 → 757 lines, the minimum for the two fixes.
- 2026-10-27 — **work/ triage.**
  - **Archived:**
    - 0002: its gaps went to work/0006 steps 3–4 and ADR-0006;
    - 0003: `era_slice` landed as step 2 (the `configs/smoke.yaml` preset was not
      built);
    - 0004: findings A–G landed as steps 1–7b;
    - 0007: every finding went to 0006 (the template stays as step 15).
  - **Refreshed 0005.** Still open: the CVE scan and the depaudit re-run (host steps),
    plus two choices for nanthony: the form of `exclude-newer`, and full `mlflow` vs
    `mlflow-skinny`.
  - **Refreshed 0011.** Still open:
    - the "one spec": feature selection by the `feature_` prefix, `cvplan` in generic
      `steps/`, and fixed slot and column names;
    - the Numerai text in DESIGN §2/§8;
    - ADR-0009's re-decision on shrinking the example, now due since step 14 is done.
- 2026-10-27 — **numerai-example smoke run on the work/0013 code (plumbing only).**
  - **Run:** `--set clean.cleaner.feature_set=small --set clean.era_slice.mode=tail
    --set clean.era_slice.n=100 --set cvplan.n_folds=2 --set
    train.model.params.n_estimators=20 --to evaluate --tag smoke`.
  - **Time and memory:** 41 s (train 36.5 s), peak memory (RSS) 1.1 GB. The identical
    rerun was 7/7 cached in 2.7 s.
  - **Plan:** `7614d02f`. That is not the baseline's plan, so no comparison is valid.
  - **Outputs:**
    - `train_history` has 80 rows: 2 folds × train/valid × 20 rounds of `l2`;
    - `feature_importance` has 300 rows;
    - `eval_chart` is 4.9 MB, mostly the inlined plotly library;
    - the MLflow parent and child runs are grouped, and the rerun's children are
      tagged cached.
  - **Fixed:**
    - the cached-run lookup now searches every MLflow experiment, so ingest and
      validate, last executed under the pre-ADR-0009 `numerai` experiment, link too;
    - the `era_slice` docstring notes that validation's newest eras have no target
      (tail 100 keeps 75).

    That docstring edit moves the example's clean signature. The pending full
    retrain absorbs it.
- 2026-10-27 — **work/0013 follow-up: Titanic serve path** (built by a subagent in a
  worktree, merged in `8df2bfd`).
  - **What runs:** `mlpipe predict --pipeline titanic-example` does `ingest_test`
    (test.csv, sha256-pinned) → `predict`. `examples/titanic/serve.py` reuses clean's
    `features` (the label is optional) and turns P(Survived) > 0.5 into
    `PassengerId, Survived` (Int64, 0/1).
  - **Generic additions:**
    - a `PredictStep(finish=...)` output hook: module-level only, in `code_deps`,
      row count checked;
    - `IngestFileStep(name=...)`.
  - **Real run** on bundle 5ce94ca3 (plan 9bb6921d): 418 rows, positive rate 0.376,
    predictions `be146739`. Lineage reaches the training run, and a rerun is all
    cached. The output is parquet; Kaggle wants CSV, one polars line away.
  - **Training rerun after the merge:** `ingest` and `clean` re-ran (their code moved)
    with byte-identical outputs, and profile through register stayed cached.
  - **Measured:** 216 tests pass, ruff clean.
- 2026-10-27 — **v0.3.0 prepared:** `pyproject.toml` version is 0.3.0. It takes 26
  commits since `v0.2.0`:
  - work/0010's streaming `save`;
  - ADR-0010 (curves, `CurveBackend`, charts, MLflow artifacts, run grouping,
    `mlpipe show`, reports);
  - the Titanic example and serve path;
  - the two numerai follow-ups.

  Host steps: `uv lock`, commit, `git tag v0.3.0`, push. Then numerai bumps its pin
  (`uv add "mlpipe @ git+https://github.com/nranthony/mlpipe.git@v0.3.0"`).
- 2026-10-27 — **work/0006 step 15: the Copier template (ADR-0008), in v0.3.0.**
  nanthony kept it to apply to a second repo outside this profile.
  - **Files:** `copier.yml` (four questions: project, package, pipeline and
    description; no tasks, migrations or extensions) and `template/`.
  - **What it renders:**
    - `AGENTS.md` with a managed section (session ritual, hard rules for a host,
      including "import from submodules", learned from numerai's closure test), the
      `CLAUDE.md` stub and a `template-sync` skill;
    - `pyproject.toml`, which pins mlpipe to the template's tag (`_commit`), so
      `copier update` moves the pin, and registers the entry point;
    - one sample `StrictConfig` step;
    - `tests/test_smoke.py` (CLI run, then all cached);
    - the DESIGN, STATUS, ADR-template, `work/README.md` and `.gitignore` skeletons.
  - **Test:** `tests/test_template.py` builds a throwaway git repo holding only the
    template, tags it, renders it, checks the layout and that pin and `_commit` match,
    and runs the rendered smoke test against this checkout. The rendered code is ruff
    clean.
  - **Measured:** 219 tests pass. work/0006 is complete, all steps done; it archives
    with the release.
- 2026-10-27 — **v0.3.0 released; work/0006 archived; adoption brief.**
  - **Release:** the human ran `uv lock` (`6c2fdcd`), tagged `v0.3.0` and pushed.
  - **numerai:** bumped its `[tool.uv.sources]` pin from `v0.2.0` to `v0.3.0`
    (`bf4a4c7` there); 121 tests pass. No API it uses changed since `v0.2.0`. The
    signature fix returns the same `None` as before without the import, so its cached
    `prep` should stay a hit. The real-data check (`prep` cached at `410367210bbb`) is
    still to run.
  - **`docs/adopting.md`:** a one-page brief for an agent adopting mlpipe into a repo
    with existing code. It covers applying the template, the first stage, what must
    reach a signature (raw-file sha256, seeds, library versions, `code_roots`), CLI or
    in-process, the cache-hit and lineage proof, and the adoption ADR. It distils
    numerai's ADR-0008 and supersedes work/0006's pre-release `host-repo-handoff.md`.
  - **Archived:** work/0006 moved to `work/archive/`, with its links fixed.
