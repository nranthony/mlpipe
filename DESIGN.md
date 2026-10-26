# mlpipe — design reference

**Status of this document:** the design reference, not law. §0's decisions are
strong defaults that stand until an ADR (docs/adr/) changes them; §2's contracts
describe the current system and are kept true to the code; §8's landscape facts
are dated snapshots. Reality (code, tests, manifests) outranks this text — when
they disagree, fix the text.

A minimal, agent-legible MLOps pipeline for a personal repo. Mirrors the elegant core of
SageMaker Pipelines / Vertex AI (typed steps, content-hash caching, lineage records)
without the cloud plumbing. Everything local. A generic library that host repos install
from a pinned git tag (ADR-0005); Numerai is its real-data test harness
(`examples/numerai/`, pipeline `numerai-example`, ADR-0009), not its purpose and not a
tournament pipeline.

## 0. Decisions already made (do not relitigate without an ADR)

- **Orchestrator:** custom minimal core. No ZenML, no Kedro. Steps talk only to
  `RunContext`, which keeps a future ZenML shim cheap. Size is governed by
  test-enforced import and surface rules, plus a soft line-count signal on the
  orchestration modules (ADR-0004; this replaces the original 400/700 ceiling).
- **DataFrame engine:** Polars is the canonical in-memory type crossing step boundaries.
  Parquet is the canonical at-rest format (Numerai ships v5.x as parquet, int8).
  DuckDB is permitted *inside* a step for SQL-shaped work; DuckDB types never cross a
  step boundary.
- **Tracker:** MLflow 3.x local — SQLite backend, `./mlruns` artifacts. MLflow is an
  *index and viewing layer only*. The manifests + artifact store are the source of
  truth; losing MLflow must lose zero provenance.
- **Data versioning:** built-in content-addressed store. No DVC.
- **Config:** pydantic v2 models composed from YAML. Swappable slots are discriminated
  unions (e.g. a feature `transformer`) or, where hosts can plug in, a `kind` validated
  against a registry (`train.model.kind`: builtins plus the `mlpipe.backends` entry
  points). Validation errors must occur before any step runs. Every config model, step
  subtrees and their nested models included, subclasses `StrictConfig`
  (`extra="forbid"`): an unknown key fails validation instead of vanishing from the
  config and the signature. `Pipeline.run` refuses a step `config_model` that allows
  extras, and config for a step that declares no `config_model` at all. A step reads only its own subtree (plus the top-level location
  `download_dir`); anything it needs from another step arrives as an input artifact.
  `--set` values parse as JSON first, then YAML (YAML 1.1 reads `1e-05` as a string).
  A config is one subtree per step plus the top-level `PipelineConfig`
  (`core/config.py`: `tracker`, `tune`, `download_dir`); any other top-level key is
  rejected. A module's training and serve lists share one config, so a `Pipeline`
  also accepts the subtrees of its `sibling_steps` (every step of the module it doesn't
  run) without validating them; they enter none of its signatures. Inputs that change
  over time are named in config (a round, a snapshot date), never discovered: a
  signature must be computable offline (ADR-0007). Top-level keys sit outside every signature, so nothing that changes a
  step's output may live there: seeds belong in the subtree of the step they affect.
- **Hardware:** 12 GB VRAM GPU, i9-11900K. TorchBackend centralizes device placement
  and memory guards. Flag any config that would exceed 12 GB.
- **No hosted/cloud storage of run data. Everything stays local.**

## 1. Core concepts

**Artifact** — any persisted output: a parquet table, a joblib'd transformer, a model
file, a fold plan, a metrics JSON. Addressed by the SHA-256 of its serialized bytes.
Stored at `store/<hash[:2]>/<hash>.<ext>`. Immutable, deduplicated. A file snapshot
(a download) is copied and hashed in one streamed pass, never read whole into memory
and never hard-linked (the source may be rewritten in place).

**Step** — a named, pure function from named input artifacts to named output artifacts.
Declares `inputs: list[str]` and `outputs: list[str]` (artifact keys). Never touches the
filesystem, tracker SDK, or model libraries directly — only `RunContext`, which serves
declared inputs only (reading any other key raises).

**Signature** — hash over (step name, step code version, resolved config subtree for the
step, input artifact hashes, `CORE_SIGNATURE_VERSION`). Code version = hash of the step's source file **closure**:
its own module, every module under `mlpipe.steps.*` / `mlpipe.backends.*` — and, for a
host step, under the host's own top-level package — reachable from it by import, and
any module it resolves lazily and declares via `Step.code_deps` (the model backend for
the configured `kind`), closed over with that module's own package as first-party, so a
host backend's helpers count. `Step.code_roots` adds prefixes for code outside both
(e.g. `("shrec.",)` for a sibling library). Data files never reach a closure: a step
that depends on one takes it as a seeded artifact or a config value. Relative imports resolve against the importing module's
package, so they count the same as absolute ones. Code that runs from inside an artifact is a separate case: a
pickled model names its class, not its source, so `train` stores
`module_code_hash(backend)` in the model artifact. A predict-only backend edit then
moves the model's bytes, and every consumer that calls `predict` re-runs. Serving a
pinned bundle retrains nothing, so `PredictStep` instead refuses a model whose stored
backend hash differs from the code on disk (retrain). The closure is found by `find_spec` + `ast`
parsing, never by importing — a signature must be computable on a cache hit without
pulling in lightgbm or torch. `mlpipe.core.*` is deliberately **excluded**: it is the
orchestrator, not the computation, and including it would invalidate every artifact on
any core edit. Instead, a core edit that changes what a step computes (codecs, override
parsing, config normalisation) bumps `CORE_SIGNATURE_VERSION` by hand, and every
manifest records which mlpipe ran (`environment.mlpipe`).

**Caching** — before running a step, look up its signature in the manifest index. If a
manifest exists and all its output hashes are present in the store, skip and return the
cached outputs. This one mechanism provides: SageMaker-style step caching, iterate-on-
any-block (a plain `run` after a config change re-runs only the changed step and
downstream), and lineage. `--from X` is an *assertion*: every step upstream of X must
be a cache hit under the current config, or the run fails. It never serves outputs
that a different config produced.

**Run manifest** — one JSON file per step execution, append-only, in `manifests/`:
step name, signature, timestamps, status, the project's git commit (`Pipeline(project_root=…)`,
default the cwd), config hash and resolved config subtree, environment (the project's
lockfile + hash and `git_dirty`, Python version, platform, `mlpipe` version and source
commit, versions of key packages), input
hashes, output hashes, metrics summary, and the `run_id` shared by every manifest of
one `Pipeline.run` (in no signature); failed runs add `error` and `traceback`.
`index.jsonl` appends are `flock`ed. Every output hash
appears as an input hash in some later manifest → the lineage graph is implicit and
walkable from any model back to raw data + config + commit.

**Promotion** — accepting a run, only via `mlpipe promote <bundle-hash>` (ADR-0006):
moves the MLflow alias and writes the `baselines.yaml` slot `{pipeline: {alias: pin}}`
with the config each lineage step ran with, config/plan/bundle hashes, metrics and
commit. Other slots are left alone. `predict --alias` (default `champion`) serves the
pinned bundle. The next iteration cycle builds on the pinned baseline. A host that
promotes by its own gate keeps its record but should still leave a `promote` manifest
(`{"step": "promote", "status": "promoted", "inputs": {"model_bundle": <hash>}}`), or
pass that record to `gc --root-file`, so the champion's lineage stays a gc root.

## 2. Pipeline blocks (each = one Claude Code cycle)

| # | Block | Contract (inputs → outputs) | Notes |
|---|-------|------------------------------|-------|
| 1 | Ingest & snapshot | source config (`version`, `snapshot` date) → `raw_train`, `raw_validation`, `features_meta` | Reads `<download_dir>/<version>/<snapshot>/`; a missing folder is downloaded only when its date is today, so a folder never holds later bytes than its name. Freeze exact bytes, hash. Record dataset name/version/snapshot/size in the manifest. Live data is the serve list's job. The download folder may be shared: the example only adds paths to it. |
| 2 | Validate | raw tables → `validated_*` (zero-copy aliases) + `validation_report` | pandera schemas: columns, the configured era column, feature dtypes against an allow-list, per-dtype null bounds, era monotonicity, row counts vs previous snapshot. Hard gate — fail loud. The report holds content only; the previous row counts it compared against go in manifest meta. |
| 3 | Clean & transform | validated → `clean_table` + `clean_validation` + `serve_recipe` | Stateless only. `serve_recipe` (cleaner + source era column) is what serving replays on live rows; register takes it as an input. Renames the source's era column to the canonical `era` (v6.0: `date`) and drops `exclude_features`, so nothing downstream knows source names. Same input ⇒ same output. Polars exprs; validation cleaned with the same exprs so downstream skew checks have a target. |
| 4 | Feature engineering | clean → `feature_table` + `feature_validation` + `fitted_transformer` | Fitted transforms learn from train only and are **unsupervised**: `fit` sees the feature matrix, never the target. A target-aware transform (target encoding, supervised selection) fitted here would leak across CV folds, so it fits per fold inside the ModelBackend instead. The transformer is a versioned artifact that travels with the model; `feature_validation` is what the skew guard compares against. |
| 5 | Split / CV plan | feature table + cv config → `fold_plan` | Era-aware, purged, embargoed. Eras are sorted into time order (ISO dates and padded ids as strings, `(len, str)` for unpadded ids), never taken in row order. The plan is an artifact with its own hash; model comparisons must cite the same plan hash. The `FoldPlan` type is domain-free (`folds`, `plan_hash`, `group_column`, `meta`): era, purge and embargo are this step's semantics and travel in `meta`. Generic alternative for the same slot: `GroupKFoldStep` (`steps/groupcv.py`; a group never straddles a split; no group column = shuffled row k-fold). |
| 6 | Train & tune | features + fold_plan + model config → `model`, `oof_preds`, `train_history`, `feature_importance` | Via ModelBackend port. `model` is a `{full, fold_models}` bundle so stored state alone reproduces OOF (5x size, no retraining needed for audit). A backend with the optional CurveBackend members (ADR-0010) gets each fold's validation rows as an eval set: its per-iteration history (`fold, iteration, split, metric, value`) and gain importances (per fold model and full) are artifacts, and the curves also go to the tracker step-indexed. Other backends give empty frames. Optuna via `mlpipe tune`. |
| 7 | Evaluate | model + oof_preds + fold_plan + feature_validation → `eval_report` + `eval_chart` (metrics JSON + standalone plotly HTML) | Era-wise corr, sharpe, drawdown, feature exposure — for OOF (`mean_corr`, …) and for the full model on the validation holdout (`val_mean_corr`, …), side by side. Eras with unresolved targets are skipped, and so are the first `purge_eras` validation eras, whose labels overlap the full model's last train eras (`val_skipped_eras` in the report). The per-era scorer is configured by import path (`evaluate.scorer: "pkg.mod:fn"`, declared via `code_deps`); the step default is rank-Pearson, the Numerai example uses a `numerai_corr` port. Payout-accurate scoring (NCORR, MMC) is the host's job. A host's frozen holdout must never be the input of a cached step: every run would spend it. Log metrics to MLflow. Generic alternative for the same slot: `TabularEvalStep` (`task: regression` → rmse, mae, r2, corr; `task: binary` → roc_auc, pr_auc, log_loss, brier, accuracy, which need probabilities, so use a classifier backend such as `lightgbm_binary` whose `predict` returns P(y=1)). `TabularEvalStep` also reads `train_history` and `feature_importance` and writes an `eval_chart` (ROC, PR, calibration, confusion / predicted-vs-actual, curves, importance) and a `curves` summary (best vs final iteration per metric) in the report. Generic `ProfileStep` (`steps/profile.py`, ADR-0010): `clean_table` → `data_profile` + `profile_chart`; nothing consumes it, so it never invalidates training. |
| 8 | Register & package | model + transformer + eval → `model_bundle` | The bundle holds model + fitted transformer + feature list + plan hash, plus the serving recipe when the pipeline declares one (`RegisterStep(recipe="serve_recipe")`: Numerai's clean emits the fill and source era column as an artifact). Registry versions are created only by `mlpipe promote` (ADR-0006). |

**Serving (ADR-0007)** is a second step list in the same module and manifest log:
`build_serve_steps()` returns a source step (no inputs, one output), checks, and a
`PredictStep` last. Numerai: `ingest_live` (round named in config; fetched into
`<version>/live/<round>/live.parquet` with numerapi's `round_num`; the live date must
fall within `max_age_days` before the round's open time, checked on the downloaded
file before it is renamed into place) → `validate_live` (the bundle's features,
dtypes, null bounds) → `predict` (`PredictStep(frame_key="validated_live",
prepare=serve.prepare)`: the cleaner's fill, ids `numerai_ticker` + `era`).

The stateless/stateful boundary between blocks 3 and 4 is deliberate: block 3 caches on
content alone; block 4 produces fitted state that must be versioned with the model.

## 3. Class architecture (ports and adapters)

One ABC, three ports, composition everywhere else. See `src/mlpipe/core/interfaces.py`
for the authoritative signatures. Summary:

- `Step` (ABC): `name`, `inputs`, `outputs`, `signature(ctx) -> str`,
  `run(ctx) -> StepResult`.
- `RunContext`: the only door steps have to the outside world. The whole step-facing
  surface is declared (abstract) in `interfaces.py` and a test holds steps to it:
  `config`; inputs `get`, `get_lazy`, `input_hashes`, `previous`; outputs `put`,
  `put_alias`; records `log_metric`, `log_meta`, `log_curve` (tracker only: the step
  also puts the series as an artifact, ADR-0010).
- `ArtifactStore` (port): `save`, `load`, `exists`. Impl: `LocalCasStore`.
- `Tracker` (port): `start_run(manifest)`, `log_metrics`, `register_model`, and the
  ADR-0010 views `log_curve(name, values)` and `log_artifact(artifact)`: after a step
  succeeds the pipeline hands it every `.html`/`.json` output. A tracker without them
  still works (curves fall back to step-indexed `log_metrics`). Impl: `MlflowTracker`
  (also logs the step's config subtree as `cfg.*` params). Must be a thin view — no
  logic lives here.
- `ModelBackend` (port): `fit(X, y, params)`, `predict(X)`. Optional richer protocol
  `CurveBackend` (ADR-0010): `fit(..., eval_set=None)`, `history()`,
  `feature_importance()`; the train step detects it by `history`.
  Impls: `LightGBMBackend`, `LightGBMBinaryBackend`, `TorchMLPBackend`. The train step
  owns the fold loop, so backends never see a `FoldPlan`, and it persists the fitted
  backend object itself (joblib, via the store), so an adapter must be picklable and
  has no save/load. Signatures may only mention *our* types (polars.DataFrame, numpy,
  Path).
- `FoldPlan` (data type, kept in core): `folds`, `plan_hash`, `group_column`, `meta`.
  It is the contract between the CV, train and evaluate steps, and the carrier of the
  plan-hash comparability rule, which is generic to any CV pipeline. If `torch.Tensor` appears in the
  interface, the port is broken.
- `Pipeline`: ordered steps (plus `sibling_steps`, names whose config subtrees it
  accepts), `run(from_step, to_step, overrides, tag, seeds)`, `resolved_config(overrides)`
  (validation only, before anything runs), `lineage(hash)`.

Rules: no inheritance deeper than Step → ConcreteStep. Fitted transforms use the
sklearn fit/transform protocol — accept anything sklearn-compatible, don't invent a
protocol. Plotting/metrics hide behind plain functions, not classes.

## 4. Swappability tiers (governs how each module is wrapped)

- **Trivial (function):** plotting (plotly ⇄ matplotlib), metrics, report rendering.
- **Cheap (borrowed sklearn protocol):** scalers, encoders, imputers, splitters.
- **Adapter (port):** model families — GBM / Torch / sklearn via ModelBackend.
- **Pick-once (boundary-wrapped):** DataFrame engine (Polars; Parquet keeps the at-rest
  format engine-neutral). Tracker (demoted to a view, so swappable to Aim later).
- **Keep-tiny (effectively unswappable):** the core itself. Boundary enforced by tests
  (ADR-0004).

## 5. Agent surface

- **Write path:** one CLI verb.
  `mlpipe run --pipeline <name> --from <block> --to <block> --set key=value ... --tag <exp>`
  Every agent action is a shell command that lands in the manifest. `--pipeline` is
  required on every verb but `gc`: the CLI is domain-free and has no default pipeline.
- **Serving:** `mlpipe predict --pipeline <name> [--bundle <hash> | --alias <slot>]
  [--set <source>.<key>=…] [--input <parquet>] [--tag <t>] [--out <path>]` runs the
  module's serve steps (§2) with the bundle, its `model` and its `fitted_transformer`
  seeded, into the same manifest log (no tracker), and copies the `predictions`
  artifact to `--out`. Numerai: `--set ingest_live.round=N` (required; mlpipe never
  asks what round is current). `--input` replaces the source step with a snapshot of
  the file and writes an `input` record naming it, so lineage doesn't stop at a bare
  hash. A module without `build_serve_steps()` is served by `PredictStep()` alone and
  needs `--input`; its default prepare selects the bundle's features and `id_columns`
  (`register.id_columns`). The old `serve_input`/`prepare` module hooks are refused
  (ADR-0007). Output: ids + `prediction`.
- **Store hygiene:** `mlpipe gc --dry-run [--keep-days 30] [--root-file <path>]` lists
  store files that no pinned, promoted or root-file bundle's lineage, and no manifest
  from the window, reaches. A root file is any text naming 64-hex hashes. It
  reads every pipeline's manifests (the store is shared) and never deletes; removing
  the listed files is a human step.
- **Read path:** manifests (grep-able JSON), MLflow UI + MLflow MCP server for run
  comparison (scalar metrics, step-indexed training curves, `cfg.*` params, and each
  run's HTML/JSON outputs as artifacts). `mlpipe show --pipeline <name> [--step S]
  [--json]` maps each step's latest completed manifest to its outputs' store files,
  read-only. **Agent reports** (ADR-0010) are scripts in gitignored
  `reports/<pipeline>/<date>-<slug>.py` that load those files and call
  `mlpipe.report.write_report`; the page lists every hash it read, and reports never
  write to `store/` or `manifests/`. Procedure: the `report` skill.
- Config typos must die at pydantic validation, not at hour six of training.
- **Smoke runs** prove plumbing, not model quality: `--set clean.era_slice.n=100` (plus
  a small `feature_set`, few folds and few trees), always with `--tag smoke`. A slice
  changes the plan hash, so a smoke run is never comparable with a full run.

## 6. Build order (cycles)

0. **Spine**: Artifact, LocalCasStore, manifest writer, signature hashing, cache
   hit/miss logic. Unit tests proving: identical config+inputs ⇒ cache hit; any bit
   change in config, code, or inputs ⇒ miss. Nothing else starts until these pass.
1. Ingest & snapshot. 2. Validate. 3. Clean. 4. Features. 5. CV plan.
6. Train (LightGBM only). 7. Evaluate + MLflow wiring. 
8. Second backend (Torch) — the test that proves the ModelBackend port is real.
   Optionally: throwaway ZenML shim spike to verify the orchestrator exit door.
9. Tune (Optuna). 10. Register & package.

Each cycle's goal file is in `goals/`. A cycle is done when its acceptance tests pass
and the block can be iterated via the CLI with upstream served from cache.

## 7. Distribution: library first, scaffolding template later (ADR-0005)

The engine (`core/`, `cli.py`, generic backends and trackers behind extras) ships as
this repo's `mlpipe` distribution. Host repos pin it to a git tag and get fixes as
version bumps, not merges. Host pipelines and backends register through entry points
(`mlpipe.plugins`):

```toml
[project.entry-points."mlpipe.pipelines"]
myproj = "myproj.pipeline"             # build_steps(), DEFAULT_CONFIG, optional build_serve_steps()
[project.entry-points."mlpipe.backends"]
xgboost = "myproj.backends:XGBBackend"  # a ModelBackend class
```

A name claimed twice is an error. Entry points come from installed metadata, so they
appear after the host package is (re)installed. Before the first tag:

- the core is domain-neutral: no era concepts in ports, and the import boundary is
  test-enforced;
- the Numerai pipeline lives in `mlpipe.examples.numerai` (done; inside the wheel as a
  builtin pipeline rather than a separate package, decision 2026-10-21, so it runs
  without a reinstall). It is registered as `numerai-example`, so a host's own
  `numerai` pipeline name stays free (ADR-0009);
- a second, non-Numerai pipeline passes (done: builtin `demo-tabular`,
  `mlpipe.steps.demo_tabular`, grouped customer data, regression and binary). The CLI
  tests for register, promote, tune, predict and lineage run on it. A small real-data
  harness, `titanic-example` (`mlpipe.examples.titanic`, Kaggle `train.csv` via the
  generic `IngestFileStep`, which pins each file's sha256), runs in seconds and is where
  the viewing surfaces are tried (work/0013).

A Copier template, if built, carries **scaffolding only**: agent docs, ADR and `work/`
conventions, a config skeleton and CLI wiring. It is built only after a host repo has
adopted the library, and synced via `copier update` driven by a `/template-sync`
procedure. Its zones are in TEMPLATE_OWNERSHIP.md. Rollout: work/0006 plan.

## 8. Known landscape facts (as of Aug 2026, verified)

- Numerai data (2026-10-20): **v6.0** is the only version still serving `train.parquet`.
  v5.0–v5.3 serve only `live.parquet` and `features.json`. In v6.0 the era column is
  `date` (weekly ISO; 835 train dates, 2003–2018; validation 2019 to the present, incl.
  `test` rows with null target). Features: 1017 Int8 + 22 Float32 factors (sparse,
  ≤7.1% null) + 1 String (`feature_country`, in every named set). Rows also carry
  `numerai_ticker`. Pull via numerapi. (Aug 2026: v5.2, all-int8, `era`.)
  Live (2026-10-23, checked host-side): `download_dataset(..., round_num=N)` serves a
  past round's `live.parquet` (round 1378, fetched during round 1383: one `date`,
  `2026-10-15`), so a backfill can be rebuilt. Rounds 1378 and 1384 (2026-10-26):
  the live `date` is the day before the round's `openTime` (Sat ~12:00 UTC), well
  inside `ingest_live`'s 7-day window.
- W&B self-hosting is enterprise-only → not usable under the local-only constraint.
- Neptune: acquired by OpenAI, public service winding down → off the table.
- MLflow: open source, Linux Foundation; local SQLite + ./mlruns works fine; its MCP
  server + official Claude Code skills exist. Compare-UI is weak beyond ~20 runs →
  generate plotly comparisons from the API when needed.
- Aim: viable self-hosted alternative UI if MLflow's comparison view chafes.
- DVC: skipped (acquisition/maintenance-risk signal; CAS store covers the need).
