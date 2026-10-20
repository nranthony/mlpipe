# mlpipe — design reference

**Status of this document:** the design reference, not law. §0's decisions are
strong defaults that stand until an ADR (docs/adr/) changes them; §2's contracts
describe the current system and are kept true to the code; §8's landscape facts
are dated snapshots. Reality (code, tests, manifests) outranks this text — when
they disagree, fix the text.

A minimal, agent-legible MLOps pipeline for a personal repo. Mirrors the elegant core of
SageMaker Pipelines / Vertex AI (typed steps, content-hash caching, lineage records)
without the cloud plumbing. Everything local. A generic library that host repos install
from a pinned git tag (ADR-0005); Numerai is its real-data example and test case
(moving to `examples/numerai/`), not its purpose.

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
  unions (e.g. `model: LightGBMConfig | TorchConfig`). A small registry maps config
  type → implementation class. Validation errors must occur before any step runs.
  A config is one subtree per step plus the top-level `PipelineConfig`
  (`core/config.py`: `tracker`, `tune`, `download_dir`); any other top-level key is
  rejected. Top-level keys sit outside every signature, so nothing that changes a
  step's output may live there: seeds belong in the subtree of the step they affect.
- **Hardware:** 12 GB VRAM GPU, i9-11900K. TorchBackend centralizes device placement
  and memory guards. Flag any config that would exceed 12 GB.
- **No hosted/cloud storage of run data. Everything stays local.**

## 1. Core concepts

**Artifact** — any persisted output: a parquet table, a joblib'd transformer, a model
file, a fold plan, a metrics JSON. Addressed by the SHA-256 of its serialized bytes.
Stored at `store/<hash[:2]>/<hash>.<ext>`. Immutable, deduplicated.

**Step** — a named, pure function from named input artifacts to named output artifacts.
Declares `inputs: list[str]` and `outputs: list[str]` (artifact keys). Never touches the
filesystem, tracker SDK, or model libraries directly — only `RunContext`.

**Signature** — hash over (step name, step code version, resolved config subtree for the
step, input artifact hashes). Code version = hash of the step's source file **closure**:
its own module, every module under `mlpipe.steps.*` / `mlpipe.backends.*` reachable from
it by import, and any module it resolves lazily and declares via `Step.code_deps` (the
model backend for the configured `kind`). The closure is found by `find_spec` + `ast`
parsing, never by importing — a signature must be computable on a cache hit without
pulling in lightgbm or torch. `mlpipe.core.*` is deliberately **excluded**: it is the
orchestrator, not the computation, and including it would invalidate every artifact on
any core edit; core is covered by tests and by the `git_commit` in every manifest.

**Caching** — before running a step, look up its signature in the manifest index. If a
manifest exists and all its output hashes are present in the store, skip and return the
cached outputs. This one mechanism provides: SageMaker-style step caching, iterate-on-
any-block (a plain `run` after a config change re-runs only the changed step and
downstream), and lineage. `--from X` is an *assertion*: every step upstream of X must
be a cache hit under the current config, or the run fails. It never serves outputs
that a different config produced.

**Run manifest** — one JSON file per step execution, append-only, in `manifests/`:
step name, signature, timestamps, status, git commit, config hash and resolved config
subtree, environment (lockfile + hash, Python version, platform, `git_dirty`), input
hashes, output hashes, metrics summary; failed runs add `error` and `traceback`.
`index.jsonl` appends are `flock`ed. Every output hash
appears as an input hash in some later manifest → the lineage graph is implicit and
walkable from any model back to raw data + config + commit.

**Promotion** — accepting a run, only via `mlpipe promote <bundle-hash>` (ADR-0006):
moves the MLflow `@champion` alias and writes `baselines.yaml` with the config each
lineage step ran with, config/plan/bundle hashes, metrics and commit. `predict` serves
the pinned bundle. The next iteration cycle builds on the pinned baseline.

## 2. Pipeline blocks (each = one Claude Code cycle)

| # | Block | Contract (inputs → outputs) | Notes |
|---|-------|------------------------------|-------|
| 1 | Ingest & snapshot | source config → `raw_train`, `raw_validation`, `raw_live`, `features_meta` | numerapi pull into the shared download cache, freeze exact bytes, hash. Record dataset name/version/size in the manifest. |
| 2 | Validate | raw tables → `validated_*` (zero-copy aliases) + `validation_report` | pandera schemas: columns, the configured era column, feature dtypes against an allow-list, per-dtype null bounds, era monotonicity, row counts vs previous snapshot. Hard gate — fail loud. |
| 3 | Clean & transform | validated → `clean_table` + `clean_validation` | Stateless only. Renames the source's era column to the canonical `era` (v6.0: `date`) and drops `exclude_features`, so nothing downstream knows source names. Same input ⇒ same output. Polars exprs; validation cleaned with the same exprs so downstream skew checks have a target. |
| 4 | Feature engineering | clean → `feature_table` + `feature_validation` + `fitted_transformer` | Fitted transforms learn from train only. The transformer is a versioned artifact that travels with the model; `feature_validation` is what the skew guard compares against. |
| 5 | Split / CV plan | feature table + cv config → `fold_plan` | Era-aware, purged, embargoed. The plan is an artifact with its own hash; model comparisons must cite the same plan hash. |
| 6 | Train & tune | features + fold_plan + model config → `model`, `oof_preds` | Via ModelBackend port. `model` is a `{full, fold_models}` bundle so stored state alone reproduces OOF (5x size, no retraining needed for audit). Optuna via `mlpipe tune`. |
| 7 | Evaluate | model + oof_preds + fold_plan + feature_validation → `eval_report` + `eval_chart` (metrics JSON + standalone plotly HTML) | Era-wise corr, sharpe, drawdown, feature exposure — for OOF (`mean_corr`, …) and for the full model on the validation holdout (`val_mean_corr`, …), side by side. Eras with unresolved targets are skipped. Log metrics to MLflow. |
| 8 | Register & package | model + transformer + eval → `model_bundle` | The bundle holds model + fitted transformer + feature list + cleaner recipe + plan hash. Registry versions are created only by `mlpipe promote` (ADR-0006). |

The stateless/stateful boundary between blocks 3 and 4 is deliberate: block 3 caches on
content alone; block 4 produces fitted state that must be versioned with the model.

## 3. Class architecture (ports and adapters)

One ABC, three ports, composition everywhere else. See `src/mlpipe/core/interfaces.py`
for the authoritative signatures. Summary:

- `Step` (ABC): `name`, `inputs`, `outputs`, `signature(ctx) -> str`,
  `run(ctx) -> StepResult`.
- `RunContext`: `config`, `get(key) -> Artifact`, `put(obj, key) -> Artifact`,
  `log_metric(name, value)`. The only door steps have to the outside world.
- `ArtifactStore` (port): `save`, `load`, `exists`. Impl: `LocalCasStore`.
- `Tracker` (port): `start_run(manifest)`, `log_metrics`, `register_model`.
  Impl: `MlflowTracker`. Must be a thin view — no logic lives here.
- `ModelBackend` (port): `fit(X, y, params, folds)`, `predict(X)`, `save(path)`.
  Impls: `LightGBMBackend`, later `TorchBackend`. Signatures may only mention *our*
  types (polars.DataFrame, numpy, Path, FoldPlan). If `torch.Tensor` appears in the
  interface, the port is broken.
- `Pipeline`: ordered steps, `run(from_step, to_step, overrides)`, `lineage(hash)`.

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
  `mlpipe run --from <block> --to <block> --set key=value ... --tag <exp>`
  Every agent action is a shell command that lands in the manifest.
- **Read path:** manifests (grep-able JSON), MLflow UI + MLflow MCP server for run
  comparison. Optional: generate plotly HTML comparisons from the tracking API.
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
(`mlpipe.pipelines`, `mlpipe.backends`). Before the first tag:

- the core is domain-neutral: no era concepts in ports, and the import boundary is
  test-enforced;
- the Numerai pipeline lives in `examples/numerai/`;
- a second, non-Numerai smoke pipeline passes.

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
- W&B self-hosting is enterprise-only → not usable under the local-only constraint.
- Neptune: acquired by OpenAI, public service winding down → off the table.
- MLflow: open source, Linux Foundation; local SQLite + ./mlruns works fine; its MCP
  server + official Claude Code skills exist. Compare-UI is weak beyond ~20 runs →
  generate plotly comparisons from the API when needed.
- Aim: viable self-hosted alternative UI if MLflow's comparison view chafes.
- DVC: skipped (acquisition/maintenance-risk signal; CAS store covers the need).
