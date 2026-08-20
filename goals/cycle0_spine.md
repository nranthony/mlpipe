# Cycle 0 — the provenance spine

Goal: implement Artifact, LocalCasStore, manifest writing, signature hashing, and
cache hit/miss logic against `src/mlpipe/core/interfaces.py`. Nothing else in this
repo starts until this cycle's tests pass.

## Scope
- `core/store.py` — LocalCasStore: sha256 content addressing, `store/<h[:2]>/<h>.<ext>`,
  atomic writes (write temp, rename), dedup on save, polars⇄parquet and joblib codecs.
- `core/manifest.py` — append-only JSON manifests in `manifests/`, one per step
  execution, schema per DESIGN.md §1; an index (`manifests/index.jsonl`) for fast
  signature lookup; `lineage(content_hash)` walking output→input references.
- `core/context.py` — RunContext.get/put/log_metric; put() computes hash, saves,
  records in the in-flight manifest.
- `core/signature.py` — hash over (step name, code hash of the step's module file,
  canonical-JSON of the step's resolved config subtree, sorted input hashes).
- `core/pipeline.py` — Pipeline.run(from_step, to_step, overrides): topological order,
  per-step cache check (signature in index AND all output hashes exist in store →
  skip and reuse), manifest written for every execution including cache hits
  (status: "cached").
- `cli.py` — `mlpipe run --from X --to Y --set k=v --tag t` (typer or argparse).
- A NoopTracker satisfying the Tracker port (MLflow arrives in cycle 7).

## Out of scope
Any real step, MLflow, model backends, tuning.

## Acceptance tests (tests/test_cycle0_spine.py)
1. Round-trip: put a polars frame, get it back identical (schema + values + dtypes,
   int8 preserved).
2. Dedup: saving identical bytes twice yields one file, same hash.
3. Cache hit: run a toy 2-step pipeline twice with identical config → second run's
   manifests all status "cached"; step bodies not re-executed (assert via counter).
4. Cache miss on config: change one config value consumed by step 2 → step 1 cached,
   step 2 re-executed.
5. Cache miss on code: touch step 2's source (change a constant) → step 2 re-executed.
6. Cache miss on data: change the seed input artifact → both steps re-executed.
7. Lineage: from the final output hash, walk back to the seed input hash and the
   config hash recorded in the first manifest.
8. Manifest completeness: every execution (fresh or cached) produced a manifest with
   git commit, config hash, input hashes, output hashes, timestamps.

## Budget
Core files in scope ≤ 400 lines total (excluding tests, CLI). Report the count.
