# ADR-0007: Live data is identified by round; serving is a pipeline with manifests

- Status: Proposed
- Date: 2026-10-23
- Deciders: nanthony + Claude (work/0009; numerai `work/0012-ml-core-review` D4, P0-2,
  P1-18, mlops C9/C14/C15)

## Context

The Numerai example serves stale data and leaves no record of what it served:

- **Live data is never refreshed.** `examples/numerai/ingest.py` `fetch` skips any file
  that exists, and ingest's signature is `{version}` + code. A new round's `live.parquet`
  is never downloaded and ingest stays `cached`. numerapi's own `download_file` also
  returns early when the local size matches the server's.
- **Hashing the file can't fix this.** Size, mtime or sha256 in the signature are all
  computed before `run()`. On a cache hit `fetch` never runs, so nothing refreshes the
  file. (STATUS.md suggested size + mtime on 2026-10-22; that was wrong.)
- **`serve_input` serves "latest ever"**: `latest_outputs()["raw_live"]`, the last live
  snapshot any run wrote, whatever its round or tag.
- **`mlpipe predict` writes no manifest**, so a submission has no lineage.
- **No `run_id`**: one run's manifests can be grouped only by timestamp.

Checked 2026-10-23 in numerapi's source (sandbox venv): `download_dataset(filename,
dest_path, round_num)` sends `round` to the server's `dataset` query. The client
supports a past round. Whether the server still serves an old round's *live* file is
unverified.

## Decision

1. **A round's live data has its own immutable path:**
   `<download_dir>/<version>/live/<round>/live.parquet`, fetched with
   `round_num=<round>` and never overwritten. A new round is a new path, so neither
   the exists-check nor numerapi's size check can return an old file.
2. **The round is config, never discovered.** `ingest_live: {version, round}`, with
   `round` required. The signature stays computable offline, and two rounds can't
   share a cache entry. The caller (a human, or numerai's round loop) resolves the
   current round and passes `--set ingest_live.round=N`. mlpipe never asks the API
   for the current round.
3. **Serving is a pipeline.** The Numerai pipeline module gains a serve step list,
   `ingest_live` → `validate_live` → `predict`, logged to the same manifest log as
   training. The bundle enters as a seed (`Pipeline.run(seeds=…)`, which exists), and
   `predictions` is a store artifact with a manifest. So a submission's lineage reaches
   the bundle, and through the bundle, the training run.
   - `mlpipe predict` becomes a thin wrapper that runs the module's serve steps.
   - The `serve_input` hook and its "latest ever" lookup are removed. `prepare` folds
     into the predict step.
   - **`--input <parquet>` stays** for ad-hoc scoring. The file is snapshotted into the
     store and seeded in place of `ingest_live`'s output, so it gets a manifest too.
4. **Training ingest stops fetching live.** `ingest` takes `train`, `validation` and
   `features.json` only, under `ingest: {version, snapshot}`, with the path
   `<download_dir>/<version>/<snapshot>/`. `validation.parquet` gains targets as they
   resolve, so a snapshot date names which copy a run used.
5. **Freshness check at fetch time.** `validate_live` fails if the live frame's era
   date is more than 7 days before the fetch time `ingest_live` records. This catches
   a server returning an old file for the requested round. On a cache hit nothing is
   fetched, so nothing is checked: that is correct, because the round is pinned.
6. **`run_id`**: every `Pipeline.run` draws a uuid and stamps it on each manifest it
   writes. (Core, a few lines.)

Generic pieces (`run_id`, the predict step, the serve wiring in `cli.py`) go in `core/`,
`steps/` and `cli.py`. The round, the paths and the freshness rule stay in
`examples/numerai/`.

## Consequences

- Each round adds one live file (~tens of MB) to the download cache and one snapshot
  to the store. `gc --dry-run` already reports old snapshots once they leave the
  retention window.
- **Human migration:** move the existing `/workspace/raw_data/numerai/v6.0/{train,
  validation}.parquet` and `features.json` into `v6.0/2026-10-20/` (their download
  date). Raw bytes are unchanged, so snapshot hashes stay the same; only ingest's
  signature moves, plus whatever follows from the config change. Removing the
  existing `live.parquet` is a human step.
- The example's training pipeline loses `raw_live` and `validated_live`, so
  validate's outputs and the cycle-1/2 tests change.
- `predict` gains a manifest and a store write per call. That is the point.
- numerai keeps its own round loop and submission path. It can reuse the pattern, but
  this ADR binds only mlpipe's example and the generic serve wiring.

## Open, to check before building

- **Past rounds:** does `NumerAPI().download_dataset("v6.0/live.parquet",
  "/tmp/live_old.parquet", round_num=<a round from last week>)` return that round's
  file? This decides whether a backfill can be rebuilt. It needs credentials-free
  network access, so it is a host-side check.
- **Validation rewrites:** whether Numerai rewrites `validation.parquet` in place within
  v6.0. Decision 4 is safe either way; the answer only changes how often a new
  snapshot is worth taking.

## Alternatives considered

- **sha256, size or mtime of the local file in the signature.** Rejected: nothing
  triggers the refresh (see Context).
- **Re-download in place, gated by a round ledger.** Rejected: it overwrites a path
  that other readers share, and ties correctness to gitignored state.
- **mlpipe resolves the current round itself.** Rejected: the signature would depend
  on a network call, and a cached run would mean different things on different days.
