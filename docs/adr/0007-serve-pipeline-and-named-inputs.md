# ADR-0007: Serving is a pipeline; inputs that change over time are named in config

- Status: Proposed
- Date: 2026-10-23
- Deciders: nanthony + Claude (work/0009; findings from an external review of the core,
  numerai `work/0012-ml-core-review` D4, P0-2, P1-18, mlops C9/C14/C15)

## Context

Two library gaps, both found through the Numerai example:

- **A cached step can't see that its source changed.** A step's signature is computed
  before `run()`. If the source is a location whose contents change over time (a weekly
  file, a table, an API) and nothing in the config says *which* contents, the signature
  never changes and the step stays `cached`. Hashing the local file (sha256, size or
  mtime) doesn't help: on a cache hit the step doesn't run, so nothing refreshes it.
  (STATUS.md suggested size + mtime on 2026-10-22; that was wrong.)
- **Serving leaves no record.** `mlpipe predict` writes no manifest, so a prediction
  file can't be traced to its bundle. Its default input comes from a pipeline's
  `serve_input(log, store)` hook, which can only pick from what earlier runs left behind.
- **No `run_id`**: one run's manifests can be grouped only by timestamp.

How it shows in the Numerai example: `examples/numerai/ingest.py` `fetch` skips any file
that exists, and ingest's signature is `{version}` + code, so a new round's
`live.parquet` is never downloaded. The example's `serve_input` returns the last live
snapshot any run wrote ("latest ever"), whatever its round or tag.

## Decision

### Library (`core/`, `steps/`, `cli.py`)

1. **Inputs that change over time are named in config, never discovered.** A step that
   reads such a source takes an identifier in its config subtree (a round, a date, a
   release) and reads from a location that identifier names. The caller (a human, a
   host's scheduler) resolves "current" and passes it with `--set`. Signatures stay
   computable offline, and two identifiers never share a cache entry. mlpipe never asks
   the outside world what is current.
2. **Serving is a pipeline.** A pipeline module may declare serve steps beside its
   training steps. `mlpipe predict` runs them with the pinned bundle as a seed
   (`Pipeline.run(seeds=…)`, which exists), logged to the same manifest log as training.
   `predictions` is a store artifact with a manifest, so its lineage reaches the bundle
   and, through it, the training run.
   - A generic `predict` step goes in `steps/`; it absorbs the `prepare` hook.
   - The `serve_input` hook is removed.
   - **`--input <parquet>` stays** for ad-hoc scoring. The file is snapshotted into the
     store and seeded in place of the serve steps' ingest output, so it gets a manifest
     too.
3. **`run_id`**: every `Pipeline.run` draws a uuid and stamps it on each manifest it
   writes. (Core, a few lines.)

### Numerai example (`examples/numerai/` only)

4. **Live data is named by round.** `ingest_live: {version, round}`, `round` required.
   The path is `<download_dir>/<version>/live/<round>/live.parquet`, fetched with
   numerapi's `round_num=<round>` and never overwritten. A new round is a new path, so
   neither the exists-check nor numerapi's own size check can return an old file.
   Serve steps: `ingest_live` → `validate_live` → `predict`.
5. **Training data is named by snapshot date.** `ingest` takes `train`, `validation`
   and `features.json` only (no live), under `ingest: {version, snapshot}`, with the
   path `<download_dir>/<version>/<snapshot>/`. `validation.parquet` gains targets as
   they resolve, so the date says which copy a run used. A snapshot folder that doesn't
   exist yet is downloaded on first use.
6. **Freshness check at fetch time.** `validate_live` fails if the live frame's era
   date is more than 7 days before the fetch time `ingest_live` records. This catches
   a server returning an old file for the requested round. On a cache hit nothing is
   fetched, so nothing is checked: that is correct, because the round is pinned.
7. **The download folder may be shared**, so the example only adds paths to it. It
   never moves, rewrites or deletes a file it didn't create. Existing files at the old
   flat paths (`<version>/train.parquet`, `<version>/live.parquet`, …) are left alone.

## Consequences

- Library: `cli.py`'s predict changes shape, and pipeline modules that used
  `serve_input`/`prepare` move to serve steps. The demo pipeline needs a serve list
  too, or `predict --input` only.
- Every predict call adds a manifest and a store write. That is the point.
- Example: each round adds one live file (~tens of MB) to the download folder and one
  snapshot to the store. `gc --dry-run` reports old snapshots once they leave the
  retention window.
- Example: the first run with a `snapshot` date downloads train and validation again
  (~2.5 GB) into the new folder. A human may copy the existing files there instead, to
  save the download. Only then do the raw snapshot hashes stay the same.
- Example: the training pipeline loses `raw_live` and `validated_live`, so validate's
  outputs and the cycle-1/2 tests change.
- Hosts keep their own scheduling and submission. This ADR binds the library's serve
  wiring and this repo's example, nothing in a host.

## Open, to check before building

- **Past rounds:** does `download_dataset("v6.0/live.parquet", …, round_num=<a past
  round>)` return that round's file? Checked 2026-10-23: numerapi's client sends
  `round` to the server's `dataset` query; whether the server honours it for live is
  unverified. It decides whether an old round's predictions can be rebuilt. Host-side
  check (network).
- **Validation rewrites:** whether Numerai rewrites `validation.parquet` in place within
  v6.0. Decision 5 is safe either way; the answer only changes how often a new snapshot
  is worth taking.

## Alternatives considered

- **sha256, size or mtime of the local file in the signature.** Rejected: nothing
  triggers the refresh (see Context).
- **Re-download in place, gated by a ledger of what was fetched.** Rejected: it
  overwrites a path other readers share, and ties correctness to gitignored state.
- **mlpipe resolves "current" itself.** Rejected: the signature would depend on a
  network call, and a cached run would mean different things on different days.
- **Move the existing flat files into a snapshot folder.** Rejected: the download folder
  is shared, and other readers use those paths.
