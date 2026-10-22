# Proposal: Live-data identity and a serve pipeline with lineage

- Status: Draft
- Author: Claude, from numerai `work/0012-ml-core-review` §5 (P0-2, P1-18; mlops C9, C14,
  C15), re-checked here 2026-10-22
- Decision on acceptance: an ADR "Live data is identified by round" (review D4)

## Summary

Every round's live data gets a path and a step config of its own, keyed by round number.
Serving becomes a small pipeline that leaves manifests: `ingest_live` → `validate_live`
→ `predict` (bundle seeded by hash). Every `Pipeline.run` gets a `run_id`.

## Motivation

- **Stale live (P0 for the example).** `examples/numerai/ingest.py:24-31` skips the
  download when the file exists, and ingest's signature is `{version}` + code. A new
  round's `live.parquet` is never fetched, and ingest stays cached. numerapi's own
  download also skips when the size matches. The cached file dates from the 2026-10-19
  round; rounds are daily.
- **The fix recorded in STATUS.md on 2026-10-22 (size + mtime in the signature) is
  wrong.** The signature is computed *before* `run()`, so a cache hit means `fetch()`
  never runs and nothing refreshes the file. mtime isn't content, since bind-mount copies
  change it, and neither carries round identity.
- **`serve_input` serves "latest ever".** It takes `latest_outputs()["raw_live"]`
  (`examples/numerai/pipeline.py:91-95`): the last live snapshot written by any run,
  whatever its round or tag.
- **`predict` leaves no manifest** (`cli.py`), so a submission has no lineage.
- **No `run_id`**: a dashboard can't group one run's manifests except by timestamp.

## Proposal

1. **Round-keyed path, immutable.** `<download_dir>/<version>/live/<round>/live.parquet`,
   never overwritten. A new round is a new path, so neither the exists-check nor
   numerapi's size-skip can fire. Fetch with an explicit `round_num=round`.
2. **The round is config.** `ingest_live: {version, round}`. Its signature stays
   computable offline, and two rounds can't share a cache entry. The host resolves the
   current round once and passes `--set ingest_live.round=N`; mlpipe never asks
   "current".
3. **Serve pipeline.** `ingest_live` → `validate_live` → `predict`. The bundle enters as
   a seed (`Pipeline.run(seeds=…)`, already supported), and `predictions` is a store
   artifact with a manifest. `mlpipe predict` becomes a thin wrapper over it.
   `serve_input` and its "latest ever" lookup go.
4. **Training snapshots name their date.** `ingest: {version, snapshot}`, because
   `validation.parquet` gains targets as they mature. Whether Numerai rewrites it in
   place within v6.0 is unverified; check docs.numer.ai before relying on either answer.
5. **Freshness guard** in `validate_live`: the live frame's era date must be within
   7 days of the round; fail closed.
6. **`run_id`** (uuid) per `Pipeline.run`, stamped on every manifest it writes.

The generic part (round-keyed snapshot steps, the serve pipeline shape, `run_id`) lives in
`steps/` and `core/`; the Numerai specifics stay in `examples/numerai/`.

## Open questions

- Can numerapi download a *past* round's live file with `round_num`? (Unverified; it
  decides whether a backfill can be rebuilt.)
- Should `mlpipe predict` keep a `--input <parquet>` escape hatch for ad-hoc scoring? It
  would write a manifest with the file hash as a seed. Recommend yes.

## Alternatives

- **sha256 of the local file in the signature:** right about bytes, but has the same
  flaw as size + mtime: nothing triggers the refresh.
- **Re-download in place, gated by a round ledger:** overwrites a path other readers
  share, and ties correctness to gitignored state.
