# 0010 spec — Store, registry and Numerai-example hygiene

- Status: **Built 2026-10-23** (go-ahead from Neil the same day). No item here is a
  wrong-result bug in mlpipe itself; each is cheap and makes the next host adoption or
  real-data run less surprising. Shipped after [work/0008](../archive/0008-cache-correctness/spec.md).
- Changed while building:
  - Item 1 copies, never hard-links. A link shares the inode with the download, and a
    source rewritten in place (work/0009's stale live file) would silently change a
    stored artifact. One streamed pass copies and hashes, so RAM stays flat; the disk
    copy remains.
  - Item 3 **dropped**, measured: `entries()` takes 0.55 ms on the real 74-entry log
    and ~100 ms on a synthetic 10,000-entry one. That is noise next to any step;
    revisit if a log passes ~100k entries.
  - Item 6 also applies to clean's `era_slice` (same first-appearance order).
  - Item 7 is in the example's `EvaluateStep` (the only era-wise evaluate); the generic
    `TabularEvalStep` keeps its fixed metrics. The port uses stdlib `NormalDist`, so
    no scipy dependency; a test matches it to a scipy reference to 1e-6.
  - The legacy `promoted:` block in `baselines.yaml` is left as written: it pins
    nothing, and removing it is a human call.
- Origin: numerai `work/0012-ml-core-review`. The review's P2-11 and P1-10 cover the
  store, registry and gc items. Its `reviews/datascience.md` §1d, §2 and P2-13 cover the
  example items. All were re-checked here 2026-10-22.

## Store, registry, gc

| # | Finding (verified) | Fix |
|---|---|---|
| 1 | **Snapshots read the whole file into RAM.** `LocalCasStore._serialize` does `Path.read_bytes()` (`core/store.py:28-29`). The v6 train snapshot is 2.18 GB in memory, then written as a second full copy on disk. | For a `Path`: stream sha256 in chunks, then `os.link` into the store, falling back to a streamed copy across filesystems. Stdlib only. Test: a large-ish temp file round-trips and the hash matches. |
| 2 | **`baselines.yaml` is one global slot.** `promote` writes `{"promoted": pin}` whatever `--alias` or `--pipeline` (`cli.py:188`). Promoting `staging`, or a second pipeline, overwrites `champion`. | Key it `{pipeline: {alias: pin}}`; `predict` and `gc` read the new shape. The current file is pre-ADR-0006 and roots nothing, so there is nothing to migrate. |
| 3 | **`entries()` re-parses the whole index on every lookup.** `find_cached` and `previous` both call it. | In-memory index keyed by signature, refreshed on write. Measure on the 74-entry real log first; skip if it's noise. |
| 4 | **A host's champion isn't a gc root.** A host that promotes by its own gate (numerai: gate + git) would see its champion listed unreachable after 30 days. | Document that host promotion writes a `promote` manifest (`{"step": "promote", "status": "promoted", "inputs": {"model_bundle": h}}`); add `gc --root-file <path>` for hash lists kept in git. |
| 5 | **Validate's report depends on history.** `previous_rows` goes into the `validation_report` artifact (`examples/numerai/validate.py`), so identical inputs can produce different report bytes. | Move `previous_rows` to manifest meta. |

## Numerai example fidelity

| # | Finding (verified) | Fix |
|---|---|---|
| 6 | **`cvplan` orders eras by first appearance.** `eras.unique(maintain_order=True)` (`steps/cvplan.py:30`). If an upstream step ever reorders rows, folds stop being contiguous in time and purging is silently wrong. | Sort the unique eras. ISO-date eras sort chronologically; numeric ids need a key, so take a `group_order` hook or sort by `(len, str)`. Pin the plan hash for already-sorted input. |
| 7 | **Its metric isn't Numerai's.** `eval_metrics.era_scores` is Pearson on rank-percentiled predictions (`examples/numerai/eval_metrics.py:10-19`), not `numerai_corr` (gaussianised ranks, `^1.5` tails). Per the review, on the same model class mlpipe reports exposure 0.302 against the host's 0.252 (not re-measured here). Ranking can diverge from payout. | Generic: `EvaluateStep` takes a scorer by import path (`evaluate.scorer: "pkg.mod:fn"`), hashed into the signature like `backend_code`. Rank-Pearson stays the default. The example adopts a `numerai_corr` port. Payout-accurate NCORR is the host's job. |
| 8 | **Full fit vs validation boundary.** The full model trains on all train eras, and its last 12 eras' 60-day labels overlap the first 12 validation eras' (about 3% of them). `val_*` metrics are slightly optimistic. | Drop the last `purge_eras` train eras from the full fit, or skip the first `purge_eras` validation eras when scoring. Prefer the latter: it doesn't change the model. |
| 9 | **The example's `val_*` would spend a host's holdout.** Every run scores all of validation. Fine here; a hazard if a host ever reused this evaluate. | Doc only: DESIGN.md notes that a host's frozen holdout must never be an input to a cached step. The review's adoption plan already keeps the host on its own scorer. |

## Done when

- Each item is fixed or explicitly dropped with a reason here.
- `pytest` is green and `ruff` is clean.
- After 6–8, one full v6.0 run under purge 12 gives the example's comparable baseline.
  Cite its plan hash in STATUS.md.
