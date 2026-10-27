# Proposal: re-decide the Numerai example (ADR-0009 trigger) and make feature selection configurable

- Status: Draft (2026-10-27). Opened at nanthony's "yes" to joining the two items in
  the wrap-up. Needs answers to the open questions; on acceptance an ADR supersedes
  ADR-0009 decision 5 and, if option B or C, ADR-0007 decisions 4–7.
- Author: Claude
- Related: [ADR-0009](../../docs/adr/0009-numerai-example-is-a-real-data-harness.md)
  decision 5 (the trigger); [ADR-0007](../../docs/adr/0007-serve-pipeline-and-named-inputs.md)
  decisions 4–7; [work/0011](../0011-numerai-separation-audit/handoff.md) finding A
  ("shrink the example") and the "one spec" (findings B High/Med); work/0006 step 14
  (done 2026-10-23, confirmed 2026-10-27).

## Summary

ADR-0009 kept the Numerai example as mlpipe's real-data harness until the numerai repo
ran on mlpipe's core with a real-data cache hit. That has happened. Its decision 5 asks
one question: can the numerai repo now be mlpipe's real-data check? If yes, the
example shrinks to a synthetic fixture. If no, the harness stays and the new ADR says
why.

The honest answer is **partly**. This proposal recommends option C below: keep a
smaller real-data harness for the generic training steps, and drop the parts the
numerai repo already covers. The same item lands work/0011's "one spec", because
moving `cvplan` and freeing feature selection touch the same files.

## What each real-data check covers today

| Surface | numerai repo (pins a tag) | numerai-example | titanic-example |
|---|---|---|---|
| Store, signatures, manifests, lineage, entry point | **yes, at scale** (3.6 GB snapshot, stages A–B) | yes | yes |
| Generic `train`, `tabular_eval`, `register`, `features` | no (its own `cv.py`, loop and scorer) | **yes, at scale** (2.7M rows) | yes, tiny (891 rows) |
| `cvplan` (era purge and embargo) | no (its own folds) | yes | no (`groupcv`) |
| `predict` and serve lists (ADR-0007) | not yet (stage E) | yes, by round (numerapi) | yes, file-pinned |
| `tune`, `promote` | no | yes | no |
| `MlflowTracker` | no (stage C plans its own adapter) | yes | yes |
| When a change is proven | **after** a tag, unless pinned to a candidate commit | before release | before release |

So the numerai repo proves the core at scale, but not the generic steps, and only
after release.

## Options

- **A. Shrink fully (ADR-0009's default at the trigger).** The example becomes a
  synthetic Numerai-shaped fixture. numerapi, the raw-data path and round-keyed live
  serving go (ADR-0007 decisions 4–7 superseded). Generic steps keep only Titanic as
  real data.
  - Lost: the only scale test of `train`, `tabular_eval` and serving. The work/0013
    per-fold eval sets are exactly the kind of change that needs one (memory scales
    with rows).
- **B. Keep the harness as is** (ADR-0009's other branch). Record why: the numerai
  repo doesn't use the generic steps. The divergence and agent-confusion costs that
  ADR-0009 listed remain.
- **C. Keep a training-only real-data harness (recommended).**
  - Keep: v6.0 train/validation ingest by snapshot date (ADR-0007 decision 5), clean,
    features, cvplan, train, evaluate and register on real data. That is the scale
    test for the generic steps.
  - Drop: live serving by round and the freshness check (ADR-0007 decisions 4 and 6)
    and the `numerapi` call. Serve-list wiring is proven by Titanic (file-pinned) and
    by the numerai repo's stage E when it lands.
  - The example then never touches the Numerai API. The download folder becomes
    read-only input placed by a human, so `numerapi` leaves the library entirely,
    including the `numerai` extra.
  - Add a release check: before tagging, the numerai repo pins the candidate commit
    and runs its spine once. That makes it a pre-release check for the core, which is
    what decision 5 asked for.

## The "one spec" (work/0011 B High/Med), in this item for every option

1. **Feature selection is configurable.** Today it is the `feature_` prefix in
   `steps/features.py:49`, `train.py:83`, `tabular_eval.py:106` and `register.py:43`,
   and Titanic renames its columns to fit (`examples/titanic/clean.py:35`).
   - Proposed: `features: {prefix: "feature_"} | {columns: [...]} | {exclude: [...]}` on
     `FeatureStep`'s config. It emits a `feature_list` artifact that train, evaluate and
     register read, instead of each re-applying the prefix.
   - The default stays `prefix: feature_`, so existing signatures move once (the code
     changes) but outputs stay byte-identical.
   - Test: a domain whose columns don't start with `feature_`.
2. **`cvplan` moves to `examples/numerai/`.** Era purge and embargo are Numerai
   conventions; `groupcv` is the generic step.
3. **The B Med rows:**
   - fixed slot names (`features.py:41`);
   - per-fold Pearson `fold{i}_corr` in train (`train.py:103`), which should use the
     task's metric from `tabular_metrics.TASKS`;
   - the fixed `prediction` column (`train.py:123`).
4. **DESIGN §2 and §8** drop the Numerai facts that remain (work/0011 D1–D2).

## Cost

- Option C or A: the example's ingest and serve code shrinks. Its next full run
  retrains once (~30 min), which the work/0013 changes already require. Doing both in
  the same release costs one retrain, not two.
- The "one spec": every pipeline's features, train, evaluate and register re-run once.
  Titanic takes seconds; numerai-example is folded into the same retrain.
- The numerai repo uses neither `cvplan` nor the generic steps, so nothing changes
  there.

## Open questions

1. **A, B or C?** (Recommended: C.)
2. **The pre-release check in C:** is pinning numerai to a candidate commit before each
   tag acceptable as a standing release step? It is a host-side `uv add` of a commit,
   then reverting to the tag.
3. **Order:** the "one spec" first (library API, affects every host), then the example
   change? Or both in one release (one retrain)? (Recommended: one release, `v0.4.0`.)
