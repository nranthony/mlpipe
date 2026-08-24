# Proposal: subset ("smoke") pipeline runs on real data

- Status: Draft — not started; capability work, not a build-defect fix
- Author: Claude (agent, windows-ai-sandbox)
- Date: 2026-08-24
- Related: [work/0002](../0002-post-build-followups/proposal.md) — separate item, but see
  "Timing coupling" below: the `clean.py` edit is free *now* and costs a full recompute later.

## Summary

There is no way to run the Numerai pipeline on a reduced slice of the data. Every real
run is the full 2.7M-row / 780-feature / 200-tree job (~64 min, dominated by training),
which is too slow for verifying plumbing after a code change. This proposes a cheap
smoke path: use the knobs that already exist, and add one missing one (era slicing).

## Motivation — measured, not assumed (2026-08-24)

**The data.** train 2,746,268 rows / 574 eras (0001–0574); validation 4,121,080 rows /
659 eras (0575–1233); live 6,997 rows / 1 era. 2,748 Int8 features, 41 target columns,
2,792 columns total. Train has zero null targets; validation drops ~35k rows on null
target during cleaning. Eras average 4,784 rows (2,314–6,009) — a uniform enough unit to
slice by, which matters because CV is era-blocked.

Named feature sets already wired to `clean.cleaner.feature_set`: intelligence 35,
small 42, dexterity 51, serenity 95, strength 135, wisdom 140, agility 145, midnight 244,
charisma 290, sunshine 325, constitution 335, faith 372, fncv3 400, rain 666,
**medium 780 (current default)**, v3_equivalent 1000, all 2748.

**Reads are not the bottleneck.** Parquet column pruning makes the data path nearly free
(scan → select → fill → collect, warm cache):

    train  small(42)   all 574 eras     0.1s   2,746,268 rows    131 MB in memory
    train  medium(780) all 574 eras     0.9s   2,746,268 rows  2,064 MB
    train  small(42)   tail 100 eras    0.1s     515,451 rows     25 MB

**Training is essentially the whole cost.** Single LightGBM fits at the default params;
the pipeline does 5 (4 folds + full):

    small(42)   tail100   20 trees   515k rows     5.8s/fit  ->  0.5 min
    small(42)   all       20 trees   2.75M rows   24.1s/fit  ->  2.0 min
    small(42)   all      200 trees   2.75M rows  144.4s/fit  -> 12.0 min
    medium(780) tail100   20 trees   515k rows    37.1s/fit  ->  3.1 min
    medium(780) tail100  200 trees   515k rows    89.2s/fit  ->  7.4 min
    medium(780) all      200 trees   2.75M rows  ~13 min/fit -> ~64 min  (cycle 6 baseline)

Feature count is a *fixed* cost (histogram binning: medium still takes 37s at 20 trees);
tree count is the *marginal* cost. Narrowing features is the biggest single lever.

## Proposal

**1. Document the zero-code smoke run** (works today, ~2-3 min end to end, 3 fits):

    mlpipe run --pipeline numerai \
      --set clean.cleaner.feature_set=small \
      --set cvplan.n_folds=2 \
      --set train.model.params.n_estimators=20 \
      --to evaluate --tag smoke

**2. Add `era_slice` to `CleanConfig`** — the one knob genuinely missing; nothing can cut
rows today. On `CleanConfig`, not the cleaner union: it is not cleaner-specific. Must be a
pure stateless polars filter, so clean's byte-determinism contract holds.

    clean:
      era_slice: {mode: tail, n: 100}     # modes: tail | head | stride

`tail` is the recommended default: eras stay **contiguous** (purge/embargo remain
meaningful) and it uses the most recent regimes. `stride` spans full history but breaks
contiguity — at stride 5, `purge_eras: 4` silently purges 20 real eras. Applies to both
train and validation.

**3. Optionally add `configs/smoke.yaml`** as a version-controlled preset instead of a wall
of `--set` flags. Note `--config` **replaces** DEFAULT_CONFIG wholesale (`cli.py:43`) — it
does not merge — so the preset must be a complete config or it silently drops
`download_dir`, tracker settings, etc. `configs/**` is already reserved as a skeleton zone
in TEMPLATE_OWNERSHIP.md.

## Rejected: row-subsampling within eras

`plan_hash` covers the era list and CV config only — deliberately invariant to row counts
(proven in cycle 5). A row-subsampled run therefore keeps the **same plan hash** as a full
run, passes `assert_same_plan`, and looks comparable to the pinned baseline while having
trained on a fraction of the data. Era slicing changes the hash honestly. Do not add row
sampling.

## Hazards to encode wherever this gets documented

- **Any slice invalidates comparison with the baseline** (different eras → different plan
  hash). Per AGENTS.md, say the comparison is invalid rather than report a winner. Smoke
  runs prove plumbing, not model quality.
- **Stop before `register`.** RegisterStep auto-registers every fresh run (gap 3 in
  work/0002), so a full smoke run would publish a 20-tree toy as `numerai-lgbm v2`. Use
  `--to evaluate` or override `register.model_name`.
- **Avoid `feature_set=all`**: FeatureStep materializes via `.to_numpy()` — 2,748 × 2.75M
  int8 ≈ 7.5 GB, and `standard_scaler` casts to float32 ≈ 30 GB against 47 GB of RAM. A
  SIGKILL leaves no failed manifest (AGENTS.local.md).
- **Store growth**: every distinct smoke config permanently adds artifacts; nothing GCs.

## Timing coupling with work/0002

The gap-1 signature fix already invalidated every cached signature, so the pipeline will
recompute ingest→register on its next real run regardless. Landing the `clean.py` era knob
**before** that run folds its invalidation into one recompute; landing it after buys a
second full recompute for no benefit.

## Open questions

- Do smoke runs deserve a `--tag smoke` convention (or a separate manifests dir) so they
  are trivially filterable out of experiment history?
- Should `era_slice` also apply to `raw_live`, or is live (1 era, 6,997 rows) always cheap
  enough to leave whole? Current thinking: leave it whole.
- Is a `configs/` preset worth it before the template extraction settles what lives there?
