"""Numerai-shaped real-data test harness (ADR-0005, ADR-0009). Not a tournament
pipeline: it never submits, and its scores are never compared with the numerai
repo's (the plan hashes differ).

    mlpipe run --pipeline numerai-example

Known divergences from the numerai repo, recorded rather than fixed (ADR-0009
decision 3; the numerai repo's version is the reference):

- correlation: a hand-port of `numerai_corr` (`eval_metrics.numerai_corr_scores`);
  the numerai repo calls `numerai_tools`.
- sharpe: `ddof=1` here, `ddof=0` there.
- target: `target` here (v6.0's alias for `target_jupiter_60`), named explicitly there.
- CV: `steps/cvplan.py` (purge 12, embargo 4) here; the numerai repo has its own fold
  arithmetic, so fold-plan hashes never match.
- live data: snapshotted into the store here; fetched per round, never
  snapshotted, there.
"""
