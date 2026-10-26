# Cycle 5 — split / CV plan
Goal: era-aware purged + embargoed CV producing a FoldPlan artifact with its own hash.
Accept: no era appears in both train and valid of any fold; purge/embargo widths
honored; plan_hash changes iff cv config or input eras change; a helper asserts two
runs share a plan hash before any comparison is reported.

Since ADR-0009: the era, purge and embargo terms are the Numerai harness's
(`numerai-example`). The generic counterpart is `steps/groupcv.py` (unordered groups).
