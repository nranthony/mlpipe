# Cycle 1 — ingest & snapshot
Goal: pull Numerai v5.2 data via numerapi, freeze as parquet artifacts, hash.
Outputs: raw_train, raw_validation, raw_live (+ features.json metadata artifact).
Record dataset name/version/size in the manifest. One scoped network pull; no other
API surface. Accept: re-running with unchanged upstream data is a cache hit; artifacts
load as polars with int8 feature dtypes intact.
