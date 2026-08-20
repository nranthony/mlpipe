# Cycle 3 — clean & transform
Goal: stateless ETL in polars expressions only. No fitted state — same input must
always produce the same output (that is what makes pure content caching valid here).
Output: clean_table. Accept: property test that two runs on identical input yield
byte-identical parquet; a config-selected alternative cleaner (discriminated union)
swaps in via --set cleaner=... and produces a different hash.
