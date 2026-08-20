# Cycle 2 — validate
Goal: pandera schemas as a hard gate. Check: expected columns, dtypes (int8 features),
null bounds, era column present and monotonic, row counts within tolerance of the
previous snapshot. Outputs: validated_* (pass-through artifacts) + validation_report
(JSON). Accept: corrupting a fixture (drop a column / poison dtypes) fails loudly with
an actionable message; clean data passes and caches.
