# Cycle 4 — feature engineering
Goal: fitted transforms (sklearn fit/transform protocol) learned from train only.
Outputs: feature_table AND fitted_transformer (joblib artifact — it must travel with
the model; this is the train/serve-skew guard). Accept: transformer round-trips via
the store; applying the loaded transformer to validation equals the pipeline's own
validation features; leakage test — fitting statistics computed on train partition only.
