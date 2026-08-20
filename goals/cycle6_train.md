# Cycle 6 — train (LightGBM backend)
Goal: TrainStep written once against the ModelBackend port; LightGBMBackend adapter
(~50–100 lines). Outputs: model artifact + oof_preds. Accept: TrainStep source
contains no lightgbm import; per-fold training respects the FoldPlan; model artifact
reloads and reproduces oof predictions; VRAM/RAM stays in budget on full data.
