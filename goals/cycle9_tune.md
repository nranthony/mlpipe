# Cycle 9 — tune (Optuna)
Goal: Optuna study over the model config subtree; each trial is a full pipeline run
(cache makes upstream free); study storage in SQLite; best trial promoted via
baselines.yaml. Accept: trials appear as MLflow runs tagged with study + trial ids;
interrupted study resumes; search space is declared in config, not code.
