# Cycle 7 — evaluate + MLflow wiring
Goal: era-wise metrics (mean corr, sharpe, max drawdown, feature exposure) as plain
functions; plotly HTML report artifact; MlflowTracker impl (SQLite backend, ./mlruns)
logging params/metrics + artifact pointers. Accept: MLflow shows two runs comparable
side by side; deleting mlruns/ loses zero provenance (manifests still reconstruct
everything); report artifact renders standalone.

Since ADR-0009: the era-wise metrics are the Numerai harness's
(`examples/numerai/eval_metrics.py`). The generic counterpart is `steps/tabular_eval.py`
(row-wise r2, rmse, roc_auc, ...).
