# Cycle 7 — evaluate + MLflow wiring
Goal: era-wise metrics (mean corr, sharpe, max drawdown, feature exposure) as plain
functions; plotly HTML report artifact; MlflowTracker impl (SQLite backend, ./mlruns)
logging params/metrics + artifact pointers. Accept: MLflow shows two runs comparable
side by side; deleting mlruns/ loses zero provenance (manifests still reconstruct
everything); report artifact renders standalone.
