# Cycle 10 — register & package
Goal: registered model version = bundle of model + fitted_transformer + feature list
+ plan hash + config hash, via MLflow registry. Accept: a fresh process can load the
bundle by name/version and produce live predictions from raw live data with no access
to training internals; lineage from the registered version walks back to the raw
snapshot hash.
