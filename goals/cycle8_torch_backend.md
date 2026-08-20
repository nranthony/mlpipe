# Cycle 8 — second backend (Torch) — the port proof
Goal: TorchBackend adapter (MLP baseline): tensors/DataLoader/device handling fully
inside the adapter; centralized 12 GB VRAM guard; checkpointing to the store.
Accept: switching --set model=torch_mlp requires zero edits to TrainStep or configs
outside the model subtree; both backends evaluated on the same plan hash produce a
valid comparison in MLflow. Optional spike (throwaway branch): wrap two steps as ZenML
steps to verify the orchestrator exit door; record findings in an ADR, then delete.
