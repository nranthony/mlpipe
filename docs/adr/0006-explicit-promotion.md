# ADR-0006: Promotion is an explicit `mlpipe promote` verb

- Status: Accepted (implementation pending — work/0006 plan, step 4)
- Date: 2026-10-19
- Deciders: nanthony + Claude (work/0004 finding A, work/0002 gap 3)

## Context

DESIGN.md §1 defines promotion as tagging the run in MLflow and pinning its config
hash in `baselines.yaml`. In the code, promotion is split across three partial
mechanisms that disagree:

- `mlpipe tune` overwrites `baselines.yaml` with only the searched params. It
  leaves out `n_estimators` and the config, plan and bundle hashes.
- Nothing reads `baselines.yaml`.
- `RegisterStep` creates a registry version on every fresh run.

The result is that the "pinned baseline" can't be reproduced and can't be compared
with anything.

## Decision

- **`mlpipe promote <bundle-hash>`** is the only way to promote. It:
  - checks that the bundle exists in the store and that its lineage resolves;
  - moves the MLflow alias `@champion` on the registered model to that bundle's
    version;
  - writes `baselines.yaml` with the full resolved config, `config_hash`,
    `plan_hash`, bundle hash, key metrics and git commit;
  - appends a manifest entry, so the promotion is part of the lineage.
- **`register` no longer auto-registers on every run.** It produces the bundle
  artifact only. Registry versions are created when a bundle is promoted.
- **`tune` stops writing `baselines.yaml`.** It reports the best trial's bundle hash
  for a human or agent to promote.
- Manifests and the store remain the source of truth. If `mlruns/` is lost, the
  alias can be rebuilt from `baselines.yaml`.

## Consequences

- `baselines.yaml` becomes something the pipeline actually reads. The comparison
  rule in AGENTS.md ("cite the fold-plan hash") can be applied to the baseline.
- There is one more CLI verb, which counts toward the CLI, not the core.

## Alternatives considered

- **Keep auto-registration and have `tune` write the baseline.** That is the current
  inconsistent state.
- **MLflow alias only.** It would make MLflow the source of truth, which goes
  against DESIGN.md §0.
