# ADR-0002: Pipeline contract amendments discovered during the build

- Status: Proposed
- Date: 2026-08-21
- Deciders: nanthony + Claude (build cycles 3-6, 2026-08-20)

## Context

DESIGN.md §2 fixes each block's contract (inputs → outputs) and §0 forbids
relitigating decisions without an ADR. During the build, three deviations from
the §2 table proved necessary or clearly better. They were recorded in commit
messages and goals/STATUS.md at the time; this ADR is the durable record.

## Decision

1. **Clean emits `clean_validation` alongside `clean_table`.** Cycle 4's
   acceptance test ("applying the loaded transformer to validation equals the
   pipeline's own validation features" — the train/serve-skew guard) requires
   cleaned validation data. The same stateless expressions are applied to
   `validated_validation`; statelessness and byte-determinism are unchanged.

2. **Features emits `feature_validation` alongside `feature_table` and
   `fitted_transformer`.** Same driver: the skew guard needs pipeline-produced
   validation features to compare against, and evaluation-era work needs
   transformed validation data without refitting anything.

3. **The model artifact is a bundle `{full, fold_models}`, not a single
   model.** Cycle 6's acceptance ("model artifact reloads and reproduces OOF
   predictions") is only satisfiable if the per-fold backends travel with the
   full-data model: OOF predictions come from fold models, so stored state
   alone must contain them. Serving uses `bundle["full"]`; reproduction uses
   `bundle["fold_models"]`.

## Consequences

- DESIGN.md §2's table is amended by this record rather than edited in place;
  at template-extraction time the table should be regenerated to match.
- Validation tables flow through two more artifacts per run (aliased/cheap for
  clean; real bytes for features), enlarging the store slightly.
- Model artifacts are ~5x larger (fold models included) but self-sufficient
  for audit: OOF reproducibility needs no retraining.

## Alternatives considered

- **Separate validation-transform step:** rejected — it would re-implement
  FeatureStep's transform path and split the skew guard across two steps.
- **Store only the full model, regenerate OOF by retraining folds:** rejected —
  "reproduce from stored state" is the point; retraining reproduces only if
  code/config/data are all still identical, which is what we refuse to assume.
- **Editing DESIGN.md §2 silently:** rejected — §0 requires an ADR, and the
  design doc's authority depends on deviations being append-only records.
