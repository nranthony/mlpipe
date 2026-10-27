# Proposal: extract the core as a library (not a template), with numerai as first consumer

- Status: Accepted → ADR-0004, ADR-0005, ADR-0006 (2026-10-19). Answers: no new
  repo; this repo *is* the library, with Numerai moved to `examples/`. Execution:
  [plan.md](plan.md)
- Author: Claude (agent, windows-ai-sandbox)
- Date: 2026-10-19
- Related: [work/0004](../archive/0004-consistency-review-and-orchestrator-question/proposal.md)
  (findings A–C are prerequisites here), DESIGN.md §7 (the extraction plan this would
  amend), numerai repo `work/0010-mlpipe-spine-adoption/spec.md` (the consumer side)

## Summary

Two findings from a 2026-10-19 review change the shape of DESIGN.md §7.

1. **The Numerai host is not this repo.** `/workspace/numerai` has its own
   pipeline (`numerai_client`, ~2.7k lines) that never references mlpipe. It is
   stronger on the domain than `mlpipe/steps/`: explicit `target_ender_20` (its
   ADR-0006), a CORR+MMC payout-aligned objective, a frozen ask-gated holdout, a
   working promotion gate → champion → git commit, dry-run submission, a round
   ledger. mlpipe is stronger on the substrate: CAS, signature caching, manifests,
   lineage, `--from`. So the right split is **numerai keeps its domain pipeline;
   mlpipe supplies the spine underneath it**. `mlpipe/steps/` stays as a
   reference implementation, not something to port across.
2. **Ship the core as a pip library, not as template-owned files.** External
   practice (Kedro framework/starter/project split; ZenML templates; Kedro
   deprecating micro-packaging for "pip-install the library") is consistent:
   the engine is a versioned dependency; templates carry scaffolding only.
   Distributing `core/` through Copier turns every core fix into a three-way
   merge in every host repo.

## Motivation

Current DESIGN.md §7 says: extract to a Copier template after several real
iteration loops, with `core/**` and `cli.py` as template-owned zones
(TEMPLATE_OWNERSHIP.md). Problems:

- The "real iteration loops" are happening in the numerai repo, on a different
  pipeline — so the precondition can never be met here as written.
- Template-owned code is the high-maintenance way to share an engine (see above).
- No second domain has exercised the core; its genericity is asserted, not proven.

## Proposal

Target shape:

| Piece | Form | Contents |
|---|---|---|
| `mlpipe-core` | pip package, git-tag versioned, pinned in each host's lockfile | `core/` (store, manifest, signature, context, pipeline, interfaces), `cli.py` |
| host project | its own repo | domain steps + backends + config, registered via entry points |
| `mlpipe-template` (optional, later) | Copier template | AGENTS.md/DESIGN stub, ADR + work/ conventions, config skeleton, CLI wiring, CI |

Prerequisites, in order (each is a small, testable change):

1. **Fix work/0004 A–C first** — promotion (one mechanism, read by something;
   MLflow `@champion` alias + `baselines.yaml` pinning config/plan/bundle hashes),
   top-level `PipelineConfig` pydantic model, `--from` resolving upstream from the
   current config's signatures. Otherwise every consumer inherits them.
2. **Domain-neutral core.** `FoldPlan` (`core/interfaces.py:35-39`) is defined as
   "era-aware" with `era_column`; generalise to `group_column` and move era
   semantics into `steps/cvplan.py`.
3. **Tracker out of core.** Move `core/mlflow_tracker.py` to an optional adapter
   (`mlpipe.trackers` or `backends/`). Frees budget (core at 697/700).
4. **Pipeline discovery via entry points.** Replace module-name lookup of
   `--pipeline` with `importlib.metadata.entry_points(group="mlpipe.pipelines")`
   (and `mlpipe.backends` for the backend registry), so a host package registers
   its `build_steps()` + config without core importing it. Stdlib only.
5. **Second-domain smoke test.** A tiny non-Numerai pipeline (plain tabular,
   group k-fold) in `tests/` proving the core runs with no era/Numerai concepts.
6. **Package split.** Separate distribution for the core (same repo, or a new
   repo — open question). Tag `v0.1.0`.
7. **ADR** superseding the template-owned-core part of DESIGN.md §7 and updating
   TEMPLATE_OWNERSHIP.md (`core/**`, `cli.py` move from "template-owned" to "library").

Human steps: creating/pushing a remote and tags (no remote git in the sandbox);
any new tooling (Copier) is a dependency → ADR + host-side install.

## Open questions

1. Accept the direction (numerai keeps `numerai_client`; mlpipe = spine library)?
2. Core package: same repo with two distributions, or a new `mlpipe-core` repo?
3. Do 0004 A–C here first, or resolve 0004's open questions separately and then
   start this?
4. Copier template: now, later (after numerai adopts the library), or never?
5. What happens to `mlpipe/steps/` Numerai code once numerai adopts the core —
   keep as an example pipeline, or retire it?

## Alternatives

- **Keep DESIGN.md §7 as is (template-owned core via Copier).** Rejected in the
  recommendation: core fixes become merges in every repo; no consumer exists.
- **mlpipe absorbs numerai's domain code** (port scorer/holdout/loop/submit into
  `mlpipe/steps/`). Rebuilds working, more correct code; discards numerai's Hydra
  decision and its ask-gated fitness surface.
- **Keep the two repos independent.** Zero cost now; numerai never gets caching or
  lineage, and mlpipe's core never meets a real consumer.
- **Adopt a framework instead** (ZenML/Kedro/Hamilton). Already assessed in work/0004
  "buy-vs-build"; not reopened here.
