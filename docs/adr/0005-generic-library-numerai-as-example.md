# ADR-0005: mlpipe is a generic library; Numerai becomes an example; template is scaffolding only

- Status: Accepted
- Date: 2026-10-19
- Deciders: nanthony + Claude (work/0006)

## Context

DESIGN.md called this repo the "Numerai host repo" and planned (§7) to extract a
Copier template later, with `core/**` and `cli.py` as files the template owns. Three
facts changed that:

- The real Numerai repo (`/workspace/numerai`) built its own pipeline and never used
  this one. Its domain layer is more complete. Tournament work lives there now.
- In common practice the engine is a versioned pip dependency and templates carry
  only scaffolding. Kedro, ZenML templates and Copier usage all work this way.
  Engine code shipped by template turns every fix into a three-way merge in every
  host repo.
- The core has only ever run one domain. Its claim to be generic is untested.

## Decision

1. **This repo is the engine.** The `mlpipe` distribution is `core/` + `cli.py` +
   generic adapters (model backends, trackers). Heavy libraries sit behind extras.
   Host repos install it from a git URL pinned to a version tag. No new repo and no
   second package.
2. **mlpipe's work is non-Numerai.** The Numerai pipeline moves out of
   `src/mlpipe/steps/` into `examples/numerai/` and stays as a real-data test case.
   It registers the same way any host repo would.
3. **Pipelines and backends are discovered through entry points**
   (`mlpipe.pipelines`, `mlpipe.backends`), not by module name inside this package.
4. **A Copier template, if built, carries scaffolding only**: agent docs, ADR and
   `work/` conventions, a config skeleton and CLI wiring. It is built only after a
   host repo has adopted the library.
5. Before release: the core is domain-neutral (`FoldPlan` uses `group_column`, not
   `era_column`), and a second, non-Numerai smoke-test pipeline passes.

This supersedes DESIGN.md §7's "template-owned core" plan and the "Numerai host repo"
framing in DESIGN.md and AGENTS.md.

## Consequences

- Host repos get core fixes through a version bump, not a merge.
- TEMPLATE_OWNERSHIP.md now describes what a future scaffolding template would own.
  `core/**` and `cli.py` become library code that host repos don't edit.
- Human steps: pushing a remote and tagging releases. Any `copier` adoption needs its
  own dependency ADR.
- The Numerai steps' module paths change, so their code closures change and every
  Numerai signature goes cold once more (one full recompute).

## Alternatives considered

- **Template-owned core (the old §7 plan).** Rejected. Every core fix becomes a merge
  in every repo.
- **A separate `mlpipe-core` repo or a second package.** Unnecessary once this repo
  stops being a Numerai project.
- **Port the numerai repo's domain code into mlpipe.** Rejected. That would rebuild
  working code and go against decision 2.
