# Handoff template — for an agent in a repo that may adopt mlpipe

> **How to use this file.** Copy it into the other repo (suggested:
> `work/NNNN-mlpipe-alignment/handoff.md`, or whatever that repo's in-flight convention
> is), fill in the `<…>` placeholders, and delete this quote block. Everything
> outside the placeholders is meant to be true for any host repo. Source:
> `mlpipe/work/0006-core-library-extraction/` (written 2026-10-19). If that
> proposal's status has moved on, update the status table before copying.

---

# Handoff — aligning `<repo name>` with the mlpipe pipeline

- For: any agent working in this repo
- Written: `<date>`, from `<where / who>`
- Upstream source of truth: `mlpipe/work/0006-core-library-extraction/proposal.md`
  (status at time of writing: **Draft, nothing decided**)

## What mlpipe is

A small, local MLOps spine (about 700 lines of core):

- **Content-addressed artifact store.** Every output is stored by the SHA-256 of
  its bytes and is immutable.
- **Steps.** Pure functions from named input artifacts to named outputs. They
  touch the outside world only through a `RunContext`.
- **Signature caching.** A step reruns only when its code closure, config
  subtree or input hashes change. `run --from <step>` reuses upstream outputs.
- **JSON run manifests.** One per step execution, append-only. Lineage is
  walkable from any model back to raw data, config and git commit.
- **Ports.** `ArtifactStore`, `Tracker` (MLflow as a view only) and
  `ModelBackend`, with pydantic-validated config.

The first pipeline built on it was Numerai, but that was never adopted by
the real Numerai repo. That repo kept its own pipeline.

## What may come, and what it means here

| Possible arrival | Status | Shape | What it touches in this repo |
|---|---|---|---|
| **`mlpipe-core` library** | Proposed. Needs mlpipe-side fixes, package split, version tag, human push | Pinned dependency (git URL + tag) in `pyproject.toml` | Your domain steps register through entry points (`mlpipe.pipelines`, `mlpipe.backends`). Core never imports your code |
| **Copier template** | **Undecided** (now / later / never) | `.copier-answers.yml` + `copier update`, driven by a `/template-sync` procedure | Scaffolding only: agent docs, ADR and `work/` conventions, config skeleton, CLI wiring. **Never engine code** |
| **Reference steps / backends** (int8 polars loading, LightGBM/Torch backends, purged group CV, plan-hash guard) | Available to read in mlpipe; not packaged | Copied in as individual, reviewed changes if wanted | Only what you choose |

**Ownership principle:** this repo stays the owner of its domain pipeline. That
means its data sources, evaluation and scoring, promotion rules and
deployment or submission. mlpipe supplies caching, lineage and reproducibility
underneath, not domain logic.

## Rules until this repo signs off on adoption

1. **Change nothing because of this handoff.** No new dependencies (`mlpipe-core`,
   `polars`, `copier`). Each needs this repo's decision record plus a human
   install.
2. **Keep these boundaries, which are what make adoption cheap:**
   - Data loading, cleaning, feature fitting and training are separable
     functions. They take inputs and return outputs, without hidden file
     reads or writes in the middle.
   - Fitted state (scalers, feature selections) is fitted on training data
     only and saved alongside the model.
   - Evaluation and scoring, and any frozen holdout, stay outside anything
     that might become a cached step.
   - Config goes through one validated schema (pydantic, or a layer that
     produces pydantic).
3. **Don't build a parallel caching or lineage system** (artifact hashing, run
   manifests, a step cache) without first checking the upstream proposal.
   That is the piece mlpipe would supply.
4. **Keep current provenance as it is** until a decision says which record is
   authoritative.

## Decisions this repo will face if adoption goes ahead

Record each one as a decision in this repo before writing code:

1. **Adopt at all**, versus staying independent. Staying independent is a
   legitimate outcome.
2. **DataFrame engine.** mlpipe passes polars between steps and stores Parquet.
   If this repo uses pandas or something else, choose between converting at
   step boundaries only and migrating.
3. **Provenance of record.** Either mlpipe manifests, with existing sidecars or
   metadata generated from them, or the existing system, with manifests as a
   cache index only.
4. **Tracker layout.** MLflow (local, view-only): share one tracking database
   with any existing experiment loop, or keep them separate.
5. **Config.** If this repo uses Hydra or similar, keep it in front and pass
   only validated pydantic objects to step config subtrees.
6. **Template (if it ships).** Whether to adopt Copier scaffolding, and which
   local docs to keep where they conflict.

## If you're asked to start adoption

Check the upstream proposal's status first. If it isn't **Accepted** with a
tagged `mlpipe-core` release, stop and say the work is blocked upstream. If it
is, open the decisions above, then wrap one expensive, deterministic stage
(data preparation or feature fitting) as the first step. Prove a cache hit on
rerun before converting more stages.

## If the Copier template arrives

- Expect a `.copier-answers.yml` file (never edit it by hand) and a
  `/template-sync` procedure.
- Ownership zones will be published in mlpipe's `TEMPLATE_OWNERSHIP.md`.
  Don't guess them in advance.
- Default conflict policy: files owned by the template take the template's
  version unless tests break; skeleton files keep your local version; files
  owned by this repo are never touched.
- Anything you've tailored, such as configs, project sections of the agent
  docs and domain code, stays yours.

## Repo-specific notes

- Current pipeline entry point(s): `<e.g. src/pkg/pipeline.py, scripts/train.py>`
- Existing provenance / tracking: `<e.g. pickle sidecars, MLflow at …>`
- Areas that must never become cached steps: `<e.g. scorer, holdout, deploy>`
- Known blockers or human steps: `<e.g. registry closed in sandbox>`
