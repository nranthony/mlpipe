<!-- BEGIN sandbox-notice (hand-written; mirrors the windows-ai-sandbox profile's managed global instructions — do not edit here, fix the sandbox config instead) -->
## ⚠️ This repo may be edited by an agent inside `windows-ai-sandbox`

The agent's shell is restricted. The authoritative deny-list lives in the sandbox's
managed global agent instructions (loaded automatically on machines running the
sandbox); this block is only the standing reminder:

- **Denied and not retryable:** package installs (PyPI/npm are closed), arbitrary
  network, remote git (`push`/`pull`/`fetch`/`clone`, `gh`), shell escapes
  (`bash -c`, `python -c`, …), and destructive commands. A denial is policy, not a
  transient error — **reframe it as a human step and ask**, don't hunt for a bypass.
- **What works:** read/edit files, local git (`add`/`commit`/`diff`/`log`), running
  tests and builds against an already-provisioned environment, `rg`/`find`/`jq`,
  web reads via the `webfetch` broker, GPU checks via `/usr/lib/wsl/lib/nvidia-smi`.
- New dependencies additionally require an ADR (status: Proposed) before the human
  installs anything — see [docs/adr/](docs/adr/).
<!-- END sandbox-notice -->

# mlpipe

A local, modular, agent-operated MLOps pipeline (Numerai host repo, later extracted as
a Copier template). [DESIGN.md](DESIGN.md) is the authoritative system design — read it
first. Work proceeds in cycles defined by `goals/cycleN_*.md`, one cycle at a time.

## Start here (session ritual)

1. Read [DESIGN.md](DESIGN.md) if not already in context.
2. Check [goals/STATUS.md](goals/STATUS.md) for the current cycle and its state.
3. Check [work/](work/) for in-flight items and [docs/adr/](docs/adr/) for decisions
   constraining the area you're touching.
4. If `.copier-answers.yml` exists, compare its pinned template version against the
   template repo's latest tag; if behind, propose `/template-sync` before new work.

## Where things live

- System design & boundaries → [DESIGN.md](DESIGN.md) (this repo's architecture doc)
- Cycle roadmap + status/work log → [goals/](goals/) (`STATUS.md` is the change record)
- Why decisions were made → [docs/adr/](docs/adr/) (numbered, append-only; DESIGN.md §0
  holds the founding decisions — changing one of those requires a new ADR)
- Proposals & in-flight items that aren't a cycle → [work/](work/)
  (`NNNN-slug/`: proposal → spec → plan → notes; archive on completion)
- Template extraction ownership zones → [TEMPLATE_OWNERSHIP.md](TEMPLATE_OWNERSHIP.md)
- Repo procedures → [.claude/skills/](.claude/skills/) (e.g. `/next-cycle`); shared
  procedures (`/myconv:make-plan`, `/myconv:wrap-up`, …) arrive via the myconv plugin,
  never copied into this repo
- Human onboarding → [README.md](README.md)

## Hard rules

- Steps interact with the world ONLY through `RunContext`. No direct file I/O, no
  tracker SDK calls, no model-library imports in step modules outside the backends.
- Polars DataFrames cross step boundaries; Parquet at rest; DuckDB allowed inside a
  step only. Never let pandas or DuckDB types leak across a boundary.
- `src/mlpipe/core/` stays free of Numerai-specific imports. Domain code lives in
  `src/mlpipe/steps/` and `src/mlpipe/backends/`.
- Core line budget: 400 target, 700 ceiling. If a change would exceed it, stop and
  write an ADR proposing the trade instead of committing the change.
- ModelBackend method signatures may only mention our types (polars, numpy, Path,
  FoldPlan). A `torch.Tensor` in a port signature is a bug.
- Config changes go through pydantic schemas. If a new option isn't representable in
  the schema, extend the schema first; never bypass validation.
- Every run goes through the CLI (`mlpipe run ...`), never by importing internals —
  this is what guarantees a manifest exists for every execution.
- GPU budget is 12 GB VRAM. Flag any model config that would exceed it before running.
- No hosted/cloud storage of run data. Local only. Never commit secrets.

## Definition of done for a cycle

- Acceptance tests in the goal file pass (`pytest tests/ -k cycleN`).
- The block runs via CLI with upstream served from cache (verify a cache hit in the
  manifest log).
- `goals/STATUS.md` updated; a short work-log entry appended there (what changed, what
  was measured, what's pinned in baselines.yaml if anything was promoted).

## When comparing experiments

Cite the fold-plan hash for both runs. If plan hashes differ, the comparison is
invalid — say so instead of reporting a winner.

## When blocked

Sandbox blocks (installs, env config, credentials) are human-only boundaries: stop and
prompt, never route around — open a `work/` item recording the exact ask so it survives
the session. New dependencies require a short ADR (status: Proposed).

@AGENTS.local.md
