# ADR-0008: Copier scaffolding template, versioned with the library

- Status: Proposed
- Date: 2026-10-23
- Deciders: nanthony + Claude (work/0006 step 15; blueprint in work/0007 "Template
  Architecture")

## Context

ADR-0005 decision 4 allows a Copier template that carries scaffolding only (agent docs,
ADR and `work/` conventions, a config skeleton, CLI wiring), built after a host repo
has adopted the library. TEMPLATE_OWNERSHIP.md already sorts the files into zones.
AGENTS.md's session ritual (step 4) expects a `.copier-answers.yml` and a
`/template-sync` procedure.

Copier is a new dependency, so it needs this ADR. Checked 2026-10-23 on PyPI:

- `copier`, by copier-org (github.com/copier-org/copier), MIT, "Production/Stable",
  many releases since 2019.
- Newest release: 9.19.1 (2026-10-13). Newest inside this repo's resolution cutoff
  (`exclude-newer = 2026-09-21`, ADR-0003): **9.18.2, released 2026-09-07**.

numerai is adopting the library now (work/0006 step 14). What that adoption actually
needs (entry points, config layout, a smoke test, agent docs) is the best evidence for
what the skeleton should contain.

## Decision

1. **Pin `copier==9.18.2`**, used in two places:
   - **As a tool, host-side**, to create and update host repos:
     `uv tool install copier==9.18.2`. It is not a runtime dependency of mlpipe or of
     any host.
   - **As an optional extra in this repo**, `template = ["copier==9.18.2"]`, so a test
     can render the template and run the result's smoke test. The test uses
     `pytest.importorskip("copier")`, so the suite still passes without the extra.
2. **The template lives in this repo**: `copier.yml` at the root with
   `_subdirectory: template`. A template version is an mlpipe git tag, and the rendered
   `pyproject.toml` pins mlpipe to that same tag. One tag gives a matching library and
   scaffolding.
3. **Rendering never runs code.** No `_tasks`, no `_migrations`, no
   `_jinja_extensions`, so `copier copy` and `copier update` never need `--trust`. A
   test enforces this.
4. **Contents follow TEMPLATE_OWNERSHIP.md:**
   - *template-owned:* the managed section of `AGENTS.md`, the `CLAUDE.md` stub, and
     `.claude/skills/template-sync/SKILL.md` (runs `copier update --vcs-ref <tag>`,
     then the tests);
   - *skeleton:* `pyproject.toml` (the mlpipe git pin plus a `mlpipe.pipelines` entry
     point), `src/<package>/pipeline.py` with one sample step and a `StrictConfig`,
     `tests/test_smoke.py` (runs through the CLI and checks for a cache hit),
     `goals/STATUS.md`, `DESIGN.md` project sections, `docs/adr/0000-template.md`,
     `work/README.md`, and a `.gitignore` for `store/`, `manifests/`, `mlruns/`;
   - *repo-owned:* nothing is rendered (no `baselines.yaml`, no data).
5. **Sequencing.** The template is written after numerai's first wrapped stage proves a
   cache hit (step 14's done-when). Its skeleton copies what that adoption needed.
   Installing Copier can happen now.

## Consequences

- `uv.lock` gains Copier and its dependencies (jinja2, plumbum, questionary, dunamai
  and a few others; the exact set is the lock diff to review). Only the `template`
  extra pulls them in.
- Template changes ship as mlpipe tags. A host takes them with `/template-sync`
  (`copier update`), which follows the zone rules: template-owned files take the
  template's side, skeleton files keep local edits.
- A host that existed before the template (numerai) can adopt it with `copier copy`
  into its existing tree and resolve conflicts once; nothing forces it.
- The template's AGENTS.md and the CLI rules in this repo's AGENTS.md must stay in step.
  The render test catches drift in the CLI wiring, not in prose.

## Alternatives considered

- **Cookiecutter.** Rejected: it has no `update`, so hosts could never take template
  fixes.
- **A separate template repo.** Rejected: two version streams for one contract, and
  ADR-0005 chose "no new repo".
- **No template; document the setup in README.** Viable for one host. It stops scaling
  at the second, and AGENTS.md's session ritual already assumes the template exists.
