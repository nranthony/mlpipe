---
name: template-sync
description: Take a new mlpipe tag into this repo — copier update of the template-owned files plus the mlpipe pin, then lock, test and log it. Use when AGENTS.md's session ritual finds .copier-answers.yml behind mlpipe's latest tag, or when asked to upgrade mlpipe.
---

# Template sync

This repo was rendered from mlpipe's Copier template (mlpipe ADR-0008). One mlpipe tag
is both the library version and the template version: `_commit` in
`.copier-answers.yml` names it, and `pyproject.toml` pins mlpipe to it.

Copier and the lock are **host-side** steps (copier is a host tool, and an agent sandbox
usually can't reach GitHub or PyPI). Prepare the exact commands, have the human run
them, then review the result.

## 1. What changes

- The current tag: `_commit` in `.copier-answers.yml`.
- The target tag: mlpipe's latest release tag (ask the human, or read
  github.com/nranthony/mlpipe/tags).
- Read mlpipe's `goals/STATUS.md` log and any new `docs/adr/` between the two tags for
  changes that touch this repo: port changes, renamed steps, signature-version bumps
  (every cache goes cold once), new optional hooks worth adopting. Summarise them for
  the human before updating.

## 2. Update (human, host-side, clean working tree)

```
copier update --vcs-ref <tag> --defaults
uv lock --upgrade-package mlpipe
uv sync
uv run pytest
```

`--defaults` keeps the recorded answers; drop it to change one.

## 3. Resolve by zone

Copier merges three ways and leaves `*.rej` files or conflict markers where both sides
changed a line.

- **Template-owned** (take the template's side unless tests break): the managed section
  of `AGENTS.md`, `CLAUDE.md`, this skill.
- **Skeleton** (keep local edits; port a template change by hand only if its ADR says
  it matters): `pyproject.toml` except the mlpipe pin line, `src/`, `tests/`,
  `DESIGN.md`, `goals/`, `work/README.md`, `docs/adr/0000-template.md`, `.gitignore`.
- **Repo-owned** (never touched): run data, `baselines.yaml`, your ADRs and work items.

The mlpipe pin line in `pyproject.toml` must name the new tag, and `uv.lock`'s mlpipe
`rev` must change in the same diff. Any other package moving in the lock is a finding:
report it before committing.

## 4. Record

- `uv run pytest` green, including the smoke test's cache hit.
- One commit: template files, `pyproject.toml`, `uv.lock`, `.copier-answers.yml`,
  message `Sync mlpipe template to <tag>`.
- A `goals/STATUS.md` log line: old → new tag, what changed that matters here, and
  whether signatures moved (a re-run is expected then, not a bug).
