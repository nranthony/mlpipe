# ADR-0001: Adopt the agent-native repo conventions

- Status: Accepted
- Date: 2026-08-20
- Deciders: nanthony + Claude (myconv:apply-conventions)

## Context

The bootstrap package shipped a substantive `CLAUDE.md` as the agent entry point. That
file is Claude-specific, and the repo's own rules already demand decision records
("write an ADR proposing the trade", "new dependencies require a short ADR") with no
home for them. The repo is edited by agents inside a restricted sandbox
(`windows-ai-sandbox`), where blocked actions must become recorded human steps rather
than lost session context.

## Decision

Adopt the agent-native conventions blueprint, adapted to this repo's existing shape:

- `AGENTS.md` is the single entry point (rules + index + sandbox notice);
  `CLAUDE.md` becomes the two-line `@AGENTS.md` stub.
- `DESIGN.md` keeps the architecture-doc role — no separate `ARCHITECTURE.md`.
- `goals/STATUS.md` keeps the change-record role — no separate `CHANGELOG.md`.
- `docs/adr/` holds decision records from now on; DESIGN.md §0 is grandfathered as
  the founding decision set, and changing any §0 decision requires a new ADR.
- `work/` holds proposals and in-flight items that are not a cycle (cycles stay in
  `goals/`); items archive to `work/archive/` on completion, never deleted.
  Plan-mode drafts land in gitignored `work/plans/` via `plansDirectory`.
- `.claude/commands/next-cycle.md` migrates to `.claude/skills/next-cycle/SKILL.md`
  (commands merged into skills upstream).
- Shared procedures (`/myconv:make-plan`, `/myconv:wrap-up`, ClickUp skills) come
  from the myconv plugin and are never copied into this repo.

Skipped deliberately: `CODEOWNERS`, `CONTRIBUTING.md`, PR template, CI (solo repo,
no remote); `validation/` (nothing measured yet — revisit when cycle 7 produces
`baselines.yaml`); `.myclickup.toml` (no tracker link declared; opt-in later).

## Consequences

- Agents landing cold read one file and find everything by reference; per-tool
  entry points stay two lines.
- Sandbox denials have a durable escalation path: a `work/` item recording the ask.
- `TEMPLATE_OWNERSHIP.md` zones updated (`.claude/skills/**` replaces
  `.claude/commands/**`; `AGENTS.md` joins the template-owned docs) so the eventual
  Copier extraction carries the conventions with it.

## Alternatives considered

- Keep the substantive `CLAUDE.md`: Claude-only; other runtimes (Cursor,
  Antigravity) read `AGENTS.md` natively.
- Add the full ceremony tier (CODEOWNERS/CI/PR template): overhead with no
  reviewers and no remote; revisit if the repo gains either.
