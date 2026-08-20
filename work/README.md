# work/ — proposals and in-flight items

One numbered folder per unit of work that is **not a build cycle** — cycles live in
[goals/](../goals/) and log to `goals/STATUS.md`; this directory holds everything
around them: proposals, environment/dependency asks, plans that must outlive a
session, spikes. `NNNN` is the next free number across active **and** archived
items; numbers are never reused.

## Files inside an item

Each is optional except whichever one starts the item:

- `proposal.md` — "should we do this?" (status-tracked; template below)
- `spec.md` — the pinned what/why, when it needs pinning
- `plan.md` — the implementation plan (typically via `/myconv:make-plan`)
- `notes.md` — running notes while executing

## Lifecycle and exit rule

1. An item opens as a `proposal.md` (Draft) or, for pre-decided work, straight as a
   `spec.md`/`plan.md`.
2. An accepted proposal's durable rationale is **distilled into an ADR** in
   [docs/adr/](../docs/adr/); the proposal's status line links it
   (`Accepted → ADR-NNNN`).
3. When the work merges or the question resolves, the folder moves to
   `work/archive/`. **Items are archived, never deleted.** Nothing durable may live
   only in `work/`.

Archived items are historical records, never current intent — the distilled ADR is
canonical. `work/plans/` is gitignored scratch space for native plan-mode drafts;
a draft becomes durable by promotion into an item.

## Proposal template

```markdown
# Proposal: <title>

- Status: Draft | In review | Accepted → ADR-NNNN | Rejected
- Author: <name / agent>

## Summary
## Motivation
## Proposal
## Open questions
## Alternatives
```
