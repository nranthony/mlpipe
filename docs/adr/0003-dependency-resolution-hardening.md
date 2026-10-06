# ADR-0003: Harden dependency resolution (build-chain pins, quarantine, explicit index)

- Status: Accepted
- Date: 2026-10-06
- Deciders: nanthony + Claude (depaudit posture review, 2026-10-05)

## Context

A `depaudit posture` run on 2026-10-05 reported 0 FAIL / 5 WARN. Three of the
warnings describe real gaps in how this repo resolves packages:

- **P01 — `no-build = false`.** The project opted out of the sandbox's
  wheels-only default (`no-build = true` in `/etc/uv/uv.toml`) on the belief
  that mlpipe's own editable install, a hatchling source build, would be
  blocked by it. That belief was wrong: uv exempts the project's editable build
  from the gate, and every locked dependency has a wheel, so the opt-out bought
  nothing while switching the gate off for the whole tree. Separately,
  `[build-system] requires = ["hatchling"]` was unpinned and build dependencies
  are not recorded in `uv.lock`, so hatchling and its requirements resolved
  fresh, at whatever was newest, on every sync. That is the one path where
  unreviewed code runs at install time.
- **P04 — no resolution quarantine.** `uv.lock` was cut on 2026-08-20 and took
  files uploaded one or two days earlier (`uvicorn`, `databricks-sdk` on 08-19;
  `gunicorn`, `idna`, `cuda-pathfinder` on 08-18). Nothing went wrong, but a
  freshly published malicious release would have been accepted the same way.
- **P05 — index not pinned in-repo.** All 146 third-party lock entries come from
  `pypi.org/simple`, but only because ambient config happened to say so.

The same comment block also cited "ADR-0004", which does not exist in this repo
(flagged in `work/0004`). It refers to the sandbox project's wheels-only policy.

Not acted on: **X07** is a false positive (`a21ff4a` predates `uv.lock`;
`a8c2424` only added `[tool.ruff.lint]`), and **X01** (CODEOWNERS) adds nothing
while the repo has a single maintainer.

## Decision

All in `pyproject.toml`:

1. Pin `hatchling==1.32.0` in `[build-system].requires`, and pin its full
   requirement set (`packaging`, `pathspec`, `pluggy`, `tomlkit`,
   `trove-classifiers`) through `[tool.uv] build-constraint-dependencies`.
   Versions are the ones already in the sandbox's uv cache from the 2026-08-20
   provisioning; `packaging` and `pluggy` match `uv.lock`.
2. Remove `no-build = false`. The project runs under the sandbox's wheels-only
   default like any other; a comment in its place says why the key is absent.
   Verified 2026-10-06: `uv sync --frozen --all-extras` into an empty
   `.venv-sandbox` with the line removed built mlpipe, installed 142 packages
   from wheels, and the suite passed (63 tests).
3. Set `[tool.uv] exclude-newer` to a fixed timestamp, as the fallback for any
   resolution run outside the sandbox's `with-egress` wrapper. Inside the
   wrapper, `UV_EXCLUDE_NEWER` (a rolling 7-day window) overrides it and is the
   value recorded in `uv.lock`. Procedure: move the pin forward only when it
   blocks a wanted upgrade, and never to less than 7 days before that day.
4. Declare PyPI as the explicit default index via `[[tool.uv.index]]`.

No dependency is added, removed or upgraded by this ADR.

## Consequences

- Re-locked on 2026-10-06 through `with-egress`: 147 packages, no version
  changes; the lock gained only `[options] exclude-newer` and
  `[manifest] build-constraints`.
- The lock's `exclude-newer` (the wrapper's timestamp) differs from the pin in
  `pyproject.toml`, and uv re-resolves whenever the effective value differs from
  the recorded one. Any uv command run without the wrapper's env var therefore
  wants to re-lock: in the sandbox that fails at the closed registry, on the
  host it rewrites the lock's timestamp. Use `uv run --no-sync` /
  `uv sync --frozen` everywhere except a deliberate re-lock.
- Each wrapper re-lock stamps a new timestamp into `uv.lock`, so expect a
  one-line lock diff even when no package moves.
- A new dependency that ships only an sdist now fails to install in the
  sandbox. That is the gate working; the answer is a different package or a new
  ADR arguing for the opt-out, not quietly restoring `no-build = false`.
- Bumping hatchling is now a deliberate six-line edit instead of something that
  happens silently.
- Whether the explicit index entry clears depaudit's P05 is unverified; the
  check is worded around pip config.
- Follow-ups outside this ADR: run a CVE scan over `uv.lock` on the host (none
  has been done); consider whether the `track` extra needs full `mlflow` given
  the tracker is local sqlite only (a dependency change, so its own ADR).

## Alternatives considered

- **`[tool.uv] package = false`** — removes the remaining source build
  entirely. Rejected: it also drops the `mlpipe` console script, and AGENTS.md
  requires every run to go through that CLI.
- **Relative `exclude-newer = "7 days"`** (as `job_search_agent` uses) — removes
  the hand-maintained date. Open, not rejected: it makes `uv lock --check` fail
  inside an egress window, because the wrapper exports an absolute timestamp.
- **Drop the `exclude-newer` pin and rely on the wrapper alone** — the wrapper
  already gives a rolling window. Rejected for now: a `uv lock` run outside it
  would then have no quarantine at all, and the lock/manifest mismatch exists
  either way.
- **Pin only hatchling, not its requirements** — rejected: the transitive
  packages run in the same build process and would still float.
- **Commit a `pip.conf` for P05** — rejected: nothing in this repo uses pip.
