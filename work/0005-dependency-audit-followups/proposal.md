# Proposal: finish the dependency audit (CVE scan, depaudit re-run, open choices)

- Status: Draft, refreshed 2026-10-27 — task 1 **done** (`dd1e540` is on `main`; the
  branch is gone). Q4 (empty `.venv-sandbox`) resolved operationally: repopulated
  host-side 2026-10-20, cause still unknown. **Still open:** task 2, the CVE scan
  (never run; `uv.lock` has since moved: `51d0b94` v0.2.0, `f5f0ee0` copier extra,
  `a8ec809` numerai extra); task 3, the `depaudit posture` re-run (P04/P05 unconfirmed);
  task 4, `exclude-newer` is still the fixed `2026-09-21` pin (`pyproject.toml:60`);
  task 5, `track` still pulls full `mlflow>=3` (`pyproject.toml:19`). 2-3 are host
  steps; 4-5 need nanthony.
- Author: Claude (agent, windows-ai-sandbox)
- Date: 2026-10-06
- Related: [ADR-0003](../../docs/adr/0003-dependency-resolution-hardening.md) (the
  hardening this follows), [work/0004](../archive/0004-consistency-review-and-orchestrator-question/proposal.md)
  (its "ADR-0004" hygiene item is resolved by ADR-0003)

## Summary

ADR-0003 (commit `dd1e540`, branch `deps-hardening`) pinned the build chain, removed the
`no-build` opt-out, and added a resolution quarantine and an explicit index. It dealt
with *how* packages resolve. Nothing has yet checked *what* is in the lock: the 147
packages in `uv.lock` have never been scanned for known vulnerabilities. This item
holds that scan and the smaller loose ends from the same review.

## Motivation

- `depaudit posture` checks configuration only; it does not look up advisories.
- `uv.lock` package versions date from 2026-08-20 and did not move in the 2026-10-06
  re-lock, so any advisory published since then is unexamined.
- The sandbox has no scanner (`osv-scanner`, `pip-audit` absent), cannot install one,
  and cannot reach `api.osv.dev`. Every scan route is a host step.

## Proposal

### 1. Land the branch (host / human)

`git checkout main && git merge --ff-only deps-hardening`, then push. Deleting the
branch afterwards is a confirmation-gated step in the sandbox.

### 2. CVE scan of `uv.lock` (host / human runs, agent triages)

Run one of these on the host from the mlpipe checkout and paste the output into a
session:

- **No install:** a stdlib-only script that reads `uv.lock` with `tomllib` and POSTs
  name+version pairs to `https://api.osv.dev/v1/querybatch` (ecosystem `PyPI`),
  printing advisory IDs. Listed under "Appendix" below; untested, since the sandbox
  cannot reach the API. Returns IDs only, no severity.
- **`pip-audit`:** `uv export --frozen --all-extras --no-emit-project --format
  requirements-txt -o /tmp/mlpipe-req.txt`, then `uvx pip-audit -r
  /tmp/mlpipe-req.txt --require-hashes --disable-pip`. Gives severity and fixed-in
  versions. `uvx` fetches and runs the tool from PyPI — a host-side trust call. Flags
  written from memory; check `--help` if one errors.
- **`osv-scanner`:** `osv-scanner scan source --lockfile uv.lock`. Needs the Go binary
  installed from `github.com/google/osv-scanner` releases.

Triage rule: a hit means the version is affected, not that mlpipe is exploitable. Rank
by whether the affected code path is used. Expect most hits under the `track` extra
(mlflow's server stack: flask, gunicorn, uvicorn, fastapi, starlette, docker,
databricks-sdk), which matters less while the tracker is local sqlite only. Any upgrade
that results is a re-lock through `with-egress` and a reviewable `uv.lock` diff.

### 3. Re-run `depaudit posture` (host / human)

Expected after ADR-0003: **P01** clears (opt-out removed). Unconfirmed: **P04**
(`exclude-newer` is now set) and **P05** (the check is worded around pip config; the fix
was a `[[tool.uv.index]]` entry). **X07** was a false positive on `a21ff4a`/`a8c2424`;
`dd1e540` changes manifest and lock together. **X01** (CODEOWNERS) stays a WARN by
choice. Record the new tally in `goals/STATUS.md`.

### 4. Decide the `exclude-newer` form

Today: a fixed `2026-09-21T00:00:00Z` in `pyproject.toml`, which the egress wrapper
overrides with its rolling 7-day window and flags as "STRICTER than this window" on
every run. `job_search_agent` uses `exclude-newer = "7 days"` instead, which needs no
hand-maintained date. Known cost: `uv lock --check` fails inside an egress window
because the wrapper exports an absolute timestamp (their ADR-0007 says the fix belongs
in the wrapper). Switching changes what `uv.lock` records, so it needs a re-lock.

### 5. Decide whether `track` needs full `mlflow`

`mlflow` accounts for most of the lock's web-server and cloud-SDK surface, and the repo
uses only a local sqlite tracker and the model registry. `mlflow-skinny` is already in
the lock as a dependency of `mlflow`. Whether it alone covers the sqlite store and
registry calls in `core/mlflow_tracker.py` and `cli.py` is **not verified**. A swap is a
dependency change: new ADR (Proposed) first, per AGENTS.md. Best decided after task 2
shows how much of the advisory load sits in that subtree.

## Open questions

1. Task 4: keep the fixed pin, or switch to `"7 days"` and accept the
   `uv lock --check` behaviour until the wrapper is fixed?
2. Task 5: worth a spike, or leave `mlflow` whole?
3. `numerai/pyproject.toml` and `shrec/uv.toml` still opt out of wheels-only (for
   `antlr4-python3-runtime` and `python-louvain`). Out of scope here — noted because
   the wrapper warns about them on every mlpipe egress window.
4. Why was `.venv-sandbox` an empty Python 3.13 shell dated 2026-09-15? Cause not
   established; repopulated 2026-10-06. Worth knowing if it recurs.
   **It recurred on 2026-10-19:** `uv run --no-sync pytest` printed "Creating virtual
   environment at: .venv-sandbox" (Python 3.13.16) and then could not find `pytest`,
   so the populated venv from 10-06 was gone. The in-sandbox
   `uv sync --frozen --all-extras --offline` was denied. Ask recorded in
   work/0006 plan.md "Blockers".

## Alternatives

- **Add `pip-audit` as a `dev` dependency** so the scan runs from the repo: rejected —
  it still needs registry and advisory egress the sandbox does not have, and it is a
  new dependency for something a host one-liner does.
- **Commit the OSV script into the repo:** not proposed yet; reasonable if the scan
  becomes routine, since it adds no dependency. Decide after its first real run.

## Appendix: the no-install OSV check

Save as `osv_check.py` in the checkout on the host and run `python3 osv_check.py`
(needs Python 3.11+ for `tomllib`). Untested.

```python
import json, tomllib, urllib.request

pkgs = [p for p in tomllib.load(open("uv.lock", "rb"))["package"]
        if "registry" in p.get("source", {})]
body = {"queries": [{"package": {"name": p["name"], "ecosystem": "PyPI"},
                     "version": p["version"]} for p in pkgs]}
req = urllib.request.Request("https://api.osv.dev/v1/querybatch",
                             json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
results = json.load(urllib.request.urlopen(req))["results"]
hits = 0
for p, r in zip(pkgs, results):
    for v in r.get("vulns", []):
        hits += 1
        print(f'{p["name"]}=={p["version"]}  {v["id"]}  https://osv.dev/vulnerability/{v["id"]}')
print(f"{len(pkgs)} packages checked, {hits} advisories")
```
