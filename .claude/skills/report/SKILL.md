---
name: report
description: Build an agent report — an HTML page of charts and tables answering a question about a pipeline's data, model or results — from stored mlpipe artifacts, optionally published as a private claude.ai page. Use when asked to visualize, explore, explain or compare features, predictions, errors, training curves or runs beyond what the MLflow UI shows.
---

# Agent report (ADR-0010)

A report reads finished artifacts and never runs or writes pipeline state. If the
question needs a new computation inside the pipeline (a new metric every run should
have), that is a step change, not a report.

## 1. Find the artifacts

```bash
uv run --no-sync mlpipe show --pipeline <name>            # human view
uv run --no-sync mlpipe show --pipeline <name> --json     # what the script parses
```

Each step's latest completed manifest, with its config, metrics, meta, and each
output's `hash`, store `path` and `format` (parquet / json / joblib / html). If the
artifacts you need are missing or stale, run the pipeline first (`mlpipe run`),
never write to `store/` or `manifests/` yourself.

Useful outputs on tabular pipelines: `clean_table` / `clean_validation` (rows),
`oof_preds` (row-aligned with `clean_table`/`feature_table`; carries the plan's group
column if any), `train_history` (`fold, iteration, split, metric, value`),
`feature_importance` (`model, feature, gain`), `eval_report` (metrics, `plan_hash`,
`curves`), `data_profile`. A `.joblib` loads with joblib; only load ones this repo's
pipelines wrote.

## 2. Write the script

`reports/<pipeline>/<YYYY-MM-DD>-<slug>.py` (gitignored). Pattern:
`reports/titanic-example/2026-10-26-survival-and-errors.py` if present, else:

```python
import json, subprocess, sys
from pathlib import Path
import polars as pl
import plotly.graph_objects as go
from mlpipe.report import table, write_report

runs = {r["step"]: r for r in json.loads(subprocess.run(
    [sys.executable, "-m", "mlpipe.cli", "show", "--pipeline", PIPELINE, "--json"],
    check=True, capture_output=True, text=True).stdout)}
oof = pl.read_parquet(runs["train"]["outputs"]["oof_preds"]["path"])
...
write_report(Path(__file__).with_suffix(".html"), "Specific Report Name",
             [{"heading": ..., "note": ..., "figure": fig, "html": table(rows)}],
             sources={"oof_preds": runs["train"]["outputs"]["oof_preds"]["hash"],
                      "fold plan": report["plan_hash"]})
```

Run it: `uv run --no-sync python reports/<pipeline>/<file>.py`.

Rules:

- `sources` names every artifact read, by full hash, and the fold-plan hash when
  model results appear. Comparing two runs with different plan hashes is invalid:
  say so on the page instead of ranking them (AGENTS.md).
- Notes state what the chart shows. **Check every claim against the numbers after
  rendering** (read the tables back from the HTML or print them) before writing it.
- Chart colours: plotly defaults or mid-tone hex values that read on light and dark;
  `write_report` recolours text and grids from the page theme.
- Don't add `toImage` back; a published page cannot download files.

## 3. Show it

- Local: give the user the absolute path of the `.html`; it opens offline.
- Published (when the user wants it viewable away from the host): the page is
  already a valid Artifact fragment. Publish the `.html` with the Artifact tool
  (load `artifact-design` first if not loaded this session), icon `chart`, and a
  one-sentence description. It is private until the user shares it. Republish the
  same path to update it.
