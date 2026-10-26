# ADR-0010: Viewing surfaces: training curves, charts in the tracker, agent reports

- Status: Accepted 2026-10-26 (nanthony: "both — go ahead" on work/0013)
- Date: 2026-10-26
- Deciders: nanthony + Claude (work/0013)

## Context

A data scientist using mlpipe could see scalar metrics in MLflow and nothing else.
There were no training curves (backends fit with no eval set and kept no history), no
feature views, no tabular evaluation charts, and MLflow received no artifacts and no
step config. The user's rule for the fix: what fits MLflow cleanly goes there. Other
visuals and reports are generated from the data, metadata and results "in a more
agentic fashion".

Two constraints frame it:

- DESIGN §0: the tracker is a view, and losing `mlruns/` must lose no provenance.
- DESIGN §3: ports mention only our types.

## Decision

1. **Artifacts first, tracker second.** Every curve, chart and report a step makes
   is a store artifact. The tracker gets a copy to display.
2. **An optional `CurveBackend` protocol** beside `ModelBackend`:
   - `fit(X, y, params, eval_set=None)`;
   - `history() -> {"<split>_<metric>": [per iteration]}`;
   - `feature_importance() -> {feature: gain} | None`.

   The train step detects it by `history`, passes each fold's validation rows, and
   writes `train_history` and `feature_importance`. A backend without the members
   still works, and its frames are empty. LightGBM records its eval metrics per
   round, and torch records MSE per epoch.
3. **The step-facing surface gains `RunContext.log_curve(name, values)`.** It goes to
   the tracker only, step-indexed, because the artifact already holds the series.
4. **The Tracker port gains `log_curve` and `log_artifact`.** After a step succeeds,
   the pipeline hands every `.html`/`.json` output to `log_artifact`. Trackers without
   these still work: curves fall back to `log_metrics(step=i)`, and artifacts are
   skipped. `MlflowTracker` also logs the step's config subtree as `cfg.*` params.
5. **Generic charts:**
   - `ProfileStep` (`data_profile`, `profile_chart`). Nothing consumes it.
   - `TabularEvalStep` gains `eval_chart` and a `curves` summary in its report
     (best vs final iteration per metric).
   - Chart pages are byte-stable: plotly gets fixed div ids.
6. **Agent reports are read-side.** `mlpipe show [--json]` maps each step's latest
   completed manifest to its outputs' store files. A report is a script in gitignored
   `reports/<pipeline>/` that loads those files and calls `mlpipe.report.write_report`.
   - The page lists every hash it read, and is a valid claude.ai Artifact page.
   - It never writes to `store/` or `manifests/`, so it is not a run and does not go
     through `mlpipe run`.
   - The procedure is the `report` skill.

## Consequences

- **One re-run after the upgrade.** The train, evaluate and register code changed, so
  each pipeline re-runs once. For numerai-example that is a ~30 min train. Model bytes
  change with the backend code hash, but predictions do not: there is no early
  stopping.
- **Memory.** Each fold also builds a LightGBM dataset for its validation rows. That
  is roughly 1/k more memory per fold fit on Numerai-sized data; watch the first
  numerai-example run.
- **Core size.** Core grows by 18 lines (orchestration 707 → 725).
- **MLflow artifact paths.** MLflow stores an experiment's artifact location as an
  absolute path. An experiment created in the sandbox points at
  `/workspace/mlpipe/mlruns/<id>`, so a host UI whose checkout lives elsewhere sees
  metrics and params but not those artifacts. The store copy is unaffected.
- **Report scripts are not versioned** (`reports/` is gitignored, like `mlruns/`). A
  report worth keeping gets moved somewhere tracked by hand.

## Addendum 2026-10-26: grouping runs (nanthony: "do both")

Viewing the first real run showed a gap: a cached step started no tracker run, so the
only run holding the training curves aged out of view while later runs showed
evaluate and register alone. Three more optional Tracker members, same fallback rule
as item 4 (a tracker without them still works, ungrouped):

- `start_pipeline(run_id, tag)` / `end_pipeline(status)`: the pipeline brackets each
  run, so `MlflowTracker` makes one parent run per pipeline run with the steps nested
  beneath it.
- `log_cached(manifest)`: a cache hit gets a small child run (params, the manifest's
  scalar metrics, `mlpipe.source_run_id` and a note linking the run that executed the
  signature, where its curves and charts are). Curves and artifacts are linked, not
  copied again.
- Every run is tagged `mlpipe.run_id` (the manifests' `run_id`), `mlpipe.pipeline`,
  `mlpipe.kind`, and on steps `mlpipe.step`, `mlpipe.signature`, `mlpipe.cached`.

None of this enters a signature, so no pipeline re-runs. Core grows by 21 lines
(725 → 746).

## Alternatives considered

- **Curves only in MLflow.** That breaks "losing `mlruns/` loses zero provenance".
- **Changing `ModelBackend.fit` itself.** That breaks every host backend. The
  optional protocol does not.
- **A dashboard app (Streamlit or similar).** It adds a dependency and a second UI
  next to MLflow; agent-written HTML covers the ad-hoc need.
- **Reports through `mlpipe run` as steps.** Ad-hoc questions don't belong in a
  cached, signature-keyed pipeline. They read finished artifacts and cite their hashes
  instead.
