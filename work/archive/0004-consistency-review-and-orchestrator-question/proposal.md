# Proposal: post-build consistency review, and whether to replace the custom core

- Status: Draft — assessment delivered 2026-08-27; awaiting nanthony's answers to
  "Open questions" before any of the "Next steps" starts
- Author: Claude (agent, windows-ai-sandbox)
- Date: 2026-08-27
- Related: [work/0002](../0002-post-build-followups/proposal.md) (gaps 2, 3 still open;
  direction still unanswered), [work/0003](../0003-subset-smoke-runs/proposal.md)

## Summary

Verified state on 2026-08-27: `pytest tests/ -q` → 63 passed; `ruff check` clean;
working tree has one uncommitted change (`AGENTS.md` managed sandbox block, harmless);
store 37 files / 11 GB; every real-data signature is cold since the gap-1 fix (next real
run recomputes ingest→register, ~1.5–2 h). The build is logically sound where it was
tested; the inconsistencies below are all at the seams the acceptance tests did not
cover — promotion, config validation, `--from`, and what "core budget" counts.

On the "one aspect is custom-made" question: the custom parts are the **orchestrator
+ signature cache + JSON manifests** (~511 lines, `core/` minus ports and adapter) and
the **content-addressed store** (79 lines, instead of DVC). Recommendation: **keep
them** (DESIGN.md §0 stands, no ADR needed) — see "The buy-vs-build question" for why
and for the one place an existing tool *should* replace custom code (promotion →
MLflow aliases, already a dependency).

## Findings — logical consistency (ranked by consequence)

### A. "Promotion" is three half-mechanisms that disagree

- DESIGN.md §1: *promote = tag the run in MLflow and pin its config hash in
  `baselines.yaml`; the next iteration builds on the pinned baseline.*
- Reality: `mlpipe tune` (`cli.py:104-110`) overwrites `baselines.yaml` with the best
  trial's **searched params only** — not `n_estimators=50` that the study actually ran
  with, no config hash, no plan hash, no bundle hash. **Nothing reads `baselines.yaml`**
  (`rg baselines src` → only the writer). No MLflow tag is set. Meanwhile
  `RegisterStep` (`steps/register.py:47`) registers a version on every fresh run
  (work/0002 gap 3).
- Consequence: the "pinned baseline" (trial 2, mean_corr 0.0254, 50 trees) is worse
  than and not comparable to the registered `numerai-lgbm v1` (0.02793, 200 trees), and
  cannot be reproduced from `baselines.yaml` alone. AGENTS.md's "cite the fold-plan hash"
  rule cannot be applied to it because it holds none.

### B. Config validation is partial, and `seed` is recorded but inert

- Only step subtrees with a `config_model` are validated (`pipeline.py:73-76`).
  `tracker`, `tune`, `download_dir`, `seed` are raw dicts: `tracker.kind: mlflw`
  silently yields `NoopTracker` (`cli.py:44`); a typo in `tune.space` dies at runtime.
  DESIGN §5 "config typos must die at pydantic validation" holds for steps only.
- `seed: 42` is written into every manifest's `environment` (`pipeline.py:40`) but no
  step or backend reads it — LightGBM hardcodes `random_state: 42`, torch reads
  `params.seed`. Provenance records a value that changes nothing, and changing it
  changes no signature either (it is not in any step subtree).
- DESIGN §0 says "pydantic models composed from YAML"; the real config is a Python dict
  (`steps/numerai.py`), `--config` replaces it wholesale (no merge; work/0003 notes
  this), and `configs/` (a TEMPLATE_OWNERSHIP zone) does not exist.

### C. `--from X` serves upstream from "latest ever", not from the current config

`pipeline.py:86-92` registers upstream artifacts from `ManifestLog.latest_outputs()` —
the most recent hash per key regardless of config. So
`run --from train --set clean.cleaner.kind=minimal` trains on whatever `clean` last
produced and records a config that did not run. Manifests stay honest (input hashes are
real), but config ≠ execution. **work/0002 gap 2 (tune overrides below `train` are
inert) is the special case of this.** Note that with content-hash caching `--from` is
not needed for *correctness*: a full run with a warm cache is seconds (the 2026-08-20
23:58 run: 8 × cached). The fix is to resolve upstream by walking signatures (cache-hit
or run), which makes `--from` an assertion ("upstream must be cached") rather than a
shortcut with a trapdoor — and closes gap 2 for free.

### D. The core line budget has drifted in what it counts

Cycle 0 reported "400/400 (interfaces.py excluded: 526 total)". Later entries count all
of `core/`: 638, then 697/700 "interfaces+tracker included". Today: 697 with
`interfaces.py` (131, port declarations) and `mlflow_tracker.py` (55, an adapter) — or
**511** without them. "The next core edit breaches the ceiling" (work/0002) is an
accounting artifact until the rule says what counts. Needs one sentence in DESIGN §0.

### E. The port steps are coded against is not the port in `interfaces.py`

`interfaces.RunContext` declares `get / put / log_metric` and raises
`NotImplementedError("cycle 0")`; steps actually use `get_lazy`, `put_alias`,
`log_meta`, `previous`, `register_model`, `input_hashes` from the concrete
`core/context.RunContext`. DESIGN §3's Tracker summary lacks `end_run`;
`ModelBackend.fit(..., folds)` is never used by any backend (dead parameter). This is
exactly what the skipped ZenML spike would have surfaced: the "cheap shim" surface is
nine methods on the concrete class, not three on the port. Checkable without ZenML.

### F. The validation holdout is produced but never scored

`clean_validation` → `feature_validation` exist only for the cycle-4 skew guard.
`EvaluateStep` scores CV-OOF on train eras; `predict` uses `raw_live`. Numerai's real
holdout (659 eras, 0575–1233) is never evaluated, so every headline metric (including
the registered model's) is in-sample-CV only. DESIGN §2 block 7 does not say which
partition it scores. This matters before any modelling iteration is meaningful.

### G. Hygiene

- `mlpipe-claude-init-package/` — 25 tracked files: an older `DESIGN.md` (differs from
  the live one), old goals, the retired `next-cycle` command. A stale twin of the design
  doc, not in AGENTS.md's map. **Deletion/archival is a human step.**
- `pyproject.toml:39` cites "ADR-0004"; this repo has ADR-0001/0002 only. Probably the
  sandbox project's ADR — ambiguous as written.
- Demo-pipeline runs (`make_data`/`summarize`, 2026-08-24) landed in the real
  `manifests/index.jsonl` and `store/` because both pipelines share the default dirs.
  Harmless today; pollutes `latest_outputs()`/lineage history.
- `goals/STATUS.md` header says "ALL CYCLES 0-10 COMPLETE" while the 08-24 entry says
  the real-data cache is fully cold and not re-run. The header should carry the state.
- Manifests record `git_commit` but not dirty-tree state; combined with core's
  exclusion from signatures, a core edit in a dirty tree is attributed to the wrong
  commit. One line — blocked on D.

## The buy-vs-build question (ZenML / Kedro / others)

What is custom: a Merkle-DAG-cached step runner (signature = name + source-closure
hash + config subtree + input hashes), append-only JSON manifests with a lineage walk,
and a sha256 CAS. Everything else is bought: MLflow (tracker/registry), Optuna,
pandera, pydantic, polars, sklearn protocol, LightGBM/torch.

Landscape, from my knowledge (cutoff Jan 2026 — **not re-verified against Aug 2026**;
can be checked with `webfetch` on request):

| Tool | What it would replace | What it costs here | Cache semantics vs. ours |
|---|---|---|---|
| **ZenML** | Pipeline/Step, caching, artifact store, lineage UI, registry glue | Heavy dep tree + a metadata store (SQLite "zen store") and a dashboard server; read path becomes SQL/REST, not grep-able JSON; install is a sandbox human step (registries closed) | Cache key is the step *function's* source + params + input artifact IDs — helper/backend module edits are **not** detected (the exact gap 1 fixed on 08-24). Ours is stricter. |
| **Kedro** | DAG runner, DataCatalog, config loading, project layout | Framework adoption for every host repo; no step caching at all (dataset versioning is timestamped, not content-hashed) — the custom part would still have to be written | n/a — Kedro has no signature cache |
| **Hamilton** | DAG + per-node caching, pure library, no server | Closest "buy" match for the custom part; function-source hashing | Same closure limitation as ZenML, as far as I know |
| **DVC pipelines** | Stage caching by content, CAS, `dvc dag` | Git-coupled, file-level, md5; §8 rejected DVC on maintenance-risk grounds (for data versioning — the stage cache is the same tool) | Deps are whatever files you list — closure is manual |

Assessment:

1. Nothing in the set gives *better* cache semantics than the 511 lines now in core,
   and the closure-hash fix just made ours stricter than the common tools.
2. Each candidate trades away the two properties the project exists for: **legibility
   to an agent** (JSON manifests, no server, one context window of code) and
   **portability** — the README's goal is a pipeline "agentically delivered to blank or
   existing repos"; a 500-line drop-in is a product, "adopt ZenML/Kedro first" is not.
3. The one honest argument *for* a framework — humans and LLM agents already know the
   Kedro/ZenML layout — is answered here by DESIGN.md + AGENTS.md fully specifying the
   custom thing.
4. Where an existing tool **should** replace custom/half-built code: **promotion** →
   MLflow model-version aliases (`champion`) plus a `baselines.yaml` that pins the full
   resolved config + plan hash + bundle hash; and **top-level config** → a
   `PipelineConfig` pydantic model (no new dependency).
5. The "orchestrator exit door" can be proven without installing anything: enumerate
   the concrete `RunContext` surface (finding E), make `interfaces.RunContext` declare
   it, and add a test that steps use nothing else.

So: §0's orchestrator decision stands; no ADR needed to keep it. An ADR *is* needed for
promotion (change of direction) and for the budget-accounting sentence.

## Next steps (proposed order; gated on the answers below)

1. Commit the pending `AGENTS.md` change. Decide D (what the budget counts) — one-line
   ADR — so core edits are unblocked.
2. Before the ~2 h recompute: land `era_slice` (work/0003) and the `--from`-by-signature
   fix (C, closes gap 2). Then kick the full real-data run in a background shell.
3. Promotion ADR (A + gap 3): `mlpipe promote <bundle-hash>` sets the MLflow alias and
   writes a complete `baselines.yaml`; `register` stops auto-registering.
4. `PipelineConfig` top-level schema; plumb `seed` to the backends or drop it (B).
5. Score the validation holdout in `evaluate` and put both numbers in the report (F).
6. Hygiene (G): archive/delete the bootstrap twin (human step), fix the ADR-0004
   reference, separate demo from real manifests, update the STATUS header.
7. Then modelling or template extraction, per Q1.

## Open questions

1. **Direction.** Numerai score (modelling) or the deliverable (template extraction)?
   Asked in work/0002 on 08-22, unanswered. It also decides how much the
   orchestrator question matters: for a template landing in existing repos, the small
   custom core *is* the product.
2. **Constraints.** Is "no server, files are the source of truth" still hard? That is
   the constraint that rules ZenML out; if a local metadata server is acceptable, the
   comparison should be redone with UI/lineage viewing weighted in.
3. **Budget accounting.** Count orchestration only (511 today, recommended) or all of
   `core/` (697)?
4. **Promotion semantics.** Explicit `mlpipe promote` verb that does both (MLflow alias
   + full-config `baselines.yaml`), with `register` gated behind it? Is
   `baselines.yaml` meant as "the config the next iteration starts from"?
5. **Evaluation.** Should the Numerai validation holdout be scored (and cited) alongside
   CV-OOF?
6. **Human steps.** Delete or archive `mlpipe-claude-init-package/`? What does
   "ADR-0004" in `pyproject.toml` refer to?

## Alternatives

- **Adopt ZenML now** (spike on the host, since the install is a sandbox boundary):
  rejected for the reasons above; revisit only if Q2's answer changes.
- **Adopt Kedro for structure only, keep the cache**: rejected — two orchestration
  vocabularies in one 3k-line repo, and it does not cover the custom part.
- **Do nothing about A–C and proceed to modelling**: viable short-term, but every
  experiment run before A/C/F land inherits a promotion record that cannot be trusted
  and metrics that never touch the holdout.
