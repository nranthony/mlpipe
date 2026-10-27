# Adopting mlpipe in an existing repo

For the agent in a host repo that already has working code. The Copier template
(ADR-0008) gives the repo its rules and a sample step; this page covers what the
template can't: wrapping code that exists. The worked precedent is the numerai repo
(its ADR-0008, "adopt mlpipe's spine"), which adopted one stage at a time from
`v0.2.0`.

## 1. Apply the template

On a branch with a clean tree: `copier copy --vcs-ref <tag> gh:nranthony/mlpipe .`
Let copier overwrite, then restore by hand from `git diff`:

- `pyproject.toml`: keep yours and add three pieces from the rendered one: `mlpipe` in
  `dependencies`, its `[tool.uv.sources]` line (the pin), and the
  `[project.entry-points."mlpipe.pipelines"]` table.
- `AGENTS.md`: keep the managed section; move your existing rules under `## Project`.
- Keep `.copier-answers.yml`; `/template-sync` needs it for every later tag.

Then `uv lock && uv sync && uv run pytest`. The sample step can stay until your first
real step replaces it.

## 2. Pick the first stage

Start at the upstream end with one stage that is deterministic and either costly or
provenance-critical: raw files → a prepared table is the usual first one. Don't start
with training loops that retrain by design (a search's trials have nothing to reuse);
their reusable by-products, such as out-of-fold predictions, come later. Prove one
stage (§5) before adding the next, and give each later stage its own `work/` item.

## 3. Write the steps

A step is a class with `name`, `inputs`, `outputs`, a `config_model` (`StrictConfig`)
and `run(ctx)`; see the rendered `pipeline.py`. Its cache key is the code closure plus
its config subtree plus its input hashes, so **everything that changes the output must
reach one of those three**, and nothing else should:

- **Raw files outside the store**: put each file's sha256 in the config, with a path
  relative to the top-level `download_dir` (outside every signature). The step
  `ctx.put`s the file and fails if the stored hash differs from the config's. Never key
  on size or mtime: a copied or touched file would recompute although its bytes are
  the same.
- **Data that is neither code nor config** (an era list, a holdout set): save it to the
  store before the run and pass it as a seed (`Pipeline.run(seeds=[artifact])`); the
  step declares it in `inputs`. A change then busts the cache.
- **Libraries whose versions decide the numbers** but whose source is outside the
  closure: put their versions in the config subtree.
- **Sibling packages the step calls**: list their prefixes in `Step.code_roots`.
  **Modules chosen at runtime** (a backend named in config): return them from
  `code_deps`.
- **Imports**: step modules import from submodules, never the package root, or every
  edit anywhere in the package invalidates the cache. Pin the closure with a test:
  `assert set(code_modules(MyStep())) == {...}` (`mlpipe.core.signature`).
- **Evaluation surfaces** (a frozen holdout, a scoring rule you must not fit to) stay
  outside steps, or enter only as seeds that the step reads for exclusion.

Polars crosses step boundaries; convert to pandas at the edges for callers that need
it.

## 4. Register and run

`build_steps()` and `DEFAULT_CONFIG` in your pipeline module, registered under the
`mlpipe.pipelines` entry point. Run through the CLI:
`uv run mlpipe run --pipeline <name>`.

Run in-process only when callers need the data back in memory and the config is
computed per call (numerai's raw-file hashes). Record that choice in an ADR. Manifests
are written exactly as by the CLI:

```python
store = LocalCasStore(state / "store")
result = Pipeline(build_steps(), store, state / "manifests" / PIPELINE_NAME,
                  config=config, project_root=REPO_ROOT).run(to_step="prep")
table = store.load(result["artifacts"]["prep_table"].content_hash)
```

Keep your own store; never share another repo's. The `gc` reachability check counts
only the manifests it can see. A tracker is optional; the manifests are the record.

## 5. Prove it

1. Run twice. The second run reports every step `cached`, with the same output hashes.
2. `uv run mlpipe lineage <hash> --pipeline <name>` walks the output back to its raw
   inputs. `uv run mlpipe show --pipeline <name>` lists the latest outputs.
3. Change one config value and check that only that step and its downstream steps
   recompute.

## 6. Record it

Write an ADR, "adopt mlpipe", before the first step lands. It records:

- the pinned tag, and the scope (which stages);
- where the store and manifests live;
- the DataFrame engine: polars between steps, plus where pandas stays and why;
- the provenance of record: whether manifests become the record, or existing sidecars
  stay and gain a pointer such as the output hash;
- the tracker: none, mlpipe's MLflow view, or an adapter over an existing tracking
  database;
- the config front end: if Hydra or similar stays, it builds the validated subtrees
  that the steps receive;
- the choices from §3 and §4: raw-file identity, seeds, CLI or in-process.

Not adopting is a legitimate outcome. Then add a STATUS entry citing the output hashes
from §5.

## When mlpipe is wrong

Never patch it in the host. Write the problem as an "mlpipe follow-up" bullet in the
work item where you met it, with a minimal repro. The human carries it to mlpipe. A fix
ships as a tag, and the host takes it with `/template-sync`.
