# 0008 spec — Cache and provenance correctness before a host adopts the core

- Status: **Draft — awaiting sign-off.** Ships as **v0.2.0**, which numerai pins instead
  of v0.1.0. (A host must not adopt a cache that can return false hits.)
- Origin: numerai `work/0012-ml-core-review` (2026-10-22), findings P0-7, P0-8, P0-9,
  P1-15, P1-16, P1-17 there, and its `reviews/mlops.md` C1–C6 and C10–C13.
- Every finding below was re-checked in this repo on 2026-10-22: by reading the cited
  code, and for item 6 by a YAML probe.

## Why

Signatures are the whole contract of the cache: `step name + code closure + config
subtree + input hashes`. Each item below is a way something that changes a step's
output stays out of that signature, so the step comes back `cached` with a stale result.
None of them has caused a wrong result here yet, because mlpipe's own pipelines happen
to avoid them. A host writing its own steps would hit them at once.

## Items

| # | Finding (verified) | Fix | Signatures move? |
|---|---|---|---|
| 1 | **Undeclared input reads.** `RunContext.get/get_lazy/put_alias` serve any key in the run registry (`core/context.py:60-84`). Only `step.inputs` enter the signature, and `pipeline.py` checks outputs only. A step reading an undeclared key is a false cache hit. | `reset_step(declared)` records `step.inputs`. `get`, `get_lazy` and `put_alias` raise `KeyError("undeclared input …")` for anything else. Test: a step reading an undeclared key fails on its first run. | No |
| 2 | **Cross-section config reads.** Generic `steps/register.py:38,49` reads `config["clean"]` and `config["train"]["model"]`, neither in its signature. It also names an example step's config. If a cleaner edit leaves `clean_table` byte-identical (v6 Int8 has no nulls), register stays cached and the bundle carries the old recipe, which `predict` replays on live data. | `clean` emits a `serve_recipe` JSON artifact (cleaner, era column). `train`'s model artifact records its model config. `register` declares `serve_recipe` as an input. Pipelines with no recipe construct `RegisterStep` without it, so inputs become per instance. Test (AST): `ctx.config.get(<literal>)` in a step names only `self.name` or a top-level location key (`download_dir`). | register, train |
| 3 | **Typos in a step's config vanish.** Every step `config_model` is a plain `BaseModel` (`extra="ignore"`). `Pipeline.run` replaces the section with `model_validate(...).model_dump()` (`core/pipeline.py:85-88`), so `--set train.modle.kind=torch_mlp` trains LightGBM, and the typo disappears from the manifest and the signature too. This contradicts DESIGN.md §5 ("config typos must die at pydantic validation"). | A shared `StrictConfig` base (`extra="forbid"`) in `core/interfaces.py`, used by every shipped config model and its nested models. `Pipeline.run` refuses a `config_model` that doesn't forbid extras, so host steps can't regress. Test: the typo above fails before any step runs. | Only if a stored section had extras |
| 4 | **Host backend helpers are outside `train`'s closure.** `code_modules` follows `code_deps` with the *step's* first-party prefixes (`core/signature.py:113-120`). `hostpkg.backend` is hashed, but its `hostpkg.helpers` import is filtered out, so a helper edit doesn't retrain. (`module_code_hash`, used for the model artifact, does this right.) This is left over from work/0007 M1. | Close over each dep with `_first_party(dep)`. Test with a fake host package. | train, for hosts only |
| 5 | **Sibling code and data files are invisible.** Imports outside mlpipe and the host root (e.g. the editable `shrec` package) never reach a closure. Neither do non-`.py` files a step reads (`holdout_eras.json`, YAML). | `Step.code_roots: tuple[str, ...] = ()`, extra module prefixes to close over (a host sets `("shrec.",)`). DESIGN.md rule: data files reach a step only as seeded artifacts or config values. | No (opt-in) |
| 6 | **`--set` floats can parse as strings.** `apply_overrides` uses `yaml.safe_load` (`core/pipeline.py:35`). YAML 1.1 parses `1e-05` and `1e-5` as **strings** but `1.5e-05` as a float (probed). `tune` builds overrides with `f"{key}={value}"` (`cli.py:113`), and Python writes floats below 1e-4 as `1e-05`. A log-scale space reaching under 1e-4 sends strings into params and signatures. | Parse with `json.loads`, falling back to `yaml.safe_load`. `tune` formats floats so they always parse as numbers. Test both spellings. | No (fixes future runs) |
| 7 | **Provenance records the host's repo, not mlpipe's.** `git_commit`/`git_dirty` run `git` in the cwd (`core/manifest.py:15-31`), and the lockfile lookup is cwd-relative (`core/pipeline.py:45`). In a host, nothing records which mlpipe ran. That breaks the stated reason for leaving `mlpipe.core` out of signatures ("covered by the git_commit in every manifest", `signature.py:23-26`). | `environment.mlpipe = {version, commit}` from `importlib.metadata` plus the package dir. `environment.packages` for a declared list (numpy, polars, lightgbm, …). `Pipeline(project_root=…)` for the host's git and lockfile. A hand-bumped `CORE_SIGNATURE_VERSION` in every signature payload, for behaviour-changing core edits. | **All, once** (the new payload field) |

Item 7's signature field makes every cached artifact go cold once. Ship 2 and 7 together
so the store recomputes once, not twice.

## Not in this item

- Stale live data and the serve path: [work/0009](../0009-live-identity-and-serving/proposal.md).
- Store, index and example polish: [work/0010](../0010-store-and-example-hygiene/spec.md).
- `previous()` making validate's verdict depend on history (review C8): the report's
  `previous_rows` move to manifest meta in work/0010. The pass/fail tolerance itself
  is acceptable as a gate.

## Done when

- All of items 1–7 are fixed, each with a test.
- `pytest` is green and `ruff` is clean.
- DESIGN.md §1 (signature) and §5 (strict config) say what the code does.
- The demo pipeline reruns cold once, then is fully `cached` on the second run.
- `pyproject` version 0.2.0. Human: `uv lock`, tag `v0.2.0`, push.
