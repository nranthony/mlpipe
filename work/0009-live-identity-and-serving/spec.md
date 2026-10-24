# Spec: serve pipeline and inputs named in config (work/0009)

- Status: Ready 2026-10-24. Cross-checked by a fresh-context review the same day; its 17
  findings are folded in (the "Review" section at the end lists them).
- Decision: [ADR-0007](../../docs/adr/0007-serve-pipeline-and-named-inputs.md)
  (Accepted 2026-10-23). This spec says how; the ADR says what and why.
- Scope: this repo only. Part A is the library (`core/`, `steps/`, `cli.py`); part B is
  the Numerai example (`examples/numerai/`). Nothing here touches a host repo. Part A
  must not name anything Numerai (`tests/test_work0006_phase2.py` enforces it for
  `cli.py`).

## Part A — library

### A1. `run_id` (core)

- `Pipeline.run` draws `run_id = uuid.uuid4().hex` once, and adds it to `base` in
  `_run_step`, so every record it writes carries it: `success`, `cached`, `failed`.
- `mlpipe promote` writes its manifest with a fresh `run_id` of its own.
- Not part of any signature. `find_cached` is unchanged.

### A2. Sibling steps in config validation (core)

Today `Pipeline.run` calls `validate_config(config, names)` with its own step names, so
any other step's subtree is an unknown top-level key. A module's training and serve
lists share one config, so each must accept the other's subtrees.

- `Pipeline.__init__` gains `sibling_steps: Iterable[str] = ()`: names whose subtrees
  pass top-level validation untouched. They are not validated by this pipeline and
  enter no signature here.
- `validate_config(config, names)` is called with `names + sibling_steps`. A name in
  both is simply one of this pipeline's own steps (validated as usual); no error.
- **Rule for every caller:** `sibling_steps` = (all of the module's training and serve
  step names) − (the steps this `Pipeline` runs). So `predict --input`, which drops
  `ingest_live`, passes `ingest_live` as a sibling too.
- `cli._load` validates the top level against training **and** serve names, and
  returns both lists. `run`, `tune`, `promote` and `predict` all apply the rule above.
- `build_serve_steps()` is called by `run`/`tune` just for names, so serve modules
  must import cheaply (numerapi imported lazily inside functions, as `ingest.fetch`
  does today).

### A3. `PredictStep` (`steps/predict.py`, new)

```python
class PredictStep(Step):
    name = "predict"
    outputs = ["predictions"]
    def __init__(self, frame_key: str = "input_frame", prepare: Callable | None = None): ...
    # inputs = ["model_bundle", "model", "fitted_transformer", frame_key]
```

- `prepare(frame, bundle) -> (features, ids)`. Default (moved from `cli._default_prepare`):
  `frame.select(bundle["features"])` and `frame.select(bundle.get("id_columns", []))`.
- `code_deps` returns `[prepare.__module__]` when a `prepare` is given, so editing it
  re-runs predict. The constructor requires a plain module-level function
  (`inspect.isfunction`, not a lambda, not from `__main__`); a `functools.partial`
  would report `functools` and hash the wrong code. Anything else is a `TypeError`.
- `run`: `bundle = ctx.get("model_bundle")`, `transformer = ctx.get("fitted_transformer")`,
  `model = ctx.get("model")["full"]`, then the current CLI arithmetic:
  `transformer.transform(feats.to_numpy())` → `model.predict(pl.from_numpy(matrix,
  schema=bundle["features"]))` → `ids.with_columns(pl.Series("prediction", preds))`.
  `ctx.put(result, "predictions")`. Meta: `rows`, `bundle` (hash prefix is fine),
  `model_name`, `plan_hash`.
- The step checks that the seeded `model` and `fitted_transformer` hashes equal
  `bundle["model_hash"]` and `bundle["transformer_hash"]` (`ctx.input_hashes`); a
  mismatch raises. The CLI seeds them from the bundle, so this guards other callers.
- No config model: `predict` has no subtree. A `predict:` key in config is an error as
  usual (no `config_model`).
- **Backend code.** The `model` artifact's `backend_code` is written at train time;
  serving a pinned bundle retrains nothing, so a later backend edit would not move
  predict's signature. `run` therefore compares `model["backend_code"]` with
  `module_code_hash(type(model["full"]).__module__)` and raises ("backend code changed
  since this bundle was trained; retrain") on a mismatch. A miss never computes under
  code the bundle wasn't trained with; a hit returns what that code produced.
  (`code_deps` can't do this: it sees the config, not the bundle.) DESIGN.md §1's
  "consumers re-run because train re-runs" gets this serving caveat.

### A4. `mlpipe predict` runs the serve steps (cli.py)

Contract for a pipeline module (DESIGN.md §5):

- Optional `build_serve_steps() -> list[Step]`. Its first step is the **source**: it
  has no inputs and exactly one output. Its last step is a `PredictStep` (by
  `isinstance`). Checked at load; violations are a `BadParameter`.
- No `build_serve_steps`: the serve list is `[PredictStep()]` and `--input` is
  required.
- `serve_input` and `prepare` module hooks are no longer read. A module that still
  defines either gets a `BadParameter` naming ADR-0007. A silently ignored `prepare`
  would serve features without the host's fill: wrong predictions, no error.
- Order of checks: bundle resolution first (so `predict --pipeline demo` with no pin
  still says `demo.champion`, `test_work0010_hygiene.py:89`), then the module
  contract, then `--input` required-ness.
- `--set predict.x=…` (a `TypeError` in `Pipeline.run`) and pydantic validation
  errors (e.g. a missing `ingest_live.round`) are mapped to `BadParameter`.

`predict` options: `--bundle`, `--alias`, `--pipeline`, `--input`, `--set` (repeatable),
`--tag`, `--config`, `--store-dir`, `--manifests-dir`, `--baselines`, `--out`.

Flow:

1. Resolve the bundle hash exactly as today (`--bundle` prefix, else the
   `<pipeline>.<alias>` pin).
2. Seeds: `model_bundle`, plus `model` and `fitted_transformer` from the bundle's
   `model_hash` / `transformer_hash`. Each must exist in the store, else `BadParameter`
   naming the missing hash.
3. Steps: the serve list. With `--input`:
   - module with serve steps: drop the source step; snapshot the file
     (`store.save(Path, key, ext="parquet")`, streamed) under the source's output key
     and seed it;
   - module without: seed it under `PredictStep.frame_key`.
   - Either way the CLI writes an `input` record first, so lineage doesn't stop at a
     bare hash: `{"step": "input", "status": "success", "signature":
     "input:<hash>", "inputs": {}, "outputs": {<key>: <hash>}, "meta": {"file":
     <path>, "bytes": n}, "run_id": <new uuid>}` (`lineage` follows success/cached
     only, `manifest.py:98`).
4. `Pipeline(steps, store, manifests/<pipeline>, cfg, tracker=None,
   sibling_steps=<all names − steps run>).run(overrides=set_, tag=tag, seeds=seeds)`.
   No tracker: serving logs to manifests only.
5. Copy the `predictions` artifact's store file to `--out` (`shutil.copyfile`; the out
   file's sha256 equals the artifact hash).
6. Echo: model name, bundle prefix, plan prefix, out path, rows, predictions hash
   prefix, and each step's status line (as `run` prints).

Lineage: `mlpipe lineage <predictions hash>` walks predict → bundle → register → … →
ingest, because the manifest log is shared and `lineage` matches by hash.

### A5. Demo pipeline

`steps/demo.py` has no model, so it gets no serve list. `mlpipe predict --pipeline demo`
keeps failing as today (no pin), and the second-domain test keeps using `--input`.

## Part B — Numerai example

### B1. Training ingest named by snapshot (`ingest.py`)

- `IngestConfig`: `version: str = "v6.0"`, `snapshot: str` (required, `YYYY-MM-DD`,
  validated as a date).
- Files: `train.parquet`, `validation.parquet`, `features.json` (no `live.parquet`),
  read from `<download_dir>/<version>/<snapshot>/`.
- `fetch(version, snapshot, filename, download_dir)`:
  - file exists → return it;
  - missing and `snapshot` is today's UTC date → download (atomic, see B4);
  - missing otherwise → `FileNotFoundError` saying the folder doesn't exist, to copy the
    files there or use today's date. A snapshot folder must never hold bytes from a
    later day than its name.
  - *Deviation from ADR decision 5's wording* ("downloaded on first use"): only a
    folder named for today is downloaded. Consequence: a fresh checkout's `mlpipe run`
    with the default `2026-10-20` fails with that message until the files are copied,
    or `--set ingest.snapshot=<today>` downloads ~3.57 GB. Recorded in the ADR's
    status line.
- Outputs: `raw_train`, `raw_validation`, `features_meta`. Meta `file` becomes
  `<version>/<snapshot>/<name>`; `dataset` adds `snapshot`.
- `DEFAULT_CONFIG["ingest"] = {"version": "v6.0", "snapshot": "2026-10-20"}` (the
  existing files' download date; the human copies them, see Human steps).

### B2. Training validate (`validate.py`)

- `PASS_THROUGH` drops `raw_live`. Inputs: `raw_train`, `raw_validation`,
  `features_meta`. Outputs: `validated_train`, `validated_validation`,
  `validation_report`.
- `check_table` is reused unchanged by B3.

### B3. Serve steps (`examples/numerai/serve.py`, new)

`IngestLiveStep` (`ingest_live`, the source):

- Config `IngestLiveConfig`: `version: str = "v6.0"`, `round: int` (required, `> 0`),
  `era_column: str = "date"`, `max_age_days: int = 7` (`> 0`).
- No inputs; output `raw_live`.
- Path `<download_dir>/<version>/live/<round>/live.parquet`. If absent, download it
  with `NumerAPI().download_dataset(f"{version}/live.parquet", <part>,
  round_num=round)` (B4).
- `open_time = round_open_time(round)`: a module function over numerapi
  `get_competitions()` (`number`, `openTime` as a datetime). Called every time the step
  runs (never on a cache hit); tests monkeypatch it. No round record is written to the
  shared folder. A round file another tool already downloaded is checked the same way.
- **Freshness (ADR-0007 decision 6), two-sided.** Fail if the era column has more than
  one value, is not an ISO date, is more than `max_age_days` before `open_time`'s
  date, **or is after it**. The likeliest wrong file in a backfill is the *current*
  round's, whose date is later than the requested round's open time (round 1378: data
  `2026-10-15`, opened the day after).
  - On a download, the check runs on the part file before it is renamed into place
    (B4), so a wrong file never lands at the immutable path. On an existing file, a
    failure names the path; removing it is a human step (decision 7).
  - *Deviation from the ADR's wording:* the ADR puts the check in `validate_live`, and
    is one-sided. It lives in `ingest_live` because only the source knows the round,
    and `--input` replaces the source with a file that has none. Both recorded in the
    ADR's status line.
- Meta: `file`, `bytes`, `rows`, `round`, `open_time`, `live_date`.

`ValidateLiveStep` (`validate_live`):

- Inputs `raw_live`, `model_bundle`; outputs `validated_live` (alias), and
  `live_validation_report` (JSON).
- Config `ValidateLiveConfig`: `era_column`, `feature_dtypes`, `max_feature_null_frac`,
  `max_feature_null_frac_by_dtype` (same meanings as `ValidateConfig`; no
  `row_tolerance`: one live file has no previous snapshot to compare with).
- Expected features are the bundle's `features` (what the model needs), not
  `features_meta`. Runs `check_table(..., ValidateConfig(**live_cfg.model_dump()),
  prev_rows=None)`: `check_table` reads `cfg.row_tolerance` unconditionally
  (`validate.py:108`), so it gets a full `ValidateConfig` with the default. Fails loud.

`prepare(frame, bundle)`: today's `pipeline.prepare`, moved here unchanged (cleaner
fill from `bundle["recipe"]`, ids `numerai_ticker` + era as `era`).

### B4. Atomic downloads

numerapi's `download_file` is already atomic onto whatever path it is given: it writes
`<path>.temp` and `os.replace`s at the end (`numerapi/utils.py:67,100-105`), resuming
from an existing `.temp`. So the training snapshot fetch (B1) passes the final path
directly. Only `ingest_live` needs a part file, so the freshness check runs before the
file reaches the immutable path:

- download to the fixed name `<final>.part` (no pid: numerapi can then resume its
  `.part.temp`), check it, then `os.replace(part, final)`;
- on a failed check or any exception, remove `<final>.part` and `<final>.part.temp`
  (both are files this step created, so decision 7 allows it), then re-raise.

### B5. `pipeline.py`

- `build_steps()` unchanged in shape (`IngestStep` now has three outputs).
- `build_serve_steps() -> [IngestLiveStep(), ValidateLiveStep(),
  PredictStep(frame_key="validated_live", prepare=serve.prepare)]`.
- `serve_input` and `prepare` removed.
- `DEFAULT_CONFIG` adds `ingest_live: {"version": "v6.0"}` (no round: it must be passed)
  and `validate_live` with the same dtype and null settings as `validate`, and
  `era_column: "date"`.
- Docstring: `mlpipe predict --pipeline numerai --set ingest_live.round=N`.

### B6. The download folder is shared (ADR-0007 decision 7)

The example writes only under `<version>/<snapshot>/` (only when the date is today) and
`<version>/live/<round>/`, and only new files. Nothing in the example moves, rewrites
or deletes a download file. A test asserts the flat `<version>/*.parquet` files are
untouched by a run.

## Tests

New `tests/test_work0009_serve.py` (test names start `test_work0009_`):

- A1: one run's records share one `run_id`; two runs differ; cached and failed records
  carry it.
- A2: a sibling subtree passes; an unknown top-level key still fails; overlapping names
  raise.
- A3: `PredictStep` predicts like the old CLI on a toy bundle; a `prepare` edit changes
  the signature (tmp package, as `test_work0010_a_host_scorer_edit_reruns_evaluate`);
  a seeded model whose hash isn't the bundle's raises.
- A4:
  - `predict --input` on a pipeline without serve steps writes an `input` record and a
    manifest with `predictions`; `--out` bytes hash to it; a second identical call is
    `cached`.
  - `lineage <predictions>` reaches `input`, `register` and `ingest`.
  - a module with `serve_input`, or with `prepare`, fails with ADR-0007 in the message;
  - a backend edit after training makes a cache-miss predict raise;
  - `prepare` given as a lambda or a `functools.partial` is a `TypeError`;
  - `--set predict.x=1` and a missing required serve field are `BadParameter`, not
    tracebacks;
  - a serve list whose first step has inputs, or whose last isn't a `PredictStep`, is a
    `BadParameter`;
  - a bundle whose model hash isn't in the store is a `BadParameter`.
- B1: snapshot paths; missing folder with a past date raises without touching the
  network; today's date calls the (monkeypatched) downloader; `snapshot` missing is a
  validation error.
- B3:
  - two rounds → two paths and two signatures; the second round is a cache miss;
  - an absent round file is downloaded with `round_num=<round>` (monkeypatched
    `NumerAPI`);
  - freshness: a live date 8 days before `open_time` fails, 6 passes, 1 day after
    fails, two dates fail, a non-date fails;
  - a failed freshness check on a download leaves nothing at the final path and no
    `.part`/`.part.temp`;
  - `validate_live` fails on a missing bundle feature, and on a live null fraction over
    the bound (the coverage `test_cycle2_validate.py:100-107` had through `raw_live`);
  - `ingest_live` without `round` fails at validation.
- B4: a downloader that raises leaves no file at the final path and no `.part`.
- B5 end to end through the CLI (toy data): `run`, `promote`, then
  `predict --set ingest_live.round=N` → three serve manifests, output columns
  `numerai_ticker, era, prediction` (or `era, prediction` without a ticker); and
  `predict --input <file>` skips `ingest_live` but runs `validate_live`.
- B6: the flat files' bytes and mtimes are unchanged after a run plus a predict.

Updated:

- `test_cycle1_ingest.py`: snapshot folder, no live; the `round` typo test now uses
  another unknown key.
- `test_cycle2_validate.py`: no `raw_live` (its live null-bound case moves to B3).
- `test_cycle9_10_tune_register.py`: train/validation under `v0-toy/<snapshot>/`; a
  live file with a **single ISO date** in its era column under `v0-toy/live/<N>/`;
  config adds `ingest: {version: v0-toy, snapshot: …}`, `ingest_live: {version:
  v0-toy, era_column: era}`, `validate_live: {era_column: era}`. The fresh-process
  predict test can't monkeypatch `round_open_time`, so it moves to `--input` (or
  in-process `CliRunner` with monkeypatch for the serve flow).
- `test_work0006_phase1.py`: seeds without `raw_live`; `test_shipped_default_configs_
  validate` (`:152-157`) validates numerai's `DEFAULT_CONFIG` against training + serve
  names.
- `test_work0006_second_domain.py`, `test_work0010_hygiene.py`: predict flags only if
  needed; the hygiene test's `demo.champion` message must survive (check order, A4).

Done when: `uv run --no-sync pytest tests/ -q` green, `ruff check` clean, the
orchestration line count is reported in STATUS (A1 + A2 add a few lines to
`pipeline.py`).

## Docs

- DESIGN.md:
  - §0 (config: one subtree per step; siblings now allowed);
  - §1 manifest fields (`run_id`) and the backend-code caveat for serving (A3);
  - block 1 (no `raw_live`; snapshot folder), block 2;
  - a serve row or paragraph (ingest_live → validate_live → predict);
  - §3 `Pipeline` signature (`sibling_steps`);
  - §5 Serving (new contract; hooks gone; `--set ingest_live.round=N`; the `input`
    record);
  - the data notes on `live.parquet`.
- `plugins.py` docstring: the optional `build_serve_steps()`. `cli.py` predict help and
  docstring.
- `ManifestLog.latest_outputs`: remove it if nothing else uses it once `serve_input`
  is gone.
- STATUS.md current state + log. ADR-0007 status line: the B1 and B3 deviations, and
  the corrected size (~3.57 GB, not ~2.5 GB).

## Real data (after the tests)

- Training: with the copied snapshot, a rerun re-executes `ingest` and `validate` only.
  Raw hashes are unchanged (same bytes), `validated_train`/`validated_validation` alias
  the same hashes, so `clean` onward are cache hits and the bundle stays `1d3c948c`.
  This is the cache-hit proof for this item.
- Serving: one `predict --set ingest_live.round=<current>` on the real bundle. Needs
  network for numerapi, so it may be a host step.

## Human steps

1. Copy (not move, not link) the existing files into the snapshot folder:
   ```bash
   mkdir -p /workspace/raw_data/numerai/v6.0/2026-10-20
   cp -p /workspace/raw_data/numerai/v6.0/{train.parquet,validation.parquet,features.json} \
         /workspace/raw_data/numerai/v6.0/2026-10-20/
   ```
   ~3.57 GB. A hard link is wrong: a tool that rewrites the flat file in place would
   change the snapshot too.
2. Promote a bundle (e.g. `1d3c948c`) before a real `predict`, if wanted: `baselines.yaml`
   has no valid pin.
3. The real serve run, if the sandbox can't reach the Numerai API.

## Out of scope

- A host's round loop, submission, or champion record.
- A generic "source step" library class: the contract is structural (no inputs, one
  output).
- Pruning old round folders or snapshots.

## Review (2026-10-24, fresh-context Opus pass)

All 17 findings were checked against the code and folded in:

| # | Finding | Where it landed |
|---|---|---|
| 1 | `predict --input` rejects the dropped source's subtree | A2 sibling rule |
| 2 | `check_table` reads `row_tolerance` | B3 `validate_live` |
| 3 | backend code doesn't reach predict's signature | A3 backend-code check |
| 4 | freshness check is one-sided | B3, two-sided |
| 5 | wrong file committed before the check | B3/B4, check the part file |
| 6 | part file clashes with numerapi's `.temp` | B4 |
| 7 | B1 changes decision 5 unflagged | B1 deviation note |
| 8 | stale `prepare` hook silently ignored | A4 |
| 9 | `--input` origin unrecorded | A4 `input` record |
| 10 | unlisted test breaks | Tests, "Updated" |
| 11 | sibling overlap error unneeded | A2, dropped |
| 12 | `prepare.__module__` wrong for partial/lambda | A3 |
| 13 | docs and dead code | Docs |
| 14 | tracebacks instead of usage errors | A4 |
| 15 | serve module import cost | A2 |
| 16 | `round.json` unneeded | B3, dropped |
| 17 | copy size | Human steps, ADR note |
