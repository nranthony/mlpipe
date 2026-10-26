# ADR-0009: The Numerai example is a real-data test harness, not a tournament pipeline

- Status: Proposed
- Date: 2026-10-26
- Deciders: nanthony + Claude (work/0012; weighs work/0011 finding A against ADR-0007
  decisions 4-7)

## Context

ADR-0005 kept Numerai in this repo as "a real-data test case". Since then the example
has grown into a second tournament pipeline:

- work/0010 tuned it against Numerai metrics.
- work/0009 (ADR-0007 decisions 4-7) gave it round-keyed live data and real serving.

The numerai repo, meanwhile, runs the actual tournament and is adopting mlpipe's core
(work/0006 step 14).

The work/0011 audit (2026-10-23) found the cost of having both:

- **The two repos disagree on things that should match:**
  - correlation: a hand-written `numerai_corr` here, `numerai_tools` there;
  - sharpe: `ddof=1` here, `ddof=0` there;
  - target: `target` here, `target_jupiter_60` there;
  - CV: different fold maths, so the two repos' plan hashes never match.
- **Both look authoritative.** An agent in either repo can mistake the example for the
  tournament pipeline.
- **The library carries Numerai.** `numerapi` sits in the generic `data` extra, and
  `"numerai"` is a built-in pipeline that every host install sees.

The audit proposes shrinking the example to synthetic data. That would remove the only
place mlpipe runs on real data. The real-data runs have earned their keep:

- they caught the v5→v6 schema change;
- the past-round live fetch was found on real data;
- the 3.5 GB cache-hit proof (2026-10-26) and the first serve runs (rounds 1384, 1378)
  are this library's only evidence at scale.

The numerai repo can't take over that role until step 14 lands.

## Decision

1. **The example's role is a real-data test harness.** It exists to exercise mlpipe's
   library on real data at real scale. It is not a tournament pipeline, and that has
   three consequences:
   - **No submission**, and no code path toward one.
   - **Its scores are never compared with the numerai repo's.** The plan hashes differ,
     so a comparison is invalid by AGENTS.md's rule.
   - **Its metrics only check the harness.** No tournament decision is taken on its
     metrics, studies or promoted bundles.
2. **ADR-0007 stands.** Decisions 1-3 (library) are unaffected. Decisions 4-7 stay,
   because serving by round is how the harness exercises named inputs and the serve
   list.
3. **Where the repos diverge, nothing new is copied into the example.** A domain
   behaviour the harness needs either goes into a generic `steps/` module that both
   repos can use, or follows the numerai repo's version. The example's existing
   divergences are listed (work/0012), not fixed here.
4. **The library stops carrying Numerai where that is cheap** (work/0012):
   - numerapi moves to its own extra;
   - the example's registration no longer collides with a host's pipeline names;
   - Numerai wording comes out of the generic modules.
5. **Re-decide at a set trigger.** When the numerai repo runs on mlpipe's core with a
   real-data cache hit (work/0006 step 14), there are two outcomes:
   - If that repo can serve as mlpipe's real-data check (it pins a tag, so a new
     mlpipe change is proven there before release), a new ADR supersedes ADR-0007
     decisions 4-7, and the example shrinks to a synthetic Numerai-shaped fixture.
   - Otherwise the harness stays, and this ADR records why.

## Consequences

- Until the trigger, mlpipe changes keep their real-data proof here, at the cost of
  keeping numerapi and the Numerai download folder in this repo's development loop.
- The example's MLflow experiment, Optuna study and `baselines.yaml` pins are harness
  state only. work/0012 renames the defaults so they say so.
- STATUS and DESIGN keep only harness-relevant Numerai facts; tournament operations
  belong in the numerai repo.
- The work/0011 "shrink the example" fix is deferred to the trigger, not dropped.

## Alternatives considered

- **Shrink the example now** (work/0011 finding A). Rejected for now: until step 14,
  mlpipe changes would have no real-data check, including the changes step 14 itself
  needs. It becomes the default at the trigger.
- **Make the example the reference tournament pipeline and port host logic into it.**
  Rejected by ADR-0005 decision 2 / alternative 3: it rebuilds working code in the
  wrong repo.
- **Leave it as is.** Rejected: the divergence and the agent confusion grow with every
  feature the example gains.
