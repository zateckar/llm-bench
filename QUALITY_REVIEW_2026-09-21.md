# Quality review — 2026-09-21

This review records the read-only audit of runs 51–58 and the machine-generated
queue for human disposition. It does not rewrite historical grades. The source
snapshot is [data/quality-audit-2026-09-21.json](data/quality-audit-2026-09-21.json);
the answer-free queue is [data/quality-review-queue-2026-09-21.json](data/quality-review-queue-2026-09-21.json).

The audit found no stored-count, duplicate-ID, score-bound, threshold, or saved
score consistency mismatches in any included run. Replay of unchanged static
questions produced no verdict or score deltas for the covered responses. Replay
was intentionally skipped for incomplete generations, changed output contracts,
and expanded historical code fixtures; those are coverage limits, not evidence
that the original answer was correct.

| Runs | Cohort | Questions | Scored | Passes | Main runtime outcomes |
|---:|---|---:|---:|---:|---|
| 51–53 | quality-v5 | 404 each | 404, 404, 402 | 310, 328, 335 | truncation 127; repetition 46; endpoint errors 2 |
| 54–58 | quality-v6 | 434 each | 423, 425, 423, 430, 434 | 361, 372, 347, 396, 264 | endpoint errors 35; repetition 69; missing answers 12 |

The v6 cohort remains the calibration baseline because all five runs share the
same suite hash, quality protocol, development split, seed, worker count, and
output budget. The v5 cohort stays separate. Endpoint aliases are not treated as
verified model identities, and runtime failures remain separate from content
difficulty.

The generated review queue contains 638 answer-free entries across all eight
runs: 236 scored capability non-passes, 288 runtime failures, and 114 deterministic
success samples (two per populated category/evaluator group where available).
The queue preserves run, question, evaluator, outcome, score, detail excerpt,
and replay coverage while omitting saved answers. A reviewer should assign one
of the dispositions in the plan before using an item for a rescore or difficulty
calibration:

- correctly graded task failure;
- grader false failure or false pass;
- ambiguous prompt/output contract;
- incorrect or incomplete key/fixture;
- legitimate formatting violation;
- endpoint/runtime failure or premature repetition;
- unresolved and requiring an independent derivation.

Priority review groups are the structural-contract families (`S6-inheritance-*`,
`S6-bitemporal-*`, `S6-scopedstate-*`), specialist evidence disagreements,
ordered path/source requirements, partial code fixtures, and all repetition or
missing-answer outcomes. Agreement among model outputs is recorded as a review
signal only; it is not a correctness proof.

The remaining operational gate is endpoint-dependent: run at least three matched
hardening-v7 pilot reports, disposition universal successes/failures, and then
use `hardening_release.py` to select standard/challenge families. Until then,
the release artifact is explicitly `pending_model_pilot` and the frozen v6
profile remains the default.

Recreate the audit and queue with:

```powershell
uv run python audit_results.py data/bench.db --runs 51 52 53 54 55 56 57 58 --replay --output data/quality-audit-2026-09-21.json
uv run python review_queue.py data/quality-audit-2026-09-21.json --runs 51 52 53 54 55 56 57 58 --output data/quality-review-queue-2026-09-21.json
```
