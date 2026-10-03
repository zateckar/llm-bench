# Rigorous v10 design and critical review

Reviewed on 2026-10-03. This document describes the current protocol; the
[historical audit](BENCHMARK_REVIEW_2026-10-03.md) covers runs 57, 58, 62 and 65.
Historical scores are retained and are not relabelled as v10 results.

## Question storage and one runtime suite

The former questions.yaml had 51,462 lines. It was parseable but made review,
answer-key maintenance and useful diffs difficult. Static questions now live in
[tests/questions](tests/questions), grouped by category with one question per
file. There are 171 files; the largest has 1,453 lines. Multiline prompts remain
literal text and short arrays remain compact. The initial split preserved the
v9 question fingerprint exactly; subsequent v10 changes deliberately revise it.

[quality_suite.py](quality_suite.py) still assembles one fixed bank: 171 static
questions, 76 generated questions, 20 interactive tasks and eight exact-length
context questions, totalling 275 across 30 categories. Splitting source files
does not introduce selectable banks, profiles or run settings. The recursive
loader rejects duplicate IDs, duplicate YAML keys, ambiguous evaluation
declarations, malformed fields and nonfinite weights.

## Primary research and decisions

| Source inspected | Method that informed this implementation |
|---|---|
| [LiveBench repository](https://github.com/LiveBench/LiveBench) | Objective ground truth across multiple capability categories. Use exact derivations and retain balanced category/family reporting; do not infer capability from keyword mentions. |
| [EvalPlus repository](https://github.com/evalplus/evalplus) | Boundary and generated test combinations expose superficially correct code. Check reference implementations against independent algorithms and reject plausible near misses. |
| [IFEval evaluator](https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_lib.py) | Whole-prompt correctness and per-instruction correctness answer different questions. Keep strict full success as the headline and criterion achievement as a separate diagnostic. |
| [IFBench](https://github.com/allenai/IFBench) | Compose independently verifiable constraints. Add Unicode normalization, revision precedence, tombstone and ordering interactions rather than only increasing text length. |
| [RULER](https://github.com/NVIDIA/RULER) | Long context should include multihop tracing and aggregation, not just recalling a conspicuous needle. Add indirect selection, temporal joins, signed aggregation and distributed evidence. |
| [NoLiMa](https://github.com/adobe-research/NoLiMa) | Literal target matches can make retrieval too easy. The question supplies attribute conditions instead of the target identifier; the model must select the entity before joining its facts. Our tasks remain self-contained, unlike evaluations relying on outside semantic knowledge. |
| [Current tau evaluator](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/evaluator/evaluator.py) | Evaluate environment state, actions and termination separately. Distinguish reaching a target, respecting authorization and explicitly completing the protocol. |

These are original local tasks inspired by methods, not copies or implementations
of the referenced datasets. No third-party benchmark score is claimed.

## Harder tasks and independently checked answers

| New family | Difficulty beyond a simple answer | Independent check |
|---|---|---|
| Robust portfolio | Dependency and exclusion constraints; worst-scenario, total-return, cost and lexical tie breakers; distinct runner-up and objective gap | Combination enumeration versus the generator's Boolean assignment search |
| Causal counterfactual | Observation versus intervention; shared exogenous variables across potential outcomes; benefit and harm probabilities | Integer-mask weighted contingency table and exact rational arithmetic |
| SQL NULL/cardinality | Bag multiplicity, nullable amounts, left joins, distinct counts and NOT IN with NULL | Executing the specified relational operations in an independent SQLite database |
| Weighted policy repair | Required clauses cannot be removed; weighted deletion, count and lexical priorities; witness and total solution count | Optional-clause subset feasibility versus assignment-based derivation |
| Causal register | Partial order rather than latest timestamp; concurrent tombstones, equal clocks, drafts and conflicts | Pairwise dominance differences versus frontier construction |
| Unicode revisions | NFKC, strip, casefold and NFC in a specified order; equal-version precedence and tombstone suppression | Reference code in the real subprocess plus lower/casefold, tie and resurrection mutants |

Each family has four fixed instances: seeds 19 and 23, variants 0 and 1. Family
averaging prevents these repeated instances from dominating a category. Existing
reasoning, scientific, policy, translation and repository-review answer checks
remain covered by their independent oracle and corruption suites.

Context tasks have two families at both 8,192 and 32,768 cl100k_base tokens for
both seeds. Within a seed and family the short and long prompts share the core
problem, allowing context-length comparisons. A selected site is joined through
approved alias and bitemporal quota revisions, or through shipment revisions with
signed quantities and tombstones. Relevant and irrelevant records share the
identifier grammar, and records are never cut at arbitrary token boundaries.
The tests parse the emitted archive independently, recompute every answer and
verify evidence distribution, record uniqueness and exact token counts.

## Grading corrections found during repeated review

1. **Input mutation was untested.** Canonical code fixtures now check the input
   state after each call, including changes between Boolean and numeric values.
   Every protected fixture must pass; a matching return value cannot excuse an
   input-state violation. Restored transient mutations are not detected.
2. **Submitted code was rewritten.** Forbidden imports were silently removed.
   Programs now execute as submitted under the import policy. The active bank
   uses standard-library tasks; unused cloud/network/data-science packages were
   removed from the allowlist.
3. **Rubric dependencies were metadata only.** Failed dependencies now block
   dependent credit transitively and remain in its diagnostic denominator.
4. **Array partial credit was too coarse.** Wrong cardinality still fails the
   full contract, while correct positional members retain diagnostic credit.
   Missing members and blocked nested structures earn zero.
5. **Scalar type failures looked like value failures.** Boolean, string, null
   and numeric mismatches now fail the type contract. Equal numeric values such
   as 1 and 1.0 remain equivalent; numbers are not coerced from strings.
6. **JSON envelope and numeric parsing had gaps.** Explicit no-Markdown tasks
   reject fences. Duplicate keys, nonfinite literals and overflowing numeric
   exponents are invalid. There is no picking a correct object from prose.
7. **Interactive termination was imprecise.** Exhausting turns is not protocol
   completion. Goal state and authorization have separate diagnostics; an
   unauthorized action remains a full-pass failure even if the target is reached.
   Interactive evaluator versions are recorded in the comparison protocol.
8. **Context filler marked itself as irrelevant.** Distinct filler prefixes
   were removed. Every record uses the same identifier grammar, without special
   answer-bearing E1/E2 labels.
9. **Source validation accepted ambiguous data.** Duplicate YAML keys, nonfinite
   weights, fractional/Boolean token caps and multiple evaluation declarations
   now fail authoring validation instead of silently changing the fixture.
10. **Runtime packaging and scheduling needed follow-up.** Docker excludes the
    complete data directory, including local retired archives and database
    sidecars. The shared form rejects nonexistent spring-transition times and
    preserves the stored offset when editing an unchanged ambiguous autumn time.
    Scheduled and immediate runs retain identical benchmark settings and dispatch.

Review was iterative: storage and evaluator inspection, then independent oracles
and mutation controls, then integration/reporting/packaging inspection. Obsolete
tests expecting import rewriting or the previous revision were updated to assert
the new contract. The final review reruns validation, regressions, corruption
checks, queue checks, template compilation, form JavaScript and lint. No known
unresolved correctness defect remains from these review passes; this is not a
proof that the benchmark or application is defect-free.

Final local validation passed: 275 questions with zero authoring errors or
warnings; 125 regression tests; 120 evaluator assertions; 1,133 challenge checks;
5,550 corrupted ceiling, stretch and specialist answers rejected; all queue
checks; all 18 templates compiled; the form JavaScript passed payload, row-limit,
timezone and DST checks; Ruff and Git whitespace checks passed. The 16 new
frontier tests also passed separately after the final evaluator-version assertion.

## Interpretation and limits

Primary quality remains strict task success, balanced first by family and then by
capability category. Partial requirement achievement, type/envelope compliance,
availability, authorization and call efficiency remain distinguishable. Creative
writing measures formal compliance separately and does not judge artistic quality.
Endpoint errors, unsupported context and cancellations remain visible with
coverage and missing-outcome sensitivity bounds rather than being silently dropped.

Fresh matched model runs are required to establish empirical difficulty, ranking
and model separation. Four generated instances do not constitute repeated-run
reliability or a pass^k estimate. Public seeds and reference tests are not a secret
holdout. These are bounded code functions, closed-world policy exercises and
text-protocol simulations; they do not establish unrestricted agent capability,
native tool-calling performance, complete repository repair or translation fluency.
Reference token lengths can differ from provider tokenization. Local subprocess
containment is not a security boundary against determined adversarial Python;
production execution of untrusted code still requires container/VM isolation.

Reproduce the authoring and evaluation checks with the commands in the
[README](README.md#verification). The dedicated new oracle and regression suite is
[selftest_frontier.py](selftest_frontier.py).
