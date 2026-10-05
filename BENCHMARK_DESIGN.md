# Rigorous v15 design and critical review

Review completed on 2026-10-04. This document describes the current protocol.
Historical scores are retained and are not relabelled as v15 results. The v10-v14
review records below describe earlier passes; the v15 review records the current changes.

## Current scoring, caching and storage revision

As of 2026-10-04, `criterion-achievement-v1` makes saved criterion achievement the primary question score. Questions without content criteria retain their evaluator score, including format failures at zero. Family and category aggregation averages those fractional values; no category pass threshold is used. Original evaluator scores and full-pass verdicts remain available separately. Saved reports are reaggregated from their recorded criteria without another model call or evaluator replay; their original suite fingerprints and generation protocols remain unchanged. Earlier strict scores in the dated reviews below describe the earlier scoring convention.

Performance v8 pairs cold and primed warm prefixes at 8K/32K reference tokens. Context-sweep v3 pairs both modes at every supported context/load/effort cell. Suffixes differ across warm requests to avoid identical-response cache hits. Provider cache reporting distinguishes missing telemetry from explicit zero. Uncached prefill and effective input rate use client TTFT; decode uses non-burst reported output over delivery span. Queue/network/first-token effects prevent interpreting these proxies as engine kernel rates. The method follows [vLLM prefix reuse](https://docs.vllm.ai/en/v0.9.2/features/automatic_prefix_caching.html), which avoids repeated prefill computation. Priming is excluded from timed measurements and disclosed in setup budgets.

Large SQLite payloads now use lossless, versioned compression with transparent application readers. Historical maintenance reserves the writer slot, requires an idle queue, creates a compressed consistent backup, verifies compression roundtrips, updates score projections, and vacuums free pages. Indexed numeric scores and outcome labels remain queryable without decompressing audits. No answer, criterion or timing evidence is discarded. `selftest_storage.py` verifies restore, preservation, mixed legacy/compressed reads, timestamp behavior and the idle guard; `selftest_cache.py` verifies rate denominators, telemetry coverage and paired prefix structure.

## V15 empirical calibration and behavioral reconstruction

The read-only [calibration snapshot](reports/calibration-v14-2026-10-04.md),
[per-question data](reports/calibration-v14-2026-10-04.json), and
[contract review](reports/calibration-v14-2026-10-04-review.md) use completed runs
80 and 81 on identical v14 suite `b11260da421820b5` and identical recorded
single-call protocols. Their balanced strict scores are 83.78% and 83.11%.
Among 312 capability tasks, 227 pass in both runs, 52 split, and 33 fail in both.
These are descriptive observations from two models with one run each. Run 82 was
still running at the snapshot and is excluded; no endpoint call, database write,
queue change, or historical score update was performed for calibration.

The calibration separates 24 and 26 completion/format failures, respectively,
from 23 and 21 other contract failures and from completed content mismatches.
Source authoring validation had zero errors/warnings across all 315 v14 tasks.
The saved answers nevertheless exposed underspecified nested field names and
representations: queue repairs need explicit `removed`, `order`, `final_queue`,
`count` keys; transaction edges need `[source,target]` arrays; shipment evidence
needs record-ID strings; a path string needs concatenated node IDs. V15 clarifies
these 13 instances while preserving answers and strict grading. Representation-only
audit mappings make 13 additional historical submissions match the keys, but
these are diagnostic counterfactuals and are not revised benchmark results.
Families failing primarily through truncation are completion-constrained in this
panel; neither they nor ambiguous-contract failures establish semantic difficulty.

The new [calibration tool](calibrate_suite.py) reads one SQLite snapshot in
read-only/query-only mode without application startup or credentials. Complete
reports are pooled only when full protocols and complete fingerprint/category/
family/scope inventories agree. Duplicate run/question identities, invalid
verdicts and hashes, partial runs, and incompatible schemas are rejected.
Repeated evidence is balanced by model. At least two models and three fully
scored runs per model are required for repeated-panel evidence. No task is
automatically pruned; variants are never counted as repeated model runs.

The public method inspiration is [VulcanBench's behavioral-parity suite](https://github.com/morganlinton/VulcanBench#suites),
its [fail-to-pass/regression task format](https://github.com/morganlinton/VulcanBench/blob/main/docs/TASK_CONTRIBUTION.md),
and the [candidate admission charter](https://github.com/morganlinton/VulcanBench/blob/main/tasks/coding-intelligence-index-v4/CHARTER.md).
We adopt observed behavior as ground truth, preserve regression behavior, and
distinguish source correctness from empirical difficulty. We do not adopt its
latency-based difficulty threshold or composite score. No third-party tasks,
binary artifacts, evaluator code, or benchmark scores are copied.

Two original reconstruction families add eight instances in a new capability
category, using seeds 19/23 and two variants. Each instance selects one of four
legacy profiles with at least two differences from the naive rewrite. The profile
does not appear in the model prompt. The model can issue up to 12 domain-valid
probes, submit complete Python once, and explicitly finish. The final program
is graded after completion using the existing out-of-process code evaluator;
no hidden test, expected output, or grade is returned during the interaction.

| Family | Interacting behavior | Exhaustive grading | Independent oracle |
|---|---|---|---|
| Signed quantizer | Floor versus truncation; offset before versus after division; reflection versus signed saturation; final cap | Every combination of value -6..6, divisor 2..4, offset -2..2, cap 1..3: 585 inputs | Exact rational arithmetic and independent integral rounding |
| Capacity-2 TTL cache | Inclusive versus strict expiration; fixed versus sliding expiry; whether successful reads update eviction recency; overwrite and expiration order | Every sequence of zero through four operations from the seven-command alphabet: 2,801 sequences | Timestamped dictionaries with independent eviction priorities, versus an ordered-map reference interpreter |

Both groups of fixtures (regression and deviation from the naive rewrite) carry
half the diagnostic content weight. All fixtures remain mandatory for strict
success. Inputs are protected from mutation, invalid/out-of-budget probes and
actions after submission fail the protocol, at least one observation is required,
and evaluator infrastructure failures remain unscored. Cancellation and unfinished
interactions do not execute submitted code. The task browser identifies the new
grader separately from state-change simulations.

Repeated review caught an initially unobservable quantizer rounding choice when
negative values were clamped to zero. The final signed-saturation domain makes
all eight policy profiles distinguishable. Independent observing agents identify
every selected profile in two probes; this establishes solvability, not model
difficulty. All 13,544 new expected fixture outputs match the independent oracles,
and all 56 alternative-policy programs fail actual subprocess grading. Source
correctness gates pass; empirical admission remains `unmeasured` pending fresh
v15 model runs. The suite is `rigorous-v15`: 323 tasks across 31 categories,
including 28 interactive tasks, with fingerprint `7909c6325d1abd7b`.

The new tasks measure bounded black-box inference and replacement functions, not
compiled-binary reverse engineering or full repository repair. Public generation
and seeds are not a contamination-free holdout. Historical single-call results
and current bounded-finalization results remain separate protocols.

Final verification on 2026-10-04: strict validation reports zero errors and warnings
for all 323 tasks; the complete unittest discovery passes 240 tests. Evaluator
self-tests pass 120 assertions, challenge checks pass 1,133 checks over 56 questions,
and existing corruption tests reject 14,145 incorrect answers. The reconstruction
tests add the exhaustive oracle and 56 alternative-policy rejection checks above.
Ruff and Git whitespace checks pass. Repeated assembly confirms fingerprint
`7909c6325d1abd7b`. Calibration CLI tests also verify distinct JSON/Markdown outputs
and reject source-database overwrite before any read or write. No live model call
was made, and the application's active v14 run was left untouched.

## V14 research, question audit and repeated review

This pass inspected the fixed bank's construction, typed answers, partial-credit
rubrics, execution comparator and integration checks. The baseline's 182 regression
tests passed. Eight existing instances were substantially revised, eight instances
were added, and every noninteractive JSON answer key is checked through the complete
evaluator and rubric pipeline. The static anchors and other established families
retain their existing questions; this is a targeted hardening pass, not a claim
that every question has been individually redesigned.

New primary sources inspected on 2026-10-03 deliberately go beyond BBEH, IFEval
and EvalPlus:

| Third-party source | Observation and local application |
|---|---|
| [MuSiQue repository](https://github.com/StonyBrookNLP/musique) and [evaluation code](https://github.com/StonyBrookNLP/musique/blob/main/evaluate_v1.0.py) | Answer quality, support quality and answer sufficiency are distinct checks. The revised retrieval tasks require evidence for every hop and distinguish missing, deleted and cyclic paths from resolved answers. All cases occur inside a question; these are not MuSiQue scores or its paired sufficiency metric. |
| [CRUXEval repository](https://github.com/facebookresearch/cruxeval) | Forward execution and inverse input prediction exercise different code reasoning. New original tasks require both, full-domain preimage counts and extremal inputs, and comparison with a one-line aliasing mutant. A separate heap interpreter and actual execution of trusted emitted Python agree over the entire finite input domain. |
| [ZebraLogic evaluator in ZeroEval](https://github.com/WildEval/ZeroEval/blob/main/src/evaluation/zebra_grid_eval.py) | Correct cells and completely solved puzzles are different metrics. New grid tasks require complete solution counts, marginal position domains and canonical witnesses after removing clues. Partial cell/domain achievement stays diagnostic; the full task must be correct to pass. We use one response per task, not its best-of-N selection. |

No third-party questions, dataset artifacts or evaluator code were copied. Earlier
research references below remain the provenance of earlier revisions.

| Changed family | Stronger question and grading | Independent oracle |
|---|---|---|
| Selective report: four revised instances | Unknown regime shared across all draws, three simultaneous selection conditions, posterior regime mass, future-tail probability and predictive covariance. Independence conditional on the regime does not imply independence after marginalizing it. Each reduced rational pair is one diagnostic quantity. | Weighted enumeration of all five draws and both regimes, without the generator's posterior shortcut |
| Revision multihop: four revised instances | Five queries each, including revision ties, obsolete records, tombstone suppression, absent keys and cycles. IDs, record types and revisions do not label relevance. Complete path, complete evidence, status and result have equal diagnostic weight per query. | SQLite window selection plus recursive traversal; reversal and obsolete-row insertion invariance |
| Grid ambiguity: four new instances | Three all-different groups with cross-group relational clues; solve the intact dossier and two clue-deletion scenarios. Count every assignment, identify every entity's domain and provide the first two lexicographic witnesses. Individual domains do not establish joint feasibility. | Position-to-entity backtracking with relation pruning, versus exhaustive entity-to-position enumeration |
| Alias preimages: four new instances | Negative modulo, shared lists versus copies, sequential mutation, forward results, inverse multiplicity and boundary witnesses, and all-domain distinction from a one-line mutant. Each instance includes attainable and unattainable targets. | Actual execution of the trusted function from the prompt on all 625 inputs, versus an explicit object-reference heap |

Grading and review corrections:

1. **Retrieval shortcuts.** Old `L0`/`L1` answer IDs and `kind=noise` distractors
   exposed the intended chain. The replacement uses uniform record grammars and
   multiple outcomes. A second review found revision numbers still marked relevance;
   the final generator updates existing selected records and gives distractors
   equal-revision conflicts too.
2. **Fragmentary answers earned misleading credit.** Opt-in `atomic_paths` group
   rational pairs, complete evidence paths, domains and witness bundles. Matching
   a denominator or part of a required proof earns no credit for that quantity.
   Unrelated quantities retain credit. Missing structures remain in the denominator;
   exact integer contracts, strict envelopes and full-pass requirements still apply.
   Authoring validation rejects duplicate, overlapping, scalar and noncanonical paths.
3. **Boolean map keys impersonated numbers.** The real code subprocess accepted
   `{True: 'x'}` against `{1: 'x'}` because Python equates those dictionary keys.
   Key comparison now distinguishes Boolean and numeric keys recursively through
   map values. Exact numeric key equivalence (`1` and `1.0`) remains supported.
4. **Atomic diagnostics needed their own adversarial review.** Descendant value
   errors initially risked being counted again outside their grouped requirement.
   They now contribute once. Document errors cannot be confused with a field named
   `root`. Tests cover missing parents, extra keys, wrong lengths, empty containers,
   type substitutions and unaffected neighboring fields.
5. **Integration drift and test resource leak.** Full-suite CLI/browser/report
   assertions now expect 315 questions. Queue-test SQLite context managers now close
   their connections after transaction exit, eliminating the observed resource warnings.

The suite is `rigorous-v14`, JSON evaluator version 8 and code evaluator version 6.
The generation revision of unchanged families is preserved. Changed prompts,
expected values and rubric grouping participate in fingerprints; historical and
new results cannot be paired across incompatible protocols. No live model endpoint
was called. Structural difficulty has increased; measured model separation still
requires fresh matched runs.

Review proceeds through baseline inspection, independent prompt-derived oracles,
single-field corruption and structural attacks, then complete regressions and
integration checks. New controls live in [selftest_reasoning.py](selftest_reasoning.py)
and run in CI.

Final v14 verification on 2026-10-04: all 315 questions passed strict authoring
validation with zero errors or warnings; all 193 regression tests, 120 evaluator
assertions and 1,133 challenge checks passed. Corruption controls rejected 14,145
altered answers, including 2,874 v14 answer mutations and 873 v14 same-value
integer-to-float substitutions. All 264 noninteractive JSON answer keys passed the
full evaluator/rubric pipeline. Independent checks also cover three additional
generation seeds. Queue integration passed without the earlier SQLite resource
warnings; Ruff and Git whitespace checks passed. Two complete assemblies produced
fingerprint `b11260da421820b5`. No known unresolved issue remains from these review
passes; these checks do not prove the absence of every possible defect.

## Question storage and one runtime suite

The former questions.yaml had 51,462 lines. It was parseable but made review,
answer-key maintenance and useful diffs difficult. Static questions now live in
[tests/questions](tests/questions), grouped by category with one question per
file. There are 171 files; the largest has 1,453 lines. Multiline prompts remain
literal text and short arrays remain compact. The initial split preserved the
v9 question fingerprint exactly; subsequent v10 changes deliberately revise it.

[quality_suite.py](app/benchmarking/quality_suite.py) still assembles one fixed bank: 171 static
questions, 116 generated questions, 20 interactive tasks and eight exact-length
context questions, totalling 315 across 30 categories. Splitting source files
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
| [BBEH tasks and evaluator](https://github.com/google-deepmind/bbeh) | Expand general reasoning through interacting constraints. New tasks distinguish adaptive from fixed decisions, conflict from view equivalence, and grounded proof from circular justification. Its evaluator normalizes textual answers; our typed contracts explicitly specify stricter envelopes and ordering. |

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
| Adaptive minimax decisions | Adversarial hidden world; identify its label, not its identity; canonical optimal adaptive tree versus cheapest fixed sensor subset | Budget-feasibility search and independent observation-partition checks |
| Network interdiction and repair | Directed cycles, all minimal cuts, worst failure ranking, route tie breakers, and one-edge restoration with other failures retained | Heap-based shortest paths versus exhaustive simple-path enumeration |
| Transaction consistency | Same-value blind writes, writer identity, all conflict/view serial orders, dirty reads, recoverability, cascadelessness and strictness | Resource-grouped conflict checks, explicit serial execution and commit-interval inspection |
| Grounded proof and evidence repair | Signed contradictory atoms, conjunctive rules, duplicate evidence alternatives, unseeded cycles and all minimum retractions | Minimal provenance antichain propagation and hitting-set repair versus subset/closure enumeration |

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

## V11 client and evaluation review

The additional public-source review inspected the current [IFEval strict and
loose evaluator](https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_lib.py),
[BBEH evaluator](https://github.com/google-deepmind/bbeh/blob/main/bbeh/evaluate.py),
[IFBench methods](https://github.com/allenai/IFBench),
[LiveBench ground-truth design](https://github.com/LiveBench/LiveBench),
[EvalPlus](https://github.com/evalplus/evalplus), and
[tau2 evaluation](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/evaluator/evaluator.py).
Prompt success and individual-requirement achievement remain separate. No dataset
questions were copied. Added difficulty is structural: optimality, completeness,
alternative semantics and evidence sufficiency must agree in a single answer.

The new 16 instances use four families and four fixed instances per family.
Graph variants include disconnecting and non-disconnecting attacks. Transaction
variants include serial histories, view-equivalent blind-write histories with a
conflict cycle, nonrecoverable histories, and strict histories without a serial
equivalent. Uniform guesses therefore fail. All requirements have exact typed
answers, top-level balanced weights and strict full-pass gates. New oracle tests
reject 942 individual corruptions as well as missing fields and Markdown fences.
All static JSON anchors now explicitly require a bare JSON document too.

Client findings and fixes:

1. **Read buffering distorted TTFT.** Requests' default line iterator requests
   512-byte buffers. A short event on a response with Content-Length could wait
   for later bytes. The client now uses available-byte `read1` reads up to 64 KiB,
   avoiding both read-to-fill delay and Python calls per byte. Older/custom
   transports have an explicit single-byte fallback. See the
   [Requests implementation](https://github.com/psf/requests/blob/main/src/requests/models.py)
   and [urllib3 response API](https://urllib3.readthedocs.io/en/stable/reference/urllib3.response.html).
2. **Closing at DONE could discard a warmed socket.** The client consumes the
   remaining HTTP framing before closing the response. Real local HTTP/1.1 tests
   confirm two chunked completions reuse the same TCP connection, and a gated
   Content-Length response delivers its first event before its body finishes.
3. **Malformed or interrupted streams could look successful.** SSE events now
   obey blank-line dispatch, multiline data, CR/LF/CRLF, UTF-8 fragments and BOM
   rules. Invalid JSON/UTF-8, unexpected choices and missing completion markers
   are endpoint failures. EOF after an explicit finish reason remains valid;
   EOF inside an event is invalid. See the
   [SSE specification](https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation).
4. **Reasoning could disappear from accounting.** An event containing both
   reasoning and final text now retains both for usage estimation while only
   final text is graded. Blocking estimates also include delivered reasoning.
   Unavailable or malformed telemetry becomes a labelled estimate; explicit zero
   remains a provider count. Partial streaming usage updates preserve previously
   reported fields. Interactive aggregates preserve estimation flags.
5. **Retry timing hid failed work.** Latency and TTFT now include prior attempts
   and backoff; successful-attempt latency is also persisted. Performance samples
   still make exactly one attempt. Every worker negotiates capabilities during
   untimed warmup, and negotiated settings are recorded per worker.
6. **The repetition cutoff changed the experiment.** A heuristic cannot prove
   that repeated structured output is invalid or that generation cannot recover.
   Canonical quality runs now disable early cutoff and use their full fixed
   budget. This can increase run time for looping models. Timeout and output
   limits still apply; their outcomes remain visible.
7. **Evaluator errors could earn scores or disappear from coverage.** Nonfinite
   or out-of-range scores, invalid/duplicate criterion diagnostics and invalid
   contract scores now produce unscored evaluator errors. Missing required rubric
   criteria retain zero credit in the diagnostic denominator. A pass criterion
   must earn its full possible credit.
8. **Reports omitted result-affecting client settings.** Quality protocols now
   record client/dependency versions, timeout, retry/backoff, streaming requests,
   disabled repetition detection and worker count. Differing protocols cannot
   produce a paired quality comparison. Performance is revision v4; quality is
   v11. Credentials and endpoint URLs are excluded from this protocol metadata.
9. **A field named root was confused with the document root.** Existing scoped
   state tasks could lose credit for correct reads when their root field was
   malformed. Root location is now explicit, unrelated fields retain credit,
   and contract criteria have separate unique IDs. JSON evaluator version is 5.
   Simultaneous structural defects therefore remain scored task failures instead
   of accidentally becoming duplicate-criterion evaluator errors.
10. **Malformed authoring fields could reach requests or reporting.** The loader
    rejects Boolean thresholds, non-text optional fields and non-text family IDs.
    JSON validation rejects impossible nonfinite answer keys and ambiguous
    criterion paths.
11. **Inferred output schemas revealed empty arrays.** Generated contracts no
    longer infer element types from a nonempty expected answer while describing
    an empty answer differently. Network cost fields have an explicit nullable
    contract for every instance. New family contracts are checked for equality
    across all four instances, so they do not reveal which attacks disconnect or
    which histories have empty serial-order lists.

No live model was called in this review. Transport optimizations were verified
with controlled HTTP servers, not inferred from model scores. First delivery
still reflects network, provider buffering, scheduling and local processing;
it is not a model-side prefill measurement. The stream deadline is checked on
received chunks and can overrun while a socket read waits for its idle timeout.
The client does not change token caps, temperature or seed after a rejection.
Unsupported decoding fields remain explicit endpoint failures. Provider seed
support and deterministic execution cannot be established from a request alone.
Successful-response usage does not include discarded failed attempts; estimates
cannot include reasoning that the provider neither delivers nor reports.

The final v11 verification commands are in the [README](README.md#verification)
and CI includes both [selftest_adversarial.py](selftest_adversarial.py) and
[selftest_client.py](selftest_client.py).

Final v11 validation: all 291 questions passed strict authoring validation with
zero errors and warnings; 158 regression tests passed; all 120 evaluator
assertions and 1,133 challenge checks passed. The corruption suites rejected
6,492 altered answers (5,550 existing and 942 new). Combined missing-field and
extra-key failures remain scored for every noninteractive JSON question. Queue
checks, Ruff and Git whitespace checks passed. No unresolved defect was found
in the completed review passes; these checks do not prove absence of all bugs.

## V12 research, question audit and repeated review

The review revisited [BBEH's task design](https://github.com/google-deepmind/bbeh),
[IFEval's strict and loose evaluator](https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_lib.py),
[IFBench's verifiable constraints](https://github.com/allenai/IFBench),
[EvalPlus's boundary-test approach](https://github.com/evalplus/evalplus),
[LiveBench's ground-truth methods](https://github.com/LiveBench/LiveBench), and
[tau2's state/action evaluation](https://github.com/sierra-research/tau2-bench).
The resulting local tasks are original. The design inference is to make models
satisfy interacting semantic requirements and completeness checks, while keeping
each requirement independently inspectable. Strict whole-task success remains
the headline; per-requirement achievement is a diagnostic with equal total weight
for each top-level requested field. This suite's output contracts differ from
public benchmarks' answer normalization, so their scores are not interchangeable.

The audit retained established reasoning, policy, translation and code anchors
with their independent oracle checks. It found three useful extensions: reasoning
about concurrent histories with failed operations, knowledge that changes after
public information, and constraints spanning an entire writing response. Formal
writing remains a separate compliance measure. Specialist fixtures still test
closed-world policies and candidate translations; their scope does not expand
merely because all fields are objectively graded.

| New or strengthened requirement | What must agree | Independent verification |
|---|---|---|
| Concurrent bounded FIFO histories, four instances | Real-time precedence, FIFO state, capacity, failed puts, empty takes, every valid order, every minimum deletion repair and canonical witnesses | Generator prunes topological/queue transitions; oracle enumerates full permutations, checks times separately and interprets queue results from emitted prompts |
| Nested public knowledge, four instances | Simultaneous announcement filtering, recomputed observations, nested knowledge, mutual versus common knowledge and canonical shortest counterexamples | Generator propagates truth sets and searches paths; oracle evaluates individual worlds with explicit relations and all-pairs min-plus distances |
| Three writing anchors | Distinct supplied percentages; global word uniqueness with prescribed starts/ends and word counts; no internal blank lines | Correct responses plus repetition, case/punctuation and blank-line near misses |

Each new family has seeds 19 and 23 and two variants. Queue pairs retain the same
IDs, times and values and change exactly one recorded result. One member is valid;
the other requires more than one deletion and has multiple minimum repairs.
Every knowledge instance has two informative announcements, a nested announcement,
different mutual/common-knowledge answers and a counterexample of at least two
observation links. Authoring deliberately selects these contrasts; these four
instances are not a random sample of all concurrent or epistemic problems.
Contracts are identical within families and do not reveal empty answer arrays.
The suite is now v12, while existing generated families retain the v11 generation
version so an evaluator change does not unnecessarily change their inputs.

Further grading defects found and fixed:

1. Global removal of reasoning tags rewrote valid code literals. Only complete
   leading reasoning envelopes are removed. An unfinished leading scratchpad has
   no established final answer. JSON strings, code and quoted tags remain data;
   orphan closing tags are no longer treated as an implicit reasoning channel.
2. Tolerant unordered comparison greedily consumed the first matching value.
   Because tolerance is not transitive, that could reject a valid permutation.
   A complete bipartite matching now finds a consistent assignment. Set members
   retain their internal sequence order unless the fixture explicitly allows it.
3. Arbitrary objects could impersonate an expected value using __repr__. Opaque
   objects now fail structural fixtures. Actual supported scalar/container values
   still cross the subprocess boundary with typed encodings.
4. A failing child process could supply plausible JSON on stdout. Nonzero exits
   always fail; SystemExit and other BaseException failures in submitted programs
   are captured as program failures. Launch failures are explicit unscored
   evaluator errors. The subprocess is still not an adversarial security boundary.
5. Complete one-line Python definitions were rejected by a two-line heuristic.
   Complete raw Python is now parsed before the best-effort prose extraction.
6. JSON nesting behavior differed across interpreters and recursion failures could
   become unscored evaluator errors. A portable 256-level limit, respecting quoted
   strings and escapes, makes excessive nesting a scored format failure.
7. Repeated copies of one supplied percentage counted as two statistics.
   Distinct-match checks implement the revised explicit contract. Internal blank
   lines have their own criterion, supplementing the position and line-count
   checks, and global word uniqueness is now verified across lines.
8. Forbidden Markdown fences failed grading but could be labelled task failures
   because outcome attribution reparsed with different fence settings. Attribution
   now uses the same settings and identifies these as format failures.

Client review and effect on results:

1. A substring test for stream matched upstream and disabled streaming after an
   unrelated permanent error. Negotiation now requires an explicit unsupported
   field or a structured error identifying that field. Unrelated field validation
   failures retain the endpoint's negotiated capabilities.
2. A measured one-attempt rejection could change the settings of later samples.
   Capability fallback now also requires remaining retry budget. Timed requests
   preserve the warmup settings; a backend rejection stays visible as a failure.
3. Completion data after a finish reason could overwrite length with stop, hiding
   truncation. Completion choices after termination now fail the request. Trailing
   usage events remain valid, and HTTP framing is still drained for socket reuse.
4. Non-string finish metadata and Boolean/float choice indices could reach grading;
   usage-only or DONE-only streams could count as successful performance requests.
   These now fail endpoint validation. Empty content on a genuine completion choice
   remains a quality missing-answer outcome.

Client revision is chat-client-v3; performance revision is performance-v5.
Changed evaluator versions are recorded in quality reports, preventing paired
comparison with a different grading protocol. The fixed temperature, seed and
token budget remain the experimental decoding policy. Connection reuse and
available-byte reads reduce client overhead and first-event buffering, following
the [urllib3 response API](https://urllib3.readthedocs.io/en/stable/reference/urllib3.response.html)
and [SSE framing specification](https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation).
They cannot remove provider buffering, network latency or shared-server contention.
Reported throughput covers delivered completions over full measured wall time;
it cannot infer model-side prefill/decode speed. A single run also cannot establish
that quality worker concurrency has no effect on an endpoint. Timeout checks still
occur between received chunks, so an idle read can overrun the stream deadline.
No paid or live model endpoint was called during this review.

Review proceeded through task/contract checks, client and grader regressions,
independent oracles and corruption controls, then full CLI/web/report integration.
Final verification: 299 questions passed strict validation with zero errors or
warnings; 173 regression tests passed; all 120 evaluator assertions and 1,133
challenge checks passed. The corruption suites rejected 7,486 altered answers,
including 994 for the new families. Queue checks, Ruff and Git whitespace checks
passed. After the final metadata/fence-attribution corrections, all 55 affected
regressions passed. CI includes the independent v12 oracles. No unresolved
correctness defect was found in the completed review passes.
Fresh matched model runs are needed to measure difficulty, ranking changes and
repeatability. Passing these checks cannot prove the absence of every defect.

## V13 research, question audit and repeated critical review

The current pass researched additional public third-party benchmarks, rather than
relying on BBEH, IFEval or EvalPlus again. Primary sources and the decisions taken:

| Source inspected | Evaluation insight | Local decision |
|---|---|---|
| [FEVER dataset and annotation format](https://fever.ai/dataset/fever.html) | Supported, refuted and insufficient-information claims have different meanings; evidence can have alternative sufficient sets. | Require all inclusion-minimal consistent supporting/refuting sets. Separate unknown from an inconsistent dossier and never grant evidence credit through vacuous entailment. |
| [HoVer task and scoring description](https://hover-nlp.github.io/) | Joint claim/evidence success requires evidence from the necessary documents; retrieval dumping does not establish correct reasoning. | Use multihop implications, alternative routes and distractors. Exhaustive minimality, complete set lists and concrete countermodels reject incomplete proofs and source dumping. |
| [BFCL task taxonomy and executable evaluation](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html) | Function selection, argument structure, relevance and execution require distinct checks. | Add strict typed calls, nonexistent tools, failed references, error precedence, retry caches and persistent effects. Validate judgments and state independently; no real API is invoked. |
| [ComplexBench scoring dependencies](https://github.com/thu-coai/ComplexBench) | Composed instructions have dependency relations between evaluation points. | Retain the existing transitive dependency gates. Receipt version/balance credit now requires the corresponding correct result status; a wrong outcome cannot earn credit from meaningless receipt fields. |
| [BigCodeBench task and execution protocol](https://github.com/bigcode-project/bigcodebench) | Practical composed programs should be graded by execution; reproducibility depends on the recorded execution/generation setup. | Strengthen existing code fixtures across integer magnitude, arbitrary identifiers and nested JSON payloads, checked by two algorithms and real subprocess execution. This does not claim BigCodeBench's library or branch coverage. |
| [MuSR paper](https://arxiv.org/abs/2310.16049) | Narrative reasoning can combine natural language with structured reasoning instances. | Considered for future narrative tasks. This pass retains explicitly specified finite semantics; it does not equate logical evidence exercises with MuSR's soft narrative reasoning or invent a judge for ambiguous prose. |

No third-party questions were copied and no third-party score is claimed. The
research produced eight new original instances: two families, seeds 19/23 and
two variants. The canonical bank has 307 questions across the same 30 categories.

The whole-bank audit covered prompt/answer consistency, scalar and container
types, scoring relaxations, rubric coverage, answer derivations and plausible
near misses. Every noninteractive JSON answer leaf has a reachable rubric
criterion. The only scalar aliases are the already explicit numeric/v-prefixed
version alternatives in TU2-01; no new permissive grading was introduced.
The suite continues to require full success on every mandatory condition.

| Question group reviewed | Difficulty/evaluation assessment and action |
|---|---|
| Logic, arithmetic, scientific and financial exercises | Retain bounded exact derivations, completeness and optimization requirements. Existing separate arithmetic/search oracles and answer corruptions are rerun. More prompt length alone would not establish harder reasoning. |
| Truthfulness and evidence | Add minimal supports/refutations, all contradiction cores, all minimum deletion repairs and repaired-world witnesses. Distinguish absent evidence, contradictory evidence and entailment. Deduplicate repaired worlds; counts are not probabilities or votes over repairs. |
| Reading, retrieval, summarization, knowledge and long context | Retain revision precedence, bitemporal joins, indirect selection and distributed evidence. Existing prompt-derived context checks confirm exact lengths and answers. No extra filler was added as a proxy for difficulty. |
| Advanced coding, generation and review | Add six transformed fixtures to each of the eight advanced-coding anchors, 48 total. Tests expose float rounding of timestamps/amounts, assumptions about single-character IDs and loss of nested/falsey payloads. Keep boundary/generated weights balanced and every fixture mandatory. |
| Tool and agent tasks | Add 22-call traces with strict integer types, failed and future references, error ordering, stale versions, failed-key reuse, exact cached receipts and replay counts. Existing live text-protocol simulations retain final-state and authorization checks. |
| Specialist policies, security, translation and ethical tasks | Retain closed-world contracts and multiple interacting obligations/conditions; rerun independent arithmetic/state and semantic candidate checks. These tasks do not establish professional advice or unconstrained translation fluency. |
| Instruction following and writing | Retain explicit compositional stages and formal writing criteria; fix contradictory criterion diagnostics. Writing remains a separate compliance measure and is not treated as artistic quality. |

Within each seed, the evidence pair changes one assertion and the tool pair one
amount argument. Evidence variants contrast a consistent dossier with one that
requires a repair; the inconsistent version admits multiple minimum repairs.
Tool variants contrast an exact retry with an idempotency conflict while leaving
the eventual account state unchanged, so correct final balances alone cannot
pass the task. Output contracts are identical within each family and do not
reveal empty evidence lists or the existence of witnesses.

Evidence partial achievement balances each claim's judgments, proof/refutation
sets and witnesses separately, regardless of how many sources a proof contains.
Claim IDs are mandatory envelope checks and do not create free content points.
Tool results receive equal diagnostic credit per call, irrespective of whether
the result is a short error or a longer receipt. Balance/state, cache contents
and actual commit counts remain separate requirements. Failed status dependencies
block receipt-field credit and remain in the achievement denominator.

Two preexisting grader defects were reproduced and corrected:

1. **Format diagnostics parsed display prose.** A failing constraint whose label
   or regex description contained `: PASS)` could emit a passed criterion, even
   while its score was zero. Diagnostics now use the actual Boolean check result;
   labels cannot alter criterion status, credit or contract score.
2. **Huge numeric outputs could escape scoring.** An otherwise valid Python
   integer larger than binary64 compared with a float fixture raised an overflow
   and became an unscored evaluator error. Overflow comparisons now use exact
   rational arithmetic. Ordinary wrong outputs remain scored failures; integer
   counts retain exact comparison and explicit numeric tolerances still apply.

The final contract review also found that the generic JSON number comparison
would accept `1.0` where the new typed tasks explicitly require an integer
literal. JSON evaluation now supports opt-in `integer_paths`, validated against
exact integer answer leaves. Those paths cannot be ignored or given non-integer
aliases. The new count/receipt tasks use these contracts; other tasks retain
their stated numeric-equivalence policy. All 108 numerically equal float
substitutions in the actual new answers are scored type failures, and invalid
authoring paths/aliases are rejected.

The suite revision is rigorous-v13; code/format evaluator versions are both 5
and JSON evaluator version is 7.
Other evaluator versions, client and performance protocols are unchanged by this
pass. Existing generated prompts retain their generation version; changed code
fixtures, new questions and rubrics are included in fingerprints. Historical
results remain historical and fresh results cannot be paired across incompatible
protocols.

Review iterations:

1. Establish the existing baseline, inspect all answer/criterion contracts and
   research the primary sources. The baseline's 173 regression tests passed.
2. Derive new answers from the emitted prompts with independent set-based truth
   tables and a SQLite tool interpreter. Check consistent/inconsistent contrasts,
   every minimal support and repair, error precedence and cache replay behavior.
3. Reject every single-leaf mutation, omission, extra fact and structural near
   miss in the new answers; verify exact family contracts and diagnostic weights.
   Regression checks cover misleading prose labels and enormous numeric outputs.
4. Recheck code transformations with both established algorithms, run reference
   programs in real subprocesses, then rerun authoring validation, the complete
   regression/corruption suites, queue integration, lint and whitespace checks.
5. Inspect the final typed contracts, add exact integer syntax checks and reject
   same-value float substitutions. Rerun every affected evaluator and integration
   regression after that correction.

The executable v13 checks are [selftest_evidence.py](selftest_evidence.py), also
included in CI. No live or paid model endpoint is called by these checks. Fresh
matched model runs are still required to establish empirical difficulty, model
separation and reliability; structural difficulty and grader validation are not
a measured model-score improvement.

Final v13 verification: all 307 questions passed strict validation with zero
errors or warnings; 182 regression tests, 120 evaluator assertions and 1,133
challenge checks passed. Corruption controls rejected 10,398 altered answers,
including 2,804 new answer mutations and 108 numerically equal float substitutions.
Controlled float-rounding shortcuts pass the old interval/version examples but
fail the expanded canonical fixtures. Queue integration, Ruff and Git whitespace
checks passed. All 256 noninteractive JSON answer keys passed the complete scoring
and rubric contracts. Two independent suite assemblies produced fingerprint
`24cfdc40a4414da6`. No known unresolved correctness issue remains from these
review passes; these checks do not prove absence of every possible defect.

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
[README](README.md#verification). The independent task oracles include
[selftest_frontier.py](selftest_frontier.py),
[selftest_adversarial.py](selftest_adversarial.py),
[selftest_compositional.py](selftest_compositional.py) and
[selftest_evidence.py](selftest_evidence.py).


## 2026-10-04 application consolidation and deadline investigation

Read-only inspection of the local database found run #67 (Qwen3.8-Flash-Next-FP8, rigorous-v11) recorded 291 outcomes and 19 request failures. All 19 were stream deadlines: Advanced Coding 5/16, Code Generation 5/8, Terminal Algorithms 4/4, Terminal Debugging 3/4, Finances 1/12 and Mathematical Reasoning 1/20. All four Terminal Algorithms tasks belong to Q9-dependency-regression. Their total request latency was 1,045.9–1,091.2 seconds. The three-attempt policy repeated a 360-second stream budget, with backoff; a socket inactivity timeout does not limit total duration when data keeps arriving. Successful coding requests also exceeded six minutes across attempts. The saved data discarded partial stream progress and cannot distinguish continued reasoning, repetition or a stalled stream. No live endpoint was invoked and historical outcomes were not changed.

Client v4 gives quality calls an independent 1,800-second stream budget while keeping the 180-second inactivity timeout. It no longer retries total generation deadlines or invalid stream protocols. It retains progress counters and timing on failed attempts without treating partial output as a completed answer. Total deadlines are observed at receive boundaries and EOF; blocked reads can overrun by the inactivity timeout. The fixed performance budget remains 360 seconds with one measured attempt. Changing the client protocol prevents strict paired comparisons from silently mixing old and new generation policies.

Performance v6 adds request samples, p99 presentation, request/error-rate graphs, output-length distributions, chunk-gap summaries and a mean output-token-time delivery proxy. Reported completion counts may include reasoning, and one SSE event can deliver many tokens. Consequently the proxy excludes estimates and buffered bursts; chunk gaps are never called inter-token latency. Definitions are informed by [NVIDIA GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2600/user-guide/docs/perf_benchmark/genai-perf-README.html). It remains a closed-loop load test with successful output over full wall time including failures. Quality-request distributions and category deadline counts are separate from controlled load results. Old records never acquire invented stream diagnostics.

The CLI benchmark and management entry points, console/Markdown formatting and generator command entry points were removed. Runtime benchmark code is under `app/benchmarking`; verification scripts stay at the root. The task bank is unchanged: 315 tasks, hash `b11260da421820b5`, verified against the pre-change Git version. App configuration loads `.env` with process environment precedence. The UI retains the existing stack and adopts compact neutral surfaces, borders and token colors inspired by [Shadcn theming](https://ui.shadcn.com/docs/theming), with mobile navigation and run sections. Review corrected missing container closure, model-ID column collisions, desktop toggle visibility, empty-report notices and percentile graph scaling. Regression checks cover deadlines without retries, retained partial progress, invalid budgets, EOF deadline enforcement, token-time exclusions, wall-clock failure denominators, raw sample distributions and authenticated performance downloads.

## 2026-10-04 bounded finalization and configurable decoding

Run #80 used four workers. Summed quality request duration was 29,185 seconds over 7,447 seconds elapsed, indicating substantial overlap. All 24 output-limit failures exhausted 65,536 completion tokens while returning reasoning without a final answer. Several saved endings repeat calculations or irrelevant edge cases. These observations motivate bounded finalization; they do not establish that another request will solve the tasks. The historical run is preserved and no live recovery requests were issued.

[Pi](https://github.com/earendil-works/pi) was evaluated as a possible harness. Its [SDK](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/sdk.md) offers agent sessions, tools, context management and configurable resources, with a Node.js/Bun runtime. It is appropriate to consider for future repository/environment tasks. Replacing the fixed-task runner would additionally require controlled instructions, tool isolation, retry/compaction limits, usage accounting and an evaluator adapter. Those boundaries already exist in the current runner, so this change keeps it and implements bounded finalization directly. This is an execution protocol, not an unrestricted coding agent.

`bounded-quality-v1` allocates 57,344 tokens to the initial ordinary-question call and reserves 8,192 for one finalization call. Allocations share the existing 65,536-token ceiling. Smaller budgets reserve `min(8192, budget // 8)`; fewer than eight tokens permit no followup. The followup trigger is output-length termination or a nonempty reasoning-only response ending normally. Empty, refused, failed and completed wrong responses receive no additional attempt. The conversation includes the original messages and full returned response, followed by a fixed instruction requesting a complete replacement answer. Only that last response is evaluated; no hidden fixtures, expected values or evaluator feedback enter model context. Finalization uses a normal chat turn and does not promise a provider-native resumption or that internal thinking is disabled. Preserving long history requires sufficient endpoint context capacity, and all repeated input tokens are included in cost.

The protocol records allocations, per-call metrics/usage, recovery outputs, initial outcomes, initial passes and recovered passes. Latency includes both calls; their combined streams are not treated as one continuous token-delivery span. Followups execute within the existing question worker. Compatibility negotiation can explicitly remove rejected streaming fields before generation, with at most three attempts per call. Quality transport failures are not regenerated, preventing unknown partial output from being multiplied by retries. Existing interactive tasks retain their action and total-output budgets and do not gain evaluator-guided repair.

Models now configure finite temperature in [0, 2] and an optional standard `reasoning_effort`. The empty effort setting omits the field. Explicit unsupported settings fail visibly. Both quality and performance honor the selected settings; submission snapshots them so later edits do not alter queued runs. Existing models migrate to temperature 0 and provider-default effort. [Qwen's model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/README.md) recommends different sampling for thinking mode; matching those settings is a user configuration choice, not an assumed measured improvement. Client v5 and performance v7 record the settings, and quality reports additionally record the execution protocol. Strict comparisons reject differing protocols/settings. The question bank and fingerprints are unchanged.

Offline coverage in `selftest_protocol.py` exercises real client payloads and SSE parsing with controlled responses, complete replacement semantics, exhausted budgets, reasoning-only recovery, missing final answers, cancellation, endpoint failures, usage/timing aggregation, parallel worker limits, schema migration, create/edit validation, provider rejection, performance settings, corrupt queue entries and queued-setting snapshots. These checks do not establish live model recovery rates or latency improvements.

## 2026-10-05 native tool conformance and repeated runs

The full design is in [docs/design-tool-conformance-and-repeats.md](docs/design-tool-conformance-and-repeats.md). Four decisions govern it. The native suite is a separate suite (`tool-conformance-v1`) with its own fingerprint, provenance and execution protocol (`native-tools-v1`); it is not added to the rigorous suite, so the rigorous inventory (323 tasks, hash `7909c6325d1abd7b`), its evaluator versions and its protocol are unchanged and verified against the pre-change Git version. Repeats vary the model seed (`seed = repeat index`); the seed is recorded but removed from the comparison identity, while every other protocol field still blocks pairing. A deployment that rejects a request feature produces a scored `feature_rejected` failure, reported separately from outages. Tool schemas are validated with an internal JSON-schema subset, and suite validation rejects any keyword outside it so the grader never ignores a constraint shown to the model.

The native client (`tool-client-v1`) accumulates streamed tool-call deltas without raising on malformed framing; defects such as changed or duplicate IDs, non-string arguments, names changing mid-stream or deltas after the finish reason are recorded as evidence and graded as wire criteria. Synthetic IDs replace missing ones in echoed history and are recorded. The text-path client protocol (`chat-client-v6`) is unchanged.

The five agentic simulations are re-hosted on native tools by reusing the same `Environment`; prompts drop the text-protocol tool listing and `done=true` instruction. `selftest_tool_conformance.py` drives the same action policies through both protocols and requires identical final state, violations, call counts and verdicts, for correct and deliberately flawed policies. An oracle client passes all 96 tasks with full contract scores; each targeted corruption (unparseable or schema-violating arguments, unknown tool, wrong value, leaked markup, framing and ID defects, ignored `tool_choice`, parallel calls when disabled, split parallel calls, ungrounded answers, fenced or mistyped JSON, feature rejection) fails its specific criterion.

Repeat statistics (`repeat-stats-v1`) exclude members that are incomplete or differ in suite or protocol, with stated reasons. They report run spread with a Student t interval (three or more runs), pooled task-mean achievement, outcome stability, unstable tasks, and pass^k / pass@k with unbiased combinatorial estimators balanced by family and category. Paired comparisons of groups or single runs use a hierarchical bootstrap over families and repeats (identical to the historical bootstrap for single runs) and a category-weighted family-level sign-flip randomization test; single runs add exact McNemar on full passes, and multiple comparisons shown together are Holm-adjusted. These tests are offline: they establish estimator and grader correctness, not live model reliability or parser behaviour on any particular deployment.

## 2026-10-05 use-case suites

The full design is in [docs/design-usecase-suites.md](docs/design-usecase-suites.md). Application teams upload YAML suites through an admin page; they are stored in the database (`usecase_suites`), and each version is immutable and identified by its question fingerprint. Validation reuses the built-in suite checks, now in `app/benchmarking/suite_checks.py`, with `validate_suite.py` as a thin command-line wrapper. Only deterministic evaluators are allowed, so no uploaded content is ever executed. `json_match` questions with a `response_format` are converted to the native structured-output evaluator, and their expected values are checked against the declared schema subset.

Runs pin `usecase:<slug>@<version>`. The runner re-parses the stored document and refuses to run if the fingerprint no longer matches the stored one. Reports use the suite name `usecase:<slug>` with protocol revision `usecase-v1` and the unchanged `bounded-quality-v1` execution protocol, so versions pair on identical question fingerprints and never pair with built-in suites. The rigorous suite (323 tasks, hash `7909c6325d1abd7b`), evaluator versions and protocols are unchanged. `selftest_usecase_suites.py` covers parsing and rejection, versioning, check-only uploads, archiving, admin-only access, run pinning, reruns, tampering detection, end-to-end reports with a native structured question, and cross-version pairing.

## 2026-10-05 open-loop load tests

The full design is in [docs/design-open-loop-load.md](docs/design-open-loop-load.md). The new `load` run mode (`open-loop-v1`, performance report schema 5, kind `open_loop`) complements the closed-loop modes, which are unchanged. Arrivals follow a seeded Poisson or gamma schedule derived from the seed, step index, rate and duration, so runs with equal settings receive identical traffic. Workloads are weighted request classes with input/output length histograms, a shared system prefix and per-class SLOs. Prompts have exact cl100k_base reference lengths, and output length is enforced through `max_tokens` on a counting prompt.

The dispatcher never waits for completions. Arrivals beyond the in-flight cap are dropped and scored as misses. Dispatch lag is added to user-visible TTFT and latency to avoid coordinated omission. A step passes only when attainment meets the target overall and per class and the client kept up. A failing step bounds capacity only if dispatch kept up and the requests actually sent still missed the target; otherwise the sustainable rate is reported as a lower bound. Drift needs at least 20 arrivals in each compared third, and a latency rise counts only once it uses half of the tightest first-token SLO. The capacity revision is now `serving-capacity-v3`, with an `arrival` block for open-loop runs; closed-loop scenarios are unchanged. `selftest_load_test.py` covers validation, schedule statistics, exact prompt lengths, open-loop dispatch against a slow fake server, cap drops, SLO rules including lag, pass and conclusiveness rules, ladder stopping, cancellation, drift, sample thinning, submission round trips, runner persistence with URL sanitisation, run/offline/capacity pages and comparison compatibility. These tests use fake servers; they establish the protocol and accounting, not the capacity of any real deployment.

## 2026-10-05 open-ended questions, blind A/B studies and LLM judges

The full design is in [docs/design-ab-studies.md](docs/design-ab-studies.md).

**Open-ended questions.** `open_ended` is a question kind, not a scoring evaluator. The name is registered so loaders and validators accept it, but it is deliberately absent from `EVALUATOR_VERSIONS`, so no protocol identity changes. Execution is the unchanged `bounded-quality-v1` protocol. A completed answer gets the outcome `recorded` and the scope `open_ended`. `Result.is_scored` is false for every open-ended question, including truncated or empty answers, so achievement, passes, missing-outcome bounds and paired comparisons never include it. Summaries add an `open_ended` block only when such questions exist. A run with only open-ended questions completes normally.

**Built-in suite.** `assistant-open-v1` has 30 tasks, hash `0a7491b1367eba4c`. The rigorous suite (323 tasks, hash `7909c6325d1abd7b`) is unchanged.

**Studies.** A study pairs two completed runs of one suite name by fingerprint, so versions of a use-case suite pair as well. Excluded from pairs:
- answers lost to endpoint errors, cancellation or missing final answers
- multi-turn tasks

Truncated answers stay in, because users would see them. More than 1,000 pairs are reduced by a deterministic hash sample. Studies copy everything they display, and votes of deleted users are kept with a null user.

**Judge (`pairwise-judge-v1`).**
- Each answer is limited to 24,000 characters, with a truncation note to the judge.
- Output is strict JSON with a `winner` of "1", "2" or "tie", parsed after reasoning removal.
- Both orders are judged, and the mean of A=1 / tie=½ / B=0 decides.
- After eight consecutive judge errors the judge stops.
- Saved judgments make a resumed judge redo only missing or failed pairs.
- Judges still running at startup are marked *interrupted*.

**Statistics (`ab-stats-v1`).**
- The preference for B averages multiple human votes on a pair first.
- Categories weigh equally and families form clusters.
- The interval is a 2,000-draw percentile bootstrap with a fixed seed; the test is an exact two-sided sign test on decisive pairs.
- A preference needs the interval to exclude ½ and p < 0.05.
- Judge diagnostics are position consistency and the longer answer's share of decisive wins.
- Agreement with people uses the majority human label per pair (both-bad and splits count as ties) and Cohen's κ over {A, B, tie}.

`selftest_ab_studies.py` covers:
- **Open-ended handling:** outcomes and summaries, use-case validation, and the built-in suite hash.
- **Runs and studies:** open-ended-only runs through the real runner, pairing, the cap and creation rules.
- **Voting:** blind voting with fewest-votes assignment, side mapping, finality, closed studies and anonymised deleted users.
- **Judging:** prompts, parsing and combination, background judging with errors, resume, cancel, the consecutive-error stop and startup interruption.
- **Statistics:** checked against hand-computed values.
- **Pages and access:** results, voting and pair pages, export, the compare-page entry point, admin-only actions and login redirects.

As with the other suites, these tests use fake models; they validate the procedure, not the preferences of any real users or judges.

## 2026-10-05 deployment fingerprints and canaries

The full design is in [docs/design-deployment-monitoring.md](docs/design-deployment-monitoring.md).

**Snapshots.** Every run records a snapshot of its endpoint (`deployment-probe-v1`) before the first measured request. This applies to all run modes. The snapshot's hard fields are:
- the sanitised endpoint and configured model;
- the matching `/models` entry (without its volatile `created` field);
- `/version`, the `Server` header, the responding model and `system_fingerprint`;
- the prompt-token counts of five fixed probes (plain, system, multilingual, multi-turn, tools), or the HTTP status that rejected them.

A short temperature-0 behaviour probe is stored for display but excluded from the fingerprint, because batching makes such output nondeterministic. The fingerprint is a 16-hex SHA-256 of the canonical hard fields. Each successful snapshot is diffed against the model's previous successful one. A change of only the configured endpoint or model is a configuration change (info); anything else is a deployment change (alert). A failed snapshot never fails a run and is never compared.

**Canaries (`canary-v1`).** Canaries are ordinary pending quality runs created by the queue tick. A canary is skipped while its previous run is queued or running, and its next time counts from the current time. Evaluation runs from the queue's finish callback, and the tick also evaluates runs that missed it, for example after a restart:
- **Baseline:** the first completed run becomes the baseline automatically.
- **Quality:** comparisons use the existing `compare_groups` (family bootstrap, sign-flip test, McNemar on full passes). A significant decrease is an alert; a significant increase is info; an incomparable pair is a warning.
- **Latency:** a median TTFT above 1.5× the baseline and at least 250 ms slower, or a median latency above 1.5× and at least 1 s slower, is a warning.
- **Failures:** endpoint errors are a warning. Failed runs and runs with nothing scored are alerts; runs stopped by an administrator are not.

Alert and warning events are posted once to an optional webhook that passes the endpoint SSRF guard. Delivery failures are recorded on the event. Retention keeps a canary's newest 100 runs, its baseline and every warning or alert run.

The rigorous suite (323 tasks, hash `7909c6325d1abd7b`), evaluator versions and all report protocols are unchanged. `selftest_monitoring.py` covers:
- **Probes:** contents, fingerprint stability despite volatile fields and behaviour text, diffs, summaries and classification, and unreachable or rejecting endpoints.
- **Snapshots:** change events and timelines; the runner storing a snapshot and surviving probe failures.
- **Canaries:** validation, scheduling without backlog, pause and run-now; every evaluation outcome, including webhook delivery and failure; pending evaluation, retention and baseline rules; the queue hooks.
- **Pages:** monitoring, canary, model and run pages, the dashboard banner, acknowledgement and access rules.

These tests use fake endpoints. They establish detection and bookkeeping, not the stability of any real deployment.

## 2026-10-05 scorecards and decision profiles

The full design is in [docs/design-decision-dashboard.md](docs/design-decision-dashboard.md).

**Evidence (`scorecard-v1`).** Evidence is computed on request from completed runs and is never stored. For each model the scorecard takes the latest completed run per evidence key:
- quality per suite key;
- closed-loop latency;
- context sweep;
- open-loop capacity per workload (the preset, or `custom:<workload hash>`).

The scorecard never picks the best of several runs. A repeat group is one piece of evidence: the mean of its usable runs, with the run-spread t interval (or the pooled task bootstrap when only two runs are usable). Failed, stopped and unscored runs are excluded, and so is the open-ended suite. Canary runs count.

**Freshness:**
- *current* if the run's snapshot fingerprint equals the model's latest successful fingerprint;
- *stale* if it differs, or, for runs without a snapshot, if a deployment change was recorded after the run started;
- *unverified* otherwise.

**Ties.** Leaders and ties use the existing `compare_groups` from the leader to every other non-stale model in a column, Holm-adjusted together. A tie means no detectable difference; incompatible protocols are labelled not comparable instead of being ranked. Comparisons are cached by run ids and computed per column after the page loads, because a group comparison takes seconds.

**Gates (`decision-gates-v1`).** Gates are a closed list: quality (suite, optional category, achievement or full pass), capacity, latency, context and operations.
- **Uncertain flag:** a quality result whose 95% interval contains the threshold is flagged uncertain. So is a capacity that is only a lower bound below the threshold. The point estimate still decides.
- **Stale evidence** gives the status *stale* together with its would-be outcome.
- **Context** uses the measured sweep context unless it is stale; otherwise it uses the context limit declared in the deployment snapshot.
- **Verdicts:** *does not meet* when any gate fails; otherwise *incomplete* when any gate is missing or stale; otherwise *meets*.

**Decision records** are append-only. Each stores the evaluation, the gate texts, the gate revision and the deployment fingerprint at the time of the decision, so later runs never rewrite it. A record is shown as drifted when the model's current fingerprint differs.

No report protocol, evaluator version or suite changed; the rigorous suite stays at 323 tasks and hash `7909c6325d1abd7b`. `selftest_scorecard.py` covers:
- **Gates:** validation, every gate's pass, fail, missing and stale paths, and the verdict rules.
- **Evidence:** latest-not-best selection and exclusions, repeat groups, every freshness rule and age, closed-loop, open-loop, sweep and declared context, and operations.
- **Ties:** the leader, ties, the Holm-adjusted below mark, incompatible protocols and a stale leader.
- **Profiles and decisions:** profile validation and evaluation, decision snapshots surviving later runs, drift, superseding and cascade deletion.
- **Pages:** pages, JSON exports, the dashboard card and access rules.
