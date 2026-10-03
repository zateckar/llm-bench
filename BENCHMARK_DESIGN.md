# Rigorous v13 design and critical review

Reviewed on 2026-10-03. This document describes the current protocol.
Historical scores are retained and are not relabelled as v13 results. The v10/v11/v12
review records below describe earlier passes; the v13 review records the current changes.

## Question storage and one runtime suite

The former questions.yaml had 51,462 lines. It was parseable but made review,
answer-key maintenance and useful diffs difficult. Static questions now live in
[tests/questions](tests/questions), grouped by category with one question per
file. There are 171 files; the largest has 1,453 lines. Multiline prompts remain
literal text and short arrays remain compact. The initial split preserved the
v9 question fingerprint exactly; subsequent v10 changes deliberately revise it.

[quality_suite.py](quality_suite.py) still assembles one fixed bank: 171 static
questions, 108 generated questions, 20 interactive tasks and eight exact-length
context questions, totalling 307 across 30 categories. Splitting source files
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
