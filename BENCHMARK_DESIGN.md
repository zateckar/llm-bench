# Rigorous v11 design and critical review

Reviewed on 2026-10-03. This document describes the current protocol.
Historical scores are retained and are not relabelled as v11 results.

## Question storage and one runtime suite

The former questions.yaml had 51,462 lines. It was parseable but made review,
answer-key maintenance and useful diffs difficult. Static questions now live in
[tests/questions](tests/questions), grouped by category with one question per
file. There are 171 files; the largest has 1,453 lines. Multiline prompts remain
literal text and short arrays remain compact. The initial split preserved the
v9 question fingerprint exactly; subsequent v10 changes deliberately revise it.

[quality_suite.py](quality_suite.py) still assembles one fixed bank: 171 static
questions, 92 generated questions, 20 interactive tasks and eight exact-length
context questions, totalling 291 across 30 categories. Splitting source files
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
[selftest_frontier.py](selftest_frontier.py) and
[selftest_adversarial.py](selftest_adversarial.py).
