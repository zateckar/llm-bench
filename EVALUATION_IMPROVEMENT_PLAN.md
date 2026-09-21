# Evaluation improvement plan

Prepared 2026-09-21 from the current source and a read-only inspection of `data/bench.db`.

**Objective:** establish whether recent grades are trustworthy, explain performance at the level of individual requirements, and raise the suite's difficulty across every category while preserving reproducible comparisons.

This document records the implementation and the operational gates that remain
endpoint-dependent. The repository now includes the read-only v5/v6 audit,
criterion-level evaluation, report/UI support, empirical calibration, a fully
validated hardening-v7 candidate profile, and a release gate. Historical rows
remain unchanged and no benchmark YAML item was added to the default suite, so
the frozen v6 suite hash and historical denominators remain comparable.

Implementation status: engineering work for all three workstreams is complete.
The hardening bank has 58 independently derived families and 116 development
cases, with simulator-backed interactive families, adversarial evaluator tests,
profile manifests, and CI checks. A live model pilot was not run because this
workspace has no authorized endpoint; `hardening_release.py` records that state
as `pending_model_pilot` and becomes the selection/release check once matched
reports are supplied. Manual historical answer disposition is recorded as a
review queue rather than being inferred from agreement among model outputs.

Generated local artifacts:

- `data/quality-audit-2026-09-21.json` — runs 51–58, with replay diagnostics for
  reconstructable saved answers and no consistency mismatches in the cohort.
- `data/difficulty-calibration-2026-09-21.json` — runs 54–58; 287 capability
  questions were fully covered by all five reports. It flags saturated families
  and categories below the five-family coverage target for the next authoring
  pass.
- `data/difficulty-hardening-backlog-2026-09-21.json` — 29 categories and 58
  candidate family specifications, prioritized from that calibration. These
  are the source descriptors for the executable hardening-v7 bank and remain
  separate from the default suite.
- `data/hardening-release-gate-2026-09-21.json` — hardening manifest, oracle
  verification, and a `pending_model_pilot` release result for the current
  endpoint-free workspace.

Validation completed: strict suite validation, evaluator, granular-evaluation,
calibration, hardening-bank, hardening-release, quality, reliability,
specialist, ceiling, stretch, report, and performance self-tests, plus Python
compilation and Ruff. No live model calls were made by the implementation work.

## 1. Evidence that determines the priorities

The previous written review covers runs 48–50. Runs **51–53** now form a separate completed v5 cohort, and runs **54–58** form the latest completed v6 cohort. Audit both, with 54–58 the primary calibration baseline.

Runs 54–58 share suite `921ffa63e20a612e`, quality-v6, temperature 0, an unset model seed, four workers, and a 65,536-token quality output cap. Each has 434 questions. They use development seed 1729, one generated variant, and no optional context-quality sizes.

| Run | Configured alias | Saved balanced capability | Ceiling-v5 subset | Stretch-v6 subset | Endpoint errors | Repetition stops |
|---|---|---:|---:|---:|---:|---:|
| 54 | Qwen3.8-27B.long | 90.58% | 84.62% | 58.33% | 11 | 16 |
| 55 | Qwen3.8-27B-FP8.long | 95.39% | 98.08% | 70.00% | 9 | 14 |
| 56 | Qwen3.8-27B-NVFP4.long | 88.87% | 80.77% | 56.67% | 11 | 33 |
| 57 | skoda-llm-primary-test | 95.07% | 94.23% | 90.00% | 4 | 4 |
| 58 | gemma4-31 | 63.13% | 11.54% | 6.67% | 0 | 2 |

These are saved scores, not independently validated judgments. Aliases do not verify model identity. There are **35 endpoint errors, 69 repetition stops, and no recorded truncations** in this cohort. Capability denominators differ: 292, 293, 295, 295, and 299 respectively.

Specific implications:

- Of 299 shared capability questions, 287 were scored in every run. **143/287** passed in all five. Across runs 54–57, **227/287** passed in all four. Report both populations; the lower-scoring model should not hide ceilings among stronger configurations.
- Language Translations and Terminal Debugging are perfect across all five on their commonly scored items. Needle Retrieval and Summarization are also perfect across runs 54–57. Terminal Debugging has eight commonly scored items out of nine, so its missing coverage must remain visible.
- The existing paired comparison of run 57 minus run 55 uses 291 questions and gives approximately **+0.0004 percentage points**, with a family-bootstrap 95% interval of **−3.20 to +3.34 points**. Their separate headline scores should not be interpreted as an established ranking.
- `eval_json_match` returns either 0 or 1 and displays at most five mismatches. `code_exec` returns a fixture-pass fraction but also truncates displayed failure details. The two evaluators expose very different levels of partial achievement.
- `S6-inheritance-*` asks for phenotype probabilities without explicitly declaring the nested object's shape. Several models supply an array. `S6-bitemporal-*` and `S6-scopedstate-*` also show recurring structural mismatches. These need prompt-contract review before their failures justify harder content.
- Some failures are clearly narrower than others: run 54 `H5-TH-worlds-07` has one incorrect support count; run 57 `AC2-02` passes 25/33 fixtures. Detailed diagnostics should preserve this distinction without calling either a full pass.
- The static suite already labels 255/427 items expert and 125 hard. Relabeling difficulty will add little information; difficulty needs measured evidence and more demanding tasks.
- Coverage consists of 28 static categories plus Interactive Tool Use. The 124 heuristic items remain outside capability, and 11 Creative Writing items measure compliance. Preserve those scope distinctions.

## 2. Workstream A — audit the new runs

### A1. Freeze the evidence and establish comparison eligibility

1. Produce an audit from one consistent read-only SQLite snapshot, recording snapshot time, included runs, result counts, source revision, and audit-tool version. Exclude credentials and endpoint URLs from exported artifacts.
2. Check expected versus stored question counts, duplicate/missing IDs and fingerprints, saved pass thresholds, score bounds, scored/excluded status, and agreement between row totals and saved reports.
3. Group runs by suite hash and saved protocol. Verify generator configuration, selected categories, worker counts, and context settings as additional dimensions. Keep v5 cohorts 48–50 and 51–53 separate from v6 runs 54–58.
4. Show coverage by category, family, evaluator, and outcome. Distinguish a completed run from a run with every question successfully evaluated.
5. Identify whether excluded requests concentrate in difficult families. Report common-item paired scores and sensitivity bounds for missing outcomes; do not treat exclusions as random or silently count them as passes.

**Deliverables:** `data/quality-audit-2026-09-21.json`, a compact cohort/coverage table, and a reproducible audit command recorded in the review.

Reproducible read-only command used for the saved artifact:

```powershell
uv run python audit_results.py data/bench.db --runs 51 52 53 54 55 56 57 58 --replay --output data/quality-audit-2026-09-21.json
```

### A2. Review grades in both directions

Review every scored capability non-pass in runs 54–58 and every endpoint error, missing answer, repetition stop, or formatting failure. Review v5 runs 51–53 for distinct failures and regressions, without duplicating identical cases unnecessarily.

Also examine apparent successes: take a deterministic sample of at least two passing answers per populated category/evaluator combination, expand every suspicious sample, and inspect families that pass universally. This catches false passes rather than only making failures more permissive.

Record each finding with run/question/fingerprint, relevant prompt text, answer excerpt or fixture result, current verdict, reviewed verdict, reason, confidence, and required action. Use these labels:

- Correctly graded task failure.
- Grader false failure or false pass.
- Ambiguous prompt/output contract.
- Incorrect or incomplete answer key/fixture.
- Legitimate formatting violation.
- Endpoint/runtime failure, missing final answer, or premature repetition stop.
- Unresolved; requires a second review or independent derivation.

Prioritize these concrete cases:

| Cases | Review question |
|---|---|
| `S6-inheritance-*`, `S6-bitemporal-*`, `S6-scopedstate-*` | Does the prompt explicitly establish every required nested key, type, and array order? Can content be assessed independently of the disputed representation? |
| `SP1-CR-03` claim C10; `SP1-SE-*` claim disagreements | Is the confirmed/refuted/unknown distinction supported by the supplied evidence and stated rule? Agreement among models is a review signal, not proof that the key is wrong. |
| `LR2-09` path representation | Is the required encoding unambiguous? If alternatives are valid, define only a question-local equivalence. |
| `SP1-FI-04` source ordering | The prompt explicitly requests sorted IDs; retain that requirement when reporting otherwise-correct arithmetic. |
| `H5-TH-worlds-*`, `S6-diagnosis-*` | Verify counts and optima with independent derivations; distinguish a single bad subanswer from a globally wrong solution. |
| Code partial failures such as `AC2-02` | Reconstruct the exact expanded fixtures, inspect boundary behavior, and reproduce in an isolated runner. |
| All 69 v6 repetition stops | Inspect saved tails for real loops versus valid repetitive data/code. Missing continuation prevents claiming what the final answer would have been. |

### A3. Make replay defensible

The current audit replays unchanged static prompts/evaluator names and skips incomplete output and expanded historical code cases. Extend it to check system prompts, output contracts, thresholds, generator settings, and fixture provenance as well.

- Recover the exact historical question manifest where possible. Reconstruct generated questions, expanded fixtures, and interactive environments only from versioned definitions and verify their fingerprints.
- Separate **historical-verdict reproduction** from **rescoring a saved response under a proposed grader**. Store both evaluator versions and the reason for every delta.
- Replay interactive transcripts against the corresponding simulator; parsing the enclosing transcript is not evidence of task completion.
- Mark unreconstructable cases explicitly. Do not score old output against a new prompt or silently substitute today's fixtures.
- Keep original grades immutable. Attach audit/rescore records and publish original-versus-reviewed summaries with covered/skipped counts.
- Turn each confirmed grading defect into a minimal positive and negative regression pair before implementing its fix.

**A exit gate:** all primary-cohort capability non-passes and reliability failures are dispositioned; success sampling is complete; each proposed grade change has evidence and a regression; unresolved/ambiguous items are identified and excluded from difficulty calibration where appropriate.

## 3. Workstream B — granular and precise evaluation

### B1. Introduce a structured evaluation result

Add a typed result carrying `evaluation_schema_version`, `evaluator_version`, legacy score, full-pass verdict, and a list of criterion results. Each criterion has:

- Stable ID, capability dimension, weight, and whether it is mandatory or a critical gate.
- Status: pass, fail, not evaluated, or evaluator error.
- Earned/possible points, machine-readable reason code, expected/observed evidence, and a JSON path, fixture ID, or interaction-turn reference.
- Optional prerequisite/group IDs to explain dependent requirements without implying an unobserved reasoning process.

Suggested dimensions are content correctness, completeness, evidence grounding, output-contract compliance, and task-specific operational constraints. Do not force every dimension onto every task or infer internal reasoning quality from answer text.

Keep the `(score, detail)` interface through an adapter while migrating callers. Unsupported legacy evaluators report unavailable criterion detail rather than invented granularity.

### B2. Preserve strict success while adding informative partial achievement

Publish three distinct measures:

1. **Full task success:** all mandatory requirements and critical gates pass.
2. **Criterion achievement:** weighted requirement credit, normalized within each task and then family/category. Group weights are declared before calibration.
3. **Contract compliance:** parsing, schema, ordering, and explicit formatting requirements, shown separately.

Initially retain the existing capability headline and add criterion achievement alongside it. A later headline change requires a scoring revision and fresh matched runs. Keep existing code fixture fractions available and clearly named.

Do not average arbitrary JSON leaves: a long array must not outweigh the central objective, and repeating equivalent fields must not increase credit. Missing answers, extra unsupported claims, and critical violations need explicit rules. A safety/authorization gate can fail the full task even when other criteria earn diagnostic credit. Never trade an unauthorized action for a few extra correct fields.

For excluded requests, criterion achievement is unavailable. Truncation/repetition/missing-answer outcomes remain scored failures; any readable partial content can appear as explicitly provisional diagnostics, not a substitute success score.

**Example:** a response correctly derives 9 of 10 equally weighted independent results but gives one wrong value may show 90% achievement, 100% contract compliance, and failed full-task success. A dependency-aware rubric may produce a different achievement score when those results are not independent.

### B3. Upgrade evaluators in priority order

| Evaluator/task class | Required improvement |
|---|---|
| Structured JSON | Emit all criterion/path results; distinguish parsing, type, missing key, extra key, array length/order, value mismatch, and cross-field inconsistency. Use explicit semantic groups rather than mismatch-count scoring. |
| Numeric/reasoning | Declare integer/exact-rational/unit/rounding contracts; permit numeric tolerances only where specified. Check requested subresults, feasibility, optimum, and tie-breaks independently. |
| Code execution | Give fixtures stable IDs and requirement tags; persist pass/fail/error/timeout per fixture. Report boundary, ordinary, invariant, and scale groups without letting hundreds of similar fixtures dominate. Distinguish runner failure from model runtime failure. |
| Instruction following | Report every constraint and distinguish content-transformation requirements from serialization constraints. Validate prompt and executable contract together. |
| Reviews/specialist tasks | Separate claims, evidence references, calculations/traces, proposed fixes, and unsupported allegations. Assess both missing findings and false positives. |
| Interactive tasks | Score observed final state, authorization, recovery, evidence/receipt validity, and protocol compliance; retain call efficiency as a separate metric. Check critical gates throughout the trace. |
| Heuristic prose | Keep its diagnostic scope; replace selected items with executable or evidence-grounded successors before promoting coverage to capability. |

Maintain strict duplicate-key/nonfinite-value rejection. Equivalences must be local and documented; ordered arrays remain ordered where requested. Prompt templates should render an explicit schema, null behavior, units, and ordering from the same contract metadata used by validation, without revealing answer values.

### B4. Version, persist, and display the new detail

- Extend `models.py`, `test_loader.py`, `validate_suite.py`, `evaluators.py`, `quality_execution.py`, and `interactive_tasks.py` for typed criteria and rubric validation.
- Persist criterion details through `quality_metadata_json` initially if size remains reasonable; use an additive normalized table only if querying/volume warrants it. Keep historical missing fields nullable/unavailable.
- Save a content-addressed run manifest containing the exact question, expected contract, generated parameters/fixtures, scorer version, simulator version, and relevant runtime/protocol settings. Keep evaluation-only material out of model prompts.
- Separate response-cache identity from evaluation identity so grader-only fixes can re-evaluate saved output without re-querying a model. A changed prompt or decoding setting still requires a new generation.
- Update `quality_report.py`, web run/comparison pages, and `app/services/html_reports.py` together. The report is now schema v2 with explicit v1 compatibility handling; `paired_comparison` accepts both versions and exposes criterion-detail availability.
- Display category → family → task → criterion drill-down, pass/achievement/compliance columns, excluded counts, all mismatch details, and evaluator version. Label item-weighted versus family-balanced values consistently.
- Preserve category/family balancing and clustered intervals. Cluster related templates across category boundaries for uncertainty estimation when they share a generator; adding a cross-category template must not create fictitious independent evidence.
- Show pairwise differences only on compatible questions, rubrics, and protocols. Explain unmatched/excluded items and disclose category coverage.

**B exit gate:** accepted gold answers get full credit; each one-requirement mutation fails strict success and affects only the intended criterion/dependencies; serialization cannot inflate content credit; critical violations always fail full success; original reports remain readable; CLI, web, JSON, and HTML agree.

## 4. Workstream C — increase difficulty across the board

### C1. Establish coverage and authoring targets

Treat difficulty as a combination of evidence reconciliation, interacting constraints, state depth, uncertainty, counterexamples, and scale. Record these parameters for every family, alongside author-estimated difficulty and observed pass rates.

For the first candidate revision, author **two distinct harder families for each of the 29 categories**, with at least two deterministic development instances per new family: **58 families / 116 candidate items**. Creative Writing remains a compliance track. Existing family counts must be reviewed for genuine independence; variants and relabeled copies do not satisfy the target.

Add further families wherever an objectively scored category would still have fewer than five independent families. For Language Translations, cover all six Czech/English/German directions across the candidate bank, even if this requires extra instances. This is a coverage floor, not a promise to append every candidate to the default suite.

Preserve a frozen v6 anchor profile. Select a new standard profile and a separate challenge profile after calibration, replacing redundant easy items in the new profile to control runtime. Give each profile a manifest/hash and report anchor/challenge results separately. Whole-suite v6/v7 scores are not directly interchangeable.

### C2. Category-by-category content plan

Each row describes the two minimum new families. Tasks must have independent ground truth and explicit, unambiguous output contracts.

| Category | New family A | New family B |
|---|---|---|
| Logical Reasoning | Constrained models with minimal counterexamples | Nonmonotonic rule updates and all optimal explanations |
| Mathematical Reasoning | Exact dependent probabilities with selection bias | Constrained discrete optimization with proofs/certificates and all ties |
| Reading Comprehension | Multi-document exceptions and precedence | Bitemporal contradictions with exact supporting sources |
| Classification | Compositional multi-label decisions with abstention | Minimal evidence changes that reverse a classification |
| Factual Knowledge | Reference-grounded mechanisms and counterfactuals | Cross-source factual reconciliation with missing evidence |
| Truthfulness | Answerable versus underdetermined linked claims | Conflicting sources and false-premise rejection with evidence IDs |
| Code Generation | Stateful parsers with malformed inputs and exact arithmetic | Transactional transformations with idempotency and invariants |
| Advanced Coding | Persistent/versioned structures under mixed operations | Multi-objective graph/interval algorithms with scale constraints |
| Code Review | Cross-file dataflow defects and minimal fixes | Migration/concurrency traces containing convincing false alarms |
| Terminal Algorithms | Streaming/external-memory algorithms | Deterministic graph/scheduling solutions under explicit resource limits |
| Terminal Debugging | Interacting faults requiring an executable patch | Regression preservation across dependency/configuration boundaries |
| Needle Retrieval | Indirect multi-hop retrieval across similar identifiers | Retrieval after revisions, deletions, and adversarial distractors |
| Long Context Coherence | Nested transactions and delayed references | Cross-document invariants after conflicting updates |
| Summarization | Evidence-linked reconciliation of events and reversals | Compact structured summaries with omissions/contradictions checked |
| Tool Using | Cost-optimal plans under prerequisites and partial observability | Recovery planning after stale reads and ambiguous tool outcomes |
| Agentic Use Cases | Adaptive policies under explicit risk/cost constraints | Multi-step workflows with rollback and evidence-dependent stopping |
| Interactive Tool Use | Multi-resource races and uncertain commit results | Paginated scope/permission changes requiring safe replanning |
| Security | Multi-hop exploitability with necessary preconditions | Least-privilege remediation and verifiable negative findings |
| Terminal System Admin | Configuration precedence and dependency diagnosis | Recovery sequences with service-state invariants in a simulator |
| Terminal File Operations | Atomic transforms over duplicates, Unicode, and links | Interrupted operations with rollback and exact directory-state checks |
| Terminal Science | Confounded experiments with valid estimands | Exact inference and numerical analysis with stability constraints |
| Instruction Following | Composed transformations with interacting exceptions | Global/local output constraints with state-dependent requirements |
| Creative Writing | Narrative continuity under interacting measurable constraints | Controlled revisions preserving facts, viewpoint, and form constraints |
| Translation | Logical scope, negation, and reference preservation | Ambiguity-aware meaning preservation under supplied terminology |
| Language Translations | Technical/legal terminology and modality in all six directions | Units, dates, gender/reference, and exception-scope preservation |
| Ethical Reasoning | Explicit-policy conflicts with counterfactual consistency | Consent/fairness allocations with infeasibility and all valid alternatives |
| Legal | Supplied jurisdiction/date-specific rules with exceptions | Evidence/claim mapping with deadlines and insufficient-fact cases |
| Finances | Liquidity/covenant/currency constraints under stress | Cash-flow reconciliation with timing, exact rounding, and sensitivities |
| R&D | Fault identifiability and optimal experimental design | Coupled mechanical/control constraints with uncertainty propagation |

Use supplied reference packets for specialist facts, with dates and authoritative sources recorded when authored. Do not turn these into tests of unstated current law or market conditions. Translation reports must distinguish semantic preservation from open-ended fluency, and factual application from broad recall. Creative-writing constraint scores do not establish artistic quality.

### C3. Validate tasks before model calibration

For every new family:

1. State what capability it measures and why it is harder than its anchor.
2. Derive answers through a second method; use brute force on small instances, algebraic derivation, differential implementations, or a reviewed policy table as appropriate. A generator calling its own solver twice is not independent verification.
3. Test a trusted solution through the actual evaluator/runner and test intentionally wrong solutions: one missing field, wrong boundary, spurious claim, wrong tie, unauthorized action, and fabricated evidence where applicable.
4. Test valid alternative representations allowed by the prompt, and reject everything outside that contract.
5. Add metamorphic invariants and randomized development instances; validate that changed irrelevant records do not change the answer.
6. Bound prompt/output size and runtime. Difficulty should come from the task, with scale tests explicit and supported by the runner.
7. Freeze development and evaluation seed lists separately. Public generators are reproducible holdouts by instance, not secret contamination-proof tests.

### C4. Calibrate, then select

- Pilot the candidate bank on the existing configurations with matched settings; retain the different-family model represented by run 58. Report the correlated Qwen configurations separately rather than treating them as independent model families.
- Use at least three generation repetitions for a stratified subset of discriminative, unstable, and apparently saturated families. Keep model repetitions separate from question variants and do not select the best answer across retries.
- Audit universal failures and universal successes before acceptance. All-fail items may be defective, ambiguous, too costly, or useful challenge items; pass rate alone cannot decide.
- Use **20–80% pilot full-pass rate** as a provisional selection band for many challenge families, not an acceptance law. Retain anchors and a controlled set of harder tail items.
- Flag categories where more than 80% of valid candidate items pass across every stronger configuration. Add distinct demands rather than more variants of the same saturated template.
- Flag excessive timeouts, repetition, or missing outputs separately from content difficulty. Maintain a common output budget and explicit task-level interaction limits.
- Select the standard/challenge profiles against both coverage and measured cost. Record prompt/output tokens, wall time, and runner overhead; reserve optional long-context sizes for a separately budgeted profile.
- Freeze rubrics, profile composition, and weights before final evaluation. Use untouched evaluation instances for the final matched runs; do not tune the suite to obtain a preferred ranking.

**C exit gate:** every category has harder validated coverage, sparse capability categories meet the family floor, saturation/floor effects are documented, all accepted tasks have independent answer verification, and fresh matched results establish observed difficulty.

## 5. Delivery sequence and validation

| Change set | Scope and principal files | Dependency | Planning estimate |
|---|---|---|---|
| 1. New-run audit | `audit_results.py`, audit artifact, new review document, replay provenance | None | 1–2 engineering days |
| 2. Contract repairs | `stretch_cases.py`, `review_cases.py`, affected YAML/oracles and regression tests | Audit findings | 1–2 days |
| 3. Criterion model and scorers | `models.py`, `evaluators.py`, `quality_execution.py`, `interactive_tasks.py`, loader/validator | Audit + rubric design | 3–5 days |
| 4. Storage and reporting | `quality_report.py`, `quality_suite.py`, database migrations if needed, CLI/web/HTML | Structured result contract | 2–3 days |
| 5. Harder candidate families | Generators, independent oracles, YAML, simulator/runner extensions | Stable contracts and scorers | 5–8 days |
| 6. Calibration and release | Matched run matrix, review, profiles, documentation, CI | Candidate validation | 2–3 days plus endpoint runtime |

Budget **14–23 engineering days** for the full scope as an initial estimate, with model availability, manual review findings, and simulator extensions the main uncertainties. Stop after change set 1 to revise the estimate against actual defects and replay coverage; this is a technical sequencing checkpoint, not permission to launch implementation as part of this planning request.

Validation should extend the existing checks rather than build a parallel testing framework:

- `validate_suite.py --strict`: criterion IDs/weights, critical gates, schemas, family metadata, malformed rubrics, output-contract consistency, and generator/manifest determinism.
- `selftest_evaluators.py`: positives, adversarial negatives, complete structured diagnostics, valid equivalences, deterministic scoring, and legacy adapters.
- `selftest_challenges.py`, `selftest_reliability.py`, `selftest_specialists.py`, `selftest_ceiling.py`, `selftest_stretch.py`: reviewed historical regression cases and independent keys for their families.
- `selftest_quality.py`: generated/repeated instances, interactive gates, saved manifest reproduction, cache behavior, runner-error attribution, persistence, and CLI/web equality.
- `selftest_reports.py`: original v1 reports, new schemas, partial/excluded runs, paired compatibility, escaping, drill-down, and offline HTML parity.
- New family-focused self-tests where new generators warrant them; add each accepted check to `.github/workflows/ci.yml`. Run Ruff and relevant existing tests for each change set, then the full existing CI suite at integration.

Release the evaluator, contracts, question bank, and report support together. Record scoring and suite revisions separately. Keep old results and profiles available, and attach any reviewed rescores explicitly. Publish a final review with original/paired scores, criterion achievement, strict success, coverage, failure modes, difficulty calibration, and known limits.

## 6. Completion criteria

- Recent runs have an auditable interpretation, including false passes, false failures, ambiguity, and reliability failures.
- Every newly scored requirement has machine-readable evidence, and near misses are distinguishable from broadly wrong answers without weakening full-success criteria.
- All 29 categories receive independently verified harder coverage; category scope and empirical difficulty are disclosed.
- Historical results remain reproducible or explicitly marked unreconstructable; grader changes and prompt changes are never conflated.
- Fresh evaluation runs use frozen compatible settings and untouched evaluation instances, with coverage and uncertainty beside the scores.
- The benchmark reveals differences that the data support, including ties and unresolved comparisons.
