Current follow-up: [rigorous-v10 design and critical review](BENCHMARK_DESIGN.md). Historical run findings below describe the protocols used by those runs.

# Review of runs 57 58 62 and 65

Implementation note: this is the historical audit snapshot. The fixed 245-task rigorous-v9 suite now supersedes the v8 profile choices and commands below; see [README.md](README.md). Historical replay tools were archived under ignored data/archive-v8-tools. Saved historical grades were not modified.

Reviewed on 3 October 2026. The saved measurements are reproducible, but the evaluation is not uniformly correct or precise. The suite supports a large advantage for run 57 over run 58 on the tested workload. It does **not** support a reliable ranking between runs 62 and 65, or a general ranking of the underlying models. Some failures come from incomplete prompts or grader defects. The H7 additions have substantial validity problems.

## Results and trust

These are the saved category/family-balanced capability scores, not revised grades. Legacy prose heuristics and creative-writing compliance are excluded from capability.

| Run | Configured model alias | Profile | Scored / attempted | Request errors | Saved capability | Scored capability items |
| --- | --- | --- | --- | --- | --- | --- |
| 57 | skoda-llm-primary-test | v6 | 430 / 434 | 4 | 95.07% | 295 |
| 58 | gemma4-31 | v6 | 434 / 434 | 0 | 63.13% | 299 |
| 62 | Qwen3.8-Flash-Next-FP8 | v6 plus H7 | 486 / 492 | 6 | 82.10% | 351 |
| 65 | Qwen3.8-27B.long | v6 plus H7 | 478 / 492 | 14 | 83.30% | 343 |

The names are configured aliases; the records do not independently establish weights, quantization, routing, or actual model identity. Runs 57/58 share suite hash `921ffa63e20a612e`; runs 62/65 share `c807bb8e677f8e3e`. Comparing their headline percentages across those two suites is misleading. All four used development seed 1729, one variant, temperature zero, an unset model seed, and a 65,536-token quality output cap. Temperature zero alone does not establish repeatability.

On the **289 identical capability questions scored in all four runs**, the balanced scores are:

| Run | Common-question capability |
| --- | --- |
| 57 | 95.24% |
| 58 | 63.05% |
| 62 | 93.25% |
| 65 | 91.13% |

This comparison controls question differences and uses the saved grades, including their limitations. It is not a corrected leaderboard.

Paired comparisons within each original suite give a clearer result:

| Comparison | Balanced difference | 95% family-bootstrap interval | Scored pairs / matched |
| --- | --- | --- | --- |
| 58 minus 57 | −31.81 percentage points | −37.00 to −26.36 | 295 / 299 |
| 65 minus 62 | −0.56 percentage points | −3.97 to +2.73 | 342 / 355 |

Run 65's higher headline score reverses direction on paired scored questions. Different missing outcomes change denominators and family/category weights. The second interval crosses zero, so these runs do not establish which endpoint is better. Bootstrap intervals measure uncertainty over sampled task families within fixed categories; they do not capture generation variability, prompt defects, infrastructure variability, or uncertainty about real-world task coverage.

Assigning every excluded capability outcome either zero or one gives sensitivity bounds of 94.29–95.13% for 57, 63.13% for 58, 81.38–82.22% for 62, and 81.10–83.80% for 65. These are **not confidence intervals**. The planned capability denominators are 299, 299, 355, and 355; exclusions are 4, 0, 4, and 12 respectively.

## What was verified

The audit read `data/bench.db` in a read-only SQLite transaction. Historical rows were not updated. Every saved question fingerprint and prompt was reconstructed: **1,852 of 1,852**, including generated questions, strengthened code fixtures, and interactive definitions. Result counts, uniqueness, score ranges, and agreement between result rows and saved report scores all checked out.

The current evaluators replayed 426 answers in 57, 432 in 58, 459 in 62, and 456 in 65. Interactive transcripts were replayed through their simulator, including matching tool replies and final state. The remaining 79 outcomes comprise request errors, stopped repetition, and missing final answers; their runtime classifications were preserved. The only binary replay change under the implemented grader fixes is `FK-05` in 57. Agreement otherwise demonstrates reproducibility, not the correctness of every answer key.

Artifacts:

- [Full audit data](data/run-review-2026-10-03.json), including per-question findings, paired comparisons, and performance checks.
- [Earlier audit export](data/quality-audit-2026-10-03.json), retaining the original audit evidence.
- [Replay tool](review_run_cohort.py), which reconstructs complete definitions and never writes historical grades.

## Grading defects and genuine failures

| Finding | Evidence | Assessment and treatment |
| --- | --- | --- |
| Undeclared JSON shape | `S6-inheritance-*` specifies phenotype order but expects an unnamed nested object. Some answers return the correct ordered array. | Two failures in 57, two in 62, and three in 65 have entirely correct content after a question-local shape conversion. New prompts explicitly declare the mapping. |
| Undeclared wrappers | Run 65 gives correct parallel arrays for `S6-bitemporal-01/02`, and a correct ordered root array for `S6-scopedstate-03`. | Three more failures are explained entirely by unspecified representations. New prompts state the nested schema. |
| Hidden output key | Run 62 uses `sequences` for both H7 probability/planning items; the grader requires `optimal_sequences`, which the prompt does not name. | Both answers otherwise match the complete key. H7 is excluded from the new profile. |
| Unicode punctuation | Run 57 `FK-05` correctly identifies Mozambique's flag and writes AK‑47 using a nonbreaking hyphen. | Literal keyword matching incorrectly gave zero. Normalize only the two typographic hyphens in this heuristic evaluator; AK‑48 still fails. |
| At least versus exactly | `CW-04` asks for at least two statistics, but its rubric demands exactly two. Run 57 supplies three and meets the other formal constraints. | A false compliance failure. Add minimum/maximum occurrence rules; the revised item uses `min_count: 2` and supplies explicitly fictional figures. This does not verify statistics or writing quality. |
| Evidence versus falsity | `SP1-CR-03` C10 asks whether migration and queries alone establish acceptable latency. The key says refuted; “not established” can reasonably mean unknown. | Revise C10 to a concrete 40-row export, a 20 ms budget, and at least 1 ms per SELECT. Its 41 SELECTs now make refutation testable. |
| Heuristic coverage | Run 57 `TH-01` gives an honest uncertainty response that its refusal patterns miss. `SE-04` expresses a password-hash protection beyond the regex's limited match window. | These are reasons to keep the legacy prose heuristics outside capability. A punctuation fix does not make regex matching a semantic judge. |

The audit's shape conversions are diagnostics with explicit scope. They are not general aliases or automatic regrades. For example, `S6-inheritance-01` in 57 still has a wrong conditional fraction after its shape is repaired; similar answers in 62 also retain substantive errors. A valid alternative shape does not excuse incorrect content.

Several run 57 failures are real: `AC2-02` passes 25/33 fixtures but incorrectly merges intervals across zero-valued gaps; `LR2-08` adds an invalid explanation that misses an alarm; `SP1-RD-03` computes the latest pose's y-error as 0.6 instead of 0.4. `SP1-FI-04` violates an explicitly requested source ordering. `SP1-SE-05` treats an unsupported private-key leakage claim as refuted rather than unknown. These examples show useful discrimination in the retained tasks.

No truncation was recorded in these four runs. Repetition failures number 4, 2, 27, and 20. Inspection found looping generations without usable final answers, so preserving them as failures is appropriate. Raising the output budget does not demonstrate a remedy for those loops.

## Why the questions compress differences

On the 289 common capability items, **141 pass in every run**. Among runs 57, 62, and 65, **240 pass in all three**: 83.0% of the common set supplies no binary separation among those endpoints. All four pass the eight common Terminal Debugging items and two Terminal System Admin items. Many familiar, short, single-operation tasks leave little room to distinguish stronger models.

The H7 bank reduces the ceiling, but often for the wrong reason. Its category labels reuse a small set of primitive generators, so many named families do not represent independent capabilities. In its 56 static additions:

- Ten code tasks request an implementation while a shared footer requires a JSON object with no Markdown. The code grader expects source. A response satisfying one instruction can fail the other.
- Twelve evidence tasks omit the complete nested schema and the rule selecting evidence only from threshold-reaching stances.
- Two creative-writing tasks require a first-person viewpoint and ABBA/ABAB pattern that are absent from their prompts.
- Planning, ledger, retrieval, and allocation tasks leave important output keys or count-versus-collection representations unspecified.

The automated audit flags prompt-contract defects in 48 of the 56 static H7 questions. A flagged question can still have a substantive model error; the flag does not turn it into a pass. The saved H7 capability subsets score 44.64% in 62 and 51.85% in 65, with 25/56 and 28/52 full passes. These percentages combine genuine difficulty with invalid requirements and different exclusions. They should not establish a model ranking.

The older granular metric also inflates some partial scores: descendants of a missing or wrongly shaped JSON field disappear from its denominator. A code fixture that raises an exception can similarly disappear from achievement while other fixtures pass. Consequently the saved 93.70% and 94.03% criterion means in 62/65 are not evidence of equivalent near-perfect capability. They also mix scopes and count raw leaves without balancing requirements.

The user's concern about compressed differences is supported by these ceilings. A benchmark percentage is still a workload success rate, not a linear measure of intelligence; harder tasks cannot establish an assumed universal gap without matched runs.

## Third-party benchmark research

The new tasks are original and use these evaluation designs as inspiration. No published leaderboard scores are imported, and these synthetic tasks should not be described as reproducing the external benchmarks.

| Primary source | Useful design | Application and limit |
| --- | --- | --- |
| [LiveBench paper](https://livebench.ai/livebench.pdf) | Diverse challenging tasks with objective ground truth and refreshed source material. | Use typed answers, checkable arithmetic, varied task mechanisms, and versioned instances. Public generators and reused anchors cannot establish contamination freedom. |
| [IFEval](https://arxiv.org/abs/2311.07911) | Explicitly verifiable instructions and separate instruction-level and prompt-level success. | Report strict pass and individual criterion achievement separately; state exact schemas and minimum counts. Formal compliance remains separate from creative quality. |
| [EvalPlus](https://evalplus.github.io/) | Expanded code tests reveal errors that a few examples miss. | Add boundary, randomized-combination, and differential fixtures. Boundary coverage gets half the new code rubric weight; passing many routine combinations cannot conceal a boundary failure. This is correctness testing, not an efficiency benchmark. |
| [SWE-bench harness](https://www.swebench.com/SWE-bench/reference/cli/) | Executable tests of repository repairs. | Use concrete parser and dependency regressions instead of prose claims about a fix. The current harness tests bounded functions, not full repository navigation and patch workflows. |
| [RULER](https://arxiv.org/abs/2404.06654) | Multi-hop tracing and aggregation beyond single-needle lookup, evaluated across context lengths. | Add revised-record traversal with tombstones and distractors; retain the separate measured context sweep. The new 189-record packet alone does not establish long-context performance. |
| [NoLiMa](https://arxiv.org/abs/2502.05167) | Retrieval requiring associations beyond literal keyword overlap. | Identify lexical matching as a remaining limitation: the new packet still contains literal keys. A true low-overlap association bank remains future work. |
| [τ-bench](https://arxiv.org/abs/2406.12045) | Policy-constrained interaction, final-state evaluation, and reliability across repeated trials. | Keep actual simulator actions, uncertain payment recovery, concurrency, authorization gates, and final-state checks. Recommend repeated trials; variants alone do not measure repeated-run reliability. |
| [BFCL multi-turn evaluation](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) | Evaluate state and responses over multiple tool turns. | Replay recorded tool replies and check state rather than accepting a proposed plan as an executed success. The portable JSON protocol does not test native provider function-calling syntax. |
| [GPQA](https://arxiv.org/abs/2311.12022) | Expert-authored questions with expert validation. | Preserve complex specialist evidence tasks and distinguish missing causal assumptions. These additions use supplied or fictional rules; they have not received independent graduate-domain expert validation. |

## Implemented candidate profile

`--quality-profile rigorous` now assembles **189 tasks across 29 categories** with the default one seed and one variant: 171 revised anchors, 13 new reasoning/code instances, and five interactive tasks. It excludes the flawed static H7 bank and the 124 legacy prose-pattern diagnostics. Three creative-writing items remain explicitly scored as formal compliance rather than capability. The default manifest is exported in [the candidate manifest](data/rigorous-manifest-2026-10-03.json).

The revised anchors retain compositional mathematical and logical reasoning, tool and transaction traces, evidence reconciliation, specialist reviews, scoped translation, and expanded coding tests. New IDs and explicit schemas distinguish revised prompts from historical questions. Exact prompt schemas describe types and keys without publishing solution values or result-array lengths.

| New family | What must be solved |
| --- | --- |
| Selective report | Conditional probability and expectation under two interacting reporting conditions, using exact fractions. |
| Unsatisfiable cores | Enumerate every inclusion-minimal inconsistent clause set and find the minimum repair size. |
| Service cascade | Track transitive stop cascades and accepted-start rate windows that persist across restarts. |
| Inode state | Distinguish directory entries, hard links, symbolic paths, writes, and intermediate rename checkpoints. |
| Confounded estimand | Compute pooled and target-standardized effects, reconcile Simpson reversal, and distinguish randomized identification from missing exchangeability. |
| Consent allocation | Optimize integer allocations with consent, budget, lower bounds, and secondary tie criteria. |
| Revised multihop records | Select current records before following a chain; honor tombstones and return ordered evidence. |
| Liquidity waterfall | Apply daily cash flows, capped borrowing, repayment, floor breaches, and net-debt accounting. |
| Deadline exception | Distinguish a fictional priority-based novelty date from an actual-filing grace deadline and unsupported inventive step. |
| Permission intersection | Combine tenant, role, token caveat, ACL, deny, and revocation conditions despite an untrusted instruction. |
| Escaped parser | Repair delimiter parsing, escapes, Unicode, empty and trailing fields, and first-error behavior; 33 fixtures. |
| Dependency regression | Produce simultaneous topological layers; handle duplicates, external dependencies, cycles, and blocked descendants; 29 fixtures. |
| Atomic settlement | Enforce prefix solvency, whole-transaction rollback, consumed rejected IDs, duplicates, and self-transfers; 30 fixtures. |

Five interactive families exercise document conflict recovery, unknown payment outcomes, paginated deletion previews, permission replanning after page revisions, and concurrent multi-document changes. The two useful H7 simulations are retained with revised contracts and new IDs. They preserve final-state and authorization gates and report efficiency separately. All side effects occur in simulations.

Evaluation now has these properties:

1. **Strict success and partial achievement are separate.** Incorrect JSON content remains a full failure; every code fixture must pass. Partial diagnostics explain what worked without turning it into a pass.
2. **Requirements have balanced weights.** Each structured top-level requirement receives one point divided among its leaves, so 100 evidence entries cannot overwhelm one incorrect conclusion. Evidence has its own dimension. New code rubrics split weight equally between boundary and generated-combination groups.
3. **Missing requirements stay in the denominator.** Wrong or absent parent structures receive zero for blocked descendants; model-code fixture exceptions receive zero achievement. Parsing, unavailable infrastructure, and efficiency remain distinguishable.
4. **Reports expose denominator limits.** Capability criterion achievement is category/family balanced and accompanied by coverage. Excluded capability counts and zero-to-one sensitivity bounds are shown in JSON, Markdown, and web/export summaries.
5. **Evaluator changes have provenance.** Evaluation schema 2 and individual evaluator versions are recorded in comparison protocols. Old and new protocols cannot silently produce a compatible paired comparison. Frozen historical suite hashes are unchanged.

The profile is available in the CLI and both run/plan forms. The default remains frozen v6 for compatibility; select `rigorous` explicitly for the candidate. H7 choices and summaries now display their known prompt defects.

## Validation and next measurement

Strict validation passes for all 427 frozen static questions and all 184 noninteractive candidate questions, with zero errors or warnings. Sixteen new regression tests cover historical fingerprints, schemas, missing-field denominators, exception accounting, balanced weights, seed limits, independent code oracles, executable reference solutions, near-miss implementations, occurrence counts, Unicode punctuation, and audit normalization. Seeded generation was exercised through the maximum ten variants at five seeds, including both integer boundaries.

Existing evaluator, granular, quality, challenge, reliability, specialist, ceiling, stretch, calibration, H7/release, review-queue, run-queue, performance, benchmark-runner, and report self-tests pass. Ruff and whitespace checks pass. The CI workflow includes candidate validation and its regression tests. Some answer keys have independent derivations or implementations; others have invariant and manually reviewed checks. This is not independent expert validation of the entire suite.

To run a fresh comparison using the existing configured endpoint:

```bash
uv run python benchmark.py --quality-profile rigorous --suite-split evaluation --suite-seeds 19,23 --variants 2 --no-cache --report rigorous.quality.md
```

This configuration has 243 tasks before optional context additions. Use exactly the same configuration, token budget, decoding settings, and workload for every model. `rigorous-only` offers the 60 new/extended instances under those settings for a smaller pilot. Repeat identical instances separately to measure run-to-run stability; do not treat seed variants as independent families or report “at least one success” as consistent reliability.

No live model calls were made for this review. The candidate's difficulty and ability to widen or stabilize model differences remain unmeasured. Four terminal categories currently have only one family; their equal category weight makes those results particularly sensitive, and the overall family-bootstrap interval is consequently unavailable. Expand independently reviewed families there before using the candidate as a precise leaderboard. Exact-length context sweeps, full repository repair, low-overlap retrieval, subjective language/creative quality, and externally validated domain expertise remain distinct coverage gaps.

## Performance findings

All saved concurrency request-rate calculations agree with requests divided by elapsed wall time. However, some reported per-stream rates are implausible as evidence of model generation speed: at concurrency one, median rates are approximately **76,078 tokens/s in 62** and **89,607 tokens/s in 65**. Nearly all recorded request time precedes the first delivered token. Buffered or burst delivery is a plausible explanation; the records lack raw token-arrival traces needed to determine the cause. Interpret these as client-observed delivery calculations, not verified hardware decode rates.

End-to-end latency and wall-clock workload throughput remain useful measurements of those particular endpoints and workloads. They still need comparable conditions: 57/58 used the uniform performance workload, while 62/65 used a mixed workload and a different initial concurrency grid. Low-load levels contain only eight requests, making tail percentiles and capacity boundaries imprecise. Saved SLO capacities are 8, 12, 4, and no passing tested level respectively. The extrapolated active-user counts depend on the supplied SLO and request-frequency model; they are not validated user capacities.

None of these four runs enabled the optional long-context **quality** sweep. A performance context sweep or a model alias ending in `.long` cannot establish long-context answer accuracy.
