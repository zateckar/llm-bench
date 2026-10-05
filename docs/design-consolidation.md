# One quality suite, one performance test

Status: accepted, 2026-10-05. It supersedes the selectable parts of the separate suite and performance-test designs.

## Problem

A person evaluating a model had to choose between four built-in quality suites and three performance tests. These suites were:
- rigorous;
- tool-calling conformance;
- safety & language;
- open-ended.

The three performance tests were the fixed workload, the context & reasoning sweep and open-loop load.

Every combination gave a different, partial picture. Two models could only be compared when someone had happened to run the same choices on both. The scorecard needed separate runs to fill its quality, latency, context and capacity columns.

## Decisions

### Quality: one standard suite

1. **`standard` (revision `standard-v1`) is the union of the four built-in suites**, 495 tasks. Question IDs, fingerprints, categories and families do not collide. Every question keeps its metadata exactly, so its fingerprint is unchanged and the rigorous subset still hashes to `7909c6325d1abd7b`.
2. **Areas.** A question's area is derived from its `metadata.cohort`:

   | Area | Source | Tasks | Scored |
   |---|---|---|---|
   | Reasoning & knowledge | rigorous-v15 | 323 | yes |
   | Tool calling & structured output | tool-conformance-v1 | 96 | yes |
   | Safety & language | safety-language-v1 | 46 | yes |
   | Open-ended requests | assistant-open-v1 | 30 | no; recorded for blind A/B studies |

   The quality summary adds an `areas` breakdown with category-balanced score, full-pass rate and counts per area.
3. **Headline score.** It is still the category-balanced achievement over all scored categories (31 + 6 + 6). Areas are reported next to it, not instead of it. Equal weight per category is the existing, documented method; weighting areas instead would give 46 safety tasks a third of the score.
4. **Execution.**
   - Text tasks use the bounded quality protocol; native tool tasks use `native-tools-v1`. Dispatch is per question, exactly as the safety suite already did.
   - The protocol identity is new, so standard runs never pair with runs of the old suites.
5. **Retired suites.**
   - `rigorous`, `tool-conformance`, `assistant-open` and `safety-language` are no longer offered anywhere: not in the run form, canaries, scorecard columns or gates.
   - They stay loadable so historical runs, reports, A/B studies and comparisons keep working, and runs of them are still labelled.
   - "Run again" on an old run runs the standard suite.
   - Stored canaries and decision-profile gates on a retired suite are migrated to `standard` at start-up. A migrated canary's baseline is cleared, because the old baseline is not comparable.
6. **Use-case suites** uploaded by application teams stay separate and versioned, and are still never paired with built-in suites.
7. **Stored options.** New runs record the quality suite explicitly. A run without a stored suite is a historical rigorous run.

### Performance: one staged test

8. **`standard-performance-v1` runs three stages in order and stores one report** (`schema_version` 6, `kind` "staged"):

   | Stage | What it measures | Engine | Defaults |
   |---|---|---|---|
   | 1. Latency | First-token and end-to-end latency, throughput and prefix-cache effect at increasing concurrency | Fixed closed-loop workload (`performance-v8`) | Concurrency 1, 2, 4, 8 |
   | 2. Context | Latency and completion as the input grows, up to the usable context | Context sweep (`context-sweep-v3`), bounded | One worker, the model's configured reasoning effort, contexts 256, 8k, 32k, 64k, 128k, … up to the declared context limit (128k when unknown), 4,096 output tokens |
   | 3. Capacity | Sustainable arrival rate under SLOs | Open-loop load (`open-loop-v1`) | Mixed workload, rates 0.5, 1, 2, 4 req/s, 120 s per rate, in-flight cap 256 |

   A full test with defaults takes about 20–40 minutes, depending on the model.
9. **Bounded context stage.**
   - The sweep engine gains two options: explicit context lengths and a fixed list of reasoning efforts.
   - The stage measures one effort, the model's configured one, and does not probe all eight.
   - The context ladder ends at the deployment's declared limit minus output and framing headroom, so it verifies the declared limit instead of guessing one.
   - Explicit rejections and inferred ceilings work as before.
10. **Checkpoints.** The report is saved after every stage, and within the context and capacity stages as they progress.
    - Stopping a run keeps the finished stages.
    - A failing stage does not stop the stages after it unless the endpoint is down. The run fails with the stage's message and keeps everything measured.
11. **Display.**
    - The Performance view renders stage by stage: Latency, Context, Capacity. Each stage uses its existing view, so comparisons align stage with stage across runs.
    - The capacity estimate is computed once per run, from all stages together.
    - Historical single-kind reports appear as the matching stage, so old and new runs remain comparable stage by stage.
12. **Scorecard.** One staged run provides latency, context and capacity evidence at once.

### Settings

13. **What the run form asks for.** A run asks for a model and what to measure: Quality, Performance or both.
    - Quality offers the standard suite or an uploaded use-case suite.
    - Everything else uses standard defaults, so runs are comparable across models.
    - An **Advanced** section, collapsed by default, holds the latency concurrency, the context limit override, and the capacity workload, rates, step length, arrival process, attainment target, seed and in-flight cap.
14. **Legacy options.** The old performance kinds `fixed`, `sweep` and `load`, and the modes `sweep` and `load`, are accepted from saved plans and API callers and mapped to the standard test. Compatible settings carry over: concurrency up to 32, the sweep's context maximum, and the load settings. Runs already queued with an old kind still execute their old test.

## Not changed

- Evaluators, evaluator versions, question content and the three performance engines' measurement methods.
- Use-case suites, A/B study mechanics (now fed by the standard suite's open-ended area) and repeat-group statistics.
