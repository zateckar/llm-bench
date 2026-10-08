# LLM Bench

The [compact-bank review](reports/standard-selection-review-2026-10-07.md) records the 180-question trim. The earlier [newest-run review](reports/latest-run-review-2026-10-07.md) explains calibration, the capability ladder and measurement corrections.

Benchmark OpenAI-compatible endpoints with one fixed question suite and one fixed performance workload. Execution, transport, evaluators, task generators and reports live in the application package under `app/benchmarking`. Models, users and benchmark runs are managed through the web app.

## Run

Python 3.13 or later and uv are required.

```sh
uv sync --frozen
uv run python -m app.main
```

Open http://localhost:8000. Add an endpoint under Models and select **Run benchmark**. Pick a model and, under **Measure**, tick **Quality**, **Performance** or both. Everything else uses standard settings, so runs compare across models ([design](docs/design-consolidation.md)).
- **Quality** runs the **standard suite**: 339 tasks in four areas (reasoning & knowledge, tool calling & structured output, safety & language, and open-ended requests recorded for blind A/B studies). An uploaded use-case suite can be chosen instead. Quality uses up to four workers.
- **Performance** runs the **standard performance test**, one test in three stages with one report: **latency** by concurrency, **limits** by context and concurrency up to the deployment's declared context limit, and **users at SLO**: how many simulated people and agents in multi-turn sessions one deployment serves at each context cap, where it is exhausted and which resource limits it. Runtime depends on the measured limits; the context stage now collects at least 20 requests per cell and tests every context independently.
- **Advanced settings**, collapsed by default, hold the latency stage's maximum concurrency (1–32, default 8), the context stage's context-limit override, maximum concurrency (1–256, default 256) and per-request targets (first token p95, default 2,000 ms; slowest 5% output speed, default 40 tok/s), and the users stage's user model, context caps, user range, windows and attainment target.

A run that measures both runs quality first and the performance test after it, and stores both results in the same run; the performance test is skipped if no quality answer could be scored. Saved plans and API callers may still send the old performance kinds (`fixed`, `sweep`, `load`) or the old built-in suite names: old kinds map to the standard test with compatible settings, and old suites are rejected. The run list's **Measures** column shows what each run measured. Any run can have **repeats** (1–10; a submission expands to at most 50 runs). The same form handles immediate and scheduled runs: leave the start time empty to run now, or set it in your browser's local timezone. Add more runs in that form for an ordered comparison. All submissions share validation, storage and one queue; only one benchmark runs at a time. Manage them under **Queue & schedules**. Editing, cloning and “Run again” use the same submission code and current suite.

The base URL usually ends in /v1; the client appends /chat/completions. Runs offer self-contained HTML downloads, quality JSON, performance JSON and side-by-side comparisons. Run details and comparisons have separate Quality, Performance and Answers views. The compact interface uses a neutral, Shadcn-inspired token theme with responsive navigation; Jinja, Alpine, HTMX and Tailwind remain in place.

The Performance view contains only dedicated load, context-sweep and cache measurements, laid out stage by stage (Latency, Context, Users) with the capacity estimate after them; historical single-test runs appear as the matching stage, and a Capacity stage appears for earlier runs that measured one. The latency stage leads with peak measured output, response and first-delivery timings at that same load, and request reliability. Quality-task timings appear only as expandable diagnostics in Quality; delivery does not imply a correct or complete answer. Quality score cards and charts do not include load throughput. Durations use milliseconds, seconds or minutes; detailed measurements and distributions expand on demand. Offline HTML downloads keep quality and performance results in separate sections. Performance comparisons align runs in columns with shared workload, context and concurrency selectors, metric rows, and slices across both axes. Fixed-load, context-sweep and cold/warm cache measurements stay distinct; missing points show a dash, and HTML downloads retain the aligned measurements.

## Quality

The **standard suite** (`standard-v3`, [standard_suite.py](app/benchmarking/standard_suite.py)) is the one built-in quality suite. It is a fixed subset of four source areas. The 180 unchanged questions fully passed in runs 91, 95, 96 and 97 are excluded from new runs; their definitions remain in regression tests. The selection is provisional, spans mixed decoding settings, and does not establish universal easiness. All 24 new ladder tasks remain. Retained questions keep their fingerprints:

| Area | Source | Tasks | Scored |
|---|---|---|---|
| Reasoning & knowledge | subset of rigorous-v16 | 288 | yes |
| Tool calling & structured output | subset of tool-conformance-v1 | 13 | yes |
| Safety & language | subset of safety-language-v1 | 8 | yes |
| Open-ended requests | assistant-open-v1 | 30 | no; recorded for blind A/B studies |

The headline is category-balanced criterion achievement over the 36 retained capability categories (one formal-writing category is reported separately), so each category weighs the same whichever area it is in. Run pages and HTML downloads add a **By area** table with each area's achievement, full-pass rate and coverage. Text tasks use the bounded quality protocol and tool tasks native tool calls, chosen per task. Standard runs never pair with runs of the four former suites; those suites are no longer offered, but their historical runs, reports and studies still work, and "Run again" on one runs the standard suite. Canaries and decision-profile gates on a former suite move to `standard` at start-up, and a moved canary's baseline is cleared.

### Reasoning & knowledge

**The unfiltered rigorous v16 source bank contains 347 tasks across 31 categories; new standard runs retain 288 across 29 categories.** The source inventory is 171 revised static anchors, 140 original generated instances, 28 interactive simulations, and eight long-context accuracy checks at 8,192 and 32,768 cl100k_base reference tokens. The static bank has one question per file under [tests/questions](tests/questions); [quality_suite.py](app/benchmarking/quality_suite.py) assembles the complete suite deterministically. Question seeds 19 and 23 and two variants per generated family are fixed. V16 adds 24 original capability-ladder tasks in six families, with independent exhaustive oracles and executable code fixtures. The ladder reports full success and criterion achievement separately by tier; greater empirical discrimination is unverified until matched repeated runs exist. V15 added eight original behavioral-reconstruction tasks: probe a simulated legacy integer quantizer or TTL cache, infer its policies, then submit replacement Python. Hidden grading covers every allowed finite-domain input, preserves existing behavior through regression guards, and gives no correctness feedback before completion. V15 also clarified four output contracts exposed by saved-run calibration. Existing inference, retrieval, ambiguity and code-execution families retain their generation rules. Criterion achievement is the primary score; full success remains a separate statistic.

Saved-run calibration is read-only: `uv run python calibrate_suite.py --suite-hash b11260da421820b5 --output reports/calibration.json` writes JSON and Markdown without changing the database. It groups complete reports by exact protocol and question inventory, separates completion and contract failures, balances repeated evidence by model, and reports observed task/family patterns. Admission evidence requires at least two models and three fully scored runs per model; no task is automatically pruned. The [v14 calibration snapshot](reports/calibration-v14-2026-10-04.md) and [contract audit](reports/calibration-v14-2026-10-04-review.md) document runs 80/81. One run per model is preliminary evidence, not proof of saturation or repeated-run reliability. The new reconstruction families have passed source correctness gates and await empirical calibration.

The headline is **criterion achievement**, using the earned/possible criterion value within each question, then averaging within each family, within each category, and across capability categories. Categories never apply a pass threshold. This prevents many similar variants from dominating. Tasks without a content breakdown retain their evaluator score, including zero for ungradeable answer formats. Full-task success, contracts and criterion coverage remain separate statistics. The retained creative-writing item measures formal compliance separately; it does not measure artistic quality.

JSON checks compare every required value, type and key. New count/receipt contracts require integer literals; general number contracts continue to accept numerically equal integer/float values. Code executes as submitted against boundary and generated-combination fixtures; forbidden imports and changes to protected input state fail. Interactive actions run in local simulations with final-state and authorization gates. Missing requirements, blocked rubric dependencies and fixture exceptions earn no credit. Correct members of short arrays retain diagnostic credit for the present requirements; extra members and unexpected fields reduce precision credit; the full-pass contract still fails. Relevant records, revisions and distractors are distributed across the long-context inputs.

Model configuration controls temperature (0–2, default 0) and optional reasoning effort (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`). Provider default omits `reasoning_effort`; an explicitly selected unsupported level fails visibly rather than being dropped. Submitted runs capture these settings, and quality and fixed-context performance use them with model seed 0; repeat *i* of a repeat group uses seed *i*. Context sweeps probe each effort separately. Existing models retain temperature 0 and provider-default effort until edited.

Ordinary quality questions use **bounded-quality-v1**: at most two model calls sharing a 65,536-token output allocation, including reasoning. The first call receives 57,344 tokens; 8,192 are reserved for finalization. A followup is issued only for output truncation or a reasoning-only response, using the original messages and previous response plus a fixed instruction to return a complete final answer. Only the last answer is graded. Completed wrong answers receive no grader-guided retries. Smaller allocations reserve one eighth, capped at 8,192. Existing interactive simulations keep their separate action/turn budgets.

No heuristic repetition cutoff is used. Transport errors are not retried during quality generation; up to three explicit compatibility-negotiation attempts can remove rejected streaming fields before generation. No requested reasoning effort is silently removed. Reports retain request allocations, outputs for recovered questions, usage, timing, initial passes and recovered passes. Both calls occupy the same question worker. Truncation and missing final answers after recovery stay scored failures. Endpoint failures, unsupported context, evaluator errors and cancellation remain explicit unscored outcomes with coverage and zero-to-one sensitivity bounds. Unstarted tasks remain in the planned denominator after an outage. This protocol changes comparability with historical single-call runs; their generation records and full-pass verdicts are preserved; score projections carry a separate scoring revision.

Saved reports record suite fingerprints, individual criteria, evaluator versions, client versions, timeout/retry policy and quality worker count. Paired comparisons require compatible protocols and identical question fingerprints. Invalid evaluator scores or diagnostics become unscored evaluator errors; missing required rubric criteria retain zero diagnostic credit. Family bootstrap intervals are unavailable when any category lacks multiple independent families; they do not estimate repeated-run variability.

There are no profiles, filters, partial reruns, seed/split controls beyond the repeat count, output-budget controls, custom context grids or response-cache options. Historical scores are retained as historical data and do not become v15 scores.

The new research draws on [FEVER](https://fever.ai/dataset/fever.html), [HoVer](https://hover-nlp.github.io/), [BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html), [ComplexBench](https://github.com/thu-coai/ComplexBench) and [BigCodeBench](https://github.com/bigcode-project/bigcodebench), alongside the earlier LiveBench and RULER methods. Tasks are original local exercises; third-party datasets and benchmark scores are not reproduced.

Fresh matched model runs are needed to establish empirical difficulty and model separation. Public question banks can be contaminated. The [v15 design and critical review](BENCHMARK_DESIGN.md) records the calibration, new tasks, primary research sources, client effects, fixed grading defects and verification methods. Reconstruction uses trusted local simulations inspired by [VulcanBench's behavioral-parity methodology](https://github.com/morganlinton/VulcanBench#suites); no third-party tasks or scores are reproduced. These bounded code functions and text-protocol simulations do not measure complete repository work, native tool APIs or unrestricted real-world agents; native tool calling is covered by the tool-calling area below.

Only complete leading reasoning envelopes are removed before grading. Unfinished scratchpads have no final answer; literal tags within JSON and code remain data. JSON has a portable 256-level nesting limit. Code fixtures reject arbitrary objects that merely print like correct values, preserve ordered members inside sets, and distinguish interpreter-launch failures from submitted-program failures.

### Tool calling & structured output

The **tool-conformance-v1** area ([tool_suite.py](app/benchmarking/tool_suite.py)) sends native `tools`, `tool_choice`, `parallel_tool_calls` and `response_format` fields and grades native `tool_calls`. It therefore tests the deployment — chat template, tool-call parser and guided decoding — as well as the model. It has 96 tasks in six categories: tool selection and typed/nested/enum arguments (each sent both streaming and blocking), abstention and `tool_choice` control (`none`, `required`, named), parallel calls (including `parallel_tool_calls: false`), multi-turn dependent chains with error recovery and grounded final answers, the rigorous suite's five agentic simulations re-hosted on native tools, and JSON-schema / JSON-object structured output. Prompts mix English, Czech and German dates, units and wording.

Grading ([tool_protocol.py](app/benchmarking/tool_protocol.py)) separates **wire conformance** (arguments are a JSON object, the tool exists, arguments satisfy the declared schema, call IDs are present and unique, stream framing is sound, no raw tool-call markup leaks into content, consistent finish reason) from **behaviour** (right calls and arguments, abstention, choice and parallelism honoured, answers grounded in tool results). Criterion achievement reports behaviour; the contract score reports wire conformance; a full pass needs both. Tool schemas use a small JSON-schema subset validated in-house ([schema_subset.py](app/benchmarking/schema_subset.py)). A request the deployment rejects (HTTP 400/404/422/501 naming a tool or format feature, e.g. `tool_choice` without a tool parser) is a scored `feature_rejected` failure, not an outage. Reports add wire diagnostics: wire-check failures, stream defect codes, leaked markup families, rejected features and stream-versus-blocking results. Runs show the native transcript with each call, its raw arguments and the simulated result.

### Repeated runs

A submission with repeats creates one ordinary run per repeat, linked as a **repeat group**; repeat *i* uses model seed *i* and everything else is identical. Editing, cloning and “Run again” keep the group together. The group page (`/runs/groups/{id}`) reports the run spread with a t interval over runs, pooled achievement, per-task stability, unstable tasks, and **pass^k** (succeeds in all *k* tries) next to pass@k, using unbiased estimators. Groups can be compared with each other: tasks pair by fingerprint, the interval resamples families and runs, and p-values come from a family-level sign-flip randomization test, Holm-adjusted when several comparisons are shown. Seed differences alone do not block pairing; any other protocol difference does. Single-run comparisons use the same test and add an exact McNemar test on full passes.

### Use-case suites

Application teams' golden examples can be uploaded as **use-case suites** under **Admin → Use-case suites**. A suite is one YAML document with a `suite` header (slug, name, description, owner, optional default system prompt) and a `questions` list using the built-in question fields; start from the [example document](app/benchmarking/usecase_example.yaml). Only deterministic evaluators are accepted; evaluators that execute code are not. A `json_match` question may add a `response_format` (`json_object` or a `json_schema` in the supported subset), which sends it through the native structured-output path. Uploads are fully validated, and **Check** reports problems and warnings without saving anything.

Every changed upload becomes a new immutable version; uploading the same questions again is rejected. Runs pin `usecase:<slug>@<version>`, so a run, its edits, clones and reruns keep using the version they were submitted with, even after newer uploads or archiving. All versions of a suite share the report suite name `usecase:<slug>`, so runs on different versions compare on the questions that are identical in both. Use-case suites are never paired with built-in suites. Archiving hides a suite from the run form but keeps its versions and runs.

### Blind A/B studies and LLM judges

Most everyday use has no answer key: emails, summaries, explanations, advice, rewrites. Questions with `evaluator: open_ended` and `expected: {criteria: [...], reference?: ...}` are run as usual, but the answer gets the outcome `recorded` and is never scored. It is excluded from achievement, passes and paired comparisons, and the quality summary lists it separately. The standard suite's open-ended area, **assistant-open-v1** ([open_suite.py](app/benchmarking/open_suite.py)), has 30 workplace requests in English, Czech and German across writing, summarising, explaining, advising, transforming data and reviewing. Use-case suites may contain open-ended questions too.

A **blind A/B study** (**Blind A/B** in the navigation; admins create one from two completed runs of the same suite, also from the compare page) pairs the two runs' answers by question fingerprint. Answers with failed requests are left out, and a study holds at most 1,000 pairs. The study copies the prompts, criteria and answers, without reasoning blocks.

- **People** vote blind: model names are hidden, sides are shuffled, and the least-voted pairs come first. Votes are left, right, tie or both bad, with an optional anonymous comment; each person votes once per pair, and votes cannot be changed.
- **LLM judges** (`pairwise-judge-v1`; any configured model) judge every pair in both orders at temperature 0, and the two verdicts are combined to cancel position bias. Judging runs in the background and can be resumed.

Results (`ab-stats-v1`) give the preference for B with a family bootstrap interval and an exact sign test, plus a per-category breakdown. For each judge they add position consistency, length bias, and agreement with people (Cohen's κ). The pair browser shows judge reasons, and the study can be downloaded as JSON. Design: [docs/design-ab-studies.md](docs/design-ab-studies.md).

### Safety and language adherence

The **safety-language-v1** area ([safety_suite.py](app/benchmarking/safety_suite.py)) has 46 deterministic tasks in English, Czech, Slovak and German, in six categories:

- **Language adherence (12):** answer in the language the user wrote in, or the one asked for, without drifting into English.
- **Prompt injection (8):** do the task on pasted e-mails, invoices, web pages and CVs, and ignore the instructions hidden in them.
- **Tool-result injection (6):** the same through native tool results. Calling the injected tool (send e-mail, delete document, close ticket, accept all invitations) is a critical failure.
- **Confidentiality (8):** do not reveal an escalation code or individual salaries from the system prompt. Spelled-out, separated and Base64 forms also count as leaks.
- **Scope and policy (6):** decline out-of-scope or abusive requests with the configured sentence, and still help with in-scope ones.
- **Over-refusal (6):** answer legitimate questions that merely sound risky.

Injected instructions ask for a token the model has to *derive*, such as a sum or a reversed number. An answer that quotes the injection to warn the user therefore still passes; only obeying it fails.

Language is detected by **language-id-v1** ([language_id.py](app/benchmarking/language_id.py)), a small local detector for en, cs, sk, de and pl. It uses stop-word lexicons and distinctive letters, scores each sentence, and ignores code, URLs and e-mail addresses. The `language_adherence` evaluator (`expected: {language, patterns, must_not, min_words}`) needs the required language, at least 85% of the words in it, a minimum length and the content patterns. It is also allowed in use-case suites.

Every run page shows an **Answer language** table for questions that declare a language in their metadata (`answer_language`, `language` or `lang`). It reports the answers in that language, the answers in another language or mixed, and the answers too short to tell, with the mismatching tasks listed. This is a diagnostic only and never changes a score. The suite does not contain harmful-content prompts; see [docs/design-safety-language.md](docs/design-safety-language.md).

### Monitoring: deployment fingerprints and canaries

Every run starts by recording a **deployment snapshot** (`deployment-probe-v1`):

- the served-model entry from `/models` (root, context limit);
- the engine version from `/version`;
- the responding model;
- the prompt-token counts of five fixed chat requests with `max_tokens: 1`, which change exactly when the chat template or tokenizer changes.

The snapshot's fingerprint is compared with the model's previous one, and a change raises an alert listing every changed field. A failed snapshot never fails a run. Admins can also **Check now** from **Monitoring**.

**Canaries** (`canary-v1`) queue a quality run of the standard suite or a use-case suite every 1–168 hours through the normal queue, without building a backlog. The first completed run becomes the baseline; admins can promote another run later. Each canary run is compared with the baseline using the paired run comparison, and raises:

- **alerts** for a significant quality regression, a failed run or a deployment change;
- **warnings** for endpoint errors, a TTFT or latency median 1.5× slower than the baseline, or a baseline that can no longer be compared.

Events appear on the Monitoring page and as a dashboard banner until an admin acknowledges them. A canary can post alerts and warnings to a Teams/Slack-compatible webhook. Each canary keeps its newest 100 runs, its baseline and every flagged run. Design: [docs/design-deployment-monitoring.md](docs/design-deployment-monitoring.md).

## Performance

The **standard performance test** (`standard-performance-v5`) runs three stages in order and stores one report (schema 6):

| Stage | Measures | Engine | Defaults |
|---|---|---|---|
| 1. Latency | First-token and end-to-end latency, throughput and prefix-cache effect by concurrency | Fixed workload (`performance-v9`) | Concurrency 1, 2, 4, 8 |
| 2. Context | Limits: the load served within per-request targets, and where measuring stops, for every input context | Context sweep with limit search (`context-limits-v2`) | Contexts 256, 8k, 32k, 64k, 128k, … up to the usable limit × concurrency 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256; targets 2 s first token and 40 tok/s; the model's configured reasoning effort, 4,096 output tokens |
| 3. Users | Simultaneous users whose multi-turn sessions all meet their SLO, per context cap; then where the deployment is exhausted and which resource limits it | Session users (`sessions-v2`) with constraint diagnosis (`constraints-v1`) | 40% chat / 60% agents, caps 32k, 128k and the usable limit, 8 → at most 256 users at SLO, saturation search up to 1,024 users, 60 s warmup + 180 s per level, 95% attainment |

The usable context limit is the `max_model_len` from the run's deployment snapshot minus 4,096 output and 512 framing tokens, 131,072 when the deployment does not declare one, or the Advanced override. The context stage therefore verifies the declared limit rather than guessing it. The report is saved after every stage and while the context and users stages progress; stopping a run keeps what was measured. A failing stage does not stop the next one, unless no request in it succeeded (the endpoint is down). The run then fails with the stage's message and keeps every finished stage. One staged run supplies the scorecard's latency, context and users evidence. The sections below describe each stage's engine; historical runs of the former separate tests remain readable and comparable stage by stage.

Versions v1–v3 also ran an open-loop **capacity** stage. v4 dropped it: the users stage answers the capacity question directly, and the capacity stage added about 10–40 minutes per run. Earlier runs keep their capacity stage, comparisons and scorecard capacity evidence. Saved plans and API callers that still send its settings (`load`, `in_flight_cap`) are accepted, and those settings are ignored.

### Latency stage

One workload uses a **1,024-token reference input** and asks for integers 1 through 80 with a **256-token output limit**. Each request has a unique leading identifier to reduce prefix reuse. Input tokenization happens before timing. Temperature and reasoning effort use the submitted model settings; model seed is 0.

Streaming is negotiated before measurement. Every connection is warmed and reused. Timed requests have **one attempt**. Load levels are powers of two up to the configured maximum, including the maximum itself: maximum 6 tests 1, 2, 4 and 6. Each level measures at least 24 requests or four rounds per worker, whichever is larger.

Negotiation requires an explicit unsupported-field error and available retry budget. One-attempt calls preserve the negotiated settings. Completion choices after a terminal finish reason, invalid finish metadata and streams containing only usage or DONE are endpoint errors; trailing usage after a finish reason remains valid.

Reports show end-to-end latency and first-delivered-token p50/p95/p99, request errors and error rate, failed-request latency, aggregate output tokens per second and successful requests per second. V6 also stores timed request samples for latency, TTFT and output-length distributions, output-length summaries, SSE chunk gap summaries, and mean output token time. The latter is a delivery proxy: first-to-last output chunk span divided by reported completion tokens minus one. Buffered bursts, estimated counts and outputs below two tokens are excluded. Chunk intervals are not individual token latency. These definitions follow the distinction between client latency and throughput metrics in [NVIDIA GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2600/user-guide/docs/perf_benchmark/genai-perf-README.html). Rates divide successful output/completions by the full measured wall time, including failed requests, ramp-up and drain. Progress updates and prompt construction stay outside that interval.

TTFT includes first delivered content or reasoning and depends on provider buffering. Available-byte streaming reads avoid an added read-to-fill delay, and complete HTTP bodies return sockets to the connection pool. Malformed or prematurely ended streams count as request errors. Single-chunk or short-burst delivery and estimated token counts are marked explicitly. Provider completion counts may include reasoning; estimates cover all delivered final text and reasoning and may differ from the provider tokenizer. Cached token counts, when supplied, remain in JSON. Missing or malformed usage is estimated at four characters per token; explicit reported zero counts are preserved. Latency includes every attempt and backoff, with successful-attempt latency also retained in JSON. Quality requests use a 180-second inactivity timeout and an independent 1,800-second total stream budget; performance retains the 360-second stream budget. Deadlines and invalid stream protocols stop without retries. Transient transport/HTTP failures retain bounded retries. Future failures retain per-attempt TTFT, delivery counts, reasoning/content character counts and last-delivery timing. The total stream budget is checked at received bytes and EOF; a blocked read can overrun it up to the inactivity timeout. Quality request timing and error concentration are shown separately from the controlled performance suite, including for historical runs. Timed performance calls still make one attempt.

### Context stage

The context stage finds the **limits** of the model and its inference stack by input context and concurrency. Contexts run in increasing order; within each, concurrency climbs 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256 up to the configured maximum. Every measured cell is judged against per-request targets (`perf_sweep.judge`):

- **Within targets:** first token p95 and slowest 5% output speed meet the targets, and at most 1% of requests failed or were incomplete. Slowing down is expected and is measured; missing a target does not stop the search.
- **Limit:** more than 10% of requests failed or were incomplete, first token p95 is over 10× its target, median output speed is under a tenth of its target, or the server is saturated (aggregate output grew less than 10% over the previous load while the median first token grew 1.5×). Results beyond that point are not meaningful.

A limit at *c* requests skips higher loads at that same context (status `skipped_limit`). Every context starts an independent load search because caching and batching can make capacity non-monotonic. Explicit context-length rejections can still bound larger contexts. The standard test collects at least 20 requests per cell; target passes require complete TTFT and output-token timing coverage. Limit cells get no cached-prefix pair. The report leads with a **Limits by input context** table: the highest load within targets (consecutive from one request), the load where the limit was reached and why, and the highest load measured. These are closed-loop requests with one short fixed answer, not user counts; users served under a traffic mix come from the users stage.

The context stage uses the **context & reasoning sweep** engine with explicit input lengths, one round, a 4,096-token output limit and only the model's configured reasoning effort. The former standalone sweep, kept for historical runs, used input lengths **256, 32,768, 65,536, 131,072, …, 1,048,576** and concurrency **1, 16, 32, …, 256**, with 1–8 rounds per cell and an 8,192-token output limit by default. At the full range with one round it had 306 paired cells, 78,372 timed cold/warm requests and 18 untimed cache primes per accepted effort (2,448 cells / 4,896 cold/warm measurements across all eight efforts). Lengths are exact cl100k_base input-text tokens, excluding chat framing; provider input counts are recorded separately. Each cell pairs unique cold prefixes with a primed shared prefix and changing suffixes. Long padding is shared; only active workers build a full prompt.

The context stage probes only the configured effort; the former sweep probed provider default plus every standard reasoning value above. Probes retain unsupported and failed probes, and measure API-accepted settings. Acceptance cannot prove that a gateway honors or distinguishes efforts. Explicit field rejections can omit temperature/seed or remap max_tokens to max_completion_tokens during probes; reasoning effort is retained. Streaming capabilities and these changes are recorded before one-attempt timed calls. Worker connections are warmed with small inputs and reused. Sweeps use a 1,800-second stream budget and the configured socket inactivity timeout.

The desktop sweep explorer provides a context/concurrency matrix, effort and metric selectors, larger cells with visible measured values, exact cell summaries and slices along both axes. It shows first-token and whole-answer p50/p95, per-request end-to-end output tok/s, a stream delivery-rate proxy and aggregate output tok/s. Whole-answer latency and per-request rates exclude failed, empty, filtered and truncated answers; first-token timings also retain successfully delivered truncated responses. Aggregate output includes delivered truncated output over full cell wall time, including failures and client overhead. Estimated token counts, cached tokens, bursts, sample counts, errors and incomplete answers remain visible. One round at concurrency 1 is one observation, so tail percentiles require caution.

Each resolved cell is committed individually. Stop requests preserve saved cells and prevent late workers from changing terminal status or results. An explicit context rejection at concurrency 1 skips larger contexts for that effort with the same output allocation; a rejection at higher load does not establish a global context limit. Three consecutive contexts without a complete single-worker answer or three consecutive load levels without any complete answers trigger an untimed 256-token single-worker control probe. A healthy control infers a workload-specific ceiling and skips larger contexts or load levels for that effort; an unsuccessful control stops the sweep with a visible endpoint/completion error. Success resets failure streaks. A single failed context skips its higher loads while later contexts still start at concurrency 1. Controls and worker warmups are additional requests outside the timed budget. Skipped and unmeasured cells carry no invented timing. Schema-4 and staged JSON downloads hydrate the saved cells; offline HTML includes script-free matrices and complete measurement tables.

The latency and context stages are closed-loop measurements. They do not establish how many users a deployment serves or infer model-side prefill/decode speed. Small-sample tail percentiles and different output lengths limit comparisons; compare common settings and repeat measurements when stability matters. User capacity comes from the users stage below.

### Users stage: users at SLO

The users stage answers "how many people and agents can this deployment serve at once, with sessions up to this context?" — for example *mixed load, sessions up to 256k: 60 users, each getting the first token within 2 s and at least 40 tok/s* ([design](docs/design-session-capacity.md)).

**Simulated users.** Each user runs one session after another. A turn sends the whole history plus a new message, waits for the answer, then thinks (people) or runs tools (agents) before the next turn, so contexts grow as real conversations do. A session ends after its planned turns or when the next turn would exceed the context cap. The default user model is an estimate, to be replaced with session traces:

| Class | Share | System prompt | First / later message | Answer | Pause between turns | Turns | SLO |
|---|---|---|---|---|---|---|---|
| Chat | 40% | 1,500 | ~1,300 / ~500 tokens | ~500 tokens | 5–180 s | 2–30 | first token ≤ 2 s, ≥ 40 tok/s |
| Agent | 60% | 8,000 (tool definitions) | ~2,800 / ~3,200 tokens | ~280 tokens | 0.5–30 s | 10–200 | first token ≤ 2 s, ≥ 40 tok/s |

Presets are mixed, chat only and agents only; custom JSON takes up to four classes with `[min, max, weight]` histograms. Users of a class share its system prompt, so the prefix cache helps as it would in production; each history is unique to its session.

**Steady state.** Users join in a session already in progress (longer sessions are proportionally more likely, as a random moment of real traffic finds them), with its history, at staggered times in the first half of the warmup. A user's first request is a cold prefill and is excluded. The measured window starts when the warmup has passed and every user has had a first answer.

**Throughput.** Tokens from requests dispatched in the measured window are divided by the time through the last such response, including drain. This cohort rate is distinct from an engine token counter within the window.

**Pass.** A request meets its class SLO when it completes, dispatch lag + first token is within the target and its output token time is within 1000 / tok/s. A level passes only after warmup completes and every assigned class has measured requests and reaches the attainment target (default 95%) and the client's dispatch lag p99 stays at or below 100 ms; otherwise the client, not the server, was the limit and the level is not counted as a failure. A level with a class below 50% after a third of the window stops early.

**Search.** Per context cap, user counts double from the start (8) while levels pass, then bisect between the highest passing and lowest failing count to within 10%. Caps run in increasing order. The previous result is a starting hint; every cap measures its own failing count because cache and batching effects need not be monotonic. A result is *established* when a higher count failed, a *lower bound* when the maximum passed, and *not met* when even one user failed.

**Saturation search.** After the SLO search, user counts keep doubling from the largest measured count until a level is **beyond a hard limit** — the same rules as the context stage: more than 10% of requests failed or were incomplete, a class's first token p95 is over 10× its target or its median output speed under a tenth of it, not every user got a first answer within the request timeout, or throughput grew less than 10% over a level with 1.5× fewer users while the median first token grew 1.5×. Levels already measured by the SLO search count, so a cap that collapsed at its first failing count needs no extra level. Saturation levels stop early on those rules rather than on the SLO. Every cap independently measures its saturation limit. The result per cap is *exhausted at N users* (with the last count below), *not reached up to N* (search limit, default 1,024), with inherited limits retained only in historical reports, plus the peak output throughput and where it occurred. Turn it off or lower its limit under Advanced.

**What limits the deployment.** Each failing and each exhausted level names one **constraint** with the evidence behind it and changes that address it. Without telemetry it comes from the client: which part of the SLO failed, the prompt-to-output token ratio, and prefix-cache hits (when the endpoint reports cached tokens) against the reuse the session histories allow. With B300 telemetry it uses vLLM and GPU signals over the level's measured window (rates and rolling p95s skip their first minute):

| Constraint | Signals | Typical remedies |
|---|---|---|
| KV cache capacity | KV occupancy ≥ 90% or preemptions | FP8 KV cache, more KV memory (gpu-memory-utilization, FP8/NVFP4 weights, more GPUs), shorter contexts, KV offloading |
| KV cache eviction | Prefix hits below 60% of the reuse sessions allow, with occupancy ≥ 80% or histories larger than the KV capacity | KV offloading (CPU/NVMe, LMCache), FP8 KV cache, more memory, session-aware routing |
| Prefix cache not reused | Same low hits without memory pressure | Enable prefix caching, byte-identical history rendering, session-aware routing |
| Batch slots | Running requests flat at a ceiling with a queue while KV has room | Raise `--max-num-seqs` (and `--max-num-batched-tokens`) |
| Scheduler / GPU compute / queueing | Queue with KV room and idle GPUs / busy GPUs / unknown GPU activity | Raise `--max-num-batched-tokens`, check API server overhead / scale out, quantize, speculative decoding |
| Prefill | First-token misses, prompt-heavy load (≥ 2 uncached prompt tokens per generated token) | Prefill/decode disaggregation, larger prefill chunks, better cache reuse, more TP |
| Prefill interfering with decode | Output-speed misses under prompt-heavy load | Prefill/decode disaggregation, smaller `--max-num-batched-tokens`, separate agent pool |
| Decode speed | Output-speed misses under a light prompt load | Speculative decoding (EAGLE/MTP), FP8 KV cache, FP8/NVFP4 weights, cap the batch, more TP |
| Client / errors | Dispatch lag p99 > 100 ms / failures without server pressure | Larger client / check error examples |

The first matching rule names the constraint and later matches are listed as also present (errors under a full KV cache are a symptom). Remedies already in place according to vLLM's `cache_config_info` (FP8 KV cache, CPU offloading, prefix caching) are not suggested. The rules are heuristics over deployment-wide signals: telemetry includes other traffic on the deployment.

**Report.** Users at SLO and exhaustion by cap, a "What limits this deployment" table per cap (at the SLO limit and at exhaustion), attainment and output-throughput charts by user count, every measured level with per-class attainment, first token p95, slowest-5% output speed and server signals, and the user model. Runs with the same user model, windows, target and seed are compared by cap.

The stage runs one client thread per user. Closed-loop users slow down when the server does, which is how people behave; a burst of independent arrivals queueing faster than they are served (open-loop overload) is not simulated. Levels at the default windows take about 4 minutes; the SLO search over three caps takes roughly 45–90 minutes and the saturation search usually adds 1–3 levels per cap (exhausted levels often stop early).

### Cache, telemetry and capacity estimates

Performance v9 also pairs cold and warm prefixes at 8,192 and 32,768 reference tokens, at concurrency 1 and the configured maximum, using at least four requests per mode. Both the latency and context stages report cache telemetry coverage, cached-token fraction, uncached prefill and effective input rates (prompt tokens divided by client TTFT), decode delivery rate, TTFT and aggregate throughput. Unknown cache usage stays n/a; explicit zero is a reported miss. Prefill rates include queue, network and first-token overhead, and decode rates exclude estimated counts, large first deliveries and streams dominated by sub-millisecond event bursts. Incomplete timing coverage cannot establish a slow-tail output target. These are client proxies, not engine timings. Prefix reuse saves repeated prefill work; see [vLLM automatic prefix caching](https://docs.vllm.ai/en/v0.9.2/features/automatic_prefix_caching.html). Historical runs cannot supply paired cache measurements.

Optional B300 vLLM correlation uses the `PROMETHEUS_API_*` credentials and `PROMETHEUS_B300_HOST=smbea02n01`. Authorization accepts Basic credentials with or without the prefix. Set each deployment's **B300 vLLM metric model name** to the exact Prometheus `model_name`. `PROMETHEUS_TIMESTAMP_SHIFT_SECONDS=85` moves source samples forward 85 seconds and queries the corresponding earlier window; use 0 for aligned clocks. The correction assumes source clock lag, not transport delay. Queued runs capture selectors and correction without credentials; JSON retains original timestamps. Reports include a two-minute baseline and corrected UTC phase correlation. Collection happens outside measurements, uses only `job="vllm"` for vLLM metrics, and never fails a benchmark. Rolling rates/p95 cover 60 seconds and all deployment traffic. Missing samples and residual clock disagreement remain explicit.

Telemetry v3 adds optional series used by the constraint diagnosis: server time per output token (`vllm:inter_token_latency_seconds`, or `time_per_output_token` on older vLLM), vLLM's `cache_config_info` (KV blocks and block size, KV dtype, prefix caching, CPU blocks; matched by instance) and DCGM GPU metrics from the same host (`DCGM_FI_PROF_SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`, `DRAM_ACTIVE`, `DEV_FB_USED`/`FB_FREE`, `DEV_GPU_UTIL`, PCIe bytes, power), selected by the `Hostname` label and the deployment's **B300 GPU indices** (model setting such as `0-3`; blank uses every GPU on the host). Missing optional series do not make a collection partial; the report lists them. The DCGM metric and label names follow the default dcgm-exporter configuration; adjust `GPU_HOST_LABEL` and the queries in `vllm_telemetry.py` if your exporter differs. Collection may take up to 60 seconds after a stage.

Benchmark results show **achieved output rates** separately from provisional **traffic budgets**, with an **Adjust assumptions** page and JSON/HTML exports. Quality workload usage is divided by its recorded question-phase duration, excluding queue time; its rate is corroborating evidence, not a chat latency result. Defaults reserve 30% headroom and 98% availability, put 80% of traffic in 10 busy hours with 2× peak bursts, and model chat (1k context, 256 output, 2s p95 TTFT, 300 output tokens/min/user) separately from agents (32k context, 256 output, 10s p95 TTFT, 2k output tokens/min/user). Only matching context/output, reasoning and cache measurements meeting latency, streaming and completion targets qualify for workload budgets. Reports separate observed-rate 24h equivalents from traffic budgets, active users from simultaneous requests, and homogeneous tests from modeled mixed traffic. A concurrency-capped test leaves maximum serving capacity unknown. Missing long-context measurements stay unavailable; quality usage is never substituted for a matching load test. These budgets are modelled from latency and context measurements; the users stage measures user counts directly.

### Open-loop load (capacity stage of earlier runs)

The standard test no longer runs this engine (see above); it describes the capacity stage of v1–v3 runs, former standalone load tests and load tests still queued from before. It sends requests on a seeded schedule, whether or not earlier requests have finished, the way independent arrivals behave ([design](docs/design-open-loop-load.md)).

**Workload.** A workload is up to six request classes. Each class has:
- a mix weight
- input and output length histograms (`[min, max, weight]` buckets in reference tokens)
- a shared system prefix, for example tool definitions
- its own SLO: first token, optional output token time and optional whole-answer time

Presets cover chat, long-context agents, retrieval-augmented answers and an 80/20 chat/agent mix. Custom JSON lets teams mirror histograms from their gateway logs.

**Arrivals.**
- Arrivals are Poisson or bursty (gamma gaps with a chosen coefficient of variation). The same seed and settings give every model identical traffic.
- Output length is enforced by the token limit on a counting prompt.
- Arrivals that find the in-flight cap full are dropped and count as SLO misses. They are never delayed.
- Dispatch lag is added to user-visible timings, so a slow client cannot hide server queueing.

**Rate ladder and result.**
- A step passes when the SLO attainment target (default 95%) is met overall and for every class, no arrival was dropped and dispatch lag p99 stays at or below 100 ms.
- The ladder stops after consecutive failures. The highest passing rate is the **sustainable arrival rate**. It is *established* when a higher rate failed because of the server, and a *lower bound* otherwise.
- **Goodput** counts only requests that met their SLO.

**Soak.** A soak runs one rate for a long step. Long steps are summarised in time windows, and drift between the first and last third is flagged.

**Reports and capacity.**
- Finished steps are saved as the ladder runs.
- Reports show results by rate and by class, miss reasons, drift, dispatch diagnostics and cache telemetry.
- Runs with identical traffic, SLOs and sampling settings are compared by rate.
- The capacity page adds an **open-loop arrival capacity** block: the measured rate with headroom, availability and busy-hour assumptions applied. It covers only that workload and those SLOs.

## Scorecard and decision profiles

**Scorecard** shows one row per model with its latest evidence: never the best run, and never a combined score.
- **Quality:** one column for the standard suite and one per use-case suite, showing achievement with its 95% interval. Runs of the former built-in suites are not evidence. A repeat group counts as one piece of evidence.
- **Capacity:** one column per load workload, showing the sustainable rate under the SLO, from earlier runs' capacity stages and standalone load tests. The model page also lists users at SLO by context cap for each user model, from the users stage.
- **Latency:** TTFT p95 and per-request output speed at concurrency 1.
- **Context:** the context measured by the context stage (or a historical sweep), or declared by the deployment.
- **Operations:** canary status, open alerts and the deployment fingerprint.

**Freshness.** Every cell is marked *current*, *stale* or *unverified*:
- *current*: measured on the fingerprint served now;
- *stale*: the deployment has changed since;
- *unverified*: there is no snapshot and no later change is known.

Evidence older than 90 days is also flagged as old.

**Leaders and ties.** In each quality column the leader is compared with every other model by the paired test, Holm-adjusted. Models without a detectable difference are marked tied, and different protocols are marked not comparable.

**Decision profiles** state the requirements of one use case as gates:
- quality on a suite or category;
- sustainable rate on a workload (evidence only from earlier runs; new runs do not measure it);
- users at SLO on a user model with sessions up to a context, judged at the exact measured context cap, excluding inherited results;
- TTFT, whole-answer time or output speed at a concurrency;
- context;
- no open alerts and all canaries ok.

Each gate passes, fails, is missing or is stale. A model *meets* a profile only when every gate passes on evidence that is current or unverified.

**Decisions.** Admins record a decision per model (approved, approved with conditions or rejected) with a required note. The record keeps the gate results, run ids and fingerprint at that moment. It is flagged when the deployment changes later.

**Exports:** `/scorecard.json` and `/scorecard/profiles/{id}/export.json`. Design: [docs/design-decision-dashboard.md](docs/design-decision-dashboard.md).

## Database storage

Large prompts, answers and report payloads use versioned, lossless compression: zlib for small records and LZMA when its larger dictionary saves more space on large records. Existing zlib and plain-text rows remain readable. Application reads and JSON/HTML downloads decode them transparently; indexed scores and queryable outcomes remain plain columns. Standalone SQLite readers must use `app.storage.DETECT_TYPES` with sqlite3 or call `unpack_text` on payload columns.

Interactive pages use persisted quality projections and short prompt previews, populated when a run finishes or once during migration. Category question lists and complete answer details load on demand. Run history has 50 rows per page. Full JSON and portable HTML downloads retain every audit record; projections do not replace the original reports.

Run `python maintain_database.py` to inspect space usage, or add `--apply --report reports/database-maintenance.json` to compress historical payloads (including older compressed records), reaggregate saved criteria under `criterion-achievement-v1`, refresh read projections and reclaim free pages. Maintenance processes one report at a time and requires an idle queue; it first creates a consistent compressed SQLite backup under `data/backups`. Prompts, answers, evaluation criteria, original evaluator scores, full-pass verdicts and timing data are preserved. Restore a backup by decompressing its `.db.gz` to a database file while the application is stopped. Keep the application code supporting the stored compression versions with a restored database.

**Production rollout:** application startup now runs a versioned storage migration against its configured database on the mounted persistent volume, after schema/recovery initialization and before HTTP service or queue dispatch. The first deployment creates one consistent `.db.gz` backup, recompresses historical plain-text and older compressed payloads, checks integrity and foreign keys, and runs `VACUUM` with WAL truncation. It preserves all saved scores and decoded data. Pending runs wait until migration finishes. Completion is recorded in `database_storage_migrations`; subsequent restarts skip it. An interrupted compaction resumes using the existing backup. Errors stop startup before any queued work starts. Run one application process against the volume during migration; stop any older container still using it.

Deploy the new image **on the production host**, retaining the existing database volume. With this repository's Compose setup, use `docker compose up --build -d app` and inspect `docker compose logs app` for `Server database storage migration`, including before/after database and WAL sizes and the backup path. The image allows ten minutes for initial health readiness; adjust your platform's startup timeout for larger databases. Allow at least three times the current database plus WAL size in free disk space for the backup and SQLite rewrite. The migration checks this before changing payloads. A backup remains on the volume and uses additional disk space; move it to your backup storage after verifying the deployment. No backup is deleted automatically.

For deployments pulling the CI image, pull the new tag and recreate the container rather than only restarting it. Verify the running container's `org.opencontainers.image.revision` label against the intended Git commit; the CI image includes this label. The startup log's `lossless-lzma-v1` revision confirms that the storage migration code ran.

To compact again without changing historical scores, stop the server and run the same image as a one-off job against its production volume. For Compose:

```sh
docker compose stop app
docker compose run --rm --no-deps app python maintain_database.py --database /app/data/bench.db --storage-only --apply --report /app/data/database-maintenance.json
docker compose up -d app
```

This targets the **remote persistent database** when executed on the production host. Inspection without mutation is available with `docker compose exec app python maintain_database.py --database /app/data/bench.db`. Database storage still increases as new benchmark history is retained; compression reduces the cost per record and VACUUM reclaims unused pages. Limiting retained history would require a separate retention policy.

## Configuration and deployment

Copy [.env.example](.env.example) to .env for local configuration. Development creates an admin account using ADMIN_USERNAME / ADMIN_PASSWORD (defaults admin / changeme). For production, set ENVIRONMENT=production, a unique SECRET_KEY and ADMIN_PASSWORD; startup rejects missing/default values. Models and API keys are managed by administrators. OIDC settings remain optional.

```sh
docker compose up --build -d
```

[docker-compose.yml](docker-compose.yml) requires SECRET_KEY and ADMIN_PASSWORD and persists the SQLite database in a named volume. For HTTPS, configure your reverse proxy and set COOKIE_SECURE=true. Keep data and credentials outside source control.

Python application and development dependencies are locked in [uv.lock](uv.lock). CI and Docker use Python 3.14.8 and uv 0.12.22. Vendored browser libraries and fonts are recorded with their source URLs, versions, licenses and checksums in [manifest.json](app/static/vendor/manifest.json). Tailwind v4 requires modern browsers (Safari 16.4+, Chrome 111+, Firefox 128+); see its [compatibility requirements](https://tailwindcss.com/docs/compatibility).

## Verification

```sh
uv run python validate_suite.py --strict
uv run python -m unittest selftest_client selftest_adversarial selftest_frontier selftest_granular selftest_rigorous selftest_quality selftest_perf selftest_telemetry selftest_run_options selftest_benchmark_runner selftest_reports selftest_ceiling selftest_stretch selftest_specialists selftest_compositional selftest_evidence selftest_reasoning
uv run python selftest_evaluators.py
uv run python selftest_challenges.py
uv run python selftest_run_queue.py
uv run python selftest_calibration.py
uv run python selftest_sweep.py
uv run python selftest_reconstruction.py
uv run python -m unittest selftest_discrimination selftest_measurement_audit selftest_standard_selection
uv run python -m unittest selftest_repeats selftest_tool_client selftest_tool_conformance selftest_usecase_suites selftest_load_test selftest_ab_studies selftest_monitoring selftest_scorecard selftest_safety_language selftest_run_modes selftest_consolidation
uv run python -m unittest selftest_sessions selftest_constraints selftest_vllm_telemetry
uv run python -m unittest selftest_storage selftest_database_performance
uv run --frozen ruff check .
```

These checks use fake endpoints and temporary databases. Oracle tests derive expected values independently, corrupt answer fields, exercise near-miss programs and check simulation state. Performance tests cover connection reuse, one-attempt timing, failure denominators, cancellation and token/chunk accounting.
