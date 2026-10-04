# LLM Bench

Benchmark OpenAI-compatible endpoints with one fixed question suite and one fixed performance workload. Execution, transport, evaluators, task generators and reports live in the application package under `app/benchmarking`. Models, users and benchmark runs are managed through the web app.

## Run

Python 3.13 or later and uv are required.

```sh
uv sync --frozen
uv run python -m app.main
```

Open http://localhost:8000. Add an endpoint under Models and select **Run benchmark**. Configure **mode** (quality + performance, quality, or performance) and **maximum concurrency** (default 8, range 1–32). Quality uses up to four workers, bounded by that maximum. The same form handles immediate and scheduled runs: leave the start time empty to run now, or set it in your browser's local timezone. Add more runs in that form for an ordered comparison. All submissions share validation, storage and one queue; only one benchmark runs at a time. Manage them under **Queue & schedules**. Editing, cloning and “Run again” use the same submission code and current suite.

The base URL usually ends in /v1; the client appends /chat/completions. Runs offer self-contained HTML downloads, quality JSON, performance JSON and side-by-side comparisons. The run detail has Overview, Performance and Answers views. The compact interface uses a neutral, Shadcn-inspired token theme with responsive navigation; Jinja, Alpine, HTMX and Tailwind remain in place.

## Quality

**Rigorous v14 contains 315 tasks across 30 categories:** 171 revised static anchors, 116 original generated instances, 20 interactive simulations, and eight long-context accuracy checks at 8,192 and 32,768 cl100k_base reference tokens. The static bank has one question per file under [tests/questions](tests/questions); [quality_suite.py](app/benchmarking/quality_suite.py) assembles the complete suite deterministically. Question seeds 19 and 23 and two variants per generated family are fixed. V14 strengthens eight existing probability/retrieval instances and adds eight grid-ambiguity and code-execution instances. Tasks require posterior prediction under shared uncertainty, revision-aware retrieval with missing/deleted/cyclic paths, complete constraint counts and domains after clue deletion, and inverse execution with alias-sensitive program comparisons. Compound quantities, evidence paths and witnesses have explicit atomic diagnostic criteria; strict full success remains the primary metric. Unchanged families retain their established generation version.

Every task requires a full pass. The headline is **strict task success**, averaged first within each family, then within each category, then across capability categories. This prevents many similar variants from dominating. Partial criterion achievement and its coverage are separate diagnostics. Three creative-writing items measure formal compliance separately; they do not measure artistic quality.

JSON checks compare every required value, type and key. New count/receipt contracts require integer literals; general number contracts continue to accept numerically equal integer/float values. Code executes as submitted against boundary and generated-combination fixtures; forbidden imports and changes to protected input state fail. Interactive actions run in local simulations with final-state and authorization gates. Missing requirements, blocked rubric dependencies and fixture exceptions earn no credit. Correct array members retain diagnostic credit when cardinality is wrong; the full-pass contract still fails. Relevant records, revisions and distractors are distributed across the long-context inputs.

All quality calls use temperature 0, model seed 0 and a 65,536-token output cap, including reasoning. This fixed cap preserves headroom for reasoning models. Quality runs allow the full budget without a heuristic repetition cutoff. Truncation and missing final answers stay scored failures. Endpoint failures, unsupported context, evaluator errors and cancellation remain explicit unscored outcomes with coverage and zero-to-one sensitivity bounds. Unstarted tasks remain in the planned denominator after an outage.

Saved reports record suite fingerprints, individual criteria, evaluator versions, client versions, timeout/retry policy and quality worker count. Paired comparisons require compatible protocols and identical question fingerprints. Invalid evaluator scores or diagnostics become unscored evaluator errors; missing required rubric criteria retain zero diagnostic credit. Family bootstrap intervals are unavailable when any category lacks multiple independent families; they do not estimate repeated-run variability.

There are no profiles, filters, partial reruns, seed/split controls, output-budget controls, custom context grids or response-cache options. Historical scores are retained as historical data and do not become v14 scores.

The new research draws on [FEVER](https://fever.ai/dataset/fever.html), [HoVer](https://hover-nlp.github.io/), [BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html), [ComplexBench](https://github.com/thu-coai/ComplexBench) and [BigCodeBench](https://github.com/bigcode-project/bigcodebench), alongside the earlier LiveBench and RULER methods. Tasks are original local exercises; third-party datasets and benchmark scores are not reproduced.

Fresh matched model runs are needed to establish empirical difficulty and model separation. Public question banks can be contaminated. The [v14 design and critical review](BENCHMARK_DESIGN.md) records the new tasks, primary research sources, client effects, fixed grading defects and verification methods. These bounded code functions and text-protocol simulations do not measure complete repository work, native tool APIs or unrestricted real-world agents.

Only complete leading reasoning envelopes are removed before grading. Unfinished scratchpads have no final answer; literal tags within JSON and code remain data. JSON has a portable 256-level nesting limit. Code fixtures reject arbitrary objects that merely print like correct values, preserve ordered members inside sets, and distinguish interpreter-launch failures from submitted-program failures.

## Performance

One workload uses a **1,024-token reference input** and asks for integers 1 through 80 with a **256-token output limit**. Each request has a unique leading identifier to reduce prefix reuse. Input tokenization happens before timing. Temperature and model seed are 0.

Streaming is negotiated before measurement. Every connection is warmed and reused. Timed requests have **one attempt**. Load levels are powers of two up to the configured maximum, including the maximum itself: maximum 6 tests 1, 2, 4 and 6. Each level measures at least 24 requests or four rounds per worker, whichever is larger.

Negotiation requires an explicit unsupported-field error and available retry budget. One-attempt calls preserve the negotiated settings. Completion choices after a terminal finish reason, invalid finish metadata and streams containing only usage or DONE are endpoint errors; trailing usage after a finish reason remains valid.

Reports show end-to-end latency and first-delivered-token p50/p95/p99, request errors and error rate, failed-request latency, aggregate output tokens per second and successful requests per second. V6 also stores timed request samples for latency, TTFT and output-length distributions, output-length summaries, SSE chunk gap summaries, and mean output token time. The latter is a delivery proxy: first-to-last output chunk span divided by reported completion tokens minus one. Buffered bursts, estimated counts and outputs below two tokens are excluded. Chunk intervals are not individual token latency. These definitions follow the distinction between client latency and throughput metrics in [NVIDIA GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2600/user-guide/docs/perf_benchmark/genai-perf-README.html). Rates divide successful output/completions by the full measured wall time, including failed requests, ramp-up and drain. Progress updates and prompt construction stay outside that interval.

TTFT includes first delivered content or reasoning and depends on provider buffering. Available-byte streaming reads avoid an added read-to-fill delay, and complete HTTP bodies return sockets to the connection pool. Malformed or prematurely ended streams count as request errors. Single-chunk or short-burst delivery and estimated token counts are marked explicitly. Provider completion counts may include reasoning; estimates cover all delivered final text and reasoning and may differ from the provider tokenizer. Cached token counts, when supplied, remain in JSON. Missing or malformed usage is estimated at four characters per token; explicit reported zero counts are preserved. Latency includes every attempt and backoff, with successful-attempt latency also retained in JSON. Quality requests use a 180-second inactivity timeout and an independent 1,800-second total stream budget; performance retains the 360-second stream budget. Deadlines and invalid stream protocols stop without retries. Transient transport/HTTP failures retain bounded retries. Future failures retain per-attempt TTFT, delivery counts, reasoning/content character counts and last-delivery timing. The total stream budget is checked at received bytes and EOF; a blocked read can overrun it up to the inactivity timeout. Quality request timing and error concentration are shown separately from the controlled performance suite, including for historical runs. Timed performance calls still make one attempt.

This is a closed-loop concurrency measurement. It does not establish an arrival-rate capacity or infer model-side prefill/decode speed. There are no SLO thresholds, user-capacity estimates, cache probes, mixed workloads or performance context sweeps. Small-sample tail percentiles and different output lengths limit comparisons; compare common load levels and repeat measurements when stability matters.

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
uv run --frozen ruff check .
```

These checks use fake endpoints and temporary databases. Oracle tests derive expected values independently, corrupt answer fields, exercise near-miss programs and check simulation state. Performance tests cover connection reuse, one-attempt timing, failure denominators, cancellation and token/chunk accounting.
