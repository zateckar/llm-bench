# LLM Bench

Benchmark OpenAI-compatible endpoints with one fixed question suite and one fixed performance workload. The CLI and web app share the same execution and reporting code.

## Run

Python 3.13 or later and uv are required.

```sh
uv sync --frozen
uv run python -m app.main
```

Open http://localhost:8000. Add an endpoint under Models and select **Run benchmark**. Configure **mode** (quality + performance, quality, or performance) and **maximum concurrency** (default 8, range 1–32). Quality uses up to four workers, bounded by that maximum. The same form handles immediate and scheduled runs: leave the start time empty to run now, or set it in your browser's local timezone. Add more runs in that form for an ordered comparison. All submissions share validation, storage and one queue; only one benchmark runs at a time. Manage them under **Queue & schedules**. Editing, cloning and “Run again” use the same submission code and current suite.

The CLI reads OPENAI_BASE_URL, OPENAI_KEY and OPENAI_MODEL from the environment or .env. The base URL usually ends in /v1; the client appends /chat/completions.

```sh
uv run python benchmark.py
uv run python benchmark.py --mode quality --report quality.md
uv run python benchmark.py --mode performance --max-concurrency 4 --report performance.md
```

The default runs both phases. Reports contain Markdown and a companion JSON file. Web runs also offer self-contained HTML downloads and side-by-side comparisons. “Run again” runs the complete current suite.

## Quality

**Rigorous v10 contains 275 tasks across 30 categories:** 171 revised static anchors, 76 original generated instances, 20 interactive simulations, and eight long-context accuracy checks at 8,192 and 32,768 cl100k_base reference tokens. The static bank has one question per file under [tests/questions](tests/questions); [quality_suite.py](quality_suite.py) assembles the complete suite deterministically. Question seeds 19 and 23 and two variants per generated family are fixed.

Every task requires a full pass. The headline is **strict task success**, averaged first within each family, then within each category, then across capability categories. This prevents many similar variants from dominating. Partial criterion achievement and its coverage are separate diagnostics. Three creative-writing items measure formal compliance separately; they do not measure artistic quality.

JSON checks compare every required value, type and key. Code executes as submitted against boundary and generated-combination fixtures; forbidden imports and changes to protected input state fail. Interactive actions run in local simulations with final-state and authorization gates. Missing requirements, blocked rubric dependencies and fixture exceptions earn no credit. Correct array members retain diagnostic credit when cardinality is wrong; the full-pass contract still fails. Relevant records, revisions and distractors are distributed across the long-context inputs.

All quality calls use temperature 0, model seed 0 and a 65,536-token output cap, including reasoning. This fixed cap preserves headroom for reasoning models. Truncation, missing final answers and detected repetition stay scored failures. Endpoint failures, unsupported context, evaluator errors and cancellation remain explicit unscored outcomes with coverage and zero-to-one sensitivity bounds. Unstarted tasks remain in the planned denominator after an outage.

Saved reports record suite fingerprints, individual criteria and evaluator versions. Paired comparisons require compatible protocols and identical question fingerprints. Family bootstrap intervals are unavailable when any category lacks multiple independent families; they do not estimate repeated-run variability.

There are no profiles, filters, partial reruns, seed/split controls, output-budget controls, custom context grids or response-cache options. Historical scores are retained as historical data and do not become v10 scores.

The [review of runs 57, 58, 62 and 65](BENCHMARK_REVIEW_2026-10-03.md) documents the original grading defects and third-party benchmark research. The current design uses ideas from [LiveBench](https://livebench.ai/livebench.pdf), [IFEval](https://arxiv.org/abs/2311.07911), [EvalPlus](https://evalplus.github.io/) and [RULER](https://arxiv.org/abs/2404.06654). It is an original local suite, not an implementation of those benchmarks.

Fresh matched model runs are needed to establish empirical difficulty and model separation. Public question banks can be contaminated. The [v10 design and critical review](BENCHMARK_DESIGN.md) records the new tasks, primary research sources, fixed grading defects and verification methods. These bounded code functions and text-protocol simulations do not measure complete repository work, native tool APIs or unrestricted real-world agents.

## Performance

One workload uses a **1,024-token reference input** and asks for integers 1 through 80 with a **256-token output limit**. Each request has a unique leading identifier to reduce prefix reuse. Input tokenization happens before timing. Temperature and model seed are 0.

Streaming is negotiated before measurement. Every connection is warmed and reused. Timed requests have **one attempt**. Load levels are powers of two up to the configured maximum, including the maximum itself: maximum 6 tests 1, 2, 4 and 6. Each level measures at least 24 requests or four rounds per worker, whichever is larger.

Reports show end-to-end latency and first-delivered-token p50/p95, request errors, failed-request latency, aggregate output tokens per second and successful requests per second. Rates divide successful output/completions by the full measured wall time, including failed requests, ramp-up and drain. Progress updates and prompt construction stay outside that interval.

TTFT includes first delivered content or reasoning and depends on provider buffering. Single-chunk or short-burst delivery and estimated token counts are marked explicitly. Provider completion counts may include reasoning; estimates cover visible output and may differ from the provider tokenizer. Cached token counts, when supplied, remain in JSON. Unreported usage is estimated at four characters per token.

This is a closed-loop concurrency measurement. It does not establish an arrival-rate capacity or infer model-side prefill/decode speed. There are no SLO thresholds, user-capacity estimates, cache probes, mixed workloads or performance context sweeps. Small-sample tail percentiles and different output lengths limit comparisons; compare common load levels and repeat measurements when stability matters.

## Configuration and deployment

Copy [.env.example](.env.example) to .env for local configuration. Development creates an admin account using ADMIN_USERNAME / ADMIN_PASSWORD (defaults admin / changeme). For production, set ENVIRONMENT=production, a unique SECRET_KEY and ADMIN_PASSWORD; startup rejects missing/default values. Models and API keys are managed by administrators. OIDC settings remain optional.

```sh
docker compose up --build -d
```

[docker-compose.yml](docker-compose.yml) requires SECRET_KEY and ADMIN_PASSWORD and persists the SQLite database in a named volume. [docker-compose.prod.yml](docker-compose.prod.yml) adds the production reverse-proxy setup. Keep data and credentials outside source control.

## Verification

```sh
uv run python validate_suite.py --strict
uv run python -m unittest selftest_frontier selftest_granular selftest_rigorous selftest_quality selftest_perf selftest_run_options selftest_benchmark_runner selftest_reports selftest_ceiling selftest_stretch selftest_specialists
uv run python selftest_evaluators.py
uv run python selftest_challenges.py
uv run python selftest_run_queue.py
uvx ruff check .
```

These checks use fake endpoints and temporary databases. Oracle tests derive expected values independently, corrupt answer fields, exercise near-miss programs and check simulation state. Performance tests cover connection reuse, one-attempt timing, failure denominators, cancellation and token/chunk accounting.
