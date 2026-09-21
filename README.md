# LLM Bench

A web application and CLI suite for benchmarking remote OpenAI-compatible LLM endpoints on **quality** and on **performance**, using curated questions and rule-based evaluation. No second LLM is used as a judge, so results are deterministic and auditable.

Built with **Python 3.13**, **FastAPI**, **HTMX**, **Tailwind CSS** and **SQLite**. Dependencies are managed with **`uv`**; the container image is a hardened multi-stage **Docker** build.

---

## What it measures

### Quality

The **reliability-v4** revision adds 46 verified tasks across the original 23 static categories, corrects database-discovered grading defects, and separates 124 legacy prose-pattern diagnostics from capability scores. Quality output budgets now default to 65,536 tokens uniformly across models, including reasoning. See [the database audit](QUALITY_AUDIT.md) for evidence, limitations, and migration details. Historical grades remain unchanged; the revised suite requires fresh matched runs.

**Specialist-v1** adds 30 tasks: Legal (4, including patents and trademarks), Finances (4, including Treasury), R&D (4, covering automotive, mechanical design and robotics), Language Translations (6, all directions between Czech, English and German), Code Review (4 repository snapshots), and 8 additional complex Security reviews. See [specialist coverage and grading](tests/SPECIALISTS.md).

**Ceiling-v5** fixes six false failures found in the latest four runs and adds 52 compositional challenges across 13 categories, including 215 code fixtures. It reports the challenge subset separately and groups related variants into families. See [the latest audit and benchmark design](QUALITY_AUDIT_V5.md). JSON equivalences are explicitly scoped to individual fields; incorrect values, missing actions and truncated answers still fail.

**Stretch-v6** adds 30 original tasks in ten categories that still showed ceilings in runs 48–50: adaptive minimax planning, dependency-constrained tool selection, bitemporal evidence, event reconciliation, revised graph paths, nested configuration state, circuit fault isolation, linked inheritance, ordered transformations, and translation of logical scope. The UUID instruction grader now checks every field. A separate stretch score appears in JSON, Markdown, web diagnostics, and HTML exports. See [the latest database review and design](BENCHMARK_REVIEW_V6.md).

427 static questions across 28 categories, plus seeded reasoning and interactive tasks (434 questions by default). Every question declares its own **pass threshold** (1.0 by default), so partial credit never counts as a pass:

| Area | Categories |
|------|-----------|
| Reasoning | Logical Reasoning, Mathematical Reasoning, Reading Comprehension, Classification |
| Knowledge | Factual Knowledge, Truthfulness |
| Code | Code Generation, Advanced Coding, Code Review, Terminal Algorithms, Terminal Debugging |
| Long context | Needle Retrieval, Long Context Coherence, Summarization |
| Agentic | Tool Using, Agentic Use Cases |
| Ops & security | Security, Terminal System Admin, Terminal File Operations, Terminal Science |
| Language & style | Instruction Following, Creative Writing, Translation, Language Translations, Ethical Reasoning |
| Specialist domains | Legal, Finances, R&D |

Each question carries a **difficulty** tier (`easy` / `medium` / `hard` / `expert`) that feeds a difficulty-weighted score alongside the raw average, so a model that only clears the easy items cannot hide behind a flat percentage.

66 items execute the model's code in a sandboxed subprocess and compare returned values structurally, rather than pattern-matching prose about the code.

The **challenge-v2** revision replaces 106 familiar puzzles, routine calculations, and generic planning questions with 56 harder questions across logical reasoning, mathematics, advanced coding, agentic use cases, tool use, and reading comprehension. Tasks include constrained optimization, exact conditional probabilities, transaction replay, concurrent-update recovery, and code with boundary and tie-breaking requirements. All requested JSON fields must match; code must pass every fixture. All quality questions use the run's uniform output cap (currently 65,536 tokens by default).

Difficulty tiers are author estimates, not measured model rankings. Compare models on the same suite hash and decoding settings; scores from before this revision are not directly comparable. Establishing how well the new questions separate particular models requires fresh runs of those models.

Quality runs now add four generated problem families (assignment optimization, constrained binary strings, selective-report probability, and Boolean constraints) and three interactive environments (concurrent document updates, uncertain payment outcomes, and paginated deletion previews). They vary deterministically by **question seed**, variant index, and development/evaluation split. The evaluation split generates different instances; its public generator is not a secret test set. The fixed YAML questions remain shared anchors.

Advanced coding receives 197 additional deterministic cases by default, including larger inputs, boundary conditions, and randomized combinations. Independent implementations and metamorphic checks verify the fixtures. Interactive models issue one JSON action per turn and receive the simulated result before choosing the next action. Scoring checks final state and authorization, while reporting excess calls separately. These environments never access real files, payments, or services. They test interaction through a portable text protocol, not native provider function-calling syntax.

Optional **long-context quality** tasks measure answer accuracy at exact `cl100k_base` reference-token lengths. Relevant records occur near the beginning, middle, and end, mixed with drafts, older revisions, near-matching identifiers, and unrelated records. Reports show provider-reported prompt-token counts separately: different tokenizers need not agree. Unsupported context sizes are recorded separately from incorrect answers. This is independent of the performance context sweep.

### Performance

Latency and throughput are recorded for every question, and an optional dedicated performance suite measures the things that fail independently of each other:

- **Latency distribution** — p50/p90/p95/p99 and max, end to end, plus **time to first token** when the endpoint supports streaming. Reported as a distribution rather than a mean, because tail latency is what users notice.
- **Decode throughput** — output tokens per second on a single stream, measured *after* the first token so prefill and queueing do not flatter the number.
- **Prefill throughput** — prompt tokens per second, estimated from TTFT on a deliberately long prompt with a 1-token generation cap.
- **Concurrency and capacity** — an **open-loop** sweep over concurrency levels (**1 to 256** in flight): requests are scheduled to arrive on a wall-clock cadence rather than one worker waiting on the previous request, so a server that falls behind is *measured* falling behind (a closed loop self-throttles and hides the collapse). Each level reports aggregate requests/s and tokens/s, **per-stream decode tok/s**, p95 TTFT, error rate, and an **SLO verdict**. The suite then bisects the gap between the highest passing and lowest failing level and reports an **SLO capacity** ("meets SLO up to concurrency N"), an estimated number of **active users** (Little's law from the knee's request rate and latency plus a configurable requests-per-user-hour), the **saturation point**, and **scaling efficiency** against linear. Defaults: p95 TTFT ≤ 2 s, per-stream decode ≥ 15 tok/s, errors ≤ 1%, 60 req/user/hour — all `PerfConfig` knobs.
- **Context scalability** — a sweep over prompt sizes from **32k to 1M tokens** measuring TTFT, latency and prefill tok/s at each size. Passing several levels to `--context-concurrency` (e.g. `1,4,8`) measures the full **context × concurrency grid**: one point per (size, level), so you can see whether prefill cost grows linearly with context and whether that growth worsens under load. To bound token spend, sizes ≥ 192k clamp their concurrency to 4 by default, and a size the endpoint refuses outright (context-window error) is recorded as `skipped` instead of being dragged through guaranteed failures. The web UI draws one TTFT / prefill-tok/s line per concurrency level against context size.

Each scheduled level fires enough probes to cover a few nominal request durations (at least two per arrival slot), so the measured window contains a steady state rather than only ramp-up and drain. Above 64 in flight the load generator's own thread scheduling and socket handling start to contribute to the measured latency; those rows are marked `*` in the report and should be read as a **lower bound** on the endpoint's capacity. To push higher, run the sweep from a host close to the endpoint or from several hosts at once.

---

## Getting started

### Prerequisites

- [Python 3.13+](https://www.python.org/)
- [uv](https://github.com/astral-sh/uv)

### Installation

```bash
git clone <repository-url>
cd llm-bench
cp .env.example .env
uv sync
```

### Running the web application

```bash
uv run uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The admin credentials come from your `.env` (defaults to `admin` / `changeme` in development).

From the admin UI you can filter by category and difficulty, choose how many questions to run concurrently, and optionally run the performance suite as part of the same run. Results, latency percentiles, throughput, the concurrency sweep and a difficulty breakdown all appear on the run detail page, and runs can be compared side by side on both quality and performance.

### Downloading and comparing reports

Click **Download HTML** on a run page, or select two or more runs on **Compare Runs** and click **Download comparison HTML**. Each download is one self-contained file with embedded SVG charts and CSS. It opens offline without JavaScript, fonts, or other external assets and includes printable summaries, quality diagnostics, provenance, and expandable saved prompts, responses, and evaluator details.

Comparisons include question-phase latency and throughput plus dedicated performance measurements: single-stream decode, prefill, peak output and request rates, SLO capacity, cache effects, concurrency curves, and context scaling. The online comparison and downloaded reports share the same performance tables and charts. Missing measurements remain `n/a`; measured zeroes remain zero. Different workloads, worker counts, cache conditions, and tested concurrency ranges can affect comparisons.

Authenticated download endpoints are `/runs/{id}/report.html` and `/compare/report.html?runs=1&runs=2` (comma-separated IDs also work). Older runs can be exported; unavailable diagnostics are omitted or marked `n/a`. Active or failed runs are clearly labeled as partial snapshots. Exports contain saved benchmark content but omit model configuration credentials and endpoint URLs.

### Running the CLI benchmark

```bash
uv run python benchmark.py
```

```bash
uv run python benchmark.py --perf --concurrency 1,16,64,256
```

A sweep at 256 holds 256 sockets open and fires 512 requests at that level alone, so check the host's file-descriptor limit (`ulimit -n`) before running it.

Useful options:

| Option | Effect |
|--------|--------|
| `--category "Security"` | run one category |
| `--difficulty hard` | run one difficulty tier |
| `--limit N` | run at most N questions |
| `--workers N` | run N questions concurrently |
| `--perf` | also run the performance suite |
| `--perf-only` | run only the performance suite |
| `--concurrency 1,16,64,256` | concurrency levels for the sweep, each 1–256 |
| `--perf-requests N` | minimum requests per concurrency level (default 8) |
| `--no-cache` | re-query the model; the cache file is neither read nor modified |
| `--report path.md` | where to write the markdown report |
| `--suite-seeds 19,23` | reproducible question seeds; distinct from the model's decoding seed |
| `--suite-split evaluation` | use the evaluation generator stream (default `development`) |
| `--variants 3` | variants per generated family and seed (1–10) |
| `--quality-profile hardening` | append the independently verified hardening-v7 candidate profile; `hardening-only` runs candidates without v6 anchors |
| `--static-only` | disable generated reasoning, interactive tasks, and extra code fixtures |
| `--quality-max-tokens 65536` | uniform quality output cap including reasoning; 1,024–262,144. Reasoning models spend most of it inside `<think>`, so keep real headroom: a cap that bites scores itself, not the model. Changing it does not change the suite hash. |
| `--quality-context-sizes 8192,32768,131072` | add answer-accuracy tests at these reference context sizes |
| `--input-price 2 --output-price 8` | estimate USD cost using supplied per-million-token rates |

The run writes `report.md`, `report.quality.json` for quality runs, and `report.perf.json` when the performance suite ran. Interactive tasks always use fresh environments and bypass the response cache.

For example, run reproducible evaluation variants and long-context quality checks:

```bash
uv run python benchmark.py --suite-split evaluation --suite-seeds 19,23 --variants 2 --quality-context-sizes 8192,32768,131072 --no-cache
```

Long-context sizes are optional because they substantially increase prompt-token spend. `tiktoken` downloads its hash-verified reference vocabulary on first use; `TIKTOKEN_CACHE_DIR` can point to a prepopulated cache. Docker images preload it during the build, so production runs need no vocabulary download.

Compare two saved reports without model calls:

```bash
uv run python quality_report.py model-a.quality.json model-b.quality.json
```

The web run form exposes the same seeds, split, variants, context sizes, and pricing. Run detail pages show diagnostics and a downloadable quality JSON; the comparison page shows capability alongside cost and latency plus paired differences.

Responses are cached in `.benchmark_cache.json`, keyed by a fingerprint of the endpoint, prompt, evaluator, expected value, pass threshold and decoding parameters — so editing a question, or pointing the same model name at a different endpoint, invalidates its cached result automatically, and re-scoring after an evaluator fix costs no tokens.

Configure the endpoint through the environment:

```ini
OPENAI_BASE_URL=https://api.example.com/v1
OPENAI_KEY=...
OPENAI_MODEL=model-name
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0        # deterministic by default, for reproducible runs
OPENAI_SEED=                # optional
OPENAI_STREAM=true          # streaming enables time-to-first-token measurement
OPENAI_TIMEOUT=180
```

---

## Trusting the results

A benchmark is only as good as its grading, so both the fixtures and the evaluators are gated in CI.

**`validate_suite.py`** statically checks the whole suite and fails on the mistakes that silently corrupt results rather than raising an error:

```bash
uv run python validate_suite.py --strict
```

- regexes that do not compile, or that match the empty string (and are therefore satisfied by any response);
- `expected` values whose shape does not match the chosen evaluator;
- a bare `contains_keywords` list that is really a list of alternatives — `["bell", "alexander graham bell"]` under all-of semantics;
- `set_match` decoys that overlap the required items, which would fail a correct answer;
- `code_exec` fixtures with no expected value, an invalid function name or an unknown harness;
- `mcq` answers outside the option set, and multiple-choice items guessable enough that a coin flip passes.

**`selftest_evaluators.py`** asserts that the grading behaves as specified, in both directions:

```bash
uv run python selftest_evaluators.py
uv run python selftest_reliability.py
uv run python selftest_specialists.py
uv run python selftest_ceiling.py
uv run python selftest_stretch.py
```

- **No false passes.** Every evaluator has at least one adversarial case — a plausible-looking wrong answer — that must score below the pass bar. Each corresponds to a real scoring defect: a number that appears only as an intermediate step, half the required keywords, required steps in the wrong order, a correct label under the wrong item number, a response that hedges and then fabricates specifics, a review that finds most issues and then declares the code secure.
- **No false failures.** Every evaluator has correct answers that must score 1.0, including awkward-but-valid forms: an `int` where the fixture says `float`, a tuple where it says a list, a dict with different key ordering, and a reasoning model's `<think>` block preceding the real answer.
- **End to end.** A hand-written ideal answer and a hand-written lazy answer are run through real questions from `tests/`, and must pass and fail respectively.

**`selftest_challenges.py`** verifies all 56 challenge questions against executable
answer derivations and reviewed policy traces. It accepts correct solutions and
rejects answers with a single wrong or missing field, reordered results, and code
with deliberately broken boundary handling. This check also runs in CI:

```bash
uv run python selftest_challenges.py
```

**`selftest_quality.py`** tests generator reproducibility and separate evaluation streams, independent answer derivations, differential code oracles, metamorphic properties, exact long-context sizes, adaptive tool agents, failed and unauthorized actions, output truncation, SQLite storage, and complete CLI/web runner agreement using fake clients. No live model endpoint is needed:

```bash
uv run python selftest_quality.py
```

**`selftest_granular.py`** checks the versioned criterion-level diagnostics used by
new quality reports. Structured JSON answers expose individual paths, code answers
expose fixture-level results, formatting checks expose one criterion per check, and
legacy evaluators retain an explicit single-criterion fallback. Full-task pass/fail
and the existing capability headline are unchanged; criterion achievement is a
diagnostic partial measure:

```bash
uv run python selftest_granular.py
```

**`difficulty_calibration.py`** reads saved completed reports in a read-only
transaction and identifies saturated categories, under-covered families, and
coverage gaps. It never changes difficulty labels or historical scores:

```bash
uv run python difficulty_calibration.py data/bench.db --runs 54 55 56 57 58 \
  --output data/difficulty-calibration-2026-09-21.json
```

**Hardening-v7** is an explicit profile, separate from the frozen v6 default.
It contains two deterministic candidate families in each of the 29 categories,
two development instances per family (116 candidate cases), and simulator-backed
interactive races and permission replanning. Every static case has an independent
derivation, executable code fixtures where applicable, and adversarial mutation
checks. The profile is not selected for release until a matched pilot is available.

```bash
uv run python selftest_hardening.py
uv run python selftest_hardening_release.py
uv run python hardening_release.py \
  --split development --variants 2 \
  --output data/hardening-release-gate-2026-09-21.json
uv run python benchmark.py --quality-profile hardening-only \
  --suite-split evaluation --variants 2 --no-cache
```

The release gate uses a provisional 20–80% full-pass selection band, requires
three matched reports and coverage of all 29 categories, and keeps runtime
failures separate from content difficulty. A run without supplied pilot reports
is recorded as `pending_model_pilot`; it does not change the default suite.

**`selftest_reports.py`** verifies HTML downloads, offline assets, authentication, comparison metrics, legacy/partial runs, excluded answers, skipped context measurements, and escaping of untrusted content using a temporary database:

```bash
uv run python selftest_reports.py
```

**`review_queue.py`** turns an audit artifact into an answer-free, deterministic
queue containing every scored capability non-pass and runtime failure, plus a
small success sample for each populated category/evaluator group. It preserves
replay coverage and marks the queue as requiring human disposition; it never
changes a historical grade. The current review is documented in
[`QUALITY_REVIEW_2026-09-21.md`](QUALITY_REVIEW_2026-09-21.md).

### Evaluation methodology & known limitations

Scoring is entirely rule-based — there is no second LLM acting as a judge. That makes grades reproducible and free of judge bias, but it also means grading is syntactic, not semantic. The known limits, kept deliberately because tightening them would risk more unfair *failures* than unfair passes:

- **Negation blindness.** `contains_keywords`, `none:` lists, and `not_contains` checks match substrings/words, not meaning: "the capital is **not** Paris" still counts as mentioning Paris. Such errors can affect models differently. In reliability-v4 these prose-pattern tasks are diagnostic only and excluded from the capability headline.
- **`numeric_set` matches anywhere.** Required values are searched in the whole response, so a value appearing as an intermediate step counts. `numeric_match` is the strict variant and grades only the asserted final answer.
- **Heuristic final-answer extraction.** `numeric_match` uses ordered cues ("answer/result/total …", last line, last number). A response that asserts the wrong number last is graded as wrong even if the right one appeared earlier — by design: the final assertion is what a reader would take away.
- **MCQ strictness.** Hedging across options scores 0; a prose answer with no recognisable option letter scores 0. Fallback letter-scanning only activates when an answer cue is present, so prose articles ("a", "I") are not mistaken for option letters.
- **Sandbox is not a hard security boundary.** `code_exec` strips non-allowlisted imports and runs with resource limits in a subprocess, but the allowlist includes `os`/`requests`-adjacent modules; run untrusted models at your own risk. Verdicts are also environment-dependent: an allow-listed import that is not installed in the sandbox interpreter fails the question.
- **Integer rounding window.** An integer expected value (a count) accepts a more precise answer that genuinely rounds to it (±0.5): 533.33 passes for 533, but 532 and 533.99 do not.

### Code quality

```bash
uv run ruff check .
```

```bash
uv run ruff format .
```

---

## Adding or editing questions

Questions live in `tests/*.yaml`, one file per category:

```yaml
- id: LR-01
  category: Logical Reasoning
  difficulty: hard                  # easy | medium | hard | expert
  description: "what this item probes"
  prompt: |
    ...
  evaluator: numeric_match
  expected: 17.0
  # pass_threshold: 1.0             # lower it only where partial credit is meaningful
  # rubric:                         # optional named diagnostic criteria
  #   - id: final-value
  #     dimension: content
  #     weight: 1
  #     critical: true
```

Available evaluators:

| Evaluator | Grades |
|-----------|--------|
| `exact_match` | whole normalized response equals the expected answer |
| `mcq` | a single option letter; naming several options scores zero |
| `contains_keywords` | `all` / `any` / `groups` / `n_of` / `none` keyword requirements |
| `numeric_match` | the model's final asserted number, within the precision the prompt asks for |
| `numeric_set` | every expected number appears, in any order |
| `code_exec` | the model's code is executed against fixtures and compared structurally |
| `format_check` | formatting and instruction constraints (counts, JSON structure, table shape, allowed vocabulary, regexes) |
| `json_match` | deep comparison against an expected JSON document, exact or subset |
| `ordered_labels` | an enumerated answer, graded item by item with explicit wrong-label rejection |
| `set_match` | exactly the required items and none of the listed decoys |
| `regex_all` | every pattern must match; any `must_not` pattern fails the item |
| `refusal_calibration` | declines to answer AND does not fabricate specifics |
| `security_analysis` | required findings present; a disqualifying claim scores zero |
| `command_correctness` | the specific commands and flags, with `forbidden` entries failing the item |
| `multi_step_solution` | all required steps, in order |
| `file_content_match` | both the filename and the required content |

New questions may add a `rubric` list with stable criterion IDs, dimensions,
weights, mandatory/critical gates, and dependencies. Evaluators use matching IDs
when they can expose field or fixture evidence; legacy evaluators retain one
explicit task criterion until upgraded. Rubrics explain partial achievement and
never lower the question's full-pass threshold.

Run `validate_suite.py --strict` after editing; it catches the classes of fixture bug that would otherwise show up as a model failure.

The empirical calibration and hardening tools keep the next difficulty revision
separate from the frozen suite. Re-run calibration on matched completed runs,
then build the candidate-family backlog:

```bash
uv run python difficulty_calibration.py data/bench.db --runs 54 55 56 57 58 \
  --output data/difficulty-calibration-2026-09-21.json
uv run python difficulty_hardening.py \
  --calibration data/difficulty-calibration-2026-09-21.json \
  --output data/difficulty-hardening-backlog-2026-09-21.json
```

The backlog contains two independently reviewable family specifications for
each objective category. It is not benchmark content until an independent
oracle, adversarial mutations, and a matched pilot validate each family.

---

## Deployment with Docker Compose

LLM Bench is pre-configured to run behind a reverse proxy (Nginx, Traefik, Caddy) with a WAF.

### 1. Configure production environment variables

In production mode the application refuses to start with missing or default secrets:

```ini
ENVIRONMENT=production
SECRET_KEY=generate-a-strong-random-key-here
ADMIN_PASSWORD=generate-a-strong-unique-password-here
```

The container serves plain HTTP, so if you access it directly (no HTTPS
reverse proxy in front) leave `COOKIE_SECURE` at `false` — with it `true`
browsers refuse the session cookie and login silently loops back to /login.

### 2. Pull and start

Use the production compose file, which pulls the pre-built CI image and does
not bind-mount a host `tests/` directory:

```bash
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

The test suite ships inside the image, so deploying a newer image is all it
takes to update the questions — repeat the two commands above.

> **Note:** the default `docker-compose.yml` is for development. It bind-mounts
> `./tests` over the suite baked into the image, so in production it would keep
> serving whatever (possibly stale) files sit next to the compose file on the
> host, even after pulling a newer image.

### Security architecture

- **SSRF protection (URL guard):** DNS-resolution-based validation blocks outbound requests to private, loopback and link-local addresses when a model's base URL is configured. Set `ALLOW_PRIVATE_ENDPOINTS=true` only for deliberately internal endpoints.
- **Sandboxed code execution:** model-generated code runs in a separate short-lived subprocess with a wall-clock timeout, a per-call time limit, an import allowlist, a stripped-down builtins set and best-effort memory/file-size limits. It is not a boundary against a determined attacker — run it inside a container as well, which the provided image does.
- **Reverse-proxy header trust:** Uvicorn starts with `--proxy-headers` and `--forwarded-allow-ips=*` so standard `X-Forwarded-*` headers are respected.
- **User separation:** the container drops privileges and runs as the non-root user `llmbench`.
- **Response headers:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`.
- **CSRF defence:** SameSite=strict session cookies plus origin/referer verification on state-changing requests.

---

## Interpreting a run

- **Category/family-balanced capability** excludes heuristic prose-pattern diagnostics and gives each capability category equal weight and each question family equal weight within its category. Generated variants are averaged within their family. Creative-writing constraint compliance is reported separately and does not claim to measure artistic quality. The legacy all-item average and difficulty-weighted scores remain available.
- **95% intervals** use a deterministic 1,000-draw bootstrap of families within fixed categories. Variants stay clustered; intervals require at least two families in every included category. These describe sampled-item uncertainty, not model run-to-run variability or guaranteed generalization. Paired comparisons use only identical question fingerprints answered by both models, reject mismatched decoding protocols, and disclose unmatched and excluded items.
- **Failure diagnostics** distinguish incorrect task results, formatting failures, creative-writing compliance failures, output truncation, degenerate repetition, endpoint errors, unsupported context, evaluator errors, and cancellation. Formatting, truncation and repetition remain scored failures. Endpoint/unsupported/evaluator/cancelled items are excluded, with coverage shown; a missing answer never becomes a pass.
- **Degenerate repetition** is detected while the answer streams: if the fraction of distinct word 8-grams in the last 4,000 characters falls below 0.30, the generation is abandoned and scored zero as `repetition`. It arms only past 8,000 characters, so short structured output cannot trip it. Reported separately from truncation because the two imply different fixes — a truncation may mean the output budget was too small, a loop never does. Calibrated on a 404-question run: every one of the 326 answers that finished scored above 0.85, so the threshold sits more than 3x below the closest legitimate answer. Quality runs only; the performance suite measures decode rate on deliberately repetitive prompts and never has a generation cut short.
- **Cost estimates** require both user-supplied rates and include fresh quality-call usage only. They exclude unknown billing, unreported failed-request usage, performance-suite usage, and provider cache discounts. Interactive task latency is the sum of its model-call latencies; call counts and excess calls are separate metrics.
- **Average score** is the mean per-question score; **weighted** applies the difficulty weights (easy 1.0, medium 1.5, hard 2.0, expert 3.0).
- **Criterion achievement** is a diagnostic weighted mean of the content/task requirements the evaluator can identify (JSON paths, code fixtures, or one explicit legacy task criterion). Contract and availability checks are excluded from that mean and shown separately as **contract compliance** when measurable. It explains near misses and does not turn a failed full task into a pass.
- **Passed** counts only questions that reached their own threshold. It is not the same as "scored above 50%".
- **Request errors** are transport failures. They are excluded from every percentage — a 502 from the endpoint is not evidence about the model — and are listed separately so they can be re-run.
- **Latency from the question run** includes server-side queueing when more than one worker was used. Compare across runs only at the same worker count, or use the performance suite, which always measures single-stream latency at concurrency 1 before sweeping upward.
