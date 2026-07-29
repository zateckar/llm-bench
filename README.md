# LLM Bench

A web application and CLI suite for benchmarking remote OpenAI-compatible LLM endpoints on **quality** and on **performance**, using curated questions and rule-based evaluation. No second LLM is used as a judge, so results are deterministic and auditable.

Built with **Python 3.13**, **FastAPI**, **HTMX**, **Tailwind CSS** and **SQLite**. Dependencies are managed with **`uv`**; the container image is a hardened multi-stage **Docker** build.

---

## What it measures

### Quality

273 questions across 23 categories, each graded by a rule-based evaluator. Every question declares its own **pass threshold** (1.0 by default), so partial credit never counts as a pass:

| Area | Categories |
|------|-----------|
| Reasoning | Logical Reasoning, Mathematical Reasoning, Reading Comprehension, Classification |
| Knowledge | Factual Knowledge, Truthfulness |
| Code | Code Generation, Advanced Coding, Terminal Algorithms, Terminal Debugging |
| Long context | Needle Retrieval, Long Context Coherence, Summarization |
| Agentic | Tool Using, Agentic Use Cases |
| Ops & security | Security, Terminal System Admin, Terminal File Operations, Terminal Science |
| Language & style | Instruction Following, Creative Writing, Translation, Ethical Reasoning |

Each question carries a **difficulty** tier (`easy` / `medium` / `hard` / `expert`) that feeds a difficulty-weighted score alongside the raw average, so a model that only clears the easy items cannot hide behind a flat percentage.

Roughly 40% of items execute the model's code in a sandboxed subprocess and compare returned values structurally, rather than pattern-matching prose about the code.

### Performance

Latency and throughput are recorded for every question, and an optional dedicated performance suite measures the things that fail independently of each other:

- **Latency distribution** — p50/p90/p95/p99 and max, end to end, plus **time to first token** when the endpoint supports streaming. Reported as a distribution rather than a mean, because tail latency is what users notice.
- **Decode throughput** — output tokens per second on a single stream, measured *after* the first token so prefill and queueing do not flatter the number.
- **Prefill throughput** — prompt tokens per second, estimated from TTFT on a deliberately long prompt with a 1-token generation cap.
- **Concurrency and capacity** — a sweep over concurrency levels (**1 to 256** in flight) reporting aggregate requests/s and tokens/s, latency degradation, error rate under load, the **saturation point** (the level past which extra concurrency buys latency, not throughput) and **scaling efficiency** against linear.

Each level fires at least two requests per worker slot, so the measured window contains a steady state rather than only ramp-up and drain. Above 64 in flight the load generator's own thread scheduling and socket handling start to contribute to the measured latency; those rows are marked `*` in the report and should be read as a **lower bound** on the endpoint's capacity. To push higher, run the sweep from a host close to the endpoint or from several hosts at once.

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
| `--no-cache` | ignore the response cache and re-query |
| `--report path.md` | where to write the markdown report |

The run writes `report.md`, and `report.perf.json` when the performance suite ran.

Responses are cached in `.benchmark_cache.json`, keyed by a fingerprint of the prompt, evaluator, expected value, pass threshold and decoding parameters — so editing a question invalidates its cached result automatically, and re-scoring after an evaluator fix costs no tokens.

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
```

- **No false passes.** Every evaluator has at least one adversarial case — a plausible-looking wrong answer — that must score below the pass bar. Each corresponds to a real scoring defect: a number that appears only as an intermediate step, half the required keywords, required steps in the wrong order, a correct label under the wrong item number, a response that hedges and then fabricates specifics, a review that finds most issues and then declares the code secure.
- **No false failures.** Every evaluator has correct answers that must score 1.0, including awkward-but-valid forms: an `int` where the fixture says `float`, a tuple where it says a list, a dict with different key ordering, and a reasoning model's `<think>` block preceding the real answer.
- **End to end.** A hand-written ideal answer and a hand-written lazy answer are run through real questions from `tests/`, and must pass and fail respectively.

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

Run `validate_suite.py --strict` after editing; it catches the classes of fixture bug that would otherwise show up as a model failure.

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

### 2. Build and start

```bash
docker compose up -d --build
```

### Security architecture

- **SSRF protection (URL guard):** DNS-resolution-based validation blocks outbound requests to private, loopback and link-local addresses when a model's base URL is configured. Set `ALLOW_PRIVATE_ENDPOINTS=true` only for deliberately internal endpoints.
- **Sandboxed code execution:** model-generated code runs in a separate short-lived subprocess with a wall-clock timeout, a per-call time limit, an import allowlist, a stripped-down builtins set and best-effort memory/file-size limits. It is not a boundary against a determined attacker — run it inside a container as well, which the provided image does.
- **Reverse-proxy header trust:** Uvicorn starts with `--proxy-headers` and `--forwarded-allow-ips=*` so standard `X-Forwarded-*` headers are respected.
- **User separation:** the container drops privileges and runs as the non-root user `llmbench`.
- **Response headers:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`.
- **CSRF defence:** SameSite=strict session cookies plus origin/referer verification on state-changing requests.

---

## Interpreting a run

- **Average score** is the mean per-question score; **weighted** applies the difficulty weights (easy 1.0, medium 1.5, hard 2.0, expert 3.0).
- **Passed** counts only questions that reached their own threshold. It is not the same as "scored above 50%".
- **Request errors** are transport failures. They are excluded from every percentage — a 502 from the endpoint is not evidence about the model — and are listed separately so they can be re-run.
- **Latency from the question run** includes server-side queueing when more than one worker was used. Compare across runs only at the same worker count, or use the performance suite, which always measures single-stream latency at concurrency 1 before sweeping upward.
