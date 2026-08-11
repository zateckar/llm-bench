#!/usr/bin/env python3
"""LLM Quality, Accuracy & Performance Benchmark.

Benchmarks a remote OpenAI-compatible LLM across multiple quality categories
using curated questions and rule-based evaluation, and measures latency,
throughput and concurrency behaviour of the endpoint. Produces a markdown report.

Questions are loaded from YAML files in the tests/ directory.
Evaluators are defined in evaluators.py.
Performance measurement lives in perf.py.

Usage:
    python benchmark.py                          # quality suite
    python benchmark.py --perf                   # quality suite + performance suite
    python benchmark.py --perf-only              # performance suite only
    python benchmark.py --category "Security" --limit 5
    python benchmark.py --workers 4              # run questions concurrently
    python benchmark.py --concurrency 1,2,4,8,16 # concurrency levels to sweep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from evaluators import EVALUATORS
from llm_client import ChatClient, ClientConfig
from models import (
    CategoryResult,
    ContextPoint,
    LatencyStats,
    PerfReport,
    Question,
    RequestMetrics,
    Result,
    TokenUsage,
)
from perf import (
    MAX_CONCURRENCY,
    ContextSweepConfig,
    PerfConfig,
    format_context_sweep_markdown,
    format_perf_console,
    format_perf_markdown,
    run_context_sweep,
    run_perf_suite,
)
from test_loader import SuiteError, compute_test_suite_hash, load_all_tests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

ROOT = Path(__file__).parent
TESTS_DIR = ROOT / "tests"
CACHE_FILE = ROOT / ".benchmark_cache.json"

BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("OPENAI_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "unknown")

def _env_number(name: str, raw: str, cast) -> int | float:
    """Parse a numeric env var, failing with a readable message instead of a
    raw ValueError traceback at import time."""
    try:
        return cast(raw)
    except ValueError:
        raise SystemExit(
            f"ERROR: invalid value for {name}: {raw!r} — expected a number. "
            "Fix it in your environment or .env file."
        )


MAX_TOKENS = _env_number("OPENAI_MAX_TOKENS", os.getenv("OPENAI_MAX_TOKENS", "4096"), int)
# For a quality benchmark, default to deterministic decoding so results are
# reproducible. Override with OPENAI_TEMPERATURE / OPENAI_SEED if desired.
TEMPERATURE = _env_number("OPENAI_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0"), float)
_SEED_ENV = os.getenv("OPENAI_SEED")
SEED = _env_number("OPENAI_SEED", _SEED_ENV, int) if _SEED_ENV else None
REQUEST_TIMEOUT = _env_number("OPENAI_TIMEOUT", os.getenv("OPENAI_TIMEOUT", "180"), float)
STREAM = os.getenv("OPENAI_STREAM", "true").lower() not in ("0", "false", "no")


def build_client_config() -> ClientConfig:
    return ClientConfig(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        seed=SEED,
        timeout=REQUEST_TIMEOUT,
        stream=STREAM,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logger.warning("Cache file is unreadable (%s); starting from an empty cache", e)
    return {}


def save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError as e:
        # Losing the cache only costs re-running questions; it must not abort
        # an otherwise healthy run.
        logger.warning("Could not write cache file: %s", e)


def _is_valid_cache_entry(cached) -> bool:
    """Validate the on-disk cache schema before use.

    Entries written by older tool versions (e.g. ``text`` instead of
    ``response``, or a raw ``[response, tokens, metrics]`` tuple) would
    otherwise raise KeyError/TypeError inside run_one and be misreported as a
    transport error that never self-heals. Malformed entries are treated as a
    miss instead, which also overwrites them on the next store.
    """
    if not isinstance(cached, dict):
        return False
    if not isinstance(cached.get("response"), str):
        return False
    if not isinstance(cached.get("latency_ms", 0.0), (int, float)):
        return False
    ttft = cached.get("ttft_ms")
    if ttft is not None and not isinstance(ttft, (int, float)):
        return False
    return True


def question_fingerprint(q: Question) -> str:
    """Stable hash of the parts of a question that affect its result.

    Including this in the cache key means editing a prompt, evaluator, expected
    value, threshold or system prompt automatically invalidates the stale cached
    result. Decoding parameters are included too, since they change the output,
    and the endpoint is included so re-pointing the model name at another
    server never replays responses from the old one.
    """
    payload = json.dumps(
        {
            "endpoint": BASE_URL,
            "prompt": q.prompt,
            "system_prompt": q.system_prompt,
            "evaluator": q.evaluator,
            "expected": q.expected,
            "pass_threshold": q.pass_threshold,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_tokens": q.max_tokens or MAX_TOKENS,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def evaluate(q: Question, response: str) -> tuple[float, str]:
    """Score a response, converting evaluator crashes into a visible error."""
    evaluator = EVALUATORS.get(q.evaluator)
    if evaluator is None:
        return 0.0, f"Unknown evaluator: {q.evaluator}"
    try:
        score, detail = evaluator(response, q.expected)
    except Exception as e:  # noqa: BLE001 - a bad fixture must not kill the run
        logger.warning("Evaluator error for %s: %s", q.id, e, exc_info=True)
        return 0.0, f"Evaluator error: {e}"
    return max(0.0, min(1.0, float(score))), detail


def run_one(q: Question, client: ChatClient, cache: dict, persist: bool = True) -> Result:
    """Run (or replay from cache) a single question."""
    cache_key = f"{MODEL}:{q.id}:{question_fingerprint(q)}"
    with _cache_lock:
        cached = cache.get(cache_key)
        if cached is not None and not _is_valid_cache_entry(cached):
            # Stale or malformed entry from an older schema: treat it as a miss
            # and drop it so the next successful store cleans it out.
            logger.warning("Ignoring malformed cache entry for %s", q.id)
            cache.pop(cache_key, None)
            cached = None

    if cached:
        metrics = RequestMetrics(
            latency_ms=cached.get("latency_ms", 0.0),
            ttft_ms=cached.get("ttft_ms"),
            completion_tokens=cached.get("completion_tokens", 0),
            prompt_tokens=cached.get("prompt_tokens", 0),
            ok=cached.get("ok", True),
        )
        # Re-score from the cached response: evaluator fixes take effect without
        # re-spending tokens, and the fingerprint already covers fixture edits.
        score, detail = evaluate(q, cached["response"])
        return Result(
            question=q,
            response=cached["response"],
            score=score,
            detail=detail,
            tokens=TokenUsage(metrics.prompt_tokens, metrics.completion_tokens),
            metrics=metrics,
            cached=True,
        )

    response, tokens, metrics = client.complete(
        q.prompt, q.system_prompt, max_tokens=q.max_tokens
    )

    if metrics.ok:
        score, detail = evaluate(q, response)
    else:
        # A transport failure is not a wrong answer; keep it out of the quality
        # numbers and surface it separately.
        score, detail = 0.0, f"Request failed: {metrics.error}"

    result = Result(
        question=q,
        response=response,
        score=score,
        detail=detail,
        tokens=tokens,
        metrics=metrics,
    )

    if metrics.ok:
        # Persist after each real API call so a mid-run crash keeps prior work.
        # persist=False (--no-cache): private throwaway cache, disk untouched.
        if persist:
            with _cache_lock:
                cache[cache_key] = {
                    "response": response,
                    "score": score,
                    "detail": detail,
                    "prompt_tokens": tokens.prompt_tokens,
                    "completion_tokens": tokens.completion_tokens,
                    "latency_ms": metrics.latency_ms,
                    "ttft_ms": metrics.ttft_ms,
                    "ok": True,
                }
                # Reload+merge entries written by other concurrent benchmark
                # processes since this run loaded the cache, so a plain
                # last-writer-wins write does not silently drop their work.
                # Best effort only: without an interprocess file lock a narrow
                # race window remains (no locking library is used on purpose).
                # Malformed entries are never merged back in, which lets the
                # write clean out stale-schema entries from disk too.
                for key, value in load_cache().items():
                    if _is_valid_cache_entry(value):
                        cache.setdefault(key, value)
                save_cache(cache)

    return result


def _status_line(index: int, total: int, result: Result) -> str:
    q = result.question
    if result.is_transport_error:
        status = "ERROR"
    else:
        status = "PASS" if result.passed else "FAIL"
    bits = [f"[{index}/{total}] {q.category} — {q.id}: {status} ({result.score:.0%})"]
    if result.cached:
        bits.append("cached")
    else:
        bits.append(f"{result.metrics.latency_ms:.0f}ms")
        if result.metrics.ttft_ms is not None:
            bits.append(f"ttft {result.metrics.ttft_ms:.0f}ms")
        bits.append(f"{result.tokens.prompt_tokens}+{result.tokens.completion_tokens} tok")
    return bits[0] + "  [" + " · ".join(bits[1:]) + "]"


def run_benchmark(
    questions: list[Question],
    client_config: ClientConfig,
    workers: int = 1,
    no_cache: bool = False,
) -> tuple[list[CategoryResult], float]:
    """Run all questions and return (categorized results, wall-clock seconds).

    With ``workers > 1`` questions run concurrently. That both shortens the run
    and exercises the endpoint under load, but per-question latency will include
    server-side queueing - use the dedicated perf suite for clean latency numbers.

    With ``no_cache`` the run operates on a throwaway in-memory cache: the cache
    file is neither read nor written, so baseline results stay on disk untouched.
    """
    cache = {} if no_cache else load_cache()
    total = len(questions)
    results: list[Result | None] = [None] * total
    print_lock = threading.Lock()
    started = time.perf_counter()

    shared_client = ChatClient(client_config)
    worker_local = threading.local()
    # Register every created worker client so its session's sockets can be
    # closed once the pool finishes; thread-local clients would otherwise leak
    # one requests.Session per worker thread until process exit.
    worker_clients: list[ChatClient] = []

    def work(index: int, q: Question) -> None:
        try:
            # One client per worker thread, not per question: sessions are not
            # documented as thread-safe, but rebuilding the client per question
            # makes every concurrent question pay TCP+TLS setup and unfairly
            # inflates its recorded latency/TTFT vs. the serial path.
            if workers > 1:
                client = getattr(worker_local, "client", None)
                if client is None:
                    client = ChatClient(client_config)
                    worker_local.client = client
                    with print_lock:
                        worker_clients.append(client)
            else:
                client = shared_client
            result = run_one(q, client, cache, persist=not no_cache)
        except Exception as e:  # noqa: BLE001 - a buggy question must not vanish
            # ChatClient.complete never raises for transport problems, so an
            # exception here is an internal bug. Record it as a failed request
            # (the same convention the client uses) rather than silently
            # dropping the question from the report.
            logger.error("Unexpected error running %s: %s", q.id, e, exc_info=True)
            result = Result(
                question=q,
                response=f"[API ERROR: unexpected: {e}]",
                score=0.0,
                detail=f"Request failed: {e}",
                tokens=TokenUsage(),
                metrics=RequestMetrics(ok=False, error=str(e)),
            )
        results[index] = result
        with print_lock:
            print(_status_line(index + 1, total, result), flush=True)

    if workers > 1:
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for i, q in enumerate(questions):
                    pool.submit(work, i, q)
        finally:
            for client in worker_clients:
                client.session.close()
    else:
        for i, q in enumerate(questions):
            work(i, q)

    elapsed = time.perf_counter() - started

    categories: dict[str, CategoryResult] = {}
    for result in results:
        if result is None:
            continue
        categories.setdefault(result.question.category, CategoryResult(name=result.question.category))
        categories[result.question.category].results.append(result)

    return list(categories.values()), elapsed


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def truncate(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"


def collect_run_latency(categories: list[CategoryResult]) -> tuple[LatencyStats, LatencyStats]:
    """Latency distribution across every freshly executed question."""
    latencies, ttfts = [], []
    for c in categories:
        for r in c.results:
            if r.cached or not r.metrics.ok:
                continue
            if r.metrics.latency_ms:
                latencies.append(r.metrics.latency_ms)
            if r.metrics.ttft_ms is not None:
                ttfts.append(r.metrics.ttft_ms)
    return LatencyStats.from_samples(latencies), LatencyStats.from_samples(ttfts)


def generate_report(
    categories: list[CategoryResult],
    test_suite_hash: str = "",
    elapsed_s: float = 0.0,
    perf: PerfReport | None = None,
    workers: int = 1,
    context_points: list | None = None,
) -> str:
    """Generate a markdown report from results."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    scored = [r for c in categories for r in c.scored]
    all_results = [r for c in categories for r in c.results]
    total_questions = len(all_results)
    total_scored = len(scored)
    total_errors = total_questions - total_scored
    total_passed = sum(1 for r in scored if r.passed)

    avg_score = (sum(r.score for r in scored) / total_scored * 100) if total_scored else 0.0
    total_weight = sum(r.question.effective_weight for r in scored)
    weighted = (
        sum(r.score * r.question.effective_weight for r in scored) / total_weight * 100
        if total_weight else 0.0
    )

    total_prompt = sum(c.tokens.prompt_tokens for c in categories)
    total_completion = sum(c.tokens.completion_tokens for c in categories)
    total_tokens = total_prompt + total_completion

    latency, ttft = collect_run_latency(categories)
    fresh_completion = sum(
        r.tokens.completion_tokens for r in all_results if not r.cached and r.metrics.ok
    )
    run_throughput = fresh_completion / elapsed_s if elapsed_s else 0.0

    lines = [
        f"# LLM Benchmark Report — {MODEL}",
        "",
        f"**Date**: {now}",
        f"**Endpoint**: {BASE_URL}",
        f"**Model**: {MODEL}",
        f"**Temperature**: {TEMPERATURE}" + (f" · **Seed**: {SEED}" if SEED is not None else ""),
        f"**Concurrent workers**: {workers}",
    ]
    if test_suite_hash:
        lines.append(f"**Test Suite Version**: `{test_suite_hash}`")

    # A --perf-only run has no quality results; emit just the performance section
    # rather than a page of zeroes that reads like a total failure.
    if not categories:
        lines += ["", "---", ""]
        if perf is not None:
            lines.append(format_perf_markdown(perf))
        if context_points:
            lines.append(format_context_sweep_markdown(context_points))
        if perf is None and not context_points:
            lines.append("_No questions were run and no performance suite was requested._")
        return "\n".join(lines)

    lines += [
        "",
        "---",
        "",
        f"## Overall: {total_passed}/{total_scored} passed "
        f"({avg_score:.1f}% average score, {weighted:.1f}% difficulty-weighted)",
        "",
        "A question passes only when it reaches its own `pass_threshold` "
        "(1.0 unless stated otherwise), so partial credit never counts as a pass.",
        "",
    ]
    if total_errors:
        lines += [
            f"> **{total_errors} request(s) failed at the transport layer** and are "
            "excluded from the percentages above. They are listed under "
            "*Infrastructure Errors*.",
            "",
        ]

    lines += [
        "## Run Summary",
        "",
        f"- **Wall clock**: {elapsed_s:.1f} s",
        f"- **Total tokens**: {total_tokens:,} ({total_prompt:,} in + {total_completion:,} out)",
        f"- **Effective output throughput for this run**: {run_throughput:.1f} tok/s",
        f"- **Request latency**: p50 {_fmt_ms(latency.p50)} · p95 {_fmt_ms(latency.p95)} "
        f"· p99 {_fmt_ms(latency.p99)} · max {_fmt_ms(latency.max)}",
        f"- **Time to first token**: p50 {_fmt_ms(ttft.p50)} · p95 {_fmt_ms(ttft.p95)}",
        "",
        "## Category Breakdown",
        "",
        "| Category | Passed | Avg Score | Weighted | p50 Latency | Input Tok | Output Tok |",
        "|----------|--------|-----------|----------|-------------|-----------|------------|",
    ]

    for c in sorted(categories, key=lambda x: x.name):
        t = c.tokens
        lines.append(
            f"| {c.name} | {c.passed}/{len(c.scored)} | {c.score_pct:.1f}% | "
            f"{c.weighted_score_pct:.1f}% | {_fmt_ms(c.median_latency_ms)} | "
            f"{t.prompt_tokens:,} | {t.completion_tokens:,} |"
        )

    if perf is not None:
        lines += ["", "---", "", format_perf_markdown(perf)]
    if context_points:
        lines.append(format_context_sweep_markdown(context_points))

    lines += ["---", "", "## Detailed Results", ""]

    for c in sorted(categories, key=lambda x: x.name):
        lines += [
            f"### {c.name}",
            "",
            "| # | ID | Difficulty | Question | Result | Score | Latency | Detail |",
            "|---|----|-----------|----------|--------|-------|---------|--------|",
        ]
        for i, r in enumerate(c.results, 1):
            if r.is_transport_error:
                status = "ERROR"
            else:
                status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"| {i} | {r.question.id} | {r.question.difficulty} | "
                f"{truncate(r.question.prompt, 50)} | {status} | {r.score:.0%} | "
                f"{'cached' if r.cached else _fmt_ms(r.metrics.latency_ms)} | "
                f"{truncate(r.detail, 90)} |"
            )
        lines.append("")

    # Failures, with the actual response, are the useful part of the report.
    failures = [r for r in scored if not r.passed]
    if failures:
        lines += ["---", "", f"## Failures ({len(failures)})", ""]
        for r in failures[:40]:
            lines += [
                f"### {r.question.id} — {r.question.category} ({r.question.difficulty})",
                "",
                f"**Prompt**: {truncate(r.question.prompt, 400)}",
                "",
                f"**Score**: {r.score:.0%} (needed {r.question.pass_threshold:.0%}) — {r.detail}",
                "",
                "**Response**:",
                "```",
                r.response[:800],
                "```",
                "",
            ]
        if len(failures) > 40:
            lines.append(f"_...and {len(failures) - 40} more failures._")
            lines.append("")

    transport_errors = [r for c in categories for r in c.results if r.is_transport_error]
    if transport_errors:
        lines += ["---", "", f"## Infrastructure Errors ({len(transport_errors)})", ""]
        for r in transport_errors:
            lines.append(f"- `{r.question.id}` ({r.question.category}): {r.detail}")
        lines.append("")

    lines += ["---", "", "## Summary & Observations", ""]

    strengths = [c for c in categories if c.score_pct >= 70 and c.scored]
    weaknesses = [c for c in categories if c.score_pct < 50 and c.scored]

    if strengths:
        lines.append("**Strengths**:")
        lines += [f"- {c.name}: {c.score_pct:.1f}%" for c in sorted(strengths, key=lambda c: -c.score_pct)]
        lines.append("")
    if weaknesses:
        lines.append("**Weaknesses**:")
        lines += [f"- {c.name}: {c.score_pct:.1f}%" for c in sorted(weaknesses, key=lambda c: c.score_pct)]
        lines.append("")
    if not strengths and not weaknesses:
        lines += ["- Performance is fairly uniform across categories.", ""]

    by_difficulty: dict[str, list[Result]] = {}
    for r in scored:
        by_difficulty.setdefault(r.question.difficulty, []).append(r)
    if by_difficulty:
        lines += ["**By difficulty**:", ""]
        for tier in ("easy", "medium", "hard", "expert"):
            group = by_difficulty.get(tier)
            if not group:
                continue
            passed = sum(1 for r in group if r.passed)
            lines.append(f"- {tier}: {passed}/{len(group)} passed ({passed / len(group) * 100:.0f}%)")
        lines.append("")

    lines += [
        f"**Total API calls**: {sum(1 for r in all_results if not r.cached)} "
        f"({sum(1 for r in all_results if r.cached)} replayed from cache)",
        f"**Total tokens consumed**: {total_tokens:,} "
        f"({total_prompt:,} input + {total_completion:,} output)",
        "**Evaluation method**: Rule-based scoring (no second LLM)",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an OpenAI-compatible LLM endpoint for quality and performance.",
    )
    parser.add_argument("--category", help="only run questions in this category")
    parser.add_argument("--difficulty", help="only run questions of this difficulty tier")
    parser.add_argument("--limit", type=int, help="run at most N questions")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="run this many questions concurrently (default 1)",
    )
    parser.add_argument(
        "--perf", action="store_true",
        help="also run the performance suite (latency, throughput, concurrency)",
    )
    parser.add_argument(
        "--perf-only", action="store_true",
        help="run only the performance suite and skip the quality questions",
    )
    parser.add_argument(
        "--concurrency", default="1,2,4,8",
        help=f"comma-separated concurrency levels for the sweep, each 1-{MAX_CONCURRENCY} "
             "(default 1,2,4,8)",
    )
    parser.add_argument(
        "--perf-requests", type=int, default=8,
        help="requests per concurrency level (default 8)",
    )
    parser.add_argument(
        "--perf-samples", type=int, default=8,
        help="serial latency samples (default 8)",
    )
    parser.add_argument(
        "--context-sweep", action="store_true",
        help="also measure latency and TTFT at a range of prompt context sizes",
    )
    parser.add_argument(
        "--context-sizes", default="",
        help="comma-separated context sizes in tokens "
             "(default " + ",".join(str(s) for s in ContextSweepConfig().context_sizes) + ")",
    )
    parser.add_argument(
        "--context-concurrency", default="4",
        help="comma-separated context-sweep concurrency levels; sizes below 192k "
             "are measured at every level, larger sizes are clamped to "
             "ContextSweepConfig.max_large_concurrency (default 4)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="ignore the response cache entirely: re-query the model and leave "
             "the cache file untouched",
    )
    parser.add_argument(
        "--report", default="report.md", help="path for the markdown report",
    )
    args = parser.parse_args(argv)
    # 0 would silently mean "no limit" (running the whole suite — the opposite
    # of the intent) and a negative value would silently slice questions off
    # the end; reject both loudly instead.
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be a positive integer, got {args.limit}")
    return args


def _parse_levels(raw: str) -> tuple[int, ...]:
    """Parse --concurrency into bounded, de-duplicated levels.

    Bad input fails loudly here rather than being silently dropped by the suite:
    a typo that halves the sweep would otherwise look like a real result.
    """
    levels = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise SystemExit(f"Invalid concurrency level: {part!r}")
        if not 1 <= value <= MAX_CONCURRENCY:
            raise SystemExit(
                f"Concurrency level {value} is out of range; must be between 1 and "
                f"{MAX_CONCURRENCY}."
            )
        levels.append(value)
    if not levels:
        raise SystemExit("No concurrency levels given")
    return tuple(sorted(set(levels)))


def _parse_context_sizes(raw: str) -> tuple[int, ...]:
    """Parse --context-sizes into sorted de-duplicated token counts.

    An empty string keeps the ContextSweepConfig defaults; malformed entries
    fail loudly like --concurrency does.
    """
    if not raw.strip():
        return ()
    sizes = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise SystemExit(f"Invalid context size: {part!r}")
        if value < 1000:
            raise SystemExit(
                f"Context size {value} is too small to measure usefully; give the size "
                "in tokens, e.g. 32000."
            )
        sizes.append(value)
    return tuple(sorted(set(sizes)))


def main() -> None:
    args = parse_args()

    if not BASE_URL or not API_KEY:
        print("ERROR: OPENAI_BASE_URL and OPENAI_KEY must be set in .env")
        sys.exit(1)

    client_config = build_client_config()
    workers = max(1, args.workers)

    print(f"LLM Benchmark — {MODEL}")
    print(f"Endpoint: {BASE_URL}")

    categories: list[CategoryResult] = []
    elapsed = 0.0
    test_suite_hash = ""

    if not args.perf_only:
        try:
            questions = load_all_tests(TESTS_DIR)
        except SuiteError as e:
            print("ERROR: the test suite is invalid:\n" + str(e))
            sys.exit(2)

        if not questions:
            print("ERROR: No questions found in tests/ directory")
            sys.exit(1)

        if args.category:
            questions = [q for q in questions if q.category.lower() == args.category.lower()]
            if not questions:
                print(f"ERROR: No questions found for category '{args.category}'")
                sys.exit(1)
        if args.difficulty:
            questions = [q for q in questions if q.difficulty == args.difficulty.lower()]
            if not questions:
                print(f"ERROR: No questions found for difficulty '{args.difficulty}'")
                sys.exit(1)
        if args.limit:
            questions = questions[: args.limit]

        test_suite_hash = compute_test_suite_hash(TESTS_DIR)
        print(f"Test Suite Version: {test_suite_hash}")
        print(
            f"Running {len(questions)} questions across "
            f"{len(set(q.category for q in questions))} categories "
            f"with {workers} worker(s)...\n"
        )
        categories, elapsed = run_benchmark(
            questions, client_config, workers=workers, no_cache=args.no_cache
        )

    perf_report: PerfReport | None = None
    if args.perf or args.perf_only:
        perf_config = PerfConfig(
            concurrency_levels=_parse_levels(args.concurrency),
            requests_per_level=args.perf_requests,
            serial_samples=args.perf_samples,
        )
        print(f"\nRunning performance suite (~{perf_config.total_requests()} requests)...")

        def progress(phase: str, done: int, total: int) -> None:
            print(f"  [{done}/{total}] {phase}", end="\r", flush=True)

        perf_report = run_perf_suite(client_config, perf_config, progress=progress)
        print(" " * 60, end="\r")
        print(format_perf_console(perf_report))

    context_points: list[ContextPoint] | None = None
    if args.context_sweep:
        context_levels = _parse_levels(args.context_concurrency)
        sweep_config = ContextSweepConfig(
            context_sizes=_parse_context_sizes(args.context_sizes),
            # A single level means exactly the old single-level sweep; the grid
            # only engages when the operator actually asks for several levels.
            concurrency_per_size=context_levels[0],
            concurrency_levels=context_levels if len(context_levels) > 1 else (),
        )
        print(
            f"\nRunning context scalability sweep "
            f"(~{sweep_config.total_requests()} requests)..."
        )

        def ctx_progress(phase: str, done: int, total: int) -> None:
            print(f"  [{done}/{total}] {phase}", end="\r", flush=True)

        context_points = run_context_sweep(
            client_config, sweep_config, progress=ctx_progress
        )
        print(" " * 60, end="\r")
        for point in context_points:
            if point.skipped:
                print(f"  ctx {point.context_tokens:>9,}: skipped ({point.skip_reason})")
            else:
                print(
                    f"  ctx {point.context_tokens:>9,}: c={point.concurrency} "
                    f"ttft {point.ttft.mean or 0:8.0f} ms  "
                    f"p50 {point.latency.p50 or 0:8.0f} ms  "
                    f"errors {point.errors}/{point.requests}"
                )

    report = generate_report(
        categories, test_suite_hash, elapsed_s=elapsed, perf=perf_report,
        workers=workers, context_points=context_points,
    )
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.write_text(report, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"Report saved to: {report_path}")

    if perf_report is not None:
        perf_path = report_path.with_suffix(".perf.json")
        perf_path.write_text(json.dumps(perf_report.to_dict(), indent=2), encoding="utf-8")
        print(f"Perf JSON saved to: {perf_path}")

    if context_points is not None:
        ctx_path = report_path.with_suffix(".context.json")
        ctx_path.write_text(
            json.dumps([p.to_dict() for p in context_points], indent=2), encoding="utf-8"
        )
        print(f"Context sweep JSON saved to: {ctx_path}")

    if categories:
        scored = [r for c in categories for r in c.scored]
        passed = sum(1 for r in scored if r.passed)
        errors = sum(c.errors for c in categories)
        print(f"\nOverall: {passed}/{len(scored)} questions passed"
              + (f"  ({errors} transport error(s) excluded)" if errors else ""))
        for c in sorted(categories, key=lambda x: x.name):
            t = c.tokens
            print(
                f"  {c.name}: {c.passed}/{len(c.scored)} ({c.score_pct:.0f}% score, "
                f"{c.weighted_score_pct:.0f}% weighted) | "
                f"p50 {_fmt_ms(c.median_latency_ms)} | "
                f"{t.prompt_tokens:,} in + {t.completion_tokens:,} out"
            )
        total_t = sum(c.tokens.total_tokens for c in categories)
        print(f"\nTotal tokens consumed: {total_t:,}")


if __name__ == "__main__":
    main()
