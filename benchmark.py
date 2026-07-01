#!/usr/bin/env python3
"""LLM Quality & Accuracy Benchmark.

Benchmarks a remote OpenAI-compatible LLM across multiple quality categories
using curated questions and rule-based evaluation. Produces a markdown report.

Questions are loaded from YAML files in the tests/ directory.
Evaluators are defined in evaluators.py.

Usage:
    pip install requests python-dotenv pyyaml
    python benchmark.py [--category CATEGORY] [--limit N]
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from evaluators import EVALUATORS
from models import CategoryResult, Question, Result, TokenUsage
from test_loader import load_all_tests, compute_test_suite_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("OPENAI_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "unknown")

if not BASE_URL or not API_KEY:
    print("ERROR: OPENAI_BASE_URL and OPENAI_KEY must be set in .env")
    sys.exit(1)

ENDPOINT = f"{BASE_URL}/chat/completions"
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
# For a quality benchmark, default to deterministic decoding so results are
# reproducible. Override with OPENAI_TEMPERATURE / OPENAI_SEED if desired.
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
SEED_ENV = os.getenv("OPENAI_SEED")
SEED = int(SEED_ENV) if SEED_ENV else None
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

CACHE_FILE = Path(__file__).parent / ".benchmark_cache.json"


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def call_llm(prompt: str, system_prompt: str | None = None) -> tuple[str, TokenUsage]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    if SEED is not None:
        payload["seed"] = SEED
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    last_error = "unknown error"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(ENDPOINT, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"].get("content")
            if content is None:
                content = data["choices"][0]["message"].get("reasoning", "")
            content = content.strip() if content else ""
            usage = data.get("usage", {})
            tokens = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
            return content, tokens
        except requests.exceptions.HTTPError:
            status = resp.status_code
            last_error = f"HTTP {status}"
            # Every attempt counts toward MAX_RETRIES, including 429s, so a
            # sustained rate limit can never loop forever.
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                label = "Rate limited" if status == 429 else f"HTTP {status}"
                print(f"  {label}, retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                return f"[API ERROR: HTTP {status}]", TokenUsage()
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"  Error: {e}, retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                return f"[API ERROR: {e}]", TokenUsage()
    return f"[API ERROR: max retries exceeded: {last_error}]", TokenUsage()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def question_fingerprint(q: Question) -> str:
    """Stable hash of the parts of a question that affect its result.

    Including this in the cache key means editing a prompt, evaluator, expected
    value, or system prompt automatically invalidates the stale cached result.
    Decoding parameters are included too, since they change the model output.
    """
    payload = json.dumps(
        {
            "prompt": q.prompt,
            "system_prompt": q.system_prompt,
            "evaluator": q.evaluator,
            "expected": q.expected,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
        },
        sort_keys=True,
        default=str,
    )
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run_benchmark(questions: list[Question]) -> list[CategoryResult]:
    """Run all questions and return categorized results."""
    categories: dict[str, CategoryResult] = {}
    total = len(questions)
    cache = load_cache()

    for i, q in enumerate(questions, 1):
        if q.category not in categories:
            categories[q.category] = CategoryResult(name=q.category)

        # Check cache. The key includes a fingerprint of the question + decoding
        # params, so a stale entry is never reused after a test is edited.
        cache_key = f"{MODEL}:{q.id}:{question_fingerprint(q)}"
        if cache_key in cache:
            cached = cache[cache_key]
            tokens = TokenUsage(
                prompt_tokens=cached.get("prompt_tokens", 0),
                completion_tokens=cached.get("completion_tokens", 0),
            )
            result = Result(
                question=q,
                response=cached["response"],
                score=cached["score"],
                detail=cached["detail"],
                tokens=tokens,
            )
            categories[q.category].results.append(result)
            status = "PASS" if result.score >= 0.5 else "FAIL"
            print(f"[{i}/{total}] {q.category} — {q.id}: CACHED {status} ({result.score:.0%})")
            continue

        print(f"[{i}/{total}] {q.category} — {q.id}: {q.prompt[:60]}...", end=" ", flush=True)

        response, tokens = call_llm(q.prompt, q.system_prompt)
        evaluator = EVALUATORS[q.evaluator]
        try:
            score, detail = evaluator(response, q.expected)
        except Exception as e:
            logger.warning("Evaluator error for %s: %s", q.id, e)
            score, detail = 0.0, f"Evaluator error: {e}"

        result = Result(question=q, response=response, score=score, detail=detail, tokens=tokens)
        categories[q.category].results.append(result)

        # Persist after each real API call so a mid-run crash keeps prior work.
        # This only runs on cache misses (i.e. alongside a network call that
        # dominates runtime), so the JSON rewrite cost is negligible here.
        cache[cache_key] = {
            "response": response,
            "score": score,
            "detail": detail,
            "prompt_tokens": tokens.prompt_tokens,
            "completion_tokens": tokens.completion_tokens,
        }
        save_cache(cache)

        status = "PASS" if score >= 0.5 else "FAIL"
        print(f"{status} ({score:.0%}) [tokens: {tokens.prompt_tokens}+{tokens.completion_tokens}]")

    return list(categories.values())


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def truncate(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def generate_report(categories: list[CategoryResult], test_suite_hash: str = "") -> str:
    """Generate a markdown report from results."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_questions = sum(c.total for c in categories)
    total_passed = sum(c.passed for c in categories)
    overall_pct = sum(c.score_pct * c.total for c in categories) / total_questions if total_questions else 0

    total_prompt = sum(c.tokens.prompt_tokens for c in categories)
    total_completion = sum(c.tokens.completion_tokens for c in categories)
    total_tokens = total_prompt + total_completion

    lines = [
        f"# LLM Benchmark Report — {MODEL}",
        "",
        f"**Date**: {now}",
        f"**Endpoint**: {BASE_URL}",
        f"**Model**: {MODEL}",
        f"**Temperature**: {TEMPERATURE}" + (f" · **Seed**: {SEED}" if SEED is not None else ""),
        f"**Test Suite Version**: `{test_suite_hash}`" if test_suite_hash else "",
        "",
        "---",
        "",
        f"## Overall Score: {total_passed}/{total_questions} questions passed ({overall_pct:.1f}% average accuracy)",
        "",
        "## Token Usage Summary",
        "",
        f"- **Total tokens**: {total_tokens:,}",
        f"- **Input tokens**: {total_prompt:,}",
        f"- **Output tokens**: {total_completion:,}",
        "",
        "## Category Breakdown",
        "",
        "| Category | Score | Questions | Avg Accuracy | Input Tokens | Output Tokens | Total Tokens |",
        "|----------|-------|-----------|--------------|--------------|---------------|--------------|",
    ]

    for c in categories:
        t = c.tokens
        lines.append(
            f"| {c.name} | {c.passed}/{c.total} | {c.total} | {c.score_pct:.1f}% | {t.prompt_tokens:,} | {t.completion_tokens:,} | {t.total_tokens:,} |"
        )

    lines.extend(["", "---", "", "## Detailed Results", ""])

    for c in categories:
        lines.append(f"### {c.name}")
        lines.append("")
        lines.append("| # | ID | Question | Score | Input | Output | Detail |")
        lines.append("|---|-----|----------|-------|-------|--------|--------|")

        for i, r in enumerate(c.results, 1):
            status = "PASS" if r.score >= 0.5 else "FAIL"
            q_short = truncate(r.question.prompt, 60)
            lines.append(
                f"| {i} | {r.question.id} | {q_short} | {status} ({r.score:.0%}) | {r.tokens.prompt_tokens:,} | {r.tokens.completion_tokens:,} | {truncate(r.detail, 80)} |"
            )

        lines.append("")

    # Sample responses
    lines.extend(["---", "", "## Sample Responses", ""])

    for c in categories:
        for r in c.results[:2]:
            lines.append(f"### {r.question.id} — {c.name}")
            lines.append("")
            lines.append(f"**Prompt**: {r.question.prompt[:200]}")
            lines.append("")
            lines.append(f"**Response**:")
            lines.append("```")
            lines.append(r.response[:500])
            lines.append("```")
            lines.append("")
            lines.append(f"**Score**: {'PASS' if r.score >= 0.5 else 'FAIL'} ({r.score:.0%}) — {r.detail}")
            lines.append("")

    # Summary
    lines.extend(["---", "", "## Summary & Observations", ""])

    strengths = [c for c in categories if c.score_pct >= 70]
    weaknesses = [c for c in categories if c.score_pct < 50]

    if strengths:
        lines.append("**Strengths**:")
        for c in strengths:
            lines.append(f"- {c.name}: {c.score_pct:.1f}%")
        lines.append("")

    if weaknesses:
        lines.append("**Weaknesses**:")
        for c in weaknesses:
            lines.append(f"- {c.name}: {c.score_pct:.1f}%")
        lines.append("")

    if not strengths and not weaknesses:
        lines.append("- Performance is fairly uniform across categories.")
        lines.append("")

    # Error pattern analysis
    fail_by_category = {}
    for c in categories:
        fails = [r for r in c.results if r.score < 0.5]
        if fails:
            fail_by_category[c.name] = fails

    if fail_by_category:
        lines.append("**Error Patterns**:")
        for cat_name, fails in fail_by_category.items():
            lines.append(f"- {cat_name}:")
            for r in fails:
                lines.append(f"  - {r.question.id}: {r.detail[:100]}")
        lines.append("")

    lines.append(f"**Total API calls**: {total_questions}")
    lines.append(f"**Total tokens consumed**: {total_tokens:,} ({total_prompt:,} input + {total_completion:,} output)")
    lines.append(f"**Evaluation method**: Rule-based scoring (no second LLM)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    """Parse command line arguments."""
    args = sys.argv[1:]
    category = None
    limit = None

    i = 0
    while i < len(args):
        if args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        elif args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {args[i]}")
            print("Usage: python benchmark.py [--category CATEGORY] [--limit N]")
            sys.exit(1)

    return category, limit


def main():
    category, limit = parse_args()

    # Load questions from YAML files
    questions = load_all_tests(Path(__file__).parent / "tests")

    if not questions:
        print("ERROR: No questions found in tests/ directory")
        sys.exit(1)

    # Filter by category if specified
    if category:
        questions = [q for q in questions if q.category.lower() == category.lower()]
        if not questions:
            print(f"ERROR: No questions found for category '{category}'")
            sys.exit(1)

    # Apply limit if specified
    if limit:
        questions = questions[:limit]

    print(f"LLM Benchmark — {MODEL}")
    print(f"Endpoint: {BASE_URL}")
    test_suite_hash = compute_test_suite_hash(Path(__file__).parent / "tests")
    print(f"Test Suite Version: {test_suite_hash}")
    print(f"Running {len(questions)} questions across {len(set(q.category for q in questions))} categories...\n")

    start = time.time()
    categories = run_benchmark(questions)
    elapsed = time.time() - start

    report = generate_report(categories, test_suite_hash)
    report_path = Path(__file__).parent / "report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"Report saved to: {report_path}")

    # Print summary
    total_q = sum(c.total for c in categories)
    total_p = sum(c.passed for c in categories)
    print(f"\nOverall: {total_p}/{total_q} questions passed")
    for c in categories:
        t = c.tokens
        print(f"  {c.name}: {c.passed}/{c.total} ({c.score_pct:.0f}%) | {t.prompt_tokens:,} in + {t.completion_tokens:,} out = {t.total_tokens:,} tokens")
    total_t = sum(c.tokens.total_tokens for c in categories)
    print(f"\nTotal tokens consumed: {total_t:,}")


if __name__ == "__main__":
    main()
