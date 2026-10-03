#!/usr/bin/env python3
"""Run the single rigorous suite and/or the fixed performance workload."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from llm_client import ClientConfig
from perf import (
    DEFAULT_MAX_CONCURRENCY,
    PerfConfig,
    format_perf_console,
    format_perf_markdown,
    run_perf_suite,
)
from quality_execution import run_quality
from quality_report import make_report, markdown
from quality_suite import MAX_OUTPUT_TOKENS, load_questions, suite_hash


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("both", "quality", "performance"), default="both")
    result.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="maximum simultaneous performance requests; quality uses up to four workers",
    )
    result.add_argument("--report", type=Path, default=Path("benchmark-report.md"))
    return result


def main():
    args = parser().parse_args()
    try:
        perf_config = PerfConfig(args.max_concurrency)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL", "")
    model = os.getenv("OPENAI_MODEL", "")
    if not base_url or not model:
        raise SystemExit(
            "Set OPENAI_BASE_URL, OPENAI_MODEL, and OPENAI_KEY in .env or the environment."
        )
    client = ClientConfig(
        base_url,
        os.getenv("OPENAI_KEY", ""),
        model,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0,
        seed=0,
    )
    quality, perf = None, None
    lines = [f"# Benchmark {model}", "", datetime.now(timezone.utc).isoformat(), ""]
    failed = False
    if args.mode != "performance":
        questions = load_questions()

        def progress(index, result):
            print(
                f"{index + 1}/{len(questions)} {result.question.id}: {result.outcome}", flush=True
            )

        results, elapsed, message = run_quality(
            questions, client, args.max_concurrency, on_result=progress
        )
        quality = make_report(results, client, suite_hash(questions))
        lines += [markdown(quality), "", f"Quality wall time: {elapsed / 1000:.1f} seconds.", ""]
        failed = bool(message) or not any(r.is_scored for r in results)
        if message:
            lines.append(message)
    if args.mode != "quality" and not failed:
        perf = run_perf_suite(
            client,
            perf_config,
            progress=lambda phase, n, total: print(f"{phase}: {n}/{total}", flush=True),
        )
        print(format_perf_console(perf))
        lines += [format_perf_markdown(perf), ""]
        failed = not any(p.requests > p.errors for p in perf.concurrency)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    data = {
        "schema_version": 3,
        "mode": args.mode,
        "max_concurrency": args.max_concurrency,
        "quality": quality,
        "performance": perf.to_dict() if perf else None,
    }
    args.report.with_suffix(".json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {args.report}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
