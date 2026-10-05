#!/usr/bin/env python3
"""Static validator for the benchmark test suite.

A benchmark is only as trustworthy as its fixtures. This checks the things that
silently corrupt results rather than producing an obvious error:

* every regex compiles, and no pattern matches the empty string (a pattern that
  does is satisfied by any response, so the check is free);
* `expected` has the shape the chosen evaluator expects;
* a bare `contains_keywords` list does not look like a list of alternatives -
  ``["bell", "alexander graham bell"]`` under all-of semantics is unsatisfiable in
  spirit and was a false-failure waiting to happen;
* `set_match` decoys cannot overlap the required items, which would make a correct
  answer fail;
* `code_exec` fixtures name a callable and a known harness, and carry an expected
  value;
* `mcq` answers are inside the option set;
* multiple-choice and exact-match items are not so guessable that a coin flip
  passes them.

Usage:
    python validate_suite.py [--tests-dir tests] [--strict] [--quiet]

Exit code 0 when clean (warnings allowed), 1 on errors, or on warnings with --strict.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from app.benchmarking.quality_suite import load_questions
from app.benchmarking.suite_checks import (  # noqa: F401 - re-exported for selftests
    Report,
    check_format_check,
    check_json_match,
    validate_question,
)
from app.benchmarking.test_loader import SuiteError


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the benchmark test suite.")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    tests_dir = Path(__file__).parent / "tests"

    report = Report()
    try:
        questions = load_questions()
    except SuiteError as e:
        print("Suite failed to load:\n" + str(e))
        return 1

    if not questions:
        print(f"No questions found in {tests_dir}")
        return 1

    for q in questions:
        validate_question(report, q)

    # The tool-conformance suite is graded natively, so it gets its own oracle
    # checks; its structured-output answer keys share json_match's rules.
    from app.benchmarking import tool_suite

    native = tool_suite.load_questions()
    for problem in tool_suite.validate_suite(native):
        report.error("tool-conformance", problem)
    for q in native:
        if q.evaluator == "native_structured_output":
            check_json_match(report, f"{q.id} [{q.evaluator}]", q.expected["json"])
    if not args.quiet:
        print(f"Validated {len(native)} tool-conformance questions ({tool_suite.REVISION}).")

    from app.benchmarking import open_suite

    open_questions = open_suite.load_questions()
    for q in open_questions:
        validate_question(report, q)
    if not args.quiet:
        print(f"Validated {len(open_questions)} open-ended questions ({open_suite.REVISION}).")

    from app.benchmarking import safety_suite

    safety_questions = safety_suite.load_questions()
    for problem in safety_suite.validate_suite(safety_questions):
        report.error("safety-language", problem)
    for q in safety_questions:
        if q.metadata.get("protocol") != "native-tools-v1":
            validate_question(report, q)
    if not args.quiet:
        print(f"Validated {len(safety_questions)} safety and language questions ({safety_suite.REVISION}).")

    by_category = Counter(q.category for q in questions)
    by_evaluator = Counter(q.evaluator for q in questions)
    by_difficulty = Counter(q.difficulty for q in questions)

    if not args.quiet:
        print(f"Loaded {len(questions)} questions from {tests_dir}\n")
        print("By category:")
        for name, count in sorted(by_category.items()):
            print(f"  {count:4d}  {name}")
        print("\nBy evaluator:")
        for name, count in sorted(by_evaluator.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4d}  {name}")
        print("\nBy difficulty:")
        for tier in ("easy", "medium", "hard", "expert"):
            if by_difficulty.get(tier):
                print(f"  {by_difficulty[tier]:4d}  {tier}")
        print()

    for warning in report.warnings:
        print(f"WARN  {warning}")
    for error in report.errors:
        print(f"ERROR {error}")

    print(
        f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s) "
        f"across {len(questions)} questions."
    )

    if report.errors:
        return 1
    if args.strict and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
