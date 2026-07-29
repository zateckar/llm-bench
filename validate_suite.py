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
import re
import sys
from collections import Counter
from pathlib import Path

from evaluators import EVALUATORS, TEST_HELPERS
from models import DIFFICULTY_WEIGHTS, Question
from test_loader import SuiteError, load_all_tests

# format_check check types the evaluator implements. Kept here explicitly so a
# typo in a suite file is an error rather than a silently-failing check.
KNOWN_CHECK_TYPES = {
    "json", "json_path", "contains", "not_contains",
    "max_words", "min_words", "word_count_exact",
    "min_chars", "max_chars", "starts_with", "ends_with",
    "regex", "not_regex", "count_occurrences",
    "line_count", "paragraph_count",
    "every_line_matches", "every_line_word_count",
    "numbered_list", "bullet_list",
    "min_sentences", "max_sentences", "sentence_count_exact",
    "unique_lines", "unique_words", "only_words", "table_shape",
}

VALUELESS_CHECKS = {"json", "unique_lines", "unique_words", "numbered_list", "bullet_list", "table_shape"}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")

    @property
    def ok(self) -> bool:
        return not self.errors


def check_regex(report: Report, where: str, pattern: object, label: str) -> None:
    """Compile a pattern and reject ones that match everything."""
    if not isinstance(pattern, str):
        report.error(where, f"{label} must be a string, got {type(pattern).__name__}")
        return
    if not pattern:
        report.error(where, f"{label} is empty")
        return
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        report.error(where, f"{label} does not compile: {pattern!r} ({e})")
        return
    if compiled.search(""):
        report.error(
            where,
            f"{label} matches the empty string, so it is satisfied by any response: {pattern!r}",
        )


def check_contains_keywords(report: Report, where: str, expected: object) -> None:
    if isinstance(expected, list):
        keywords = [str(k) for k in expected]
        if not keywords:
            report.error(where, "contains_keywords has an empty keyword list")
            return
        lowered = [k.lower() for k in keywords]
        for i, a in enumerate(lowered):
            for j, b in enumerate(lowered):
                if i != j and a and a in b:
                    report.error(
                        where,
                        f"bare keyword list looks like alternatives: {keywords[i]!r} is contained in "
                        f"{keywords[j]!r}. A bare list means ALL keywords are required; use "
                        f"`groups:` or `any:` for alternative spellings.",
                    )
                    return
        return

    if not isinstance(expected, dict):
        report.error(where, f"contains_keywords expects a list or mapping, got {type(expected).__name__}")
        return

    unknown = set(expected) - {"all", "any", "groups", "none", "n_of", "partial"}
    if unknown:
        report.error(where, f"contains_keywords has unknown key(s) {sorted(unknown)}")
    if not any(expected.get(k) for k in ("all", "any", "groups", "n_of")):
        report.error(where, "contains_keywords has no positive requirement (all/any/groups/n_of)")
    for group in expected.get("groups") or []:
        if not isinstance(group, list) or not group:
            report.error(where, f"contains_keywords group must be a non-empty list, got {group!r}")
    n_of = expected.get("n_of")
    if n_of is not None:
        pool = n_of.get("of") or []
        need = int(n_of.get("n", 0))
        if not pool:
            report.error(where, "contains_keywords n_of has an empty pool")
        elif need < 1 or need > len(pool):
            report.error(where, f"contains_keywords n_of.n={need} is outside 1..{len(pool)}")
    forbidden = {str(k).lower() for k in (expected.get("none") or [])}
    required = {
        str(k).lower()
        for k in (expected.get("all") or []) + (expected.get("any") or [])
    }
    for group in expected.get("groups") or []:
        required |= {str(k).lower() for k in group}
    clash = forbidden & required
    if clash:
        report.error(where, f"contains_keywords requires and forbids the same term(s): {sorted(clash)}")


def check_format_check(report: Report, where: str, expected: object) -> None:
    if not isinstance(expected, dict):
        report.error(where, f"format_check expects a mapping with `checks`, got {type(expected).__name__}")
        return
    checks = expected.get("checks")
    if not isinstance(checks, list) or not checks:
        report.error(where, "format_check has no checks")
        return
    for i, check in enumerate(checks, 1):
        label = f"check #{i}"
        if not isinstance(check, dict):
            report.error(where, f"{label} must be a mapping")
            continue
        ctype = check.get("type")
        if ctype not in KNOWN_CHECK_TYPES:
            report.error(where, f"{label} has unknown type {ctype!r}")
            continue
        if ctype in ("regex", "not_regex", "every_line_matches"):
            check_regex(report, where, check.get("value"), f"{label} ({ctype})")
        elif ctype == "count_occurrences":
            check_regex(report, where, check.get("pattern", check.get("value")), f"{label} (pattern)")
            if not isinstance(check.get("count"), int):
                report.error(where, f"{label} count_occurrences needs an integer `count`")
        elif ctype == "json_path":
            if not check.get("path"):
                report.error(where, f"{label} json_path needs a `path`")
            if "value" not in check:
                report.error(where, f"{label} json_path needs a `value`")
        elif ctype == "table_shape":
            for key in ("rows", "columns"):
                if not isinstance(check.get(key), int):
                    report.error(where, f"{label} table_shape needs an integer `{key}`")
        elif ctype == "only_words":
            if not isinstance(check.get("value"), list) or not check["value"]:
                report.error(where, f"{label} only_words needs a non-empty list")
        elif ctype not in VALUELESS_CHECKS and check.get("value") is None:
            report.error(where, f"{label} ({ctype}) needs a `value`")

        if ctype in ("numbered_list", "bullet_list") and check.get("value") is None:
            report.warn(
                where,
                f"{label} {ctype} without a `value` only requires 2+ items; set an exact count "
                "if the prompt specifies one",
            )


def check_code_exec(report: Report, where: str, expected: object) -> None:
    if isinstance(expected, dict):
        fixtures = expected.get("tests")
        helper = expected.get("helper")
        if helper and helper not in TEST_HELPERS:
            report.error(where, f"code_exec references unknown helper {helper!r}")
    else:
        fixtures = expected
    if not isinstance(fixtures, list) or not fixtures:
        report.error(where, "code_exec has no fixtures")
        return
    for i, fixture in enumerate(fixtures, 1):
        label = f"fixture #{i}"
        if not isinstance(fixture, dict):
            report.error(where, f"{label} must be a mapping")
            continue
        fn = fixture.get("function")
        if not fn or not isinstance(fn, str):
            report.error(where, f"{label} needs a `function` name")
            continue
        if not re.fullmatch(r"[A-Za-z_]\w*", fn):
            report.error(where, f"{label} function name {fn!r} is not a valid Python identifier")
        if "expected" not in fixture:
            report.error(where, f"{label} ({fn}) has no `expected` value")
        args = fixture.get("args")
        if args is not None and not isinstance(args, list):
            report.error(where, f"{label} ({fn}) `args` must be a list, got {type(args).__name__}")
        kwargs = fixture.get("kwargs")
        if kwargs is not None and not isinstance(kwargs, dict):
            report.error(where, f"{label} ({fn}) `kwargs` must be a mapping")
        for numeric_key in ("tolerance", "relative"):
            if numeric_key in fixture and not isinstance(fixture[numeric_key], (int, float)):
                report.error(where, f"{label} ({fn}) `{numeric_key}` must be a number")


def check_security_analysis(report: Report, where: str, expected: object) -> None:
    if not isinstance(expected, dict):
        report.error(where, f"security_analysis expects a mapping, got {type(expected).__name__}")
        return
    criteria = expected.get("criteria") or []
    if not criteria:
        report.error(where, "security_analysis has no criteria")
    for i, pattern in enumerate(criteria, 1):
        check_regex(report, where, pattern, f"criterion #{i}")
    for i, pattern in enumerate(expected.get("must_not") or [], 1):
        check_regex(report, where, pattern, f"must_not #{i}")
    min_criteria = expected.get("min_criteria")
    if min_criteria is not None:
        if not isinstance(min_criteria, int):
            report.error(where, "min_criteria must be an integer")
        elif not 1 <= min_criteria <= len(criteria):
            report.error(
                where, f"min_criteria={min_criteria} is outside 1..{len(criteria)}"
            )


def check_multi_step(report: Report, where: str, expected: object) -> None:
    if isinstance(expected, dict):
        steps = expected.get("steps")
        for i, pattern in enumerate(expected.get("must_not") or [], 1):
            check_regex(report, where, pattern, f"must_not #{i}")
    else:
        steps = expected
    if not isinstance(steps, list) or not steps:
        report.error(where, "multi_step_solution has no steps")
        return
    orders = []
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            report.error(where, f"step #{i} must be a mapping")
            continue
        check_regex(report, where, step.get("pattern"), f"step #{i} pattern")
        if "order" not in step:
            report.error(where, f"step #{i} has no `order`")
        else:
            orders.append(step["order"])
    duplicates = [o for o, n in Counter(orders).items() if n > 1]
    if duplicates:
        report.error(where, f"multi_step_solution has duplicate step orders: {sorted(duplicates)}")


def check_ordered_labels(report: Report, where: str, expected: object) -> None:
    if not isinstance(expected, list) or not expected:
        report.error(where, "ordered_labels expects a non-empty list")
        return
    seen = []
    for item in expected:
        if not isinstance(item, dict):
            report.error(where, "ordered_labels entries must be mappings")
            continue
        idx = item.get("index")
        if not isinstance(idx, int):
            report.error(where, f"ordered_labels entry has a non-integer index: {idx!r}")
            continue
        seen.append(idx)
        accept = item.get("accept") or []
        if not accept:
            report.error(where, f"item {idx} has no `accept` patterns")
        for pattern in accept:
            check_regex(report, where, pattern, f"item {idx} accept")
        for pattern in item.get("reject") or []:
            check_regex(report, where, pattern, f"item {idx} reject")
        overlap = set(accept) & set(item.get("reject") or [])
        if overlap:
            report.error(where, f"item {idx} both accepts and rejects {sorted(overlap)}")
    duplicates = [i for i, n in Counter(seen).items() if n > 1]
    if duplicates:
        report.error(where, f"ordered_labels has duplicate indexes: {sorted(duplicates)}")
    if seen and sorted(seen) != list(range(1, len(seen) + 1)):
        report.warn(where, f"ordered_labels indexes are not 1..{len(seen)}: {sorted(seen)}")


def check_set_match(report: Report, where: str, expected: object) -> None:
    if isinstance(expected, dict):
        items = [str(i) for i in (expected.get("items") or [])]
        decoys = [str(d) for d in (expected.get("decoys") or [])]
    elif isinstance(expected, list):
        items, decoys = [str(i) for i in expected], []
    else:
        report.error(where, f"set_match expects a list or mapping, got {type(expected).__name__}")
        return
    if not items:
        report.error(where, "set_match has no items")
        return
    if not decoys:
        report.warn(where, "set_match has no decoys, so it does not test discrimination")
    for item in items:
        for decoy in decoys:
            if item.lower() == decoy.lower():
                report.error(where, f"{item!r} is both a required item and a decoy")
            elif decoy.lower() in item.lower():
                report.error(
                    where,
                    f"decoy {decoy!r} is a substring of required item {item!r}: a correct answer "
                    "would be scored as leaking a decoy",
                )


def check_json_match(report: Report, where: str, expected: object) -> None:
    if not isinstance(expected, dict):
        report.error(where, f"json_match expects a mapping, got {type(expected).__name__}")
        return
    if "value" not in expected:
        report.error(where, "json_match needs a `value`")
    mode = expected.get("mode", "exact")
    if mode not in ("exact", "subset"):
        report.error(where, f"json_match mode must be 'exact' or 'subset', got {mode!r}")
    ignore = expected.get("ignore_keys")
    if ignore is not None and not isinstance(ignore, list):
        report.error(where, "json_match ignore_keys must be a list")


def check_regex_all(report: Report, where: str, expected: object) -> None:
    if isinstance(expected, dict):
        patterns = expected.get("patterns") or []
        must_not = expected.get("must_not") or []
    elif isinstance(expected, list):
        patterns, must_not = expected, []
    else:
        report.error(where, f"regex_all expects a list or mapping, got {type(expected).__name__}")
        return
    if not patterns and not must_not:
        report.error(where, "regex_all has no patterns")
    for i, pattern in enumerate(patterns, 1):
        check_regex(report, where, pattern, f"pattern #{i}")
    for i, pattern in enumerate(must_not, 1):
        check_regex(report, where, pattern, f"must_not #{i}")


def check_command_correctness(report: Report, where: str, expected: object) -> None:
    if not isinstance(expected, list) or not expected:
        report.error(where, "command_correctness expects a non-empty list")
        return
    required = 0
    for i, entry in enumerate(expected, 1):
        if not isinstance(entry, dict):
            report.error(where, f"entry #{i} must be a mapping")
            continue
        check_regex(report, where, entry.get("pattern"), f"entry #{i} pattern")
        if entry.get("required", True):
            required += 1
        if entry.get("forbidden") and entry.get("required", True):
            report.error(
                where,
                f"entry #{i} is both required and forbidden; a forbidden entry must set required: false",
            )
    if required == 0:
        report.error(where, "command_correctness has no required commands")


def check_numeric(report: Report, where: str, expected: object, evaluator: str) -> None:
    if evaluator == "numeric_match":
        if isinstance(expected, dict):
            if "value" not in expected:
                report.error(where, "numeric_match mapping needs a `value`")
            elif not isinstance(expected["value"], (int, float)):
                report.error(where, "numeric_match `value` must be a number")
            for key in ("tolerance", "relative"):
                if key in expected and not isinstance(expected[key], (int, float)):
                    report.error(where, f"numeric_match `{key}` must be a number")
            if "tolerance" not in expected and "relative" not in expected:
                report.warn(where, "numeric_match mapping sets neither tolerance nor relative")
        elif not isinstance(expected, (int, float)):
            report.error(where, f"numeric_match expects a number, got {type(expected).__name__}")
    else:  # numeric_set
        values = expected.get("values") if isinstance(expected, dict) else expected
        if not isinstance(values, list) or not values:
            report.error(where, "numeric_set has no values")
        elif not all(isinstance(v, (int, float)) for v in values):
            report.error(where, "numeric_set values must all be numbers")


def check_mcq(report: Report, where: str, expected: object) -> None:
    if isinstance(expected, dict):
        answer = str(expected.get("answer", "")).strip()
        options = str(expected.get("options", "")).strip()
    else:
        answer, options = str(expected).strip(), ""
    if len(answer) != 1 or not answer.isalpha():
        report.error(where, f"mcq answer must be a single letter, got {answer!r}")
        return
    if not options:
        report.warn(where, "mcq has no explicit `options`; extraction falls back to A-H")
        return
    if answer.upper() not in options.upper():
        report.error(where, f"mcq answer {answer!r} is not in options {options!r}")
    if len(set(options.upper())) < 3:
        report.warn(
            where,
            f"mcq has only {len(set(options))} options, so a random guess passes "
            f"{100 / max(1, len(set(options))):.0f}% of the time",
        )


def check_refusal(report: Report, where: str, expected: object) -> None:
    if expected is None:
        report.warn(
            where,
            "refusal item has no `forbidden` patterns, so a response that hedges and then "
            "fabricates specifics still passes",
        )
        return
    if not isinstance(expected, dict):
        report.error(where, f"refusal item expects a mapping, got {type(expected).__name__}")
        return
    forbidden = expected.get("forbidden") or []
    if not forbidden:
        report.warn(where, "refusal item has an empty `forbidden` list")
    for i, pattern in enumerate(forbidden, 1):
        check_regex(report, where, pattern, f"forbidden #{i}")


VALIDATORS = {
    "contains_keywords": check_contains_keywords,
    "format_check": check_format_check,
    "code_exec": check_code_exec,
    "security_analysis": check_security_analysis,
    "multi_step_solution": check_multi_step,
    "ordered_labels": check_ordered_labels,
    "set_match": check_set_match,
    "json_match": check_json_match,
    "regex_all": check_regex_all,
    "command_correctness": check_command_correctness,
    "mcq": check_mcq,
}


def validate_question(report: Report, q: Question) -> None:
    where = f"{q.id} [{q.evaluator}]"

    if q.evaluator not in EVALUATORS:
        report.error(where, f"unknown evaluator {q.evaluator!r}")
        return

    if q.evaluator in ("numeric_match", "numeric_set"):
        check_numeric(report, where, q.expected, q.evaluator)
    elif q.evaluator in ("admits_uncertainty", "refusal_calibration"):
        check_refusal(report, where, q.expected)
    elif q.evaluator == "exact_match":
        options = q.expected if isinstance(q.expected, list) else [q.expected]
        if not options or any(o is None or str(o) == "" for o in options):
            report.error(where, "exact_match has an empty expected value")
        if all(str(o).strip().lower() in ("yes", "no", "true", "false") for o in options):
            report.error(
                where,
                "exact_match on a yes/no answer is a coin flip; use `mcq` with 3+ options or "
                "require a specific value instead",
            )
    elif q.evaluator == "file_content_match":
        if not isinstance(q.expected, dict) or not any(
            q.expected.get(k) for k in ("content", "content_patterns", "file_name")
        ):
            report.error(where, "file_content_match has nothing to check")
        for i, pattern in enumerate(q.expected.get("content_patterns") or [], 1):
            check_regex(report, where, pattern, f"content_pattern #{i}")
    else:
        validator = VALIDATORS.get(q.evaluator)
        if validator:
            validator(report, where, q.expected)

    if q.pass_threshold < 1.0:
        report.warn(
            where,
            f"pass_threshold is {q.pass_threshold}, so a partially correct answer passes. "
            "This is intentional only where the evaluator's partial credit is meaningful.",
        )
    if q.difficulty not in DIFFICULTY_WEIGHTS:
        report.error(where, f"unknown difficulty {q.difficulty!r}")
    if len(q.prompt.strip()) < 20:
        report.warn(where, "prompt is suspiciously short")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the benchmark test suite.")
    parser.add_argument("--tests-dir", default="tests")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    tests_dir = Path(args.tests_dir)
    if not tests_dir.is_absolute():
        tests_dir = Path(__file__).parent / tests_dir

    report = Report()
    try:
        questions = load_all_tests(tests_dir)
    except SuiteError as e:
        print("Suite failed to load:\n" + str(e))
        return 1

    if not questions:
        print(f"No questions found in {tests_dir}")
        return 1

    for q in questions:
        validate_question(report, q)

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
