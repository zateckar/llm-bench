"""Static checks for question fixtures, shared by validate_suite.py and suite uploads.

These catch defects that silently corrupt results rather than raising: regexes
that match everything, answer keys of the wrong shape, guessable items and
overlapping decoys. validate_suite.py documents the full list.
"""

from __future__ import annotations

import re
import math
from collections import Counter

from app.benchmarking.evaluators import EVALUATORS, TEST_HELPERS
from app.benchmarking.models import DIFFICULTY_WEIGHTS, Question

# format_check check types the evaluator implements. Kept here explicitly so a
# typo in a suite file is an error rather than a silently-failing check.
KNOWN_CHECK_TYPES = {
    "json",
    "json_path",
    "contains",
    "not_contains",
    "max_words",
    "min_words",
    "word_count_exact",
    "min_chars",
    "max_chars",
    "starts_with",
    "ends_with",
    "regex",
    "not_regex",
    "count_occurrences",
    "line_count",
    "paragraph_count",
    "every_line_matches",
    "every_line_word_count",
    "numbered_list",
    "bullet_list",
    "min_sentences",
    "max_sentences",
    "sentence_count_exact",
    "unique_lines",
    "unique_words",
    "only_words",
    "table_shape",
}

VALUELESS_CHECKS = {
    "json",
    "unique_lines",
    "unique_words",
    "numbered_list",
    "bullet_list",
    "table_shape",
}


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
        report.error(
            where, f"contains_keywords expects a list or mapping, got {type(expected).__name__}"
        )
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
        n = n_of.get("n", len(pool))
        if not pool:
            report.error(where, "contains_keywords n_of has an empty pool")
        elif isinstance(n, bool) or not isinstance(n, int):
            report.error(where, f"contains_keywords n_of.n must be an integer, got {n!r}")
        elif n < 1 or n > len(pool):
            report.error(where, f"contains_keywords n_of.n={n} is outside 1..{len(pool)}")
    forbidden = {str(k).lower() for k in (expected.get("none") or [])}
    required = {str(k).lower() for k in (expected.get("all") or []) + (expected.get("any") or [])}
    for group in expected.get("groups") or []:
        required |= {str(k).lower() for k in group}
    clash = forbidden & required
    if clash:
        report.error(
            where, f"contains_keywords requires and forbids the same term(s): {sorted(clash)}"
        )


def check_format_check(report: Report, where: str, expected: object) -> None:
    if not isinstance(expected, dict):
        report.error(
            where, f"format_check expects a mapping with `checks`, got {type(expected).__name__}"
        )
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
            if "distinct" in check and type(check["distinct"]) is not bool:
                report.error(where, f"{label} distinct must be boolean")
            check_regex(
                report, where, check.get("pattern", check.get("value")), f"{label} (pattern)"
            )
            keys = [k for k in ("count", "min_count", "max_count") if k in check]
            if not keys or any(type(check[k]) is not int or check[k] < 0 for k in keys):
                report.error(
                    where, f"{label} needs a nonnegative integer count/min_count/max_count"
                )
            if "count" in keys and len(keys) > 1:
                report.error(where, f"{label} exact count cannot be combined with bounds")
            if (
                "min_count" in keys
                and "max_count" in keys
                and check["min_count"] > check["max_count"]
            ):
                report.error(where, f"{label} min_count exceeds max_count")
        elif ctype == "json":
            if "strict_json" in check and type(check["strict_json"]) is not bool:
                report.error(where, f"{label} strict_json must be boolean")
            if "item_fields" in check:
                fields = check["item_fields"]
                if check.get("root") != "array" or not isinstance(fields, dict) or not fields:
                    report.error(
                        where, f"{label} item_fields needs an array root and nonempty mapping"
                    )
                else:
                    for key, spec in fields.items():
                        if not isinstance(key, str) or not key:
                            report.error(
                                where, f"{label} item field names must be nonempty strings"
                            )
                        if not isinstance(spec, dict) or spec.get("type") not in {
                            "integer",
                            "string",
                            "boolean",
                        }:
                            report.error(where, f"{label} invalid item field {key}")
                            continue
                        if set(spec) - {"type", "pattern"}:
                            report.error(where, f"{label} unknown options for item field {key}")
                        if "pattern" in spec:
                            if spec["type"] != "string":
                                report.error(where, f"{label} patterns require string fields")
                            check_regex(report, where, spec["pattern"], f"{label}.{key}")
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
            if numeric_key in fixture and (
                type(fixture[numeric_key]) not in (int, float)
                or not math.isfinite(fixture[numeric_key]) or fixture[numeric_key] < 0
            ):
                report.error(where, f"{label} ({fn}) `{numeric_key}` must be finite and nonnegative")
        if "preserve_inputs" in fixture and type(fixture["preserve_inputs"]) is not bool:
            report.error(where, f"{label} ({fn}) preserve_inputs must be boolean")


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
            report.error(where, f"min_criteria={min_criteria} is outside 1..{len(criteria)}")


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
    if "strict_json" in expected and type(expected["strict_json"]) is not bool:
        report.error(where, "json_match strict_json must be a boolean")
    if "allow_fence" in expected and type(expected["allow_fence"]) is not bool:
        report.error(where, "json_match allow_fence must be a boolean")
    for key in ("relative", "tolerance"):
        value = expected.get(key, 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            report.error(where, f"json_match {key} must be finite and nonnegative")
    mode = expected.get("mode", "exact")
    if mode not in ("exact", "subset"):
        report.error(where, f"json_match mode must be 'exact' or 'subset', got {mode!r}")
    ignore = expected.get("ignore_keys")
    if ignore is not None and not isinstance(ignore, list):
        report.error(where, "json_match ignore_keys must be a list")
    aliases = expected.get("value_aliases", {})
    if not isinstance(aliases, dict):
        report.error(where, "json_match value_aliases must map exact leaf paths to alternatives")
    else:
        # Build canonical comparator paths; do not accept misspelled or wildcard paths.
        leaves, containers = {}, set()

        def visit(value, path=""):
            if isinstance(value, dict):
                containers.add(path)
                for key, child in value.items():
                    if not isinstance(key, str) or not key or any(c in key for c in ".[]"):
                        report.error(where, f"json_match key {key!r} cannot form an unambiguous criterion path")
                    visit(child, f"{path}.{key}".lstrip("."))
            elif isinstance(value, list):
                containers.add(path)
                for i, child in enumerate(value):
                    visit(child, f"{path}[{i}]")
            else:
                if type(value) is float and not math.isfinite(value):
                    report.error(where, f"json_match expected value at {path!r} must be finite")
                if value is not None and not isinstance(value, (str, int, float, bool)):
                    report.error(where, f"json_match expected value at {path!r} must be JSON-compatible")
                leaves[path] = value

        visit(expected.get("value"))
        atomic_paths = expected.get("atomic_paths", [])
        if (not isinstance(atomic_paths, list) or any(not isinstance(p, str) or not p for p in atomic_paths)
                or len(set(atomic_paths)) != len(atomic_paths)):
            report.error(where, "json_match atomic_paths must be a list of unique nonempty paths")
            atomic_paths = []
        for path in atomic_paths:
            if path not in containers:
                report.error(where, f"json_match atomic path {path!r} must name an exact container path")
            if any(other != path and (path.startswith(other + ".") or path.startswith(other + "["))
                   for other in atomic_paths):
                report.error(where, f"json_match atomic path {path!r} overlaps another atomic path")
        integer_paths = expected.get("integer_paths", [])
        if (not isinstance(integer_paths, list) or any(not isinstance(p, str) for p in integer_paths)
                or len(set(integer_paths)) != len(integer_paths)):
            report.error(where, "json_match integer_paths must be a list of unique exact scalar paths")
            integer_paths = []
        for path in integer_paths:
            if path not in leaves or type(leaves[path]) is not int:
                report.error(where, f"json_match integer path {path!r} must name an integer answer leaf")
            if isinstance(ignore, list) and any(key in ignore for key in re.findall(r"[^.\[\]]+", path)):
                report.error(where, f"json_match integer path {path!r} cannot be ignored")
        for path, values in aliases.items():
            if not isinstance(path, str) or path not in leaves:
                report.error(where, f"json_match alias path {path!r} is not an exact scalar leaf")
            if (
                not isinstance(values, list)
                or not values
                or any(
                    isinstance(v, (dict, list)) or (type(v) is float and not math.isfinite(v))
                    for v in (values if isinstance(values, list) else [])
                )
            ):
                report.error(
                    where, f"json_match aliases for {path!r} must be a nonempty scalar list"
                )
            elif path in integer_paths and any(type(v) is not int for v in values):
                report.error(where, f"json_match aliases for integer path {path!r} must be integers")


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


def check_open_ended(report: Report, where: str, expected: object) -> None:
    if not isinstance(expected, dict) or set(expected) - {"criteria", "reference"}:
        report.error(where, "open_ended expects a mapping with `criteria` and optional `reference`")
        return
    criteria = expected.get("criteria")
    if (not isinstance(criteria, list) or not 1 <= len(criteria) <= 12
            or any(not isinstance(c, str) or not c.strip() or len(c) > 500 for c in criteria)):
        report.error(where, "open_ended `criteria` must list 1-12 non-empty statements of at most 500 characters")
    reference = expected.get("reference")
    if reference is not None and (not isinstance(reference, str) or not reference.strip() or len(reference) > 20_000):
        report.error(where, "open_ended `reference` must be non-empty text of at most 20,000 characters")


def check_language_adherence(report: Report, where: str, expected: object) -> None:
    if not isinstance(expected, dict) or set(expected) - {"language", "patterns", "must_not", "min_words"}:
        report.error(where, "language_adherence expects a mapping with `language`, optional `patterns`, "
                            "`must_not` and `min_words`")
        return
    if expected.get("language") not in {"en", "cs", "sk", "de"}:
        report.error(where, "language_adherence `language` must be one of en, cs, sk, de")
    for key in ("patterns", "must_not"):
        values = expected.get(key) or []
        if not isinstance(values, list):
            report.error(where, f"language_adherence `{key}` must be a list")
            continue
        for i, pattern in enumerate(values, 1):
            check_regex(report, where, pattern, f"{key} #{i}")
    min_words = expected.get("min_words", 8)
    if type(min_words) is not int or not 1 <= min_words <= 2000:
        report.error(where, "language_adherence `min_words` must be an integer between 1 and 2000")


VALIDATORS = {
    "open_ended": check_open_ended,
    "language_adherence": check_language_adherence,
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
        if not isinstance(q.expected, dict):
            report.error(where, "file_content_match has nothing to check")
            return
        if not any(q.expected.get(k) for k in ("content", "content_patterns", "file_name")):
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
