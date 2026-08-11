"""Load benchmark questions from YAML files.

Question schema
---------------
Required:
    id            unique test identifier (unique across the whole suite)
    prompt        the prompt to send
    category      category name

Evaluation (exactly one of):
    evaluator + expected      explicit evaluator (see evaluators.EVALUATORS)
    criteria [+ must_not]     shorthand for the security_analysis evaluator

Optional:
    system_prompt   system message to send with the prompt
    difficulty      easy | medium | hard | expert  (drives the weighted score)
    weight          explicit weight, overriding the difficulty weight
    pass_threshold  score needed to count as a pass (default 1.0)
    max_tokens      per-question generation cap
    source          provenance note
    description     what the question is probing

The loader is strict on purpose: a typo in an evaluator name used to silently
score every affected question 0, which looks like a model failure rather than a
suite bug.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is declared in pyproject
    yaml = None

from models import DIFFICULTY_WEIGHTS, Question

# Evaluators whose scores carry meaningful partial credit get a lower default
# bar; everything else must be fully correct. Individual questions may still
# override this with an explicit `pass_threshold`.
DEFAULT_PASS_THRESHOLD = 1.0

KNOWN_FIELDS = {
    "id", "category", "prompt", "evaluator", "expected", "system_prompt",
    "criteria", "must_not", "min_criteria", "keywords",
    "difficulty", "weight", "pass_threshold", "max_tokens",
    "source", "description",
}


class SuiteError(ValueError):
    """Raised when a test suite file is structurally invalid."""


def compute_test_suite_hash(tests_dir: str | Path = "tests") -> str:
    """Compute a SHA-256 fingerprint of the entire test suite."""
    tests_path = Path(tests_dir)
    if not tests_path.exists():
        return ""
    hasher = hashlib.sha256()
    for yaml_file in sorted(tests_path.glob("*.yaml")):
        hasher.update(yaml_file.name.encode("utf-8"))
        hasher.update(yaml_file.read_text(encoding="utf-8").encode("utf-8"))
    return hasher.hexdigest()[:16]


def load_yaml_tests(yaml_path: str | Path) -> list[Question]:
    """Load questions from a single YAML file."""
    if yaml is None:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise SuiteError(
            f"{path.name}: expected a list of tests, got {type(data).__name__}"
        )

    questions: list[Question] = []
    problems: list[str] = []
    seen: set[str] = set()

    for i, item in enumerate(data, 1):
        try:
            q = _parse_question(item, f"{path.name}#{i}")
        except SuiteError as e:
            problems.append(str(e))
            continue
        if q.id in seen:
            problems.append(f"{path.name}: duplicate test id {q.id!r}")
            continue
        seen.add(q.id)
        questions.append(q)

    if problems:
        raise SuiteError("\n".join(problems))

    return questions


def load_category(category_name: str, tests_dir: str | Path = "tests") -> list[Question]:
    """Load all questions belonging to a specific category."""
    return [
        q for q in load_all_tests(tests_dir)
        if q.category.lower() == category_name.lower()
    ]


def load_all_tests(tests_dir: str | Path = "tests") -> list[Question]:
    """Load every question from every YAML file in the tests directory."""
    tests_path = Path(tests_dir)
    if not tests_path.exists():
        return []

    questions: list[Question] = []
    seen: dict[str, str] = {}
    problems: list[str] = []

    for yaml_file in sorted(tests_path.glob("*.yaml")):
        file_questions = load_yaml_tests(yaml_file)
        for q in file_questions:
            if q.id in seen:
                problems.append(
                    f"duplicate test id {q.id!r} in {yaml_file.name} "
                    f"(already defined in {seen[q.id]})"
                )
                continue
            seen[q.id] = yaml_file.name
            questions.append(q)

    if problems:
        raise SuiteError("\n".join(problems))

    return questions


def _parse_question(item: Any, where: str) -> Question:
    """Parse a single YAML test item into a Question, or raise SuiteError."""
    from evaluators import EVALUATORS

    if not isinstance(item, dict):
        raise SuiteError(f"{where}: expected a mapping, got {type(item).__name__}")

    unknown = set(item) - KNOWN_FIELDS
    if unknown:
        raise SuiteError(f"{where}: unknown field(s) {sorted(unknown)}")

    raw_id = item.get("id")
    if raw_id is None or not str(raw_id).strip():
        raise SuiteError(f"{where}: missing 'id'")
    test_id = str(raw_id).strip()

    prompt = item.get("prompt")
    if not prompt or not str(prompt).strip():
        raise SuiteError(f"{where} ({test_id}): missing 'prompt'")

    category = item.get("category")
    if not category:
        raise SuiteError(f"{where} ({test_id}): missing 'category'")

    if "criteria" in item:
        if "evaluator" in item:
            raise SuiteError(
                f"{where} ({test_id}): use either 'criteria' shorthand or an explicit "
                "'evaluator', not both"
            )
        evaluator = "security_analysis"
        criteria = item.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise SuiteError(
                f"{where} ({test_id}): 'criteria' must be a non-empty list"
            )
        expected: Any = {
            "criteria": criteria,
            "must_not": item.get("must_not", []),
        }
        if "min_criteria" in item:
            expected["min_criteria"] = item["min_criteria"]
    elif "evaluator" in item:
        evaluator = str(item["evaluator"])
        expected = item.get("expected")
        if expected is None and evaluator not in ("admits_uncertainty", "refusal_calibration"):
            raise SuiteError(
                f"{where} ({test_id}): evaluator {evaluator!r} requires an 'expected' value"
            )
    elif "keywords" in item:
        evaluator = "contains_keywords"
        expected = item.get("keywords", [])
    else:
        raise SuiteError(
            f"{where} ({test_id}): no 'evaluator', 'criteria' or 'keywords' field"
        )

    if evaluator not in EVALUATORS:
        raise SuiteError(
            f"{where} ({test_id}): unknown evaluator {evaluator!r}; "
            f"known: {', '.join(sorted(EVALUATORS))}"
        )

    difficulty = str(item.get("difficulty", "medium")).lower()
    if difficulty not in DIFFICULTY_WEIGHTS:
        raise SuiteError(
            f"{where} ({test_id}): unknown difficulty {difficulty!r}; "
            f"expected one of {', '.join(DIFFICULTY_WEIGHTS)}"
        )

    threshold = item.get("pass_threshold", DEFAULT_PASS_THRESHOLD)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise SuiteError(f"{where} ({test_id}): pass_threshold must be a number")
    if not 0.0 < threshold <= 1.0:
        raise SuiteError(
            f"{where} ({test_id}): pass_threshold must be in (0, 1], got {threshold}"
        )

    weight = item.get("weight")
    if weight is not None:
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise SuiteError(f"{where} ({test_id}): weight must be a number")
        if weight <= 0:
            raise SuiteError(f"{where} ({test_id}): weight must be positive")

    max_tokens = item.get("max_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            raise SuiteError(f"{where} ({test_id}): max_tokens must be an integer")
        if max_tokens <= 0:
            raise SuiteError(f"{where} ({test_id}): max_tokens must be positive")

    return Question(
        id=test_id,
        category=str(category),
        prompt=str(prompt),
        evaluator=evaluator,
        expected=expected,
        system_prompt=item.get("system_prompt"),
        pass_threshold=threshold,
        difficulty=difficulty,
        weight=weight,
        source=item.get("source"),
        description=item.get("description"),
        max_tokens=max_tokens,
    )
