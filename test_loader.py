"""Load benchmark questions from YAML files.

YAML format (inspired by rapticore/llm-security-benchmark):
    - id: unique test identifier
    - category: category name
    - prompt: the prompt to send
    - evaluator: evaluator function name
    - expected: expected value (type depends on evaluator)
    - system_prompt: optional system prompt

For security_analysis evaluator:
    - criteria: list of regex patterns that SHOULD match
    - must_not: list of regex patterns that MUST NOT match
"""

import hashlib
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

from models import Question


def compute_test_suite_hash(tests_dir: str | Path = "tests") -> str:
    """Compute SHA-256 fingerprint of the entire test suite."""
    tests_path = Path(tests_dir)
    if not tests_path.exists():
        return ""
    hasher = hashlib.sha256()
    for yaml_file in sorted(tests_path.glob("*.yaml")):
        content = yaml_file.read_text(encoding="utf-8")
        hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()[:16]


def load_yaml_tests(yaml_path: str | Path) -> list[Question]:
    """Load questions from a YAML file."""
    if yaml is None:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of tests in {path.name}, got {type(data).__name__}")

    questions = []
    for item in data:
        q = _parse_question(item)
        if q is not None:
            questions.append(q)

    return questions


def load_category(category_name: str, tests_dir: str | Path = "tests") -> list[Question]:
    """Load all YAML files for a specific category."""
    tests_path = Path(tests_dir)
    if not tests_path.exists():
        return []

    questions = []
    for yaml_file in sorted(tests_path.glob("*.yaml")):
        file_questions = load_yaml_tests(yaml_file)
        for q in file_questions:
            if q.category == category_name:
                questions.append(q)

    return questions


def load_all_tests(tests_dir: str | Path = "tests") -> list[Question]:
    """Load all questions from all YAML files in the tests directory."""
    tests_path = Path(tests_dir)
    if not tests_path.exists():
        return []

    questions = []
    for yaml_file in sorted(tests_path.glob("*.yaml")):
        questions.extend(load_yaml_tests(yaml_file))

    return questions


def _parse_question(item: dict) -> Question | None:
    """Parse a single YAML test item into a Question."""
    if not isinstance(item, dict):
        return None

    raw_id = item.get("id")
    if raw_id is None:
        return None
    test_id = str(raw_id)
    prompt = item.get("prompt")

    if not test_id or not prompt:
        return None

    # Determine evaluator and expected from YAML fields
    if "criteria" in item:
        # Security analysis format (from rapticore/llm-security-benchmark style)
        evaluator = "security_analysis"
        expected = {
            "criteria": item.get("criteria", []),
            "must_not": item.get("must_not", []),
        }
    elif "evaluator" in item:
        # Direct evaluator specification
        evaluator = item["evaluator"]
        expected = item.get("expected")
    else:
        # Default to contains_keywords with any text keywords
        evaluator = "contains_keywords"
        expected = item.get("keywords", [])

    return Question(
        id=test_id,
        category=item.get("category", "Unknown"),
        prompt=prompt,
        evaluator=evaluator,
        expected=expected,
        system_prompt=item.get("system_prompt"),
    )
