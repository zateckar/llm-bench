"""Versioned use-case suites supplied by application teams (``usecase-v1``).

A suite is one YAML document (see docs/usecase-suite-example.yaml). Uploads are
parsed with the rigorous loader's strict rules, checked with the same static
fixture checks, and stored as immutable versions. Runs pin
``usecase:<slug>@<version>``; reports name the suite ``usecase:<slug>`` so
versions of one suite pair on their identical questions.

Only deterministic evaluators are allowed: an uploaded document must never
cause server-side code execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
import sqlite3

from app.benchmarking.models import Question
from app.benchmarking.quality_suite import MAX_OUTPUT_TOKENS, question_scope, suite_hash
from app.benchmarking.schema_subset import unsupported_keywords, validate
from app.benchmarking.suite_checks import Report, validate_question
from app.benchmarking.test_loader import SuiteError, UniqueKeyLoader, _parse_question

REVISION = "usecase-v1"
PREFIX = "usecase:"
MAX_BYTES = 2_000_000
MAX_QUESTIONS = 2000
ALLOWED_EVALUATORS = frozenset({
    "exact_match", "mcq", "numeric_match", "numeric_set", "contains_keywords", "regex_all",
    "json_match", "set_match", "format_check", "ordered_labels", "refusal_calibration",
    "admits_uncertainty", "open_ended", "language_adherence",
})
SLUG = re.compile(r"[a-z0-9][a-z0-9-]{1,47}")
KEY = re.compile(r"usecase:([a-z0-9][a-z0-9-]{1,47})@([1-9][0-9]{0,5})")
SUITE_FIELDS = {"slug", "name", "description", "owner", "system_prompt"}


class UsecaseSuiteError(SuiteError):
    def __init__(self, problems, warnings=()):
        self.problems = list(problems)
        self.warnings = list(warnings)
        super().__init__("\n".join(self.problems))


@dataclass
class ParsedSuite:
    slug: str
    name: str
    description: str | None
    owner: str | None
    questions: list[Question]
    suite_hash: str
    warnings: list[str] = field(default_factory=list)


def is_key(value) -> bool:
    return isinstance(value, str) and KEY.fullmatch(value) is not None


def make_key(slug, version) -> str:
    return f"{PREFIX}{slug}@{version}"


def split_key(key):
    match = KEY.fullmatch(key or "")
    if not match:
        raise ValueError(f"Invalid use-case suite reference {key!r}")
    return match[1], int(match[2])


def _text(meta, name, problems, *, required=False, limit=1000):
    value = meta.get(name)
    if value is None:
        if required:
            problems.append(f"suite.{name} is required")
        return None
    if not isinstance(value, str) or not value.strip():
        problems.append(f"suite.{name} must be a non-empty string")
        return None
    if len(value) > limit:
        problems.append(f"suite.{name} is longer than {limit} characters")
    return value.strip()


def _response_format(raw, q, where, problems):
    if not isinstance(raw, dict) or raw.get("type") not in {"json_object", "json_schema"}:
        problems.append(f"{where}: response_format.type must be json_object or json_schema")
        return
    if q.evaluator != "json_match":
        problems.append(f"{where}: response_format requires the json_match evaluator")
        return
    value = q.expected.get("value") if isinstance(q.expected, dict) else None
    if raw["type"] == "json_object":
        if set(raw) != {"type"}:
            problems.append(f"{where}: json_object response_format takes no other fields")
        if not isinstance(value, dict):
            problems.append(f"{where}: json_object output requires an object as expected value")
        return
    spec = raw.get("json_schema")
    if set(raw) != {"type", "json_schema"} or not isinstance(spec, dict):
        problems.append(f"{where}: json_schema response_format needs a json_schema mapping")
        return
    if set(spec) - {"name", "schema", "strict", "description"}:
        problems.append(f"{where}: json_schema has unknown field(s) {sorted(set(spec) - {'name', 'schema', 'strict', 'description'})}")
    if not isinstance(spec.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", spec["name"]):
        problems.append(f"{where}: json_schema.name must match [A-Za-z0-9_-]{{1,64}}")
    if "strict" in spec and type(spec["strict"]) is not bool:
        problems.append(f"{where}: json_schema.strict must be boolean")
    schema = spec.get("schema")
    if not isinstance(schema, dict):
        problems.append(f"{where}: json_schema.schema must be a mapping")
        return
    for path, keyword in unsupported_keywords(schema):
        problems.append(f"{where}: schema keyword {keyword!r} at {path} is not supported")
    for error in validate(value, schema):
        problems.append(f"{where}: expected value violates the schema at {error['path']}: {error['error']}")


def _native(q, response_format):
    """A response_format question runs on the native structured-output path."""
    expected = {"strict_json": True, "allow_fence": False, **q.expected}
    return replace(
        q,
        evaluator="native_structured_output",
        expected={"json": expected},
        request={"response_format": response_format},
        interaction={"kind": "structured"},
        metadata={**q.metadata, "protocol": "native-tools-v1", "transport": "stream",
                  "max_turns": 1, "kind": "structured"},
    )


def parse_suite(text) -> ParsedSuite:
    import yaml

    if not isinstance(text, str) or not text.strip():
        raise UsecaseSuiteError(["The document is empty"])
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise UsecaseSuiteError([f"The document is larger than {MAX_BYTES:,} bytes"])
    try:
        data = yaml.load(text, Loader=UniqueKeyLoader)
    except (yaml.YAMLError, SuiteError) as error:
        raise UsecaseSuiteError([f"YAML error: {error}"]) from error
    if not isinstance(data, dict) or set(data) != {"suite", "questions"}:
        raise UsecaseSuiteError(["The document must be a mapping with exactly the keys 'suite' and 'questions'"])
    meta, items = data["suite"], data["questions"]
    problems = []
    if not isinstance(meta, dict):
        raise UsecaseSuiteError(["'suite' must be a mapping"])
    if set(meta) - SUITE_FIELDS:
        problems.append(f"suite has unknown field(s) {sorted(set(meta) - SUITE_FIELDS)}")
    slug = meta.get("slug")
    if not isinstance(slug, str) or not SLUG.fullmatch(slug):
        problems.append("suite.slug must be 2-48 characters of a-z, 0-9 and '-', starting with a letter or digit")
    name = _text(meta, "name", problems, required=True, limit=120)
    description = _text(meta, "description", problems)
    owner = _text(meta, "owner", problems, limit=200)
    default_system = _text(meta, "system_prompt", problems, limit=100_000)
    if not isinstance(items, list) or not items:
        problems.append("'questions' must be a non-empty list")
        items = []
    if len(items) > MAX_QUESTIONS:
        raise UsecaseSuiteError([*problems, f"At most {MAX_QUESTIONS} questions are allowed"])

    checks = Report()
    questions, seen = [], set()
    for index, item in enumerate(items, 1):
        where = f"question #{index}"
        if not isinstance(item, dict):
            problems.append(f"{where}: expected a mapping")
            continue
        item = dict(item)
        response_format = item.pop("response_format", None)
        if default_system is not None and "system_prompt" not in item:
            item["system_prompt"] = default_system
        try:
            q = _parse_question(item, where)
        except SuiteError as error:
            problems.append(str(error))
            continue
        where = f"{q.id}"
        if q.id in seen:
            problems.append(f"{where}: duplicate question id")
            continue
        seen.add(q.id)
        if q.evaluator not in ALLOWED_EVALUATORS:
            problems.append(f"{where}: evaluator {q.evaluator!r} is not allowed in use-case suites; "
                            f"use one of {', '.join(sorted(ALLOWED_EVALUATORS))}")
            continue
        validate_question(checks, q)
        q = replace(
            q,
            max_tokens=q.max_tokens or MAX_OUTPUT_TOKENS,
            metadata={**q.metadata, "family": q.metadata.get("family", q.id),
                      "scope": question_scope(q), "cohort": f"{PREFIX}{slug}"},
        )
        if response_format is not None:
            before = len(problems)
            _response_format(response_format, q, where, problems)
            if len(problems) == before:
                q = _native(q, response_format)
        questions.append(q)
    problems.extend(checks.errors)
    if problems:
        raise UsecaseSuiteError(problems, checks.warnings)
    return ParsedSuite(slug, name, description, owner, questions, suite_hash(questions), list(checks.warnings))


# --- Storage -----------------------------------------------------------------------

def _connect():
    from app import config as app_config
    from app.storage import DETECT_TYPES

    db = sqlite3.connect(str(app_config.DATABASE_PATH), detect_types=DETECT_TYPES)
    db.row_factory = sqlite3.Row
    return db


def load_version(slug, version):
    db = _connect()
    try:
        row = db.execute("SELECT * FROM usecase_suites WHERE slug = ? AND version = ?",
                         (slug, version)).fetchone()
        return dict(row) if row else None
    finally:
        db.close()


def latest_versions(include_archived=False):
    db = _connect()
    try:
        rows = db.execute(
            f"""SELECT * FROM usecase_suites u
                WHERE version = (SELECT MAX(version) FROM usecase_suites v WHERE v.slug = u.slug)
                {'' if include_archived else 'AND archived = 0'}
                ORDER BY name, slug"""
        ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        return []  # Database not initialised yet.
    finally:
        db.close()


def choice(row):
    owner = f" Owner: {row['owner']}." if row.get("owner") else ""
    return {
        "name": make_key(row["slug"], row["version"]),
        "label": f"{row['name']} · v{row['version']}",
        "description": (row.get("description") or "Use-case suite.") + owner,
        "questions": row["question_count"],
    }


def suite_def(key):
    from app.benchmarking.quality_protocol import protocol
    from app.benchmarking.suites import SuiteDef

    slug, version = split_key(key)
    row = load_version(slug, version)
    if row is None:
        raise ValueError(f"Unknown use-case suite {key!r}")
    parsed = parse_suite(row["source_yaml"])
    if parsed.suite_hash != row["suite_hash"] or parsed.slug != slug:
        # A parser change must never silently alter a stored version's questions.
        raise ValueError(f"Use-case suite {key!r} no longer parses to its stored fingerprint")
    provenance = {
        "revision": REVISION,
        "slug": slug,
        "version": version,
        "label": f"{row['name']} v{version}",
        "owner": row.get("owner"),
        "suite_hash": row["suite_hash"],
        "questions": row["question_count"],
    }
    return SuiteDef(
        name=f"{PREFIX}{slug}",
        label=f"{row['name']} · v{version}",
        description=row.get("description") or "Use-case suite.",
        revision=lambda: REVISION,
        load=lambda: list(parsed.questions),
        provenance=lambda: dict(provenance),
        execution=protocol,
        client_extra=dict,
    )
