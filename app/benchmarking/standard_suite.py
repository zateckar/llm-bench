"""The standard quality suite: one suite covering every built-in area.

See docs/design-consolidation.md. New runs use a frozen subset of the four
built-in areas. The full bank remains available for authoring and regression
checks. Retained questions keep their metadata and fingerprints.
"""

import functools

from app.benchmarking import open_suite, quality_suite, safety_suite, tool_suite

REVISION = "standard-v3"

# (area key, label, source suite module, scored)
AREAS = (
    ("reasoning", "Reasoning & knowledge", quality_suite, True),
    ("tools", "Tool calling & structured output", tool_suite, True),
    ("safety", "Safety & language", safety_suite, True),
    ("open", "Open-ended requests", open_suite, False),
)
AREA_LABELS = {key: label for key, label, _, _ in AREAS}
_COHORTS = {module.REVISION: key for key, _, module, _ in AREAS}
# Saved inventories must retain their areas after a bank revision.
_COHORTS.update({"rigorous-v15": "reasoning"})


def area_of(metadata):
    """Area key of a question or result row, or None for questions of other suites."""
    return _COHORTS.get((metadata or {}).get("cohort"))


def load_all_questions():
    """Unfiltered source bank, including questions retired from new runs."""
    questions = []
    for _, _, module, _ in AREAS:
        questions.extend(module.load_questions())
    seen = set()
    for q in questions:
        if q.id in seen:
            raise ValueError(f"Duplicate question id in the standard suite: {q.id}")
        seen.add(q.id)
    return questions


def load_questions():
    from app.benchmarking.standard_selection import select_questions

    return select_questions(load_all_questions())


def provenance():
    import copy

    return copy.deepcopy(_provenance())


@functools.lru_cache(maxsize=1)
def _provenance():
    # Loading all areas takes over a second; the inventory is fixed per process.
    questions = load_questions()
    areas = []
    for key, label, module, scored in AREAS:
        items = [q for q in questions if area_of(q.metadata) == key]
        areas.append({"area": key, "label": label, "revision": module.REVISION, "questions": len(items),
                      "suite_hash": quality_suite.suite_hash(items), "scored": scored})
    from app.benchmarking.standard_selection import provenance as selection_provenance

    return {"revision": REVISION, "questions": len(questions), "areas": areas,
            "selection": selection_provenance(),
            "max_output_tokens": quality_suite.MAX_OUTPUT_TOKENS}


def validate_suite(questions=None):
    """Cross-area checks; each area's own oracle checks run in validate_suite.py."""
    questions = questions or load_questions()
    problems = []
    fingerprints = {}
    for q in questions:
        if area_of(q.metadata) is None:
            problems.append(f"{q.id}: no known area (cohort {q.metadata.get('cohort')!r})")
        print_ = quality_suite.fingerprint(q)
        if print_ in fingerprints:
            problems.append(f"{q.id}: same fingerprint as {fingerprints[print_]}")
        fingerprints[print_] = q.id
    families = {}
    for q in questions:
        family = q.metadata.get("family", q.id)
        area = area_of(q.metadata)
        if families.setdefault(family, area) != area:
            problems.append(f"{q.id}: family {family!r} spans several areas")
    return problems
