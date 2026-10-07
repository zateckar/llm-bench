"""Frozen, evidence-backed exclusions from new standard quality runs."""

import copy
import functools
import json
from pathlib import Path

from app.benchmarking.quality_suite import fingerprint, suite_hash

MANIFEST = Path(__file__).with_suffix(".json")


@functools.lru_cache(maxsize=1)
def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def select_questions(questions, manifest=None):
    manifest = manifest if manifest is not None else _manifest()
    if len(questions) != manifest["source_questions"] or suite_hash(questions) != manifest["source_hash"]:
        raise ValueError("Standard source bank changed; review and refresh its frozen selection")
    removed = manifest["removed"]
    exclusions = {row["id"]: row["fingerprint"] for row in removed}
    if len(exclusions) != len(removed):
        raise ValueError("Duplicate standard exclusion IDs")
    if (manifest["removed_questions"] != len(removed)
            or manifest["retained_questions"] != len(questions) - len(removed)):
        raise ValueError("Standard selection counts do not match its inventory")
    available = {q.id: q for q in questions}
    if any(qid not in available or fingerprint(available[qid]) != expected
           for qid, expected in exclusions.items()):
        raise ValueError("Standard exclusion no longer matches its question identity")
    return [q for q in questions if q.id not in exclusions]


def provenance():
    return copy.deepcopy({k: value for k, value in _manifest().items() if k != "removed"})
