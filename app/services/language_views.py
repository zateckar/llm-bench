"""Answer-language diagnostics for any run (docs/design-safety-language.md, decision 7).

Questions that declare a language (``answer_language``, ``language`` or ``lang``
in their metadata) are checked with ``language-id``. The diagnostic is
computed when the page is viewed and never changes a score.
"""

import json

from app.benchmarking import language_id

MIN_WORDS = 12
SKIPPED_EVALUATORS = {"json_match", "native_structured_output", "code_exec"}
NO_ANSWER = {"endpoint_error", "cancelled", "missing_answer", "feature_rejected", "evaluator_error"}


def _required(metadata):
    for key in ("answer_language", "language", "lang"):
        value = (metadata or {}).get(key)
        if value in language_id.NAMES:
            return value
    return None


def _answer_text(row, response):
    if (row.get("metadata") or {}).get("protocol") == "native-tools-v1":
        from app.benchmarking.tool_protocol import final_text

        try:
            transcript = json.loads(response or "[]")
        except ValueError:
            return ""
        return final_text(transcript) if isinstance(transcript, list) else ""
    return response or ""


def language_report(quality, results):
    """Per-language adherence of answers, or None when no question declares a language."""
    if not quality or quality.get("schema_version") != 3:
        return None
    responses = {r["test_id"]: r.get("response") for r in results}
    languages, mismatches, checked = {}, [], 0
    for row in quality.get("results") or []:
        required = _required(row.get("metadata"))
        evaluator = (row.get("evaluation") or {}).get("evaluator")
        if required is None or evaluator in SKIPPED_EVALUATORS or row.get("outcome") in NO_ANSWER:
            continue
        entry = languages.setdefault(required, {"language": required, "name": language_id.NAMES[required],
                                                "answers": 0, "adherent": 0, "other": 0, "short": 0})
        entry["answers"] += 1
        text = _answer_text(row, responses.get(row["id"]))
        found = language_id.detect(text)
        if found["words"] < MIN_WORDS:
            entry["short"] += 1
            continue
        checked += 1
        share = found["shares"].get(required, 0.0 if found["shares"] else 1.0)
        if found["language"] == required and share >= 0.85:
            entry["adherent"] += 1
        else:
            entry["other"] += 1
            mismatches.append({"id": row["id"], "category": row.get("category"), "required": required,
                               "detected": found["language"], "share": share})
    if not languages:
        return None
    order = {lang: i for i, lang in enumerate(language_id.LANGUAGES)}
    return {"revision": language_id.REVISION, "checked": checked,
            "languages": sorted(languages.values(), key=lambda e: order[e["language"]]),
            "mismatches": mismatches[:50], "mismatch_count": len(mismatches)}
