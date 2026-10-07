"""Propose frozen standard-bank exclusions from read-only saved evidence.

Passing every selected run makes a question a removal candidate, not proof of
universal easiness. Unlike calibration, this selection deliberately checks
success across decoding cohorts; it never pools their scores or ranks models.
"""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from app.benchmarking.calibration import read_runs, validate_report
from app.benchmarking.quality_suite import fingerprint, suite_hash
from app.benchmarking.standard_suite import area_of, load_all_questions


def propose(questions, runs):
    if len({run["id"] for run in runs}) != len(runs):
        raise ValueError("Selection needs unique runs")
    if len({run["model_id"] for run in runs}) < 3:
        raise ValueError("Selection needs evidence from at least three distinct deployments")
    reports, panel = [], []
    for run in runs:
        report = json.loads(run["quality_json"] or "{}")
        error = validate_report(run, report)
        if error:
            raise ValueError(f"Run {run['id']}: {error}")
        reports.append({row["id"]: row for row in report["results"]})
        panel.append({"run": run["id"], "model_id": run["model_id"], "model": report["model"],
                      "suite_hash": report["suite_hash"], "temperature": report["protocol"]["temperature"],
                      "reasoning_effort": report["protocol"].get("reasoning_effort")})
    if len({item["model"] for item in panel}) < 3:
        raise ValueError("Selection needs at least three distinct recorded model identities")
    removed, retained_reasons = [], Counter()
    for q in questions:
        rows = [report.get(q.id) for report in reports]
        if any(row is None for row in rows):
            retained_reasons["new_or_missing"] += 1
        elif q.evaluator == "open_ended":
            retained_reasons["ungraded"] += 1
        elif any(not isinstance(row.get("metadata"), dict)
                 or fingerprint(replace(q, metadata={**q.metadata, "cohort": row["metadata"].get("cohort")}))
                 != row["fingerprint"] for row in rows):
            retained_reasons["changed_identity"] += 1
        elif not all(row["scored"] and row["passed"] for row in rows):
            retained_reasons["failure_or_unscored"] += 1
        else:
            removed.append({"id": q.id, "fingerprint": fingerprint(q), "area": area_of(q.metadata),
                            "category": q.category, "family": q.metadata.get("family", q.id)})
    ids = {q["id"] for q in removed}
    kept = [q for q in questions if q.id not in ids]
    return {"revision": "standard-selection-v1", "source_questions": len(questions),
            "source_hash": suite_hash(questions), "retained_questions": len(kept),
            "removed_questions": len(removed), "panel": panel,
            "rule": "Remove only unchanged, graded tasks fully passed in every selected run; retain failures, missing or changed evidence, ungraded tasks and new challenges.",
            "evidence_status": "provisional; mixed decoding cohorts, no repeated three-run panel per model; no claim of universal easiness or validated discrimination",
            "dropped_categories": sorted({q.category for q in questions} - {q.category for q in kept}),
            "removed_by_area": dict(Counter(q["area"] for q in removed)),
            "retained_by_area": dict(Counter(area_of(q.metadata) for q in kept)),
            "retained_reasons": dict(retained_reasons),
            "removed": sorted(removed, key=lambda row: row["id"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/bench.db"))
    parser.add_argument("--runs", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs):
        parser.error("Run IDs must be unique")
    if args.output.resolve() == args.database.resolve():
        parser.error("Output must not overwrite the database")
    runs = [run for run in read_runs(args.database) if run["id"] in args.runs]
    if set(args.runs) != {run["id"] for run in runs}:
        parser.error("A selected run does not exist")
    try:
        result = propose(load_all_questions(), runs)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"removed": result["removed_questions"], "retained": result["retained_questions"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
