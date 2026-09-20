"""Read-only audit of saved runs; never rewrites historical model answers or grades."""

import argparse
from contextlib import closing
from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path
import sqlite3
import statistics


def audit_database(path, run_ids=None, replay=False):
    questions = {}
    if replay:
        from test_loader import load_all_tests

        questions = {q.id: q for q in load_all_tests()}
    # SQLite's read-only URI also prevents accidentally creating a missing database.
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")  # One consistent snapshot even while another run is writing.
        runs = db.execute(
            "SELECT r.id,m.name,r.status,r.test_suite_hash,r.quality_json "
            "FROM test_runs r JOIN models m ON m.id=r.model_id ORDER BY r.id"
        ).fetchall()
        report = []
        for run in runs:
            if run_ids and run["id"] not in run_ids:
                continue
            rows = db.execute(
                "SELECT test_id,category,score,passed,detail,request_ok,quality_scored,"
                "quality_metadata_json,prompt,response,evaluator FROM test_results WHERE run_id=? ORDER BY test_id",
                (run["id"],),
            ).fetchall()
            items, categories = [], defaultdict(list)
            for row in rows:
                meta = json.loads(row["quality_metadata_json"] or "{}")
                scored = meta.get("scored", bool(row["quality_scored"])
                                  if row["quality_scored"] is not None else bool(row["request_ok"]))
                item = {
                    "id": row["test_id"],
                    "category": row["category"],
                    "score": row["score"],
                    "passed": bool(row["passed"]),
                    "scored": scored,
                    "outcome": meta.get("outcome", "legacy_unknown"),
                    "detail": row["detail"],
                    "scope": meta.get("scope", "legacy_unknown"),
                }
                if row["evaluator"] == "json_match":
                    from evaluators import extract_json, strip_think_blocks

                    if meta.get("metadata", {}).get("protocol") == "json-actions-v1":
                        # response stores a transcript, not a proposed final action.
                        turns = meta.get("diagnostics", {}).get("transcript", [])
                        assistant_turns = [t for t in turns if t.get("role") == "assistant"]
                        item["json_audit"] = {
                            "kind": "interactive_transcript",
                            "assistant_turns": len(assistant_turns),
                            "invalid_turns": [
                                {"turn": i, "missing_final_action": not bool(strip_think_blocks(t["content"]).strip()),
                                 "error": extract_json(t["content"], strict=True)[1]}
                                for i, t in enumerate(assistant_turns, 1)
                                if extract_json(t["content"], strict=True)[1]
                            ],
                        }
                    else:
                        _, error = extract_json(row["response"] or "", strict=True)
                        item["json_audit"] = {"kind": "static_answer", "strict_document_valid": error is None,
                                              "parse_error": error}
                items.append(item)
                if replay:
                    q = questions.get(row["test_id"])
                    # Never grade an old response against a newly worded task.
                    if not q or q.prompt != row["prompt"] or q.evaluator != row["evaluator"]:
                        item["replay_skipped"] = (
                            "Question or output contract changed; fresh answer required"
                        )
                    elif not scored or item["outcome"] in {
                        "truncation", "missing_answer", "repetition",
                    }:
                        item["replay_skipped"] = "Original answer unavailable or incomplete"
                    elif "fixture_seed" in meta.get("metadata", {}):
                        item["replay_skipped"] = (
                            "Expanded historical code fixtures are not the current static fixtures"
                        )
                    else:
                        from models import RequestMetrics, TokenUsage
                        from quality_execution import score_response

                        metrics = RequestMetrics(**meta.get("metrics", {}))
                        result = score_response(q, row["response"] or "", TokenUsage(), metrics)
                        item["replay"] = {
                            "score": result.score,
                            "passed": result.passed,
                            "detail": result.detail,
                            "outcome": result.outcome,
                        }
                if scored:
                    categories[row["category"]].append(item)
            usable = [r for r in items if r["scored"]]
            saved = json.loads(run["quality_json"] or "{}")
            replay_summary = None
            if replay and saved.get("results"):
                from quality_report import summarize

                by_id = {i["id"]: i for i in items}
                corrected = []
                for original in saved["results"]:
                    updated = dict(original)
                    if by_id.get(original["id"], {}).get("replay"):
                        updated.update(by_id[original["id"]]["replay"])
                    corrected.append(updated)
                replay_summary = summarize(corrected)
            report.append(
                {
                    "run_id": run["id"],
                    "name": run["name"],
                    "status": run["status"],
                    "suite_hash": run["test_suite_hash"],
                    "protocol": saved.get("protocol"),
                    "saved_summary": saved.get("summary"),
                    "replay_summary": replay_summary,
                    "count": len(items),
                    "scored": len(usable),
                    "mean": statistics.mean(r["score"] for r in usable) if usable else None,
                    "passes": sum(r["passed"] for r in usable),
                    "outcomes": dict(Counter(r["outcome"] for r in items)),
                    "categories": {
                        k: {
                            "count": len(v),
                            "mean": statistics.mean(r["score"] for r in v),
                            "passes": sum(r["passed"] for r in v),
                            "truncations": sum(r["outcome"] == "truncation" for r in v),
                            "capability_count": sum(r["scope"] == "capability" for r in v),
                            "capability_passes": sum(r["passed"] for r in v if r["scope"] == "capability"),
                        }
                        for k, v in sorted(categories.items())
                    },
                    "items": items,
                }
            )
    cohorts = defaultdict(list)
    for run in report:
        if run["status"] != "completed":
            continue  # Partial runs remain in the audit, never in ceiling calibration.
        protocol = json.dumps(run["protocol"], sort_keys=True)
        # Unknown protocols cannot establish a matched historical comparison.
        digest = hashlib.sha256(protocol.encode()).hexdigest()[:12] if run["protocol"] else f"unknown-{run['run_id']}"
        cohorts[f"{run['suite_hash']}:{digest}"].append(run)
    item_analysis = {}
    for cohort_key, cohort in cohorts.items():
        by_id = defaultdict(list)
        for run in cohort:
            for item in run["items"]:
                if item["scored"]:
                    by_id[item["id"]].append(item)
        item_analysis[cohort_key] = {
            id: {"category": items[0]["category"], "scope": items[0]["scope"],
                 "observations": len(items), "passes": sum(i["passed"] for i in items),
                 "mean": statistics.mean(i["score"] for i in items),
                 "complete_coverage": len(items) == len(cohort),
                 "all_passed": len(cohort) > 1 and len(items) == len(cohort)
                 and all(i["passed"] for i in items),
                 "score_range": max(i["score"] for i in items)-min(i["score"] for i in items)}
            for id, items in sorted(by_id.items())
        }
    return {
        "schema_version": 2,
        "note": "Historical grades are unchanged. Optional replay uses current graders only for "
        "unchanged static prompts; it cannot recover truncated output or evaluate revised prompts. "
        "Endpoint aliases do not verify model identity. "
        "Compare identical suites and protocols; truncation remains a scored failure.",
        "runs": report,
        "cohorts": {
            key: {"suite_hash": runs[0]["suite_hash"], "protocol": runs[0]["protocol"],
                  "run_ids": [r["run_id"] for r in runs],
                  "distinct_model_aliases": len({r["name"] for r in runs})}
            for key, runs in cohorts.items()
        },
        "item_analysis_by_cohort": item_analysis,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("--runs", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replay", action="store_true", help="Compare current grading of unchanged static prompts"
    )
    args = parser.parse_args()
    report = audit_database(args.database, args.runs, args.replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    for run in report["runs"]:
        print(
            f"{run['run_id']}: {run['name']} passes={run['passes']}/{run['scored']} "
            f"truncated={run['outcomes'].get('truncation', 0)}"
        )


if __name__ == "__main__":
    main()
