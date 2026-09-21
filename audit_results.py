"""Read-only audit of saved runs; never rewrites historical model answers or grades."""

import argparse
from contextlib import closing
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sqlite3
import statistics
import subprocess


def _repository_revision() -> dict[str, object]:
    """Capture source identity without exposing repository contents."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def audit_database(path, run_ids=None, replay=False):
    audit_started_at = datetime.now(timezone.utc).isoformat()
    source = _repository_revision()
    questions = {}
    if replay:
        from test_loader import load_all_tests

        questions = {q.id: q for q in load_all_tests()}
    # SQLite's read-only URI also prevents accidentally creating a missing database.
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")  # One consistent snapshot even while another run is writing.
        runs = db.execute(
            "SELECT r.id,m.name,r.status,r.test_suite_hash,r.quality_json,"
            "r.total_questions,r.workers,r.quality_config_json,r.run_options_json "
            "FROM test_runs r JOIN models m ON m.id=r.model_id ORDER BY r.id"
        ).fetchall()
        report = []
        for run in runs:
            if run_ids and run["id"] not in run_ids:
                continue
            try:
                parsed_saved = json.loads(run["quality_json"] or "{}")
                saved = parsed_saved if isinstance(parsed_saved, dict) else {}
            except (TypeError, ValueError):
                saved = {}
            saved_rows = {
                item.get("id"): item
                for item in (saved.get("results") or [])
                if isinstance(item, dict) and item.get("id")
            }
            try:
                parsed_quality_config = json.loads(run["quality_config_json"] or "{}")
                quality_config = parsed_quality_config if isinstance(parsed_quality_config, dict) else {}
            except (TypeError, ValueError):
                quality_config = {}
            try:
                parsed_run_options = json.loads(run["run_options_json"] or "{}")
                run_options = parsed_run_options if isinstance(parsed_run_options, dict) else {}
            except (TypeError, ValueError):
                run_options = {}
            rows = db.execute(
                "SELECT test_id,category,score,passed,pass_threshold,detail,request_ok,quality_scored,"
                "quality_metadata_json,prompt,response,evaluator FROM test_results WHERE run_id=? ORDER BY test_id",
                (run["id"],),
            ).fetchall()
            items, categories = [], defaultdict(list)
            for row in rows:
                try:
                    parsed_meta = json.loads(row["quality_metadata_json"] or "{}")
                    meta = parsed_meta if isinstance(parsed_meta, dict) else {}
                except (TypeError, ValueError):
                    meta = {}
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
                    "evaluator": row["evaluator"],
                    "criterion_achievement": (meta.get("evaluation") or {}).get("criterion_achievement"),
                    "evaluation_schema_version": (meta.get("evaluation") or {}).get("schema_version"),
                    "evaluator_version": (meta.get("evaluation") or {}).get("evaluator_version"),
                    "failed_criteria": (meta.get("evaluation") or {}).get("failed_criteria"),
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
                            "evaluation": result.evaluation.to_dict() if result.evaluation else None,
                        }
                if scored:
                    categories[row["category"]].append(item)
            usable = [r for r in items if r["scored"]]
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
            duplicate_ids = sorted(
                ident for ident, count in Counter(r["id"] for r in items).items() if count > 1
            )
            saved_ids = [item.get("id") for item in (saved.get("results") or []) if isinstance(item, dict)]
            duplicate_saved_ids = sorted(
                ident for ident, count in Counter(saved_ids).items() if ident and count > 1
            )
            threshold_disagreements = []
            for row in rows:
                if row["score"] is None or row["pass_threshold"] is None:
                    continue
                expected_pass = bool(row["request_ok"] and float(row["score"]) >= float(row["pass_threshold"]) - 1e-9)
                if expected_pass != bool(row["passed"]):
                    threshold_disagreements.append(row["test_id"])
            metadata_score_disagreements = [
                row["test_id"]
                for row in rows
                if row["test_id"] in saved_rows
                and isinstance(saved_rows[row["test_id"]].get("score"), (int, float))
                and row["score"] is not None
                and abs(float(saved_rows[row["test_id"]]["score"]) - float(row["score"])) > 1e-9
            ]
            consistency = {
                "question_count_matches": run["total_questions"] in (None, 0, len(items)),
                "saved_count_matches": not saved.get("results") or len(saved.get("results", [])) == len(items),
                "duplicate_result_ids": duplicate_ids,
                "duplicate_saved_ids": duplicate_saved_ids,
                "score_bounds_violations": [
                    r["id"] for r in items if not isinstance(r["score"], (int, float)) or not 0 <= r["score"] <= 1
                ],
                "threshold_disagreements": threshold_disagreements,
                "metadata_score_disagreements": metadata_score_disagreements,
                "missing_saved_ids": sorted(set(saved_ids) - {r["id"] for r in items}),
                "unrecorded_result_ids": sorted({r["id"] for r in items} - set(saved_ids)) if saved.get("results") else [],
            }
            report.append(
                {
                    "run_id": run["id"],
                    "name": run["name"],
                    "status": run["status"],
                    "suite_hash": run["test_suite_hash"],
                    "protocol": saved.get("protocol"),
                    "quality_config": saved.get("quality_config") or quality_config,
                    "run_options": run_options,
                    "total_questions_configured": run["total_questions"],
                    "workers": run["workers"],
                    "saved_summary": saved.get("summary"),
                    "replay_summary": replay_summary,
                    "count": len(items),
                    "scored": len(usable),
                    "mean": statistics.mean(r["score"] for r in usable) if usable else None,
                    "passes": sum(r["passed"] for r in usable),
                    "outcomes": dict(Counter(r["outcome"] for r in items)),
                    "evaluators": dict(Counter(r["evaluator"] for r in items)),
                    "criterion_diagnostics": {
                        "available": sum(r["criterion_achievement"] is not None for r in items),
                        "mean_achievement": statistics.mean(
                            r["criterion_achievement"] for r in items
                            if r["criterion_achievement"] is not None
                        ) if any(r["criterion_achievement"] is not None for r in items) else None,
                    },
                    "consistency": consistency,
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
        "schema_version": 3,
        "audit_version": "2026-09-21.1",
        "snapshot_started_at": audit_started_at,
        "source": source,
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
