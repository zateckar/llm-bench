"""Read-only replay and measurement audit, with no endpoint calls or migrations."""

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

from app.benchmarking.models import RequestMetrics, TokenUsage
from app.benchmarking.quality_execution import score_response
from app.benchmarking.quality_report import paired_comparison
from app.benchmarking.quality_suite import fingerprint
from app.benchmarking.standard_suite import load_all_questions as load_questions
from app.storage import DETECT_TYPES


def audit(path, run_ids):
    questions = {q.id: q for q in load_questions()}
    reports, findings = {}, []
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True,
                                 detect_types=DETECT_TYPES)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        for run_id in run_ids:
            run = db.execute("SELECT id,status,quality_json,perf_json FROM test_runs WHERE id=?", (run_id,)).fetchone()
            if not run:
                raise ValueError(f"Run {run_id} does not exist")
            quality = json.loads(run["quality_json"] or "{}")
            perf = json.loads(run["perf_json"] or "{}")
            reports[run_id] = quality
            saved = {r["id"]: r for r in quality.get("results", [])}
            coverage, changes = Counter(), []
            for record in db.execute("SELECT test_id,response FROM test_results WHERE run_id=?", (run_id,)):
                old = saved.get(record["test_id"])
                q = questions.get(record["test_id"])
                if not old or not q:
                    coverage["missing_question"] += 1
                    continue
                # Cohort revision is the only change to prior text/code tasks;
                # verify the entire original identity before replaying.
                q = replace(q, metadata={**q.metadata, "cohort": old["metadata"].get("cohort")})
                if fingerprint(q) != old["fingerprint"]:
                    coverage["changed_question"] += 1
                    continue
                if q.interaction or q.request or q.evaluator == "open_ended" or not old["scored"]:
                    coverage["interactive_native_open_or_unscored"] += 1
                    continue
                result = score_response(q, record["response"], TokenUsage(**old["tokens"]),
                                        RequestMetrics(**old["metrics"]))
                coverage[q.evaluator] += 1
                if result.passed != old["passed"] or abs(result.achievement_score - old["score"]) > 1e-9:
                    changes.append({"id": q.id, "saved_achievement": old["score"],
                                    "current_achievement": result.achievement_score,
                                    "saved_pass": old["passed"], "current_pass": result.passed,
                                    "detail": result.detail})
            stages = perf.get("stages", {})
            context = stages.get("context") or {}
            cells = context.get("cells") or [json.loads(row[0]) for row in db.execute(
                "SELECT result_json FROM performance_cells WHERE run_id=?", (run_id,))]
            levels = [lv for cap in (stages.get("users") or {}).get("caps", []) for lv in cap.get("levels", [])]
            findings.append({
                "run": run_id, "status": run["status"], "model": quality.get("model"),
                "suite_hash": quality.get("suite_hash"), "scoring_revision": quality.get("scoring_revision"),
                "temperature": quality.get("protocol", {}).get("temperature"),
                "reasoning_effort": quality.get("protocol", {}).get("reasoning_effort"),
                "summary": {k: quality.get("summary", {}).get(k) for k in (
                    "category_balanced", "category_balanced_full_pass", "missing_outcome_score_bounds", "outcomes", "areas")},
                "replay": {"coverage": dict(coverage), "changes_under_current_evaluators": changes},
                "perfect_achievement_failed_tasks": [r["id"] for r in saved.values()
                                                     if r["scored"] and not r["passed"] and r["score"] == 1],
                "latency_stream_coverage": [{"concurrency": p["concurrency"], "requests": p["requests"],
                                            "stream_samples": p.get("output_token_time", {}).get("count"),
                                            "burst_requests": p.get("burst_delivery_requests")}
                                           for p in (stages.get("latency") or {}).get("concurrency", [])],
                "context": {"stored_cells": len(cells), "statuses": dict(Counter(c["status"] for c in cells)),
                            "thin_measured_cells": sum(0 < c["requests"] < 20 for c in cells),
                            "target_passes_with_incomplete_timing": [
                                [c["context_tokens"], c["concurrency"]] for c in cells
                                if c.get("within_targets") is True and
                                (c.get("output_token_time", {}).get("count", 0) < c.get("completed", 0)
                                 or c.get("ttft", {}).get("count", 0) < c.get("completed", 0))]},
                "sessions": {"inherited_caps": [cap["context_cap"] for cap in
                                                 (stages.get("users") or {}).get("caps", [])
                                                 if cap.get("inherited_failure") is not None],
                             "maximum_drain_seconds": max((lv["ended_at"] - lv["window_ended_at"] for lv in levels), default=0)},
            })
    return {"revision": "latest-run-audit-v1", "read_only": True, "endpoint_calls": 0,
            "replay_note": "Current evaluator counterfactual; saved reports are unchanged. Interactions/native calls are not replayed. Code executes only in the existing subprocess sandbox.",
            "runs": findings,
            "comparisons": [{"left": left, "right": right, **paired_comparison(reports[left], reports[right])}
                            for left, right in zip(run_ids, run_ids[1:])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/bench.db"))
    parser.add_argument("--runs", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.database.resolve():
        parser.error("Output must not overwrite the database")
    result = audit(args.database, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"runs": args.runs, "output": str(args.output)}))


if __name__ == "__main__":
    main()
