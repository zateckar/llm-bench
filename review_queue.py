#!/usr/bin/env python3
"""Create a deterministic, answer-free review queue from an audit artifact."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


RUNTIME_OUTCOMES = {"endpoint_error", "unsupported_context", "cancelled", "truncation", "repetition", "missing_answer"}


def build_review_queue(
    audit: dict[str, Any],
    *,
    run_ids: Iterable[int] | None = None,
    success_samples_per_group: int = 2,
) -> dict[str, Any]:
    selected = set(run_ids) if run_ids is not None else None
    runs = [run for run in audit.get("runs", []) if selected is None or run.get("run_id") in selected]
    queue: list[dict[str, Any]] = []
    successes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in sorted(runs, key=lambda item: item.get("run_id", 0)):
        run_id = run.get("run_id")
        for item in sorted(run.get("items") or [], key=lambda row: str(row.get("id", ""))):
            outcome = str(item.get("outcome", "unknown"))
            capability_nonpass = item.get("scope") == "capability" and item.get("scored") and not item.get("passed")
            runtime_failure = outcome in RUNTIME_OUTCOMES
            if capability_nonpass or runtime_failure:
                queue.append({
                    "run_id": run_id,
                    "id": item.get("id"),
                    "category": item.get("category"),
                    "evaluator": item.get("evaluator"),
                    "scope": item.get("scope"),
                    "score": item.get("score"),
                    "passed": item.get("passed"),
                    "scored": item.get("scored"),
                    "outcome": outcome,
                    "priority": "runtime" if runtime_failure else "capability_nonpass",
                    "detail": str(item.get("detail") or "")[:240],
                    "replay": item.get("replay") or None,
                    "replay_skipped": item.get("replay_skipped") or None,
                })
            if item.get("scored") and item.get("passed") and outcome == "pass":
                key = (str(item.get("category")), str(item.get("evaluator")))
                if len(successes[key]) < success_samples_per_group:
                    successes[key].append({
                        "run_id": run_id,
                        "id": item.get("id"),
                        "category": item.get("category"),
                        "evaluator": item.get("evaluator"),
                        "scope": item.get("scope"),
                        "score": item.get("score"),
                        "passed": True,
                        "scored": True,
                        "outcome": outcome,
                        "priority": "success_sample",
                        "detail": str(item.get("detail") or "")[:240],
                        "replay": item.get("replay") or None,
                        "replay_skipped": item.get("replay_skipped") or None,
                    })
    queue.extend(row for rows in successes.values() for row in rows)
    queue.sort(key=lambda row: (str(row["priority"]), row.get("run_id", 0), str(row.get("id", ""))))
    primary_nonpasses = sum(row["priority"] == "capability_nonpass" for row in queue)
    runtime = sum(row["priority"] == "runtime" for row in queue)
    samples = sum(row["priority"] == "success_sample" for row in queue)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_audit_version": audit.get("audit_version"),
        "source_snapshot_started_at": audit.get("snapshot_started_at"),
        "run_ids": [run.get("run_id") for run in sorted(runs, key=lambda item: item.get("run_id", 0))],
        "policy": {
            "all_scored_capability_nonpasses": True,
            "all_runtime_failures": True,
            "success_samples_per_category_evaluator": success_samples_per_group,
            "answers_omitted": True,
        },
        "summary": {
            "queue_items": len(queue),
            "capability_nonpasses": primary_nonpasses,
            "runtime_failures": runtime,
            "success_samples": samples,
            "by_priority": dict(Counter(row["priority"] for row in queue)),
            "by_outcome": dict(Counter(row["outcome"] for row in queue)),
        },
        "items": queue,
        "note": "This is a deterministic review queue, not an automatic grade correction. A reviewer must disposition each item before a historical rescore or difficulty calibration uses it.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--runs", type=int, nargs="+")
    parser.add_argument("--success-samples", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.success_samples < 0:
        parser.error("--success-samples must be nonnegative")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    queue = build_review_queue(audit, run_ids=args.runs, success_samples_per_group=args.success_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queue, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"{queue['summary']['queue_items']} queue items: "
        f"{queue['summary']['capability_nonpasses']} capability non-passes, "
        f"{queue['summary']['runtime_failures']} runtime failures, "
        f"{queue['summary']['success_samples']} success samples"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
