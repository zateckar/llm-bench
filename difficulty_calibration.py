#!/usr/bin/env python3
"""Calibrate empirical difficulty and identify where harder families are needed.

The tool is intentionally read-only. It consumes saved quality reports, keeps
matched suite/protocol cohorts separate, and never rewrites a question's
author-estimated difficulty or a historical grade.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Iterable


def _protocol_key(report: dict[str, Any]) -> str:
    protocol = json.dumps(report.get("protocol"), sort_keys=True)
    return hashlib.sha256(protocol.encode()).hexdigest()[:12] if protocol != "null" else "unknown"


def calibrate_reports(
    reports: Iterable[dict[str, Any]],
    *,
    saturation_rate: float = 0.8,
    floor_rate: float = 0.2,
    min_families: int = 5,
) -> dict[str, Any]:
    """Return empirical category/family diagnostics for saved reports."""
    if not 0 <= floor_rate <= saturation_rate <= 1:
        raise ValueError("floor_rate and saturation_rate must satisfy 0 <= floor <= saturation <= 1")
    if min_families < 1:
        raise ValueError("min_families must be positive")

    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        if not isinstance(report, dict):
            continue
        key = f"{report.get('suite_hash', 'unknown')}:{_protocol_key(report)}"
        cohorts[key].append(report)

    calibrated: dict[str, Any] = {}
    for cohort_key, cohort_reports in sorted(cohorts.items()):
        by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for report in cohort_reports:
            for row in report.get("results") or []:
                if row.get("scope") != "capability" or not row.get("scored"):
                    continue
                fingerprint = row.get("fingerprint") or row.get("id")
                if fingerprint:
                    by_fingerprint[fingerprint].append(row)

        families: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rows in by_fingerprint.values():
            if not rows:
                continue
            representative = rows[0]
            if len(rows) < len(cohort_reports):
                # A missing answer is coverage information, not a failure.
                continue
            family_key = f"{representative.get('category', 'unknown')}::{representative.get('family', representative.get('id'))}"
            families[family_key].append({
                "id": representative.get("id"),
                "category": representative.get("category"),
                "difficulty": representative.get("metadata", {}).get("difficulty") or representative.get("difficulty"),
                "pass_rate": statistics.mean(bool(row.get("passed")) for row in rows),
                "score_mean": statistics.mean(float(row.get("score", 0)) for row in rows),
                "coverage": len(rows) / len(cohort_reports),
                "runs": len(rows),
            })

        category_rows: dict[str, dict[str, Any]] = {}
        for category in sorted({row["category"] for rows in families.values() for row in rows}):
            category_families = [
                family_rows for family_key, family_rows in families.items()
                if family_key.startswith(f"{category}::")
            ]
            family_summaries = [family[0] for family in category_families if family]
            category_rows[category] = {
                "families": len(family_summaries),
                "items": sum(len(family) for family in category_families),
                "mean_pass_rate": statistics.mean(f["pass_rate"] for f in family_summaries)
                if family_summaries else None,
                "saturated_families": sum(f["pass_rate"] >= saturation_rate for f in family_summaries),
                "floor_families": sum(f["pass_rate"] <= floor_rate for f in family_summaries),
                "coverage_mean": statistics.mean(f["coverage"] for f in family_summaries)
                if family_summaries else None,
            }

        recommendations = []
        for category, summary in category_rows.items():
            if summary["families"] < min_families:
                recommendations.append({
                    "category": category,
                    "action": "add_families",
                    "reason": f"only {summary['families']} fully covered families; target at least {min_families}",
                })
            if summary["families"] and summary["saturated_families"] / summary["families"] >= 0.8:
                recommendations.append({
                    "category": category,
                    "action": "replace_saturated_items",
                    "reason": f"{summary['saturated_families']}/{summary['families']} families pass at or above {saturation_rate:.0%}",
                })
            if summary["families"] and summary["floor_families"] / summary["families"] >= 0.8:
                recommendations.append({
                    "category": category,
                    "action": "review_floor_items",
                    "reason": f"{summary['floor_families']}/{summary['families']} families pass at or below {floor_rate:.0%}; verify keys/contracts before calling them hard",
                })

        calibrated[cohort_key] = {
            "suite_hash": cohort_reports[0].get("suite_hash"),
            "protocol": cohort_reports[0].get("protocol"),
            "run_ids": [report.get("run_id") for report in cohort_reports if report.get("run_id") is not None],
            "reports": len(cohort_reports),
            "fully_covered_questions": sum(
                len(rows) == len(cohort_reports) for rows in by_fingerprint.values()
            ),
            "families": {
                key: rows for key, rows in sorted(families.items())
            },
            "categories": category_rows,
            "recommendations": recommendations,
        }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "saturation_rate": saturation_rate,
            "floor_rate": floor_rate,
            "minimum_families": min_families,
        },
        "note": "Empirical calibration is diagnostic. Missing/failed requests are coverage signals; universal failures require contract and key review before difficulty selection. No author-estimated difficulty or historical result is changed.",
        "cohorts": calibrated,
    }


def reports_from_database(path: str | Path, run_ids: set[int] | None = None) -> list[dict[str, Any]]:
    """Read saved quality reports using one read-only SQLite snapshot."""
    database = Path(path).resolve()
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        rows = db.execute(
            "SELECT id,status,quality_json FROM test_runs ORDER BY id"
        ).fetchall()
        reports = []
        for row in rows:
            if run_ids and row["id"] not in run_ids:
                continue
            if row["status"] != "completed" or not row["quality_json"]:
                continue
            try:
                report = json.loads(row["quality_json"])
            except (TypeError, ValueError):
                continue
            if isinstance(report, dict):
                report.setdefault("run_id", row["id"])
                reports.append(report)
        return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("--runs", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saturation-rate", type=float, default=0.8)
    parser.add_argument("--floor-rate", type=float, default=0.2)
    parser.add_argument("--min-families", type=int, default=5)
    args = parser.parse_args()
    report = calibrate_reports(
        reports_from_database(args.database, set(args.runs) if args.runs else None),
        saturation_rate=args.saturation_rate,
        floor_rate=args.floor_rate,
        min_families=args.min_families,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    for key, cohort in report["cohorts"].items():
        print(f"{key}: {cohort['reports']} runs, {cohort['fully_covered_questions']} fully covered questions")
        for recommendation in cohort["recommendations"]:
            print(f"  {recommendation['category']}: {recommendation['action']} — {recommendation['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
