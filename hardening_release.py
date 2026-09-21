#!/usr/bin/env python3
"""Build and audit the hardening-v7 profile before a model pilot.

The command is deliberately usable without an endpoint.  It verifies the
candidate bank and writes a manifest; when matched quality reports are passed,
it also computes the provisional 20--80% full-pass selection band and flags
coverage, saturation, floor, and runtime failures for review.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from hardening_cases import build_cases, load_questions, verify_cases
from quality_suite import QualityConfig, assemble_questions, fingerprint, suite_hash


REVISION = "hardening-v7"
SELECTION_BAND = (0.20, 0.80)


def _is_hardening(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    return metadata.get("cohort") == "hardening-v7" or str(row.get("id", "")).startswith("H7-")


def _manifest_payload(*, split: str, variants: int, seed: int) -> dict[str, Any]:
    cases = build_cases(split=split, variants=variants)
    static = load_questions(split=split, variants=variants)
    config = QualityConfig(
        profile="hardening-only",
        generated=False,
        interactive=True,
        strengthen_code=False,
        seeds=(seed,),
        split=split,
        variants=variants,
        context_sizes=(),
    )
    questions = assemble_questions(static, config)
    families = sorted({q.metadata.get("family", q.id) for q in questions})
    categories = sorted({q.category for q in questions})
    fingerprints = {
        q.id: fingerprint(q)
        for q in sorted(questions, key=lambda item: item.id)
    }
    return {
        "revision": REVISION,
        "split": split,
        "seed": seed,
        "variants": variants,
        "raw_candidate_count": len(cases),
        "static_count": len(static),
        "interactive_count": sum(q.interaction is not None for q in questions),
        "profile_count": len(questions),
        "category_count": len(categories),
        "categories": categories,
        "family_count": len(families),
        "families": families,
        "suite_hash": suite_hash(questions),
        "question_fingerprints": fingerprints,
        "oracle_check": verify_cases(split=split, variants=variants),
        "selection_band": {"min_full_pass_rate": SELECTION_BAND[0], "max_full_pass_rate": SELECTION_BAND[1]},
        "note": "Development/evaluation generator outputs are public and deterministic. A release requires a matched pilot, independent review, and frozen profile composition.",
    }


def build_manifest(*, split: str = "development", variants: int = 2, seed: int = 1729) -> dict[str, Any]:
    payload = _manifest_payload(split=split, variants=variants, seed=seed)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "profile": "hardening-only",
        **payload,
    }


def _pilot_summary(
    reports: Iterable[dict[str, Any]],
    *,
    expected_split: str | None = None,
    expected_variants: int | None = None,
) -> dict[str, Any]:
    report_list = [report for report in reports if isinstance(report, dict)]
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    category_by_family: dict[str, str] = {}
    outcomes = defaultdict(int)
    match_keys = []
    match_issues: list[str] = []
    for index, report in enumerate(report_list, 1):
        config = report.get("quality_config")
        protocol = report.get("protocol")
        suite = report.get("suite_hash")
        if not isinstance(config, dict) or not isinstance(protocol, dict) or not suite:
            match_issues.append(f"report {index} is missing suite_hash, quality_config, or protocol")
            continue
        profile = config.get("profile") or protocol.get("quality_profile")
        if profile not in {"hardening", "hardening-only"}:
            match_issues.append(f"report {index} does not use a hardening profile")
        if expected_split is not None and config.get("split") != expected_split:
            match_issues.append(f"report {index} uses split {config.get('split')!r}, expected {expected_split!r}")
        if expected_variants is not None and config.get("variants") != expected_variants:
            match_issues.append(
                f"report {index} uses {config.get('variants')!r} variants, expected {expected_variants!r}"
            )
        # Prices affect accounting only. Every generation-affecting quality and
        # protocol field must match, while the model identity may differ.
        comparable_config = {
            key: value for key, value in config.items()
            if key not in {"input_price", "output_price"}
        }
        match_keys.append(json.dumps({
            "suite_hash": suite,
            "quality_config": comparable_config,
            "protocol": protocol,
        }, sort_keys=True, separators=(",", ":")))
    if match_keys and len(set(match_keys)) != 1:
        match_issues.append("pilot reports do not share one suite, quality configuration, and protocol")
    for report in report_list:
        for row in report.get("results") or []:
            if not isinstance(row, dict) or not _is_hardening(row):
                continue
            family = str(row.get("family") or (row.get("metadata") or {}).get("family") or row.get("id"))
            category_by_family.setdefault(family, str(row.get("category", "unknown")))
            if row.get("scored"):
                rows_by_family[family].append({**row, "_report_id": report.get("run_id")})
            outcomes[str(row.get("outcome", "unknown"))] += 1

    family_rates = {
        family: {
            "category": category_by_family.get(family, "unknown"),
            "observations": len(rows),
            "pass_rate": mean(bool(row.get("passed")) for row in rows) if rows else None,
            "score_mean": mean(float(row.get("score", 0.0)) for row in rows) if rows else None,
            "runs": len({row.get("_report_id") for row in rows if row.get("_report_id") is not None}),
        }
        for family, rows in sorted(rows_by_family.items())
    }
    selected = [
        family for family, row in family_rates.items()
        if row["pass_rate"] is not None and SELECTION_BAND[0] <= row["pass_rate"] <= SELECTION_BAND[1]
    ]
    categories = defaultdict(list)
    for family, row in family_rates.items():
        categories[row["category"]].append(row)
    category_summary = {
        category: {
            "families": len(rows),
            "mean_pass_rate": mean(row["pass_rate"] for row in rows if row["pass_rate"] is not None)
            if any(row["pass_rate"] is not None for row in rows) else None,
            "selected_families": sum(
                row["pass_rate"] is not None and SELECTION_BAND[0] <= row["pass_rate"] <= SELECTION_BAND[1]
                for row in rows
            ),
        }
        for category, rows in sorted(categories.items())
    }
    return {
        "reports": len(report_list),
        "matched_settings": bool(report_list) and not match_issues,
        "match_issues": sorted(set(match_issues)),
        "hardening_rows_scored": sum(len(rows) for rows in rows_by_family.values()),
        "families_observed": len(family_rates),
        "categories_observed": len(category_summary),
        "family_rates": family_rates,
        "category_summary": category_summary,
        "selection_band": {"min": SELECTION_BAND[0], "max": SELECTION_BAND[1]},
        "selected_families": selected,
        "outcomes": dict(sorted(outcomes.items())),
    }


def evaluate_release(
    reports: Iterable[dict[str, Any]] = (),
    *,
    split: str = "development",
    variants: int = 2,
    seed: int = 1729,
) -> dict[str, Any]:
    manifest = build_manifest(split=split, variants=variants, seed=seed)
    pilot = _pilot_summary(
        reports,
        expected_split=split,
        expected_variants=variants,
    )
    oracle_ok = manifest["oracle_check"] == {
        "cases": 58 * variants,
        "families": 58,
        "variants": variants,
    }
    manifest_ok = (
        manifest["category_count"] == 29
        and manifest["family_count"] == 58
        and manifest["raw_candidate_count"] == 58 * variants
        and manifest["profile_count"] == manifest["static_count"] + manifest["interactive_count"]
    )
    if not pilot["reports"]:
        status = "pending_model_pilot"
        release_ready = False
        blockers = ["No matched hardening quality reports were supplied"]
    else:
        blockers = list(pilot["match_issues"])
        if not pilot["matched_settings"]:
            blockers.append("Pilot reports must use one matched hardening suite and protocol")
        if pilot["reports"] < 3:
            blockers.append("At least three matched generation reports are required for provisional selection")
        expected_categories = set(manifest["categories"])
        observed_categories = set(pilot["category_summary"])
        missing_categories = sorted(expected_categories - observed_categories)
        if missing_categories:
            blockers.append(f"Missing scored pilot coverage for categories: {', '.join(missing_categories)}")
        missing_families = sorted(set(manifest["families"]) - set(pilot["family_rates"]))
        if missing_families:
            blockers.append(f"Missing scored pilot coverage for {len(missing_families)} hardening families")
        if not pilot["selected_families"]:
            blockers.append("No family currently falls inside the provisional 20–80% full-pass band")
        if not manifest_ok:
            blockers.append("Hardening manifest coverage/count checks failed")
        if not oracle_ok:
            blockers.append("Hardening independent-oracle checks failed")
        status = "ready" if not blockers and manifest_ok and oracle_ok else "review_required"
        release_ready = status == "ready"
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "release_ready": release_ready,
        "blockers": blockers,
        "manifest": manifest,
        "pilot": pilot,
        "policy": {
            "minimum_reports": 3,
            "selection_band": {"min_full_pass_rate": SELECTION_BAND[0], "max_full_pass_rate": SELECTION_BAND[1]},
            "universal_success_and_failure_require_manual_review": True,
            "runtime_outcomes_are_not_content_difficulty": True,
        },
    }


def _load_reports(paths: Iterable[Path]) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            reports.append(data)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="*", default=[])
    parser.add_argument("--split", choices=("development", "evaluation"), default="development")
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_release(
        _load_reports(args.reports), split=args.split, variants=args.variants, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"{result['status']}: {result['manifest']['profile_count']} questions, "
        f"{result['manifest']['family_count']} families, "
        f"{result['pilot']['reports']} pilot reports"
    )
    for blocker in result["blockers"]:
        print(f"BLOCKER: {blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
