"""Read-only, protocol-matched empirical calibration of saved quality reports.

Observations are descriptive. Variants are not repeated runs, and a small panel
never automatically removes tasks or establishes frontier difficulty.
"""

from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import statistics

from app.benchmarking.quality_report import (
    balanced,
    clusters,
    comparable_protocol,
    paired_comparison,
)
from app.storage import DETECT_TYPES

REVISION = "calibration-v1"
CONFOUNDS = {"truncation", "missing_answer", "repetition", "formatting"}
MIN_MODELS = 2
MIN_REPEATS = 3


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_runs(path):
    """Use one SQLite snapshot; never load credentials or run app migrations."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, detect_types=DETECT_TYPES)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(
            "SELECT r.id, r.model_id, m.name AS model_name, r.status, r.started_at, "
            "r.completed_at, r.total_questions, r.test_suite_hash, r.quality_json "
            "FROM test_runs r LEFT JOIN models m ON m.id=r.model_id ORDER BY r.id"
        )]


def validate_report(run, report):
    if run["status"] != "completed":
        return "run_not_completed"
    if report.get("schema_version") != 3:
        return "unsupported_report_schema"
    if not isinstance(report.get("protocol"), dict) or not report["protocol"]:
        return "missing_protocol"
    if not isinstance(report.get("model"), str) or not report["model"]:
        return "missing_recorded_model_identity"
    required_protocol = {"revision", "evaluation_schema_version", "evaluator_versions", "temperature",
                         "model_seed", "max_output_tokens", "quality_workers", "client"}
    if not required_protocol <= report["protocol"].keys():
        return "incomplete_protocol"
    rows = report.get("results")
    if not isinstance(rows, list) or not rows or len(rows) != run["total_questions"]:
        return "incomplete_result_inventory"
    required = {"id", "fingerprint", "category", "family", "scope", "scored", "passed", "outcome"}
    if any(not isinstance(row, dict) or not required <= row.keys() for row in rows):
        return "invalid_result_record"
    if len({row["id"] for row in rows}) != len(rows):
        return "duplicate_question_ids"
    if (any(not isinstance(row["fingerprint"], str) or len(row["fingerprint"]) != 64
            or any(char not in "0123456789abcdef" for char in row["fingerprint"])
            or not all(isinstance(row[key], str) and row[key] for key in ("id", "category", "family", "scope", "outcome"))
            or row["scope"] not in {"capability", "compliance"}
            or type(row["scored"]) is not bool or type(row["passed"]) is not bool
            or (row["passed"] and not row["scored"])
            or (row["passed"] != (row["outcome"] == "pass")) for row in rows)
            or len({row["fingerprint"] for row in rows}) != len(rows)):
        return "invalid_fingerprint_or_verdict"
    if not any(row["scope"] == "capability" for row in rows):
        return "no_capability_tasks"
    actual_hash = hashlib.sha256("".join(
        row["fingerprint"] for row in sorted(rows, key=lambda row: row["id"])
    ).encode()).hexdigest()[:16]
    if actual_hash != report.get("suite_hash") or actual_hash != run["test_suite_hash"]:
        return "suite_hash_mismatch"
    return None


def summarize_observations(rows):
    scored = [row for row in rows if row["scored"]]
    passed = sum(row["passed"] for row in scored)
    contract_failures = sum(
        not row["passed"] and (row.get("evaluation") or {}).get("contract_score") == 0
        for row in scored if row["outcome"] not in CONFOUNDS
    )
    return {
        "observations": len(rows), "scored": len(scored), "passes": passed,
        "strict_success": passed / len(scored) if scored else None,
        "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        "completion_or_format_failures": sum(row["outcome"] in CONFOUNDS for row in scored),
        "other_contract_failures": contract_failures,
        "observed_pattern": (
            "unscored" if not scored else "all_observed_pass" if passed == len(scored)
            else "no_observed_pass" if not passed else "mixed_observed"
        ),
    }


def analyze_runs(runs, *, suite_hash=None, run_ids=None):
    if len({run["id"] for run in runs}) != len(runs):
        raise ValueError("Duplicate run IDs cannot count as repeated measurements")
    included, excluded, cohorts = [], [], defaultdict(list)
    for run in runs:
        if run_ids is not None and run["id"] not in run_ids:
            continue
        try:
            report = json.loads(run.get("quality_json") or "{}")
            reason = validate_report(run, report) if isinstance(report, dict) else "invalid_report"
        except (ValueError, TypeError, KeyError):
            report, reason = {}, "invalid_report"
        if reason is None and suite_hash and report["suite_hash"] != suite_hash:
            reason = "different_requested_suite"
        if reason:
            excluded.append({"run_id": run["id"], "reason": reason})
            continue
        # Both the protocol and complete identity inventory must match. A
        # truncated hash alone is not sufficient evidence of comparability.
        inventory = sorted((row["id"], row["fingerprint"], row["category"], row["family"], row["scope"])
                           for row in report["results"])
        # Repeats vary only the model seed, so they pool into one cohort.
        key = stable_hash([comparable_protocol(report["protocol"]), inventory])
        entry = {**run, "report": report}
        cohorts[key].append(entry)
        included.append(run["id"])
    output = []
    for key, entries in sorted(cohorts.items()):
        by_model = defaultdict(list)
        for entry in entries:
            # Duplicating a model configuration in the UI cannot manufacture
            # distinct models. Requested IDs are frozen in the saved report.
            by_model[entry["report"]["model"]].append(entry)
        observations = defaultdict(list)
        family_rows = defaultdict(list)
        run_summaries = []
        for entry in entries:
            rows = entry["report"]["results"]
            run_summaries.append({
                "run_id": entry["id"], "model_id": entry["model_id"],
                "model_name": entry["model_name"], "model": entry["report"].get("model"),
                "started_at": entry["started_at"], "completed_at": entry["completed_at"],
                **summarize_observations(rows),
                "category_balanced": balanced(clusters([
                    {**row, "score": int(row["passed"])} for row in rows
                    if row["scope"] == "capability" and row["scored"]
                ])),
            })
            for row in rows:
                observation = {**row, "run_id": entry["id"], "model_id": entry["model_id"],
                               "model_identity": entry["report"]["model"]}
                observations[row["id"]].append(observation)
                if row["scope"] == "capability":
                    family_rows[(row["category"], row["family"])].append(observation)
        questions = []
        for ident, rows in sorted(observations.items()):
            questions.append({
                "id": ident, "fingerprint": rows[0]["fingerprint"],
                "category": rows[0]["category"], "family": rows[0]["family"],
                "scope": rows[0]["scope"], **summarize_observations(rows),
                "runs": [{"run_id": row["run_id"], "passed": row["passed"],
                          "scored": row["scored"], "outcome": row["outcome"],
                          "contract_score": (row.get("evaluation") or {}).get("contract_score")}
                         for row in rows],
            })
        families = []
        for (category, family), rows in sorted(family_rows.items()):
            # Repeated runs cannot give one model more weight in family results.
            model_rates = defaultdict(list)
            for row in rows:
                if row["scored"]:
                    model_rates[row["model_identity"]].append(int(row["passed"]))
            families.append({
                "category": category, "family": family,
                "instances": len({row["id"] for row in rows}),
                **summarize_observations(rows),
                "model_balanced_strict_success": statistics.mean(
                    statistics.mean(values) for values in model_rates.values()
                ) if model_rates else None,
            })
        pairs = []
        for left, right in itertools.combinations(entries, 2):
            comparison = paired_comparison(left["report"], right["report"])
            lrows = {row["id"]: row for row in left["report"]["results"] if row["scope"] == "capability"}
            rrows = {row["id"]: row for row in right["report"]["results"] if row["scope"] == "capability"}
            patterns = Counter((lrows[k]["passed"], rrows[k]["passed"])
                               for k in lrows if lrows[k]["scored"] and rrows[k]["scored"])
            pairs.append({"left_run": left["id"], "right_run": right["id"], **comparison,
                          "both_pass": patterns[True, True], "both_fail": patterns[False, False],
                          "left_only_pass": patterns[True, False], "right_only_pass": patterns[False, True]})
        def complete_runs(values):
            return sum(all(row["scored"] for row in entry["report"]["results"]
                           if row["scope"] == "capability") for entry in values)

        repeats = [{"model_ids": sorted({entry["model_id"] for entry in values}),
                    "model": identity, "runs": len(values),
                    "complete_scored_runs": complete_runs(values)}
                   for identity, values in sorted(by_model.items(), key=lambda item: str(item[0]))]
        ready = len(by_model) >= MIN_MODELS and all(complete_runs(values) >= MIN_REPEATS for values in by_model.values())
        output.append({
            "cohort_id": key, "suite_hash": entries[0]["report"]["suite_hash"],
            "protocol": entries[0]["report"]["protocol"], "runs": run_summaries,
            "distinct_models": len(by_model), "repeats": repeats,
            "admission_evidence": "repeat_panel_available" if ready else "preliminary",
            "questions": questions, "families": families, "paired_comparisons": pairs,
            "capability_patterns": dict(sorted(Counter(q["observed_pattern"] for q in questions
                                                        if q["scope"] == "capability").items())),
        })
    return {
        "revision": REVISION, "snapshot_at": datetime.now(timezone.utc).isoformat(),
        "included_run_ids": included, "excluded_runs": excluded, "cohorts": output,
        "admission_policy": {"minimum_models": MIN_MODELS, "minimum_complete_repeats_per_model": MIN_REPEATS,
                             "automatic_pruning": False},
        "limitations": [
            "Patterns describe only this observed model panel; all-pass does not establish saturation.",
            "Variants and seeds are task instances, not repeated model runs.",
            "Budget, missing-answer and contract failures cannot establish semantic difficulty.",
            "Source correctness gates and adjudication of ambiguous prompts remain necessary.",
            "Historical single-call protocols stay distinct from bounded finalization protocols.",
            "Model identity uses recorded requested IDs; gateway aliases and provider revisions may obscure actual identity.",
        ],
    }


def markdown(report):
    lines = ["# Existing-task calibration", "", f"Snapshot: {report['snapshot_at']}", "",
             "Descriptive evidence only. Historical scores are unchanged; no task is automatically pruned.", ""]
    for cohort in report["cohorts"]:
        lines.extend([f"## Suite {cohort['suite_hash']}", "",
                      f"Evidence: {cohort['admission_evidence']}; {cohort['distinct_models']} models.", "",
                      "| Run | Model | Strict passes | Balanced success | Completion/format failures | Other contract failures |",
                      "|---|---|---:|---:|---:|---:|"])
        for run in cohort["runs"]:
            score = run["category_balanced"]
            label = str(run['model_name']).replace('|', '\\|').replace('\n', ' ')
            score_text = f"{score:.2%}" if score is not None else "unavailable"
            lines.append(f"| {run['run_id']} | {label} | {run['passes']}/{run['scored']} | {score_text} | "
                         f"{run['completion_or_format_failures']} | {run['other_contract_failures']} |")
        lines.extend(["", "Capability task patterns: " + json.dumps(cohort["capability_patterns"], sort_keys=True), "",
                      "### Families with at least one observed failure", "",
                      "| Family | Passes/observations | Completion/format failures | Other contract failures |",
                      "|---|---:|---:|---:|"])
        for family in cohort["families"]:
            if family["passes"] < family["observations"]:
                lines.append(f"| {family['family']} | {family['passes']}/{family['observations']} | "
                             f"{family['completion_or_format_failures']} | {family['other_contract_failures']} |")
        lines.extend(["", "### Paired observations", ""])
        for pair in cohort["paired_comparisons"]:
            lines.append(f"- Runs {pair['left_run']} and {pair['right_run']}: {pair['both_pass']} both pass, "
                         f"{pair['both_fail']} both fail, {pair['left_only_pass']} left-only passes, "
                         f"{pair['right_only_pass']} right-only passes (capability tasks).")
    lines.extend(["", "## Limits", "", *[f"- {limit}" for limit in report["limitations"]], "",
                  "## Excluded runs", "", *[f"- Run {row['run_id']}: {row['reason']}" for row in report["excluded_runs"]]])
    return "\n".join(lines) + "\n"
