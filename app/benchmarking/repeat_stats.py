"""Run-to-run variability for repeated quality runs of one model configuration.

A repeat group is a set of saved schema-3 reports that share a suite hash and a
comparison protocol; only the model seed may differ. Members that do not
qualify are excluded with a stated reason and never silently pooled.
"""

from collections import Counter, defaultdict
import json
import statistics

from app.benchmarking.paired_stats import pass_at_k, pass_hat_k, t_interval_95
from app.benchmarking.quality_report import (
    achievement_score,
    balanced,
    bootstrap,
    clusters,
    comparable_protocol,
)

REVISION = "repeat-stats-v1"


def _identity(entry):
    report = entry["report"]
    protocol = comparable_protocol(report.get("protocol"))
    return report.get("suite_hash"), json.dumps(protocol, sort_keys=True, default=str)


def group_members(entries):
    """Split ``[{"run_id", "status", "report"}]`` into usable and excluded runs.

    The reference identity is the most common (suite, protocol) pair among
    completed schema-3 reports; ties go to the earliest run."""
    candidates, excluded = [], []
    for entry in entries:
        report = entry.get("report") or {}
        if entry.get("status") != "completed":
            excluded.append({"run_id": entry["run_id"], "reason": "not_completed"})
        elif report.get("schema_version") != 3 or not report.get("results"):
            excluded.append({"run_id": entry["run_id"], "reason": "missing_quality_report"})
        else:
            candidates.append(entry)
    if not candidates:
        return [], excluded
    counts = Counter(_identity(e) for e in candidates)
    reference = max(counts, key=lambda key: (counts[key], -min(
        e["run_id"] for e in candidates if _identity(e) == key)))
    usable = []
    for entry in candidates:
        suite, protocol = _identity(entry)
        if suite != reference[0]:
            excluded.append({"run_id": entry["run_id"], "reason": "different_suite"})
        elif protocol != reference[1]:
            excluded.append({"run_id": entry["run_id"], "reason": "different_protocol"})
        else:
            usable.append(entry)
    return usable, sorted(excluded, key=lambda row: row["run_id"])


def _spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "sd": None, "min": None, "max": None, "ci95": None, "runs": 0}
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "ci95": t_interval_95(values),
        "runs": len(values),
    }


def group_summary(entries):
    usable, excluded = group_members(entries)
    usable = sorted(usable, key=lambda e: e["run_id"])
    summary = {
        "revision": REVISION,
        "planned_runs": len(entries),
        "usable_runs": len(usable),
        "excluded_runs": excluded,
    }
    if not usable:
        return {**summary, "available": False}
    runs = []
    for entry in usable:
        report = entry["report"]
        s = report.get("summary") or {}
        runs.append({
            "run_id": entry["run_id"],
            "model_seed": (report.get("protocol") or {}).get("model_seed"),
            "score": s.get("category_balanced"),
            "full_pass": s.get("category_balanced_full_pass"),
            "scored": s.get("scored"),
            "total": s.get("total"),
        })
    tasks = {}
    for position, entry in enumerate(usable):
        for row in entry["report"]["results"]:
            if row["scope"] != "capability":
                continue
            task = tasks.setdefault(row["fingerprint"], {
                "id": row["id"], "category": row["category"], "family": row["family"],
                "outcomes": [None] * len(usable), "scores": [], "passes": 0, "scored": 0,
            })
            task["outcomes"][position] = row["outcome"]
            if row["scored"]:
                task["scored"] += 1
                task["scores"].append(achievement_score(row))
                task["passes"] += bool(row["passed"])
    n = len(usable)
    complete = [t for t in tasks.values() if t["scored"] == n]
    curves = []
    for k in range(1, n + 1):
        if not complete:
            break
        curves.append({
            "k": k,
            "pass_hat_k": balanced(clusters([{**t, "value": pass_hat_k(n, t["passes"], k)}
                                             for t in complete], "value")),
            "pass_at_k": balanced(clusters([{**t, "value": pass_at_k(n, t["passes"], k)}
                                            for t in complete], "value")),
        })
    pooled_rows = [{**t, "score": statistics.mean(t["scores"])} for t in tasks.values() if t["scores"]]
    pooled_groups = clusters(pooled_rows)
    flaky = sorted(
        (
            {
                "id": t["id"], "category": t["category"], "family": t["family"],
                "passes": t["passes"], "scored": t["scored"], "outcomes": t["outcomes"],
                "score_mean": statistics.mean(t["scores"]),
                "score_min": min(t["scores"]), "score_max": max(t["scores"]),
            }
            for t in tasks.values()
            if t["scores"] and (0 < t["passes"] < t["scored"] or len(set(t["outcomes"])) > 1)
        ),
        key=lambda t: (t["category"], t["id"]),
    )
    categories = defaultdict(list)
    for entry in usable:
        for name, values in ((entry["report"].get("summary") or {}).get("categories") or {}).items():
            categories[name].append(values.get("score"))
    return {
        **summary,
        "available": True,
        "runs": runs,
        "score_spread": _spread([r["score"] for r in runs]),
        "full_pass_spread": _spread([r["full_pass"] for r in runs]),
        "pooled_score": balanced(pooled_groups),
        "pooled_ci95": bootstrap(pooled_groups),
        "tasks": len(tasks),
        "tasks_scored_in_every_run": len(complete),
        "consistent_pass_fraction": (
            sum(t["passes"] in (0, n) for t in complete) / len(complete) if complete else None
        ),
        "identical_outcome_fraction": (
            sum(len(set(t["outcomes"])) == 1 for t in complete) / len(complete)
            if complete else None
        ),
        "pass_k": curves,
        "flaky_tasks": flaky,
        "categories": {name: _spread(values) for name, values in sorted(categories.items())},
        "method": (
            "Run spread uses each run's category-balanced achievement; the 95% interval is a "
            "Student t interval over runs (three or more). pass^k and pass@k use unbiased "
            "combinatorial estimators over tasks scored in every run, balanced by family and "
            "category. The pooled interval resamples task families only."
        ),
    }
