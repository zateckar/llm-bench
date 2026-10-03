"""Auditable quality summaries, cluster bootstrap intervals and paired comparisons."""

from collections import Counter, defaultdict
from dataclasses import asdict
import random
import statistics

from models import EVALUATION_SCHEMA_VERSION, percentile
from evaluators import EVALUATOR_VERSIONS
from quality_suite import (
    REVISION,
    MAX_OUTPUT_TOKENS,
    fingerprint,
    suite_hash,
    question_scope,
    provenance,
)

EXCLUDED = {"endpoint_error", "unsupported_context", "evaluator_error", "cancelled"}


def result_record(result):
    q = result.question
    return {
        "id": q.id,
        "fingerprint": fingerprint(q),
        "category": q.category,
        "pass_threshold": q.pass_threshold,
        "difficulty": q.difficulty,
        "weight": q.effective_weight,
        "family": q.metadata.get("family", q.id),
        "scope": q.metadata.get("scope", question_scope(q)),
        "metadata": q.metadata,
        "rubric": q.rubric,
        "score": result.score,
        "passed": result.passed and result.is_scored,
        "scored": result.is_scored,
        "outcome": result.outcome
        or (
            "endpoint_error"
            if not result.metrics.ok
            else "pass"
            if result.passed
            else "task_failure"
        ),
        "detail": result.detail,
        "tokens": asdict(result.tokens),
        "metrics": asdict(result.metrics),
        "cached": result.cached,
        "diagnostics": result.diagnostics,
        "evaluation": result.evaluation.to_dict() if result.evaluation else None,
    }


def clusters(rows, field="score"):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["category"]][row["family"]].append(row[field])
    return {
        category: [statistics.mean(values) for values in families.values()]
        for category, families in sorted(groups.items())
    }


def balanced(groups):
    return (
        statistics.mean(statistics.mean(values) for values in groups.values()) if groups else None
    )


def bootstrap(groups, samples=1000):
    # Variants from a family form one cluster; they never pretend to be
    # independent questions. Categories retain equal weight in each draw.
    if not groups or any(len(group) < 2 for group in groups.values()):
        return None
    rng = random.Random(90210)
    values = [
        statistics.mean(
            statistics.mean(rng.choices(group, k=len(group))) for group in groups.values()
        )
        for _ in range(samples)
    ]
    return [percentile(values, 2.5), percentile(values, 97.5)]


def summarize(rows):
    usable = [r for r in rows if r["scored"]]
    capability = [r for r in usable if r["scope"] == "capability"]
    planned = [r for r in rows if r["scope"] == "capability"]
    compliance = [r for r in usable if r["scope"] == "compliance"]
    achievement = [
        {**r, "score": r["evaluation"]["criterion_achievement"]}
        for r in capability
        if r.get("evaluation") and r["evaluation"].get("criterion_achievement") is not None
    ]
    contract = [
        r["evaluation"]["contract_score"]
        for r in usable
        if r.get("evaluation") and r["evaluation"].get("contract_score") is not None
    ]
    bounds = [
        balanced(
            clusters(
                [{**r, "passed": r["passed"] if r["scored"] else assumed} for r in planned],
                "passed",
            )
        )
        for assumed in (0, 1)
    ]
    groups = clusters(capability, "passed")
    categories = {}
    for category in sorted({r["category"] for r in rows}):
        items = [r for r in usable if r["category"] == category]
        categories[category] = {
            "count": len(items),
            "passes": sum(r["passed"] for r in items),
            "score": balanced(clusters(items, "passed")),
            "families": len({r["family"] for r in items}),
        }
    context = []
    for size in sorted(
        {r["metadata"]["context_tokens"] for r in rows if r["metadata"].get("context_tokens")}
    ):
        items = [r for r in rows if r["metadata"].get("context_tokens") == size]
        scored = [r for r in items if r["scored"]]
        measured = [
            r["metrics"]["prompt_tokens"]
            for r in items
            if r["metrics"].get("ok") and not r["metrics"].get("prompt_tokens_estimated")
        ]
        context.append(
            {
                "reference_tokens": size,
                "count": len(items),
                "scored": len(scored),
                "unsupported": sum(r["outcome"] == "unsupported_context" for r in items),
                "accuracy": statistics.mean(r["passed"] for r in scored) if scored else None,
                "server_prompt_tokens_mean": statistics.mean(measured) if measured else None,
            }
        )
    interactions = [r for r in rows if r["metadata"].get("protocol") == "json-actions-v1"]
    return {
        "count": len(rows),
        "total": len(rows),
        "scored": len(usable),
        "capability_count": len(capability),
        "planned_capability_count": len(planned),
        "excluded_capability_count": len(planned) - len(capability),
        "passes": sum(r["passed"] for r in usable),
        "truncation_rate": sum(r["outcome"] == "truncation" for r in usable) / len(usable)
        if usable
        else None,
        "repetition_rate": sum(r["outcome"] == "repetition" for r in usable) / len(usable)
        if usable
        else None,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "category_balanced": balanced(groups),
        "category_balanced_ci95": bootstrap(groups),
        "category_balanced_criterion_achievement": balanced(clusters(achievement)),
        "capability_criterion_coverage": len(achievement) / len(capability) if capability else None,
        "contract_compliance_mean": statistics.mean(contract) if contract else None,
        "missing_outcome_score_bounds": bounds,
        "compliance_score": statistics.mean(r["passed"] for r in compliance)
        if compliance
        else None,
        "categories": categories,
        "context_quality": context,
        "confidence_method": "Family bootstrap within fixed categories; does not measure run-to-run variability. An interval needs multiple families in every category.",
        "interactive": {
            "tasks": len(interactions),
            "successes": sum(r["passed"] for r in interactions),
            "calls": sum(r["diagnostics"].get("calls", 0) for r in interactions),
            "unnecessary_calls": sum(
                r["diagnostics"].get("unnecessary_calls", 0) for r in interactions
            ),
            "violations": sum(len(r["diagnostics"].get("violations", [])) for r in interactions),
        },
    }


def make_report(results, client_config, selected_hash=None):
    rows = [result_record(r) for r in results]
    return {
        "schema_version": 3,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "suite_hash": selected_hash or suite_hash([r.question for r in results]),
        "model": client_config.model,
        "suite": provenance(),
        "protocol": {
            "revision": REVISION,
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluator_versions": dict(EVALUATOR_VERSIONS),
            "temperature": client_config.temperature,
            "model_seed": client_config.seed,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "summary": summarize(rows),
        "results": rows,
    }


def paired_comparison(left, right):
    if left.get("schema_version") not in {3} or right.get("schema_version") not in {3}:
        return {"compatible": False, "reason": "Missing or unsupported quality report version"}
    if left.get("protocol") != right.get("protocol"):
        return {
            "compatible": False,
            "reason": "Different decoding budgets/settings or quality protocol revisions",
        }
    lrows = {r["fingerprint"]: r for r in left["results"] if r["scope"] == "capability"}
    rrows = {r["fingerprint"]: r for r in right["results"] if r["scope"] == "capability"}
    matched = sorted(lrows.keys() & rrows.keys())
    pairs = [
        {**lrows[key], "score": int(rrows[key]["passed"]) - int(lrows[key]["passed"])}
        for key in matched
        if lrows[key]["scored"] and rrows[key]["scored"]
    ]
    if not pairs:
        return {
            "compatible": False,
            "reason": "No identical capability questions answered by both runs",
        }
    groups = clusters(pairs)
    return {
        "compatible": True,
        "direction": "right minus left",
        "matched": len(matched),
        "paired": len(pairs),
        "excluded_pairs": len(matched) - len(pairs),
        "left_unmatched": len(lrows) - len(matched),
        "right_unmatched": len(rrows) - len(matched),
        "balanced_difference": balanced(groups),
        "ci95": bootstrap(groups),
        "same_suite": left.get("suite_hash") == right.get("suite_hash"),
        "criterion_detail_available": all(r.get("evaluation") is not None for r in pairs),
    }


def markdown(report):
    s = report["summary"]

    def pct(value):
        return "n/a" if value is None else f"{value * 100:.1f}%"

    lines = [
        "## Quality",
        "",
        f"Suite: {REVISION} · {report['suite_hash']}",
        "",
        f"- Balanced strict success: **{pct(s['category_balanced'])}**",
        f"- Capability criterion achievement: {pct(s['category_balanced_criterion_achievement'])}; coverage {pct(s['capability_criterion_coverage'])}.",
        f"- Formal writing compliance: {pct(s['compliance_score'])}; artistic quality is not measured.",
        f"- Scored: {s['scored']}/{s['count']}; full passes: {s['passes']}.",
        "- Outcomes: " + ", ".join(f"{k}: {v}" for k, v in s["outcomes"].items()),
        "",
        s["confidence_method"],
    ]
    if s["excluded_capability_count"]:
        lo, hi = s["missing_outcome_score_bounds"]
        lines += [
            f"Excluded capability outcomes: {s['excluded_capability_count']}/{s['planned_capability_count']}. Zero-to-one sensitivity: {pct(lo)}–{pct(hi)}; not a confidence interval."
        ]
    lines += ["", "| Category | Strict passes | Balanced strict success |", "|---|---:|---:|"]
    lines += [
        f"| {name} | {c['passes']}/{c['count']} | {pct(c['score'])} |"
        for name, c in s["categories"].items()
    ]
    if s["context_quality"]:
        lines += [
            "",
            "Long-context accuracy uses fixed 8k and 32k cl100k_base reference prompts; provider tokenizers can differ.",
        ]
        lines += [
            f"{c['reference_tokens']:,}: accuracy {pct(c['accuracy'])}, {c['unsupported']} unsupported."
            for c in s["context_quality"]
        ]
    return "\n".join(lines)
