"""Auditable quality summaries, cluster bootstrap intervals and paired comparisons."""

from collections import Counter, defaultdict
from dataclasses import asdict
import json
import random
import statistics

from models import percentile
from quality_suite import REVISION, fingerprint, suite_hash, question_scope

EXCLUDED = {"endpoint_error", "unsupported_context", "evaluator_error", "cancelled"}


def result_record(result):
    q = result.question
    return {
        "id": q.id,
        "fingerprint": fingerprint(q),
        "category": q.category,
        "family": q.metadata.get("family", q.id),
        "scope": q.metadata.get(
            "scope", question_scope(q)
        ),
        "metadata": q.metadata,
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


def summarize(rows, input_price=None, output_price=None):
    usable = [r for r in rows if r["scored"]]
    capability = [r for r in usable if r["scope"] == "capability"]
    compliance = [r for r in usable if r["scope"] == "compliance"]
    heuristic = [r for r in usable if r["scope"] == "heuristic"]
    groups = clusters(capability)
    cost_rows = [r for r in rows if not r.get("cached")]
    prompt = sum(r["tokens"]["prompt_tokens"] for r in cost_rows)
    completion = sum(r["tokens"]["completion_tokens"] for r in cost_rows)
    cost = (
        (prompt * input_price + completion * output_price) / 1e6
        if input_price is not None and output_price is not None
        else None
    )
    category_rows = {}
    for category in sorted({r["category"] for r in rows}):
        all_items = [r for r in usable if r["category"] == category]
        items = [r for r in all_items if r["scope"] != "heuristic"]
        category_rows[category] = {
            "count": len(items),
            "score": statistics.mean(r["score"] for r in items) if items else None,
            "scope": "compliance" if category == "Creative Writing" else "capability",
            "heuristic_count": sum(r["scope"] == "heuristic" for r in all_items),
        }
    context_rows = []
    for size in sorted(
        {r["metadata"].get("context_tokens") for r in rows if r["metadata"].get("context_tokens")}
    ):
        items = [r for r in rows if r["metadata"].get("context_tokens") == size]
        scored = [r for r in items if r["scored"]]
        measured = [
            r["metrics"]["prompt_tokens"]
            for r in items
            if r["metrics"].get("ok") and not r["metrics"].get("prompt_tokens_estimated")
        ]
        context_rows.append(
            {
                "reference_tokens": size,
                "reference_tokenizer": "cl100k_base",
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
        "scored": len(usable),
        "capability_count": len(capability),
        "all_item_mean": statistics.mean(r["score"] for r in usable) if usable else None,
        "raw_capability_mean": statistics.mean(r["score"] for r in capability)
        if capability
        else None,
        "category_balanced": balanced(groups),
        "category_balanced_ci95": bootstrap(groups),
        "category_balanced_pass_rate": balanced(clusters(capability, "passed")),
        "truncation_rate": sum(r["outcome"] == "truncation" for r in usable) / len(usable)
        if usable else None,
        "budget_note": "Output-limit failures remain in scores. High truncation rates measure "
        "budget-constrained completion as well as capability; compare identical output budgets.",
        "confidence_method": "Percentile bootstrap, 1000 draws; families clustered within fixed categories. "
        "An interval requires at least two families in every included category. "
        "Measures sampled-item uncertainty, not model run-to-run variability.",
        "compliance_score": statistics.mean(r["score"] for r in compliance) if compliance else None,
        "heuristic_count": len(heuristic),
        "heuristic_mean": statistics.mean(r["score"] for r in heuristic) if heuristic else None,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "categories": category_rows,
        "estimated_cost_usd": cost,
        "input_price_per_million": input_price,
        "output_price_per_million": output_price,
        "cost_note": "Estimate for fresh quality calls only, using supplied flat rates; excludes unknown billing, "
        "provider cache discounts and unreported usage from failed requests.",
        "fresh_prompt_tokens": prompt,
        "fresh_completion_tokens": completion,
        "latency_p50_ms": percentile(
            [
                r["metrics"]["latency_ms"]
                for r in rows
                if not r.get("cached") and r["metrics"].get("ok")
            ],
            50,
        ),
        "context_quality": context_rows,
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


def make_report(results, config, client_config, selected_hash=None):
    rows = [result_record(r) for r in results]
    return {
        "schema_version": 1,
        "suite_hash": selected_hash or suite_hash([r.question for r in results]),
        "model": client_config.model,
        "quality_config": asdict(config),
        "protocol": {
            "revision": REVISION,
            "temperature": client_config.temperature,
            "model_seed": client_config.seed,
            "default_max_tokens": client_config.max_tokens,
            "quality_max_output_tokens": config.max_output_tokens,
        },
        "summary": summarize(rows, config.input_price, config.output_price),
        "results": rows,
    }


def paired_comparison(left, right):
    if left.get("schema_version") != 1 or right.get("schema_version") != 1:
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
        {**lrows[key], "score": rrows[key]["score"] - lrows[key]["score"]}
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
    }


def markdown(report):
    s = report["summary"]

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    ci = s["category_balanced_ci95"]
    lines = [
        "## Quality diagnostics",
        "",
        f"- Category/family-balanced capability: **{pct(s['category_balanced'])}**"
        + (
            f" (95% interval {pct(ci[0])}–{pct(ci[1])})"
            if ci
            else " (insufficient families for an interval)"
        ),
        f"- Raw capability mean: {pct(s['raw_capability_mean'])}",
        f"- Category/family-balanced full-pass rate: {pct(s.get('category_balanced_pass_rate'))}",
        f"- Legacy prose-pattern diagnostics: {s.get('heuristic_count', 0)} items, excluded from capability.",
        f"- Output-limit failure rate: {pct(s.get('truncation_rate'))}; these remain scored failures.",
        f"- Creative-writing constraint compliance: {pct(s['compliance_score'])}; artistic quality is not measured.",
        "- Outcomes: " + ", ".join(f"{k}: {v}" for k, v in sorted(s["outcomes"].items())),
        "- Quality-call cost estimate: "
        + (
            f"${s['estimated_cost_usd']:.6f}"
            if s["estimated_cost_usd"] is not None
            else "not configured"
        ),
        "",
        s["confidence_method"],
        "",
        s["cost_note"],
        "",
    ]
    if s["interactive"]["tasks"]:
        a = s["interactive"]
        lines.append(
            f"Interactive tasks: {a['successes']}/{a['tasks']} succeeded; {a['calls']} tool calls, "
            f"{a['unnecessary_calls']} excess calls, {a['violations']} authorization violations."
        )
    if s["context_quality"]:
        lines += [
            "",
            "| Reference tokens (cl100k_base) | Answered | Accuracy | Unsupported | Mean server prompt tokens |",
            "|---:|---:|---:|---:|---:|",
        ]
        for p in s["context_quality"]:
            lines.append(
                f"| {p['reference_tokens']} | {p['scored']}/{p['count']} | {pct(p['accuracy'])} | "
                f"{p['unsupported']} | {p['server_prompt_tokens_mean'] or 'unreported'} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Compare two .quality.json reports without API calls"
    )
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    print(
        json.dumps(
            paired_comparison(
                json.loads(Path(args.left).read_text(encoding="utf-8")),
                json.loads(Path(args.right).read_text(encoding="utf-8")),
            ),
            indent=2,
        )
    )
