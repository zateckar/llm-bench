"""Auditable quality summaries, cluster bootstrap intervals and paired comparisons."""

from collections import Counter, defaultdict
from dataclasses import asdict, replace
import random
import statistics

from app.benchmarking.models import EVALUATION_SCHEMA_VERSION, percentile
from app.benchmarking.evaluators import EVALUATOR_VERSIONS
from app.benchmarking.llm_client import client_protocol
from app.benchmarking.paired_stats import (
    family_differences,
    hierarchical_bootstrap,
    mcnemar_exact,
    sign_flip_test,
    verdict,
)
from app.benchmarking.quality_protocol import protocol as execution_protocol
from app.benchmarking.quality_suite import (
    REVISION,
    MAX_OUTPUT_TOKENS,
    QUALITY_WORKERS,
    fingerprint,
    suite_hash,
    question_scope,
    provenance,
)

EXCLUDED = {"endpoint_error", "unsupported_context", "evaluator_error", "cancelled"}
SCORING_REVISION = "criterion-achievement-v1"


def achievement_score(row):
    value = (row.get("evaluation") or {}).get("criterion_achievement")
    return value if value is not None else row.get("evaluator_score", row.get("score", 0.0))


def rescore_report(report):
    """Project saved criteria into current scores without regrading model answers."""
    if report.get("schema_version") != 3:
        return report
    for row in report["results"]:
        row.setdefault("evaluator_score", row["score"])
        row["score"] = achievement_score(row)
    report["scoring_revision"] = SCORING_REVISION
    report["summary"] = summarize(report["results"])
    return report


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
        "score": result.achievement_score,
        "evaluator_score": result.score,
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
    rows = [{**r, "score": achievement_score(r)} for r in rows]
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
                [{**r, "score": r["score"] if r["scored"] else assumed} for r in planned],
            )
        )
        for assumed in (0, 1)
    ]
    groups = clusters(capability)
    categories = {}
    open_ended = [r for r in rows if r["scope"] == "open_ended"]
    for category in sorted({r["category"] for r in rows if r["scope"] != "open_ended"}):
        items = [r for r in usable if r["category"] == category]
        categories[category] = {
            "count": len(items),
            "passes": sum(r["passed"] for r in items),
            "score": balanced(clusters(items)),
            "full_pass_rate": balanced(clusters(items, "passed")),
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
                "accuracy": statistics.mean(r["score"] for r in scored) if scored else None,
                "server_prompt_tokens_mean": statistics.mean(measured) if measured else None,
            }
        )
    interactions = [r for r in rows if r["metadata"].get("protocol") == "json-actions-v1"]
    recovery_rows = [r for r in rows if r["diagnostics"].get("recovery")]
    native = [r for r in rows if r["metadata"].get("protocol") == "native-tools-v1"]
    extra = {"native": native_summary(native)} if native else {}
    if open_ended:
        # Unscored: answers wait for pairwise A/B studies.
        extra["open_ended"] = {
            "count": len(open_ended),
            "recorded": sum(r["outcome"] == "recorded" for r in open_ended),
            "outcomes": dict(Counter(r["outcome"] for r in open_ended)),
            "categories": sorted({r["category"] for r in open_ended}),
        }
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
        "recovery": {
            "questions": len(recovery_rows),
            "model_calls": sum(len(r["diagnostics"]["quality_requests"]) for r in recovery_rows),
            "initial_passes": sum(r["diagnostics"]["recovery"]["initial_passed"] for r in recovery_rows),
            "attempted": sum(r["diagnostics"]["recovery"]["attempted"] for r in recovery_rows),
            "recovered": sum(r["diagnostics"]["recovery"]["recovered"] for r in recovery_rows),
        },
        "category_balanced": balanced(groups),
        "category_balanced_full_pass": balanced(clusters(capability, "passed")),
        "category_balanced_ci95": bootstrap(groups),
        "category_balanced_criterion_achievement": balanced(clusters(achievement)),
        "capability_criterion_coverage": len(achievement) / len(capability) if capability else None,
        "contract_compliance_mean": statistics.mean(contract) if contract else None,
        "missing_outcome_score_bounds": bounds,
        "compliance_score": statistics.mean(r["score"] for r in compliance)
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
        **extra,
    }


def _failed_criteria(row):
    return [c for c in (row.get("evaluation") or {}).get("criteria") or [] if c.get("status") == "fail"]


def native_summary(rows):
    """Deployment-level tool-calling diagnostics, separate from task quality.

    Counts are per task: a defect code is counted once per task that showed it,
    so one chatty task cannot dominate the picture."""
    usable = [r for r in rows if r["scored"]]
    defects, contract, leaks = Counter(), Counter(), Counter()
    for r in usable:
        defects.update({d.get("code") for d in r["diagnostics"].get("wire_defects") or [] if isinstance(d, dict)})
        for c in _failed_criteria(r):
            if c.get("dimension") == "contract":
                contract[c["id"]] += 1
            if c.get("id") == "no-markup-leak":
                leaks.update((c.get("evidence") or {}).get("patterns") or [])
    rejected = [r for r in rows if r["outcome"] == "feature_rejected"]
    transports = {}
    for transport in ("stream", "blocking"):
        items = [r for r in usable if r["metadata"].get("transport") == transport]
        if items:
            transports[transport] = {
                "tasks": len(items),
                "passes": sum(r["passed"] for r in items),
                "contract_failures": sum(any(c.get("dimension") == "contract" for c in _failed_criteria(r))
                                         for r in items),
            }
    simulations = [r for r in usable if r["metadata"].get("kind") == "simulation"]
    return {
        "tasks": len(rows),
        "scored": len(usable),
        "passes": sum(r["passed"] for r in usable),
        "feature_rejected": len(rejected),
        "rejected_families": sorted({r["family"] for r in rejected}),
        "rejection_status": dict(Counter(str(r["metrics"].get("http_status")) for r in rejected)),
        "contract_failures": dict(contract.most_common()),
        "wire_defects": dict(defects.most_common()),
        "markup_leaks": dict(leaks.most_common()),
        "transports": transports,
        "simulations": {
            "tasks": len(simulations),
            "successes": sum(r["passed"] for r in simulations),
            "calls": sum(r["diagnostics"].get("calls", 0) for r in simulations),
            "unnecessary_calls": sum(r["diagnostics"].get("unnecessary_calls", 0) for r in simulations),
            "violations": sum(len(r["diagnostics"].get("violations", [])) for r in simulations),
        },
    }


def make_report(results, client_config, selected_hash=None, max_concurrency=8, suite=None):
    rows = [result_record(r) for r in results]
    client = client_protocol(replace(client_config, detect_repetition=False,
                                     retry_transport_errors=False,
                                     max_retries=min(3, client_config.max_retries)))
    if suite is None or suite.name == "rigorous":
        # The historical rigorous report shape is unchanged so saved runs stay
        # comparable; other suites record their name, protocol and client extras.
        suite_block = provenance()
        revision = REVISION
        execution = execution_protocol(client_config.max_tokens)
    else:
        suite_block = {**suite.provenance(), "name": suite.name}
        revision = suite.revision()
        execution = suite.execution(client_config.max_tokens)
        client = {**client, **suite.client_extra()}
    return {
        "schema_version": 3,
        "scoring_revision": SCORING_REVISION,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "suite_hash": selected_hash or suite_hash([r.question for r in results]),
        "model": client_config.model,
        "suite": suite_block,
        "protocol": {
            "revision": revision,
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluator_versions": dict(EVALUATOR_VERSIONS),
            "temperature": client_config.temperature,
            "reasoning_effort": client_config.reasoning_effort,
            "execution": execution,
            "model_seed": client_config.seed,
            "max_output_tokens": min(MAX_OUTPUT_TOKENS, client_config.max_tokens),
            "quality_workers": min(QUALITY_WORKERS, max_concurrency),
            "client": client,
        },
        "summary": summarize(rows),
        "results": rows,
    }


def comparable_protocol(protocol):
    """Protocol identity used for comparisons.

    The model seed is a sampling detail: repeated runs deliberately vary it, so
    it is recorded and reported but does not by itself prevent pairing."""
    if not isinstance(protocol, dict):
        return protocol
    identity = {key: value for key, value in protocol.items() if key != "model_seed"}
    if isinstance(identity.get("client"), dict):
        identity["client"] = {
            key: value for key, value in identity["client"].items() if key != "model_seed"
        }
    return identity


def _capability_rows(report):
    return [r for r in report["results"] if r["scope"] == "capability"]


def compare_groups(lefts, rights):
    """Paired comparison of two run groups (single runs are groups of one).

    Tasks pair by fingerprint. Each task contributes the right-minus-left
    difference of its mean achievement over scored repeats; families and
    categories keep the headline weighting. The interval resamples families
    and repeats; the p-value is a family-level sign-flip randomization test."""
    reports = [*lefts, *rights]
    if not lefts or not rights:
        return {"compatible": False, "reason": "Both sides need at least one run"}
    if any(r.get("schema_version") not in {3} for r in reports):
        return {"compatible": False, "reason": "Missing or unsupported quality report version"}
    if len({(r.get("suite") or {}).get("name") or "rigorous" for r in reports}) > 1:
        return {"compatible": False, "reason": "Different quality suites are never paired"}
    identity = comparable_protocol(reports[0].get("protocol"))
    if any(comparable_protocol(r.get("protocol")) != identity for r in reports):
        return {
            "compatible": False,
            "reason": "Different decoding budgets/settings or quality protocol revisions",
        }

    def side(group):
        rows, values, passes = {}, defaultdict(list), defaultdict(list)
        for report in group:
            for row in _capability_rows(report):
                rows.setdefault(row["fingerprint"], row)
                if row["scored"]:
                    values[row["fingerprint"]].append(achievement_score(row))
                    passes[row["fingerprint"]].append(bool(row["passed"]))
        return rows, values, passes

    lrows, lvalues, lpasses = side(lefts)
    rrows, rvalues, rpasses = side(rights)
    matched = sorted(lrows.keys() & rrows.keys())
    paired = [key for key in matched if lvalues[key] and rvalues[key]]
    if not paired:
        return {
            "compatible": False,
            "reason": "No identical capability questions answered by both runs",
        }
    tasks = defaultdict(lambda: defaultdict(list))
    for key in paired:
        row = lrows[key]
        tasks[row["category"]][row["family"]].append((lvalues[key], rvalues[key]))
    groups = family_differences(tasks)
    difference = balanced(groups)
    interval = hierarchical_bootstrap(tasks)
    p_value = sign_flip_test(groups)
    single = len(lefts) == 1 and len(rights) == 1
    discordance = None
    if single:
        patterns = Counter((lpasses[key][0], rpasses[key][0]) for key in paired)
        discordance = {
            "both_pass": patterns[True, True],
            "both_fail": patterns[False, False],
            "left_only_pass": patterns[True, False],
            "right_only_pass": patterns[False, True],
        }
        discordance["mcnemar_p"] = mcnemar_exact(
            discordance["left_only_pass"], discordance["right_only_pass"]
        )
    seeds = sorted({(r.get("protocol") or {}).get("model_seed") for r in reports}, key=str)
    return {
        "compatible": True,
        "direction": "right minus left",
        "score_basis": SCORING_REVISION,
        "left_runs": len(lefts),
        "right_runs": len(rights),
        "matched": len(matched),
        "paired": len(paired),
        "excluded_pairs": len(matched) - len(paired),
        "left_unmatched": len(lrows) - len(matched),
        "right_unmatched": len(rrows) - len(matched),
        "balanced_difference": difference,
        "ci95": interval,
        "interval_method": "Hierarchical bootstrap: families within categories, then repeats within tasks."
        if not single
        else "Family bootstrap within fixed categories.",
        "p_value": p_value,
        "test": "Two-sided family-level sign-flip randomization test (10,000 permutations, fixed seed).",
        "verdict": verdict(difference, interval, p_value),
        "full_pass_discordance": discordance,
        "same_suite": len({r.get("suite_hash") for r in reports}) == 1,
        "same_seed": len(seeds) == 1,
        "model_seeds": seeds,
        "criterion_detail_available": all(
            lrows[key].get("evaluation") is not None and rrows[key].get("evaluation") is not None
            for key in paired
        ),
    }


def paired_comparison(left, right):
    return compare_groups([left], [right])


