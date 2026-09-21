"""Shared quality execution and failure attribution for CLI and web."""

from evaluators import EVALUATOR_VERSIONS, EVALUATORS, extract_json, strip_think_blocks
from llm_client import REPETITION_FINISH_REASON
from models import CriterionResult, EvaluationResult, Result


def _evaluation_for(
    q,
    *,
    score: float,
    full_pass: bool,
    outcome: str,
    sink: dict | None = None,
    reason_code: str | None = None,
) -> EvaluationResult:
    """Normalize evaluator diagnostics into the persisted evaluation schema."""
    sink = sink or {}
    raw = sink.get("criteria") or []
    criteria: list[CriterionResult] = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        criteria.append(
            CriterionResult(
                criterion_id=str(item.get("id") or f"criterion-{i:03d}"),
                status=str(item.get("status", "not_evaluated")),
                earned=float(item.get("earned", 0.0)),
                possible=float(item.get("possible", 1.0)),
                dimension=str(item.get("dimension", "content")),
                # A scorer's fraction and the question's pass_threshold remain
                # the compatibility gate. Evaluators opt a criterion into a
                # stricter gate explicitly (or through a rubric), which keeps
                # intentional partial-credit thresholds meaningful.
                mandatory=bool(item.get("mandatory", False)),
                critical=bool(item.get("critical", False)),
                group=str(item["group"]) if item.get("group") is not None else None,
                depends_on=[str(dep) for dep in (item.get("depends_on") or [])],
                reason_code=str(item.get("reason_code", "")),
                evidence=dict(item.get("evidence") or {}),
            )
        )
    rubric = {str(item.get("id")): item for item in (getattr(q, "rubric", None) or [])}
    if rubric:
        for criterion in criteria:
            spec = rubric.get(criterion.criterion_id)
            if not spec:
                continue
            raw_ratio = (
                criterion.earned / criterion.possible
                if criterion.possible
                else 0.0
            )
            weight = float(spec.get("weight", criterion.possible or 1.0))
            criterion.earned = raw_ratio * weight
            criterion.possible = weight
            criterion.dimension = str(spec.get("dimension", criterion.dimension))
            criterion.mandatory = bool(spec.get("mandatory", True))
            criterion.critical = bool(spec.get("critical", criterion.critical))
            criterion.group = str(spec["group"]) if spec.get("group") is not None else criterion.group
            criterion.depends_on = [str(dep) for dep in (spec.get("depends_on") or criterion.depends_on)]
            if spec.get("group") is not None:
                criterion.evidence = {
                    **criterion.evidence,
                    "group": str(spec["group"]),
                }
        emitted = {criterion.criterion_id for criterion in criteria}
        for criterion_id, spec in rubric.items():
            if criterion_id in emitted:
                continue
            criteria.append(
                CriterionResult(
                    criterion_id=criterion_id,
                    status="not_evaluated",
                    earned=0.0,
                    possible=float(spec.get("weight", 1.0)),
                    dimension=str(spec.get("dimension", "content")),
                    mandatory=bool(spec.get("mandatory", True)),
                    critical=bool(spec.get("critical", False)),
                    group=str(spec["group"]) if spec.get("group") is not None else None,
                    depends_on=[str(dep) for dep in (spec.get("depends_on") or [])],
                    reason_code="rubric_criterion_unavailable",
                )
            )
    if not criteria:
        # Legacy evaluators remain useful and are explicitly marked as one
        # criterion rather than receiving invented sub-requirements.
        criteria = [
            CriterionResult(
                criterion_id="task",
                status="pass" if full_pass else "fail",
                earned=score,
                possible=1.0,
                dimension="content",
                reason_code=reason_code or ("full_match" if full_pass else "task_mismatch"),
            )
        ]
    gate_failed = any(
        (criterion.mandatory or criterion.critical)
        and criterion.status in {"fail", "error", "not_evaluated"}
        for criterion in criteria
    )
    full_pass = bool(full_pass and not gate_failed)
    if gate_failed and outcome == "pass":
        outcome = "task_failure"
    return EvaluationResult(
        evaluator=q.evaluator,
        score=float(score),
        full_pass=bool(full_pass),
        outcome=outcome,
        criteria=criteria,
        contract_score=sink.get("contract_score"),
        evaluator_version=EVALUATOR_VERSIONS.get(q.evaluator, "1"),
    )


def _availability_evaluation(q, outcome: str, reason: str) -> EvaluationResult:
    return _evaluation_for(
        q,
        score=0.0,
        full_pass=False,
        outcome=outcome,
        sink={
            "criteria": [{
                "id": "response",
                "status": "not_evaluated" if outcome in {"endpoint_error", "unsupported_context", "cancelled"} else "fail",
                "earned": 0.0,
                "possible": 1.0,
                "dimension": "availability",
                "reason_code": outcome,
                "evidence": {"detail": reason[:240]},
            }],
            "contract_score": None,
        },
        reason_code=outcome,
    )


def score_response(q, response, tokens, metrics, cached=False):
    result = Result(q, response, 0.0, tokens=tokens, metrics=metrics, cached=cached)
    if not metrics.ok:
        error = str(metrics.error or "").lower()
        context_refusal = any(
            s in error
            for s in (
                "context_length",
                "context window",
                "maximum context",
                "too many tokens",
                "http 413",
            )
        )
        result.outcome = (
            "unsupported_context"
            if q.metadata.get("context_tokens") and context_refusal
            else "endpoint_error"
        )
        result.detail = f"Request failed: {metrics.error}"
        result.evaluation = _availability_evaluation(q, result.outcome, result.detail)
        return result
    if metrics.finish_reason == REPETITION_FINISH_REASON:
        # Kept separate from truncation on purpose: a truncation says the
        # output budget was too small, a loop says raising it would only buy
        # more of the same text. Both are scored failures.
        result.outcome, result.detail = (
            "repetition",
            "Degenerate repetition; generation was stopped and is not a pass",
        )
        result.evaluation = _availability_evaluation(q, result.outcome, result.detail)
        return result
    if metrics.finish_reason in {"length", "max_tokens"}:
        result.outcome, result.detail = (
            "truncation",
            "Output limit reached; incomplete answers are not passes",
        )
        result.evaluation = _availability_evaluation(q, result.outcome, result.detail)
        return result
    if not strip_think_blocks(response).strip():
        result.outcome, result.detail = "missing_answer", "No final answer; reasoning alone is not an answer"
        result.evaluation = _availability_evaluation(q, result.outcome, result.detail)
        return result
    diagnostic_sink: dict = {}
    try:
        result.score, result.detail = EVALUATORS[q.evaluator](
            response, q.expected, _diagnostics=diagnostic_sink
        )
        result.score = max(0.0, min(1.0, float(result.score)))
    except Exception as exc:
        result.outcome, result.detail = "evaluator_error", f"Evaluator error: {exc}"
        result.evaluation = _availability_evaluation(q, result.outcome, result.detail)
        return result
    if result.detail.startswith(("Misconfigured question:", "Unknown evaluator:")):
        result.outcome = "evaluator_error"
        result.detail = "Evaluator error: " + result.detail
    elif result.passed:
        result.outcome = "pass"
    elif q.metadata.get("scope") == "heuristic":
        result.outcome = "heuristic_mismatch"
    elif q.evaluator == "json_match" and extract_json(
        response, strict=isinstance(q.expected, dict) and q.expected.get("strict_json", False)
    )[1]:
        result.outcome = "formatting"
    elif q.category == "Creative Writing":
        result.outcome = "compliance_failure"
    elif q.evaluator == "format_check":
        result.outcome = "formatting"
    else:
        # A wrong answer cannot establish its internal cause; do not assert that
        # every factual mistake or runtime bug is a reasoning failure.
        result.outcome = "task_failure"
    result.evaluation = _evaluation_for(
        q,
        score=result.score,
        full_pass=result.passed,
        outcome=result.outcome,
        sink=diagnostic_sink,
    )
    if result.outcome == "pass" and not result.evaluation.full_pass:
        result.outcome = result.evaluation.outcome = "task_failure"
    return result


def execute_question(q, client, cancelled=lambda: False):
    if q.interaction:
        from interactive_tasks import run_interaction

        return run_interaction(q, client, cancelled)
    response, tokens, metrics = client.complete(q.prompt, q.system_prompt, max_tokens=q.max_tokens)
    return score_response(q, response, tokens, metrics)
