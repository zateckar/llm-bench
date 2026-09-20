"""Shared quality execution and failure attribution for CLI and web."""

from evaluators import EVALUATORS, extract_json, strip_think_blocks
from llm_client import REPETITION_FINISH_REASON
from models import Result


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
        return result
    if metrics.finish_reason == REPETITION_FINISH_REASON:
        # Kept separate from truncation on purpose: a truncation says the
        # output budget was too small, a loop says raising it would only buy
        # more of the same text. Both are scored failures.
        result.outcome, result.detail = (
            "repetition",
            "Degenerate repetition; generation was stopped and is not a pass",
        )
        return result
    if metrics.finish_reason in {"length", "max_tokens"}:
        result.outcome, result.detail = (
            "truncation",
            "Output limit reached; incomplete answers are not passes",
        )
        return result
    if not strip_think_blocks(response).strip():
        result.outcome, result.detail = "missing_answer", "No final answer; reasoning alone is not an answer"
        return result
    try:
        result.score, result.detail = EVALUATORS[q.evaluator](response, q.expected)
        result.score = max(0.0, min(1.0, float(result.score)))
    except Exception as exc:
        result.outcome, result.detail = "evaluator_error", f"Evaluator error: {exc}"
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
    return result


def execute_question(q, client, cancelled=lambda: False):
    if q.interaction:
        from interactive_tasks import run_interaction

        return run_interaction(q, client, cancelled)
    response, tokens, metrics = client.complete(q.prompt, q.system_prompt, max_tokens=q.max_tokens)
    return score_response(q, response, tokens, metrics)
