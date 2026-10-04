"""Bounded final-answer recovery without evaluator feedback or answer selection."""

from dataclasses import asdict, replace

from app.benchmarking.evaluators import strip_think_blocks
from app.benchmarking.models import RequestMetrics, TokenUsage
from app.benchmarking.quality_suite import MAX_OUTPUT_TOKENS

REVISION = "bounded-quality-v1"
FINAL_RESERVE = 8192
FINALIZE_PROMPT = (
    "The previous response did not complete the requested answer. Use the work above "
    "to produce one complete final answer following the original request. "
    "Finish your analysis and provide the answer now. Return the entire requested "
    "output, not a continuation fragment. Do not discuss this instruction."
)


def protocol(total_budget=MAX_OUTPUT_TOKENS):
    budget = min(MAX_OUTPUT_TOKENS, total_budget)
    reserve = min(FINAL_RESERVE, budget // 8)
    return {
        "revision": REVISION,
        "scope": "noninteractive_questions",
        "max_model_calls": 2,
        "total_output_budget": budget,
        "final_answer_reserve": reserve,
        "reserve_for_smaller_budgets": "min(8192, total // 8)",
        "recovery_triggers": ["length", "max_tokens", "reasoning_only"],
        "followup_prompt": FINALIZE_PROMPT,
        "history": "original messages plus full previous response",
        "answer_selection": "last response only, complete replacement",
        "evaluator_feedback": False,
        "transport_retries": False,
        "capability_negotiation_attempts_per_call": 3,
        "latency_scope": "sum of all model calls including capability negotiation",
    }


def complete_bounded(q, client, cancelled):
    client.config = replace(client.config, retry_transport_errors=False,
                            max_retries=min(3, client.config.max_retries))
    budget = min(MAX_OUTPUT_TOKENS, q.max_tokens or client.config.max_tokens)
    reserve = min(FINAL_RESERVE, budget // 8)
    limits = [budget - reserve, reserve]
    calls = []
    diagnostics = {"quality_requests": calls, "recovery": {"attempted": False}}
    totals = TokenUsage()
    response, last_metrics = "", RequestMetrics(attempts=0)
    trigger = None

    for index, limit in enumerate(limits):
        if not limit:
            break
        if cancelled():
            diagnostics["cancelled"] = True
            break
        if index == 0:
            response, usage, metrics = client.complete(
                q.prompt, q.system_prompt, max_tokens=limit
            )
        else:
            messages = []
            if q.system_prompt:
                messages.append({"role": "system", "content": q.system_prompt})
            messages.extend([
                {"role": "user", "content": q.prompt},
                {"role": "assistant", "content": response},
                {"role": "user", "content": FINALIZE_PROMPT},
            ])
            diagnostics["recovery"]["attempted"] = True
            response, usage, metrics = client.complete_messages(messages, max_tokens=limit)

        totals.prompt_tokens += usage.prompt_tokens
        totals.completion_tokens += usage.completion_tokens
        totals.cached_tokens += usage.cached_tokens
        totals.prompt_tokens_estimated |= usage.prompt_tokens_estimated
        totals.completion_tokens_estimated |= usage.completion_tokens_estimated
        calls.append({
            "request": index + 1,
            "max_output_tokens": limit,
            "response": response,
            "tokens": asdict(usage),
            "metrics": asdict(metrics),
        })
        last_metrics = metrics
        # An endpoint that ignores the allocation cannot provide a conforming
        # bounded result. Missing usage is already estimated by ChatClient.
        if not usage.completion_tokens_estimated and usage.completion_tokens > limit:
            last_metrics = replace(
                metrics, ok=False, error="Endpoint exceeded the requested output budget"
            )
            break
        if index or not metrics.ok:
            break
        if metrics.finish_reason in {"length", "max_tokens"}:
            trigger = "truncation"
        elif (response.strip() and not strip_think_blocks(response).strip()
              and metrics.finish_reason in {None, "stop"}):
            trigger = "missing_answer"
        else:
            break
        diagnostics["recovery"]["initial_outcome"] = trigger

    # A single successful answer is already stored as Result.response. Keep
    # full outputs here when there was recovery, including a cancelled followup.
    if trigger is None:
        for call in calls:
            call.pop("response", None)
    elapsed = sum(call["metrics"]["latency_ms"] for call in calls)
    totals.cached_tokens_reported = bool(calls) and all(
        call["tokens"].get("cached_tokens_reported", False) for call in calls
    )
    ttft = None
    preceding = 0.0
    for call in calls:
        metrics = call["metrics"]
        if ttft is None and metrics["ttft_ms"] is not None:
            ttft = preceding + metrics["ttft_ms"]
        preceding += metrics["latency_ms"]
    aggregate = replace(
        last_metrics,
        latency_ms=elapsed,
        ttft_ms=ttft,
        prompt_tokens=totals.prompt_tokens,
        completion_tokens=totals.completion_tokens,
        cached_tokens=totals.cached_tokens,
        cached_tokens_reported=bool(calls) and all(
            call["metrics"].get("cached_tokens_reported", False) for call in calls
        ),
        prompt_tokens_estimated=totals.prompt_tokens_estimated,
        completion_tokens_estimated=totals.completion_tokens_estimated,
        attempts=sum(call["metrics"]["attempts"] for call in calls),
        streamed=bool(calls) and all(call["metrics"]["streamed"] for call in calls),
        stream_chunks=sum(call["metrics"]["stream_chunks"] for call in calls),
        # Combining streams must not invent a continuous token-delivery span.
        stream_span_ms=last_metrics.stream_span_ms if len(calls) == 1 else None,
        successful_attempt_latency_ms=(
            last_metrics.successful_attempt_latency_ms if len(calls) == 1 else None
        ),
        attempt_diagnostics=[
            {**attempt, "request": call["request"]}
            for call in calls for attempt in call["metrics"]["attempt_diagnostics"]
        ],
    )
    return response, totals, aggregate, diagnostics
