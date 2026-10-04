"""Context × concurrency × reasoning sweeps with incremental cell results.

Lengths use cl100k_base reference tokens, not an assertion about the provider's
tokenizer. Timed calls have one attempt. Capability probes are outside timing.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import re
import threading
import time
import uuid

from app.benchmarking.llm_client import ChatClient, _rejects_feature, client_protocol
from app.benchmarking.models import LatencyStats, RequestMetrics
from app.benchmarking.evaluators import strip_think_blocks
from app.benchmarking.cache_metrics import cache_metrics

REVISION = "context-sweep-v2"
MAX_CONTEXT = 1_048_576
CONTEXT_STEP = 16_384
MAX_CONCURRENCY = 256
EFFORTS = ("default", "none", "minimal", "low", "medium", "high", "xhigh", "max")


def grid(baseline, step, maximum):
    return tuple(sorted({baseline, maximum, *range(step, maximum + 1, step)}))


@dataclass(frozen=True)
class SweepConfig:
    context_max: int = MAX_CONTEXT
    max_concurrency: int = MAX_CONCURRENCY
    sweep_rounds: int = 1
    sweep_output_tokens: int = 8192

    def __post_init__(self):
        for name, value, low, high in (
            ("Maximum context", self.context_max, 256, MAX_CONTEXT),
            ("Maximum concurrency", self.max_concurrency, 1, MAX_CONCURRENCY),
            ("Rounds per cell", self.sweep_rounds, 1, 8),
            ("Output token limit", self.sweep_output_tokens, 256, 65_536),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")

    @property
    def contexts(self):
        return grid(256, CONTEXT_STEP, self.context_max)

    @property
    def concurrencies(self):
        return grid(1, 8, self.max_concurrency)

    @property
    def requests_per_effort(self):
        return len(self.contexts) * (2 * sum(self.concurrencies) * self.sweep_rounds + 1)


class PromptBank:
    """One shared padding string per length; only active workers build a prompt.

    Each request starts with a unique nonce to prevent shared-prefix caching.
    Header/footer boundaries are checked once, outside endpoint timing. Worker
    headers are separately tokenized (a few dozen tokens), never the long body.
    """

    footer = "\nIgnore the background. Output the integers 1 through 80, separated by commas, with no prose."

    def __init__(self, context_tokens):
        import tiktoken

        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.context_tokens = context_tokens
        self.padding = " pad" * context_tokens
        self.nonce = uuid.uuid4().hex
        probe = self.prompt(0)
        if len(self.encoding.encode(probe)) != context_tokens:
            raise ValueError("Sweep reference prompt token budget mismatch")

    def prompt(self, index, cache_mode="cold"):
        if cache_mode == "warm":
            # Stable prefix, changing suffix: reuse KV without identical-answer caching.
            header = f"Session {self.nonce}. Background records follow:\n"
            footer = f"\nTurn {index:06d}." + self.footer
        else:
            header = f"Request {self.nonce}-{index}. Background records follow:\n"
            footer = self.footer
        fixed = len(self.encoding.encode(header)) + len(self.encoding.encode(footer))
        return header + self.padding[: 4 * (self.context_tokens - fixed)] + footer


class SweepClient(ChatClient):
    """Allow explicitly rejected sampling fields to be omitted during probes."""

    def __init__(self, config, omitted=()):
        super().__init__(config)
        self.omitted = set(omitted)

    def _payload(self, *args, **kwargs):
        payload = super()._payload(*args, **kwargs)
        for key in self.omitted:
            if key == "max_tokens":
                payload["max_completion_tokens"] = payload.pop(key)
            else:
                payload.pop(key, None)
        return payload


def rejects(error, feature):
    # ChatClient prefixes errors with HTTP status; preserve structured matching.
    text = str(error or "")
    start = text.find("{")
    body = text[start:] if start >= 0 else text
    return _rejects_feature(body, feature) or (
        feature == "reasoning_effort"
        and bool(
            re.search(
                r"(?:invalid|unsupported)\s+(?:value for\s+)?reasoning[_ ]effort|"
                r"reasoning[_ ]effort.{0,180}(?:must be one of|allowed values|not one of)",
                body,
                re.I,
            )
        )
    )


def context_rejection(error):
    text = str(error or "").lower()
    return bool(
        re.search(
            r"context_length_exceeded|maximum context (?:length|window)|"
            r"exceeds? (?:the )?(?:maximum |model.{0,30})?context|"
            r"too many (?:input |prompt )?tokens|input.{0,40}too long",
            text,
        )
    )


def call(client, prompt, output_tokens, retries=1):
    started = time.perf_counter()
    try:
        text, _, metrics = client.complete(prompt, max_tokens=output_tokens, retries=retries)
        return text, metrics
    except Exception as error:
        return "", RequestMetrics(
            ok=False, error=str(error), latency_ms=(time.perf_counter() - started) * 1000
        )


def scalar_stats(values):
    return {
        key.removesuffix("_ms"): value
        for key, value in LatencyStats.from_samples(values).to_dict().items()
    }


def measure_cell(clients, concurrency, bank, config, cancelled, cache_mode="cold", index_offset=0):
    barrier = threading.Barrier(concurrency + 1)

    def worker(slot):
        # First prompt is built before the timed barrier. Later rounds construct
        # only a small header; wall time includes this client overhead explicitly.
        try:
            prompt = bank.prompt(index_offset + slot, cache_mode) if cache_mode == "warm" else bank.prompt(slot)
        except Exception:
            barrier.abort()
            raise
        barrier.wait()
        samples = []
        for round_index in range(config.sweep_rounds):
            if cancelled():
                break
            if round_index:
                index = index_offset + round_index * concurrency + slot
                prompt = bank.prompt(index, cache_mode) if cache_mode == "warm" else bank.prompt(index)
            text, metrics = call(clients[slot], prompt, config.sweep_output_tokens)
            complete = (
                metrics.ok
                and bool(strip_think_blocks(text).strip())
                and metrics.finish_reason
                not in {"length", "max_tokens", "content_filter", "repetition"}
            )
            samples.append((metrics, complete))
        return samples

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        try:
            futures = [pool.submit(worker, slot) for slot in range(concurrency)]
            started = time.perf_counter()
            barrier.wait()
            samples = [sample for future in futures for sample in future.result()]
            wall_ms = (time.perf_counter() - started) * 1000
        finally:
            barrier.abort()
    good = [m for m, complete in samples if complete]
    transport_good = [m for m, _ in samples if m.ok]
    bad = [m for m, _ in samples if not m.ok]
    failures = list(dict.fromkeys(m.error for m in bad if m.error))[:3]
    all_context_rejected = (
        bool(bad) and len(bad) == len(samples) and all(context_rejection(m.error) for m in bad)
    )
    return {
        "context_tokens": bank.context_tokens,
        "cache_mode": cache_mode,
        "cache_metrics": cache_metrics([m for m, _ in samples], wall_ms),
        "concurrency": concurrency,
        "status": "unsupported_context"
        if all_context_rejected
        else ("measured" if good else "incomplete" if transport_good else "failed"),
        "requests": len(samples),
        "completed": len(good),
        "errors": len(bad),
        "incomplete": len(transport_good) - len(good),
        "wall_ms": wall_ms,
        "latency": LatencyStats.from_samples([m.latency_ms for m in good]).to_dict(),
        "ttft": LatencyStats.from_samples([m.ttft_ms for m in transport_good]).to_dict(),
        "request_tokens_per_sec": scalar_stats(
            [m.completion_tokens * 1000 / m.latency_ms for m in good if m.latency_ms > 0]
        ),
        "generation_tokens_per_sec": scalar_stats(
            [
                1000 / m.output_token_time_ms
                for m in good
                if m.output_token_time_ms is not None and m.output_token_time_ms > 0
            ]
        ),
        # Include delivered tokens from truncated responses in aggregate output;
        # complete-answer latency and per-request rates remain separate.
        "aggregate_tokens_per_sec": sum(m.completion_tokens for m in transport_good)
        * 1000
        / wall_ms
        if wall_ms
        else None,
        "prompt_tokens": scalar_stats([m.prompt_tokens for m in transport_good]),
        "estimated_output_requests": sum(m.completion_tokens_estimated for m in transport_good),
        "estimated_input_requests": sum(m.prompt_tokens_estimated for m in transport_good),
        "burst_requests": sum(m.burst_delivery for m in transport_good),
        "cached_tokens": sum(m.cached_tokens for m in transport_good),
        "error": " · ".join(failures),
    }


@dataclass
class SweepReport:
    model: str
    protocol: dict
    efforts: list = field(default_factory=list)
    cells: list = field(default_factory=list)
    measured_cells: int = 0
    resolved_cells: int = 0
    successful_requests: int = 0
    cancelled: bool = False
    finished: bool = False

    def to_dict(self):
        return {
            "schema_version": 4,
            "kind": "context_sweep",
            "model": self.model,
            "protocol": self.protocol,
            "efforts": self.efforts,
            "cells": self.cells,
            "measured_cells": self.measured_cells,
            "resolved_cells": self.resolved_cells,
            "successful_requests": self.successful_requests,
            "cancelled": self.cancelled,
            "finished": self.finished,
        }


def run_sweep(
    client_config,
    config=None,
    *,
    on_cell=None,
    checkpoint=None,
    progress=None,
    cancelled=None,
    retain_cells=True,
):
    config = config or SweepConfig()
    cancelled = cancelled or (lambda: False)
    report = SweepReport(
        client_config.model,
        {
            "revision": REVISION,
            "contexts": list(config.contexts),
            "concurrencies": list(config.concurrencies),
            "candidate_efforts": list(EFFORTS),
            "context_step": CONTEXT_STEP,
            "concurrency_step": 8,
            "reference_tokenizer": "cl100k_base",
            "rounds_per_cell": config.sweep_rounds,
            "max_output_tokens": config.sweep_output_tokens,
            "temperature": client_config.temperature,
            "attempts_per_request": 1,
            "load_model": "closed_loop",
            "requests_per_effort": config.requests_per_effort,
            "timed_requests_per_effort": config.requests_per_effort - len(config.contexts),
            "cache_priming_requests_per_effort": len(config.contexts),
            "ttft_definition": "First received content or reasoning chunk; unavailable for blocking responses.",
            "latency_definition": "Request start to complete answer; excludes failed, empty, filtered and truncated answers.",
            "request_rate_definition": "Reported completion tokens / full request latency; includes provider-reported reasoning tokens.",
            "generation_rate_definition": "(Reported completion tokens - 1) / first-to-last delivery span; excludes estimated counts and burst delivery. A client delivery proxy, not model-side decode speed.",
            "aggregate_rate_definition": "All successful delivered output, including truncated output, / cell wall time including failures and client overhead.",
            "context_definition": "Input text length in cl100k_base reference tokens, excluding chat framing; provider counts are stored separately.",
            "cache_policy": "Paired unique-prefix cold and primed shared-prefix warm cells; unique suffixes prevent identical response caching. Priming is outside timing. Cache hits require provider telemetry.",
            "prefill_rate_definition": "Uncached reported prompt tokens / client TTFT; includes queue, network and first decode overhead. Effective prompt rate includes reused tokens. These are client proxies, not engine timings.",
            "effort_verification": "Probe accepted means API acceptance, not proof that the endpoint honors or distinguishes the effort.",
            "cell_order": "Reasoning effort, increasing context, increasing concurrency; warmed first worker and reused connections.",
        },
    )
    total = len(EFFORTS) * len(config.contexts) * len(config.concurrencies)
    baseline = PromptBank(256)

    def save_report():
        if checkpoint:
            checkpoint(report.to_dict())

    def save(cell):
        report.resolved_cells += 1
        report.measured_cells += bool(cell.get("requests"))
        report.successful_requests += cell.get("completed", 0) + cell.get("cache_reuse", {}).get("completed", 0)
        if retain_cells:
            report.cells.append(cell)
        if on_cell:
            on_cell(cell)
        if progress:
            progress(
                f"{cell['effort']} · {cell['context_tokens']:,} tokens · c={cell['concurrency']}",
                report.resolved_cells,
                total,
            )

    save_report()
    for effort in EFFORTS:
        if cancelled():
            break
        config_for_effort = replace(
            client_config,
            reasoning_effort=None if effort == "default" else effort,
            seed=0,
            detect_repetition=False,
            stream_deadline=1800,
            max_retries=1,
            retry_transport_errors=False,
        )
        clients = [SweepClient(config_for_effort)]
        try:
            warm = clients[0]
            if progress:
                progress(f"Probe reasoning: {effort}", report.resolved_cells, total)
            _, metrics = call(warm, baseline.prompt(0), config.sweep_output_tokens, retries=3)
            # Negotiate only explicitly rejected fields. Reasoning models may
            # require max_completion_tokens and forbid temperature or seed.
            for _ in range(3):
                rejected = next(
                    (
                        key
                        for key in ("temperature", "seed", "max_tokens")
                        if not metrics.ok
                        and rejects(metrics.error, key)
                        and key not in warm.omitted
                    ),
                    None,
                )
                if not rejected or cancelled():
                    break
                warm.omitted.add(rejected)
                _, metrics = call(
                    warm, baseline.prompt(len(warm.omitted)), config.sweep_output_tokens, retries=3
                )
            availability = (
                "accepted"
                if metrics.ok
                else "unsupported"
                if effort != "default" and rejects(metrics.error, "reasoning_effort")
                else "probe_failed"
            )
            report.efforts.append(
                {
                    "effort": effort,
                    "status": availability,
                    "error": metrics.error or "",
                    "omitted_fields": sorted(warm.omitted),
                    "streaming": warm.streaming_supported,
                    "stream_usage": warm._stream_usage_supported,
                    "output_token_parameter": "max_completion_tokens"
                    if "max_tokens" in warm.omitted
                    else "max_tokens",
                    "client": client_protocol(warm.config),
                }
            )
            save_report()
            if not metrics.ok:
                report.resolved_cells += len(config.contexts) * len(config.concurrencies)
                save_report()
                continue
            measured_config = replace(
                warm.config,
                stream=warm.streaming_supported,
                request_stream_usage=warm._stream_usage_supported,
                max_retries=1,
            )
            clients.extend(
                SweepClient(measured_config, warm.omitted)
                for _ in range(config.max_concurrency - 1)
            )
            refused_at = None
            warmed = 1
            for context in config.contexts:
                if cancelled():
                    break
                bank = PromptBank(context) if refused_at is None else None
                cache_primed = False
                cache_prime = None
                warm_index = 1
                failed_load = None
                for concurrency in config.concurrencies:
                    if cancelled():
                        break
                    if refused_at is not None:
                        cell = {
                            "context_tokens": context,
                            "concurrency": concurrency,
                            "status": "skipped_context",
                            "requests": 0,
                            "completed": 0,
                            "errors": 0,
                            "incomplete": 0,
                            "error": f"Not measured: explicit context rejection at {refused_at:,} reference tokens with this output limit.",
                        }
                    elif failed_load is not None:
                        cell = {
                            "context_tokens": context,
                            "concurrency": concurrency,
                            "status": "skipped_failure",
                            "requests": 0,
                            "completed": 0,
                            "errors": 0,
                            "incomplete": 0,
                            "error": f"Not measured: all requests failed at concurrency {failed_load} for this context. Larger contexts still start at concurrency 1.",
                        }
                    else:
                        if concurrency > warmed:
                            if progress:
                                progress(
                                    f"Warm connections · {effort} · c={concurrency}",
                                    report.resolved_cells,
                                    total,
                                )

                            def warm_worker(slot):
                                if not cancelled():
                                    call(
                                        clients[slot],
                                        baseline.prompt(slot + 1000),
                                        config.sweep_output_tokens,
                                    )

                            with ThreadPoolExecutor(max_workers=concurrency - warmed) as pool:
                                list(pool.map(warm_worker, range(warmed, concurrency)))
                            warmed = concurrency
                        if cancelled():
                            break
                        if progress:
                            progress(
                                f"{effort} · {context:,} tokens · c={concurrency}",
                                report.resolved_cells,
                                total,
                            )
                        cell = measure_cell(clients, concurrency, bank, config, cancelled)
                        if cell["status"] == "unsupported_context" and concurrency == 1:
                            refused_at = context
                        elif cell["requests"] and cell["errors"] == cell["requests"]:
                            failed_load = concurrency
                        elif cell.get("completed") and not cancelled():
                            if not cache_primed:
                                _, cache_prime = call(warm, bank.prompt(0, "warm"), config.sweep_output_tokens)
                                cache_primed = True
                            if cache_prime.ok and not cancelled():
                                if progress:
                                    progress(f"Cached prefix · {effort} · {context:,} tokens · c={concurrency}",
                                             report.resolved_cells, total)
                                cell["cache_reuse"] = measure_cell(
                                    clients, concurrency, bank, config, cancelled, "warm", warm_index
                                )
                                cell["cache_reuse"]["priming_ok"] = True
                                warm_index += concurrency * config.sweep_rounds
                            else:
                                cell["cache_reuse"] = {"status": "priming_failed", "priming_ok": False,
                                                       "error": cache_prime.error or "Cancelled"}
                    cell["effort"] = effort
                    if cancelled():
                        cell["cancelled"] = True
                    save(cell)
                save_report()
        finally:
            for client in clients:
                client.session.close()
    report.cancelled = cancelled()
    report.finished = not report.cancelled
    save_report()
    return report
