"""Fixed-workload latency and wall-clock throughput at bounded concurrency.

A closed-loop load test: concurrency is a number of simultaneous requests,
not an offered arrival rate or a claim about user capacity. No model-side
decode/prefill speed is inferred from client stream timings.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import threading
import time
import uuid

from app.benchmarking.llm_client import ChatClient, client_protocol
from app.benchmarking.models import ConcurrencyPoint, LatencyStats, PerfReport, RequestMetrics
from app.benchmarking.cache_metrics import cache_metrics

REVISION = "performance-v8"
DEFAULT_MAX_CONCURRENCY = 8
MAX_CONCURRENCY = 32
MIN_SAMPLES = 24
ROUNDS = 4
INPUT_REFERENCE_TOKENS = 1024
OUTPUT_TOKENS = 256
CACHE_CONTEXTS = (8192, 32768)


@dataclass(frozen=True)
class PerfConfig:
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY

    def __post_init__(self):
        if (
            type(self.max_concurrency) is not int
            or not 1 <= self.max_concurrency <= MAX_CONCURRENCY
        ):
            raise ValueError(
                f"Maximum concurrency must be an integer between 1 and {MAX_CONCURRENCY}"
            )

    @property
    def levels(self):
        levels = {1, self.max_concurrency}
        level = 2
        while level < self.max_concurrency:
            levels.add(level)
            level *= 2
        return tuple(sorted(levels))

    def requests_for_level(self, level):
        return max(MIN_SAMPLES, ROUNDS * level)

    def total_requests(self):
        cache_requests = len(CACHE_CONTEXTS) * sum(1 + 2 * max(4, c) for c in {1, self.max_concurrency})
        return self.max_concurrency + 1 + sum(self.requests_for_level(c) for c in self.levels) + cache_requests


def _prompt(nonce, index):
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    header = f"Request {nonce}-{index:06d}. The following records are background only.\n"
    footer = "\nIgnore the background. Output the integers 1 through 80, separated by commas, with no prose."
    filler = (
        "Archive record: the amber station reports normal operations and no pending maintenance. "
    )
    budget = INPUT_REFERENCE_TOKENS - len(encoding.encode(header + footer)) - 8
    text = encoding.decode(encoding.encode(filler * 200)[:budget])
    prompt = header + text + footer
    while len(encoding.encode(prompt)) < INPUT_REFERENCE_TOKENS:
        prompt = prompt.replace(footer, " pad" + footer, 1)
    if len(encoding.encode(prompt)) != INPUT_REFERENCE_TOKENS:
        raise ValueError("Performance prompt token budget mismatch")
    return prompt


def _probe(client, prompt):
    started = time.perf_counter()
    try:
        _text, _usage, metrics = client.complete(prompt, max_tokens=OUTPUT_TOKENS, retries=1)
        return metrics
    except Exception as error:
        return RequestMetrics(
            ok=False, error=str(error), latency_ms=(time.perf_counter() - started) * 1000
        )


def _measure_level(clients, level, prompts, cancelled):
    barrier = threading.Barrier(level + 1)
    lock = threading.Lock()
    samples = []

    def worker(slot):
        barrier.wait()
        for index in range(slot, len(prompts), level):
            if cancelled():
                break
            metrics = _probe(clients[slot], prompts[index])
            with lock:
                samples.append(metrics)

    with ThreadPoolExecutor(max_workers=level) as pool:
        futures = [pool.submit(worker, i) for i in range(level)]
        started = time.perf_counter()
        started_at = time.time()
        barrier.wait()
        for future in futures:
            future.result()
        wall_ms = (time.perf_counter() - started) * 1000
        ended_at = time.time()
    good = [m for m in samples if m.ok]
    bad = [m for m in samples if not m.ok]
    return ConcurrencyPoint(
        level,
        len(samples),
        len(bad),
        wall_ms,
        LatencyStats.from_samples([m.latency_ms for m in good]),
        LatencyStats.from_samples([m.ttft_ms for m in good if m.ttft_ms is not None]),
        sum(m.completion_tokens for m in good),
        sum(m.prompt_tokens for m in good),
        failure_latency=LatencyStats.from_samples([m.latency_ms for m in bad]),
        estimated_token_requests=sum(m.completion_tokens_estimated for m in good),
        streamed_requests=sum(m.streamed for m in good),
        burst_delivery_requests=sum(m.burst_delivery for m in good),
        cached_tokens=sum(m.cached_tokens for m in good),
        cache_metrics=cache_metrics(samples, wall_ms),
        estimated_prompt_requests=sum(m.prompt_tokens_estimated for m in good),
        started_at=started_at,
        ended_at=ended_at,
        output_token_time=LatencyStats.from_samples([m.output_token_time_ms for m in good]),
        chunk_gap=LatencyStats.from_samples([gap for m in good for gap in m.chunk_gaps_ms]),
        output_length=LatencyStats.from_samples([m.completion_tokens for m in good]),
        samples=[{
            "ok": m.ok, "latency_ms": m.latency_ms, "ttft_ms": m.ttft_ms,
            "completion_tokens": m.completion_tokens,
            "prompt_tokens": m.prompt_tokens, "cached_tokens": m.cached_tokens,
            "cached_tokens_reported": m.cached_tokens_reported,
            "output_token_time_ms": m.output_token_time_ms,
            "estimated_tokens": m.completion_tokens_estimated,
            "burst_delivery": m.burst_delivery, "error": m.error,
        } for m in samples],
    )


def run_perf_suite(client_config, config=None, progress=None, cancelled=None):
    client_config = replace(client_config, stream_deadline=None)
    config = config or PerfConfig()
    cancelled = cancelled or (lambda: False)
    report = PerfReport(
        client_config.model,
        protocol={
            "revision": REVISION,
            "load_model": "closed_loop",
            "max_concurrency": config.max_concurrency,
            "levels": list(config.levels),
            "minimum_samples": MIN_SAMPLES,
            "rounds_per_level": ROUNDS,
            "input_reference_tokens": INPUT_REFERENCE_TOKENS,
            "reference_tokenizer": "cl100k_base",
            "max_output_tokens": OUTPUT_TOKENS,
            "temperature": client_config.temperature,
            "reasoning_effort": client_config.reasoning_effort,
            "model_seed": 0,
            "attempts_per_request": 1,
            "output_token_time_definition": "first-to-last delivery span / (reported completion tokens - 1); excludes estimated counts and bursts",
            "chunk_gap_definition": "interval between content/reasoning SSE chunks; not individual token latency",
            "sample_scope": "timed requests only, including errors; successful requests for distribution percentiles",
            "cache_context_tokens": list(CACHE_CONTEXTS),
            "cache_policy": "Paired unique-prefix cold and primed shared-prefix warm requests, with changing suffixes. Priming outside timing; actual hits require provider telemetry.",
            "prefill_rate_definition": "Uncached reported prompt tokens / client TTFT; includes queue, network and first decode overhead. Effective prompt rate includes reused tokens; rates are client proxies.",
        },
    )
    nonce = uuid.uuid4().hex
    count, total = 0, config.total_requests()
    clients = []

    def notify(phase):
        nonlocal count
        count += 1
        if progress:
            progress(phase, count, total)

    try:
        # Negotiate streaming before timed samples; no retries in measured calls.
        warm = ChatClient(replace(client_config, seed=0, detect_repetition=False))
        clients.append(warm)
        warm_success = False
        for i in range(2):
            if cancelled():
                report.cancelled = True
                return report
            _, _, metrics = warm.complete(_prompt(nonce, i), max_tokens=OUTPUT_TOKENS, retries=3)
            warm_success |= metrics.ok
            notify("Warmup")
        if not warm_success:
            report.notes.append("Endpoint did not answer warmup requests; measurement stopped.")
            return report
        measured_config = replace(
            warm.config,
            stream=warm.streaming_supported,
            request_stream_usage=warm._stream_usage_supported,
            max_retries=1,
        )
        for i in range(1, config.max_concurrency):
            client = ChatClient(measured_config)
            clients.append(client)

        def warm_worker(pair):
            i, client = pair
            if cancelled():
                return
            _, _, metrics = client.complete(_prompt(nonce, i + 1), max_tokens=OUTPUT_TOKENS, retries=3)
            return metrics.ok

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as pool:
            for success in pool.map(warm_worker, list(enumerate(clients[1:], start=1))):
                if success is False:
                    report.notes.append("A worker connection failed warmup; its timed failures remain visible.")
                notify("Warmup")
        report.protocol["streaming"] = warm.streaming_supported
        report.protocol["client"] = client_protocol(measured_config)
        report.protocol["negotiated_workers"] = [
            {"streaming": c.streaming_supported, "stream_usage": c._stream_usage_supported}
            for c in clients
        ]
        start_index = config.max_concurrency + 1
        for level in config.levels:
            if cancelled():
                report.cancelled = True
                break
            # Build/tokenize prompts before timing the endpoint.
            prompts = [
                _prompt(nonce, start_index + i) for i in range(config.requests_for_level(level))
            ]
            start_index += len(prompts)
            if progress:
                progress(f"Concurrency {level}", count, total)
            point = _measure_level(clients, level, prompts, cancelled)
            count += point.requests
            if progress:
                progress(f"Concurrency {level}", count, total)
            report.concurrency.append(point)
            if cancelled():
                report.cancelled = True
                break
            if point.requests and point.errors == point.requests:
                report.notes.append(
                    f"All requests failed at concurrency {level}; further load stopped."
                )
                break
        if report.concurrency and report.concurrency[-1].requests > report.concurrency[-1].errors:
            from app.benchmarking.perf_sweep import PromptBank
            for context in CACHE_CONTEXTS:
                for level in sorted({1, config.max_concurrency}):
                    if cancelled():
                        report.cancelled = True
                        break
                    bank = PromptBank(context)
                    samples = max(4, level)
                    if progress:
                        progress(f"Cold prefix · {context:,} tokens · c={level}", count, total)
                    cold = _measure_level(clients, level, [bank.prompt(i) for i in range(samples)], cancelled)
                    count += cold.requests
                    pair = {"context_tokens": context, "concurrency": level, "cold": cold.to_dict()}
                    report.cache_reuse.append(pair)
                    if cold.requests == cold.errors or cancelled():
                        pair["status"] = "cold_failed" if not cancelled() else "cancelled"
                        if cancelled():
                            report.cancelled = True
                        break
                    prime = _probe(clients[0], bank.prompt(0, "warm"))
                    notify("Cache priming")
                    pair["priming_ok"] = prime.ok
                    if not prime.ok:
                        pair["status"] = "priming_failed"
                        pair["error"] = prime.error
                        break
                    warm_prompts = [bank.prompt(i + 1, "warm") for i in range(samples)]
                    if progress:
                        progress(f"Cached prefix · {context:,} tokens · c={level}", count, total)
                    reused = _measure_level(clients, level, warm_prompts, cancelled)
                    pair["warm"] = reused.to_dict()
                    pair["status"] = "measured"
                    count += reused.requests
                    if progress:
                        progress("Cache comparison", count, total)
        if cancelled():
            report.cancelled = True
        if any(p.estimated_token_requests for p in report.concurrency):
            report.notes.append("Some output token counts are estimates; see counts per level.")
        if any(p.burst_delivery_requests for p in report.concurrency):
            report.notes.append(
                "Some responses arrived in a single chunk or burst; TTFT measures first delivery, not model-side prefill."
            )
        return report
    finally:
        for client in clients:
            client.session.close()


