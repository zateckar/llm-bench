"""Performance measurement for an OpenAI-compatible endpoint.

Quality benchmarks say whether a model is *right*; this module says whether it is
*usable*. It measures four different things, because they fail independently:

* **Latency** - end-to-end wall clock per request, plus time-to-first-token
  (TTFT) when streaming is available. Reported as a distribution (p50/p90/p95/p99),
  not a mean, because tail latency is what users notice.
* **Decode throughput** - output tokens per second on a single stream, measured
  after the first token so queueing and prefill do not flatter the number.
* **Prefill throughput** - prompt tokens per second, estimated from TTFT on a
  deliberately long prompt with a 1-token generation cap.
* **Concurrency / capacity** - a sweep over concurrency levels reporting
  aggregate requests/s and tokens/s, latency degradation, and error rate. This is
  what tells you whether concurrency buys throughput or only queueing delay.

Every phase can be run independently; a phase that fails records a note rather
than aborting the sweep.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from llm_client import ChatClient, ClientConfig
from models import ConcurrencyPoint, LatencyStats, PerfReport, RequestMetrics

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY_LEVELS = (1, 2, 4, 8)

# Highest concurrency the sweep will accept. 256 in-flight streaming requests is
# already enough to saturate most hosted endpoints and costs ~256 sockets and
# threads on the client; beyond that the client becomes the bottleneck and the
# numbers stop describing the server.
MAX_CONCURRENCY = 256

# Each level fires at least this many requests per worker slot, so the measured
# window contains a steady state rather than only ramp-up and drain.
MIN_ROUNDS_PER_LEVEL = 2

# Above this level the client's own scheduling overhead starts to show up in the
# latency figures, so the report says so rather than letting it be read as server
# behaviour.
CLIENT_OVERHEAD_CONCURRENCY = 64

# A block of filler text used to build long prompts for the prefill probe.
_FILLER = (
    "The quick brown fox jumps over the lazy dog while seventeen amber lanterns "
    "sway above the harbour and the tide retreats across the flats. "
)


@dataclass
class PerfConfig:
    """Knobs for the performance suite.

    Defaults are sized to finish in roughly a minute against a typical hosted
    endpoint while still producing a stable distribution.
    """

    warmup_requests: int = 2
    serial_samples: int = 8
    serial_max_tokens: int = 24

    long_output_max_tokens: int = 384
    prefill_prompt_tokens: int = 3000

    concurrency_levels: tuple[int, ...] = DEFAULT_CONCURRENCY_LEVELS
    # Requests fired at each level. A level always sends at least `concurrency`
    # requests so every worker slot is used; see `requests_for_level`.
    requests_per_level: int = 8
    load_max_tokens: int = 96

    # Phase toggles, so a slow endpoint can be probed cheaply.
    measure_serial: bool = True
    measure_decode: bool = True
    measure_prefill: bool = True
    measure_concurrency: bool = True

    # Per-request retries during perf measurement. Retries distort timing, so a
    # perf request gets exactly one attempt and a failure is counted as an error.
    attempts_per_request: int = 1

    def requests_for_level(self, level: int) -> int:
        """How many requests a given concurrency level will fire.

        At least one per worker slot, so a level above ``requests_per_level`` is
        not measured with idle workers. Levels are also given a couple of rounds
        per slot where that is cheap, so the in-flight count reaches a steady
        state instead of only capturing ramp-up and drain.
        """
        return max(level * MIN_ROUNDS_PER_LEVEL, self.requests_per_level)

    def total_requests(self) -> int:
        """Upper bound on how many API calls the suite will make.

        This drives the progress display, so it has to match what the sweep
        actually fires - using ``len(levels) * requests_per_level`` here reported
        72 requests for a sweep that sent several hundred.
        """
        total = self.warmup_requests
        if self.measure_serial:
            total += self.serial_samples
        if self.measure_decode:
            total += 3
        if self.measure_prefill:
            total += 3
        if self.measure_concurrency:
            total += sum(
                self.requests_for_level(level)
                for level in sorted(set(self.concurrency_levels))
            )
        return total


@dataclass
class _Sample:
    metrics: RequestMetrics
    text: str = ""

    @property
    def ok(self) -> bool:
        return self.metrics.ok


def _new_client(base: ClientConfig) -> ChatClient:
    """Build a fresh client for a worker thread.

    ``requests.Session`` is not documented as thread-safe, and sharing one
    across the load generator would add lock contention to the very number we
    are trying to measure.
    """
    return ChatClient(base)


def _probe(client: ChatClient, prompt: str, max_tokens: int, attempts: int) -> _Sample:
    text, _usage, metrics = client.complete(
        prompt, max_tokens=max_tokens, retries=attempts
    )
    return _Sample(metrics=metrics, text=text)


def _ack_prompt(nonce: int) -> str:
    # A unique nonce defeats server-side prompt caching, which would otherwise
    # make the second request look artificially fast.
    return (
        f"Request {nonce}: reply with exactly the token ACK and nothing else. "
        "Do not explain."
    )


def _count_prompt(nonce: int, upto: int = 150) -> str:
    return (
        f"Task {nonce}: list the integers from 1 to {upto}, one per line, "
        "with no commentary."
    )


def _long_prompt(nonce: int, target_tokens: int) -> str:
    repeats = max(1, int(target_tokens * 4 / len(_FILLER)))
    return (
        f"Document {nonce}:\n" + (_FILLER * repeats) +
        "\n\nReply with exactly the token ACK and nothing else."
    )


def run_perf_suite(
    config: ClientConfig,
    perf: PerfConfig | None = None,
    progress=None,
) -> PerfReport:
    """Run every enabled phase and return a :class:`PerfReport`.

    ``progress`` is an optional ``callable(phase: str, done: int, total: int)``
    so a UI can show what is happening; the suite makes real network calls and
    can take a while.
    """
    perf = perf or PerfConfig()
    report = PerfReport(
        model=config.model,
        endpoint=config.base_url,
        streaming=config.stream,
    )

    client = ChatClient(config)
    attempts = max(1, perf.attempts_per_request)
    total = perf.total_requests()
    done = 0

    def tick(phase: str, n: int = 1) -> None:
        nonlocal done
        done += n
        if progress:
            try:
                progress(phase, done, total)
            except Exception:  # noqa: BLE001 - a broken UI must not fail the run
                logger.debug("perf progress callback raised", exc_info=True)

    # --- warm-up: the first request pays connection setup and model load ----
    for i in range(perf.warmup_requests):
        _probe(client, _ack_prompt(-1 - i), perf.serial_max_tokens, attempts)
        tick("warmup")

    if not client.streaming_supported:
        report.streaming = False
        report.notes.append(
            "Endpoint does not support streaming; TTFT and decode rate are unavailable "
            "and throughput is derived from end-to-end latency."
        )

    # --- phase 1: serial latency distribution -------------------------------
    if perf.measure_serial:
        samples = []
        for i in range(perf.serial_samples):
            samples.append(_probe(client, _ack_prompt(i), perf.serial_max_tokens, attempts))
            tick("latency")
        ok = [s for s in samples if s.ok]
        report.serial_latency = LatencyStats.from_samples([s.metrics.latency_ms for s in ok])
        report.serial_ttft = LatencyStats.from_samples(
            [s.metrics.ttft_ms for s in ok if s.metrics.ttft_ms is not None]
        )
        failures = len(samples) - len(ok)
        if failures:
            report.notes.append(f"{failures}/{len(samples)} latency probes failed.")

    # --- phase 2: single-stream decode throughput ---------------------------
    if perf.measure_decode:
        rates: list[float] = []
        produced = 0
        for i in range(3):
            sample = _probe(
                client, _count_prompt(1000 + i), perf.long_output_max_tokens, attempts
            )
            tick("decode")
            if not sample.ok:
                continue
            produced += sample.metrics.completion_tokens
            rate = sample.metrics.output_tokens_per_sec
            if rate:
                rates.append(rate)
        if rates:
            report.decode_tokens_per_sec = sum(rates) / len(rates)
            report.long_output_tokens_per_sec = max(rates)
        else:
            report.notes.append("Decode-throughput probe produced no usable samples.")
        if produced and produced < perf.long_output_max_tokens // 4:
            report.notes.append(
                f"Decode probe generated only {produced} tokens across 3 requests; "
                "the model may be ignoring the length instruction, so the decode "
                "rate is measured over a short window."
            )

    # --- phase 3: prefill throughput ----------------------------------------
    if perf.measure_prefill:
        rates = []
        for i in range(3):
            sample = _probe(
                client, _long_prompt(2000 + i, perf.prefill_prompt_tokens), 8, attempts
            )
            tick("prefill")
            if not sample.ok or sample.metrics.ttft_ms is None:
                continue
            prompt_tokens = sample.metrics.prompt_tokens
            if prompt_tokens and sample.metrics.ttft_ms > 0:
                rates.append(prompt_tokens / (sample.metrics.ttft_ms / 1000.0))
        if rates:
            report.prefill_tokens_per_sec = sum(rates) / len(rates)
        else:
            report.notes.append(
                "Prefill throughput unavailable (requires streaming and reported "
                "prompt_tokens)."
            )

    # --- phase 4: concurrency sweep -----------------------------------------
    if perf.measure_concurrency:
        levels, dropped = normalise_levels(perf.concurrency_levels)
        if dropped:
            report.notes.append(
                f"Concurrency level(s) {', '.join(str(d) for d in dropped)} were skipped: "
                f"levels must be between 1 and {MAX_CONCURRENCY}."
            )
        for level in levels:
            point = _run_concurrency_level(
                config, level, perf, attempts, lambda n=1: tick(f"load c={level}", n)
            )
            report.concurrency.append(point)
            if point.error_rate >= 0.5:
                report.notes.append(
                    f"Concurrency {level}: {point.errors}/{point.requests} requests "
                    "failed; the endpoint is rate limiting or saturated. Higher "
                    "levels were still measured but are not comparable."
                )
            elif point.errors:
                report.notes.append(
                    f"Concurrency {level}: {point.errors}/{point.requests} requests failed. "
                    "Throughput is computed from the successful requests only, so it "
                    "understates capacity at this level."
                )
        high = [p.concurrency for p in report.concurrency
                if p.concurrency > CLIENT_OVERHEAD_CONCURRENCY]
        if high:
            report.notes.append(
                f"Level(s) {', '.join(str(h) for h in high)} exceed "
                f"{CLIENT_OVERHEAD_CONCURRENCY} in-flight requests. At that point the load "
                "generator's own thread scheduling and socket handling contribute to the "
                "measured latency, so treat those rows as a lower bound on the endpoint's "
                "capacity rather than an exact figure. Run the sweep from a host close to "
                "the endpoint, or from several hosts, to push higher."
            )

    return report


def normalise_levels(levels) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Sort, de-duplicate and bound a set of concurrency levels.

    Returns ``(accepted, dropped)`` so the caller can report what it discarded
    instead of silently measuring something other than what was asked for.
    """
    accepted, dropped = [], []
    for raw in levels:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            dropped.append(raw)
            continue
        if 1 <= value <= MAX_CONCURRENCY:
            accepted.append(value)
        else:
            dropped.append(value)
    return tuple(sorted(set(accepted))), tuple(sorted(set(dropped)))


def _run_concurrency_level(
    config: ClientConfig,
    level: int,
    perf: PerfConfig,
    attempts: int,
    tick,
) -> ConcurrencyPoint:
    """Fire ``perf.requests_for_level(level)`` requests with ``level`` in flight."""
    count = perf.requests_for_level(level)
    results: list[_Sample] = []

    # One client per worker thread, reused across that thread's requests. Building
    # a fresh Session per request would mean a new TCP and TLS handshake every
    # time, which at high concurrency measures connection setup rather than the
    # endpoint - and would leave hundreds of sockets in TIME_WAIT.
    local = threading.local()

    def job(idx: int) -> _Sample:
        worker = getattr(local, "client", None)
        if worker is None:
            worker = _new_client(config)
            local.client = worker
        return _probe(
            worker, _count_prompt(5000 + idx, upto=80), perf.load_max_tokens, attempts,
        )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=level) as pool:
        futures = [pool.submit(job, i) for i in range(count)]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:  # noqa: BLE001
                results.append(
                    _Sample(metrics=RequestMetrics(ok=False, error=str(e)))
                )
            tick()
    wall_ms = (time.perf_counter() - started) * 1000.0

    ok = [s for s in results if s.ok]
    return ConcurrencyPoint(
        concurrency=level,
        requests=len(results),
        errors=len(results) - len(ok),
        wall_ms=wall_ms,
        latency=LatencyStats.from_samples([s.metrics.latency_ms for s in ok]),
        ttft=LatencyStats.from_samples(
            [s.metrics.ttft_ms for s in ok if s.metrics.ttft_ms is not None]
        ),
        output_tokens=sum(s.metrics.completion_tokens for s in ok),
        prompt_tokens=sum(s.metrics.prompt_tokens for s in ok),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.0f} ms"


def _rate(value: float | None, unit: str = "tok/s") -> str:
    return "n/a" if value is None else f"{value:.1f} {unit}"


def format_perf_markdown(report: PerfReport) -> str:
    """Render a PerfReport as a markdown section."""
    lines = [
        "## Performance",
        "",
        f"- **Streaming**: {'yes' if report.streaming else 'no (TTFT unavailable)'}",
        f"- **Single-stream decode**: {_rate(report.decode_tokens_per_sec)}",
        f"- **Prefill**: {_rate(report.prefill_tokens_per_sec)}",
        f"- **Peak aggregate output**: {_rate(report.peak_output_tokens_per_sec)}",
        f"- **Peak request rate**: {_rate(report.peak_requests_per_sec, 'req/s')}",
    ]
    if report.saturation_concurrency:
        lines.append(
            f"- **Saturation point**: concurrency {report.saturation_concurrency} "
            "(more concurrency adds latency, not throughput)"
        )
    if report.scaling_efficiency is not None:
        lines.append(
            f"- **Scaling efficiency at max concurrency**: "
            f"{report.scaling_efficiency * 100:.0f}% of linear"
        )
    lines.append("")

    lines += [
        "### Latency (single stream)",
        "",
        "| Metric | Samples | Mean | p50 | p90 | p95 | p99 | Max |",
        "|--------|---------|------|-----|-----|-----|-----|-----|",
    ]
    for label, stats in (
        ("End-to-end", report.serial_latency),
        ("Time to first token", report.serial_ttft),
    ):
        lines.append(
            f"| {label} | {stats.count} | {_ms(stats.mean)} | {_ms(stats.p50)} | "
            f"{_ms(stats.p90)} | {_ms(stats.p95)} | {_ms(stats.p99)} | {_ms(stats.max)} |"
        )
    lines.append("")

    if report.concurrency:
        lines += [
            "### Concurrency sweep",
            "",
            "| Concurrency | Requests | Errors | Req/s | Output tok/s | p50 latency | p95 latency | p50 TTFT |",
            "|-------------|----------|--------|-------|--------------|-------------|-------------|----------|",
        ]
        for point in sorted(report.concurrency, key=lambda p: p.concurrency):
            marker = "*" if point.concurrency > CLIENT_OVERHEAD_CONCURRENCY else ""
            lines.append(
                f"| {point.concurrency}{marker} | {point.requests} | {point.errors} "
                f"({point.error_rate * 100:.0f}%) | {point.requests_per_sec:.2f} | "
                f"{point.output_tokens_per_sec:.1f} | {_ms(point.latency.p50)} | "
                f"{_ms(point.latency.p95)} | {_ms(point.ttft.p50)} |"
            )
        if any(p.concurrency > CLIENT_OVERHEAD_CONCURRENCY for p in report.concurrency):
            lines.append("")
            lines.append(
                f"`*` above {CLIENT_OVERHEAD_CONCURRENCY} in-flight requests the load generator "
                "contributes to the measured latency; read those rows as a lower bound on capacity."
            )
        lines.append("")

    if report.notes:
        lines.append("**Measurement notes**:")
        lines += [f"- {note}" for note in report.notes]
        lines.append("")

    return "\n".join(lines)


def format_perf_console(report: PerfReport) -> str:
    """Compact console summary."""
    lines = [
        "Performance:",
        f"  latency  p50 {_ms(report.serial_latency.p50)} | "
        f"p95 {_ms(report.serial_latency.p95)} | p99 {_ms(report.serial_latency.p99)}",
        f"  ttft     p50 {_ms(report.serial_ttft.p50)} | p95 {_ms(report.serial_ttft.p95)}",
        f"  decode   {_rate(report.decode_tokens_per_sec)} single-stream | "
        f"prefill {_rate(report.prefill_tokens_per_sec)}",
        f"  capacity peak {_rate(report.peak_output_tokens_per_sec)} | "
        f"{_rate(report.peak_requests_per_sec, 'req/s')}"
        + (
            f" | saturates at c={report.saturation_concurrency}"
            if report.saturation_concurrency else ""
        ),
    ]
    for point in sorted(report.concurrency, key=lambda p: p.concurrency):
        lines.append(
            f"    c={point.concurrency:<3} {point.requests_per_sec:6.2f} req/s  "
            f"{point.output_tokens_per_sec:7.1f} tok/s  "
            f"p50 {_ms(point.latency.p50):>9}  p95 {_ms(point.latency.p95):>9}  "
            f"errors {point.errors}/{point.requests}"
        )
    for note in report.notes:
        lines.append(f"  ! {note}")
    return "\n".join(lines)
