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
from models import ConcurrencyPoint, ContextPoint, LatencyStats, PerfReport, RequestMetrics

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
# Context-length scalability sweep
# ---------------------------------------------------------------------------

# Default context sizes (tokens) to probe. These are the operating points
# commonly advertised for modern models; probing across two orders of
# magnitude shows where prefill cost and rate limits start to dominate.
DEFAULT_CONTEXT_SIZES: tuple[int, ...] = (
    32_000, 64_000, 128_000, 192_000, 256_000, 384_000, 512_000, 720_000, 1_000_000,
)

# One 1M-token prompt at 4 chars/token is ~4 MB of text; a slow model can spend
# minutes just prefilling it, so the per-request timeout must scale with the
# context size instead of using the fixed 180 s default.
CONTEXT_TIMEOUT_BASE_S = 180.0
CONTEXT_TIMEOUT_PER_TOKEN_S = 1.0 / 8_000.0  # ~120 s extra per 1M tokens
CONTEXT_TIMEOUT_MAX_S = 600.0


@dataclass
class ContextSweepConfig:
    """Knobs for the context-length sweep.

    Small sizes (below ``large_threshold_tokens``) get the full concurrency
    treatment so the sweep measures real scaling; large sizes are deliberately
    sampled serially because the alternative - a 4-way burst of 1M-token
    prompts - spends tens of millions of tokens per model and tells you more
    about the rate limiter than about the serving stack.

    ``concurrency_levels`` turns the sweep into a context x concurrency grid:
    every size is probed at each requested level, one ContextPoint per
    (size, level) cell. A concurrency level on a large size multiplies the
    prompt spend of that size - c=4 at 1M tokens is a ~4M-token burst per
    probe round - so ``max_large_concurrency`` caps what large sizes accept.
    Levels above the cap are clamped down per large size, not dropped, so the
    grid stays rectangular and the clamp is recorded on the emitted point.
    """

    context_sizes: tuple[int, ...] = DEFAULT_CONTEXT_SIZES
    # Concurrent probes per context size below the large threshold. Bounded by
    # MAX_CONCURRENCY like the load sweep; the client is the bottleneck beyond
    # ~64 in-flight streaming requests.
    concurrency_per_size: int = 4
    # Explicit grid levels. Empty (the default) keeps the single-level
    # behaviour driven by ``concurrency_per_size``; when given, it overrides
    # that for every size.
    concurrency_levels: tuple[int, ...] = ()
    # Hard cap on concurrency for sizes at or above ``large_threshold_tokens``
    # when a grid is requested. A multi-way burst of sub-megabyte prompts is
    # cheap enough to measure scaling, but the same burst at 1M tokens spends
    # tens of millions of tokens per model; this is the compromise between
    # "grid everywhere" and "keep the token bill sane".
    max_large_concurrency: int = 4
    # Probes per small context size; at least ``concurrency_per_size`` are run
    # so every worker slot is exercised.
    probes_per_size_small: int = 3
    # Probes per large context size - serial, by design.
    probes_per_size_large: int = 1
    # Above this context size only ``probes_per_size_large`` serial probes run.
    large_threshold_tokens: int = 192_000
    # Generation is capped tiny on purpose: the sweep measures how context size
    # changes spot-check cost, not how the model answers. 8 tokens gets TTFT,
    # prefill cost and usage back without paying decode at 1M scale.
    probe_max_tokens: int = 8

    def levels_for_size(self, size: int) -> tuple[int, ...]:
        """Concurrency levels actually measured at ``size``.

        With no explicit grid the size gets exactly one level - the
        ``concurrency_per_size`` burst for small sizes, serial probing for
        large ones, as before. With a grid every requested level is clamped
        into 1..MAX_CONCURRENCY, and large sizes are additionally clamped to
        ``max_large_concurrency``; clamping merges levels, so duplicates are
        dropped after clamping.
        """
        if not self.concurrency_levels:
            return (self.concurrency_for_size(size),)
        cap = MAX_CONCURRENCY
        if size >= self.large_threshold_tokens:
            cap = min(cap, max(1, self.max_large_concurrency))
        return tuple(sorted({max(1, min(cap, level)) for level in self.concurrency_levels}))

    def probes_for_size(self, size: int, level: int | None = None) -> int:
        """Probes fired for one grid cell ``(size, level)``.

        Each cell fires at least one probe per worker slot so no slot sits
        idle, and at least ``probes_per_size_small``/``probes_per_size_large``
        so the probe count does not shrink below what a single-level sweep
        would have sent. ``level=None`` keeps the pre-grid signature: it means
        the size's single default level.
        """
        if level is None:
            level = self.levels_for_size(size)[0]
        if size >= self.large_threshold_tokens:
            return max(level, self.probes_per_size_large)
        return max(level, self.probes_per_size_small)

    def concurrency_for_size(self, size: int) -> int:
        if size >= self.large_threshold_tokens:
            return 1
        return max(1, min(MAX_CONCURRENCY, self.concurrency_per_size))

    def timeout_for_size(self, size: int) -> float:
        """Per-request timeout that grows with the context being prefilled."""
        scaled = CONTEXT_TIMEOUT_BASE_S + size * CONTEXT_TIMEOUT_PER_TOKEN_S
        return min(CONTEXT_TIMEOUT_MAX_S, scaled)

    def total_requests(self) -> int:
        """Upper bound on API calls, for progress reporting and cost planning.

        Sums over the whole (size, level) grid, so the progress display cannot
        under-report what a multi-level sweep is about to spend.
        """
        return sum(
            self.probes_for_size(size, level)
            for size in sorted(set(self.context_sizes))
            for level in self.levels_for_size(size)
        )


def _context_prompt(nonce: int, target_tokens: int) -> str:
    """Build a filler prompt calibrated to roughly ``target_tokens``.

    Uses the same chars-per-token heuristic as ``_long_prompt`` so the expected
    context is the input to the server; the server-reported usage is what gets
    recorded, so a tokenizer mismatch shows up in the measured prompt_tokens
    rather than silently skewing the curve.
    """
    repeats = max(1, int(target_tokens * 4 / len(_FILLER)))
    return (
        f"Calibration document {nonce}:\n" + (_FILLER * repeats) +
        "\n\nIgnoring the document above, reply with exactly one token: OK."
    )


def _is_context_limit_error(sample: _Sample) -> bool:
    """Heuristic: the server rejected the request because the prompt exceeded
    its context window, which is different from a generic transport failure."""
    if sample.ok:
        return False
    err = (sample.metrics.error or "").lower()
    return any(
        marker in err
        for marker in (
            "context length", "context_length", "maximum context", "max_tokens",
            "context window", "token limit", "too many tokens",
        )
    )


def run_context_sweep(
    config: ClientConfig,
    sweep: ContextSweepConfig | None = None,
    progress=None,
) -> list[ContextPoint]:
    """Probe the endpoint at a fixed set of context sizes.

    Each (size, level) grid cell gets one :class:`ContextPoint` (one per size
    when no grid is requested, as before). Small sizes are exercised with the
    requested concurrency; sizes at or above
    ``ContextSweepConfig.large_threshold_tokens`` are clamped to
    ``max_large_concurrency`` so the sweep finishes in reasonable time and
    token budget. A size whose level-1 probes all report a context-window
    error is recorded as ``skipped`` at every level rather than dragged
    through the rest of the grid: the remaining levels of that size are not
    probed at all, since they are guaranteed to fail the same way.
    """
    sweep = sweep or ContextSweepConfig()
    attempts = 1  # perf measurements never retry; a retry would distort timing
    total = sweep.total_requests()
    done = 0

    def tick(size: int, level: int, n: int = 1) -> None:
        nonlocal done
        done += n
        if progress:
            try:
                progress(f"context {size} c{level}", done, total)
            except Exception:  # noqa: BLE001 - a broken UI must not fail the run
                logger.debug("context sweep progress callback raised", exc_info=True)

    points: list[ContextPoint] = []
    for size in sorted(set(sweep.context_sizes)):
        if size <= 0:
            continue
        levels = sweep.levels_for_size(size)
        for index, level in enumerate(levels):
            point = _run_context_job(config, sweep, size, level, attempts,
                                     lambda n=1, s=size, c=level: tick(s, c, n))
            points.append(point)
            if point.skipped and index == 0 and len(levels) > 1:
                # The level-1 probes were all context-limit refusals, so the
                # prompt itself is too long - firing the same request louder
                # does not change the answer, it just multiplies the bill.
                # Complete the grid with skipped copies instead of probing.
                skipped_note = (
                    f"skipped without probing: level-1 probes at this size hit "
                    f"the context limit ({point.skip_reason})"
                )
                for skipped_level in levels[1:]:
                    points.append(ContextPoint(
                        context_tokens=size,
                        concurrency=skipped_level,
                        requests=0,
                        errors=0,
                        wall_ms=0.0,
                        latency=LatencyStats(),
                        ttft=LatencyStats(),
                        prompt_tokens=0,
                        output_tokens=0,
                        skipped=True,
                        skip_reason=point.skip_reason,
                        notes=[skipped_note],
                    ))
                break

    return points


def _run_context_job(
    config: ClientConfig,
    sweep: ContextSweepConfig,
    size: int,
    level: int,
    attempts: int,
    tick,
) -> ContextPoint:
    """Run the probes for one grid cell and aggregate them into a point."""
    probes = sweep.probes_for_size(size, level)
    concurrency = sweep.concurrency_for_size(size) if not sweep.concurrency_levels else level
    timeout = sweep.timeout_for_size(size)
    nonce_base = 10_000 + (size + level) % 10_000

    def job(idx: int) -> _Sample:
        # Each probe gets its own client config so the per-size timeout
        # cannot leak into the next size's measurements.
        worker_config = ClientConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            seed=config.seed,
            timeout=timeout,
            stream=config.stream,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            request_stream_usage=config.request_stream_usage,
            extra_headers=config.extra_headers,
        )
        client = ChatClient(worker_config)
        try:
            return _probe(
                client,
                _context_prompt(nonce_base + idx, size),
                sweep.probe_max_tokens,
                attempts,
            )
        finally:
            client.session.close()

    started = time.perf_counter()
    results: list[_Sample] = []
    if concurrency > 1 and probes > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(job, i) for i in range(probes)]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:  # noqa: BLE001
                    results.append(
                        _Sample(metrics=RequestMetrics(ok=False, error=str(e)))
                    )
                tick()
    else:
        for i in range(probes):
            results.append(job(i))
            tick()
    wall_ms = (time.perf_counter() - started) * 1000.0

    skipped = results and all(_is_context_limit_error(r) for r in results)
    skip_reason = ""
    if skipped:
        skip_reason = results[0].metrics.error or "context length exceeded"

    ok = [s for s in results if s.ok]
    point = ContextPoint(
        context_tokens=size,
        concurrency=concurrency,
        requests=len(results),
        errors=len(results) - len(ok),
        wall_ms=wall_ms,
        latency=LatencyStats.from_samples([s.metrics.latency_ms for s in ok]),
        ttft=LatencyStats.from_samples(
            [s.metrics.ttft_ms for s in ok if s.metrics.ttft_ms is not None]
        ),
        prompt_tokens=sum(s.metrics.prompt_tokens for s in ok),
        output_tokens=sum(s.metrics.completion_tokens for s in ok),
        skipped=skipped,
        skip_reason=skip_reason,
    )
    if sweep.concurrency_levels:
        requested = max(
            {max(1, min(MAX_CONCURRENCY, lv)) for lv in sweep.concurrency_levels}
        )
        if concurrency < requested:
            cap = min(MAX_CONCURRENCY, max(1, sweep.max_large_concurrency))
            point.notes.append(
                f"requested concurrency {requested} clamped to {concurrency} "
                f"(cap {cap} for sizes >= {sweep.large_threshold_tokens:,} tokens)"
            )
    if not ok and not skipped:
        point.notes.append(
            f"All {len(results)} probes failed (not a context-limit refusal)."
        )
    return point


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


def format_context_sweep_markdown(points: list[ContextPoint]) -> str:
    """Render a context-length sweep as a markdown section."""
    lines = [
        "## Context scalability",
        "",
        "Time-to-first-token and single-request latency at increasing prompt sizes. "
        "Prefill throughput is derived from TTFT, so sizes with an unavailable TTFT "
        "show `n/a` there. 'skipped' means the endpoint rejected the size outright.",
        "",
        "| Context (tokens) | Concurrency | Probes | Errors | p50 latency | p95 latency | "
        "TTFT mean | Prefill tok/s | Output tok/s | Note |",
        "|------------------|-------------|--------|--------|-------------|-------------|"
        "-----------|-----------------|----------------|------|",
    ]
    for point in sorted(points, key=lambda p: (p.context_tokens, p.concurrency)):
        if point.skipped:
            note = point.skip_reason or "context limit"
            lines.append(
                f"| {point.context_tokens:,} | - | {point.requests} | - | - | - | - | - | - | skipped: {note} |"
            )
            continue
        lines.append(
            f"| {point.context_tokens:,} | {point.concurrency} | {point.requests} "
            f"| {point.errors} | {_ms(point.latency.p50)} | {_ms(point.latency.p95)} "
            f"| {_ms(point.ttft.mean)} | {_rate(point.prompt_tokens_per_sec)} "
            f"| {_rate(point.output_tokens_per_sec)} | {'; '.join(point.notes) or ''} |"
        )
    lines.append("")

    # One scalability line per concurrency level: how much prefill throughput
    # is lost between the smallest and largest size that level actually
    # measured. Only worth stating when both ends of the range exist.
    for level in sorted({p.concurrency for p in points if not p.skipped}):
        measured = sorted(
            (p for p in points
             if p.concurrency == level and not p.skipped and p.prompt_tokens_per_sec),
            key=lambda p: p.context_tokens,
        )
        if len(measured) < 2:
            continue
        first, last = measured[0], measured[-1]
        ratio = first.prompt_tokens_per_sec / last.prompt_tokens_per_sec
        lines.append(
            f"- c={level}: prefill fell {ratio:.1f}x from "
            f"{first.context_tokens // 1000}k to {last.context_tokens // 1000}k."
        )
    if lines and lines[-1] != "":
        lines.append("")
    return "\n".join(lines)


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
