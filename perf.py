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
* **Concurrency / capacity** - an open-loop sweep over concurrency levels:
  requests are scheduled to arrive on a wall-clock cadence instead of one
  worker waiting for its previous request, so a server that falls behind is
  actually measured falling behind (a closed-loop worker pool self-throttles
  and never surfaces queueing). Each level reports aggregate requests/s and
  tokens/s, per-stream decode rate, SLO verdicts, and error rate - this is
  what tells you how many users the stack can serve before it hurts.

Every phase can be run independently; a phase that fails records a note rather
than aborting the sweep.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from llm_client import ChatClient, ClientConfig
from models import (
    CacheProbe,
    ConcurrencyPoint,
    ContextPoint,
    LatencyStats,
    PerfReport,
    RequestMetrics,
)

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

# The prefix-cache probe uses a *different* filler from ``_FILLER`` so chunks
# the prefill/context phases already pushed through a prefix cache do not warm
# this probe's cold request - that would silently understate the cold/warm gap.
_CACHE_FILLER = (
    "Under a copper sky the observatory catalogues slow drift of granite buoys "
    "and tallymen count the gulls that never land. "
)


def _cache_probe_prompt(run_nonce: int, target_tokens: int) -> str:
    """One large prompt for the cold/warm prefix-cache pair.

    The per-run nonce at the head keeps this run's prefix distinct from other
    runs (and from phases that share ``_FILLER``); the warm probes reuse the
    identical string so the cache can hit.
    """
    repeats = max(1, int(target_tokens * 4 / len(_CACHE_FILLER)))
    return (
        f"Cache probe {run_nonce}:\n" + (_CACHE_FILLER * repeats) +
        "\n\nAcknowledge with exactly: OK."
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
    # requests so every scheduled arrival exists; see `requests_for_level`.
    requests_per_level: int = 8
    load_max_tokens: int = 96
    # Load shape. None keeps the classic uniform count-to-80 probe; a mix of
    # chat/summarize/codegen models what an organisation actually sends.
    workload_mix: tuple[WorkloadTask, ...] | None = None
    # Shared prompt prefix reused across every load probe of a level. Serving
    # stacks with prefix caching serve this much better than nonce-everywhere
    # probes - which is what production traffic actually looks like.
    shared_prefix: bool = False

    # Acceptance thresholds the capacity verdict is computed against. They
    # model "would a colleague find this usable", not "is the server up":
    # users notice first-token delay above ~2 s and stop trusting answers that
    # stream slower than reading speed.
    slo_ttft_p95_ms: float = 2000.0
    slo_stream_tps_p50: float = 15.0
    slo_error_rate: float = 0.01
    # How many requests one active user generates in an hour; converts the
    # measured request rate into "people we can serve". A 0 stops that
    # estimate from being reported.
    requests_per_user_hour: float = 60.0

    # Phase toggles, so a slow endpoint can be probed cheaply.
    measure_serial: bool = True
    measure_decode: bool = True
    measure_prefill: bool = True
    measure_concurrency: bool = True
    # Cold/warm probe against a large shared prefix: 1 cold + this many
    # identical warm requests, reporting cache hit ratio and speedups.
    measure_prefix_cache: bool = True
    prefix_cache_warm_probes: int = 2

    # Per-request retries during perf measurement. Retries distort timing, so a
    # perf request gets exactly one attempt and a failure is counted as an error.
    attempts_per_request: int = 1

    def requests_for_level(self, level: int) -> int:
        """How many requests a given concurrency level will fire.

        At least one per scheduled arrival, so a level above
        ``requests_per_level`` is not measured with idle capacity. Levels are
        also given a couple of rounds where that is cheap, so the in-flight
        count reaches a steady state instead of only capturing ramp-up and
        drain.
        """
        return max(level * MIN_ROUNDS_PER_LEVEL, self.requests_per_level)

    def total_requests(self) -> int:
        """Upper bound on how many API calls the suite will make.

        This drives the progress display. When concurrency is on, the open-loop
        levels may be re-measured at up to three times the configured budget
        (short probes scale sub-linearly with c, so a long-requested level
        would otherwise only cover ramp-up), and the knee refinement may probe
        one additional pair of midpoint levels once.
        """
        total = self.warmup_requests
        if self.measure_serial:
            total += self.serial_samples
        if self.measure_decode:
            total += 3
        if self.measure_prefill:
            total += 3
        if self.measure_concurrency:
            base = sum(
                self.requests_for_level(level)
                for level in sorted(set(self.concurrency_levels))
            )
            total += base * 3 + 2 * max(self.requests_per_level, 1)
        if self.measure_prefix_cache:
            total += 1 + max(0, self.prefix_cache_warm_probes)
        return total


@dataclass
class _Sample:
    metrics: RequestMetrics
    text: str = ""
    # Which workload task produced the probe, for per-task SLO insight.
    task: str = ""

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


# --- Workload mix -----------------------------------------------------------
# Real traffic is not one synthetic prompt: it is mostly short Q&A, with a
# tail of long summarisation and a codegen/drafting block. Tasks encode the
# three shapes and each probe picks one by weight, so the load phase measures
# the mix an organisation actually sends - and prefix-sharing stacks get to
# show the caching they will do in production.


@dataclass(frozen=True)
class WorkloadTask:
    """One shape of request in the load mix. weight is relative."""

    name: str
    weight: int
    prompt_tokens: int
    max_tokens: int


WORKLOAD_MIX: tuple[WorkloadTask, ...] = (
    WorkloadTask("chat", weight=60, prompt_tokens=48, max_tokens=32),
    WorkloadTask("summarize", weight=25, prompt_tokens=1200, max_tokens=96),
    WorkloadTask("codegen", weight=15, prompt_tokens=200, max_tokens=160),
)


DEFAULT_WORKLOAD_MIX: tuple[WorkloadTask, ...] = WORKLOAD_MIX


def _expand_mix(mix: tuple[WorkloadTask, ...]) -> list[WorkloadTask]:
    """Weights -> a flat, interleaved bag; the sampler draws round-robin.

    Tasks are interleaved rather than concatenated so a level with few probes
    still sees every task shape - concatenating weights would feed a 10-probe
    level nothing but the heaviest task.
    """
    bag: list[WorkloadTask] = []
    rounds = max((task.weight for task in mix), default=1)
    for _ in range(rounds):
        bag.extend(mix)
    return bag or [WorkloadTask("chat", 1, 48, 32)]


def _task_prompt(task: WorkloadTask, nonce: int, shared_prefix: str) -> str:
    """A task-shaped prompt; the shared prefix is prepend-only so prefix
    caches see the same leading tokens on every probe of this level."""
    tail_nonce = f" r{nonce}"
    marker = f"[{task.name}] "
    if task.name == "summarize":
        filler_repeats = max(1, int(task.prompt_tokens * 4 / len(_FILLER)))
        return (
            f"{shared_prefix}{marker}Document{tail_nonce}:\n" + (_FILLER * filler_repeats) +
            f"\n\nSummarise the document in at most {task.max_tokens} tokens."
        )
    if task.name == "codegen":
        return (
            f"{shared_prefix}{marker}Code task{tail_nonce}: write a Python function "
            "f(n) returning the sum of squares 1..n, with a docstring and "
            "a type hint. Code only."
        )
    return (
        f"{shared_prefix}{marker}Q{tail_nonce}: give a two-sentence practical tip about "
        "staying focused during long meetings."
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
        slo_ttft_p95_ms=perf.slo_ttft_p95_ms,
        slo_stream_tps_p50=perf.slo_stream_tps_p50,
        slo_error_rate=perf.slo_error_rate,
        requests_per_user_hour=perf.requests_per_user_hour,
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

    # --- phase 4: open-loop concurrency sweep with knee refinement -----------
    if perf.measure_concurrency:
        levels, dropped = normalise_levels(perf.concurrency_levels)
        if dropped:
            report.notes.append(
                f"Concurrency level(s) {', '.join(str(d) for d in dropped)} were skipped: "
                f"levels must be between 1 and {MAX_CONCURRENCY}."
            )
        if levels:
            _run_capacity_sweep(config, perf, report, levels, attempts, tick)

    # --- phase 5: prefix-cache cold/warm probe --------------------------------
    if perf.measure_prefix_cache:
        _run_cache_probe(config, perf, report, attempts, tick)

    return report


def _run_cache_probe(
    config: ClientConfig,
    perf: PerfConfig,
    report: PerfReport,
    attempts: int,
    tick,
) -> None:
    """One cold request against a large shared prefix, then warm repeats.

    Measures whether the endpoint's prefix cache helps a repeated long prompt:
    the server-reported hit count plus the warm-vs-cold TTFT/prefill delta.
    Runs after the load sweep so any cache state the sweep left behind is the
    *point* (a hot cache is what production looks like), and uses a filler the
    prefill/context phases never emit so the cold request is genuinely cold.
    A failure records a note and leaves ``report.cache_probe`` as None rather
    than aborting the suite.
    """
    run_nonce = int(time.time() * 1000) % 1_000_000
    prompt = _cache_probe_prompt(run_nonce, perf.prefill_prompt_tokens)
    client = ChatClient(config)
    try:
        cold = _probe(client, prompt, 8, attempts)
        tick("prefix cache")
        if not cold.ok:
            report.notes.append(
                "Prefix-cache probe failed on the cold request; cache benefit not measured."
            )
            return

        warm: list[_Sample] = []
        for _ in range(max(0, perf.prefix_cache_warm_probes)):
            warm.append(_probe(client, prompt, 8, attempts))
            tick("prefix cache")
        warm_ok = [s for s in warm if s.ok]
        if not warm_ok:
            report.notes.append("Prefix-cache probe produced no successful warm request.")
            return

        def _prefill(s: _Sample) -> float | None:
            pt, ttft = s.metrics.prompt_tokens, s.metrics.ttft_ms
            # NaN is truthy, so a NaN ttft would slip past a plain truthiness
            # check and poison the whole prefill average; guard explicitly.
            if pt is not None and pt >= 0 and ttft is not None and math.isfinite(ttft) and ttft > 0:
                return pt / (ttft / 1000.0)
            return None

        warm_ttfts = [s.metrics.ttft_ms for s in warm_ok if s.metrics.ttft_ms is not None]
        # ``is not None``, not truthiness: a legitimately computed 0.0 rate is a
        # valid sample and must not be dropped from the average.
        warm_prefill = [r for r in (_prefill(s) for s in warm_ok) if r is not None]
        report.cache_probe = CacheProbe(
            prompt_tokens=cold.metrics.prompt_tokens,
            warm_probes=len(warm_ok),
            warm_prompt_tokens=sum(s.metrics.prompt_tokens for s in warm_ok),
            cached_tokens=sum(s.metrics.cached_tokens for s in warm_ok),
            cold_ttft_ms=cold.metrics.ttft_ms,
            warm_ttft_ms=(
                sum(warm_ttfts) / len(warm_ttfts) if warm_ttfts else None
            ),
            cold_prefill_tokens_per_sec=_prefill(cold),
            warm_prefill_tokens_per_sec=(
                sum(warm_prefill) / len(warm_prefill) if warm_prefill else None
            ),
        )
        if report.cache_probe.cold_ttft_ms is None or not warm_ttfts:
            report.cache_probe.notes.append(
                "TTFT unavailable (endpoint not streaming); only the hit ratio is reliable."
            )
        if report.cache_probe.cached_tokens == 0 and report.cache_probe.warm_prompt_tokens:
            report.cache_probe.notes.append(
                "Server reported no prefix-cache hits; either it does not cache, this "
                "prompt is not cacheable at this size, or it does not report hits."
            )
    finally:
        client.session.close()


def _run_capacity_sweep(
    config: ClientConfig,
    perf: PerfConfig,
    report: PerfReport,
    levels: tuple[int, ...],
    attempts: int,
    tick,
) -> None:
    """Measure each requested level open-loop, then bisect the SLO knee.

    Two passes over the grid:

    1. the configured levels, which answer "how far did I ask to push";
    2. one refinement round that inserts the midpoint of every adjacent
       (highest passing, lowest failing) level pair, so the capacity verdict
       is not just the coarsest grid point below the knee.

    A level whose SLO failure is pure error-rate (the endpoint is rate
    limiting or refusing) is a different signal from a level whose latency
    and per-stream rate degrade: the first marks the configured cap, the
    second the serving stack's actual limit, and the report notes say which.
    """
    measured: dict[int, ConcurrencyPoint] = {}

    def measure(level: int) -> ConcurrencyPoint:
        point = measured.get(level)
        if point is None:
            point = _open_loop_level(
                config, level, perf.requests_for_level(level), perf, report, attempts,
                lambda n=1: tick(f"load c={level}", n),
            )
            measured[level] = point
        return point

    for level in levels:
        measure(level)

    for _ in range(REFINEMENT_ROUNDS):
        ordered = sorted(measured)
        verdict = {c: report.point_meets_slo(measured[c]) for c in ordered}
        # Adjacent pass->fail pairs bound the knee; bisect each of them once.
        gaps = [
            (low, high)
            for low, high in zip(ordered, ordered[1:])
            if verdict[low] and not verdict[high] and high - low > 1
        ]
        if not gaps:
            break
        for low, high in gaps:
            mid = (low + high) // 2
            measure(mid)

    report.concurrency = [measured[c] for c in sorted(measured)]

    for point in report.concurrency:
        verdict = report.point_meets_slo(point)
        if point.error_rate >= 0.5 and verdict:
            report.notes.append(
                f"Concurrency {point.concurrency}: {point.errors}/{point.requests} requests "
                "failed; the endpoint is rate limiting or saturated. Higher "
                "levels were still measured but are not comparable."
            )
        elif point.errors and verdict:
            report.notes.append(
                f"Concurrency {point.concurrency}: {point.errors}/{point.requests} requests failed. "
                "Throughput is computed from the successful requests only, so it "
                "understates capacity at this level."
            )

    capacity = report.slo_capacity
    measured_levels = sorted(measured)
    if capacity is not None:
        knee = next(
            (c for c in measured_levels if c > capacity and not report.point_meets_slo(measured[c])),
            None,
        )
        cause = _knee_cause(report, measured[knee]) if knee is not None else ""
        report.notes.append(
            f"Meets SLO up to c={capacity}"
            + (f"; fails at c={knee} ({cause})." if knee is not None else " (highest measured level).")
        )
    elif measured_levels:
        first = measured[measured_levels[0]]
        cause = _knee_cause(report, first)
        report.notes.append(
            f"Fails SLO already at c={measured_levels[0]} ({cause}); try lower levels "
            "or relax the SLO thresholds."
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


def _knee_cause(report: PerfReport, point: ConcurrencyPoint) -> str:
    """Which SLO criterion(s) a failing level tripped, in plain language."""
    causes = []
    if report.slo_ttft_p95_ms > 0:
        p95 = point.ttft.p95
        if p95 is None or p95 > report.slo_ttft_p95_ms:
            causes.append(
                f"p95 TTFT {'unavailable' if p95 is None else f'{p95:.0f} ms'} "
                f"> {report.slo_ttft_p95_ms:.0f} ms"
            )
    if report.slo_stream_tps_p50 > 0:
        tps = point.stream_tps.p50
        if tps is None or tps < report.slo_stream_tps_p50:
            causes.append(
                f"per-stream decode {'unavailable' if tps is None else f'{tps:.1f} tok/s'} "
                f"< {report.slo_stream_tps_p50:.1f} tok/s"
            )
    if report.slo_error_rate > 0 and point.error_rate > report.slo_error_rate:
        causes.append(f"errors {point.error_rate * 100:.0f}% > {report.slo_error_rate * 100:.0f}%")
    return ", ".join(causes) or "unknown"


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


# Beyond the nominal grid the knee between adjacent levels is refined with one
# extra level per gap; the extra probe runs against a cached session so its
# cost is bounded by the en-route request budget.
REFINEMENT_ROUNDS = 1


def _open_loop_level(
    config: ClientConfig,
    level: int,
    count: int,
    perf: PerfConfig,
    report: PerfReport,
    attempts: int,
    tick,
) -> ConcurrencyPoint:
    """Schedule ``count`` requests on a wall-clock cadence with ``level`` in the
    air at a time *if the server keeps up*.

    A closed loop (one worker per slot, next request when the previous
    finishes) cannot find a capacity limit: when the server slows down the
    load slows with it, and the collapse never shows up. Here the schedule is
    absolute - request k is due at k / level of a nominal request duration -
    and late submissions are recorded as lateness, so an overloaded level
    shows up first in TTFT/latency and per-stream rate, then in error rate,
    exactly the way an operator sees it.
    """
    start = time.perf_counter()

    # Nominal request duration guessing: the suite's serial phase has already
    # measured a short round trip at c=1. Scale it by the load probe's token
    # count to estimate what one load request costs the server, then derive
    # each request's scheduled offset. This turns the probe count into a
    # duration rather than a guess: a slow endpoint gets few, widely spaced
    # probes, which is exactly when the open-loop cadence matters.
    nominal_s: float | None = None
    decoded = report.decode_tokens_per_sec
    if report.serial_latency.count and report.serial_latency.mean:
        # serial probes ask for 24 tokens; a load request asks for 96, and
        # past c=1 the prefill part is unchanged. Linearity in tokens is rough
        # but beats a constant.
        nominal_s = (report.serial_latency.mean / 1000.0) * (perf.load_max_tokens / 24.0)
    elif decoded:
        nominal_s = perf.load_max_tokens / decoded
    interval_s = (nominal_s or 1.0) / level
    # Never less than one request per worker slot per nominal duration - an
    # endpoint that answers instantly would otherwise fire thousands of
    # requests in a second.
    count = max(count, level * MIN_ROUNDS_PER_LEVEL)

    executor = ThreadPoolExecutor(max_workers=min(MAX_CONCURRENCY, level * 4))
    local = threading.local()
    clients: list[ChatClient] = []
    clients_lock = threading.Lock()

    # A shared preamble, prepend-only across every probe of this level: prefix
    # caches see the same leading tokens and serve the way they would under
    # real org traffic. A nonce at the head of the suffix keeps the *cache key*
    # fresh (otherwise the whole mix collapses onto one cache line).
    prefix = _FILLER * 40 if perf.shared_prefix else ""
    mix_bag = _expand_mix(perf.workload_mix) if perf.workload_mix else None

    def job(idx: int, scheduled_s: float) -> _Sample:
        now = time.perf_counter() - start
        if now < scheduled_s:
            time.sleep(scheduled_s - now)
        worker = getattr(local, "client", None)
        if worker is None:
            worker = _new_client(config)
            local.client = worker
            # Register every created client so its session's sockets can be
            # closed once the level finishes; thread-local clients would
            # otherwise leak one requests.Session per worker thread per level.
            with clients_lock:
                clients.append(worker)
        if mix_bag is not None:
            task = mix_bag[idx % len(mix_bag)]
            sample = _probe(
                worker, _task_prompt(task, idx, prefix), task.max_tokens, attempts,
            )
            sample.task = task.name
            return sample
        return _probe(
            worker, prefix + _count_prompt(5000 + idx, upto=80),
            perf.load_max_tokens, attempts,
        )

    try:
        started = time.perf_counter()
        futures = [
            executor.submit(job, i, i * interval_s)
            for i in range(count)
        ]
        results: list[_Sample] = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:  # noqa: BLE001
                results.append(
                    _Sample(metrics=RequestMetrics(ok=False, error=str(e)))
                )
            tick()
    finally:
        executor.shutdown(wait=True)
        for client in clients:
            client.session.close()
    wall_ms = (time.perf_counter() - started) * 1000.0

    ok = [s for s in results if s.ok]
    # Per-task breakdown: latency + per-stream rate grouped by task name,
    # so the report can show which workload shape trips the SLO first.
    task_stats: dict[str, dict] = {}
    if mix_bag is not None:
        for name in {t.name for t in perf.workload_mix or ()}:
            task_ok = [s for s in ok if s.task == name]
            if not task_ok:
                continue
            task_stats[name] = {
                "count": len(task_ok),
                "latency_ms": LatencyStats.from_samples(
                    [s.metrics.latency_ms for s in task_ok]
                ).to_dict(),
                "stream_tps": LatencyStats.from_samples(
                    [s.metrics.output_tokens_per_sec
                     for s in task_ok if s.metrics.output_tokens_per_sec is not None]
                ).to_dict(),
            }
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
        stream_tps=LatencyStats.from_samples(
            [s.metrics.output_tokens_per_sec
             for s in ok if s.metrics.output_tokens_per_sec is not None]
        ),
        task_stats=task_stats,
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
    # Extra "warm" probes per grid cell that re-run cold probe 0's *identical*
    # prompt after the burst, so a prefix cache shows up as a warm-vs-cold
    # TTFT/prefill difference rather than polluting the cold number. 0 disables
    # the pairing. Serial by design so warm TTFT is not polluted by in-flight
    # siblings.
    warm_probes: int = 1

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
        under-report what a multi-level sweep is about to spend. Warm-pair
        probes are counted per cell; skipped/refused cells fire fewer (this is
        an upper bound, same as for short-circuited levels).
        """
        cells = sum(
            1
            for size in sorted(set(self.context_sizes))
            for level in self.levels_for_size(size)
        )
        return (
            sum(
                self.probes_for_size(size, level)
                for size in sorted(set(self.context_sizes))
                for level in self.levels_for_size(size)
            )
            + cells * max(0, self.warm_probes)
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

    def probe(prompt: str) -> _Sample:
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
            return _probe(client, prompt, sweep.probe_max_tokens, attempts)
        finally:
            client.session.close()

    def job(idx: int) -> _Sample:
        # Cold probes get a per-probe nonce so server-side caching cannot make
        # later probes of the burst look artificially fast.
        return probe(_context_prompt(nonce_base + idx, size))

    def warm_job() -> _Sample:
        # Re-run cold probe 0's exact prompt - the cache is now warm.
        return probe(_context_prompt(nonce_base, size))

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

    # Warm pair: only worth firing when cold probe 0 was actually served -
    # there is nothing to warm against otherwise. Keyed on the first cold
    # result, not the burst aggregate, so a cell whose probes all failed for a
    # non-context reason does not waste warm budget either.
    warm_results: list[_Sample] = []
    if sweep.warm_probes > 0 and not skipped and results and results[0].ok:
        for _ in range(sweep.warm_probes):
            warm_results.append(warm_job())
            tick()
    warm_ok = [s for s in warm_results if s.ok]

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
        warm_ttft=LatencyStats.from_samples(
            [s.metrics.ttft_ms for s in warm_ok if s.metrics.ttft_ms is not None]
        ),
        warm_prompt_tokens=sum(s.metrics.prompt_tokens for s in warm_ok),
        cached_tokens=sum(s.metrics.cached_tokens for s in warm_ok),
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
        "TTFT mean | Prefill tok/s | Warm TTFT | Warm prefill tok/s | Cached tok | "
        "Probe output tok/s | Note |",
        "|------------------|-------------|--------|--------|-------------|-------------|"
        "-----------|-----------------|-----------|--------------------|------------|"
        "-----------------------|------|",
    ]
    for point in sorted(points, key=lambda p: (p.context_tokens, p.concurrency)):
        if point.skipped:
            note = point.skip_reason or "context limit"
            lines.append(
                f"| {point.context_tokens:,} | - | {point.requests} | - | - | - | - | - | - | - | - | - | skipped: {note} |"
            )
            continue
        cached = f"{point.cached_tokens:,}" if point.cached_tokens else "-"
        lines.append(
            f"| {point.context_tokens:,} | {point.concurrency} | {point.requests} "
            f"| {point.errors} | {_ms(point.latency.p50)} | {_ms(point.latency.p95)} "
            f"| {_ms(point.ttft.mean)} | {_rate(point.prompt_tokens_per_sec)} "
            f"| {_ms(point.warm_ttft.mean)} | {_rate(point.warm_prompt_tokens_per_sec)} "
            f"| {cached} "
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
    capacity = report.slo_capacity
    if capacity is not None:
        slo_bits = []
        if report.slo_ttft_p95_ms > 0:
            slo_bits.append(f"p95 TTFT <= {report.slo_ttft_p95_ms:.0f} ms")
        if report.slo_stream_tps_p50 > 0:
            slo_bits.append(f"p50 per-stream decode >= {report.slo_stream_tps_p50:.1f} tok/s")
        if report.slo_error_rate > 0:
            slo_bits.append(f"errors <= {report.slo_error_rate * 100:.0f}%")
        users = report.capacity_users
        lines.append(
            f"- **Meets SLO up to**: concurrency {capacity}"
            + (f" (~{users:.0f} active users)" if users else "")
            + (" (" + ", ".join(slo_bits) + ")" if slo_bits else "")
        )
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
    cp = report.cache_probe
    if cp is not None:
        parts = []
        if cp.cache_hit_ratio is not None:
            parts.append(f"hit ratio {cp.cache_hit_ratio * 100:.0f}%")
        if cp.ttft_speedup is not None:
            parts.append(f"TTFT {cp.ttft_speedup:.1f}x faster warm")
        if cp.prefill_gain is not None:
            parts.append(f"prefill {cp.prefill_gain:.1f}x")
        detail = " · ".join(parts) if parts else "measured (no TTFT/hits to compare)"
        lines.append(
            f"- **Prefix cache**: {detail} "
            f"({cp.prompt_tokens:,}-token prompt, {cp.warm_probes} warm probe(s))"
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
            "| Concurrency | Requests | Errors | Req/s | Aggregate tok/s | Per-stream tok/s | p50 latency | p95 latency | p50 TTFT |",
            "|-------------|----------|--------|-------|-----------------|------------------|-------------|-------------|----------|",
        ]
        for point in sorted(report.concurrency, key=lambda p: p.concurrency):
            marker = "*" if point.concurrency > CLIENT_OVERHEAD_CONCURRENCY else ""
            stream = _rate(point.stream_tps.p50)
            lines.append(
                f"| {point.concurrency}{marker} | {point.requests} | {point.errors} "
                f"({point.error_rate * 100:.0f}%) | {point.requests_per_sec:.2f} | "
                f"{point.output_tokens_per_sec:.1f} | {stream} | {_ms(point.latency.p50)} | "
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
        stream = _rate(point.stream_tps.p50)
        lines.append(
            f"    c={point.concurrency:<3} {point.requests_per_sec:6.2f} req/s  "
            f"{point.output_tokens_per_sec:7.1f} tok/s  {stream:>12}/stream  "
            f"p50 {_ms(point.latency.p50):>9}  p95 {_ms(point.latency.p95):>9}  "
            f"errors {point.errors}/{point.requests}"
        )
    capacity = report.slo_capacity
    if capacity is not None:
        users = report.capacity_users
        lines.append(
            f"  SLO capacity: c={capacity}"
            + (f" (~{users:.0f} active users at {report.requests_per_user_hour:.0f} req/user/h)"
               if users else "")
        )
    cp = report.cache_probe
    if cp is not None:
        bits = []
        if cp.cache_hit_ratio is not None:
            bits.append(f"hits {cp.cache_hit_ratio * 100:.0f}%")
        if cp.ttft_speedup is not None:
            bits.append(f"TTFT {cp.ttft_speedup:.1f}x")
        if cp.prefill_gain is not None:
            bits.append(f"prefill {cp.prefill_gain:.1f}x")
        lines.append(f"  cache    {' | '.join(bits) if bits else 'measured'}")
    for note in report.notes:
        lines.append(f"  ! {note}")
    return "\n".join(lines)
