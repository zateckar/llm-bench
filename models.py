"""Shared data structures for the LLM benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Scores are floats; comparing against a threshold needs a little slack so that
# 5/5 == 1.0 does not fail a `>= 1.0` test because of accumulated FP error.
SCORE_EPSILON = 1e-9

# Difficulty tiers, used for weighted scoring in reports.
DIFFICULTY_WEIGHTS = {
    "easy": 1.0,
    "medium": 1.5,
    "hard": 2.0,
    "expert": 3.0,
}
DEFAULT_DIFFICULTY = "medium"


@dataclass
class Question:
    id: str
    category: str
    prompt: str
    evaluator: str
    expected: Any = None
    system_prompt: str | None = None
    # A question passes only when its score reaches this bar. Default 1.0: the
    # answer must be fully correct. Questions whose evaluator produces genuinely
    # meaningful partial credit (a security review finding 4 of 5 issues) lower
    # it explicitly in the YAML.
    pass_threshold: float = 1.0
    difficulty: str = DEFAULT_DIFFICULTY
    # Relative weight in the weighted score. Defaults to the difficulty weight.
    weight: float | None = None
    source: str | None = None
    description: str | None = None
    # Per-question generation overrides (long-output tests need more headroom).
    max_tokens: int | None = None

    @property
    def effective_weight(self) -> float:
        if self.weight is not None:
            return float(self.weight)
        return DIFFICULTY_WEIGHTS.get(self.difficulty, DIFFICULTY_WEIGHTS[DEFAULT_DIFFICULTY])


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Prompt tokens the server served from its prefix cache instead of
    # prefilling. Both common report shapes are normalised here (vLLM's
    # prompt_cache_hit_tokens, OpenAI's prompt_tokens_details.cached_tokens).
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class RequestMetrics:
    """Timing for a single model call.

    ``ttft_ms`` (time to first token) is only available when the request was
    streamed; it is ``None`` otherwise. ``latency_ms`` is always the full
    wall-clock time of the call including retries-free single attempt.
    """

    latency_ms: float = 0.0
    ttft_ms: float | None = None
    completion_tokens: int = 0
    prompt_tokens: int = 0
    ok: bool = True
    error: str | None = None
    attempts: int = 1
    streamed: bool = False
    # Prompt tokens served from the server's prefix cache on this call. 0
    # either means a genuine miss or that the server does not report hits;
    # callers can only rely on it being *nonzero*.
    cached_tokens: int = 0

    @property
    def decode_ms(self) -> float | None:
        """Time spent generating after the first token arrived."""
        if self.ttft_ms is None:
            return None
        return max(0.0, self.latency_ms - self.ttft_ms)

    @property
    def output_tokens_per_sec(self) -> float | None:
        """Sustained decode rate: output tokens divided by post-TTFT time.

        Falls back to total latency when TTFT is unknown, which understates the
        rate but is still comparable across models measured the same way.
        """
        if not self.completion_tokens:
            return None
        window = self.decode_ms if self.decode_ms not in (None, 0.0) else self.latency_ms
        if not window:
            return None
        return self.completion_tokens / (window / 1000.0)


@dataclass
class Result:
    question: Question
    response: str
    score: float
    detail: str = ""
    tokens: TokenUsage = field(default_factory=TokenUsage)
    metrics: RequestMetrics = field(default_factory=RequestMetrics)
    cached: bool = False

    @property
    def passed(self) -> bool:
        return self.score >= self.question.pass_threshold - SCORE_EPSILON

    @property
    def is_transport_error(self) -> bool:
        """True when the model never answered (network/HTTP failure).

        These are excluded from quality percentages: a 502 from the endpoint is
        not evidence about the model's capability. ``metrics.ok`` is set by the
        LLM client only for transport failures, so the response text is never
        inspected here - a genuine answer may legitimately start with "[API ERROR".
        """
        return not self.metrics.ok


@dataclass
class CategoryResult:
    name: str
    results: list[Result] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def scored(self) -> list[Result]:
        """Results that actually produced a model answer."""
        return [r for r in self.results if not r.is_transport_error]

    @property
    def errors(self) -> int:
        return self.total - len(self.scored)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.scored if r.passed)

    @property
    def score_pct(self) -> float:
        scored = self.scored
        if not scored:
            return 0.0
        return sum(r.score for r in scored) / len(scored) * 100

    @property
    def pass_rate_pct(self) -> float:
        scored = self.scored
        if not scored:
            return 0.0
        return self.passed / len(scored) * 100

    @property
    def weighted_score_pct(self) -> float:
        """Score weighted by question difficulty."""
        scored = self.scored
        total_weight = sum(r.question.effective_weight for r in scored)
        if not total_weight:
            return 0.0
        earned = sum(r.score * r.question.effective_weight for r in scored)
        return earned / total_weight * 100

    @property
    def tokens(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=sum(r.tokens.prompt_tokens for r in self.results),
            completion_tokens=sum(r.tokens.completion_tokens for r in self.results),
            cached_tokens=sum(r.tokens.cached_tokens for r in self.results),
        )

    @property
    def latencies_ms(self) -> list[float]:
        return [
            r.metrics.latency_ms
            for r in self.results
            if not r.cached and r.metrics.ok and r.metrics.latency_ms > 0
        ]

    @property
    def median_latency_ms(self) -> float | None:
        return percentile(self.latencies_ms, 50)


# ---------------------------------------------------------------------------
# Statistics helpers (shared by the report and the perf suite)
# ---------------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile. Returns None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


@dataclass
class LatencyStats:
    """Summary of a latency sample, in milliseconds."""

    count: int = 0
    mean: float | None = None
    stdev: float | None = None
    min: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    max: float | None = None

    @classmethod
    def from_samples(cls, values: list[float]) -> "LatencyStats":
        clean = [v for v in values if v is not None]
        if not clean:
            return cls()
        return cls(
            count=len(clean),
            mean=mean(clean),
            stdev=stdev(clean),
            min=min(clean),
            p50=percentile(clean, 50),
            p90=percentile(clean, 90),
            p95=percentile(clean, 95),
            p99=percentile(clean, 99),
            max=max(clean),
        )

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_ms": self.mean,
            "stdev_ms": self.stdev,
            "min_ms": self.min,
            "p50_ms": self.p50,
            "p90_ms": self.p90,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "max_ms": self.max,
        }


@dataclass
class ConcurrencyPoint:
    """One rung of the concurrency sweep."""

    concurrency: int
    requests: int
    errors: int
    wall_ms: float
    latency: LatencyStats
    ttft: LatencyStats
    output_tokens: int
    prompt_tokens: int
    # Per-request sustained decode rates (tok/s per stream). This is what a
    # user actually watches; the aggregate figure can stay flat while every
    # individual stream has already slowed to a crawl.
    stream_tps: LatencyStats = field(default_factory=LatencyStats)
    # Per-workload-task breakdown, keyed by task name - the mix tells you
    # *which* shape of request hurt most at this level, aggregate stats do not.
    task_stats: dict[str, dict] = field(default_factory=dict)

    @property
    def requests_per_sec(self) -> float:
        return (self.requests - self.errors) / (self.wall_ms / 1000.0) if self.wall_ms else 0.0

    @property
    def output_tokens_per_sec(self) -> float:
        """Aggregate generation throughput across all in-flight requests."""
        return self.output_tokens / (self.wall_ms / 1000.0) if self.wall_ms else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    def to_dict(self) -> dict:
        return {
            "concurrency": self.concurrency,
            "requests": self.requests,
            "errors": self.errors,
            "error_rate": self.error_rate,
            "wall_ms": self.wall_ms,
            "requests_per_sec": self.requests_per_sec,
            "output_tokens_per_sec": self.output_tokens_per_sec,
            "output_tokens": self.output_tokens,
            "prompt_tokens": self.prompt_tokens,
            "latency": self.latency.to_dict(),
            "ttft": self.ttft.to_dict(),
            "stream_tps": self.stream_tps.to_dict(),
            "task_stats": self.task_stats,
        }


@dataclass
class ContextPoint:
    """One rung of the context-length scalability sweep.

    ``skipped`` is set (with ``skip_reason``) when the endpoint refused the
    size entirely - e.g. its context window is smaller than the target. A
    skipped point is excluded from charts rather than drawn as an error bar.
    """

    context_tokens: int
    concurrency: int
    requests: int
    errors: int
    wall_ms: float
    latency: LatencyStats
    ttft: LatencyStats
    prompt_tokens: int
    output_tokens: int
    skipped: bool = False
    skip_reason: str = ""
    notes: list[str] = field(default_factory=list)
    # Warm (cache-hot) pair probes re-running cold probe 0's exact prompt after
    # the burst. warm_ttft is the warm-pair latency distribution;
    # warm_prompt_tokens / cached_tokens are the server-reported sums so the
    # cache benefit is measured, not inferred from a timing dip. Serial by
    # design, after the burst, so warm TTFT is not polluted by in-flight
    # siblings - which also means the largest sizes may see post-burst cache
    # eviction; that is the honest answer for "what does a repeat query cost".
    warm_ttft: LatencyStats = field(default_factory=LatencyStats)
    warm_prompt_tokens: int = 0
    cached_tokens: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    @property
    def prompt_tokens_per_sec(self) -> float | None:
        """Server-side ingestion rate: prompt tokens per second of TTFT window.

        Uses TTFT (not total latency) so queueing behind concurrent probes does
        not hide prefill throughput. None when no streamed TTFT was observed.
        """
        mean_ttft = self.ttft.mean
        if not mean_ttft or mean_ttft <= 0:
            return None
        # prompt_tokens is summed over successful probes only, so divide by the
        # successful count - not all requests - or errors understate the rate.
        succeeded = self.requests - self.errors
        if succeeded <= 0:
            return None
        avg_prompt = self.prompt_tokens / succeeded
        return avg_prompt / (mean_ttft / 1000.0)

    @property
    def output_tokens_per_sec(self) -> float | None:
        """Aggregate generation rate across the probes at this size."""
        if not self.wall_ms:
            return None
        return self.output_tokens / (self.wall_ms / 1000.0)

    @property
    def warm_prompt_tokens_per_sec(self) -> float | None:
        """Prefill rate on the warm pair - the cache-hit ingestion rate.

        Mirrors ``prompt_tokens_per_sec`` over the warm probes. When the server
        reports cache hits, this and ``cached_tokens`` answer "what did the
        prefix cache actually save"; without hits it is simply the warm TTFT
        expressed as a rate.
        """
        mean_ttft = self.warm_ttft.mean
        if not mean_ttft or mean_ttft <= 0 or not self.warm_ttft.count:
            return None
        avg_prompt = self.warm_prompt_tokens / self.warm_ttft.count
        return avg_prompt / (mean_ttft / 1000.0)

    def to_dict(self) -> dict:
        return {
            "context_tokens": self.context_tokens,
            "concurrency": self.concurrency,
            "requests": self.requests,
            "errors": self.errors,
            "error_rate": self.error_rate,
            "wall_ms": self.wall_ms,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "prompt_tokens_per_sec": self.prompt_tokens_per_sec,
            "output_tokens_per_sec": self.output_tokens_per_sec,
            "latency": self.latency.to_dict(),
            "ttft": self.ttft.to_dict(),
            "warm_ttft": self.warm_ttft.to_dict(),
            "warm_prompt_tokens": self.warm_prompt_tokens,
            "warm_prompt_tokens_per_sec": self.warm_prompt_tokens_per_sec,
            "cached_tokens": self.cached_tokens,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "notes": self.notes,
        }


@dataclass
class CacheProbe:
    """One cold/warm pair against a large shared prefix.

    Answers "does this stack's prefix cache help my workload" with numbers,
    not a timing guess: ``cached_tokens`` is the server-reported hit count
    (`prompt_cache_hit_tokens` / `prompt_tokens_details.cached_tokens`), and
    the speedups are warm-vs-cold on an identical prompt. Prompt-level fields
    are per-probe values; the ``cached_/warm_`` sums aggregate the warm probes
    so a partial failure still yields a valid ratio.
    """

    prompt_tokens: int = 0         # server-reported prompt size (per probe)
    warm_probes: int = 0
    warm_prompt_tokens: int = 0    # summed over warm probes
    cached_tokens: int = 0         # summed over warm probes (server-reported)
    cold_ttft_ms: float | None = None
    warm_ttft_ms: float | None = None      # mean over warm probes
    cold_prefill_tokens_per_sec: float | None = None
    warm_prefill_tokens_per_sec: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def cache_hit_ratio(self) -> float | None:
        """Fraction of warm prompt tokens the cache served. None = unmeasured."""
        if not self.warm_prompt_tokens:
            return None
        return self.cached_tokens / self.warm_prompt_tokens

    @property
    def ttft_speedup(self) -> float | None:
        """cold TTFT / warm TTFT. >1 means the repeat is faster to first token."""
        if not self.cold_ttft_ms or not self.warm_ttft_ms:
            return None
        return self.cold_ttft_ms / self.warm_ttft_ms

    @property
    def prefill_gain(self) -> float | None:
        """warm prefill rate / cold prefill rate."""
        if not self.cold_prefill_tokens_per_sec or not self.warm_prefill_tokens_per_sec:
            return None
        return self.warm_prefill_tokens_per_sec / self.cold_prefill_tokens_per_sec

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "warm_probes": self.warm_probes,
            "warm_prompt_tokens": self.warm_prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cold_ttft_ms": self.cold_ttft_ms,
            "warm_ttft_ms": self.warm_ttft_ms,
            "cold_prefill_tokens_per_sec": self.cold_prefill_tokens_per_sec,
            "warm_prefill_tokens_per_sec": self.warm_prefill_tokens_per_sec,
            "cache_hit_ratio": self.cache_hit_ratio,
            "ttft_speedup": self.ttft_speedup,
            "prefill_gain": self.prefill_gain,
            "notes": self.notes,
        }


@dataclass
class PerfReport:
    """Everything the performance suite measured."""

    model: str = ""
    endpoint: str = ""
    streaming: bool = False
    # Single-stream (concurrency 1) behaviour
    serial_latency: LatencyStats = field(default_factory=LatencyStats)
    serial_ttft: LatencyStats = field(default_factory=LatencyStats)
    decode_tokens_per_sec: float | None = None
    prefill_tokens_per_sec: float | None = None
    long_output_tokens_per_sec: float | None = None
    # Load behaviour
    concurrency: list[ConcurrencyPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Acceptance thresholds the capacity verdict is computed against. Zero or
    # None disables that criterion. Stored on the report so a reader can see
    # which contract the verdict refers to.
    slo_ttft_p95_ms: float = 2000.0
    slo_stream_tps_p50: float = 15.0
    slo_error_rate: float = 0.01
    # How many requests one active user generates in an hour; converts the
    # measured request rate into "people we can serve".
    requests_per_user_hour: float = 60.0
    # Prefix-cache probe result; None when the phase was off or failed.
    cache_probe: CacheProbe | None = None

    @property
    def peak_output_tokens_per_sec(self) -> float | None:
        if not self.concurrency:
            return None
        return max(p.output_tokens_per_sec for p in self.concurrency)

    @property
    def peak_requests_per_sec(self) -> float | None:
        if not self.concurrency:
            return None
        return max(p.requests_per_sec for p in self.concurrency)

    @property
    def saturation_concurrency(self) -> int | None:
        """Lowest concurrency that reached (within 5% of) peak throughput.

        Past this point extra concurrency buys latency, not throughput.
        """
        peak = self.peak_output_tokens_per_sec
        if not peak:
            return None
        for point in sorted(self.concurrency, key=lambda p: p.concurrency):
            if point.output_tokens_per_sec >= peak * 0.95:
                return point.concurrency
        return None

    @property
    def scaling_efficiency(self) -> float | None:
        """Throughput at max concurrency vs. perfect linear scaling from c=1."""
        if len(self.concurrency) < 2:
            return None
        points = sorted(self.concurrency, key=lambda p: p.concurrency)
        base, top = points[0], points[-1]
        if not base.output_tokens_per_sec or base.concurrency != 1:
            return None
        ideal = base.output_tokens_per_sec * top.concurrency
        return top.output_tokens_per_sec / ideal if ideal else None

    def point_meets_slo(self, point: "ConcurrencyPoint") -> bool:
        """Whether one concurrency level satisfies every enabled SLO criterion."""
        if self.slo_ttft_p95_ms > 0:
            p95 = point.ttft.p95
            if p95 is None or p95 > self.slo_ttft_p95_ms:
                return False
        if self.slo_stream_tps_p50 > 0:
            tps = point.stream_tps.p50
            if tps is None or tps < self.slo_stream_tps_p50:
                return False
        if self.slo_error_rate > 0 and point.error_rate > self.slo_error_rate:
            return False
        return True

    @property
    def slo_capacity(self) -> int | None:
        """Largest measured concurrency that still meets the SLO.

        Levels are sorted, so the highest passing level whose predecessors all
        pass too is the capacity; a gap in the pass chain ends the search -
        a level that "recovers" after a failing level is usually luck, not
        headroom.
        """
        if not self.concurrency:
            return None
        capacity = None
        for point in sorted(self.concurrency, key=lambda p: p.concurrency):
            if not self.point_meets_slo(point):
                break
            capacity = point.concurrency
        return capacity

    @property
    def capacity_users(self) -> float | None:
        """Rough number of simultaneously active users the endpoint can serve.

        ``slo_capacity`` is in-flight requests, not people: between requests a
        user reads the answer and thinks. With ``requests_per_user_hour`` r, a
        think time of (3600/r - request duration) follows, and the Little's-law
        mapping is users = c * 3600 / (3600 - c*r*L/1000), where L is the mean
        request latency in ms at the knee. When the request duration alone
        consumes the whole think budget (c*r*L/1000 approaches 3600) the model
        breaks down and None is returned instead of an absurd figure.
        """
        if not self.concurrency or self.requests_per_user_hour <= 0:
            return None
        capacity = self.slo_capacity
        if capacity is None:
            return None
        point = next(
            (p for p in self.concurrency if p.concurrency == capacity), None
        )
        mean_ms = point.latency.mean if point else None
        if not mean_ms:
            return None
        service_s = capacity * self.requests_per_user_hour * mean_ms / 3600_000.0
        if service_s >= 0.95:
            return None
        return capacity / (1.0 - service_s)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "streaming": self.streaming,
            "serial_latency": self.serial_latency.to_dict(),
            "serial_ttft": self.serial_ttft.to_dict(),
            "decode_tokens_per_sec": self.decode_tokens_per_sec,
            "prefill_tokens_per_sec": self.prefill_tokens_per_sec,
            "long_output_tokens_per_sec": self.long_output_tokens_per_sec,
            "peak_output_tokens_per_sec": self.peak_output_tokens_per_sec,
            "peak_requests_per_sec": self.peak_requests_per_sec,
            "saturation_concurrency": self.saturation_concurrency,
            "scaling_efficiency": self.scaling_efficiency,
            "slo_ttft_p95_ms": self.slo_ttft_p95_ms,
            "slo_stream_tps_p50": self.slo_stream_tps_p50,
            "slo_error_rate": self.slo_error_rate,
            "requests_per_user_hour": self.requests_per_user_hour,
            "slo_capacity": self.slo_capacity,
            "capacity_users": self.capacity_users,
            "cache_probe": self.cache_probe.to_dict() if self.cache_probe else None,
            "concurrency": [p.to_dict() for p in self.concurrency],
            "notes": self.notes,
        }
