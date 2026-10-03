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
EVALUATION_SCHEMA_VERSION = 2


@dataclass
class CriterionResult:
    """One auditable requirement inside a question evaluation.

    Criteria are deliberately small and explicit.  A criterion can explain a
    partial result without changing the question's strict ``passed`` verdict.
    ``evidence`` is a short machine-readable pointer (for example a JSON path
    or fixture id); it should not contain an answer key that was not already
    part of the saved benchmark record.
    """

    criterion_id: str
    status: str = "not_evaluated"  # pass | fail | not_evaluated | error
    earned: float = 0.0
    possible: float = 1.0
    dimension: str = "content"
    mandatory: bool = True
    critical: bool = False
    group: str | None = None
    depends_on: list[str] = field(default_factory=list)
    reason_code: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail", "not_evaluated", "error"}:
            raise ValueError(f"unknown criterion status: {self.status!r}")
        self.earned = max(0.0, float(self.earned))
        self.possible = max(0.0, float(self.possible))
        if self.possible and self.earned > self.possible:
            self.earned = self.possible

    @property
    def achievement(self) -> float | None:
        if not self.possible or self.status in {"not_evaluated", "error"}:
            return None
        return self.earned / self.possible

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.criterion_id,
            "status": self.status,
            "earned": self.earned,
            "possible": self.possible,
            "achievement": self.achievement,
            "dimension": self.dimension,
            "mandatory": self.mandatory,
            "critical": self.critical,
            "group": self.group,
            "depends_on": self.depends_on,
            "reason_code": self.reason_code,
            "evidence": self.evidence,
        }


@dataclass
class EvaluationResult:
    """Versioned, criterion-level evaluation attached to a model result."""

    evaluator: str
    score: float = 0.0
    full_pass: bool = False
    outcome: str = ""
    criteria: list[CriterionResult] = field(default_factory=list)
    contract_score: float | None = None
    evaluator_version: str = "1"
    schema_version: int = EVALUATION_SCHEMA_VERSION

    @property
    def criterion_achievement(self) -> float | None:
        # Contract, availability, and efficiency checks are reported separately. Keeping them
        # out of content achievement prevents a valid JSON envelope with an
        # invalid schema from looking like a fully correct task.
        scored = [
            c
            for c in self.criteria
            if (
                c.achievement is not None
                or c.reason_code in {"blocked_by_structure", "blocked_by_dependency", "fixture_error", "input_mutation"}
            )
            and c.dimension not in {"contract", "availability", "efficiency"}
        ]
        if not scored:
            return None
        possible = sum(c.possible for c in scored)
        return sum(c.earned for c in scored) / possible if possible else None

    @property
    def criterion_count(self) -> int:
        return len(self.criteria)

    @property
    def failed_criteria(self) -> int:
        return sum(c.status in {"fail", "error"} for c in self.criteria)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator": self.evaluator,
            "evaluator_version": self.evaluator_version,
            "score": self.score,
            "full_pass": self.full_pass,
            "outcome": self.outcome,
            "criterion_achievement": self.criterion_achievement,
            "achievement_note": "Unmet required fields and code-fixture exceptions remain in the achievement denominator. Parsing, infrastructure availability and efficiency are separate.",
            "contract_score": self.contract_score,
            "criterion_count": self.criterion_count,
            "failed_criteria": self.failed_criteria,
            "criteria": [c.to_dict() for c in self.criteria],
        }


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
    metadata: dict = field(default_factory=dict)
    # Local simulation definition, never included in the model's prompt.
    interaction: dict | None = None
    # Optional criterion contract for new questions. Legacy questions keep the
    # evaluator-defined fallback until they opt into named requirements.
    rubric: list[dict[str, Any]] | None = None

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
    prompt_tokens_estimated: bool = False
    completion_tokens_estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class RequestMetrics:
    """Timing for a single model call.

    ``ttft_ms`` (time to first token) is only available when the request was
    streamed; it is ``None`` otherwise. ``latency_ms`` measures the successful
    attempt, or all attempts and backoff when the request fails.
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
    finish_reason: str | None = None
    prompt_tokens_estimated: bool = False
    completion_tokens_estimated: bool = False

    stream_chunks: int = 0
    stream_span_ms: float | None = None

    @property
    def burst_delivery(self) -> bool:
        return (
            self.streamed
            and self.stream_chunks > 0
            and (
                self.stream_chunks == 1
                or (self.stream_span_ms is not None and self.stream_span_ms < 10)
            )
        )


@dataclass
class Result:
    question: Question
    response: str
    score: float
    detail: str = ""
    tokens: TokenUsage = field(default_factory=TokenUsage)
    metrics: RequestMetrics = field(default_factory=RequestMetrics)
    cached: bool = False
    outcome: str = ""
    diagnostics: dict = field(default_factory=dict)
    evaluation: EvaluationResult | None = None

    @property
    def is_scored(self) -> bool:
        return self.metrics.ok and self.outcome not in {
            "endpoint_error",
            "unsupported_context",
            "evaluator_error",
            "cancelled",
        }

    @property
    def passed(self) -> bool:
        if not self.is_scored:
            return False
        if self.evaluation is not None:
            return self.evaluation.full_pass
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
        return [r for r in self.results if r.is_scored]

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
    concurrency: int
    requests: int
    errors: int
    wall_ms: float
    latency: LatencyStats
    ttft: LatencyStats
    output_tokens: int
    prompt_tokens: int
    failure_latency: LatencyStats = field(default_factory=LatencyStats)
    estimated_token_requests: int = 0
    streamed_requests: int = 0
    burst_delivery_requests: int = 0
    cached_tokens: int = 0

    @property
    def requests_per_sec(self):
        return (self.requests - self.errors) * 1000 / self.wall_ms if self.wall_ms else 0.0

    @property
    def output_tokens_per_sec(self):
        return self.output_tokens * 1000 / self.wall_ms if self.wall_ms else 0.0

    @property
    def error_rate(self):
        return self.errors / self.requests if self.requests else 0.0

    def to_dict(self):
        return {
            **{
                key: value
                for key, value in self.__dict__.items()
                if key not in {"latency", "ttft", "failure_latency"}
            },
            "latency": self.latency.to_dict(),
            "ttft": self.ttft.to_dict(),
            "failure_latency": self.failure_latency.to_dict(),
            "requests_per_sec": self.requests_per_sec,
            "output_tokens_per_sec": self.output_tokens_per_sec,
            "error_rate": self.error_rate,
        }


@dataclass
class PerfReport:
    model: str
    protocol: dict = field(default_factory=dict)
    concurrency: list[ConcurrencyPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cancelled: bool = False

    @property
    def peak_output_tokens_per_sec(self):
        return max((p.output_tokens_per_sec for p in self.concurrency), default=None)

    @property
    def peak_requests_per_sec(self):
        return max((p.requests_per_sec for p in self.concurrency), default=None)

    def to_dict(self):
        return {
            "schema_version": 3,
            "model": self.model,
            "protocol": self.protocol,
            "concurrency": [p.to_dict() for p in self.concurrency],
            "notes": self.notes,
            "cancelled": self.cancelled,
            "peak_output_tokens_per_sec": self.peak_output_tokens_per_sec,
            "peak_requests_per_sec": self.peak_requests_per_sec,
        }
