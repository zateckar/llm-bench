"""Open-loop workload definitions, seeded arrival schedules and prompt construction.

A workload is a weighted mix of request classes. Each class draws input and
output lengths from bucket histograms, shares a fixed system prefix and
carries its own latency SLO. Schedules are a pure function of the settings and
the step index, so equal settings reproduce identical traffic on every run.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_INPUT_TOKENS = 1_048_576
MAX_OUTPUT_TOKENS = 65_536
MAX_SHARED_PREFIX = 524_288
MIN_BODY_TOKENS = 128
MIN_SHARED_PREFIX = 32
MAX_CLASSES = 6
MAX_BUCKETS = 16
MAX_RATES = 12
MIN_RATE, MAX_RATE = 0.01, 500.0
# Module constants (not Field bounds) so offline tests can use short steps.
MIN_STEP_SECONDS = 10
MAX_STEP_SECONDS = 14_400
MAX_TOTAL_SECONDS = 12 * 3600
MAX_EXPECTED_ARRIVALS = 500_000
MAX_IN_FLIGHT = 1024
DEFAULT_IN_FLIGHT = 256
NAME = re.compile(r"^[a-z0-9-]{1,24}$")


class SLO(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ttft_ms: float = Field(gt=0, le=600_000)
    tpot_ms: float | None = Field(None, gt=0, le=10_000)
    e2e_ms: float | None = Field(None, gt=0, le=3_600_000)


def _buckets(value, label, maximum):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_BUCKETS:
        raise ValueError(f"{label} needs 1-{MAX_BUCKETS} buckets")
    clean = []
    for bucket in value:
        if (not isinstance(bucket, (list, tuple)) or len(bucket) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bucket)):
            raise ValueError(f"{label} buckets are [min, max, weight]")
        low, high, weight = bucket
        if low != int(low) or high != int(high) or not 1 <= low <= high <= maximum:
            raise ValueError(f"{label} bucket bounds must be integers with 1 <= min <= max <= {maximum:,}")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{label} bucket weights must be positive")
        clean.append([int(low), int(high), float(weight)])
    return clean


class RequestClass(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    weight: float = Field(gt=0, le=1_000_000)
    input: list
    output: list
    shared_prefix: int = Field(0, ge=0, le=MAX_SHARED_PREFIX)
    slo: SLO

    @field_validator("name")
    @classmethod
    def _name(cls, value):
        if not NAME.match(value):
            raise ValueError("Class names use 1-24 lowercase letters, digits or hyphens")
        return value

    @field_validator("input")
    @classmethod
    def _input(cls, value):
        return _buckets(value, "Input", MAX_INPUT_TOKENS)

    @field_validator("output")
    @classmethod
    def _output(cls, value):
        return _buckets(value, "Output", MAX_OUTPUT_TOKENS)

    @model_validator(mode="after")
    def _prefix_fits(self):
        if 0 < self.shared_prefix < MIN_SHARED_PREFIX:
            raise ValueError(f"Class {self.name}: a shared prefix needs at least {MIN_SHARED_PREFIX} tokens")
        smallest = min(low for low, _, _ in self.input)
        if smallest < self.shared_prefix + MIN_BODY_TOKENS:
            raise ValueError(f"Class {self.name}: every input bucket must be at least the shared prefix "
                             f"plus {MIN_BODY_TOKENS} tokens")
        return self

    def mean(self, key):
        buckets = getattr(self, key)
        total = sum(w for _, _, w in buckets)
        return sum((low + high) / 2 * w for low, high, w in buckets) / total


class Workload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    classes: list[RequestClass] = Field(min_length=1, max_length=MAX_CLASSES)

    @model_validator(mode="after")
    def _unique(self):
        names = [c.name for c in self.classes]
        if len(set(names)) != len(names):
            raise ValueError("Class names must be unique")
        return self

    def shares(self):
        total = sum(c.weight for c in self.classes)
        return [c.weight / total for c in self.classes]

    def fingerprint(self):
        canonical = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class LoadSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    preset: str = Field("custom", pattern=r"^[a-z-]{1,20}$")
    workload: Workload
    rates: list[float] = Field(min_length=1, max_length=MAX_RATES)
    step_seconds: int = 120
    arrival: Literal["poisson", "gamma"] = "poisson"
    burstiness: float = Field(2.0, ge=1, le=10)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    attainment: float = Field(0.95, ge=0.5, le=0.999)
    request_timeout: int = Field(300, ge=30, le=1800)
    stop_after_failed_steps: int = Field(2, ge=1, le=MAX_RATES)

    @field_validator("rates")
    @classmethod
    def _rates(cls, value):
        if any(isinstance(r, bool) or not math.isfinite(r) or not MIN_RATE <= r <= MAX_RATE for r in value):
            raise ValueError(f"Rates must be between {MIN_RATE} and {MAX_RATE:g} requests per second")
        if any(b <= a for a, b in zip(value, value[1:])):
            raise ValueError("Rates must be strictly increasing")
        return value

    @field_validator("step_seconds")
    @classmethod
    def _step(cls, value):
        if not MIN_STEP_SECONDS <= value <= MAX_STEP_SECONDS:
            raise ValueError(f"Step duration must be between {MIN_STEP_SECONDS} and {MAX_STEP_SECONDS:,} seconds")
        return value

    @model_validator(mode="after")
    def _budget(self):
        if len(self.rates) * self.step_seconds > MAX_TOTAL_SECONDS:
            raise ValueError(f"The ladder exceeds {MAX_TOTAL_SECONDS // 3600} hours of arrivals")
        if self.expected_arrivals > MAX_EXPECTED_ARRIVALS:
            raise ValueError(f"The ladder expects more than {MAX_EXPECTED_ARRIVALS:,} requests")
        return self

    @property
    def expected_arrivals(self):
        return round(sum(self.rates) * self.step_seconds)

    def comparison_key(self):
        """Settings that must match before two runs' steps may be compared."""
        return {"workload_hash": self.workload.fingerprint(), "arrival": self.arrival,
                "burstiness": self.burstiness if self.arrival == "gamma" else None,
                "step_seconds": self.step_seconds, "seed": self.seed, "attainment": self.attainment}


def _cls(name, weight, inputs, outputs, prefix, ttft, tpot, e2e=None):
    return {"name": name, "weight": weight, "input": inputs, "output": outputs,
            "shared_prefix": prefix, "slo": {"ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e}}


# Shapes follow typical internal traffic; teams should replace them with
# histograms from their gateway logs (advanced JSON) when available.
CHAT = _cls("chat", 80, [[500, 800, 30], [800, 1500, 40], [1500, 4000, 22], [4000, 8000, 8]],
            [[50, 150, 30], [150, 400, 45], [400, 800, 20], [800, 1500, 5]], 300, 2000, 50)
AGENT = _cls("agent", 20, [[8000, 16000, 30], [16000, 32000, 45], [32000, 64000, 25]],
             [[50, 200, 50], [200, 600, 35], [600, 1500, 15]], 6000, 10000, 100)
RAG = _cls("rag", 100, [[2000, 6000, 40], [6000, 12000, 45], [12000, 24000, 15]],
           [[100, 300, 50], [300, 700, 50]], 500, 4000, 50)
PRESETS = {
    "chat": {"classes": [CHAT]},
    "agent": {"classes": [AGENT]},
    "rag": {"classes": [RAG]},
    "mixed": {"classes": [CHAT, AGENT]},
}
PRESET_LABELS = {"chat": "Chat", "agent": "Agents · long context", "rag": "Retrieval-augmented answers",
                 "mixed": "Mixed · 80% chat, 20% agents", "custom": "Custom workload"}


def default_settings(preset="mixed"):
    return {"preset": preset, "workload": PRESETS[preset], "rates": [0.5, 1, 2, 4], "step_seconds": 120,
            "arrival": "poisson", "burstiness": 2.0, "seed": 0, "attainment": 0.95,
            "request_timeout": 300, "stop_after_failed_steps": 2}


def parse_settings(raw):
    """Validate a settings dict; presets supply the workload unless it is custom."""
    if not isinstance(raw, dict):
        raise ValueError("Load settings must be an object")
    raw = dict(raw)
    preset = raw.get("preset", "custom")
    if preset in PRESETS and "workload" not in raw:
        raw["workload"] = PRESETS[preset]
    settings = LoadSettings.model_validate(raw)
    if settings.preset in PRESETS and settings.workload.model_dump() != Workload.model_validate(PRESETS[settings.preset]).model_dump():
        settings = settings.model_copy(update={"preset": "custom"})
    if settings.preset not in PRESET_LABELS:
        raise ValueError("Unknown workload preset")
    return settings


def error_text(error):
    """Short pydantic/ValueError text suitable for a 422 response."""
    errors = getattr(error, "errors", None)
    if callable(errors):
        parts = []
        for item in errors()[:3]:
            where = ".".join(str(p) for p in item.get("loc", ()))
            message = str(item.get("msg", "")).removeprefix("Value error, ")
            parts.append(f"{where}: {message}" if where else message)
        return "; ".join(parts)
    return str(error)


@dataclass(frozen=True)
class Arrival:
    offset: float
    klass: int
    input_tokens: int
    output_tokens: int


def _draw(rng, buckets):
    low, high, _ = rng.choices(buckets, weights=[w for _, _, w in buckets])[0]
    return rng.randint(low, high)


def schedule(settings, step_index):
    rate = settings.rates[step_index]
    rng = random.Random(f"open-loop-v1:{settings.seed}:{step_index}:{rate!r}:{settings.step_seconds}")
    classes = settings.workload.classes
    weights = [c.weight for c in classes]
    shape = 1 / settings.burstiness ** 2
    arrivals, t = [], 0.0
    while True:
        if settings.arrival == "poisson":
            t += rng.expovariate(rate)
        else:
            # Gamma gaps with mean 1/rate and coefficient of variation `burstiness`.
            t += rng.gammavariate(shape, 1 / (rate * shape))
        if t >= settings.step_seconds:
            return arrivals
        k = rng.choices(range(len(classes)), weights=weights)[0]
        arrivals.append(Arrival(t, k, _draw(rng, classes[k].input), _draw(rng, classes[k].output)))


class PromptBuilder:
    """Exact cl100k_base reference lengths: the shared system prefix plus a unique body."""

    footer = ("\nIgnore the background. Count upward from 1, writing every integer separated by "
              "commas. Continue until you are stopped.")

    def __init__(self, workload, nonce):
        import tiktoken

        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.nonce = nonce
        longest = max(high for c in workload.classes for _, high, _ in c.input)
        self.padding = " pad" * longest
        self.footer_tokens = self.count(self.footer)
        self.systems = []
        for c in workload.classes:
            if not c.shared_prefix:
                self.systems.append(None)
                continue
            header = f"Workload {c.name} context {nonce}. Shared instructions follow:\n"
            text = header + self.padding[: 4 * (c.shared_prefix - self.count(header))]
            if self.count(text) != c.shared_prefix:
                raise ValueError("Shared prefix token budget mismatch")
            self.systems.append(text)
        self.shared = [c.shared_prefix for c in workload.classes]

    def count(self, text):
        return len(self.encoding.encode(text))

    def build(self, arrival, index):
        header = f"Request {self.nonce}-{index}. Background records follow:\n"
        body = arrival.input_tokens - self.shared[arrival.klass] - self.count(header) - self.footer_tokens
        return self.systems[arrival.klass], header + self.padding[: 4 * body] + self.footer
