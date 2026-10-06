"""Simulated users for the users-at-SLO test: user models, session plans and prompts.

A user model is a weighted mix of user classes (chat people, agents). Each
simulated user runs one session after another. A session is a conversation
whose history grows turn by turn and ends when its planned turns are used up
or the next turn would exceed the context cap. Between turns a user thinks
(people) or runs tools (agents). Plans are a pure function of the seed, the
context cap, the user count and the user index, so every model receives the
same sessions. See docs/design-session-capacity.md.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import random
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CLASSES = 4
MAX_BUCKETS = 16
MAX_SYSTEM_TOKENS = 131_072
MAX_INPUT_TOKENS = 524_288
MAX_OUTPUT_TOKENS = 32_768
MAX_TURNS = 1_000
MAX_THINK_SECONDS = 3_600.0
MIN_INPUT_TOKENS = 48
MAX_USERS = 1024
MIN_CAP, MAX_CAP = 1_024, 1_048_576
MAX_CAPS = 6
# Module constants (not Field bounds) so offline tests can use short levels.
MIN_WARMUP_SECONDS = 10
MIN_MEASURE_SECONDS = 30
MAX_PHASE_SECONDS = 3_600
# Chat framing allowance per message when checking a session against its cap.
MESSAGE_FRAMING = 8
NAME = re.compile(r"^[a-z0-9-]{1,24}$")


def _buckets(value, label, low, high, integer=True):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_BUCKETS:
        raise ValueError(f"{label} needs 1-{MAX_BUCKETS} buckets")
    clean = []
    for bucket in value:
        if (not isinstance(bucket, (list, tuple)) or len(bucket) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bucket)
                or not all(math.isfinite(v) for v in bucket)):
            raise ValueError(f"{label} buckets are [min, max, weight]")
        first, last, weight = bucket
        if integer and (first != int(first) or last != int(last)):
            raise ValueError(f"{label} bucket bounds must be integers")
        if not low <= first <= last <= high:
            raise ValueError(f"{label} bucket bounds must satisfy {low:g} <= min <= max <= {high:,g}")
        if weight <= 0:
            raise ValueError(f"{label} bucket weights must be positive")
        clean.append([int(first), int(last), float(weight)] if integer else [float(first), float(last), float(weight)])
    return clean


def _mean(buckets):
    total = sum(w for _, _, w in buckets)
    return sum((a + b) / 2 * w for a, b, w in buckets) / total


class SessionSLO(BaseModel):
    """Per-request targets: first token p95 and output speed p95 (tokens per second)."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    ttft_ms: int = Field(ge=100, le=600_000)
    output_tokens_per_sec: int = Field(ge=1, le=1_000)

    @property
    def tpot_ms(self):
        return 1000 / self.output_tokens_per_sec


class UserClass(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    weight: float = Field(gt=0, le=1_000_000)
    system_tokens: int = Field(0, ge=0, le=MAX_SYSTEM_TOKENS)
    first_input: list
    turn_input: list
    output: list
    think_seconds: list
    turns: list
    slo: SessionSLO

    @field_validator("name")
    @classmethod
    def _name(cls, value):
        if not NAME.match(value):
            raise ValueError("Class names use 1-24 lowercase letters, digits or hyphens")
        return value

    @field_validator("first_input", "turn_input")
    @classmethod
    def _inputs(cls, value, info):
        return _buckets(value, info.field_name.replace("_", " ").capitalize(), MIN_INPUT_TOKENS, MAX_INPUT_TOKENS)

    @field_validator("output")
    @classmethod
    def _output(cls, value):
        return _buckets(value, "Output", 1, MAX_OUTPUT_TOKENS)

    @field_validator("think_seconds")
    @classmethod
    def _think(cls, value):
        return _buckets(value, "Think seconds", 0, MAX_THINK_SECONDS, integer=False)

    @field_validator("turns")
    @classmethod
    def _turns(cls, value):
        return _buckets(value, "Turns", 1, MAX_TURNS)

    def mean(self, key):
        return _mean(getattr(self, key))

    def smallest_session(self):
        """Tokens the shortest possible first turn needs."""
        return (self.system_tokens + min(a for a, _, _ in self.first_input) + min(a for a, _, _ in self.output)
                + 3 * MESSAGE_FRAMING)


class UserModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    classes: list[UserClass] = Field(min_length=1, max_length=MAX_CLASSES)

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

    def assign(self, users):
        """Class index of each of ``users`` users: exact shares by largest remainder, interleaved."""
        shares = self.shares()
        exact = [users * s for s in shares]
        counts = [math.floor(x) for x in exact]
        for k in sorted(range(len(shares)), key=lambda k: (counts[k] - exact[k], k))[:users - sum(counts)]:
            counts[k] += 1
        # Interleave so that every prefix of the user list keeps the mix.
        slots = sorted((((i + 0.5) / counts[k], k) for k in range(len(counts)) for i in range(counts[k])))
        return [k for _, k in slots]


class UserSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    preset: str = Field("custom", pattern=r"^[a-z-]{1,20}$")
    model: UserModel
    # Session context caps in reference tokens; empty = 32k, 128k and the usable limit.
    context_caps: list[int] = Field(default_factory=list, max_length=MAX_CAPS)
    start_users: int = Field(8, ge=1, le=MAX_USERS)
    max_users: int = Field(256, ge=1, le=MAX_USERS)
    # Bisection stops when the bracket is within this share of the passing count.
    resolution: float = Field(0.1, ge=0.01, le=0.5)
    warmup_seconds: int = 60
    measure_seconds: int = 180
    request_timeout: int = Field(300, ge=30, le=1800)
    attainment: float = Field(0.95, ge=0.5, le=0.999)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    # After the SLO search, keep doubling users until the deployment hits a hard limit.
    saturation: bool = True
    saturation_max_users: int = Field(1024, ge=1, le=MAX_USERS)

    @field_validator("context_caps")
    @classmethod
    def _caps(cls, value):
        if any(isinstance(c, bool) or not MIN_CAP <= c <= MAX_CAP for c in value):
            raise ValueError(f"Context caps must be between {MIN_CAP:,} and {MAX_CAP:,} tokens")
        if any(b <= a for a, b in zip(value, value[1:])):
            raise ValueError("Context caps must be strictly increasing")
        return value

    @field_validator("warmup_seconds")
    @classmethod
    def _warmup(cls, value):
        if not MIN_WARMUP_SECONDS <= value <= MAX_PHASE_SECONDS:
            raise ValueError(f"Warmup must be between {MIN_WARMUP_SECONDS} and {MAX_PHASE_SECONDS:,} seconds")
        return value

    @field_validator("measure_seconds")
    @classmethod
    def _measure(cls, value):
        if not MIN_MEASURE_SECONDS <= value <= MAX_PHASE_SECONDS:
            raise ValueError(f"The measured window must be between {MIN_MEASURE_SECONDS} and "
                             f"{MAX_PHASE_SECONDS:,} seconds")
        return value

    @model_validator(mode="after")
    def _users(self):
        if self.start_users > self.max_users:
            raise ValueError("The starting user count cannot exceed the maximum")
        return self

    def comparison_key(self):
        """Settings that must match before two runs' results may be compared."""
        return {"model_hash": self.model.fingerprint(), "warmup_seconds": self.warmup_seconds,
                "measure_seconds": self.measure_seconds, "attainment": self.attainment, "seed": self.seed}


def _class(name, weight, system, first, turn, output, think, turns, ttft=2000, rate=40):
    return {"name": name, "weight": weight, "system_tokens": system, "first_input": first, "turn_input": turn,
            "output": output, "think_seconds": think, "turns": turns,
            "slo": {"ttft_ms": ttft, "output_tokens_per_sec": rate}}


# Estimates of typical internal use, to be replaced with session traces from
# gateway logs (custom JSON) when available.
CHAT = _class("chat", 40, 1500,
              [[100, 500, 50], [500, 2000, 35], [2000, 8000, 15]],
              [[48, 200, 60], [200, 1000, 30], [1000, 4000, 10]],
              [[100, 300, 35], [300, 700, 45], [700, 1500, 20]],
              [[5, 20, 30], [20, 60, 50], [60, 180, 20]],
              [[2, 6, 50], [6, 15, 35], [15, 30, 15]])
AGENT = _class("agent", 60, 8000,
               [[500, 2000, 60], [2000, 8000, 40]],
               [[200, 1000, 40], [1000, 4000, 40], [4000, 16000, 20]],
               [[30, 150, 60], [150, 500, 30], [500, 2000, 10]],
               [[0.5, 2, 50], [2, 8, 40], [8, 30, 10]],
               [[10, 30, 40], [30, 80, 40], [80, 200, 20]])
PRESETS = {
    "mixed": {"classes": [CHAT, AGENT]},
    "chat": {"classes": [{**CHAT, "weight": 100}]},
    "agent": {"classes": [{**AGENT, "weight": 100}]},
}
PRESET_LABELS = {"mixed": "Mixed · 40% chat, 60% agents", "chat": "Chat users", "agent": "Agents",
                 "custom": "Custom user model"}


def default_settings(preset="mixed"):
    return {"preset": preset, "model": PRESETS[preset], "context_caps": [], "start_users": 8, "max_users": 256,
            "resolution": 0.1, "warmup_seconds": 60, "measure_seconds": 180, "request_timeout": 300,
            "attainment": 0.95, "seed": 0, "saturation": True, "saturation_max_users": 1024}


def parse_settings(raw):
    """Validate a settings dict; presets supply the user model unless it is custom."""
    if not isinstance(raw, dict):
        raise ValueError("Users-stage settings must be an object")
    raw = dict(raw)
    preset = raw.get("preset", "custom")
    if preset in PRESETS and "model" not in raw:
        raw["model"] = PRESETS[preset]
    settings = UserSettings.model_validate(raw)
    if settings.preset in PRESETS and settings.model.model_dump() != UserModel.model_validate(PRESETS[settings.preset]).model_dump():
        settings = settings.model_copy(update={"preset": "custom"})
    if settings.preset not in PRESET_LABELS:
        raise ValueError("Unknown user-model preset")
    return settings


def caps_for(settings, usable):
    """Context caps to measure: the explicit list (bounded by ``usable``), else 32k, 128k and ``usable``."""
    if settings.context_caps:
        caps = [c for c in settings.context_caps if c <= usable]
        return tuple(caps) or (usable,)
    return tuple(sorted({*(c for c in (32_768, 131_072) if c < usable), usable}))


def check_fits(model, cap):
    too_big = [c.name for c in model.classes if c.smallest_session() > cap]
    if too_big:
        raise ValueError(f"The {cap:,}-token context cap cannot hold one turn of: {', '.join(too_big)}")


# --- Session plans -------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    input_tokens: int
    output_tokens: int
    think_seconds: float


@dataclass(frozen=True)
class SessionPlan:
    klass: int
    turns: tuple
    # Turn the user is at when the level starts (> 0 for a session already in progress).
    start: int = 0


def _draw(rng, buckets, integer=True):
    low, high, _ = rng.choices(buckets, weights=[w for _, _, w in buckets])[0]
    return rng.randint(low, high) if integer else rng.uniform(low, high)


def plan_session(model, klass, cap, rng):
    """Turns of one session, cut where the next turn would not fit ``cap``."""
    c = model.classes[klass]
    planned = _draw(rng, c.turns)
    context, turns = c.system_tokens + MESSAGE_FRAMING, []
    for t in range(planned):
        turn = Turn(_draw(rng, c.first_input if t == 0 else c.turn_input), _draw(rng, c.output),
                    round(_draw(rng, c.think_seconds, integer=False), 3))
        if context + turn.input_tokens + turn.output_tokens + 2 * MESSAGE_FRAMING > cap:
            break
        turns.append(turn)
        context += turn.input_tokens + turn.output_tokens + 2 * MESSAGE_FRAMING
    return SessionPlan(klass, tuple(turns))


def user_rng(seed, cap, users, user):
    return random.Random(f"sessions-v1:{seed}:{cap}:{users}:{user}")


def starting_plan(model, klass, cap, rng):
    """A session already in progress, as a random moment of steady traffic finds it.

    Longer sessions are proportionally more likely to be in progress
    (length-biased), and the current turn is uniform within the session."""
    longest = max(b for _, b, _ in model.classes[klass].turns)
    plan = None
    for _ in range(100):
        plan = plan_session(model, klass, cap, rng)
        if plan.turns and rng.random() < len(plan.turns) / longest:
            break
    if not plan.turns:
        raise ValueError(f"The {cap:,}-token cap cannot hold a session of {model.classes[klass].name}")
    return SessionPlan(klass, plan.turns, rng.randrange(len(plan.turns)))


# --- Prompts ---------------------------------------------------------------------------

class SessionPrompts:
    """Messages with exact cl100k_base reference lengths.

    The class system prompt is shared by every user of the class (like tool
    definitions); everything after a user's first turn header is unique to
    that session, so only its own history can be reused from the prefix cache."""

    footer = "\nIgnore the notes. Count upward from 1, separated by commas, until stopped."

    def __init__(self, model, nonce):
        import tiktoken

        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.nonce = nonce
        longest = max(max(b for _, b, _ in getattr(c, key)) for c in model.classes
                      for key in ("first_input", "turn_input", "output"))
        self.padding = " pad" * max(longest, max(c.system_tokens for c in model.classes), 1)
        self.footer_tokens = self.count(self.footer)
        self.systems = []
        for c in model.classes:
            header = f"Users {c.name} {nonce}. Shared instructions follow:\n"
            body = max(0, c.system_tokens - self.count(header))
            self.systems.append(header + self.padding[: 4 * body] if c.system_tokens else None)

    def count(self, text):
        return len(self.encoding.encode(text))

    def system(self, klass):
        text = self.systems[klass]
        return [{"role": "system", "content": text}] if text else []

    def user(self, tag, turn, tokens):
        header = f"Session {self.nonce}-{tag} turn {turn}. Notes follow:\n"
        body = max(0, tokens - self.count(header) - self.footer_tokens)
        return header + self.padding[: 4 * body] + self.footer

    def assistant(self, tokens):
        """Stand-in answer for history the level starts with."""
        return self.padding[1: 4 * tokens]

    def history(self, plan, tag):
        """System prompt plus the turns before ``plan.start``."""
        messages = self.system(plan.klass)
        for t, turn in enumerate(plan.turns[: plan.start]):
            messages.append({"role": "user", "content": self.user(tag, t, turn.input_tokens)})
            messages.append({"role": "assistant", "content": self.assistant(turn.output_tokens)})
        return messages
