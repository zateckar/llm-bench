"""Users at SLO: simulated multi-turn users against one deployment.

Each simulated user runs sessions whose history grows turn by turn (people
think between messages, agents run tools between steps). A level runs a fixed
number of users: a warmup brings every user's first, cold request through
the server, then a measured window judges every request against its class
SLO. A level passes when every class reaches the attainment target.

The user count is searched per context cap: doubling from the start count,
then bisection between the highest passing and the lowest failing count. A
level that is clearly failing stops early. Caps run in increasing order and
each cap uses the previous result as a search hint. Failed counts and
saturation limits are measured independently for every cap.

After the SLO search a saturation search keeps doubling users until a level is
beyond a hard limit (many failures, first tokens or output speed an order of
magnitude off target, or throughput that stopped growing while latency rose),
so each cap reports both users at SLO and where the deployment is exhausted.
Failing and exhausted levels are diagnosed (constraints.py): which resource
limits them and what would relieve it. See docs/design-session-capacity.md.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import math
import random
import threading
import time
import uuid

from app.benchmarking import constraints
from app.benchmarking.cache_metrics import cache_metrics
from app.benchmarking.llm_client import ChatClient, client_protocol
from app.benchmarking.models import LatencyStats, RequestMetrics
from app.benchmarking.session_workload import (
    SessionPrompts, caps_for, check_fits, plan_session, starting_plan, user_rng,
)

REVISION = "sessions-v2"
SCHEMA_VERSION = 7
KIND = "sessions"
LAG_LIMIT_MS = 100.0
CHECK_INTERVAL = 0.5
# Stop a level once a class with enough judged requests is far below target.
EARLY_MIN_SECONDS = 30
EARLY_SHARE = 1 / 3
EARLY_MIN_REQUESTS = 20
EARLY_ATTAINMENT = 0.5
# Fewer measured requests in a class than this are flagged as thin evidence.
THIN_REQUESTS = 20
SAMPLE_LIMIT = 1_000
# A level is beyond a hard limit when more than this share of requests fails or is incomplete,
# a class's first token p95 or median output speed is HARD_FACTOR off its target, or
# throughput grew less than PLATEAU_GAIN over a level with PLATEAU_RATIO fewer users while the
# median first token grew PLATEAU_TTFT-fold.
HARD_FAILURE_SHARE = 0.10
HARD_FACTOR = 10
PLATEAU_RATIO = 1.5
PLATEAU_GAIN = 1.10
PLATEAU_TTFT = 1.5
INCOMPLETE_FINISH = {"content_filter", "repetition"}
REASONS = {
    "error": "Request error",
    "incomplete": "Empty or filtered answer",
    "ttft": "First token over target",
    "ttft_unmeasured": "First token not measurable",
    "tpot": "Output speed under target",
    "tpot_unmeasured": "Output speed not measurable",
}
LIMITS = {
    "ttft": "First token over target",
    "prefill": "Prefill: first token too slow while the server had no queue",
    "queue": "Queueing: requests waited for a batch slot",
    "kv_cache": "KV cache full: session histories no longer fit",
    "tpot": "Output speed under target as the batch grows",
    "error": "Request errors",
    "incomplete": "Incomplete answers",
    "unmeasured": "Timings not measurable (no streaming or burst delivery)",
    "client": "The client could not keep up; capacity may be higher",
    "inherited": "Failed at a smaller context cap",
}


@dataclass
class Record:
    user: int
    klass: int
    sent: float
    lag_ms: float
    context_tokens: int
    output_tokens: int
    turn: int
    metrics: RequestMetrics
    first: bool = False
    # Leading context the prefix cache could hold from earlier turns (system prompt and history).
    reusable: int = 0
    finished: float | None = None

    @property
    def status(self):
        m = self.metrics
        if not m.ok:
            return "error"
        if m.completion_tokens <= 0 or m.finish_reason in INCOMPLETE_FINISH:
            return "incomplete"
        return "complete"

    @property
    def ttft(self):
        return None if self.metrics.ttft_ms is None else self.metrics.ttft_ms + self.lag_ms

    def misses(self, slo):
        status = self.status
        if status != "complete":
            return [status]
        reasons = []
        if self.ttft is None:
            reasons.append("ttft_unmeasured")
        elif self.ttft > slo.ttft_ms:
            reasons.append("ttft")
        tpot = self.metrics.output_token_time_ms
        if tpot is None:
            reasons.append("tpot_unmeasured")
        elif tpot > slo.tpot_ms:
            reasons.append("tpot")
        return reasons


def _stats(values):
    return LatencyStats.from_samples([v for v in values if v is not None]).to_dict()


def _plain(values):
    return {k.removesuffix("_ms"): v for k, v in _stats(values).items()}


def _class_summary(records, slo, users, target):
    complete = [r for r in records if r.status == "complete"]
    reasons = Counter(reason for r in records for reason in r.misses(slo))
    met = sum(not r.misses(slo) for r in records)
    speeds = [1000 / r.metrics.output_token_time_ms for r in complete
              if r.metrics.output_token_time_ms and r.metrics.output_token_time_ms > 0]
    return {
        "users": users,
        "requests": len(records),
        "completed": len(complete),
        "errors": reasons["error"],
        "incomplete": reasons["incomplete"],
        "slo_met": met,
        "attainment": met / len(records) if records else None,
        "meets_target": bool(records) and met / len(records) >= target,
        "thin": len(records) < THIN_REQUESTS,
        "failure_reasons": dict(reasons),
        "ttft": _stats(r.ttft for r in complete),
        "output_token_time": _stats(r.metrics.output_token_time_ms for r in complete),
        # The slow tail of output speed is the 5th percentile, i.e. 1000 / token-time p95.
        "output_speed": {"p50": LatencyStats.from_samples(speeds).p50,
                         "p05": (1000 / t) if (t := LatencyStats.from_samples(
                             [r.metrics.output_token_time_ms for r in complete
                              if r.metrics.output_token_time_ms]).p95) else None},
        "context_tokens": _plain(r.context_tokens for r in records),
        "prompt_tokens": _plain(r.metrics.prompt_tokens for r in complete),
        "output_tokens": _plain(r.metrics.completion_tokens for r in complete),
        "cache_metrics": cache_metrics([r.metrics for r in records], None),
        "slo": {"ttft_ms": slo.ttft_ms, "output_tokens_per_sec": slo.output_tokens_per_sec},
    }


def _limiting(level, model):
    """Why a failed level failed, from the misses of the classes below target."""
    if level["passed"] or level["cancelled"]:
        return None
    if level["client_limited"]:
        return {"factor": "client", "label": LIMITS["client"],
                "detail": f"Dispatch lag p99 {level['dispatch_lag'].get('p99_ms') or 0:,.0f} ms"}
    counts, parts = Counter(), []
    for c in model.classes:
        item = level["classes"].get(c.name) or {}
        if not item.get("requests") or item.get("meets_target"):
            continue
        reasons = Counter({k: v for k, v in item["failure_reasons"].items()})
        counts.update(reasons)
        main = reasons.most_common(1)[0][0] if reasons else None
        text = f"{c.name}: {item['attainment']:.0%} met the SLO"
        if main in ("ttft", "ttft_unmeasured"):
            text += f"; first token p95 {(item['ttft'].get('p95_ms') or 0) / 1000:,.1f} s vs {c.slo.ttft_ms / 1000:g} s"
        elif main in ("tpot", "tpot_unmeasured"):
            speed = item["output_speed"]["p05"]
            text += (f"; slowest 5% at {speed:,.1f} tok/s" if speed else "") + f" vs {c.slo.output_tokens_per_sec} tok/s"
        elif main:
            text += f"; {item['errors']} errors, {item['incomplete']} incomplete"
        parts.append(text)
    if not counts:
        return {"factor": "error", "label": "No measured request", "detail": level.get("early_stop") or ""}
    main = counts.most_common(1)[0][0]
    factor = {"ttft_unmeasured": "unmeasured", "tpot_unmeasured": "unmeasured"}.get(main, main)
    return {"factor": factor, "label": LIMITS[factor], "detail": " · ".join(parts)}


def analyse_level(model, settings, cap, users, records, *, window, started_at, ended_at, cancelled=False,
                  early_stop="", warmup_complete=True, sample_limit=SAMPLE_LIMIT, phase="slo", working_set=None):
    start, end = window
    measured = [r for r in records if not r.first and start <= r.sent < end]
    assignment = Counter(model.assign(users))
    classes = {}
    for k, c in enumerate(model.classes):
        classes[c.name] = _class_summary([r for r in measured if r.klass == k], c.slo, assignment[k],
                                         settings.attainment)
    lag = LatencyStats.from_samples([r.lag_ms for r in measured])
    client_limited = lag.p99 is not None and lag.p99 > LAG_LIMIT_MS
    judged = [v for v in classes.values() if v["users"]]
    met = sum(v["slo_met"] for v in judged)
    total = sum(v["requests"] for v in judged)
    passed = (not cancelled and not early_stop and warmup_complete
              and total > 0 and not client_limited
              and all(v["meets_target"] for v in judged))
    seconds = max(end - start, 1e-9)
    rate_end = max([end, *(r.finished if r.finished is not None else
                          r.sent + r.metrics.latency_ms / 1000 for r in measured)])
    rate_seconds = max(rate_end - start, 1e-9)
    good = [r.metrics for r in measured if r.metrics.ok]
    complete = [r for r in measured if r.status == "complete"]
    context = sum(r.context_tokens for r in complete)
    level = {
        "users": users,
        "context_cap": cap,
        "phase": phase,
        "started_at": started_at,
        "ended_at": ended_at,
        # Wall-clock measured window, for server telemetry.
        "window_started_at": started_at + start,
        "window_ended_at": started_at + end,
        "warmup_seconds": round(start, 3),
        "warmup_complete": warmup_complete,
        "measured_seconds": round(seconds, 3),
        "rate_seconds": rate_seconds,
        "drain_seconds": max(0, rate_end - end),
        "requests": total,
        "all_requests": len(records),
        "completed": sum(v["completed"] for v in judged),
        "errors": sum(v["errors"] for v in judged),
        "slo_met": met,
        "attainment": met / total if total else None,
        "passed": passed,
        "cancelled": cancelled,
        "early_stop": early_stop,
        "client_limited": client_limited,
        "dispatch_lag": lag.to_dict(),
        "sessions_started": sum(r.turn == 0 for r in measured),
        "context_tokens": _plain(r.context_tokens for r in measured),
        "ttft": _stats(r.ttft for r in complete),
        # Share of prompt tokens a prefix cache that kept every history could serve.
        "expected_prefix_reuse": sum(r.reusable for r in complete) / context if context else None,
        "working_set_tokens": working_set,
        "output_tokens_per_sec": sum(m.completion_tokens for m in good) / rate_seconds,
        "input_tokens_per_sec": sum(m.prompt_tokens for m in good) / rate_seconds,
        "requests_per_sec": len(measured) / rate_seconds,
        "cache_metrics": cache_metrics([r.metrics for r in measured], rate_seconds * 1000),
        "classes": classes,
        "error_examples": list(dict.fromkeys(r.metrics.error for r in measured
                                             if not r.metrics.ok and r.metrics.error))[:3],
        "samples": _samples(measured, sample_limit, f"{settings.seed}:{cap}:{users}"),
        "server": None,
        "hard_limit": "",
        "constraint": None,
    }
    level["limiting"] = _limiting(level, model)
    return level


def _samples(records, limit, seed):
    picked = records if len(records) <= limit else sorted(
        random.Random(f"samples:{seed}").sample(records, limit), key=lambda r: r.sent)
    codes = {"complete": 0, "incomplete": 1, "error": 2}
    columns = {key: [] for key in ("t", "c", "u", "turn", "ctx", "s", "ttft", "tpot", "out", "pin", "cached")}
    for r in picked:
        m = r.metrics
        for key, value in (("t", round(r.sent, 3)), ("c", r.klass), ("u", r.user), ("turn", r.turn),
                           ("ctx", r.context_tokens), ("s", codes[r.status]),
                           ("ttft", None if r.ttft is None else round(r.ttft, 1)),
                           ("tpot", None if m.output_token_time_ms is None else round(m.output_token_time_ms, 2)),
                           ("out", m.completion_tokens), ("pin", m.prompt_tokens), ("cached", m.cached_tokens)):
            columns[key].append(value)
    return columns


# --- Search ----------------------------------------------------------------------------

def next_users(passed, failed, *, start, maximum, resolution):
    """The next user count to measure, or None when the bracket is resolved.

    ``failed`` may include an inherited failure from a smaller context cap."""
    lo = max(passed, default=0)
    hi = min((f for f in failed if f > lo), default=None)
    if not passed and not failed:
        return min(start, maximum)
    if hi is None:
        return None if lo >= maximum else min(lo * 2, maximum)
    if lo == 0:
        return hi // 2 or None
    if hi - lo <= max(1, math.ceil(lo * resolution)):
        return None
    return (lo + hi) // 2


def cap_result(entry):
    """Users at SLO for one context cap from its measured levels and inherited bounds."""
    levels = [lv for lv in entry["levels"] if not lv["cancelled"]]
    conclusive = [lv for lv in levels if lv["passed"] or not lv["client_limited"]]
    passed = [lv["users"] for lv in conclusive if lv["passed"]]
    failed = [lv["users"] for lv in conclusive if not lv["passed"]]
    inherited = entry.get("inherited_failure")
    users = max(passed, default=0)
    measured_hi = min((f for f in failed if f > users), default=None)
    hi = min((f for f in (measured_hi, inherited) if f is not None and f > users), default=None)
    if entry.get("inferred_not_met"):
        return {"users": 0, "status": "not_met", "first_failed": inherited, "inherited": True,
                "limiting": {"factor": "inherited", "label": LIMITS["inherited"],
                             "detail": "No user count met the SLO at a smaller cap"}}
    if not levels:
        return {"users": None, "status": "not_measured", "first_failed": None, "inherited": False, "limiting": None}
    if not conclusive:
        return {"users": None, "status": "inconclusive", "first_failed": None, "inherited": False,
                "limiting": levels[-1]["limiting"], "client_limited": True, "at_result": None}
    status = "not_met" if not passed else "established" if hi is not None else "lower_bound"
    from_prior = hi is not None and hi != measured_hi
    limiting, constraint = None, None
    if hi is not None and not from_prior:
        first = next((lv for lv in conclusive if lv["users"] == hi and not lv["passed"]), None)
        limiting, constraint = first["limiting"], first.get("constraint")
    elif from_prior:
        limiting = {"factor": "inherited", "label": LIMITS["inherited"],
                    "detail": f"{hi} users failed at a smaller context cap"}
    best = next((lv for lv in conclusive if lv["users"] == users and lv["passed"]), None)
    return {"users": users, "status": status, "first_failed": hi, "inherited": from_prior,
            "limiting": limiting, "constraint": constraint,
            "client_limited": any(lv["client_limited"] and not lv["passed"] for lv in levels),
            "at_result": _at(best)}


def _at(level):
    if not level:
        return None
    return {"attainment": level["attainment"], "requests": level["requests"],
            "context_p50": level["context_tokens"].get("p50"), "context_p95": level["context_tokens"].get("p95"),
            "output_tokens_per_sec": level["output_tokens_per_sec"],
            "classes": {name: {"attainment": c["attainment"], "ttft_p95_ms": c["ttft"].get("p95_ms"),
                               "output_speed_p05": c["output_speed"]["p05"], "requests": c["requests"]}
                        for name, c in level["classes"].items()}}


def summarise(caps):
    results = [{"context_cap": c["context_cap"], **c["result"], "saturation": c.get("saturation")} for c in caps]
    measured = [r for r in results if r["status"] != "not_measured"]
    return {"results": results, "headline": measured[-1] if measured else None}


# --- Saturation ----------------------------------------------------------------------------

def hard_limit(level, previous=None):
    """Why a level is beyond what the deployment can sustain, or "" when it is not.

    ``previous`` is a measured level with at most 1/PLATEAU_RATIO of the users,
    for the throughput plateau rule."""
    if level["cancelled"]:
        return ""
    if not level.get("warmup_complete", True):
        return "Not every user received a first answer within the request timeout"
    requests = level["requests"]
    if not requests:
        return "No request was measured"
    classes = level["classes"]
    failed = sum(c["errors"] + c["incomplete"] for c in classes.values())
    if failed / requests > HARD_FAILURE_SHARE:
        return f"{failed / requests:.0%} of {requests} measured requests failed or were incomplete"
    for name, c in classes.items():
        if not c["requests"]:
            continue
        slo = c["slo"]
        p95 = c["ttft"].get("p95_ms")
        if p95 and p95 > HARD_FACTOR * slo["ttft_ms"]:
            return (f"{name}: first token p95 {p95 / 1000:,.1f} s, over {HARD_FACTOR}× the "
                    f"{slo['ttft_ms'] / 1000:g} s target")
        speed = c["output_speed"]["p50"]
        if speed is not None and speed < slo["output_tokens_per_sec"] / HARD_FACTOR:
            return (f"{name}: median output speed {speed:,.1f} tok/s, under a {HARD_FACTOR}th of the "
                    f"{slo['output_tokens_per_sec']} tok/s target")
    if previous and previous["output_tokens_per_sec"]:
        gain = level["output_tokens_per_sec"] / previous["output_tokens_per_sec"]
        now, before = (level.get("ttft") or {}).get("p50_ms"), (previous.get("ttft") or {}).get("p50_ms")
        if gain < PLATEAU_GAIN and now and before and now >= PLATEAU_TTFT * before:
            return (f"Throughput stopped growing: {level['output_tokens_per_sec']:,.0f} tok/s with {level['users']} users "
                    f"vs {previous['output_tokens_per_sec']:,.0f} with {previous['users']}, while the median first "
                    f"token rose from {before / 1000:,.2f} s to {now / 1000:,.2f} s")
    if level.get("phase") == "saturation" and level.get("early_stop"):
        return level["early_stop"]
    return ""


def _measured(entry):
    return sorted((lv for lv in entry["levels"] if not lv["cancelled"]), key=lambda lv: lv["users"])


def judge_limits(entry):
    levels = _measured(entry)
    for level in levels:
        previous = max((lv for lv in levels if lv["users"] * PLATEAU_RATIO <= level["users"]),
                       key=lambda lv: lv["users"], default=None)
        level["hard_limit"] = hard_limit(level, previous)


def next_saturation(entry, maximum):
    """The next user count of the saturation search, or None when it is done."""
    levels = _measured(entry)
    if not levels or any(lv.get("hard_limit") for lv in levels):
        return None
    if any(lv["client_limited"] and not lv["passed"] for lv in levels):
        return None
    top = levels[-1]["users"]
    if top >= maximum:
        return None
    users = min(top * 2, maximum)
    inherited = entry.get("inherited_saturation")
    if inherited and users >= inherited:
        return None
    return users


def saturation_result(entry):
    """Where the deployment is exhausted at one context cap."""
    if not entry.get("saturation_search"):
        return None
    levels = _measured(entry)
    if not levels:
        return {"status": "not_measured", "users": None, "below": None, "reason": None, "constraint": None,
                "peak_output_tokens_per_sec": None, "peak_users": None}
    peak = max(levels, key=lambda lv: lv["output_tokens_per_sec"])
    result = {"peak_output_tokens_per_sec": peak["output_tokens_per_sec"], "peak_users": peak["users"],
              "reason": None, "constraint": None, "below": None}
    reached = next((lv for lv in levels if lv.get("hard_limit")), None)
    if reached:
        below = [lv["users"] for lv in levels if lv["users"] < reached["users"]]
        return {**result, "status": "reached", "users": reached["users"], "below": max(below, default=None),
                "reason": reached["hard_limit"], "constraint": reached.get("constraint")}
    top = levels[-1]["users"]
    if any(lv["client_limited"] and not lv["passed"] for lv in levels):
        return {**result, "status": "client_limited", "users": top}
    inherited = entry.get("inherited_saturation")
    if inherited and top * 2 >= inherited:
        return {**result, "status": "inherited", "users": inherited, "below": top,
                "reason": f"Exhausted with {inherited} users at a smaller context cap"}
    return {**result, "status": "not_reached", "users": top}


# --- Report ------------------------------------------------------------------------------

def refresh_entry(entry, telemetry=None):
    """Judge hard limits, diagnose failing and exhausted levels, and recompute the cap results."""
    judge_limits(entry)
    config = (telemetry or {}).get("cache_config")
    for level in entry["levels"]:
        signals = constraints.window_signals(telemetry, level.get("window_started_at"),
                                             level.get("window_ended_at")) if telemetry else None
        level["server"] = signals
        level["constraint"] = constraints.diagnose(level, signals, config) if constraints.needs_diagnosis(level) else None
    entry["result"] = cap_result(entry)
    entry["saturation"] = saturation_result(entry)


@dataclass
class SessionReport:
    model: str
    protocol: dict
    caps: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    cancelled: bool = False
    stop_reason: str = ""
    telemetry: dict | None = None

    def attach_telemetry(self, data):
        self.telemetry = data
        for entry in self.caps:
            refresh_entry(entry, data)

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "kind": KIND, "model": self.model, "protocol": self.protocol,
                "caps": self.caps, "summary": summarise(self.caps), "notes": self.notes,
                "cancelled": self.cancelled, "stop_reason": self.stop_reason, "telemetry": self.telemetry}


def protocol(settings, client_config, caps):
    model = settings.model
    return {
        "revision": REVISION,
        "load_model": "closed_loop_users",
        "preset": settings.preset,
        "user_model": model.model_dump(),
        "user_model_hash": model.fingerprint(),
        "context_caps": list(caps),
        "start_users": settings.start_users,
        "max_users": settings.max_users,
        "resolution": settings.resolution,
        "warmup_seconds": settings.warmup_seconds,
        "measure_seconds": settings.measure_seconds,
        "request_timeout": settings.request_timeout,
        "attainment_target": settings.attainment,
        "seed": settings.seed,
        "comparison_key": settings.comparison_key(),
        "temperature": client_config.temperature,
        "reasoning_effort": client_config.reasoning_effort,
        "model_seed": client_config.seed,
        "reference_tokenizer": "cl100k_base",
        "attempts_per_request": 1,
        "lag_limit_ms": LAG_LIMIT_MS,
        "early_stop": {"min_seconds": EARLY_MIN_SECONDS, "share": EARLY_SHARE,
                       "min_requests": EARLY_MIN_REQUESTS, "attainment": EARLY_ATTAINMENT},
        "saturation": settings.saturation,
        "saturation_max_users": settings.saturation_max_users,
        "hard_limit": {"failure_share": HARD_FAILURE_SHARE, "factor": HARD_FACTOR, "plateau_ratio": PLATEAU_RATIO,
                       "plateau_gain": PLATEAU_GAIN, "plateau_ttft": PLATEAU_TTFT},
        "diagnosis_revision": constraints.REVISION,
        "user_definition": "A simulated user runs one session after another. Each turn sends the whole history plus a new message, waits for the full answer, then thinks (people) or runs tools (agents) before the next turn. A session ends after its planned turns or when the next turn would exceed the context cap.",
        "steady_state_definition": "Users start in a session already in progress (length-biased) with its history, at random times within the first half of the warmup. Each user's first request is a cold prefill and is excluded; the measured window starts when the warmup has passed and every user has had a first answer.",
        "slo_definition": "A request meets its class SLO when it completes, dispatch lag + first token <= ttft_ms and its output token time <= 1000 / output_tokens_per_sec. Unmeasurable timings are misses.",
        "rate_definition": "Tokens and responses of requests dispatched in the measurement window, divided by the time from window start through the last such response (drain included). This is cohort throughput, not an in-window engine token counter.",
        "pass_definition": "Every class with users reaches the attainment target over the requests sent in the measured window, and dispatch lag p99 <= lag_limit_ms.",
        "search_definition": "Per context cap: double from start_users while levels pass, then bisect between the highest passing and the lowest failing count until the gap is within resolution of the passing count. Caps run in increasing order; the previous result is only a starting hint. Every cap measures its own failing count. A level stops early when a class with enough judged requests is far below the target.",
        "saturation_definition": "After the SLO search, users keep doubling from the largest measured count until a level is beyond a hard limit or saturation_max_users is reached. A level is beyond a hard limit when more than failure_share of its requests fail or are incomplete, a class's first token p95 exceeds factor x its target or its median output speed is under 1/factor of its target, not every user got a first answer within the request timeout, or throughput grew less than plateau_gain over a level with plateau_ratio fewer users while the median first token grew plateau_ttft-fold. Saturation levels stop early on the failure and first-token rules instead of the SLO rule. Every cap measures its own saturation limit; cache and batching effects need not be monotonic.",
        "constraint_definition": "Failing and exhausted levels name one constraint from what the client saw and, with B300 telemetry over the measured window, from vLLM queue, batch, KV cache, preemption and prefix-cache signals and DCGM GPU activity; see constraints-v1 in docs/design-session-capacity.md.",
        "context_definition": "System prompt, history and the new message in cl100k_base reference tokens, plus a per-message framing allowance; provider counts are stored separately.",
        "output_definition": "Each message asks the model to count until stopped; max_tokens is the planned output length. Counts include reasoning tokens. Answers become history.",
    }


def run_session_test(client_config, settings, usable_context, *, progress=None, cancelled=None, on_level=None,
                     client_factory=None):
    """Measure users at SLO for every context cap; returns a SessionReport."""
    cancelled = cancelled or (lambda: False)
    client_factory = client_factory or ChatClient
    caps = caps_for(settings, usable_context)
    for cap in caps:
        check_fits(settings.model, cap)
    base = replace(client_config, detect_repetition=False, stream=True, max_retries=1,
                   retry_transport_errors=False, stream_deadline=float(settings.request_timeout),
                   timeout=min(client_config.timeout, float(settings.request_timeout)))
    report = SessionReport(client_config.model, protocol(settings, client_config, caps))
    prompts = SessionPrompts(settings.model, uuid.uuid4().hex[:12])
    clients = []
    last_check = [0.0, False]

    def is_cancelled():
        now = time.perf_counter()
        if now - last_check[0] >= CHECK_INTERVAL:
            last_check[0], last_check[1] = now, bool(cancelled())
        return last_check[1]

    try:
        warm = client_factory(replace(base, max_retries=3))
        clients.append(warm)
        ok = False
        for i in range(2):
            if cancelled():
                report.cancelled = True
                return report
            _, _, metrics = warm.complete(f"Warmup {prompts.nonce}-{i}. Reply with OK.", max_tokens=16, retries=3)
            ok |= metrics.ok
        if not ok:
            report.stop_reason = "Endpoint did not answer warmup requests; measurement stopped."
            return report
        if not warm.streaming_supported:
            report.stop_reason = "Endpoint does not stream; first-token SLOs cannot be measured."
            return report
        measured_config = replace(warm.config, stream=True, request_stream_usage=warm._stream_usage_supported,
                                  max_retries=1)
        report.protocol["client"] = client_protocol(measured_config)
        report.protocol["streaming"] = True
        user_clients = []
        done, estimate = 0, len(caps) * (9 if settings.saturation else 7)
        prior_users = None

        def measure(entry, cap, users, phase):
            nonlocal done
            while len(user_clients) < users:
                client = client_factory(measured_config)
                user_clients.append(client)
                clients.append(client)
            if progress:
                label = " · saturation search" if phase == "saturation" else ""
                progress(f"{cap:,}-token cap · {users} users{label}", done, max(estimate, done + 1))
            level = _run_level(settings, prompts, user_clients, cap, users,
                               sum(len(c["levels"]) for c in report.caps), is_cancelled, phase=phase)
            done += 1
            entry["levels"].append(level)
            refresh_entry(entry)
            if on_level:
                on_level(report)
            if level["cancelled"] or is_cancelled():
                report.cancelled = True
            return level

        for cap in caps:
            entry = {"context_cap": cap, "levels": [], "inherited_failure": None,
                     "prior_users": prior_users, "inferred_not_met": False,
                     "saturation_search": settings.saturation,
                     "inherited_saturation": None}
            report.caps.append(entry)
            refresh_entry(entry)
            while not report.cancelled and not report.stop_reason and not entry["inferred_not_met"]:
                levels = [lv for lv in entry["levels"] if not lv["cancelled"]]
                if any(lv["client_limited"] and not lv["passed"] for lv in levels):
                    break
                passed = [lv["users"] for lv in levels if lv["passed"]]
                failed = [lv["users"] for lv in levels if not lv["passed"]]
                if not levels and prior_users:
                    users = prior_users
                else:
                    users = next_users(passed, failed, start=settings.start_users,
                                       maximum=settings.max_users, resolution=settings.resolution)
                if users is None or users in passed + failed:
                    break
                level = measure(entry, cap, users, "slo")
                if not report.cancelled and not level["all_completed"]:
                    report.stop_reason = (f"No request completed with {users} users at the {cap:,}-token cap; "
                                          "the endpoint may be down. Larger counts and caps were not attempted.")
            while entry["saturation_search"] and not report.cancelled and not report.stop_reason:
                users = next_saturation(entry, settings.saturation_max_users)
                if users is None:
                    break
                # No completed request here is the limit itself, not a down endpoint.
                measure(entry, cap, users, "saturation")
            result = entry["result"]
            if report.cancelled or report.stop_reason:
                break
            prior_users = result["users"]
        if progress:
            progress("Users stage finished", done, done)
        if any(entry["result"].get("client_limited") for entry in report.caps):
            report.notes.append(f"Some levels were client-limited (dispatch lag p99 over {LAG_LIMIT_MS:g} ms); "
                                "they do not bound capacity.")
        return report
    finally:
        for client in clients:
            client.session.close()


def _run_level(settings, prompts, clients, cap, users, index, cancelled, phase="slo"):
    model = settings.model
    assignment = model.assign(users)
    # History each user holds now; class system prompts are shared, so count each once.
    contexts = [0] * users
    shared = (sum(model.classes[k].system_tokens for k in assignment)
              - sum(model.classes[k].system_tokens for k in set(assignment)))
    records, lock = [], threading.Lock()
    first_answers = [0]
    any_completed = [False]
    failures = []
    stop = threading.Event()
    started_at = time.time()
    t0 = time.perf_counter()
    stagger = settings.warmup_seconds / 2

    def user_loop(u):
        klass = assignment[u]
        rng = user_rng(settings.seed, cap, users, u)
        plan = starting_plan(model, klass, cap, rng)
        session = 0
        tag = f"{index}-{u}-{session}"
        messages = prompts.history(plan, tag)
        system = model.classes[klass].system_tokens
        context = system + sum(t.input_tokens + t.output_tokens for t in plan.turns[: plan.start])
        turn = plan.start
        contexts[u] = context
        due = t0 + rng.uniform(0, stagger)
        first = True
        while True:
            while not stop.is_set() and (wait := due - time.perf_counter()) > 0:
                stop.wait(min(wait, CHECK_INTERVAL))
            if stop.is_set():
                return
            step = plan.turns[turn]
            content = prompts.user(tag, turn, step.input_tokens)
            request = [*messages, {"role": "user", "content": content}]
            sent = time.perf_counter()
            try:
                text, _, metrics = clients[u].complete_messages(request, max_tokens=step.output_tokens, retries=1)
            except Exception as error:  # noqa: BLE001 - recorded as a failed request
                text, metrics = "", RequestMetrics(ok=False, error=str(error))
            record = Record(u, klass, sent - t0, (sent - due) * 1000, context + step.input_tokens,
                            step.output_tokens, turn, metrics, first, context,
                            finished=time.perf_counter() - t0)
            with lock:
                records.append(record)
                if first:
                    first_answers[0] += 1
                if record.status == "complete":
                    any_completed[0] = True
            first = False
            if metrics.ok:
                messages = [*request, {"role": "assistant", "content": text}]
                context += step.input_tokens + step.output_tokens
            turn += 1
            if turn >= len(plan.turns):
                session += 1
                tag = f"{index}-{u}-{session}"
                for _ in range(20):
                    plan = plan_session(model, klass, cap, rng)
                    if plan.turns:
                        break
                messages, context, turn = prompts.system(klass), system, 0
            contexts[u] = context
            due = time.perf_counter() + step.think_seconds

    def guarded(u):
        try:
            user_loop(u)
        except Exception as error:  # noqa: BLE001 - surfaced on the level, never silently lost
            with lock:
                failures.append(f"User {u}: {type(error).__name__}: {error}")

    level_cancelled, early, warmup_complete = False, "", True
    with ThreadPoolExecutor(max_workers=users) as pool:
        for u in range(users):
            pool.submit(guarded, u)
        try:
            warm_until = t0 + settings.warmup_seconds
            give_up = warm_until + settings.request_timeout
            while True:
                now = time.perf_counter()
                if cancelled():
                    level_cancelled = True
                    break
                if now >= warm_until and first_answers[0] + len(failures) >= users:
                    break
                if now >= give_up:
                    warmup_complete = False
                    break
                time.sleep(0.2)
            window_start = time.perf_counter()
            next_check = window_start + max(EARLY_MIN_SECONDS, settings.measure_seconds * EARLY_SHARE)
            working = []
            while not level_cancelled:
                now = time.perf_counter()
                if now - window_start >= settings.measure_seconds:
                    break
                if cancelled():
                    level_cancelled = True
                    break
                working.append(sum(contexts) - shared)
                if now >= next_check:
                    next_check = now + 5
                    with lock:
                        snapshot = list(records)
                    if phase == "saturation":
                        early = _early_limit(model, snapshot, window_start - t0)
                    else:
                        early = _early_failure(model, snapshot, window_start - t0, settings.attainment)
                    if early:
                        break
                time.sleep(0.2)
            window_end = time.perf_counter()
        finally:
            stop.set()
    level = analyse_level(model, settings, cap, users, records, window=(window_start - t0, window_end - t0),
                          started_at=started_at, ended_at=time.time(), cancelled=level_cancelled,
                          early_stop=early, warmup_complete=warmup_complete, phase=phase,
                          working_set={"mean": sum(working) / len(working), "max": max(working)} if working else None)
    level["all_completed"] = any_completed[0]
    level["user_failures"] = failures[:3]
    return level


def _early_failure(model, records, window_start, target):
    """A reason to stop the level now, when a class is far below the attainment target."""
    for k, c in enumerate(model.classes):
        judged = [r for r in records if r.klass == k and not r.first and r.sent >= window_start]
        if len(judged) >= EARLY_MIN_REQUESTS:
            share = sum(not r.misses(c.slo) for r in judged) / len(judged)
            if share < min(EARLY_ATTAINMENT, target):
                return (f"Stopped early: {c.name} met the SLO on {share:.0%} of {len(judged)} measured requests, "
                        f"far below the {target:.0%} target")
    return ""


def _early_limit(model, records, window_start):
    """A reason to stop a saturation level now, when it is clearly beyond a hard limit."""
    judged = [r for r in records if not r.first and r.sent >= window_start]
    if len(judged) >= EARLY_MIN_REQUESTS:
        failed = sum(r.status != "complete" for r in judged)
        if failed / len(judged) > HARD_FAILURE_SHARE:
            return f"Stopped early: {failed / len(judged):.0%} of {len(judged)} measured requests failed or were incomplete"
    for k, c in enumerate(model.classes):
        ttfts = [r.ttft for r in judged if r.klass == k and r.ttft is not None]
        if len(ttfts) >= EARLY_MIN_REQUESTS:
            p95 = LatencyStats.from_samples(ttfts).p95
            if p95 > HARD_FACTOR * c.slo.ttft_ms:
                return (f"Stopped early: {c.name} first token p95 {p95 / 1000:,.1f} s, over {HARD_FACTOR}× the "
                        f"{c.slo.ttft_ms / 1000:g} s target")
    return ""
