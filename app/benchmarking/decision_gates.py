"""Requirement gates of decision profiles (``decision-gates-v1``).

A gate checks one piece of a model's scorecard evidence against a threshold.
See docs/design-decision-dashboard.md. Everything here is pure: the evidence
is assembled by ``app.services.scorecard``.
"""

from __future__ import annotations

import math
import re

REVISION = "decision-gates-v1"
MAX_GATES = 20
TYPES = ("quality", "capacity", "users", "latency", "context", "operations")
QUALITY_METRICS = {"score": "achievement", "full_pass": "full pass"}
LATENCY_METRICS = {
    # key: (label, unit, lower_is_better)
    "ttft_p95": ("TTFT p95", "ms", True),
    "latency_p95": ("whole answer p95", "ms", True),
    "request_output": ("per-request output p50", "tok/s", False),
}
LIMITS = {"capacity": (0.001, 10_000.0), "context": (1, 100_000_000), "concurrency": (1, 4096),
          "ms": (1, 3_600_000), "tok/s": (0.001, 1_000_000), "users": (1, 1_000_000)}
CUSTOM_WORKLOAD = re.compile(r"custom:[0-9a-f]{16}")
STATUSES = ("pass", "fail", "missing", "stale")


class GateError(ValueError):
    pass


def _number(raw, low, high, what):
    if isinstance(raw, bool):
        raise GateError(f"{what} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise GateError(f"{what} must be a number") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise GateError(f"{what} must be between {low:g} and {high:,g}")
    return value


def _whole(raw, kind, what):
    value = _number(raw, *LIMITS[kind], what)
    if value != int(value):
        raise GateError(f"{what} must be a whole number")
    return int(value)


def validate(raw_gates, *, suite_exists, presets, user_presets=()):
    """Normalised gates or ``GateError``. ``suite_exists(key)`` checks quality suites;
    ``presets`` and ``user_presets`` name capacity workloads and users-stage user models."""
    if not isinstance(raw_gates, list) or not raw_gates:
        raise GateError("A profile needs at least one gate")
    if len(raw_gates) > MAX_GATES:
        raise GateError(f"A profile can have at most {MAX_GATES} gates")
    gates = []
    for index, raw in enumerate(raw_gates, 1):
        where = f"Gate {index}"
        if not isinstance(raw, dict) or raw.get("type") not in TYPES:
            raise GateError(f"{where}: unknown gate type")
        kind = raw["type"]
        if kind == "quality":
            suite = raw.get("suite")
            if not isinstance(suite, str) or not suite_exists(suite):
                raise GateError(f"{where}: unknown quality suite")
            metric = raw.get("metric", "score")
            if metric not in QUALITY_METRICS:
                raise GateError(f"{where}: unknown quality metric")
            category = raw.get("category") or None
            if category is not None and (not isinstance(category, str) or len(category) > 120):
                raise GateError(f"{where}: invalid category")
            gates.append({"type": kind, "suite": suite, "metric": metric, "category": category,
                          "threshold": _number(raw.get("threshold"), 0.001, 1, f"{where}: threshold")})
        elif kind == "capacity":
            workload = raw.get("workload")
            if not isinstance(workload, str) or not (workload in presets or CUSTOM_WORKLOAD.fullmatch(workload)):
                raise GateError(f"{where}: choose a workload preset or custom:<workload hash>")
            gates.append({"type": kind, "workload": workload,
                          "threshold": _number(raw.get("threshold"), *LIMITS["capacity"], f"{where}: rate")})
        elif kind == "users":
            user_model = raw.get("user_model")
            if not isinstance(user_model, str) or not (user_model in user_presets
                                                       or CUSTOM_WORKLOAD.fullmatch(user_model)):
                raise GateError(f"{where}: choose a user-model preset or custom:<user model hash>")
            gates.append({"type": kind, "user_model": user_model,
                          "context": _whole(raw.get("context"), "context", f"{where}: context"),
                          "threshold": _whole(raw.get("threshold"), "users", f"{where}: users")})
        elif kind == "latency":
            metric = raw.get("metric")
            if metric not in LATENCY_METRICS:
                raise GateError(f"{where}: unknown latency metric")
            concurrency = _number(raw.get("concurrency", 1), *LIMITS["concurrency"], f"{where}: concurrency")
            if concurrency != int(concurrency):
                raise GateError(f"{where}: concurrency must be a whole number")
            unit = LATENCY_METRICS[metric][1]
            gates.append({"type": kind, "metric": metric, "concurrency": int(concurrency),
                          "threshold": _number(raw.get("threshold"), *LIMITS[unit], f"{where}: threshold")})
        elif kind == "context":
            value = _number(raw.get("threshold"), *LIMITS["context"], f"{where}: context")
            gates.append({"type": kind, "threshold": int(value)})
        else:
            gates.append({"type": kind})
    return gates


def describe(gate, suite_labels=None, workload_labels=None, user_labels=None):
    """Readable requirement, e.g. ``Rigorous suite achievement ≥ 70%``."""
    kind = gate["type"]
    if kind == "users":
        label = (user_labels or {}).get(gate["user_model"], gate["user_model"])
        return f"Users at SLO on {label} with sessions up to {gate['context']:,} tokens ≥ {gate['threshold']:,}"
    if kind == "quality":
        suite = (suite_labels or {}).get(gate["suite"], gate["suite"])
        scope = f" · {gate['category']}" if gate.get("category") else ""
        return f"{suite}{scope}: {QUALITY_METRICS[gate['metric']]} ≥ {gate['threshold'] * 100:g}%"
    if kind == "capacity":
        workload = (workload_labels or {}).get(gate["workload"], gate["workload"])
        return f"Sustainable rate on {workload} ≥ {gate['threshold']:g} req/s"
    if kind == "latency":
        label, unit, lower = LATENCY_METRICS[gate["metric"]]
        return f"{label} at concurrency {gate['concurrency']} {'≤' if lower else '≥'} {gate['threshold']:,g} {unit}"
    if kind == "context":
        return f"Context ≥ {gate['threshold']:,} tokens"
    return "No open monitoring alert and every canary ok"


def _result(status, *, value=None, unit=None, evidence=None, note=None, uncertain=False, outcome=None):
    evidence = evidence or {}
    return {"status": status, "outcome": outcome, "value": value, "unit": unit, "uncertain": uncertain,
            "note": note, "run_id": evidence.get("run_id"), "group_id": evidence.get("group_id"),
            "freshness": evidence.get("freshness"), "at": evidence.get("at"),
            "fingerprint": evidence.get("fingerprint")}


def _judge(passed, evidence, **fields):
    outcome = "pass" if passed else "fail"
    status = "stale" if evidence.get("freshness") == "stale" else outcome
    return _result(status, outcome=outcome, evidence=evidence, **fields)


def _quality(gate, evidence):
    item = (evidence.get("quality") or {}).get(gate["suite"])
    if not item:
        return _result("missing", note="No completed run of this suite")
    threshold = gate["threshold"]
    if gate.get("category"):
        category = (item.get("categories") or {}).get(gate["category"])
        if category is None:
            return _result("missing", evidence=item, note="The suite has no such category")
        value = category.get(gate["metric"])
        interval = None
    else:
        value = item.get(gate["metric"])
        interval = item.get(f"{gate['metric']}_ci95")
    if value is None:
        return _result("missing", evidence=item, note="Nothing was scored")
    uncertain = bool(interval and interval[0] is not None and interval[1] is not None
                     and interval[0] <= threshold <= interval[1])
    return _judge(value >= threshold, item, value=value, unit="%", uncertain=uncertain,
                  note="The 95% interval contains the threshold" if uncertain else None)


def _capacity(gate, evidence):
    item = (evidence.get("capacity") or {}).get(gate["workload"])
    if not item:
        return _result("missing", note="No open-loop load test of this workload")
    status, rate = item.get("status"), item.get("rate")
    if status == "not_measured":
        return _result("missing", evidence=item, note="The load test finished no step")
    if rate is None:
        return _judge(False, item, unit="req/s", note="No offered rate met the SLO")
    if rate >= gate["threshold"]:
        return _judge(True, item, value=rate, unit="req/s")
    bound = status == "lower_bound"
    return _judge(False, item, value=rate, unit="req/s", uncertain=bound,
                  note="Only a lower bound was established; test higher rates" if bound else None)


def _users(gate, evidence):
    """Judged at the smallest measured context cap that holds the required sessions:
    a larger cap serves at most as many users, so it is conservative evidence."""
    item = (evidence.get("users") or {}).get(gate["user_model"])
    if not item:
        return _result("missing", note="No users-stage run of this user model")
    usable = [r for r in item.get("results") or [] if r.get("context_cap", 0) >= gate["context"]
              and r.get("status") in ("established", "lower_bound", "not_met")]
    if not usable:
        return _result("missing", evidence=item,
                       note=f"No conclusive result at a context cap of at least {gate['context']:,} tokens")
    result = min(usable, key=lambda r: r["context_cap"])
    users = result.get("users") or 0
    cap = f"at the {result['context_cap']:,}-token cap"
    if users >= gate["threshold"]:
        return _judge(True, item, value=users, unit="users", note=cap)
    bound = result["status"] == "lower_bound"
    return _judge(False, item, value=users, unit="users", uncertain=bound,
                  note=f"Only a lower bound was established {cap}; test more users" if bound else cap)


def _latency(gate, evidence):
    item = evidence.get("latency")
    if not item:
        return _result("missing", note="No closed-loop performance run")
    level = (item.get("levels") or {}).get(gate["concurrency"]) or (item.get("levels") or {}).get(
        str(gate["concurrency"]))
    if not level:
        return _result("missing", evidence=item, note=f"Not measured at concurrency {gate['concurrency']}")
    value = level.get(gate["metric"])
    _, unit, lower = LATENCY_METRICS[gate["metric"]]
    if value is None:
        return _result("missing", evidence=item, note="The metric was not measured")
    passed = value <= gate["threshold"] if lower else value >= gate["threshold"]
    return _judge(passed, item, value=value, unit=unit)


def _context(gate, evidence):
    measured, declared = evidence.get("context"), evidence.get("declared_context")
    item = measured if measured and measured.get("freshness") != "stale" else (declared or measured)
    if not item or item.get("tokens") is None:
        return _result("missing", note="No context sweep and no declared context limit")
    source = "measured" if item is measured else "declared by the deployment"
    return _judge(item["tokens"] >= gate["threshold"], item, value=item["tokens"], unit="tokens",
                  note=f"Context {source}")


def _operations(gate, evidence):
    ops = evidence.get("operations") or {}
    item = {"freshness": "current", "fingerprint": ops.get("fingerprint")}
    if ops.get("open_alerts"):
        return _judge(False, item, note=f"{ops['open_alerts']} unacknowledged alert{'s' if ops['open_alerts'] != 1 else ''}")
    canaries = ops.get("canaries") or []
    unevaluated = [c for c in canaries if not c.get("status")]
    failing = [c for c in canaries if c.get("status") not in (None, "ok")]
    if failing:
        return _judge(False, item, note="Canary " + ", ".join(f"{c['name']}: {c['status']}" for c in failing))
    if unevaluated:
        return _result("missing", evidence=item, note="A canary has not been evaluated yet")
    return _judge(True, item, note=f"{len(canaries)} canar{'ies' if len(canaries) != 1 else 'y'} ok"
                  if canaries else "No open alerts; no canaries configured")


EVALUATORS = {"quality": _quality, "capacity": _capacity, "users": _users, "latency": _latency, "context": _context,
              "operations": _operations}


def evaluate(gates, evidence):
    """Per-gate results and the profile verdict for one model's evidence."""
    results = [{**EVALUATORS[gate["type"]](gate, evidence), "gate": gate} for gate in gates]
    return {"revision": REVISION, "results": results, "verdict": verdict(results)}


def verdict(results):
    statuses = {r["status"] for r in results}
    if not results:
        return "incomplete"
    if "fail" in statuses:
        return "fails"
    if statuses & {"missing", "stale"}:
        return "incomplete"
    return "meets"
