"""Open-loop load tests: seeded arrivals, per-request SLOs, goodput and drift.

Requests are sent at their scheduled times whether or not earlier requests
finished. Arrivals that find the in-flight cap full are dropped and counted as
SLO misses, never delayed. Dispatch lag is added to user-visible timings so a
slow client cannot hide server queueing (coordinated omission).
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import math
import random
import threading
import time
import uuid

from app.benchmarking.cache_metrics import cache_metrics
from app.benchmarking.llm_client import ChatClient, client_protocol
from app.benchmarking.load_workload import PromptBuilder, schedule
from app.benchmarking.models import LatencyStats, RequestMetrics

REVISION = "open-loop-v1"
LAG_LIMIT_MS = 100.0
SAMPLE_LIMIT = 100_000
WINDOW_MIN_SECONDS = 10
WINDOWS_PER_STEP = 30
DRIFT_MIN_WINDOWS = 6
# Arrivals needed in each compared third; fewer cannot support a trend claim.
DRIFT_MIN_REQUESTS = 20
DRIFT_TTFT_RATIO = 1.25
DRIFT_ATTAINMENT_DROP = 0.05
DRIFT_SLO_SHARE = 0.5
PREWARM = 16
CHECK_INTERVAL = 0.5
INCOMPLETE_FINISH = {"content_filter", "repetition"}
REASONS = {
    "dropped": "Dropped at the in-flight cap",
    "error": "Request error",
    "incomplete": "Empty or filtered answer",
    "ttft": "First token over target",
    "ttft_unmeasured": "First token not measurable",
    "tpot": "Output token time over target",
    "tpot_unmeasured": "Output token time not measurable",
    "e2e": "Whole answer over target",
}


@dataclass
class Record:
    arrival: object
    metrics: RequestMetrics | None = None
    lag_ms: float | None = None
    fresh_connection: bool = False

    @property
    def status(self):
        m = self.metrics
        if m is None:
            return "dropped"
        if not m.ok:
            return "error"
        if m.completion_tokens <= 0 or m.finish_reason in INCOMPLETE_FINISH:
            return "incomplete"
        return "complete"

    def visible(self, value):
        return None if value is None or self.lag_ms is None else value + self.lag_ms

    def misses(self, slo):
        status = self.status
        if status != "complete":
            return [status]
        m, reasons = self.metrics, []
        ttft = self.visible(m.ttft_ms)
        if ttft is None:
            reasons.append("ttft_unmeasured")
        elif ttft > slo.ttft_ms:
            reasons.append("ttft")
        if slo.tpot_ms is not None:
            tpot = m.output_token_time_ms
            if tpot is None:
                reasons.append("tpot_unmeasured")
            elif tpot > slo.tpot_ms:
                reasons.append("tpot")
        if slo.e2e_ms is not None and self.visible(m.latency_ms) > slo.e2e_ms:
            reasons.append("e2e")
        return reasons


def _stats(values):
    return LatencyStats.from_samples([v for v in values if v is not None]).to_dict()


def _round(value, digits=1):
    return None if value is None else round(value, digits)


def _group_summary(records, slos, duration, target):
    complete = [r for r in records if r.status == "complete"]
    met = [r for r in records if not r.misses(slos[r.arrival.klass])]
    reasons = Counter(reason for r in records for reason in r.misses(slos[r.arrival.klass]))
    arrivals = len(records)
    return {
        "arrivals": arrivals,
        "dispatched": sum(r.metrics is not None for r in records),
        "dropped": reasons["dropped"],
        "errors": reasons["error"],
        "incomplete": reasons["incomplete"],
        "completed": len(complete),
        "slo_met": len(met),
        "attainment": len(met) / arrivals if arrivals else None,
        "meets_target": bool(arrivals) and len(met) / arrivals >= target,
        "goodput_rps": len(met) / duration if duration else None,
        "failure_reasons": dict(reasons),
        "ttft": _stats(r.visible(r.metrics.ttft_ms) for r in complete),
        "latency": _stats(r.visible(r.metrics.latency_ms) for r in complete),
        "output_token_time": _stats(r.metrics.output_token_time_ms for r in complete),
        "input_tokens": {k.removesuffix("_ms"): v for k, v in _stats(r.arrival.input_tokens for r in records).items()},
        "output_tokens": {k.removesuffix("_ms"): v for k, v in _stats(r.metrics.completion_tokens for r in complete).items()},
        "short_outputs": sum(r.metrics.completion_tokens < r.arrival.output_tokens for r in complete),
    }


def _windows(records, slos, step_seconds, target):
    width = max(WINDOW_MIN_SECONDS, step_seconds / WINDOWS_PER_STEP)
    count = max(1, math.ceil(step_seconds / width - 1e-9))
    buckets = [[] for _ in range(count)]
    for r in records:
        buckets[min(count - 1, int(r.arrival.offset // width))].append(r)
    windows = []
    for i, items in enumerate(buckets):
        complete = [r for r in items if r.status == "complete"]
        met = sum(not r.misses(slos[r.arrival.klass]) for r in items)
        seconds = min(width, step_seconds - i * width)
        windows.append({
            "start_s": i * width, "seconds": seconds, "arrivals": len(items), "slo_met": met,
            "attainment": met / len(items) if items else None,
            "errors": sum(r.status == "error" for r in items),
            "dropped": sum(r.status == "dropped" for r in items),
            "ttft_p95_ms": LatencyStats.from_samples([r.visible(r.metrics.ttft_ms) for r in complete
                                                      if r.metrics.ttft_ms is not None]).p95,
            "output_tokens_per_sec": sum(r.metrics.completion_tokens for r in complete) / seconds if seconds else None,
        })
    drift = None
    third = count // 3
    first = [r for w in buckets[:third] for r in w]
    last = [r for w in buckets[-third:] for r in w] if third else []
    if count >= DRIFT_MIN_WINDOWS and min(len(first), len(last)) >= DRIFT_MIN_REQUESTS:

        def part(items):
            done = [r.visible(r.metrics.ttft_ms) for r in items
                    if r.status == "complete" and r.metrics.ttft_ms is not None]
            met = sum(not r.misses(slos[r.arrival.klass]) for r in items)
            return LatencyStats.from_samples(done).p95, (met / len(items) if items else None)

        (ttft_first, att_first), (ttft_last, att_last) = part(first), part(last)
        ratio = ttft_last / ttft_first if ttft_first and ttft_last is not None else None
        drop = att_first - att_last if att_first is not None and att_last is not None else None
        error_windows = sum(1 for w in windows if w["arrivals"] and w["errors"] / w["arrivals"] > 1 - target)
        # A relative rise far below every SLO is noise, not lost headroom.
        floor = DRIFT_SLO_SHARE * min(s.ttft_ms for s in slos)
        drift = {
            "ttft_p95_first_ms": ttft_first, "ttft_p95_last_ms": ttft_last, "ttft_ratio": ratio,
            "attainment_first": att_first, "attainment_last": att_last, "attainment_drop": drop,
            "error_burst_windows": error_windows,
            "degrading": bool((ratio is not None and ratio > DRIFT_TTFT_RATIO and ttft_last > floor)
                              or (drop is not None and drop > DRIFT_ATTAINMENT_DROP)),
        }
    return windows, drift


def _samples(records, limit, seed):
    indices = list(range(len(records)))
    if len(indices) > limit:
        indices = sorted(random.Random(f"samples:{seed}").sample(indices, limit))
    codes = {"complete": 0, "incomplete": 1, "error": 2, "dropped": 3}
    columns = {key: [] for key in ("t", "c", "i", "o", "s", "lag", "ttft", "lat", "tpot", "out", "pin", "cached", "fresh")}
    for index in indices:
        r = records[index]
        m = r.metrics
        columns["t"].append(round(r.arrival.offset * 1000, 1))
        columns["c"].append(r.arrival.klass)
        columns["i"].append(r.arrival.input_tokens)
        columns["o"].append(r.arrival.output_tokens)
        columns["s"].append(codes[r.status])
        columns["lag"].append(_round(r.lag_ms))
        columns["ttft"].append(_round(r.visible(m.ttft_ms)) if m else None)
        columns["lat"].append(_round(r.visible(m.latency_ms)) if m else None)
        columns["tpot"].append(_round(m.output_token_time_ms, 2) if m else None)
        columns["out"].append(m.completion_tokens if m else None)
        columns["pin"].append(m.prompt_tokens if m else None)
        columns["cached"].append(m.cached_tokens if m else None)
        columns["fresh"].append(int(r.fresh_connection))
    return columns


def analyse_step(settings, index, records, *, duration, drain_seconds, started_at, ended_at,
                 max_in_flight, cancelled=False, sample_limit=SAMPLE_LIMIT):
    """Summarise one ladder step from its arrival records."""
    classes = settings.workload.classes
    slos = [c.slo for c in classes]
    target = settings.attainment
    overall = _group_summary(records, slos, duration, target)
    dispatched = [r for r in records if r.metrics is not None]
    lag = LatencyStats.from_samples([r.lag_ms for r in dispatched])
    client_limited = bool(overall["dropped"]) or (lag.p99 is not None and lag.p99 > LAG_LIMIT_MS)
    per_class = {}
    for k, c in enumerate(classes):
        group = [r for r in records if r.arrival.klass == k]
        per_class[c.name] = {**_group_summary(group, slos, duration, target), "slo": c.slo.model_dump()}
    class_ok = all(v["meets_target"] for v in per_class.values() if v["arrivals"])
    windows, drift = _windows(records, slos, settings.step_seconds if not cancelled else max(duration, 1e-9), target)
    good = [r.metrics for r in dispatched if r.metrics.ok]
    output = sum(m.completion_tokens for m in good)
    prompt = sum(m.prompt_tokens for m in good)
    busy = sum(r.metrics.latency_ms for r in dispatched) / 1000
    errors = list(dict.fromkeys(r.metrics.error for r in dispatched if not r.metrics.ok and r.metrics.error))[:3]
    passed = (not cancelled and overall["arrivals"] > 0 and overall["meets_target"]
              and class_ok and not client_limited)
    # A failure bounds capacity only if the server, not the client, caused it:
    # dispatch kept up, and requests that were sent still missed the target.
    sent_met = overall["slo_met"] / overall["dispatched"] if overall["dispatched"] else 0.0
    lag_ok = lag.p99 is None or lag.p99 <= LAG_LIMIT_MS
    conclusive_failure = (not passed and not cancelled and overall["arrivals"] > 0 and lag_ok
                          and (not overall["dropped"] or sent_met < target))
    return {
        "offered_rate": settings.rates[index],
        "step_seconds": settings.step_seconds,
        "measured_seconds": duration,
        "drain_seconds": drain_seconds,
        "started_at": started_at,
        "ended_at": ended_at,
        "achieved_arrival_rate": overall["arrivals"] / duration if duration else None,
        **overall,
        "passed": passed,
        "conclusive_failure": conclusive_failure,
        "dispatched_attainment": sent_met if overall["dispatched"] else None,
        "client_limited": client_limited,
        "cancelled": cancelled,
        "dispatch_lag": lag.to_dict(),
        "max_in_flight": max_in_flight,
        # Little's law over the step plus drain: busy request-seconds / elapsed seconds.
        "mean_in_flight": busy / (duration + drain_seconds) if duration + drain_seconds > 0 else None,
        "fresh_connections": sum(r.fresh_connection for r in dispatched),
        "input_tokens_reported": prompt,
        "output_tokens_delivered": output,
        "output_tokens_per_sec": output / (duration + drain_seconds) if duration + drain_seconds > 0 else None,
        "input_tokens_per_sec": prompt / (duration + drain_seconds) if duration + drain_seconds > 0 else None,
        "cache_metrics": cache_metrics([r.metrics for r in dispatched], (duration + drain_seconds) * 1000),
        "classes": per_class,
        "windows": windows,
        "drift": drift,
        "error_examples": errors,
        "samples": _samples(records, sample_limit, f"{settings.seed}:{index}"),
        "samples_total": len(records),
    }


def summarise(steps):
    finished = [s for s in steps if not s["cancelled"]]
    passing = [s for s in finished if s["passed"]]
    failed = [s for s in finished if not s["passed"]]
    if not passing:
        return {"sustainable_rate": None, "sustainable_status": "not_met" if finished else "not_measured",
                "goodput_at_sustainable": None,
                "first_failed_rate": failed[0]["offered_rate"] if failed else None, "non_monotonic": False}
    best = max(passing, key=lambda s: s["offered_rate"])
    higher = [s for s in failed if s["offered_rate"] > best["offered_rate"]]
    bounded = [s for s in higher if s.get("conclusive_failure")]
    return {
        "sustainable_rate": best["offered_rate"],
        # Client-limited failures above the best step leave the boundary open.
        "sustainable_status": "established" if bounded else "lower_bound",
        "goodput_at_sustainable": best["goodput_rps"],
        "first_failed_rate": min((s["offered_rate"] for s in bounded), default=None),
        "inconclusive_rates": [s["offered_rate"] for s in higher if not s.get("conclusive_failure")],
        # A failure below the sustainable rate suggests noise or instability.
        "non_monotonic": any(s["offered_rate"] < best["offered_rate"] for s in failed),
    }


@dataclass
class LoadReport:
    model: str
    protocol: dict
    steps: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    cancelled: bool = False
    stop_reason: str = ""
    telemetry: dict | None = None

    def to_dict(self):
        return {
            "schema_version": 5,
            "kind": "open_loop",
            "model": self.model,
            "protocol": self.protocol,
            "steps": self.steps,
            "summary": summarise(self.steps),
            "notes": self.notes,
            "cancelled": self.cancelled,
            "stop_reason": self.stop_reason,
            "telemetry": self.telemetry,
        }


def protocol(settings, client_config, in_flight_cap):
    return {
        "revision": REVISION,
        "load_model": "open_loop",
        "preset": settings.preset,
        "workload": settings.workload.model_dump(),
        "workload_hash": settings.workload.fingerprint(),
        "arrival": settings.arrival,
        "burstiness": settings.burstiness if settings.arrival == "gamma" else 1.0,
        "seed": settings.seed,
        "rates": list(settings.rates),
        "step_seconds": settings.step_seconds,
        "request_timeout": settings.request_timeout,
        "in_flight_cap": in_flight_cap,
        "attainment_target": settings.attainment,
        "stop_after_failed_steps": settings.stop_after_failed_steps,
        "expected_arrivals": settings.expected_arrivals,
        "temperature": client_config.temperature,
        "reasoning_effort": client_config.reasoning_effort,
        "model_seed": client_config.seed,
        "reference_tokenizer": "cl100k_base",
        "attempts_per_request": 1,
        "lag_limit_ms": LAG_LIMIT_MS,
        "sample_limit": SAMPLE_LIMIT,
        "comparison_key": settings.comparison_key(),
        "slo_definition": "A request meets its class SLO when it completes, dispatch lag + TTFT <= ttft_ms, its output token time <= tpot_ms (if set) and dispatch lag + latency <= e2e_ms (if set). Unmeasurable required metrics are misses.",
        "attainment_definition": "SLO-met requests / all arrivals, including dropped and failed requests.",
        "goodput_definition": "SLO-met requests / arrival window seconds.",
        "pass_definition": "Overall and every class reach the attainment target, no arrival was dropped and dispatch lag p99 <= lag_limit_ms.",
        "throughput_definition": "Delivered tokens / (arrival window + drain).",
        "output_definition": "The prompt asks the model to count until stopped; max_tokens is the sampled output length, so finish_reason=length delivers the target. Counts include reasoning tokens.",
        "context_definition": "System prefix plus user text in cl100k_base reference tokens, excluding chat framing.",
    }


def run_load_test(client_config, settings, in_flight_cap, *, progress=None, cancelled=None,
                  on_step=None, client_factory=None, sample_limit=SAMPLE_LIMIT):
    cancelled = cancelled or (lambda: False)
    client_factory = client_factory or ChatClient
    base = replace(client_config, detect_repetition=False, stream=True, max_retries=1,
                   retry_transport_errors=False, stream_deadline=float(settings.request_timeout),
                   timeout=min(client_config.timeout, float(settings.request_timeout)))
    report = LoadReport(client_config.model, protocol(settings, client_config, in_flight_cap))
    builder = PromptBuilder(settings.workload, uuid.uuid4().hex[:12])
    total = settings.expected_arrivals
    clients, pool, pool_lock = [], [], threading.Lock()
    last_check = [0.0, False]

    def is_cancelled():
        now = time.perf_counter()
        if now - last_check[0] >= CHECK_INTERVAL:
            last_check[0], last_check[1] = now, bool(cancelled())
        return last_check[1]

    def take():
        with pool_lock:
            if pool:
                return pool.pop(), False
        client = client_factory(measured)
        with pool_lock:
            clients.append(client)
        return client, True

    def give(client):
        with pool_lock:
            pool.append(client)

    try:
        warm = client_factory(replace(base, max_retries=3))
        clients.append(warm)
        warm_ok = False
        for i in range(2):
            if cancelled():
                report.cancelled = True
                return report
            _, _, metrics = warm.complete(f"Warmup {builder.nonce}-{i}. Reply with OK.", max_tokens=16, retries=3)
            warm_ok |= metrics.ok
        if not warm_ok:
            report.stop_reason = "Endpoint did not answer warmup requests; measurement stopped."
            return report
        if not warm.streaming_supported:
            report.stop_reason = "Endpoint does not stream; first-token SLOs cannot be measured."
            return report
        measured = replace(warm.config, stream=True, request_stream_usage=warm._stream_usage_supported, max_retries=1)
        report.protocol["client"] = client_protocol(measured)
        report.protocol["streaming"] = True
        prewarm = [client_factory(measured) for _ in range(min(in_flight_cap, PREWARM))]
        clients.extend(prewarm)

        def warm_one(pair):
            i, client = pair
            client.complete(f"Warmup {builder.nonce}-c{i}. Reply with OK.", max_tokens=16, retries=1)

        with ThreadPoolExecutor(max_workers=len(prewarm)) as executor:
            list(executor.map(warm_one, enumerate(prewarm)))
        pool.extend(prewarm)

        dispatched_total, failures = 0, 0
        for index, rate in enumerate(settings.rates):
            if cancelled():
                report.cancelled = True
                break
            arrivals = schedule(settings, index)
            records = [Record(a) for a in arrivals]
            slots = threading.BoundedSemaphore(in_flight_cap)
            state = {"in_flight": 0, "max": 0}
            state_lock = threading.Lock()
            label = f"{rate:g} req/s"

            def worker(record, request_index, due):
                try:
                    client, fresh = take()
                    try:
                        record.fresh_connection = fresh
                        try:
                            system, user = builder.build(record.arrival, request_index)
                            record.lag_ms = (time.perf_counter() - due) * 1000
                            _, _, metrics = client.complete(user, system, max_tokens=record.arrival.output_tokens, retries=1)
                        except Exception as error:  # noqa: BLE001 - recorded as a failed request
                            if record.lag_ms is None:
                                record.lag_ms = (time.perf_counter() - due) * 1000
                            metrics = RequestMetrics(ok=False, error=str(error))
                        record.metrics = metrics
                    finally:
                        give(client)
                finally:
                    with state_lock:
                        state["in_flight"] -= 1
                    slots.release()

            step_cancelled = False
            started_at = time.time()
            t0 = time.perf_counter()
            last_progress = t0
            with ThreadPoolExecutor(max_workers=in_flight_cap) as executor:
                for i, record in enumerate(records):
                    due = t0 + record.arrival.offset
                    while True:
                        if is_cancelled():
                            step_cancelled = True
                            break
                        wait = due - time.perf_counter()
                        if wait <= 0:
                            break
                        time.sleep(min(wait, CHECK_INTERVAL))
                    if step_cancelled:
                        records = records[:i]
                        break
                    if slots.acquire(blocking=False):
                        with state_lock:
                            state["in_flight"] += 1
                            state["max"] = max(state["max"], state["in_flight"])
                        executor.submit(worker, record, f"{index}-{i}", due)
                    else:
                        record.lag_ms = None  # dropped: never sent
                    dispatched_total += 1
                    now = time.perf_counter()
                    if progress and now - last_progress >= 1:
                        last_progress = now
                        progress(f"Open-loop · {label}", dispatched_total, total)
                arrival_end = time.perf_counter()
            ended = time.perf_counter()
            duration = (arrival_end - t0) if step_cancelled else float(settings.step_seconds)
            step = analyse_step(
                settings, index, records, duration=duration,
                drain_seconds=max(0.0, ended - t0 - duration), started_at=started_at, ended_at=time.time(),
                max_in_flight=state["max"], cancelled=step_cancelled,
                sample_limit=max(1, sample_limit // len(settings.rates)),
            )
            report.steps.append(step)
            if on_step:
                on_step(report)
            if progress:
                progress(f"Open-loop · {label}", dispatched_total, total)
            if step_cancelled or cancelled():
                report.cancelled = True
                break
            if not step["completed"]:
                report.stop_reason = f"No request completed at {rate:g} req/s; higher rates were not attempted."
                break
            failures = 0 if step["passed"] else failures + 1
            if failures >= settings.stop_after_failed_steps and index + 1 < len(settings.rates):
                report.stop_reason = (f"Stopped after {failures} consecutive steps below the "
                                      f"{settings.attainment:.0%} SLO attainment target; higher rates were not attempted.")
                break
        if any(s["client_limited"] for s in report.steps):
            report.notes.append("Some steps were client-limited (dropped arrivals or dispatch lag p99 over "
                                f"{LAG_LIMIT_MS:g} ms); they cannot establish capacity.")
        if any(s["samples_total"] > len(s["samples"]["t"]) for s in report.steps):
            report.notes.append("Per-request samples were thinned to a seeded uniform subset; statistics use every request.")
        return report
    finally:
        for client in clients:
            client.session.close()
