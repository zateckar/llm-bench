"""Explicit planning assumptions applied to measured, matching workloads.

No hardware multiplier or input/output token equivalence. Arrival-rate claims
come only from open-loop runs and apply to their own workload and SLOs.
The user population includes think/tool time; in-flight requests do not.
"""

from collections import Counter
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.benchmarking.staged_performance import is_staged, stages_of

REVISION = "serving-capacity-v3"


class CapacityAssumptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    headroom_percent: float = Field(30, ge=0, le=80)
    availability_percent: float = Field(98, ge=1, le=100)
    busy_hours: float = Field(10, gt=0, lt=24)
    busy_traffic_percent: float = Field(80, gt=0, lt=100)
    peak_factor: float = Field(2, ge=1, le=20)
    chat_mix_percent: float = Field(80, ge=0, le=100)
    effort: str = Field("default", max_length=20, pattern=r"^[a-z]+$")
    cache_mode: Literal["cold", "warm"] = "cold"
    chat_context: int = Field(1024, ge=128, le=1_048_576)
    chat_output: int = Field(256, ge=16, le=65_536)
    chat_ttft_seconds: float = Field(2, gt=0, le=1800)
    chat_response_seconds: float = Field(30, gt=0, le=3600)
    chat_stream_tokens_minute: float = Field(1200, ge=1, le=60000)
    chat_user_tokens_minute: float = Field(300, ge=1, le=60000)
    agent_context: int = Field(32768, ge=128, le=1_048_576)
    agent_output: int = Field(256, ge=16, le=65_536)
    agent_ttft_seconds: float = Field(10, gt=0, le=1800)
    agent_response_seconds: float = Field(120, gt=0, le=3600)
    agent_stream_tokens_minute: float = Field(600, ge=1, le=60000)
    agent_user_tokens_minute: float = Field(2000, ge=1, le=60000)

    def profile(self, key):
        return {name: getattr(self, f"{key}_{name}") for name in (
            "context", "output", "ttft_seconds", "response_seconds",
            "stream_tokens_minute", "user_tokens_minute")}


def finite(value, default=None):
    return value if type(value) in (int, float) and math.isfinite(value) else default


def statistic(point, key, name):
    value = point.get(key) or {}
    return finite(value.get(name)) if isinstance(value, dict) else None


def evidence(perf):
    """Never combine runs, reasoning levels or cold/warm cells."""
    if is_staged(perf):
        # Each stage keeps its own source label, effort and cache mode.
        return [record for report in stages_of(perf).values() for record in evidence(report)]
    protocol = perf.get("protocol") or {}
    if not protocol.get("revision"):
        return []
    records = []

    def add(point, context, mode, effort, source):
        requests = finite(point.get("requests"), 0)
        completed = finite(point.get("completed"), requests-finite(point.get("errors"), 0))
        wall = finite(point.get("wall_ms"), 0) / 1000
        out_rate = finite(point.get("output_tokens_per_sec", point.get("aggregate_tokens_per_sec")))
        prompt_mean = statistic(point, "prompt_tokens", "mean")
        if prompt_mean is None and completed > 0:
            prompt_mean = finite(point.get("prompt_tokens", 0), 0) / completed
        mean_out = finite(point.get("output_tokens"))
        mean_out = mean_out / completed if mean_out is not None and completed > 0 else statistic(point, "output_tokens", "mean")
        if mean_out is None:
            mean_out = statistic(point, "output_length", "mean")
        if mean_out is None and wall > 0 and completed > 0 and out_rate is not None:
            mean_out = out_rate * wall / completed
        tpot = statistic(point, "output_token_time", "p95_ms")
        stream_rate = 1000/tpot if tpot is not None and tpot > 0 else None
        # Older sweep cells retain minimum observed decode rate, not TPOT p95.
        if stream_rate is None:
            stream_rate = statistic(point, "generation_tokens_per_sec", "min")
        records.append({
            "context": finite(context), "cache_mode": mode, "effort": effort, "source": source,
            "concurrency": finite(point.get("concurrency"), 0), "requests": requests,
            "completed": completed, "wall_seconds": wall, "output_rate": out_rate,
            "mean_input": prompt_mean, "mean_output": mean_out, "stream_rate": stream_rate,
            "ttft_p95_ms": statistic(point, "ttft", "p95_ms"),
            "latency_p95_ms": statistic(point, "latency", "p95_ms"),
            "mean_latency_ms": statistic(point, "latency", "mean_ms"),
            "estimated_output": finite(point.get("estimated_token_requests", point.get("estimated_output_requests")), 0),
            "estimated_input": finite(point.get("estimated_prompt_requests", point.get("estimated_input_requests")), 0),
            "burst_requests": finite(point.get("burst_delivery_requests", point.get("burst_requests")), 0),
            "stream_samples": statistic(point, "output_token_time", "count") or statistic(point, "generation_tokens_per_sec", "count") or 0,
            "status": point.get("status", "measured"),
        })

    if perf.get("schema_version") == 3:
        effort = protocol.get("reasoning_effort") or "default"
        for point in perf.get("concurrency", []):
            add(point, protocol.get("input_reference_tokens"), "cold", effort, "Fixed load")
        for pair in perf.get("cache_reuse", []):
            for mode in ("cold", "warm"):
                if pair.get(mode) and (mode == "cold" or pair.get("priming_ok")):
                    add(pair[mode], pair.get("context_tokens"), mode, effort, "Prefix pair")
    elif perf.get("schema_version") == 4 and perf.get("kind") == "context_sweep":
        for cell in perf.get("cells", []):
            add(cell, cell.get("context_tokens"), "cold", cell.get("effort", "default"), "Context sweep")
            warm = cell.get("cache_reuse")
            if warm and warm.get("priming_ok"):
                add(warm, cell.get("context_tokens"), "warm", cell.get("effort", "default"), "Context sweep")
    return records


def rejection(point, profile):
    if not point["context"] or not 0.8 <= profile["context"]/point["context"] <= 1.25:
        return "No matching context measurement"
    if point["requests"] < 4 or point["concurrency"] < 1 or point["wall_seconds"] <= 0:
        return "Too few timed requests"
    if point["status"] != "measured" or not 0.99 <= point["completed"]/point["requests"] <= 1:
        return "Failures or incomplete answers exceed 1%"
    if point["estimated_output"] or point["estimated_input"]:
        return "Estimated token counts"
    if not point["mean_output"] or not 0.8 <= profile["output"]/point["mean_output"] <= 1.25:
        return "No matching output length measurement"
    if not point["mean_input"] or not point["output_rate"] or point["output_rate"] <= 0:
        return "Missing input/output throughput"
    if point["ttft_p95_ms"] is None or point["latency_p95_ms"] is None:
        return "Missing p95 latency measurements"
    projected_ttft = point["ttft_p95_ms"]*max(1, profile["context"]/point["context"])
    if projected_ttft > profile["ttft_seconds"]*1000:
        return "First-token latency exceeds target"
    # Only small length differences are allowed; longer contexts need their own tests.
    projected = projected_ttft + max(0, point["latency_p95_ms"]-point["ttft_p95_ms"])*max(1, profile["output"]/point["mean_output"])
    if projected > profile["response_seconds"]*1000:
        return "Response latency exceeds target"
    if point["burst_requests"] or point["stream_samples"] < point["completed"] or point["stream_rate"] is None:
        return "Incomplete streaming delivery evidence"
    if point["stream_rate"]*60 < profile["stream_tokens_minute"]:
        return "Streaming token rate below target"
    return None


def daily_budget(rate, assumptions):
    share = assumptions.busy_traffic_percent/100
    busy = assumptions.busy_hours*3600/(share*assumptions.peak_factor)
    quiet = (24-assumptions.busy_hours)*3600/(1-share)
    return rate*min(busy, quiet)*assumptions.availability_percent/100


def projection(rate, input_tokens, output_tokens, users, assumptions):
    requests = daily_budget(rate, assumptions)
    flat_requests = rate*86400*assumptions.availability_percent/100
    return {"requests_per_second": rate, "input_tokens_per_second": rate*input_tokens,
            "output_tokens_per_second": rate*output_tokens,
            "daily_requests": requests, "daily_input_tokens": requests*input_tokens,
            "daily_output_tokens": requests*output_tokens,
            "daily_total_tokens": requests*(input_tokens+output_tokens),
            "flat_daily_output_tokens": flat_requests*output_tokens,
            "active_users": math.floor(users)}


def observed_quality(run):
    """Recorded workload throughput is corroborating evidence, never a chat SLO result.

    Use question-phase duration, not created/completed timestamps: creation can
    include hours in the queue. Usage can include reasoning/recovery calls.
    """
    seconds = finite(run.get("duration_ms"), 0)/1000
    output = finite(run.get("total_completion_tokens"), 0)
    questions = finite(run.get("total_questions"), 0)
    if run.get("status") not in {"completed", "failed"} or seconds <= 0 or output <= 0 or questions <= 0:
        return None
    rate = output/seconds
    return {"source": "Quality workload · recorded usage", "seconds": seconds, "output_tokens": output,
            "output_tokens_per_second": rate, "output_tokens_per_hour": rate*3600,
            "flat_24h_output_tokens": rate*86400, "workers": finite(run.get("workers")),
            "mean_output_tokens_per_task": output/questions,
            "limit": "Quality workers were capped; server saturation and chat latency capacity were not established."}


def load_limit(selected, candidates, profile):
    higher = [point for point in candidates if point["concurrency"] > selected["concurrency"]
              and point["requests"] > 0 and point["wall_seconds"] > 0
              and point["context"] and 0.8 <= profile["context"]/point["context"] <= 1.25
              and point["mean_output"] and 0.8 <= profile["output"]/point["mean_output"] <= 1.25]
    if not higher:
        return "Test concurrency ceiling reached; higher serving capacity remains unmeasured."
    if any(rejection(point, profile) in {"First-token latency exceeds target", "Response latency exceeds target",
                                        "Streaming token rate below target"} for point in higher):
        return "Higher tested concurrency exceeded a latency/delivery target; repeat sustained arrival-rate tests to establish the boundary."
    return "Higher loads were tested, but the sustained serving limit was not established."


def monitoring(perf):
    from app.benchmarking.vllm_telemetry import alignment_shift
    if is_staged(perf):
        # Every stage collects its own telemetry; report warnings from all of them.
        stages = [monitoring(report) for report in stages_of(perf).values() if isinstance(report.get("telemetry"), dict)]
        if not stages:
            return monitoring({})
        result = dict(stages[0])
        result["warnings"] = list(dict.fromkeys(w for item in stages for w in item["warnings"]))
        result["baseline_output"] = next((item["baseline_output"] for item in stages
                                          if item["baseline_output"] is not None), None)
        return result
    data = perf.get("telemetry")
    if not isinstance(data, dict):
        return {"status": "No saved vLLM monitoring for this run", "warnings": [], "baseline_output": None}
    shift, aligned = alignment_shift(data)
    window = data.get("window") or {}
    start = finite(window.get("measurement_start"))
    result = {"status": "Configured timestamp correction" if aligned and shift else "Aligned source timestamps" if aligned else "Telemetry alignment uncertain",
              "warnings": [], "baseline_output": None, "scope": data.get("scope")}
    if data.get("status") != "collected":
        result["warnings"].append("vLLM collection is incomplete; missing server metrics are not evidence of spare capacity.")
    if not aligned or start is None:
        result["warnings"].append("Capacity uses client measurements; server timing cannot support phase-level diagnosis.")
        return result
    for metric in data.get("metrics", []):
        baseline, active = {}, {}
        for series in metric.get("series", []):
            for stamp, value in series.get("values", []):
                if finite(value) is not None:
                    (baseline if stamp+shift < start else active).setdefault(stamp, []).append(value)
        if metric["id"] == "output" and len(baseline) >= 3:
            result["baseline_output"] = sum(sum(v) for v in baseline.values())/len(baseline)
        if len(active) < 3:
            continue
        if metric["id"] == "waiting" and max(sum(v) for v in active.values()) > 0:
            result["warnings"].append("vLLM queueing occurred during measurement; validate peak arrivals with the proposed workload mix.")
        if metric["id"] == "kv_cache" and max(max(v) for v in active.values()) >= 0.9:
            result["warnings"].append("KV occupancy reached 90%; long-context concurrency may be limited by cache capacity.")
        if metric["id"] == "preemptions" and max(sum(v) for v in active.values()) > 0:
            result["warnings"].append("Rolling preemptions were observed; increase headroom or reduce long-context concurrency.")
    return result


def arrival_capacity(perf, assumptions):
    """Measured open-loop evidence: the highest seeded arrival rate meeting every class SLO."""
    if is_staged(perf):
        perf = stages_of(perf).get("capacity") or {}
    if perf.get("schema_version") != 5 or perf.get("kind") != "open_loop":
        return None
    summary = perf.get("summary") or {}
    protocol = perf.get("protocol") or {}
    classes = (protocol.get("workload") or {}).get("classes") or []
    status = summary.get("sustainable_status", "not_measured")
    rate = finite(summary.get("sustainable_rate"))
    result = {"status": status, "requests_per_second": rate, "goodput_rps": finite(summary.get("goodput_at_sustainable")),
              "workload_hash": protocol.get("workload_hash"), "preset": protocol.get("preset"),
              "attainment_target": protocol.get("attainment_target"), "estimate": None, "classes": [],
              "inconclusive_rates": summary.get("inconclusive_rates") or []}
    step = next((s for s in perf.get("steps", []) if s.get("passed") and s.get("offered_rate") == rate), None)
    if rate is None or step is None:
        result["reason"] = ("No tested arrival rate met the SLO target." if status == "not_met"
                            else "No finished open-loop step.")
        return result
    answered = finite(step.get("completed"), 0) + finite(step.get("incomplete"), 0)
    mean_input = finite(step.get("input_tokens_reported"), 0) / answered if answered else 0
    mean_output = finite(step.get("output_tokens_delivered"), 0) / answered if answered else 0
    weight = sum(finite(c.get("weight"), 0) for c in classes) or 1
    demand = {"chat": assumptions.chat_user_tokens_minute, "agent": assumptions.agent_user_tokens_minute}
    planned = rate * (1 - assumptions.headroom_percent / 100)
    estimate = projection(planned, mean_input, mean_output, 0, assumptions)
    if classes and all(c.get("name") in demand for c in classes):
        # Users per class: that class's planned output tokens/min / its per-user demand.
        users = sum(planned * finite(c.get("weight"), 0) / weight * 60 * mean_output / demand[c["name"]] for c in classes)
        estimate["active_users"] = math.floor(users)
    else:
        estimate["active_users"] = None
    result.update(
        estimate=estimate, mean_input=mean_input, mean_output=mean_output,
        output_tokens_per_second=rate * mean_output,
        classes=[{"name": name, "share": finite(next((c.get("weight") for c in classes if c.get("name") == name), 0), 0) / weight,
                  "attainment": item.get("attainment"), "ttft_p95_ms": (item.get("ttft") or {}).get("p95_ms"),
                  "slo": item.get("slo")} for name, item in (step.get("classes") or {}).items()],
        reason=("A higher tested rate missed the SLO because of the server." if status == "established"
                else "Every tested rate passed or failed only because of the client; capacity may be higher."),
    )
    return result


def estimate_capacity(run, assumptions=None):
    assumptions = assumptions or CapacityAssumptions()
    perf = run.get("perf") or {}
    arrival = arrival_capacity(perf, assumptions)
    points = evidence(perf)
    ready = run.get("status", "completed") in {"completed", "failed"} and not perf.get("cancelled")
    scenarios = []
    for key, title in (("chat", "Simple chat"), ("agent", "Long-context agents")):
        profile = assumptions.profile(key)
        matches = [p for p in points if p["effort"] == assumptions.effort and p["cache_mode"] == assumptions.cache_mode]
        rejected = Counter()
        eligible = []
        for point in matches if ready else []:
            reason = rejection(point, profile)
            if reason:
                rejected[reason] += 1
                continue
            # Input rate is wall-clock throughput at this same point, not engine prefill speed.
            input_tokens = point["mean_input"]*profile["context"]/point["context"]
            rate = min(point["completed"]/point["wall_seconds"], point["output_rate"]/profile["output"],
                       point["completed"]*point["mean_input"]/point["wall_seconds"]/input_tokens)
            eligible.append((rate, input_tokens, point))
        row = {"key": key, "title": title, "profile": profile, "status": "unavailable",
               "rejections": dict(rejected), "estimate": None, "evidence": None}
        if eligible:
            rate, input_tokens, point = max(eligible, key=lambda item: item[0])
            rate *= 1-assumptions.headroom_percent/100
            row.update(status="provisional", evidence=point, estimate=projection(
                rate, input_tokens, profile["output"], rate*60*profile["output"]/profile["user_tokens_minute"], assumptions))
            row["mean_inflight"] = rate*point["mean_latency_ms"]/1000 if point["mean_latency_ms"] is not None else None
            row["reason"] = "Traffic budget at an observed load point; arrival-rate and sustained-load validation pending."
            row["limit"] = load_limit(point, matches, profile)
        else:
            row["reason"] = "Run must finish without cancellation." if not ready else "; ".join(rejected) or "No measurements for this reasoning/cache setting."
        scenarios.append(row)
    share = assumptions.chat_mix_percent/100
    components = [(scenarios[0], share), (scenarios[1], 1-share)]
    used = [(row, weight) for row, weight in components if weight > 0]
    mix = {"title": "Shared deployment mix", "status": "unavailable", "estimate": None}
    if all(row["estimate"] for row, _ in used):
        rate = 1/sum(weight/row["estimate"]["requests_per_second"] for row, weight in used)
        input_tokens = sum(weight*row["estimate"]["input_tokens_per_second"]/row["estimate"]["requests_per_second"] for row, weight in used)
        output_tokens = sum(weight*row["profile"]["output"] for row, weight in used)
        users = sum(rate*weight*60*row["profile"]["output"]/row["profile"]["user_tokens_minute"] for row, weight in used)
        mix.update(status="modeled", estimate=projection(rate, input_tokens, output_tokens, users, assumptions))
    observations = []
    quality = observed_quality(run)
    if quality:
        observations.append(quality)
    for row in scenarios:
        point = row["evidence"]
        if point:
            observations.append({"source": row["title"]+" · "+point["source"], "seconds": point["wall_seconds"],
                "output_tokens": point["mean_output"]*point["completed"], "output_tokens_per_second": point["output_rate"],
                "output_tokens_per_hour": point["output_rate"]*3600, "flat_24h_output_tokens": point["output_rate"]*86400,
                "workers": point["concurrency"], "mean_output_tokens_per_task": point["mean_output"], "limit": row["limit"]})
    maximum = {"status": "not_established", "output_tokens_per_second": None,
               "reason": "A capped closed-loop test does not establish the maximum serving rate or maximum user population."}
    if arrival and arrival["estimate"]:
        maximum = {"status": arrival["status"], "requests_per_second": arrival["requests_per_second"],
                   "output_tokens_per_second": arrival["output_tokens_per_second"],
                   "reason": "Highest seeded arrival rate meeting every class SLO for this workload only. " + arrival["reason"]}
    return {"revision": REVISION, "run_id": run.get("id"), "label": run.get("label", "Run"),
            "assumptions": assumptions.model_dump(), "scenarios": scenarios, "mix": mix,
            "observations": observations, "arrival": arrival,
            "maximum_capacity": maximum,
            "monitoring": monitoring(perf), "reasoning": assumptions.effort, "cache_mode": assumptions.cache_mode,
            "notes": ["Each standalone scenario uses the whole measured deployment. Do not add their capacities or multiply by GPU count.",
                      "The mix uses a weighted service-demand model, not a measured mixed-load result. Prefill/decode interference can lower it.",
                      "Daily traffic budgets apply the chosen distribution, headroom and availability to a tested workload. They are not server throughput ceilings.",
                      "The 24-hour equivalents in achieved rates assume the observed workload and rate continue; they are not measured daily production or saturation limits.",
                      "Active users include thinking/tool time at the specified average token demand. Simultaneous requests are bounded by tested concurrency.",
                      "Input and output token counts are reported separately; their sum is volume, not equivalent compute cost."]}
