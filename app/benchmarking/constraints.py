"""What limits a deployment under a given load: one named constraint with evidence and remedies.

A users-stage level that misses its SLO, or that is beyond a hard limit, is
diagnosed from what the client saw (which SLO part failed, cache reuse the
session history allows) and, when B300 telemetry covers its measured window,
from vLLM and GPU signals over that window: queue, batch occupancy, KV cache
occupancy and preemptions, prefix-cache hits against the reuse the sessions
allow, the share of prompt work against generation, and GPU activity.

The rules are ordered: a full KV cache explains queueing and slow first
tokens, evicted session histories explain extra prefill, so the first matching
rule names the constraint and later matches are listed as also present. See
docs/design-session-capacity.md.
"""

from app.benchmarking.vllm_telemetry import alignment_shift

REVISION = "constraints-v1"
# KV occupancy (vLLM: blocks held by running requests).
KV_FULL = 0.9
KV_HIGH = 0.8
# Observed prefix hits below this share of the reuse the sessions allow mean histories were lost.
EVICTION_SHARE = 0.6
# Sessions that allow less reuse than this cannot show lost histories.
MIN_EXPECTED_REUSE = 0.3
# Uncached prompt tokens per generated token from which prompt processing dominates.
PREFILL_HEAVY = 2.0
# SM activity from which GPUs count as busy, and under which they are idle.
BUSY = 0.7
IDLE = 0.4
# Running requests at their ceiling with a queue for this share of the window: a batch-slot cap.
FLAT_SHARE = 0.5
FAILURE_SHARE = 0.10
# Rolling rates and percentiles start including the window one minute in.
ROLLING_SECONDS = 60

LABELS = {
    "kv_capacity": "KV cache capacity",
    "cache_eviction": "KV cache eviction of session histories",
    "cache_miss": "Prefix cache not reused",
    "batch_slots": "Batch slots (max-num-seqs)",
    "scheduler": "Scheduler token budget or serving overhead",
    "compute": "GPU compute",
    "queue": "Queueing with KV cache room",
    "prefill": "Prefill (prompt processing)",
    "prefill_interference": "Prefill interfering with decode",
    "decode": "Decode speed",
    "client": "Benchmark client",
    "errors": "Request errors",
    "prefill_or_queue": "First token: prefill or queueing",
    "unknown": "Not identified",
}
SUMMARY = {
    "kv_capacity": "Running requests fill the KV cache, so new requests wait or running ones are preempted and recomputed.",
    "cache_eviction": "Idle session histories are evicted between turns and every turn prefills them again.",
    "cache_miss": "Prompts that should reuse a cached history are prefilled from scratch, without memory pressure.",
    "batch_slots": "The number of running requests stops at a ceiling while requests queue and KV cache has room.",
    "scheduler": "Requests queue while KV cache has room and the GPUs are not busy.",
    "compute": "Requests queue while the GPUs are busy and the KV cache has room.",
    "queue": "Requests queue while the KV cache has room; GPU metrics are needed to tell a compute limit from a scheduler limit.",
    "prefill": "First tokens are slow because prompt processing takes long, not because of a queue.",
    "prefill_interference": "Output speed drops because long prompt chunks share each step with decoding requests.",
    "decode": "Output speed per user drops as the batch grows.",
    "client": "The benchmark client could not send requests on time; the deployment may serve more.",
    "errors": "Requests fail or come back incomplete.",
    "prefill_or_queue": "First tokens are slow; without server metrics prefill and queueing cannot be told apart.",
    "unknown": "No rule matched the available signals.",
}


def _remedy(title, detail):
    return {"title": title, "detail": detail}


def _flag(config, key):
    values = {str(c.get(key, "")).lower() for c in config or [] if c.get(key) is not None}
    return values.pop() if len(values) == 1 else None


def remedies(constraint, config=None, signals=None):
    """Changes that address ``constraint``, skipping ones vLLM's cache config shows are in place."""
    dtype = _flag(config, "cache_dtype")
    fp8_kv = dtype is not None and "fp8" in dtype
    prefix_off = _flag(config, "enable_prefix_caching") == "false"
    cpu_blocks = _flag(config, "num_cpu_blocks")
    offloading = cpu_blocks not in (None, "", "0", "none") or _flag(config, "kv_offloading_size") not in (None, "", "none")
    kv_fp8 = _remedy("FP8 KV cache", "--kv-cache-dtype fp8 roughly doubles the tokens the cache holds and halves attention reads; check quality on long contexts.")
    offload = _remedy("KV cache offloading", "Offload evicted blocks to CPU memory or NVMe (vLLM native offloading or LMCache), so returning sessions load their history instead of prefilling it.")
    memory = _remedy("More KV memory per replica", "Raise --gpu-memory-utilization, use FP8/NVFP4 weights, or add GPUs (tensor parallel) so more memory is left for the cache.")
    disagg = _remedy("Disaggregate prefill and decode", "Separate prefill and decode instances (vLLM P/D with a KV connector) so long prompts stop stalling generation and each side scales on its own.")
    replicas = _remedy("Scale out", "Add data-parallel replicas behind prefix- or session-aware routing; one replica's GPUs are fully used.")
    spec = _remedy("Speculative decoding", "EAGLE or MTP drafts raise tokens per step when the batch is memory-bandwidth bound; gains shrink at large batches.")
    out = {
        "kv_capacity": [*([] if fp8_kv else [kv_fp8]), memory,
                        _remedy("Shorter contexts per replica", "A lower context cap or max-model-len per user class leaves room for more running sessions."),
                        *([] if offloading else [offload])],
        "cache_eviction": [*([] if offloading else [offload]), *([] if fp8_kv else [kv_fp8]), memory,
                           _remedy("Session-aware routing", "With several replicas, route a session to the replica that holds its history.")],
        "cache_miss": [*([_remedy("Enable prefix caching", "vLLM reports prefix caching disabled; enable it (--enable-prefix-caching).")] if prefix_off else []),
                       _remedy("Keep history byte-identical", "The chat template must render earlier turns the same way each time (e.g. not dropping or re-rendering reasoning), and system prompts must not change between turns."),
                       _remedy("Session-aware routing", "A load balancer that spreads one session across replicas loses its cached history; route by session or prefix.")],
        "batch_slots": [_remedy("Raise --max-num-seqs", "The KV cache has room for more running requests; check output speed per user stays above target as the batch grows."),
                        _remedy("Raise --max-num-batched-tokens with it", "A larger step token budget keeps more running requests progressing per step.")],
        "scheduler": [_remedy("Raise --max-num-batched-tokens", "The per-step token budget admits too little prompt work; a larger budget admits waiting prefills sooner."),
                      _remedy("Check serving overhead", "Idle GPUs with a queue can also mean a CPU-bound API server or scheduler: try more API server processes and async scheduling.")],
        "compute": [*([spec] if signals and not signals.get("prefill_heavy") else []), replicas,
                    _remedy("Quantize weights", "FP8 or NVFP4 weights raise compute throughput on B300."),
                    *([disagg] if signals and signals.get("prefill_heavy") else [])],
        "queue": [_remedy("Raise --max-num-batched-tokens", "If GPUs turn out idle, the per-step token budget admits too little work."),
                  replicas],
        "prefill": [disagg,
                    _remedy("Raise --max-num-batched-tokens", "Larger prefill chunks finish long prompts in fewer steps, at some cost to output speed of running requests."),
                    _remedy("Reuse more of the prompt", "Check prefix-cache hits; shared system prompts and session histories should not be prefilled again."),
                    _remedy("More GPUs per replica", "Tensor parallelism splits prompt processing; FP8/NVFP4 weights speed it up.")],
        "prefill_interference": [disagg,
                                 _remedy("Lower --max-num-batched-tokens", "Smaller prefill chunks stall decoding requests less per step, at some cost to first-token time."),
                                 _remedy("Limit concurrent long prompts", "Admission control or a separate pool for agent traffic keeps chat output speed stable.")],
        "decode": [spec, *([] if fp8_kv else [kv_fp8]),
                   _remedy("Quantize weights", "FP8 or NVFP4 weights cut the bytes read per decode step."),
                   _remedy("Cap the batch", "A lower --max-num-seqs protects output speed per user at the cost of users per replica; add replicas for the rest."),
                   _remedy("More GPUs per replica", "Tensor parallelism adds memory bandwidth per step.")],
        "client": [_remedy("Larger client", "Run the benchmark from a machine with more CPU or closer to the endpoint; the result is a lower bound.")],
        "errors": [_remedy("Check the error examples", "Timeouts under load point to queueing; HTTP errors to gateway limits or crashes. Compare the server logs for the measured window.")],
        "prefill_or_queue": [_remedy("Map the deployment to B300 metrics", "With vLLM telemetry the diagnosis tells queueing, KV pressure, cache eviction and prefill apart."), disagg],
        "unknown": [],
    }
    return out[constraint]


# --- Server signals over a window --------------------------------------------------------

def _combine(values, how):
    if how == "sum":
        return sum(values)
    if how == "max":
        return max(values)
    return sum(values) / len(values)


def window_signals(telemetry, start, end):
    """vLLM and GPU signals while a level was measured, or None without aligned telemetry.

    Rates and rolling p95s skip their first minute when the window is long enough,
    so they describe the level rather than the one before it."""
    if not isinstance(telemetry, dict) or not telemetry.get("metrics") or start is None or end is None:
        return None
    shift, aligned = alignment_shift(telemetry)
    skip = ROLLING_SECONDS if end - start > 2 * ROLLING_SECONDS else 0
    metrics, timelines = {}, {}
    for metric in telemetry["metrics"]:
        key = metric.get("id")
        if not key or metric.get("status") != "collected":
            continue
        begin = start + (skip if metric.get("kind") in ("rate", "histogram") else 0)
        by_time = {}
        for series in metric.get("series", []):
            for stamp, value in series.get("values", []):
                if value is not None and begin <= stamp + shift <= end:
                    by_time.setdefault(stamp, []).append(value)
        if not by_time:
            continue
        timeline = {t: _combine(v, metric.get("combine", "sum")) for t, v in by_time.items()}
        values = list(timeline.values())
        metrics[key] = {"mean": sum(values) / len(values), "max": max(values), "min": min(values),
                        "samples": len(values)}
        timelines[key] = timeline
    if not metrics:
        return None
    signals = {"metrics": metrics, "aligned": aligned}
    running, waiting = timelines.get("running"), timelines.get("waiting")
    if running and waiting:
        ceiling = max(running.values())
        stamps = [t for t in running if t in waiting]
        if ceiling > 0 and stamps:
            signals["running_at_ceiling"] = sum(running[t] >= 0.97 * ceiling and waiting[t] > 0 for t in stamps) / len(stamps)
    return signals


def kv_capacity_tokens(config):
    """Tokens the KV cache holds, summed over instances, from vLLM's cache_config_info."""
    total = 0
    for c in config or []:
        try:
            total += int(float(c["num_gpu_blocks"])) * int(float(c["block_size"]))
        except (KeyError, TypeError, ValueError):
            return None
    return total or None


# --- Diagnosis ---------------------------------------------------------------------------

def _pct(value):
    return f"{value:.0%}"


def _metric(signals, key, field="mean"):
    item = (signals or {}).get("metrics", {}).get(key)
    return item.get(field) if item else None


def needs_diagnosis(level):
    return not level.get("cancelled") and (not level.get("passed") or bool(level.get("hard_limit")))


def diagnose(level, signals=None, config=None):
    """The constraint that limits a level, with the evidence that names it and remedies."""
    limiting = (level.get("limiting") or {}).get("factor")
    requests = level.get("requests") or 0
    failed = sum((c.get("errors") or 0) + (c.get("incomplete") or 0) for c in (level.get("classes") or {}).values())
    failure_share = failed / requests if requests else 1.0
    server = bool(signals and signals.get("aligned") and signals.get("metrics"))
    evidence, found = [], []

    expected = level.get("expected_prefix_reuse")
    observed, source = None, None
    if server and _metric(signals, "prefix_hit") is not None:
        observed, source = _metric(signals, "prefix_hit"), "server"
    elif (level.get("cache_metrics") or {}).get("cache_hit_fraction") is not None:
        observed, source = level["cache_metrics"]["cache_hit_fraction"], "client"
    reuse_lost = (expected is not None and observed is not None and expected >= MIN_EXPECTED_REUSE
                  and observed < EVICTION_SHARE * expected)
    if expected is not None and observed is not None:
        evidence.append(f"Prefix cache hits {_pct(observed)} ({source}) vs {_pct(expected)} the session histories allow")

    output = _metric(signals, "output") if server else None
    prompt = _metric(signals, "input") if server else None
    if output is None:
        output, prompt = level.get("output_tokens_per_sec"), level.get("input_tokens_per_sec")
    # Without an observed hit rate, assume the cache kept what the sessions allow.
    hit = observed if observed is not None else expected or 0.0
    prefill_ratio = prompt * (1 - hit) / output if output and prompt is not None else None
    prefill_heavy = prefill_ratio is not None and prefill_ratio >= PREFILL_HEAVY
    if prefill_ratio is not None:
        evidence.append(f"{prefill_ratio:,.1f} uncached prompt tokens per generated token"
                        + (" (prompt-heavy)" if prefill_heavy else ""))

    working = (level.get("working_set_tokens") or {}).get("mean")
    capacity = kv_capacity_tokens(config)
    if working and capacity:
        evidence.append(f"Session histories ≈ {working:,.0f} tokens vs KV cache capacity {capacity:,} tokens")

    if level.get("client_limited"):
        found.append("client")
        evidence.insert(0, f"Dispatch lag p99 {(level.get('dispatch_lag') or {}).get('p99_ms') or 0:,.0f} ms")
    if failure_share > FAILURE_SHARE:
        evidence.append(f"{_pct(failure_share)} of measured requests failed or were incomplete")

    if server:
        kv_max, kv_mean = _metric(signals, "kv_cache", "max"), _metric(signals, "kv_cache")
        preempt = _metric(signals, "preemptions", "max") or 0
        waiting_mean, waiting_max = _metric(signals, "waiting") or 0, _metric(signals, "waiting", "max") or 0
        running_max = _metric(signals, "running", "max")
        flat = signals.get("running_at_ceiling") or 0
        # SM activity, not whole-GPU busy time, which reads 100% with any kernel running.
        sm = _metric(signals, "sm_active")
        queued = waiting_max >= 1 and waiting_mean >= 0.5
        kv_full = (kv_max is not None and kv_max >= KV_FULL) or preempt > 0
        kv_high = kv_max is not None and kv_max >= KV_HIGH
        if kv_max is not None:
            evidence.append(f"KV cache occupancy max {_pct(kv_max)}, mean {_pct(kv_mean)}")
        if preempt > 0:
            evidence.append(f"Preemptions up to {preempt:,.2f}/s")
        evidence.append(f"Waiting requests mean {waiting_mean:,.1f}, max {waiting_max:,.0f}"
                        + (f"; running up to {running_max:,.0f}" if running_max is not None else ""))
        if flat:
            evidence.append(f"Running requests at their ceiling with a queue for {_pct(flat)} of the window")
        gpu = [f"{label} {_pct(v)}" for label, key in (("SM", "sm_active"), ("tensor", "tensor_active"),
                                                        ("memory bandwidth", "dram_active"), ("memory used", "gpu_memory"))
               if (v := _metric(signals, key)) is not None]
        if gpu:
            evidence.append("GPU activity: " + ", ".join(gpu))
        parts = [f"{label} {v:,.2f} s" for label, key in (("queue", "queue"), ("prefill", "prefill"), ("first token", "ttft"))
                 if (v := _metric(signals, key)) is not None]
        if (tpot := _metric(signals, "tpot")) is not None:
            parts.append(f"time per token {tpot * 1000:,.0f} ms")
        if parts:
            evidence.append("Server p95 (rolling): " + ", ".join(parts))
        busy = sm is not None and sm >= BUSY
        idle = sm is not None and sm < IDLE
        if kv_full:
            found.append("kv_capacity")
        if reuse_lost:
            found.append("cache_eviction" if kv_high or kv_full or (working and capacity and working > capacity) else "cache_miss")
        if queued and flat >= FLAT_SHARE and not kv_high:
            found.append("batch_slots")
        if limiting == "tpot":
            found.append("prefill_interference" if prefill_heavy else "decode")
        if queued and not kv_full:
            found.append("scheduler" if idle else "prefill" if prefill_heavy else "compute" if busy else "queue")
        if limiting == "ttft":
            found.append("prefill")
        if limiting == "tpot" and busy:
            found.append("compute")
    else:
        if reuse_lost:
            found.append("cache_eviction" if working and capacity and working > capacity else "cache_miss")
        if limiting == "tpot":
            found.append("prefill_interference" if prefill_heavy else "decode")
        elif limiting == "ttft":
            found.append("prefill_or_queue")
    if failure_share > FAILURE_SHARE:
        # Errors under load are usually timeouts of a queue; they name the constraint only on their own.
        found.append("errors")
    found = list(dict.fromkeys(found)) or ["unknown"]
    main = found[0]
    return {"revision": REVISION, "constraint": main, "label": LABELS[main], "summary": SUMMARY[main],
            "basis": "server" if server else "client",
            "also": [{"constraint": c, "label": LABELS[c]} for c in found[1:]],
            "evidence": evidence, "remedies": remedies(main, config, {"prefill_heavy": prefill_heavy}),
            "kv_capacity_tokens": capacity}
