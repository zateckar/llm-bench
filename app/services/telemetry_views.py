"""Saved B300 time series and benchmark phases, shared by online/offline reports."""


def measurement_phases(perf):
    phases = []

    def add(point, label):
        if point.get("started_at") is not None and point.get("ended_at") is not None:
            phases.append({**point, "label": label})

    for point in perf.get("concurrency", []):
        add(point, f"Load · c={point['concurrency']}")
    for pair in perf.get("cache_reuse", []):
        for mode in ("cold", "warm"):
            if pair.get(mode):
                add(pair[mode], f"{mode.title()} prefix · {pair['context_tokens']:,} tokens · c={pair['concurrency']}")
    for cell in perf.get("cells", []):
        label = f"{cell.get('effort', 'default')} · {cell['context_tokens']:,} tokens · c={cell['concurrency']}"
        add(cell, "Cold · " + label)
        if cell.get("cache_reuse"):
            add(cell["cache_reuse"], "Warm · " + label)
    for entry in perf.get("caps", []) if perf.get("kind") == "sessions" else []:
        for level in entry.get("levels", []):
            # The measured window, not the warmup.
            window = {**level, "started_at": level.get("window_started_at", level.get("started_at")),
                      "ended_at": level.get("window_ended_at", level.get("ended_at"))}
            search = " · saturation" if level.get("phase") == "saturation" else ""
            add(window, f"Users · {level['users']} · {entry.get('context_cap', 0):,}-token cap{search}")
    return sorted(phases, key=lambda p: p["started_at"])


def telemetry_view(run):
    from app.services.html_reports import COLORS, at, duration, fmt, line_chart, number
    from app.benchmarking.vllm_telemetry import alignment_shift

    perf = run.get("perf", {})
    data = perf.get("telemetry")
    if not isinstance(data, dict):
        return None
    shift, aligned = alignment_shift(data)
    scope = data.get("scope") or {}
    window = data.get("window") or {}
    origin = number(window.get("measurement_start")) or 0
    start, end = number(window.get("start")), number(window.get("end"))
    x_range = ((start-origin)/60, (end-origin)/60) if start is not None and end is not None and end >= start else None
    metrics = {m["id"]: m for m in data.get("metrics", [])}
    phases = measurement_phases(perf)
    series_by_metric = {}
    for key, metric in metrics.items():
        factor = 1000 if metric.get("unit") == "seconds" else 1
        series_by_metric[key] = [
            {"label": metric["title"] + " · " + str(s.get("labels", {}).get("instance", "deployment")),
             "color": COLORS[i % len(COLORS)],
             "points": [((stamp+shift-origin)/60, value*factor if value is not None else None) for stamp, value in s.get("values", [])]}
            for i, s in enumerate(metric.get("series", []))
        ]

    def client_series(key, title):
        points = []
        for phase in phases:
            value = number(at(phase, key))
            if value is None and key == "output_tokens_per_sec":
                value = number(phase.get("aggregate_tokens_per_sec"))
            start, end = (phase[k] - origin for k in ("started_at", "ended_at"))
            points.extend([(start/60, value), (end/60, value), (end/60, None)])
        return {"label": title, "color": COLORS[2], "points": points}

    charts = []
    for title, keys, unit, client in (
        ("Server and benchmark output", ("output",), "tok/s", ("output_tokens_per_sec", "Benchmark output · phase average")),
        ("Running and waiting requests", ("running", "waiting"), "requests", None),
        ("KV occupancy and prefix reuse", ("kv_cache", "prefix_hit"), "%", None),
        ("First token and client first delivery", ("ttft",), "ms", ("ttft.p95_ms", "Benchmark first delivery · phase p95")),
        ("Server queue, prefill and decode", ("queue", "prefill", "decode"), "ms", None),
        ("Server and client response time", ("latency",), "ms", ("latency.p95_ms", "Benchmark response · phase p95")),
        ("Preemptions", ("preemptions",), "events/s", None),
        ("Server prompt and generation", ("input", "output"), "tok/s", None),
        ("Server time per output token", ("tpot",), "ms", None),
        ("GPU activity and memory", ("sm_active", "tensor_active", "dram_active", "gpu_memory"), "%", None),
    ):
        series = []
        for i, key in enumerate(keys):
            for s in series_by_metric.get(key, []):
                series.append({**s, "color": COLORS[i % len(COLORS)]})
        if client:
            series.append(client_series(*client))
        series = [s for s in series if any(number(y) is not None for _, y in s["points"])]
        svg = line_chart(series, title, "Minutes from performance start", unit, x_range=x_range, x_precision=2)
        if svg:
            charts.append({"title": title, "svg": svg, "series": series})

    offset = number(data.get("api_clock_offset_seconds"))
    rows = []
    for phase in phases[:120]:
        start, end = phase["started_at"], phase["ended_at"]

        def samples(key):
            by_time = {}
            for s in metrics.get(key, {}).get("series", []):
                for t, v in s.get("values", []):
                    if start <= t+shift <= end and number(v) is not None:
                        by_time.setdefault(t, []).append(v)
            return [(max(values) if key == "kv_cache" else sum(values)) for values in by_time.values()]

        waiting, kv, output, preemptions = (samples(k) for k in ("waiting", "kv_cache", "output", "preemptions"))
        enough = aligned and len(waiting) >= 3
        signals = []
        if not aligned:
            signals.append("Clock alignment uncertain")
        elif not enough:
            signals.append("Insufficient queue samples")
        else:
            if max(waiting) > 0:
                signals.append("Queueing observed")
            if kv and max(kv) >= 0.9:
                signals.append("High KV occupancy")
            if preemptions and max(preemptions) > 0:
                signals.append("Rolling preemptions present")
        if end-start < 60:
            signals.append("Phase shorter than rate window")
        rows.append({"label": phase["label"], "duration": duration((end-start)*1000),
                     "client_output": fmt(phase.get("output_tokens_per_sec", phase.get("aggregate_tokens_per_sec")), "tok/s"),
                     "client_ttft": duration(at(phase, "ttft.p95_ms")), "samples": len(waiting),
                     "waiting": fmt(max(waiting) if enough else None, "int"),
                     "kv": fmt(max(kv) if enough and kv else None, "%"),
                     "server_output": fmt(sum(output)/len(output) if enough and output else None, "tok/s"),
                     "signals": "; ".join(signals) or "No queue/cache signal observed"})
    return {"label": run.get("label", "Run"), "scope": scope, "status": data.get("status", "unavailable"),
            "warnings": data.get("warnings", []), "charts": charts, "rows": rows,
            "phase_count": len(phases), "offset": fmt(offset, "s"), "shift": fmt(shift, "s"), "step": window.get("step_seconds"),
            "missing": [m["title"] for m in data.get("metrics", []) if m.get("status") != "collected"]}
