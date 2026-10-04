"""Align measured performance points by workload for run-column comparisons."""

from collections import OrderedDict


METRICS = [
    {"key": "ttft_p50", "label": "First token · p50", "unit": "ms"},
    {"key": "ttft_p95", "label": "First token · p95", "unit": "ms"},
    {"key": "ttft_p99", "label": "First token · p99", "unit": "ms"},
    {"key": "latency_p50", "label": "Whole answer · p50", "unit": "ms"},
    {"key": "latency_p95", "label": "Whole answer · p95", "unit": "ms"},
    {"key": "latency_p99", "label": "Whole answer · p99", "unit": "ms"},
    {"key": "aggregate_output", "label": "Aggregate output", "unit": "tok/s"},
    {"key": "request_rate", "label": "Successful request rate", "unit": "req/s"},
    {"key": "request_output", "label": "Per-request output · p50", "unit": "tok/s"},
    {"key": "stream_output", "label": "Stream output · p50", "unit": "tok/s"},
    {"key": "error_rate", "label": "Request error rate", "unit": "%"},
    {"key": "cache_hit", "label": "Reported cache hit", "unit": "%"},
    {"key": "uncached_prefill", "label": "Uncached prefill · p50", "unit": "tok/s"},
    {"key": "effective_input", "label": "Effective input · p50", "unit": "tok/s"},
    {"key": "decode", "label": "Decode delivery · p50", "unit": "tok/s"},
]


def comparison_point(point, *, sweep):
    from app.services.html_reports import at, duration, fmt, number

    requests, errors = number(point.get("requests")), number(point.get("errors"))
    complete = point.get("completed") if sweep else (
        requests - errors if requests is not None and errors is not None else None
    )
    telemetry = point.get("cache_metrics", {})
    values = {
        **{f"{kind}_{q}": number(at(point, f"{path}.{q}_ms"))
           for kind, path in (("ttft", "ttft"), ("latency", "latency"))
           for q in ("p50", "p95", "p99")},
        "aggregate_output": number(point.get("aggregate_tokens_per_sec" if sweep else "output_tokens_per_sec")),
        "request_rate": number(point.get("requests_per_sec")),
        "request_output": number(at(point, "request_tokens_per_sec.p50")),
        "stream_output": number(at(point, "generation_tokens_per_sec.p50")),
        "error_rate": errors / requests if requests and errors is not None else None,
        "cache_hit": number(telemetry.get("cache_hit_fraction")),
        "uncached_prefill": number(at(telemetry, "uncached_prefill_tokens_per_sec.p50")),
        "effective_input": number(at(telemetry, "effective_prompt_tokens_per_sec.p50")),
        "decode": number(at(telemetry, "decode_tokens_per_sec.p50")),
    }
    status = point.get("status") or ("measured" if complete else "failed")
    return {
        "status": status,
        "error": point.get("error", ""),
        "requests": requests,
        "completed": complete,
        "errors": errors,
        "incomplete": point.get("incomplete") if sweep else None,
        "values": values,
        "formatted": {m["key"]: duration(values[m["key"]]) if m["unit"] == "ms"
                      else fmt(values[m["key"]], m["unit"]) for m in METRICS},
        "samples": f"{fmt(complete, 'int')} / {fmt(requests, 'int')} "
                   f"{'complete answers' if sweep else 'successful requests'}",
        "failures": f"{fmt(errors, 'int')} errors" + (
            f" · {fmt(point.get('incomplete'), 'int')} incomplete" if sweep else ""
        ),
    }


def performance_comparison(runs):
    """Keep different protocols, budgets, sampling settings and prefix modes distinct.

    Revisions can share a workload (e.g. different sweep grids), but remain
    visible as provenance. Only actual saved points become comparison rows.
    """
    if len(runs) < 2:
        return None
    from app.services.html_reports import fmt, number, points

    groups = OrderedDict()
    provenance = []
    for run_index, run in enumerate(runs):
        perf = run.get("perf", {})
        protocol = perf.get("protocol", {})
        is_sweep = perf.get("schema_version") == 4 and perf.get("kind") == "context_sweep"
        is_fixed = perf.get("schema_version") == 3
        provenance.append({
            "label": run["label"],
            "revision": protocol.get("revision", "Not recorded"),
            "status": run.get("status", "Not recorded"),
            "partial": bool(perf.get("cancelled") or perf.get("stop_error")
                            or (is_sweep and not perf.get("finished"))),
        })
        if not (is_sweep or is_fixed):
            continue

        def add(kind, effort, prefix, context, concurrency, point):
            if number(context) is None or number(concurrency) is None:
                return
            # Missing settings stay distinct from a known setting, including zero.
            output = protocol.get("max_output_tokens")
            temperature = protocol.get("temperature")
            tokenizer = protocol.get("reference_tokenizer", "Not recorded")
            key = (kind, effort, prefix, output, temperature, tokenizer)
            if key not in groups:
                label = (f"{kind} · {'Provider default' if effort == 'default' else effort} · "
                         f"{prefix} · {fmt(output, 'int')} output limit · "
                         f"temperature {fmt(temperature)} · {tokenizer}")
                groups[key] = {"label": label, "points": {}, "rounds": {}}
            group = groups[key]
            group["points"].setdefault((context, concurrency), [None] * len(runs))[run_index] = (
                comparison_point(point, sweep=is_sweep)
            )
            group["rounds"][run_index] = (protocol.get("rounds_per_cell") if is_sweep else
                                         protocol.get("rounds_per_level") if kind == "Fixed workload" else None)

        if is_sweep:
            for cell in perf.get("cells", []):
                effort = cell.get("effort", "default")
                context, concurrency = cell.get("context_tokens"), cell.get("concurrency")
                add("Context sweep", effort, "cold", context, concurrency, cell)
                warm = cell.get("cache_reuse")
                if warm:
                    add("Context sweep", effort, "warm", context, concurrency, warm)
        else:
            effort = protocol.get("reasoning_effort") or "default"
            for point in points(perf, "concurrency"):
                add("Fixed workload", effort, "cold", protocol.get("input_reference_tokens"),
                    point.get("concurrency"), point)
            for pair in perf.get("cache_reuse", []):
                for prefix in ("cold", "warm"):
                    point = pair.get(prefix)
                    if point:
                        add("Prefix cache", pair.get("effort", effort), prefix,
                            pair.get("context_tokens"), pair.get("concurrency"), point)

    workloads = []
    for i, group in enumerate(groups.values()):
        rows = [{"context": context, "concurrency": concurrency, "points": aligned}
                for (context, concurrency), aligned in sorted(group["points"].items())]
        # Avoid selectors for metrics that no run reports for this workload.
        metrics = [m for m in METRICS if any(
            point is not None and point["values"][m["key"]] is not None
            for row in rows for point in row["points"]
        )]
        workloads.append({
            "id": str(i), "label": group["label"], "rows": rows,
            "contexts": sorted({row["context"] for row in rows}),
            "concurrencies": sorted({row["concurrency"] for row in rows}),
            "metrics": metrics or METRICS[:1],
            "rounds": [group["rounds"].get(j) for j in range(len(runs))],
        })
    if not workloads:
        return None
    return {"runs": provenance, "workloads": workloads}
