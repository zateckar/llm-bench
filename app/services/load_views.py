"""Open-loop load reports: sustainable rate, per-step and per-class SLO results, drift."""

import json

from app.benchmarking.load_workload import PRESET_LABELS


def is_open_loop(perf):
    return isinstance(perf, dict) and perf.get("schema_version") == 5 and perf.get("kind") == "open_loop"


STATUS_TEXT = {
    "established": "Established: a higher tested rate missed the SLO",
    "lower_bound": "Lower bound: every higher tested rate passed or was inconclusive",
    "not_met": "No tested rate met the SLO",
    "not_measured": "No step finished",
}
REASON_TEXT = {
    "dropped": "dropped at cap", "error": "errors", "incomplete": "empty/filtered",
    "ttft": "first token slow", "ttft_unmeasured": "first token unmeasured",
    "tpot": "token time slow", "tpot_unmeasured": "token time unmeasured", "e2e": "whole answer slow",
}


def _slo_text(slo):
    from app.services.html_reports import duration

    parts = [f"first token ≤ {duration(slo.get('ttft_ms'))}"]
    if slo.get("tpot_ms"):
        parts.append(f"token time ≤ {slo['tpot_ms']:g} ms")
    if slo.get("e2e_ms"):
        parts.append(f"answer ≤ {duration(slo['e2e_ms'])}")
    return " · ".join(parts)


def _reasons(counts):
    ordered = sorted((counts or {}).items(), key=lambda item: -item[1])
    return ", ".join(f"{REASON_TEXT.get(k, k)} {v:,}" for k, v in ordered[:3]) or "—"


def _step_status(step):
    if step.get("cancelled"):
        return "stopped"
    if step.get("passed"):
        return "pass"
    if step.get("conclusive_failure"):
        return "fail"
    return "inconclusive (client-limited)"


def _mean(buckets):
    total = sum(b[2] for b in buckets)
    return sum((b[0] + b[1]) / 2 * b[2] for b in buckets) / total if total else None


def run_view(run):
    from app.services.html_reports import COLORS, at, duration, fmt, line_chart

    perf = run["perf"]
    protocol = perf.get("protocol", {})
    summary = perf.get("summary", {})
    steps = [s for s in perf.get("steps", []) if isinstance(s, dict)]
    classes = (protocol.get("workload") or {}).get("classes", [])
    weight = sum(c.get("weight", 0) for c in classes) or 1
    best = next((s for s in steps if s.get("offered_rate") == summary.get("sustainable_rate")
                 and s.get("passed")), None)
    status = summary.get("sustainable_status", "not_measured")
    rate_text = fmt(summary.get("sustainable_rate"), "req/s")
    metrics = [
        {"label": "Sustainable arrival rate", "value": ("≥ " if status == "lower_bound" else "") + rate_text
         if summary.get("sustainable_rate") is not None else "Not met",
         "detail": STATUS_TEXT.get(status, status)},
        {"label": "Goodput at that rate", "value": fmt(summary.get("goodput_at_sustainable"), "req/s"),
         "detail": "Requests meeting their SLO per second"},
        {"label": "SLO attainment", "value": fmt(at(best, "attainment"), "%"),
         "detail": f"Target {fmt(protocol.get('attainment_target'), '%')} of all arrivals, per class"},
        {"label": "First token · p95", "value": duration(at(best, "ttft.p95_ms")),
         "detail": "User-visible, including dispatch lag, at the sustainable rate"},
    ]
    step_rows, class_rows = [], []
    for step in steps:
        step_rows.append({
            "rate": fmt(step.get("offered_rate"), "req/s"),
            "achieved": fmt(step.get("achieved_arrival_rate"), "req/s"),
            "arrivals": fmt(step.get("arrivals"), "int"),
            "attainment": fmt(step.get("attainment"), "%"),
            "goodput": fmt(step.get("goodput_rps"), "req/s"),
            "ttft": duration(at(step, "ttft.p95_ms")),
            "latency": duration(at(step, "latency.p95_ms")),
            "tpot": fmt(at(step, "output_token_time.p95_ms"), "ms"),
            "errors": fmt(step.get("errors"), "int"),
            "dropped": fmt(step.get("dropped"), "int"),
            "lag": duration(at(step, "dispatch_lag.p99_ms")),
            "in_flight": f"{fmt(step.get('mean_in_flight'))} / {fmt(step.get('max_in_flight'), 'int')}",
            "output": fmt(step.get("output_tokens_per_sec"), "tok/s"),
            "cache": fmt(at(step, "cache_metrics.cache_hit_fraction"), "%"),
            "status": _step_status(step),
            "passed": bool(step.get("passed")),
            "reasons": _reasons(step.get("failure_reasons")),
            "errors_text": " · ".join(step.get("error_examples") or []),
            "samples": f"{len((step.get('samples') or {}).get('t', [])):,} of {step.get('samples_total', 0):,} retained",
        })
        for name, item in (step.get("classes") or {}).items():
            class_rows.append({
                "rate": fmt(step.get("offered_rate"), "req/s"), "name": name,
                "arrivals": fmt(item.get("arrivals"), "int"),
                "attainment": fmt(item.get("attainment"), "%"),
                "meets": bool(item.get("meets_target")),
                "ttft": duration(at(item, "ttft.p95_ms")),
                "tpot": fmt(at(item, "output_token_time.p95_ms"), "ms"),
                "latency": duration(at(item, "latency.p95_ms")),
                "output": fmt(at(item, "output_tokens.mean"), "tokens"),
                "reasons": _reasons(item.get("failure_reasons")),
                "slo": _slo_text(item.get("slo") or {}),
            })
    rates = [s.get("offered_rate") for s in steps]
    charts = []
    for title, unit, key, description in (
        ("SLO attainment by offered rate", "%", "attainment", "Share of all arrivals that met their class SLO."),
        ("Goodput by offered rate", "req/s", "goodput_rps", "SLO-met requests per second; the offered line is perfect service."),
        ("First token p95 by offered rate", "ms", "ttft.p95_ms", "User-visible first-token time, dispatch lag included."),
    ):
        series = [{"label": run["label"], "color": COLORS[0],
                   "points": [(s.get("offered_rate"), at(s, key)) for s in steps]}]
        if key == "goodput_rps":
            series.append({"label": "Offered", "color": COLORS[2], "points": [(r, r) for r in rates]})
        if key == "attainment":
            target = protocol.get("attainment_target")
            series.append({"label": "Target", "color": COLORS[2], "points": [(r, target) for r in rates]})
        svg = line_chart(series, title, "Offered arrivals · req/s", unit, x_precision=2)
        if svg:
            charts.append({"title": title, "description": description, "svg": svg, "series": series})
    drift = []
    for step in steps:
        windows = step.get("windows") or []
        if not step.get("drift"):
            continue
        d = step["drift"]
        minutes = (step.get("step_seconds") or 0) >= 600
        scale, axis = (60, "Minutes into step") if minutes else (1, "Seconds into step")
        series_ttft = [{"label": "First token p95", "color": COLORS[1],
                        "points": [(w["start_s"] / scale, w.get("ttft_p95_ms")) for w in windows]}]
        series_att = [{"label": "Attainment", "color": COLORS[0],
                       "points": [(w["start_s"] / scale, w.get("attainment")) for w in windows]}]
        drift.append({
            "rate": fmt(step.get("offered_rate"), "req/s"),
            "degrading": d.get("degrading"),
            "text": (f"First-token p95 {duration(d.get('ttft_p95_first_ms'))} → {duration(d.get('ttft_p95_last_ms'))}"
                     f" (×{fmt(d.get('ttft_ratio'))}) · attainment {fmt(d.get('attainment_first'), '%')} → "
                     f"{fmt(d.get('attainment_last'), '%')} · {d.get('error_burst_windows', 0)} windows with error bursts"),
            "charts": [c for c in (
                {"title": f"First token p95 over time · {fmt(step.get('offered_rate'), 'req/s')}", "series": series_ttft,
                 "svg": line_chart(series_ttft, "First token p95 over time", axis, "ms", x_precision=1 if minutes else 0)},
                {"title": f"Attainment over time · {fmt(step.get('offered_rate'), 'req/s')}", "series": series_att,
                 "svg": line_chart(series_att, "Attainment over time", axis, "%", x_precision=1 if minutes else 0)},
            ) if c["svg"]],
        })
    arrival = ("Poisson arrivals" if protocol.get("arrival") == "poisson"
               else f"Bursty gamma arrivals, CV {fmt(protocol.get('burstiness'))}")
    return {
        "label": run["label"],
        "metrics": metrics,
        "status": status,
        "protocol": {
            "workload": PRESET_LABELS.get(protocol.get("preset"), "Custom workload"),
            "workload_hash": protocol.get("workload_hash", "n/a"),
            "arrival": arrival,
            "step": duration((protocol.get("step_seconds") or 0) * 1000),
            "seed": protocol.get("seed"),
            "cap": fmt(protocol.get("in_flight_cap"), "int"),
            "rates": ", ".join(f"{r:g}" for r in protocol.get("rates", [])),
            "timeout": duration((protocol.get("request_timeout") or 0) * 1000),
            "revision": protocol.get("revision", "n/a"),
            "expected": fmt(protocol.get("expected_arrivals"), "int"),
        },
        "classes": [{
            "name": c.get("name"), "share": fmt(c.get("weight", 0) / weight, "%"),
            "input": fmt(_mean(c.get("input", [])), "int"), "output": fmt(_mean(c.get("output", [])), "int"),
            "prefix": fmt(c.get("shared_prefix"), "int"), "slo": _slo_text(c.get("slo") or {}),
        } for c in classes],
        "steps": step_rows,
        "class_rows": class_rows,
        "charts": charts,
        "drift": drift,
        "stop_reason": perf.get("stop_reason", ""),
        "cancelled": perf.get("cancelled", False),
        "inconclusive": summary.get("inconclusive_rates") or [],
        "non_monotonic": summary.get("non_monotonic", False),
        "notes": perf.get("notes", []),
    }


def comparison_key(perf):
    protocol = perf.get("protocol", {})
    return json.dumps({**(protocol.get("comparison_key") or {}), "temperature": protocol.get("temperature"),
                       "reasoning_effort": protocol.get("reasoning_effort"), "revision": protocol.get("revision")},
                      sort_keys=True)


def comparison(runs):
    """Align steps by offered rate only when traffic, SLOs and sampling are identical."""
    from app.services.html_reports import at, duration, fmt

    loads = [r for r in runs if is_open_loop(r.get("perf"))]
    if len(loads) < 2:
        return None
    groups = {}
    for run in loads:
        groups.setdefault(comparison_key(run["perf"]), []).append(run)
    key, members = max(groups.items(), key=lambda item: len(item[1]))
    excluded = [r["label"] for r in loads if r not in members]
    if len(members) < 2:
        return {"runs": [], "rows": [], "excluded": excluded,
                "note": "No two open-loop runs share workload, arrivals, seed, step duration, SLOs and sampling settings."}
    rates = sorted({s["offered_rate"] for r in members for s in r["perf"].get("steps", []) if not s.get("cancelled")})
    rows = []
    for rate in rates:
        cells = []
        for run in members:
            step = next((s for s in run["perf"].get("steps", []) if s.get("offered_rate") == rate
                         and not s.get("cancelled")), None)
            cells.append(None if step is None else {
                "attainment": fmt(step.get("attainment"), "%"), "goodput": fmt(step.get("goodput_rps"), "req/s"),
                "ttft": duration(at(step, "ttft.p95_ms")), "status": _step_status(step), "passed": step.get("passed"),
            })
        rows.append({"rate": fmt(rate, "req/s"), "cells": cells})
    return {
        "runs": [{"label": r["label"], "sustainable": fmt(at(r["perf"], "summary.sustainable_rate"), "req/s"),
                  "status": at(r["perf"], "summary.sustainable_status")} for r in members],
        "rows": rows, "excluded": excluded,
        "note": "Same seeded arrivals, lengths, SLOs and sampling settings; differences come from the deployments.",
    }


def open_loop_views(runs):
    views = [run_view(r) for r in runs if is_open_loop(r.get("perf"))]
    return {"views": views, "comparison": comparison(runs)} if views else None
