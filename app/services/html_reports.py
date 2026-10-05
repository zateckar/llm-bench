"""Offline HTML reports and the performance view shared with online comparisons.

Charts are inline SVG: downloads need neither a server nor JavaScript/CDNs.
Only presentation fields are rendered; model credentials and endpoint URLs are
never part of the report context exposed by the templates.
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
import math

from markupsafe import Markup, escape

from app.database import fetch_all, fetch_one
from app.templates_config import templates
from app.benchmarking.paired_stats import holm, verdict
from app.benchmarking.quality_report import paired_comparison, rescore_report, achievement_score

COLORS = ("#3b82f6", "#14b8a6", "#e99b16", "#e879ad", "#9b87f5", "#ef744d")


def object_json(raw):
    try:
        value = json.loads(raw or "null")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def _pretty(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def native_turns(transcript):
    """Assistant text, native tool calls and simulated tool results per turn."""
    turns = []
    for t in transcript:
        if not isinstance(t, dict):
            continue
        number_ = t.get("turn")
        if "error" in t:
            status = t.get("http_status")
            turns.append({"role": f"request rejected (HTTP {status})" if status else "request failed",
                          "number": number_, "content": str(t["error"]), "native": True})
            continue
        if t.get("content"):
            turns.append({"role": f"assistant · finish {t.get('finish_reason') or 'n/a'}",
                          "number": number_, "content": t["content"], "native": True})
        for call in t.get("tool_calls") or []:
            if isinstance(call, dict):
                turns.append({"role": f"tool call {call.get('name') or '(no name)'} · id {call.get('id') or 'missing'}",
                              "number": number_, "content": str(call.get("arguments", "")), "native": True})
        for reply in t.get("tool_results") or []:
            if isinstance(reply, dict):
                turns.append({"role": f"tool result {reply.get('name') or ''}".strip(),
                              "number": number_, "content": _pretty(reply.get("reply")), "native": True})
        if t.get("wire_defects"):
            turns.append({"role": "wire defects", "number": number_,
                          "content": _pretty(t["wire_defects"]), "native": True})
        if not t.get("content") and not t.get("tool_calls"):
            turns.append({"role": f"assistant · finish {t.get('finish_reason') or 'n/a'}",
                          "number": number_, "content": "(no content)", "native": True})
    return turns


def interactive_turns(result):
    """Display stored actions separately from the enclosing transcript JSON."""
    meta = object_json(result.get("quality_metadata_json"))
    protocol = meta.get("metadata", {}).get("protocol")
    if protocol == "native-tools-v1":
        transcript = meta.get("diagnostics", {}).get("transcript", [])
        return native_turns(transcript) if isinstance(transcript, list) else []
    if protocol != "json-actions-v1":
        return []
    transcript = meta.get("diagnostics", {}).get("transcript", [])
    if not isinstance(transcript, list):
        return []
    turns, number = [], 0
    for t in transcript:
        if (
            not isinstance(t, dict)
            or t.get("role") not in {"assistant", "tool"}
            or "content" not in t
        ):
            continue
        number += t["role"] == "assistant"
        turns.append(
            {
                "role": t["role"],
                "number": number,
                "content": t["content"]
                if isinstance(t["content"], str)
                else json.dumps(t["content"], ensure_ascii=False, indent=2),
            }
        )
    return turns


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def at(value, path):
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def fmt(value, unit=""):
    value = number(value)
    if value is None:
        return "n/a"
    if unit == "%":
        return f"{value * 100:,.1f}%"
    if unit == "$":
        return f"${value:,.6f}"
    if unit == "int":
        return f"{value:,.0f}"
    if unit == "req/s":
        return f"{value:,.3f} req/s"
    return f"{value:,.1f}" + (f" {unit}" if unit else "")


def duration(value):
    """Compact client-observed timings; stored measurements remain milliseconds."""
    value = number(value)
    if value is None:
        return "n/a"
    if value >= 60_000:
        return f"{value / 60_000:,.2f} min"
    if value >= 1000:
        return f"{value / 1000:,.2f} s"
    return f"{value:,.0f} ms"


def points(perf, key):
    raw = perf.get(key)
    return [p for p in raw if isinstance(p, dict)] if isinstance(raw, list) else []


async def load_run(run_id):
    if not 0 < run_id <= 2**63 - 1:
        return None
    run = await fetch_one(
        """SELECT tr.*, m.name AS model_name, m.model_id AS model_identifier
           FROM test_runs tr JOIN models m ON tr.model_id = m.id WHERE tr.id = ?""",
        (run_id,),
    )
    if not run:
        return None
    # The templates historically call the provider's identifier model_id.
    run["model_id"] = run.pop("model_identifier")
    run["label"] = f"#{run['id']} · {run['model_name']}"
    run["quality"] = object_json(run.get("quality_json"))
    if run["quality"].get("schema_version") != 3:
        run["quality"] = {}
    else:
        run["quality"] = rescore_report(run["quality"])
    run["perf"] = object_json(run.get("perf_json"))
    from app.services.sweep_reports import hydrate_sweep
    run["perf"] = await hydrate_sweep(run_id, run["perf"])
    run["results"] = await fetch_all(
        "SELECT * FROM test_results WHERE run_id = ? ORDER BY category, question_index, id",
        (run_id,),
    )
    groups = defaultdict(list)
    for result in run["results"]:
        result["interactive_turns"] = interactive_turns(result)
        result["scored"] = bool(
            result["quality_scored"]
            if result.get("quality_scored") is not None
            else result.get("request_ok", 1)
        )
        metadata = object_json(result.get("quality_metadata_json"))
        result["evaluation"] = metadata.get("evaluation")
        result["score"] = achievement_score({**result, "evaluation": result["evaluation"]})
        result["attempt_diagnostics"] = metadata.get("metrics", {}).get("attempt_diagnostics", [])
        result["quality_requests"] = metadata.get("diagnostics", {}).get("quality_requests", [])
        result["outcome"] = metadata.get("outcome") or (
            "excluded" if not result["scored"] else "pass" if result["passed"] else "task_failure"
        )
        groups[result["category"]].append(result)
    run["categories"] = {}
    for name, items in groups.items():
        scored = [r for r in items if r["scored"]]
        run["categories"][name] = {
            "total": len(items),
            "scored": len(scored),
            "passed": sum(bool(r["passed"]) for r in scored),
            "avg_score": sum(r["score"] for r in scored) / len(scored) if scored else None,
        }
    # Use the saved answers for coverage, including legacy and in-flight runs.
    run["recorded_questions"] = len(run["results"])
    run["scored_questions"] = sum(r["scored"] for r in run["results"])
    run["passed_questions"] = sum(bool(r["passed"]) for r in run["results"] if r["scored"])
    scored = [r for r in run["results"] if r["scored"]]
    run["avg_score"] = sum(r["score"] for r in scored) / len(scored) if scored else 0
    if run["quality"].get("schema_version") == 3:
        run["avg_score"] = run["quality"]["summary"]["category_balanced"] or 0.0
        for name, summary in run["quality"]["summary"]["categories"].items():
            if name in run["categories"] and run["categories"][name]["scored"]:
                run["categories"][name]["avg_score"] = summary["score"]
    return run


def _adjust_for_multiplicity(comparisons):
    """Holm-adjust the baseline-versus-each p-values shown together."""
    compatible = [c for c in comparisons if c.get("compatible")]
    adjusted = holm([c.get("p_value") for c in compatible])
    for comparison, value in zip(compatible, adjusted):
        comparison["p_adjusted"] = value
        comparison["verdict"] = verdict(comparison["balanced_difference"], comparison["ci95"], value)
        comparison["adjustment"] = "Holm" if len(compatible) > 1 else None


def comparison_context(runs):
    comparisons = []
    if len(runs) > 1:
        for run in runs[1:]:
            if runs[0]["quality"] and run["quality"]:
                comparisons.append(
                    {
                        "left": runs[0]["id"],
                        "right": run["id"],
                        "comparison": paired_comparison(runs[0]["quality"], run["quality"]),
                    }
                )
    _adjust_for_multiplicity([row["comparison"] for row in comparisons])
    categories = sorted({name for run in runs for name in run["categories"]})
    indexed = {
        run["id"]: {(r["category"], r["test_id"]): r for r in run["results"]} for run in runs
    }
    details = []
    for category in categories:
        ids = sorted(
            {r["test_id"] for run in runs for r in run["results"] if r["category"] == category}
        )
        rows = []
        for test_id in ids:
            row = {"test_id": test_id}
            for run in runs:
                result = indexed[run["id"]].get((category, test_id), {})
                row[f"run_{run['id']}_score"] = (
                    result.get("score") if result.get("scored") else None
                )
                row[f"run_{run['id']}_detail"] = result.get("detail", "")
            rows.append(row)
        details.append({"category": category, "results": rows})
    return {
        "quality_comparisons": comparisons,
        "comparison_data": details,
        "category_names": categories,
    }


def svg_text(x, y, text, extra=""):
    return f'<text x="{x:g}" y="{y:g}" {extra}>{escape(text)}</text>'


def line_chart(series, title, x_label, unit, *, percentile=False, x_range=None, x_precision=0):
    """A true numeric x axis; missing observations break the line."""
    valid = [
        (x, y)
        for s in series
        for x, y in s["points"]
        if number(x) is not None and number(y) is not None
    ]
    if not valid:
        return None
    xs = sorted({x for x, _ in valid})
    xmin, xmax = (0, 100) if percentile else (min(xs), max(xs))
    if x_range is not None:
        xmin, xmax = x_range
    maximum = max(y for _, y in valid)
    scale = (
        60_000
        if unit == "ms" and maximum >= 120_000
        else (1000 if unit == "ms" and maximum >= 1000 else 1)
    )
    axis_unit = "min" if scale == 60_000 else "s" if scale == 1000 else unit
    # Round the upper bound to readable ticks instead of e.g. 1.9k milliseconds.
    target = max(0.01 if unit in {"%", "req/s"} else 1 / scale, maximum / scale) * 1.08
    magnitude = 10 ** math.floor(math.log10(target / 4))
    step = next(s * magnitude for s in (1, 2, 2.5, 5, 10) if s * magnitude >= target / 4)
    ymax = 4 * step * scale
    if unit == "%":
        ymax = min(1, ymax)
    width, height = 560, 260
    left, top, pw, ph = 62, 24, 476, 174

    def cx(x):
        return left + (x - xmin) / (xmax - xmin) * pw if xmax > xmin else left + pw / 2

    def cy(y):
        return top + ph - y / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<title>{escape(title)}</title><g font-family="system-ui,sans-serif" font-size="16" fill="currentColor">',
    ]
    for i in range(5):
        value = ymax * i / 4
        y = cy(value)
        parts.append(f'<path d="M{left} {y:g}H{left + pw}" stroke="currentColor" opacity=".12"/>')
        label = (
            f"{value * 100:g}%"
            if unit == "%"
            else f"{value / scale:g}"
            if scale != 1
            else f"{value / 1000:g}k"
            if value >= 1000
            else f"{value:g}"
        )
        parts.append(svg_text(left - 12, y + 4, label, 'text-anchor="end"'))
    ticks = [xmin + (xmax-xmin)*i/6 for i in range(7)] if x_range is not None else (
        [0, 25, 50, 75, 100]
        if percentile
        else (xs if len(xs) <= 7 else [xs[round(i * (len(xs) - 1) / 6)] for i in range(7)])
    )
    for x in ticks:
        label = f"{x:,.{x_precision}f}"
        if x_precision:
            label = label.rstrip("0").rstrip(".")
        parts.append(svg_text(cx(x), top + ph + 24, label, 'text-anchor="middle"'))
    parts.append(svg_text(left + pw / 2, height - 12, x_label, 'text-anchor="middle"'))
    parts.append(svg_text(left, 14, axis_unit))
    for s in series:
        color = s["color"]
        previous = None
        for x, y in sorted(s["points"], key=lambda p: number(p[0]) or 0):
            if number(x) is None or number(y) is None:
                previous = None
                continue
            px, py = cx(x), cy(y)
            if previous:
                parts.append(
                    f'<path d="M{previous[0]:g} {previous[1]:g}L{px:g} {py:g}" fill="none" stroke="{color}" stroke-width="2.5"/>'
                )
            radius = 2 if len(s["points"]) > 40 else 4
            display = duration(y) if unit == "ms" else fmt(y, unit)
            parts.append(
                f'<circle cx="{px:g}" cy="{py:g}" r="{radius}" fill="{color}"><title>{escape(s["label"])}: {x:g}, {escape(display)}</title></circle>'
            )
            previous = (px, py)
    parts.append("</g></svg>")
    return Markup("".join(parts))


def bar_chart(runs, getter, title, unit="%"):
    values = [number(getter(run)) for run in runs]
    if all(v is None for v in values):
        return None
    maximum = 1 if unit == "%" else max(1, max(v for v in values if v is not None))
    height = 30 + len(runs) * 58
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 {height}" role="img" aria-label="{escape(title)}"><title>{escape(title)}</title><g font-family="system-ui,sans-serif" font-size="13" fill="currentColor">'
    ]
    for i, (run, value) in enumerate(zip(runs, values)):
        y = 20 + i * 58
        label = run["label"]
        out.append(svg_text(10, y, label[:68] + ("…" if len(label) > 68 else "")))
        out.append(
            f'<rect x="10" y="{y + 10}" width="555" height="12" rx="6" fill="currentColor" opacity=".08"/>'
        )
        if value is not None:
            out.append(
                f'<rect x="10" y="{y + 10}" width="{max(0, min(1, value / maximum)) * 555:g}" height="12" rx="6" fill="{COLORS[i % len(COLORS)]}"><title>{escape(label)}: {escape(fmt(value, unit))}</title></rect>'
            )
        out.append(svg_text(680, y + 21, fmt(value, unit), 'text-anchor="end"'))
    return Markup("".join(out) + "</g></svg>")


def load_performance_view(run):
    """Tie the headline timings to the same observed load as peak throughput."""
    perf = run["perf"]
    loads = sorted(
        (p for p in points(perf, "concurrency") if number(p.get("concurrency")) is not None),
        key=lambda p: p["concurrency"],
    )
    eligible = [
        p
        for p in loads
        if number(p.get("output_tokens_per_sec")) is not None
        and number(p.get("requests")) is not None
        and number(p.get("errors")) is not None
        and p["requests"] > p["errors"]
    ]
    peak = max(eligible, key=lambda p: p["output_tokens_per_sec"], default=None)
    counts_known = bool(loads) and all(
        number(p.get("requests")) is not None and number(p.get("errors")) is not None for p in loads
    )
    total = sum(p["requests"] for p in loads) if counts_known else None
    errors = sum(p["errors"] for p in loads) if counts_known else None
    successful = total - errors if counts_known else None
    level = fmt(peak["concurrency"], "int") if peak else None
    return {
        "label": run["label"],
        "levels": ", ".join(fmt(p["concurrency"], "int") for p in loads),
        "level_count": len(loads),
        "peak_concurrency": peak["concurrency"] if peak else None,
        "metrics": [
            {
                "label": "Peak aggregate output",
                "value": fmt(at(peak, "output_tokens_per_sec"), "tok/s"),
                "detail": f"At {level} concurrent request{'s' if peak['concurrency'] != 1 else ''}"
                if peak
                else "No successful load measurement",
            },
            {
                "label": "Response time · p95",
                "value": duration(at(peak, "latency.p95_ms")),
                "detail": "Request start to completion at peak output",
            },
            {
                "label": "First delivery · p95",
                "value": duration(at(peak, "ttft.p95_ms")),
                "detail": "First content or reasoning at peak output",
            },
            {
                "label": "Request success",
                "value": fmt(successful / total if total else None, "%"),
                "detail": f"{fmt(successful, 'int')} / {fmt(total, 'int')} timed requests · {fmt(errors, 'int')} errors",
            },
        ],
        "protocol": {
            "revision": at(perf, "protocol.revision") or "Not recorded",
            "input": fmt(at(perf, "protocol.input_reference_tokens"), "int"),
            "output": fmt(at(perf, "protocol.max_output_tokens"), "int"),
            "attempts": fmt(at(perf, "protocol.attempts_per_request"), "int"),
        },
    }


def performance_view(runs, *, offline=False):
    from app.services.telemetry_views import telemetry_view
    from app.services.capacity import estimate_capacity
    from app.services.performance_comparison import performance_comparison
    def current_perf(run):
        raw = run["perf"]
        return raw if raw.get("schema_version") == 3 else {}

    current = [{**r, "perf": current_perf(r), "is_sweep": r["perf"].get("schema_version") == 4
                and r["perf"].get("kind") == "context_sweep"} for r in runs]
    suite_rows = [
        {"label": label, "values": [fmt(at(r["perf"], key), unit) for r in current]}
        for label, key, unit in [
            ("Peak aggregate output", "peak_output_tokens_per_sec", "tok/s"),
            ("Peak successful request rate", "peak_requests_per_sec", "req/s"),
            ("Reference input tokens", "protocol.input_reference_tokens", "int"),
            ("Output limit", "protocol.max_output_tokens", "int"),
        ]
    ]
    charts = []
    descriptions = {
        "output_tokens_per_sec": "Successful output across all concurrent requests, divided by elapsed test time.",
        "latency.p95_ms": "95% of successful requests finished within this time at each load level.",
    }
    for key, title, unit in [
        ("output_tokens_per_sec", "Delivered output under load", "tok/s"),
        ("latency.p95_ms", "Response time under load · p95", "ms"),
        ("requests_per_sec", "Successful requests under load", "req/s"),
        ("ttft.p95_ms", "First delivery under load · p95", "ms"),
        ("error_rate", "Request failure rate", "%"),
        ("output_token_time.p50_ms", "Mean output token time proxy · p50", "ms/token"),
    ]:
        series = [
            {
                "label": r["label"],
                "color": COLORS[i % len(COLORS)],
                "points": [
                    (p["concurrency"], number(at(p, key)))
                    for p in points(r["perf"], "concurrency")
                    if number(p.get("concurrency")) is not None
                ],
            }
            for i, r in enumerate(current)
        ]
        svg = line_chart(series, title, "Concurrent requests", unit)
        if svg:
            charts.append(
                {
                    "title": title,
                    "primary": key in descriptions,
                    "description": descriptions.get(key, ""),
                    "svg": svg,
                    "series": [s for s in series if any(y is not None for _, y in s["points"])],
                }
            )
    load_rows, notes = [], []
    for r in current:
        perf = r["perf"]
        load_view = load_performance_view(r)
        if not perf and not r["is_sweep"]:
            notes.append(f"{r['label']}: no performance report for the current protocol.")
        if perf.get("cancelled"):
            notes.append(f"{r['label']}: stopped early; measurements are incomplete.")
        notes.extend(f"{r['label']}: {note}" for note in perf.get("notes", []))
        for p in points(perf, "concurrency"):
            load_rows.append(
                {
                    "run": r["label"],
                    "concurrency": fmt(p.get("concurrency"), "int"),
                    "peak": load_view["peak_concurrency"] is not None
                    and p.get("concurrency") == load_view["peak_concurrency"],
                    "output": fmt(p.get("output_tokens_per_sec"), "tok/s"),
                    "request_rate": fmt(p.get("requests_per_sec"), "req/s"),
                    "requests": fmt(p.get("requests"), "int"),
                    "errors": fmt(p.get("errors"), "int"),
                    "latency": duration(at(p, "latency.p95_ms")),
                    "ttft": duration(at(p, "ttft.p95_ms")),
                    "values": [
                        fmt(at(p, key), unit)
                        for key, unit in [
                            ("concurrency", "int"),
                            ("requests", "int"),
                            ("errors", "int"),
                            ("wall_ms", "ms"),
                            ("output_tokens_per_sec", "tok/s"),
                            ("requests_per_sec", "req/s"),
                            ("latency.p50_ms", "ms"),
                            ("latency.p95_ms", "ms"),
                            ("ttft.p50_ms", "ms"),
                            ("ttft.p95_ms", "ms"),
                            ("failure_latency.p95_ms", "ms"),
                            ("estimated_token_requests", "int"),
                            ("burst_delivery_requests", "int"),
                            ("latency.p99_ms", "ms"),
                            ("ttft.p99_ms", "ms"),
                            ("error_rate", "%"),
                            ("output_length.mean", "tokens"),
                            ("output_token_time.p50_ms", "ms/token"),
                            ("chunk_gap.p95_ms", "ms"),
                        ]
                    ],
                }
            )
    distributions = []
    for key, title, unit in [
        ("latency_ms", "Completed-request latency distribution", "ms"),
        ("ttft_ms", "First-delivery latency distribution", "ms"),
        ("completion_tokens", "Delivered output length distribution", "tokens"),
    ]:
        series = []
        for i, r in enumerate(current):
            for p in points(r["perf"], "concurrency"):
                values = sorted(number(s.get(key)) for s in points(p, "samples")
                                if s.get("ok") and number(s.get(key)) is not None)
                if values:
                    series.append({"label": f"{r['label']} · c={p['concurrency']}",
                                   "color": COLORS[len(series) % len(COLORS)],
                                   "points": [(100 * (j + 1) / len(values), v)
                                              for j, v in enumerate(values)]})
        svg = line_chart(series, title, "Successful requests · percentile", unit, percentile=True)
        if svg:
            distributions.append({"title": title, "svg": svg, "series": series})
    revisions = {at(r["perf"], "protocol.revision") for r in current if r["perf"]}
    if len(revisions) > 1:
        notes.append("These runs use different performance revisions; compare protocols before interpreting differences.")
    from app.services.sweep_views import sweep_view
    sweep_views = [sweep_view(r, offline) for r in runs if r.get("perf", {}).get("schema_version") == 4
                   and r["perf"].get("kind") == "context_sweep"]
    cache_rows = []
    for run in runs:
        pairs = list(run.get("perf", {}).get("cache_reuse", []))
        for cell in run.get("perf", {}).get("cells", []):
            if cell.get("cache_reuse"):
                pairs.append({"context_tokens": cell["context_tokens"], "concurrency": cell["concurrency"],
                              "effort": cell.get("effort", "default"), "cold": cell,
                              "warm": cell["cache_reuse"]})
        for pair in pairs:
            for mode in ("cold", "warm"):
                point = pair.get(mode)
                if not point:
                    continue
                telemetry = point.get("cache_metrics", {})
                cache_rows.append({
                    "run": run["label"], "context": fmt(pair["context_tokens"], "int"),
                    "concurrency": pair["concurrency"], "effort": pair.get("effort", "configured"),
                    "mode": mode, "status": point.get("status", pair.get("status", "measured")),
                    "ttft": duration(at(point, "ttft.p50_ms")),
                    "hit": fmt(telemetry.get("cache_hit_fraction"), "%"),
                    "coverage": f"{telemetry.get('cache_reporting_requests', 0)}/{telemetry.get('successful_requests', 0)}",
                    "prefill": fmt(at(telemetry, "uncached_prefill_tokens_per_sec.p50"), "tok/s"),
                    "effective": fmt(at(telemetry, "effective_prompt_tokens_per_sec.p50"), "tok/s"),
                    "decode": fmt(at(telemetry, "decode_tokens_per_sec.p50"), "tok/s"),
                    "output": fmt(point.get("output_tokens_per_sec", point.get("aggregate_tokens_per_sec")), "tok/s"),
                    "requests": point.get("requests", 0), "errors": point.get("errors", 0),
                })
    return {
        "comparison": performance_comparison(runs),
        "cache_rows": cache_rows,
        "capacity_views": [estimate_capacity(r) for r in runs if r.get("perf")],
        "telemetry_views": [view for r in runs if (view := telemetry_view(r))],
        "sweep_views": sweep_views,
        "suite_rows": suite_rows,
        "load_views": [load_performance_view(r) for r in current if r["perf"]],
        "charts": charts,
        "distributions": distributions,
        "load_rows": load_rows,
        "notes": notes,
        "labels": [r["label"] for r in current],
    }


def quality_timing_view(run):
    """Model-call timings belong to quality diagnostics, independently of load tests."""
    from app.benchmarking.models import LatencyStats
    from app.services.capacity import observed_quality

    results = run.get("results", [])
    good = [r for r in results if r.get("request_ok")]
    failed = [r for r in results if not r.get("request_ok")]
    latency = LatencyStats.from_samples([number(r.get("latency_ms")) for r in good])
    ttft = LatencyStats.from_samples([number(r.get("ttft_ms")) for r in good])
    categories = defaultdict(list)
    for r in results:
        categories[r["category"]].append(r)
    rows = []
    for category, items in sorted(categories.items()):
        successful = [r for r in items if r.get("request_ok")]
        stats = LatencyStats.from_samples([number(r.get("latency_ms")) for r in successful])
        errors = sum(not r.get("request_ok") for r in items)
        deadlines = sum("stream deadline exceeded" in (r.get("detail") or "") for r in items)
        rows.append(
            {
                "category": category,
                "requests": len(items),
                "errors": errors,
                "deadlines": deadlines,
                "p50": fmt(stats.p50, "ms"),
                "p95": fmt(stats.p95, "ms"),
                "time_p50": duration(stats.p50),
                "time_p95": duration(stats.p95),
                "p95_ms": stats.p95,
            }
        )
    series = []
    for label, items, key, color in [
        ("Request delivered", good, "latency_ms", COLORS[0]),
        ("Request error", failed, "latency_ms", COLORS[2]),
        ("First delivery", good, "ttft_ms", COLORS[1]),
    ]:
        values = sorted(number(r.get(key)) for r in items if number(r.get(key)) is not None)
        if values:
            series.append(
                {
                    "label": label,
                    "color": color,
                    "points": [(100 * (i + 1) / len(values), v) for i, v in enumerate(values)],
                }
            )
    completion_series = [s for s in series if s["label"] != "First delivery"]
    delivery_series = [s for s in series if s["label"] == "First delivery"]
    chart = line_chart(
        completion_series,
        "Quality task completion time distribution",
        "Tasks · percentile",
        "ms",
        percentile=True,
    )
    delivery_chart = line_chart(
        delivery_series,
        "Quality task first-delivery distribution",
        "Tasks · percentile",
        "ms",
        percentile=True,
    )
    slowest = sorted(
        (r for r in rows if r["p95_ms"] is not None), key=lambda r: (-r["p95_ms"], r["category"])
    )[:5]
    maximum = max((r["p95_ms"] for r in slowest), default=0)
    slowest = [{**r, "width": 100 * r["p95_ms"] / maximum if maximum else 0} for r in slowest]
    deadlines = sum(r["deadlines"] for r in rows)
    metrics = [
        {
            "label": "Request delivery",
            "value": fmt(len(good) / len(results) if results else None, "%"),
            "detail": f"{len(good):,} / {len(results):,} recorded tasks",
        },
        {
            "label": "Typical response · p50",
            "value": duration(latency.p50),
            "detail": "Median model-call time for tasks with delivered responses",
        },
        {
            "label": "Slower responses · p95",
            "value": duration(latency.p95),
            "detail": "95th percentile of model-call time for delivered responses",
        },
        {
            "label": "First delivery · p50",
            "value": duration(ttft.p50),
            "detail": "Median time to first content or reasoning",
        },
    ]
    return {
        "label": run["label"],
        "usage": observed_quality(run),
        "rows": rows,
        "chart": chart,
        "series": series,
        "metrics": metrics,
        "slowest": slowest,
        "error_count": len(failed),
        "completion_series": completion_series,
        "delivery_chart": delivery_chart,
        "delivery_series": delivery_series,
        "failed_categories": [r for r in rows if r["errors"]],
        "count": len(results),
        "success_rate": fmt(len(good) / len(results) if results else None, "%"),
        "latency_p50": fmt(latency.p50, "ms"),
        "latency_p95": fmt(latency.p95, "ms"),
        "ttft_p50": fmt(ttft.p50, "ms"),
        "deadline_count": deadlines,
    }


def render_report(runs):
    context = comparison_context(runs)
    charts = []
    # Never silently substitute a legacy average for balanced capability.
    svg = bar_chart(runs, lambda r: at(r, "quality.summary.category_balanced"),
                    "Criterion achievement", "%")
    if svg:
        charts.append({"title": "Criterion achievement", "svg": svg})
    return templates.env.get_template("report_download.html").render(
        runs=runs,
        comparison=len(runs) > 1,
        has_quality_results=any(run.get("results") or run.get("quality") for run in runs),
        performance=performance_view(runs, offline=True),
        quality_timings=[quality_timing_view(r) for r in runs if r.get("results")],
        offline=True,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        overview_charts=charts,
        colors=COLORS,
        fmt=fmt,
        **context,
    )
