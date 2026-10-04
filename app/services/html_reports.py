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
from app.benchmarking.quality_report import paired_comparison

COLORS = ("#3b82f6", "#14b8a6", "#e99b16", "#e879ad", "#9b87f5", "#ef744d")


def object_json(raw):
    try:
        value = json.loads(raw or "null")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def interactive_turns(result):
    """Display stored actions separately from the enclosing transcript JSON."""
    meta = object_json(result.get("quality_metadata_json"))
    if meta.get("metadata", {}).get("protocol") != "json-actions-v1":
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
    run["perf"] = object_json(run.get("perf_json"))
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
        result["attempt_diagnostics"] = metadata.get("metrics", {}).get("attempt_diagnostics", [])
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


def line_chart(series, title, x_label, unit):
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
    xmin, xmax = min(xs), max(xs)
    ymax = max(.01 if unit in {"%", "req/s"} else 1, max(y for _, y in valid)) * 1.1
    if unit == "%":
        ymax = min(1, ymax)
    width, height = 700, 300
    left, top, pw, ph = 78, 25, 595, 210

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
        label = (f"{value * 100:.1f}%" if unit == "%" else
                 f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.1f}")
        parts.append(svg_text(left - 12, y + 4, label, 'text-anchor="end"'))
    ticks = xs if len(xs) <= 7 else [xs[round(i * (len(xs) - 1) / 6)] for i in range(7)]
    for x in ticks:
        parts.append(svg_text(cx(x), top + ph + 24, f"{x:,.0f}", 'text-anchor="middle"'))
    parts.append(svg_text(left + pw / 2, height - 12, x_label, 'text-anchor="middle"'))
    parts.append(svg_text(left, 14, unit))
    for s in series:
        color = s["color"]
        previous = None
        for x, y in sorted(s["points"], key=lambda p: p[0]):
            if number(x) is None or number(y) is None:
                previous = None
                continue
            px, py = cx(x), cy(y)
            if previous:
                parts.append(
                    f'<path d="M{previous[0]:g} {previous[1]:g}L{px:g} {py:g}" fill="none" stroke="{color}" stroke-width="2.5"/>'
                )
            parts.append(
                f'<circle cx="{px:g}" cy="{py:g}" r="4.5" fill="{color}"><title>{escape(s["label"])}: {x:g}, {escape(fmt(y, unit))}</title></circle>'
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


def performance_view(runs):
    def current_perf(run):
        raw = run["perf"]
        return raw if raw.get("schema_version") == 3 else {}

    current = [{**r, "perf": current_perf(r)} for r in runs]
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
    for key, title, unit in [
        ("output_tokens_per_sec", "Delivered output under load", "tok/s"),
        ("requests_per_sec", "Successful requests under load", "req/s"),
        ("latency.p95_ms", "Completed-request latency · p95", "ms"),
        ("ttft.p95_ms", "First delivered token · p95", "ms"),
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
                    "svg": svg,
                    "series": [s for s in series if any(y is not None for _, y in s["points"])],
                }
            )
    load_rows, notes = [], []
    for r in current:
        perf = r["perf"]
        if not perf:
            notes.append(f"{r['label']}: no performance report for the current protocol.")
        if perf.get("cancelled"):
            notes.append(f"{r['label']}: stopped early; measurements are incomplete.")
        notes.extend(f"{r['label']}: {note}" for note in perf.get("notes", []))
        for p in points(perf, "concurrency"):
            load_rows.append(
                {
                    "run": r["label"],
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
        svg = line_chart(series, title, "Observed percentile (successful requests)", unit)
        if svg:
            distributions.append({"title": title, "svg": svg, "series": series})
    revisions = {at(r["perf"], "protocol.revision") for r in current if r["perf"]}
    if len(revisions) > 1:
        notes.append("These runs use different performance revisions; compare protocols before interpreting differences.")
    request_views = [request_performance_view(r) for r in runs if r.get("results")]
    return {
        "suite_rows": suite_rows,
        "charts": charts,
        "distributions": distributions,
        "request_views": request_views,
        "load_rows": load_rows,
        "notes": notes,
        "labels": [r["label"] for r in current],
    }


def request_performance_view(run):
    """Historical request timings remain separate from the controlled load suite."""
    from app.benchmarking.models import LatencyStats

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
        rows.append({"category": category, "requests": len(items), "errors": errors,
                     "deadlines": deadlines, "p50": fmt(stats.p50, "ms"), "p95": fmt(stats.p95, "ms")})
    series = []
    for label, items, key, color in [
        ("Successful latency", good, "latency_ms", COLORS[0]),
        ("Failed latency", failed, "latency_ms", COLORS[2]),
        ("First delivery", good, "ttft_ms", COLORS[1]),
    ]:
        values = sorted(number(r.get(key)) for r in items if number(r.get(key)) is not None)
        if values:
            series.append({"label": label, "color": color,
                           "points": [(100 * (i + 1) / len(values), v) for i, v in enumerate(values)]})
    chart = line_chart(series, "Quality request timing distribution", "Observed percentile", "ms")
    return {"label": run["label"], "rows": rows, "chart": chart, "series": series,
            "count": len(results), "success_rate": fmt(len(good) / len(results) if results else None, "%"),
            "latency_p50": fmt(latency.p50, "ms"), "latency_p95": fmt(latency.p95, "ms"),
            "ttft_p50": fmt(ttft.p50, "ms"), "deadline_count": sum(r["deadlines"] for r in rows)}


def render_report(runs):
    context = comparison_context(runs)
    charts = []
    # Never silently substitute a legacy average for balanced capability.
    for title, getter, unit in [
        ("Strict task success", lambda r: at(r, "quality.summary.category_balanced"), "%"),
        ("Peak aggregate output", lambda r: at(r, "app.benchmarking.perf.peak_output_tokens_per_sec"), "tok/s"),
    ]:
        svg = bar_chart(runs, getter, title, unit)
        if svg:
            charts.append({"title": title, "svg": svg})
    return templates.env.get_template("report_download.html").render(
        runs=runs,
        comparison=len(runs) > 1,
        performance=performance_view(runs),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        overview_charts=charts,
        colors=COLORS,
        fmt=fmt,
        **context,
    )
