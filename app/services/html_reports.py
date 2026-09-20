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
from quality_report import paired_comparison

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
        if not isinstance(t, dict) or t.get("role") not in {"assistant", "tool"} or "content" not in t:
            continue
        number += t["role"] == "assistant"
        turns.append({"role": t["role"], "number": number,
                      "content": t["content"] if isinstance(t["content"], str)
                      else json.dumps(t["content"], ensure_ascii=False, indent=2)})
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
    weight = sum(r.get("weight") or 1 for r in scored)
    run["weighted_score"] = (
        sum(r["score"] * (r.get("weight") or 1) for r in scored) / weight if weight else 0
    )
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
    ymax = max(1, max(y for _, y in valid)) * 1.1
    width, height = 700, 300
    left, top, pw, ph = 78, 25, 595, 210

    def cx(x):
        return left + (x - xmin) / (xmax - xmin) * pw if xmax > xmin else left + pw / 2

    def cy(y):
        return top + ph - y / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<title>{escape(title)}</title><g font-family="system-ui,sans-serif" font-size="12" fill="currentColor">',
    ]
    for i in range(5):
        value = ymax * i / 4
        y = cy(value)
        parts.append(f'<path d="M{left} {y:g}H{left + pw}" stroke="currentColor" opacity=".12"/>')
        label = f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.1f}"
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
    def rows(specs, source):
        return [
            {"label": label, "values": [fmt(at(source(r), key), unit) for r in runs]}
            for label, key, unit in specs
        ]

    question_rows = rows(
        [
            ("Workers", "workers", "int"),
            ("Wall clock", "duration_ms", "ms"),
            ("Latency p50", "latency_p50_ms", "ms"),
            ("Latency p95", "latency_p95_ms", "ms"),
            ("Latency p99", "latency_p99_ms", "ms"),
            ("TTFT p50", "ttft_p50_ms", "ms"),
            ("TTFT p95", "ttft_p95_ms", "ms"),
            ("Aggregate output", "output_tokens_per_sec", "tok/s"),
            ("Prompt tokens", "total_prompt_tokens", "int"),
            ("Output tokens", "total_completion_tokens", "int"),
            ("Request errors", "error_count", "int"),
        ],
        lambda r: r,
    )
    suite_rows = rows(
        [
            ("Single-stream decode", "decode_tokens_per_sec", "tok/s"),
            ("Long-output decode", "long_output_tokens_per_sec", "tok/s"),
            ("Prefill", "prefill_tokens_per_sec", "tok/s"),
            ("Serial latency p50", "serial_latency.p50_ms", "ms"),
            ("Serial latency p95", "serial_latency.p95_ms", "ms"),
            ("Serial TTFT p50", "serial_ttft.p50_ms", "ms"),
            ("Serial TTFT p95", "serial_ttft.p95_ms", "ms"),
            ("Peak aggregate output", "peak_output_tokens_per_sec", "tok/s"),
            ("Peak request rate", "peak_requests_per_sec", "req/s"),
            ("Saturation concurrency", "saturation_concurrency", "int"),
            ("Scaling efficiency", "scaling_efficiency", "%"),
            ("Concurrency meeting SLO", "slo_capacity", "int"),
            ("Estimated active users", "capacity_users", "int"),
            ("SLO: maximum TTFT p95", "slo_ttft_p95_ms", "ms"),
            ("SLO: minimum stream decode p50", "slo_stream_tps_p50", "tok/s"),
            ("SLO: maximum error rate", "slo_error_rate", "%"),
            ("Requests per user per hour", "requests_per_user_hour", "int"),
            ("Cache TTFT speedup", "cache_probe.ttft_speedup", "×"),
            ("Cache hit ratio", "cache_probe.cache_hit_ratio", "%"),
            ("Cache prefill gain", "cache_probe.prefill_gain", "×"),
        ],
        lambda r: r["perf"],
    )
    charts = []
    for key, title, unit in [
        ("output_tokens_per_sec", "Throughput under load", "tok/s"),
        ("ttft.p95_ms", "Time to first token under load · p95", "ms"),
        ("stream_tps.p50_ms", "Per-stream decode under load · p50", "tok/s"),
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
            for i, r in enumerate(runs)
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
    context_series = []
    context_rows = []
    load_rows = []
    notes = []
    for r in runs:
        perf = r["perf"]
        if not perf:
            notes.append(f"{r['label']}: dedicated performance suite not recorded.")
        if perf.get("error"):
            notes.append(f"{r['label']}: performance suite incomplete — {perf['error']}")
        if perf.get("streaming") is False:
            notes.append(f"{r['label']}: streaming unavailable; TTFT may be unmeasured.")
        raw_notes = perf.get("notes", [])
        if isinstance(raw_notes, list):
            notes.extend(f"{r['label']}: {note}" for note in raw_notes)
        cache_notes = at(perf, "cache_probe.notes")
        if isinstance(cache_notes, list):
            notes.extend(f"{r['label']} · cache: {note}" for note in cache_notes)
        groups = defaultdict(list)
        for p in points(perf, "concurrency"):
            load_rows.append(
                {
                    "run": r["label"],
                    "values": [
                        fmt(at(p, key), unit)
                        for key, unit in [
                            ("concurrency", "int"),
                            ("requests", "int"),
                            ("output_tokens_per_sec", "tok/s"),
                            ("requests_per_sec", "req/s"),
                            ("latency.p95_ms", "ms"),
                            ("ttft.p95_ms", "ms"),
                            ("stream_tps.p50_ms", "tok/s"),
                            ("error_rate", "%"),
                        ]
                    ],
                }
            )
        for p in points(perf, "context_sweep"):
            size, level = number(p.get("context_tokens")), number(p.get("concurrency")) or 1
            skipped = bool(p.get("skipped"))
            if size is not None:
                groups[level].append((size, None if skipped else number(at(p, "ttft.p50_ms"))))
            context_rows.append(
                {
                    "run": r["label"],
                    "status": p.get("skip_reason")
                    or ("Skipped" if skipped else "Measured")
                    + (
                        " · " + "; ".join(map(str, p["notes"]))
                        if isinstance(p.get("notes"), list) and p["notes"]
                        else ""
                    ),
                    "values": [fmt(size, "int"), fmt(level, "int")]
                    + [
                        fmt(None if skipped else at(p, key), unit)
                        for key, unit in [
                            ("ttft.p50_ms", "ms"),
                            ("warm_ttft.p50_ms", "ms"),
                            ("prompt_tokens_per_sec", "tok/s"),
                            ("output_tokens_per_sec", "tok/s"),
                            ("error_rate", "%"),
                        ]
                    ],
                }
            )
        for level, values in sorted(groups.items()):
            context_series.append(
                {
                    "label": f"{r['label']} · c={level:g}",
                    "color": COLORS[len(context_series) % len(COLORS)],
                    "points": values,
                }
            )
    svg = line_chart(
        context_series, "Context scaling · cold TTFT p50", "Target context tokens", "ms"
    )
    if svg:
        charts.append(
            {"title": "Context scaling · cold TTFT p50", "svg": svg, "series": context_series}
        )
    return {
        "question_rows": question_rows,
        "suite_rows": suite_rows,
        "charts": charts,
        "load_rows": load_rows,
        "context_rows": context_rows,
        "notes": notes,
        "labels": [r["label"] for r in runs],
    }


def render_report(runs):
    context = comparison_context(runs)
    charts = []
    # Never silently substitute a legacy average for balanced capability.
    for title, getter, unit in [
        ("Balanced capability", lambda r: at(r, "quality.summary.category_balanced"), "%"),
        (
            "All-item average",
            lambda r: r.get("avg_score") if r.get("scored_questions") else None,
            "%",
        ),
        ("Peak aggregate output", lambda r: at(r, "perf.peak_output_tokens_per_sec"), "tok/s"),
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
