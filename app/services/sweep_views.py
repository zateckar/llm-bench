"""Bounded presentation data and script-free export matrices for sweeps."""

from html import escape
from markupsafe import Markup

METRICS = [
    {"key": "first_p50", "label": "First token · p50", "unit": "ms", "better": "Lower is faster"},
    {"key": "first_p95", "label": "First token · p95", "unit": "ms", "better": "Lower is faster"},
    {"key": "answer_p50", "label": "Whole answer · p50", "unit": "ms", "better": "Lower is faster"},
    {"key": "answer_p95", "label": "Whole answer · p95", "unit": "ms", "better": "Lower is faster"},
    {
        "key": "tokens_per_sec",
        "label": "Per-request output · p50",
        "unit": "tok/s",
        "better": "Higher is faster",
    },
    {
        "key": "generation_tokens_per_sec",
        "label": "Stream output · p50",
        "unit": "tok/s",
        "better": "Higher is faster",
    },
    {
        "key": "aggregate_tokens_per_sec",
        "label": "Aggregate output",
        "unit": "tok/s",
        "better": "Higher is faster",
    },
]


def sweep_view(run, offline=False):
    from app.services.html_reports import at, duration, fmt

    report = run["perf"]
    protocol = report.get("protocol", {})
    cells = []
    for cell in report.get("cells", []):
        values = {
            "first_p50": at(cell, "ttft.p50_ms"),
            "first_p95": at(cell, "ttft.p95_ms"),
            "answer_p50": at(cell, "latency.p50_ms"),
            "answer_p95": at(cell, "latency.p95_ms"),
            "tokens_per_sec": at(cell, "request_tokens_per_sec.p50"),
            "generation_tokens_per_sec": at(cell, "generation_tokens_per_sec.p50"),
            "aggregate_tokens_per_sec": cell.get("aggregate_tokens_per_sec"),
        }
        cells.append(
            {
                "context": cell["context_tokens"],
                "concurrency": cell["concurrency"],
                "effort": cell["effort"],
                "status": cell["status"],
                "values": values,
                "requests": cell.get("requests", 0),
                "completed": cell.get("completed", 0),
                "errors": cell.get("errors", 0),
                "incomplete": cell.get("incomplete", 0),
                "ttft_count": at(cell, "ttft.count") or 0,
                "error": cell.get("error", ""),
                "estimated": cell.get("estimated_output_requests", 0),
                "estimated_input": cell.get("estimated_input_requests", 0),
                "cached": cell.get("cached_tokens", 0),
                "bursts": cell.get("burst_requests", 0),
                "provider_input": at(cell, "prompt_tokens.mean"),
                "cancelled": cell.get("cancelled", False),
                "formatted": {
                    key: duration(value)
                    if key.startswith(("first_", "answer_"))
                    else fmt(value, "tok/s")
                    for key, value in values.items()
                },
            }
        )
    data = {
        "contexts": protocol.get("contexts", []),
        "concurrencies": protocol.get("concurrencies", []),
        "efforts": report.get("efforts", []),
        "metrics": METRICS,
        "cells": cells,
    }
    if not offline:
        for cell in cells:
            cell.pop("formatted", None)
    view = {
        "label": run["label"],
        "data": data,
        "protocol": protocol,
        "measured": sum(bool(cell["requests"]) for cell in cells),
        "complete": report.get("finished", False) and not report.get("cancelled"),
        "accepted": sum(e.get("status") == "accepted" for e in data["efforts"]),
        "exports": [],
    }
    if offline:
        for effort in data["efforts"]:
            effort_cells = [c for c in cells if c["effort"] == effort["effort"]]
            matrices = [
                {"title": metric["label"], "svg": export_matrix(data, effort, metric)}
                for metric in METRICS
                if effort_cells
            ]
            view["exports"].append({**effort, "cells": effort_cells, "matrices": matrices})
    return view


def export_matrix(data, effort, metric):
    """Every cell has an exact-value tooltip; missing results have distinct colors."""
    contexts, concurrencies = data["contexts"], data["concurrencies"]
    if not contexts or not concurrencies:
        return ""
    cells = {
        (c["context"], c["concurrency"]): c
        for c in data["cells"]
        if c["effort"] == effort["effort"]
    }
    numbers = [
        c["values"][metric["key"]] for c in cells.values() if c["values"][metric["key"]] is not None
    ]
    low, high = min(numbers, default=0), max(numbers, default=0)
    width, height = 100 + 29 * len(concurrencies), 58 + 18 * len(contexts)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(metric["label"])} by context and concurrency" viewBox="0 0 {width} {height}" class="sweep-export-matrix">',
        '<g fill="currentColor" font-size="10" font-family="sans-serif">',
        '<text x="100" y="14">Concurrent requests →</text>',
    ]
    for x, concurrency in enumerate(concurrencies):
        svg.append(f'<text x="{114 + x * 29}" y="34" text-anchor="middle">{concurrency}</text>')
    for y, context in enumerate(contexts):
        top = 42 + y * 18
        svg.append(f'<text x="88" y="{top + 12}" text-anchor="end">{context:,}</text>')
        for x, concurrency in enumerate(concurrencies):
            cell = cells.get((context, concurrency))
            value = cell["values"][metric["key"]] if cell else None
            if value is not None:
                ratio = (value - low) / (high - low) if high > low else 0.5
                fill = f"hsl({215 + ratio * 55:.0f},70%,{75 - ratio * 35:.0f}%)"
            elif cell and cell["status"] in {"failed", "unsupported_context", "incomplete"}:
                fill = "#a55a37"
            else:
                fill = "#494953"
            label = f"{context:,} tokens · c={concurrency} · {cell['formatted'][metric['key']] if cell else 'Not measured'} · {cell['status'] if cell else effort['status']}"
            svg.append(
                f'<rect x="{100 + x * 29}" y="{top}" width="26" height="15" rx="2" fill="{fill}"><title>{escape(label)}</title></rect>'
            )
    svg.append("</g></svg>")
    return Markup("".join(svg))
