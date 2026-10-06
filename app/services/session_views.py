"""Presentation of users-at-SLO (session) reports for run pages and offline HTML."""

from app.benchmarking.session_load import KIND, LIMITS, SCHEMA_VERSION

STATUS_LABELS = {
    "established": "Established",
    "lower_bound": "Lower bound",
    "not_met": "Not met",
    "inconclusive": "Inconclusive",
    "not_measured": "Not measured",
}
SATURATION_LABELS = {
    "reached": "Exhausted",
    "not_reached": "Not reached",
    "inherited": "From a smaller cap",
    "client_limited": "Client-limited",
    "not_measured": "Not measured",
}


def is_sessions(perf):
    return isinstance(perf, dict) and perf.get("schema_version") == SCHEMA_VERSION and perf.get("kind") == KIND


def tokens(value):
    from app.services.html_reports import number

    value = number(value)
    if value is None:
        return "n/a"
    return f"{value / 1024:,.0f}k" if value >= 1024 and value % 1024 == 0 else f"{value:,.0f}"


def _users(result):
    users = result.get("users")
    if users is None:
        return "n/a"
    if result.get("status") == "not_met":
        return "0"
    return f"≥ {users}" if result.get("status") == "lower_bound" else str(users)


def _class_cells(level, names, slos):
    from app.services.html_reports import duration, fmt

    cells = []
    for name in names:
        item = (level.get("classes") or {}).get(name) or {}
        if not item.get("users"):
            cells.append(None)
            continue
        slo = slos.get(name) or {}
        speed = (item.get("output_speed") or {}).get("p05")
        cells.append({
            "attainment": fmt(item.get("attainment"), "%"),
            "meets": item.get("meets_target"),
            "ttft": duration((item.get("ttft") or {}).get("p95_ms")),
            "ttft_over": ((item.get("ttft") or {}).get("p95_ms") or 0) > slo.get("ttft_ms", float("inf")),
            "speed": fmt(speed, "tok/s"),
            "speed_under": speed is not None and speed < slo.get("output_tokens_per_sec", 0),
            "requests": item.get("requests", 0),
            "users": item.get("users", 0),
            "thin": item.get("thin"),
        })
    return cells


def _server(server):
    from app.services.html_reports import fmt

    if not server:
        return "—"
    if "metrics" in server:
        metrics = server["metrics"]

        def value(key, field="mean"):
            return (metrics.get(key) or {}).get(field)

        parts = []
        if value("kv_cache", "max") is not None:
            parts.append(f"KV max {fmt(value('kv_cache', 'max'), '%')}")
        if value("waiting") is not None:
            parts.append(f"waiting {value('waiting'):,.1f} (max {value('waiting', 'max'):,.0f})")
        if value("running", "max") is not None:
            parts.append(f"running ≤ {value('running', 'max'):,.0f}")
        if value("preemptions", "max"):
            parts.append(f"preemptions {value('preemptions', 'max'):,.2f}/s")
        if value("prefix_hit") is not None:
            parts.append(f"prefix hits {fmt(value('prefix_hit'), '%')}")
        if value("sm_active") is not None:
            parts.append(f"SM {fmt(value('sm_active'), '%')}")
        if not server.get("aligned", True):
            parts.append("clock alignment uncertain")
        return " · ".join(parts) or "—"
    # sessions-v1 reports before the constraint diagnosis.
    parts = []
    if server.get("kv_cache_max") is not None:
        parts.append(f"KV {fmt(server['kv_cache_max'], '%')}")
    if server.get("waiting_max") is not None:
        parts.append(f"waiting {fmt(server['waiting_max'], 'int')}")
    if server.get("preemptions_max"):
        parts.append(f"preemptions {server['preemptions_max']:,.2f}/s")
    if server.get("prefix_hit_mean") is not None:
        parts.append(f"prefix hits {fmt(server['prefix_hit_mean'], '%')}")
    return " · ".join(parts) or "—"


def _saturation(saturation):
    """Users at which the deployment is exhausted, as table text."""
    if not saturation:
        return "—"
    users, status = saturation.get("users"), saturation.get("status")
    if status == "reached":
        return str(users)
    if status == "inherited":
        return f"≤ {users}"
    if status in ("not_reached", "client_limited") and users:
        return f"> {users}"
    return "n/a"


def _constraint(item, where):
    if not item:
        return None
    return {"where": where, "label": item.get("label"), "summary": item.get("summary"),
            "basis": "vLLM and GPU telemetry" if item.get("basis") == "server" else "client measurements only",
            "evidence": item.get("evidence") or [], "remedies": item.get("remedies") or [],
            "also": [a.get("label") for a in item.get("also") or []]}


def session_view(run):
    from app.services.html_reports import COLORS, fmt, line_chart
    from app.benchmarking.session_workload import PRESET_LABELS

    report = run["perf"]
    protocol = report.get("protocol") or {}
    model = protocol.get("user_model") or {}
    classes = model.get("classes") or []
    names = [c.get("name") for c in classes]
    slos = {c.get("name"): c.get("slo") or {} for c in classes}
    total_weight = sum(c.get("weight", 0) for c in classes) or 1
    results, levels, series, throughput, diagnoses = [], [], [], [], []
    for i, entry in enumerate(report.get("caps") or []):
        cap, result = entry.get("context_cap"), entry.get("result") or {}
        limiting = result.get("limiting") or {}
        at = result.get("at_result") or {}
        saturation = entry.get("saturation")
        results.append({
            "cap": tokens(cap), "users": _users(result), "status": STATUS_LABELS.get(result.get("status"), "n/a"),
            "status_key": result.get("status"), "first_failed": result.get("first_failed") or "—",
            "inherited": result.get("inherited"), "limiting": limiting.get("label") or "—",
            "detail": limiting.get("detail") or "",
            "context": f"{tokens(at.get('context_p50'))} / {tokens(at.get('context_p95'))}" if at else "—",
            "output": fmt(at.get("output_tokens_per_sec"), "tok/s") if at else "—",
            "levels": len(entry.get("levels") or []),
            "saturation": _saturation(saturation),
            "saturation_status": SATURATION_LABELS.get((saturation or {}).get("status"), ""),
            "saturation_reason": (saturation or {}).get("reason") or "",
            "peak": (f"{fmt(saturation.get('peak_output_tokens_per_sec'), 'tok/s')} at {saturation.get('peak_users')} users"
                     if saturation and saturation.get("peak_users") else "—"),
        })
        found = [c for c in (_constraint(result.get("constraint"), "At the SLO limit"),
                             _constraint((saturation or {}).get("constraint"), "At exhaustion")) if c]
        if len(found) == 2 and found[0]["label"] == found[1]["label"]:
            found = [{**found[1], "where": "At the SLO limit and at exhaustion"}]
        if found:
            diagnoses.append({"cap": tokens(cap), "items": found})
        ordered = sorted(entry.get("levels") or [], key=lambda lv: lv["users"])
        live = [lv for lv in ordered if not lv.get("cancelled")]
        series.append({"label": f"{tokens(cap)} cap", "color": COLORS[i % len(COLORS)],
                       "points": [(lv["users"], lv.get("attainment")) for lv in live]})
        throughput.append({"label": f"{tokens(cap)} cap", "color": COLORS[i % len(COLORS)],
                           "points": [(lv["users"], lv.get("output_tokens_per_sec")) for lv in live]})
        for level in entry.get("levels") or []:
            limit = level.get("limiting") or {}
            constraint = level.get("constraint") or {}
            levels.append({
                "cap": tokens(cap), "users": level["users"], "passed": level.get("passed"),
                "phase": "Saturation" if level.get("phase") == "saturation" else "SLO",
                "hard_limit": level.get("hard_limit") or "",
                "constraint": constraint.get("label") or "",
                "result": ("Cancelled" if level.get("cancelled") else "Pass" if level.get("passed")
                           else "Stopped early" if level.get("early_stop") else "Fail"),
                "attainment": fmt(level.get("attainment"), "%"),
                "classes": _class_cells(level, names, slos),
                "context": f"{tokens((level.get('context_tokens') or {}).get('p50'))} / "
                           f"{tokens((level.get('context_tokens') or {}).get('p95'))}",
                "requests": level.get("requests", 0),
                "output": fmt(level.get("output_tokens_per_sec"), "tok/s"),
                "server": _server(level.get("server")),
                "note": level.get("early_stop") or (limit.get("label") if limit else "") or "",
                "detail": limit.get("detail", "") if limit else "",
                "warmup": fmt(level.get("warmup_seconds"), "s"),
                "warmup_complete": level.get("warmup_complete", True),
                "errors": " · ".join([*(level.get("error_examples") or []), *(level.get("user_failures") or [])]),
            })
    headline = (report.get("summary") or {}).get("headline") or {}
    chart = line_chart(series, "SLO attainment by simulated users", "Simulated users", "%")
    output_chart = line_chart(throughput, "Output throughput by simulated users", "Simulated users", "tok/s")
    target = protocol.get("attainment_target")
    exhausted = headline.get("saturation") or {}
    return {
        "label": run["label"],
        "headline": {"users": _users(headline) if headline else "n/a", "cap": tokens(headline.get("context_cap")),
                     "status": STATUS_LABELS.get(headline.get("status"), "Not measured"),
                     "limiting": ((headline.get("constraint") or {}).get("label")
                                  or (headline.get("limiting") or {}).get("label") or "")},
        "exhaustion": {"users": _saturation(exhausted), "status": SATURATION_LABELS.get(exhausted.get("status"), ""),
                       "constraint": (exhausted.get("constraint") or {}).get("label") or ""} if exhausted else None,
        "saturation_search": protocol.get("saturation"),
        "diagnoses": diagnoses,
        "output_chart": {"title": "Output throughput by simulated users", "svg": output_chart, "series": throughput,
                         "description": "Generated tokens per second over each measured window. Where the curve flattens "
                                        "while first tokens grow, more users only add waiting."} if output_chart else None,
        "results": results,
        "levels": levels,
        "names": names,
        "classes": [{"name": c.get("name"), "share": fmt(c.get("weight", 0) / total_weight, "%"),
                     "system": f"{c.get('system_tokens', 0):,}",
                     "first": _mean(c.get("first_input")), "turn": _mean(c.get("turn_input")),
                     "output": _mean(c.get("output")), "think": _mean(c.get("think_seconds"), "s"),
                     "turns": _mean(c.get("turns"), ""),
                     "slo": f"first token ≤ {(c.get('slo') or {}).get('ttft_ms', 0) / 1000:g} s · "
                            f"≥ {(c.get('slo') or {}).get('output_tokens_per_sec', 0)} tok/s"} for c in classes],
        "chart": {"title": "SLO attainment by simulated users", "svg": chart, "series": series,
                  "description": f"Share of measured requests that met their class SLO; the target is "
                                 f"{fmt(target, '%')}."} if chart else None,
        "protocol": {
            "revision": protocol.get("revision"), "preset": PRESET_LABELS.get(protocol.get("preset"), "Custom"),
            "hash": protocol.get("user_model_hash"), "caps": ", ".join(tokens(c) for c in protocol.get("context_caps") or []),
            "warmup": protocol.get("warmup_seconds"), "measure": protocol.get("measure_seconds"),
            "target": fmt(target, "%"), "start": protocol.get("start_users"), "max": protocol.get("max_users"),
            "resolution": fmt(protocol.get("resolution"), "%"), "seed": protocol.get("seed"),
            "saturation_max": protocol.get("saturation_max_users"),
            "definitions": [protocol.get(k) for k in ("user_definition", "steady_state_definition", "slo_definition",
                                                      "pass_definition", "search_definition", "saturation_definition",
                                                      "constraint_definition", "context_definition",
                                                      "output_definition") if protocol.get(k)],
        },
        "cancelled": report.get("cancelled"),
        "stop_reason": report.get("stop_reason"),
        "notes": report.get("notes") or [],
        "limits": LIMITS,
    }


def _mean(buckets, unit="tokens"):
    if not buckets:
        return "n/a"
    total = sum(w for _, _, w in buckets) or 1
    value = sum((a + b) / 2 * w for a, b, w in buckets) / total
    if unit == "s":
        return f"{value:,.1f} s"
    return f"{value:,.0f}" + (f" {unit}" if unit else "")


def comparison(runs):
    """Users at SLO by context cap across runs with the same user model and window."""
    from app.services.html_reports import at

    views = [r for r in runs if is_sessions(r.get("perf"))]
    if len(views) < 2:
        return None
    key = lambda r: at(r["perf"], "protocol.comparison_key")  # noqa: E731
    reference = key(views[0])
    same = [r for r in views if key(r) == reference]
    excluded = [r["label"] for r in views if key(r) != reference]
    caps = sorted({c["context_cap"] for r in same for c in r["perf"].get("caps") or []})
    rows = []
    for cap in caps:
        cells = []
        for r in same:
            entry = next((c for c in r["perf"].get("caps") or [] if c["context_cap"] == cap), None)
            result = (entry or {}).get("result") or {}
            saturation = (entry or {}).get("saturation")
            cells.append({"users": _users(result), "status": STATUS_LABELS.get(result.get("status"), ""),
                          "saturation": _saturation(saturation) if saturation else None,
                          "constraint": ((saturation or {}).get("constraint") or result.get("constraint") or {}).get("label")}
                         if entry else None)
        rows.append({"cap": tokens(cap), "cells": cells})
    return {"runs": [r["label"] for r in same], "rows": rows, "excluded": excluded}


def session_views(runs):
    views = [session_view(r) for r in runs if is_sessions(r.get("perf"))]
    return {"views": views, "comparison": comparison(runs)} if views else None
