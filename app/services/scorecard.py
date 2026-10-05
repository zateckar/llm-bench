"""Model scorecards, decision profiles and decision records (``scorecard-v1``).

See docs/design-decision-dashboard.md. Evidence is assembled on request from
completed runs, deployment checks and monitoring events, so it can never
disagree with them. Completed runs never change, so the parsed parts of their
reports and the paired comparisons between them are cached by run id.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import json
import sqlite3
import threading

from app.benchmarking import decision_gates
from app.benchmarking.decision_gates import GateError
from app.benchmarking.suites import DEFAULT_SUITE, HISTORICAL_SUITE, RETIRED_SUITES
from app.services import run_modes

REVISION = "scorecard-v1"
OLD_DAYS = 90
NAME_LIMIT = 120
DESCRIPTION_LIMIT = 2000
NOTE_LIMIT = 4000
DECISIONS = {"approved": "Approved", "conditional": "Approved with conditions", "rejected": "Rejected"}


class ScorecardError(ValueError):
    pass


class _LRU:
    def __init__(self, size):
        self.size, self.items, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, key, make):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        value = make()
        with self.lock:
            self.items[key] = value
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return value

    def clear(self):
        with self.lock:
            self.items.clear()


_reports = _LRU(512)
_perf = _LRU(1024)
_groups = _LRU(256)
_comparisons = _LRU(2048)


def clear_caches():
    for cache in (_reports, _perf, _groups, _comparisons):
        cache.clear()


def _now():
    return datetime.now(timezone.utc)


def _connect():
    from app import config as app_config
    from app.storage import DETECT_TYPES

    db = sqlite3.connect(str(app_config.DATABASE_PATH), timeout=30, detect_types=DETECT_TYPES)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def _rows(db, query, params=()):
    return [dict(r) for r in db.execute(query, params).fetchall()]


def _one(db, query, params=()):
    row = db.execute(query, params).fetchone()
    return dict(row) if row else None


def _loads(raw, default=None):
    try:
        value = json.loads(raw) if raw else default
    except (TypeError, ValueError):
        return default
    return value if value is not None else default


def parse_time(text):
    """Stored timestamps are SQLite ``CURRENT_TIMESTAMP`` (UTC) or ISO with an offset."""
    if not text:
        return None
    try:
        value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# --- Parsed run parts (cached) ------------------------------------------------------

def _slim_report(db, run_id):
    """The parts of a rescored quality report that summaries and comparisons use."""
    from app.benchmarking.quality_report import achievement_score, rescore_report

    row = _one(db, "SELECT quality_json FROM test_runs WHERE id = ?", (run_id,))
    report = _loads(row and row["quality_json"], {})
    if not isinstance(report, dict) or report.get("schema_version") != 3 or not report.get("results"):
        return None
    report = rescore_report(report)
    keep = ("id", "fingerprint", "category", "family", "scope", "scored", "passed", "outcome")
    return {
        "schema_version": 3, "suite": report.get("suite"), "suite_hash": report.get("suite_hash"),
        "protocol": report.get("protocol"), "summary": report.get("summary") or {},
        "results": [{**{k: r.get(k) for k in keep}, "score": achievement_score(r), "evaluation": None}
                    for r in report["results"]],
    }


def _closed_loop(perf):
    from app.services.html_reports import points
    from app.services.performance_comparison import comparison_point

    levels = {}
    for point in points(perf, "concurrency"):
        concurrency = point.get("concurrency")
        if isinstance(concurrency, int) and point.get("requests"):
            values = comparison_point(point, sweep=False)["values"]
            levels[concurrency] = {k: values[k] for k in ("ttft_p50", "ttft_p95", "latency_p95", "request_output",
                                                          "aggregate_output", "error_rate")}
    protocol = perf.get("protocol") or {}
    return {"kind": "closed_loop", "levels": levels, "input_tokens": protocol.get("input_reference_tokens"),
            "max_output_tokens": protocol.get("max_output_tokens"),
            "peak_output": perf.get("peak_output_tokens_per_sec"), "cancelled": bool(perf.get("cancelled"))}


def _open_loop(perf):
    from app.benchmarking.load_workload import PRESET_LABELS

    protocol, summary = perf.get("protocol") or {}, perf.get("summary") or {}
    preset, workload_hash = protocol.get("preset") or "custom", protocol.get("workload_hash") or ""
    key = preset if preset != "custom" else f"custom:{workload_hash}"
    label = PRESET_LABELS.get(preset, preset) if preset != "custom" else f"Custom workload {workload_hash[:8]}"
    return {"kind": "open_loop", "workload": key, "label": label, "workload_hash": workload_hash,
            "rate": summary.get("sustainable_rate"), "status": summary.get("sustainable_status"),
            "goodput": summary.get("goodput_at_sustainable"), "attainment_target": protocol.get("attainment_target"),
            "cancelled": bool(perf.get("cancelled"))}


def _sweep(db, run_id):
    tokens = None
    for row in db.execute("SELECT context_tokens, result_json FROM performance_cells WHERE run_id = ?", (run_id,)):
        if (_loads(row["result_json"], {}) or {}).get("status") == "measured":
            tokens = max(tokens or 0, row["context_tokens"])
    return {"kind": "sweep", "tokens": tokens}


def _perf_extracts(db, run_id):
    """Latency, context and capacity evidence of one run; a staged run can provide all three."""
    from app.benchmarking.staged_performance import stages_of

    row = _one(db, "SELECT perf_json FROM test_runs WHERE id = ?", (run_id,))
    perf = _loads(row and row["perf_json"], {})
    if not isinstance(perf, dict):
        return []
    stages = stages_of(perf)
    out = []
    if "latency" in stages:
        out.append(_closed_loop(stages["latency"]))
    if "context" in stages:
        out.append(_sweep(db, run_id))
    if "capacity" in stages:
        out.append(_open_loop(stages["capacity"]))
    return out


# --- Evidence -----------------------------------------------------------------------

def suite_label(key):
    from app.benchmarking import usecase_suites
    from app.benchmarking.suites import get_suite, is_builtin

    if is_builtin(key):
        return get_suite(key).label
    if usecase_suites.is_key(key):
        slug, version = usecase_suites.split_key(key)
        row = usecase_suites.load_version(slug, version)
        if row:
            return f"{row['name']} · v{version}"
    return key


def suite_exists(key):
    from app.benchmarking import usecase_suites
    from app.benchmarking.suites import SUITE_NAMES

    if key in SUITE_NAMES:
        return True
    if usecase_suites.is_key(key):
        return usecase_suites.load_version(*usecase_suites.split_key(key)) is not None
    return False


def workload_label(key):
    from app.benchmarking.load_workload import PRESET_LABELS

    if key.startswith("custom:"):
        return f"Custom workload {key[7:15]}"
    return PRESET_LABELS.get(key, key)


def _freshness(run, model_id, run_prints, current, changes):
    printed = run_prints.get(run["id"])
    if printed:
        return ("current" if printed == current.get(model_id) else "stale"), printed
    started = parse_time(run.get("started_at") or run.get("created_at"))
    later = started is not None and any(t and t > started for t in changes.get(model_id, ()))
    return ("stale" if later else "unverified"), None


def _stamp(item, runs, model_id, context, now):
    """Freshness, fingerprint and age of evidence made of ``runs`` (newest first)."""
    labels = [_freshness(r, model_id, *context) for r in runs]
    freshness = ("stale" if any(f == "stale" for f, _ in labels) else
                 "unverified" if any(f == "unverified" for f, _ in labels) else "current")
    at = runs[0].get("completed_at") or runs[0].get("created_at")
    when = parse_time(at)
    age = (now - when).days if when else None
    item.update({"run_id": runs[0]["id"], "freshness": freshness, "fingerprint": labels[0][1], "at": at,
                 "age_days": age, "old": age is not None and age > OLD_DAYS})
    return item


def _single_quality(report, run_id):
    summary = report["summary"]
    return {
        "group_id": None, "runs": 1, "planned": 1, "units": (run_id,), "reports": [report],
        "score": summary.get("category_balanced"), "score_ci95": summary.get("category_balanced_ci95"),
        "full_pass": summary.get("category_balanced_full_pass"), "full_pass_ci95": None,
        "categories": {name: {"score": c.get("score"), "full_pass": c.get("full_pass_rate")}
                       for name, c in (summary.get("categories") or {}).items()},
        "scored": summary.get("scored"), "total": summary.get("total"),
    }


def _group_quality(db, group_id, members):
    """Evidence of a repeat group from its completed members (newest first)."""
    from app.benchmarking.repeat_stats import group_members, group_summary

    entries = []
    for run in reversed(members):
        key = (run["id"], run.get("completed_at"))
        report = _reports.get(key, lambda: _slim_report(db, run["id"]))
        if report:
            entries.append({"run_id": run["id"], "status": "completed", "report": report})
    if not entries:
        return None
    units = tuple(e["run_id"] for e in entries)
    summary = _groups.get(units, lambda: group_summary(entries))
    if not summary.get("available"):
        return None
    usable = group_members(entries)[0]
    if len(usable) == 1:
        return {**_single_quality(usable[0]["report"], usable[0]["run_id"]), "group_id": group_id,
                "planned": max(summary["planned_runs"], members[0].get("repeat_count") or 0)}
    spread, passes = summary["score_spread"], summary["full_pass_spread"]
    return {
        "group_id": group_id, "runs": len(usable), "units": tuple(e["run_id"] for e in usable),
        "planned": max(summary["planned_runs"], members[0].get("repeat_count") or 0),
        "reports": [e["report"] for e in usable],
        # The run-spread interval needs three runs; two fall back to the pooled task bootstrap.
        "score": spread["mean"], "score_ci95": spread["ci95"] or summary.get("pooled_ci95"),
        "full_pass": passes["mean"], "full_pass_ci95": passes["ci95"],
        "categories": {name: {"score": s.get("mean"), "full_pass": None}
                       for name, s in (summary.get("categories") or {}).items()},
        "scored": None, "total": None,
    }


def _operations(db, models):
    out = {m["id"]: {"open_alerts": 0, "open_warnings": 0, "canaries": []} for m in models}
    try:
        for row in _rows(db, """SELECT model_id, severity, COUNT(*) AS n FROM monitor_events
                                 WHERE acknowledged_at IS NULL AND severity IN ('alert', 'warning')
                                 GROUP BY model_id, severity"""):
            if row["model_id"] in out:
                out[row["model_id"]]["open_alerts" if row["severity"] == "alert" else "open_warnings"] = row["n"]
        for row in _rows(db, """SELECT c.id, c.name, c.model_id, c.suite,
                                       (SELECT r.id FROM test_runs r WHERE r.canary_id = c.id
                                          AND r.canary_status IS NOT NULL ORDER BY r.id DESC LIMIT 1) AS run_id,
                                       (SELECT r.canary_status FROM test_runs r WHERE r.canary_id = c.id
                                          AND r.canary_status IS NOT NULL ORDER BY r.id DESC LIMIT 1) AS status
                                  FROM canaries c WHERE c.enabled = 1 ORDER BY c.name"""):
            if row["model_id"] in out:
                out[row["model_id"]]["canaries"].append(row)
    except sqlite3.OperationalError:
        pass
    return out


def collect(now=None):
    """Evidence for every model: ``{"models": [...], "suites": [...], "workloads": [...]}``."""
    now = now or _now()
    db = _connect()
    try:
        models = _rows(db, "SELECT id, name, model_id AS identifier FROM models ORDER BY name, id")
        current, declared, checked = {}, {}, {}
        run_prints, changes = {}, {}
        try:
            for row in _rows(db, """SELECT model_id, fingerprint, snapshot_json, created_at FROM deployment_checks
                                     WHERE id IN (SELECT MAX(id) FROM deployment_checks WHERE ok = 1
                                                  GROUP BY model_id)"""):
                current[row["model_id"]] = row["fingerprint"]
                checked[row["model_id"]] = row["created_at"]
                served = ((_loads(row["snapshot_json"], {}) or {}).get("hard") or {}).get("served") or {}
                if isinstance(served.get("max_model_len"), int):
                    declared[row["model_id"]] = {"tokens": served["max_model_len"], "freshness": "current",
                                                 "fingerprint": row["fingerprint"], "at": row["created_at"],
                                                 "source": "declared"}
            for row in _rows(db, """SELECT run_id, fingerprint FROM deployment_checks
                                     WHERE run_id IS NOT NULL AND ok = 1 ORDER BY id"""):
                run_prints[row["run_id"]] = row["fingerprint"]
            for row in _rows(db, "SELECT model_id, created_at FROM deployment_checks WHERE changed = 'deployment'"):
                changes.setdefault(row["model_id"], []).append(parse_time(row["created_at"]))
        except sqlite3.OperationalError:
            pass
        context = (run_prints, current, changes)
        runs = _rows(db, """SELECT id, model_id, created_at, started_at, completed_at, run_options_json,
                                   repeat_group_id, repeat_count, canary_id, scored_questions,
                                   perf_json IS NOT NULL AS has_perf
                              FROM test_runs WHERE status = 'completed' ORDER BY id DESC""")
        by_model = {}
        for run in runs:
            by_model.setdefault(run["model_id"], []).append(run)
        operations = _operations(db, models)
        suites, workloads = set(), {}
        out = []
        for model in models:
            evidence = {"model": model, "fingerprint": current.get(model["id"]),
                        "last_check_at": checked.get(model["id"]), "declared_context": declared.get(model["id"]),
                        "quality": {}, "capacity": {}, "latency": None, "context": None,
                        "operations": {**operations[model["id"]], "fingerprint": current.get(model["id"])}}
            mine = by_model.get(model["id"], [])
            seen_groups = set()
            for run in mine:
                options = _loads(run["run_options_json"], {}) or {}
                measures_quality, measures_performance = run_modes.parts(options)
                if run["scored_questions"] and measures_quality:
                    suite = options.get("suite") or HISTORICAL_SUITE
                    # Runs of retired built-in suites are history, not a scorecard column.
                    if suite not in RETIRED_SUITES and suite not in evidence["quality"] \
                            and run["repeat_group_id"] not in seen_groups:
                        item = None
                        if run["repeat_group_id"] is not None:
                            seen_groups.add(run["repeat_group_id"])
                            members = [r for r in mine if r["repeat_group_id"] == run["repeat_group_id"]]
                            item = _group_quality(db, run["repeat_group_id"], members)
                            if item:
                                used = set(item["units"])
                                item = _stamp(item, [r for r in members if r["id"] in used], model["id"],
                                              context, now)
                        if item is None:
                            report = _reports.get((run["id"], run["completed_at"]),
                                                  lambda run=run: _slim_report(db, run["id"]))
                            if report:
                                item = _stamp(_single_quality(report, run["id"]), [run], model["id"], context, now)
                        if item:
                            item["suite"] = suite
                            evidence["quality"][suite] = item
                            suites.add(suite)
                if run["has_perf"] and measures_performance:
                    extracts = _perf.get((run["id"], run["completed_at"]), lambda run=run: _perf_extracts(db, run["id"]))
                    for extract in extracts:
                        kind = extract["kind"]
                        if kind == "closed_loop" and evidence["latency"] is None and extract["levels"]:
                            evidence["latency"] = _stamp(dict(extract), [run], model["id"], context, now)
                        elif kind == "sweep" and evidence["context"] is None and extract["tokens"]:
                            evidence["context"] = {**_stamp(dict(extract), [run], model["id"], context, now),
                                                   "source": "measured"}
                        elif kind == "open_loop" and extract["workload"] not in evidence["capacity"]:
                            evidence["capacity"][extract["workload"]] = _stamp(dict(extract), [run], model["id"],
                                                                               context, now)
                            workloads[extract["workload"]] = extract["label"]
            out.append(evidence)
    finally:
        db.close()
    order = {DEFAULT_SUITE: 0}
    return {
        "revision": REVISION, "generated_at": now.isoformat(), "models": out,
        "suites": [{"key": key, "label": suite_label(key)}
                   for key in sorted(suites, key=lambda k: (order.get(k, 2), k))],
        "workloads": [{"key": key, "label": label} for key, label in sorted(workloads.items())],
    }


def model_evidence(collected, model_id):
    return next((m for m in collected["models"] if m["model"]["id"] == model_id), None)


def public(value):
    """Evidence without cached report bodies, for templates' JSON and exports."""
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k not in ("reports", "units")}
    if isinstance(value, (list, tuple)):
        return [public(v) for v in value]
    return value


# --- Ties with the leader -----------------------------------------------------------

def _compare(leader, other):
    from app.benchmarking.quality_report import compare_groups

    key = (leader["units"], other["units"])
    return dict(_comparisons.get(key, lambda: compare_groups(leader["reports"], other["reports"])))


def ties(collected, suite):
    """``{model_id: mark}`` for one quality column; see design decision 4."""
    from app.services.html_reports import _adjust_for_multiplicity

    items = {m["model"]["id"]: m["quality"][suite] for m in collected["models"]
             if suite in m["quality"] and m["quality"][suite].get("score") is not None}
    out = {mid: {"mark": "stale"} for mid, item in items.items() if item["freshness"] == "stale"}
    candidates = {mid: item for mid, item in items.items() if item["freshness"] != "stale"}
    if not candidates:
        return out
    names = {m["model"]["id"]: m["model"]["name"] for m in collected["models"]}
    leader_id = max(candidates, key=lambda mid: (candidates[mid]["score"], [-ord(c) for c in names[mid]]))
    leader = candidates[leader_id]
    out[leader_id] = {"mark": "leader"}
    rows = {mid: _compare(leader, item) for mid, item in candidates.items() if mid != leader_id}
    _adjust_for_multiplicity(list(rows.values()))
    for mid, comparison in rows.items():
        if not comparison.get("compatible"):
            out[mid] = {"mark": "not_comparable", "reason": comparison.get("reason")}
            continue
        out[mid] = {
            "mark": "below" if comparison["verdict"] == "left_higher" else "tied",
            "difference": comparison["balanced_difference"], "ci95": comparison["ci95"],
            "p_adjusted": comparison.get("p_adjusted", comparison.get("p_value")),
            "adjustment": comparison.get("adjustment"),
        }
    return out


# --- Profiles -----------------------------------------------------------------------

def _presets():
    from app.benchmarking.load_workload import PRESETS

    return set(PRESETS)


def _validated(name, description, gates):
    name = (name or "").strip()
    description = (description or "").strip()
    if not name or len(name) > NAME_LIMIT:
        raise ScorecardError(f"The name must be 1–{NAME_LIMIT} characters")
    if len(description) > DESCRIPTION_LIMIT:
        raise ScorecardError(f"The description is limited to {DESCRIPTION_LIMIT:,} characters")
    if isinstance(gates, str):
        try:
            gates = json.loads(gates)
        except ValueError:
            raise ScorecardError("The gates are not valid JSON") from None
    try:
        gates = decision_gates.validate(gates, suite_exists=suite_exists, presets=_presets())
    except GateError as error:
        raise ScorecardError(str(error)) from None
    return name, description, gates


def _profile(row):
    if row is None:
        return None
    row["gates"] = _loads(row.pop("gates_json"), [])
    return row


def list_profiles():
    db = _connect()
    try:
        rows = _rows(db, """SELECT p.*, u.username AS created_by_name,
                                   (SELECT COUNT(*) FROM decision_records d WHERE d.profile_id = p.id) AS decisions
                              FROM decision_profiles p LEFT JOIN users u ON u.id = p.created_by
                             ORDER BY p.name, p.id""")
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()
    return [_profile(r) for r in rows]


def get_profile(profile_id):
    db = _connect()
    try:
        return _profile(_one(db, """SELECT p.*, u.username AS created_by_name FROM decision_profiles p
                                     LEFT JOIN users u ON u.id = p.created_by WHERE p.id = ?""", (profile_id,)))
    finally:
        db.close()


def create_profile(user_id, *, name, description, gates):
    name, description, gates = _validated(name, description, gates)
    now = _now().isoformat()
    db = _connect()
    try:
        profile_id = db.execute(
            """INSERT INTO decision_profiles (name, description, gates_json, created_by, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, description, json.dumps(gates), user_id, now, now)).lastrowid
        db.commit()
    finally:
        db.close()
    return profile_id


def update_profile(profile_id, *, name, description, gates):
    name, description, gates = _validated(name, description, gates)
    db = _connect()
    try:
        changed = db.execute("""UPDATE decision_profiles SET name = ?, description = ?, gates_json = ?, updated_at = ?
                                 WHERE id = ?""",
                             (name, description, json.dumps(gates), _now().isoformat(), profile_id)).rowcount
        db.commit()
    finally:
        db.close()
    if not changed:
        raise ScorecardError("Profile not found")


def delete_profile(profile_id):
    """Delete a profile and its decision records; returns the number of records deleted."""
    db = _connect()
    try:
        records = db.execute("SELECT COUNT(*) FROM decision_records WHERE profile_id = ?", (profile_id,)).fetchone()[0]
        db.execute("DELETE FROM decision_profiles WHERE id = ?", (profile_id,))
        db.commit()
    finally:
        db.close()
    return records


def evaluate_profile(profile, collected):
    """One row per model: ``{"model", "evaluation"}``, models meeting the profile first."""
    rank = {"meets": 0, "incomplete": 1, "fails": 2}
    rows = [{"model": m["model"], "fingerprint": m["fingerprint"],
             "evaluation": decision_gates.evaluate(profile["gates"], m)} for m in collected["models"]]
    return sorted(rows, key=lambda r: (rank[r["evaluation"]["verdict"]], r["model"]["name"]))


def gate_labels(profile, collected=None):
    suites = {s["key"]: s["label"] for s in (collected or {}).get("suites", [])}
    workloads = {w["key"]: w["label"] for w in (collected or {}).get("workloads", [])}
    for gate in profile["gates"]:
        if gate["type"] == "quality" and gate["suite"] not in suites:
            suites[gate["suite"]] = suite_label(gate["suite"])
        if gate["type"] == "capacity" and gate["workload"] not in workloads:
            workloads[gate["workload"]] = workload_label(gate["workload"])
    return [decision_gates.describe(g, suites, workloads) for g in profile["gates"]]


# --- Decision records ---------------------------------------------------------------

def record_decision(profile_id, model_id, *, decision, note, user_id, collected=None):
    if decision not in DECISIONS:
        raise ScorecardError("Choose approved, approved with conditions or rejected")
    note = (note or "").strip()
    if not note or len(note) > NOTE_LIMIT:
        raise ScorecardError(f"A note of 1–{NOTE_LIMIT:,} characters is required")
    profile = get_profile(profile_id)
    if profile is None:
        raise ScorecardError("Profile not found")
    collected = collected or collect()
    evidence = model_evidence(collected, model_id)
    if evidence is None:
        raise ScorecardError("Model not found")
    evaluation = {**decision_gates.evaluate(profile["gates"], evidence),
                  "requirements": gate_labels(profile, collected), "profile_name": profile["name"],
                  "evaluated_at": collected["generated_at"]}
    db = _connect()
    try:
        record_id = db.execute(
            """INSERT INTO decision_records (profile_id, model_id, model_name, decision, note, revision,
                                             evaluation_json, fingerprint, created_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (profile_id, model_id, evidence["model"]["name"], decision, note, decision_gates.REVISION,
             json.dumps(public(evaluation), ensure_ascii=False), evidence["fingerprint"], user_id,
             _now().isoformat())).lastrowid
        db.commit()
    finally:
        db.close()
    return record_id


def decisions(*, profile_id=None, model_id=None, limit=200):
    """Records newest first; ``drifted`` when the model is served from another fingerprint now."""
    clauses, params = [], []
    if profile_id is not None:
        clauses.append("d.profile_id = ?")
        params.append(profile_id)
    if model_id is not None:
        clauses.append("d.model_id = ?")
        params.append(model_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    db = _connect()
    try:
        rows = _rows(db, f"""SELECT d.*, u.username AS created_by_name, p.name AS profile_name,
                                    (SELECT c.fingerprint FROM deployment_checks c WHERE c.model_id = d.model_id
                                       AND c.ok = 1 ORDER BY c.id DESC LIMIT 1) AS current_fingerprint
                               FROM decision_records d
                               LEFT JOIN users u ON u.id = d.created_by
                               JOIN decision_profiles p ON p.id = d.profile_id
                               {where} ORDER BY d.id DESC LIMIT ?""", (*params, limit))
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()
    latest = set()
    for row in rows:
        row["evaluation"] = _loads(row.pop("evaluation_json"), {})
        row["decision_label"] = DECISIONS.get(row["decision"], row["decision"])
        row["drifted"] = bool(row["fingerprint"] and row["current_fingerprint"]
                              and row["fingerprint"] != row["current_fingerprint"])
        key = (row["profile_id"], row["model_id"])
        row["superseded"] = key in latest
        latest.add(key)
    return rows


# --- A/B studies involving a model ------------------------------------------------------

def studies_for(model_id, limit=10):
    from app.services import ab_studies

    db = _connect()
    try:
        rows = _rows(db, """SELECT s.id, s.name, s.label_a, s.label_b, s.suite_name, s.status, s.created_at,
                                   ra.model_id AS model_a, rb.model_id AS model_b
                              FROM ab_studies s
                              LEFT JOIN test_runs ra ON ra.id = s.run_a
                              LEFT JOIN test_runs rb ON rb.id = s.run_b
                             WHERE ra.model_id = ? OR rb.model_id = ?
                             ORDER BY s.id DESC LIMIT ?""", (model_id, model_id, limit))
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()
    out = []
    for row in rows:
        side = "a" if row["model_a"] == model_id else "b"
        result = ab_studies.results(row["id"])

        def outcome(preference):
            verdict = preference.get("verdict")
            if verdict == "no_data":
                return "no_data"
            if verdict == "unclear":
                return "unclear"
            return "preferred" if verdict == side else "not_preferred"

        out.append({**row, "side": side, "opponent": row["label_b"] if side == "a" else row["label_a"],
                    "people": {"outcome": outcome(result["people"]), "votes": result["people"]["votes"],
                               "pairs": result["people"]["pairs"]},
                    "judges": [{"name": j["model_name"], "status": j["status"],
                                "outcome": outcome(j["preference"])} for j in result["judges"]]})
    return out


# --- Overviews ----------------------------------------------------------------------------

def profile_overview(collected=None):
    """Every profile with its verdict counts and the models meeting it (dashboard and matrix)."""
    profiles = list_profiles()
    if not profiles:
        return []
    collected = collected or collect()
    out = []
    for profile in profiles:
        rows = evaluate_profile(profile, collected)
        counts = {"meets": 0, "incomplete": 0, "fails": 0}
        for row in rows:
            counts[row["evaluation"]["verdict"]] += 1
        out.append({"id": profile["id"], "name": profile["name"], "gates": len(profile["gates"]),
                    "counts": counts, "verdicts": {r["model"]["id"]: r["evaluation"]["verdict"] for r in rows},
                    "meeting": [r["model"]["name"] for r in rows if r["evaluation"]["verdict"] == "meets"]})
    return out
