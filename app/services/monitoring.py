"""Deployment snapshots, scheduled canaries and monitoring events (``canary-v1``).

See docs/design-deployment-monitoring.md. Everything here runs in worker
threads (runner, queue tick) or behind ``run_in_threadpool`` and uses plain
sqlite3 connections, like the run queue.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
import sqlite3

from app.benchmarking import deployment_probe
from app.storage import DETECT_TYPES

logger = logging.getLogger(__name__)

REVISION = "canary-v1"
KEEP_RUNS = 100
MIN_HOURS, MAX_HOURS = 1, 168
MAX_CONCURRENCY = 16
NAME_LIMIT = 120
LATENCY_RATIO = 1.5
TTFT_MARGIN_MS = 250
LATENCY_MARGIN_MS = 1000
RANK = {"ok": 0, "info": 0, "warning": 1, "alert": 2}
STOPPED = "Stopped by administrator."


class MonitoringError(ValueError):
    pass


def _now():
    return datetime.now(timezone.utc)


def _connect():
    from app import config as app_config

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
        return json.loads(raw) if raw else default
    except (TypeError, ValueError):
        return default


# --- Events and notifications -----------------------------------------------------

def _post_webhook(url, payload):
    """Deliver one notification; returns None or an error text."""
    import requests

    from app.services.url_guard import validate_endpoint

    try:
        validate_endpoint(url)
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code >= 300:
            return f"HTTP {response.status_code}"
    except Exception as error:  # noqa: BLE001 - recorded on the event, never retried
        return f"{type(error).__name__}: {error}"[:300]
    return None


def _webhooks(db, model_id, canary_id):
    if canary_id is not None:
        row = _one(db, "SELECT webhook_url FROM canaries WHERE id = ?", (canary_id,))
        return [row["webhook_url"]] if row and row["webhook_url"] else []
    return [r["webhook_url"] for r in _rows(
        db, "SELECT DISTINCT webhook_url FROM canaries WHERE model_id = ? AND webhook_url IS NOT NULL AND enabled = 1",
        (model_id,))]


def _add_event(db, *, model_id, kind, severity, title, detail=None, canary_id=None, run_id=None, check_id=None):
    event_id = db.execute(
        """INSERT INTO monitor_events (model_id, canary_id, run_id, check_id, kind, severity, title, detail_json,
                                       created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (model_id, canary_id, run_id, check_id, kind, severity, title,
         json.dumps(detail, separators=(",", ":")) if detail else None, _now().isoformat())).lastrowid
    return {"id": event_id, "model_id": model_id, "canary_id": canary_id, "severity": severity, "title": title,
            "run_id": run_id}


def _notify(events):
    """Post alert and warning events to the relevant webhooks after the data is committed."""
    pending = [e for e in events if e["severity"] in {"alert", "warning"}]
    if not pending:
        return
    db = _connect()
    try:
        for event in pending:
            urls = _webhooks(db, event["model_id"], event["canary_id"])
            if not urls:
                continue
            model = _one(db, "SELECT name FROM models WHERE id = ?", (event["model_id"],)) or {"name": "deleted model"}
            text = f"[LLM Bench] {event['severity'].upper()} · {model['name']}: {event['title']}"
            if event["run_id"]:
                text += f" (run #{event['run_id']})"
            errors = [error for url in urls if (error := _post_webhook(url, {"text": text}))]
            db.execute("UPDATE monitor_events SET notified = ? WHERE id = ?",
                       ("; ".join(errors)[:500] if errors else "sent", event["id"]))
        db.commit()
    finally:
        db.close()


def acknowledge(event_ids, user_id):
    db = _connect()
    try:
        db.executemany("""UPDATE monitor_events SET acknowledged_by = ?, acknowledged_at = ?
                           WHERE id = ? AND acknowledged_at IS NULL""",
                       [(user_id, _now().isoformat(), int(i)) for i in event_ids])
        db.commit()
    finally:
        db.close()


def events(*, open_only=False, model_id=None, canary_id=None, limit=100):
    clauses, params = [], []
    if open_only:
        clauses.append("e.acknowledged_at IS NULL AND e.severity IN ('alert', 'warning')")
    if model_id is not None:
        clauses.append("e.model_id = ?")
        params.append(model_id)
    if canary_id is not None:
        clauses.append("e.canary_id = ?")
        params.append(canary_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    db = _connect()
    try:
        rows = _rows(db, f"""SELECT e.*, m.name AS model_name, c.name AS canary_name, u.username AS acknowledged_by_name
                               FROM monitor_events e
                               JOIN models m ON m.id = e.model_id
                               LEFT JOIN canaries c ON c.id = e.canary_id
                               LEFT JOIN users u ON u.id = e.acknowledged_by
                               {where} ORDER BY e.id DESC LIMIT ?""", (*params, limit))
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()
    for row in rows:
        row["detail"] = _loads(row.pop("detail_json"), {})
    return rows


def open_counts():
    """Unacknowledged alerts and warnings, for the dashboard banner."""
    db = _connect()
    try:
        rows = _rows(db, """SELECT severity, COUNT(*) AS n FROM monitor_events
                             WHERE acknowledged_at IS NULL AND severity IN ('alert', 'warning')
                             GROUP BY severity""")
    except sqlite3.OperationalError:
        return {"alert": 0, "warning": 0}
    finally:
        db.close()
    counts = {"alert": 0, "warning": 0}
    counts.update({r["severity"]: r["n"] for r in rows})
    return counts


# --- Deployment snapshots ---------------------------------------------------------

def record_check(model, snapshot, run_id=None):
    """Store a snapshot, compare it with the model's previous one, raise change events."""
    db = _connect()
    created = []
    try:
        previous = None
        if snapshot.get("ok"):
            previous = _one(db, """SELECT id, fingerprint, snapshot_json FROM deployment_checks
                                    WHERE model_id = ? AND ok = 1 ORDER BY id DESC LIMIT 1""", (model["id"],))
        changes, changed = [], None
        if previous is not None:
            changes = deployment_probe.diff(_loads(previous["snapshot_json"]), snapshot)
            changed = deployment_probe.classify(changes)
        check_id = db.execute(
            """INSERT INTO deployment_checks (model_id, run_id, revision, created_at, ok, error, fingerprint,
                                              snapshot_json, previous_id, changed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model["id"], run_id, snapshot.get("revision", deployment_probe.REVISION), _now().isoformat(),
             int(bool(snapshot.get("ok"))), snapshot.get("error"), snapshot.get("fingerprint"),
             json.dumps(snapshot, ensure_ascii=False), previous["id"] if previous else None, changed)).lastrowid
        if changed:
            fields = deployment_probe.summary(changes)
            canary = _one(db, "SELECT canary_id FROM test_runs WHERE id = ?", (run_id,)) if run_id else None
            created.append(_add_event(
                db, model_id=model["id"], run_id=run_id, check_id=check_id,
                canary_id=canary["canary_id"] if canary else None,
                kind=f"{changed}_changed", severity="alert" if changed == "deployment" else "info",
                title=f"{'Deployment' if changed == 'deployment' else 'Configuration'} changed: {fields}",
                detail={"changes": changes, "previous_fingerprint": previous["fingerprint"],
                        "fingerprint": snapshot.get("fingerprint"), "previous_check_id": previous["id"]}))
        db.commit()
    finally:
        db.close()
    _notify(created)
    return check_id


def capture(model):
    return deployment_probe.capture(model["base_url"], model["api_key"], model["model_id"])


def capture_for_run(run_id, model):
    """Runner hook: snapshot the deployment before any measured request. Never raises."""
    try:
        record_check(model, capture(model), run_id)
    except Exception:  # noqa: BLE001 - monitoring must never fail a run
        logger.exception("Deployment snapshot failed for run %s", run_id)


def check_model(model_id):
    from app.services.url_guard import validate_endpoint

    db = _connect()
    try:
        model = _one(db, "SELECT * FROM models WHERE id = ?", (model_id,))
    finally:
        db.close()
    if model is None:
        raise MonitoringError("Model not found")
    validate_endpoint(model["base_url"])
    return record_check(model, capture(model))


def _with_diff(check, previous):
    check["snapshot"] = _loads(check.pop("snapshot_json"), {})
    check["changes"] = (deployment_probe.diff(previous["snapshot"], check["snapshot"])
                        if previous and check["ok"] and previous["ok"] else [])
    return check


def model_checks(model_id, limit=100):
    db = _connect()
    try:
        rows = _rows(db, "SELECT * FROM deployment_checks WHERE model_id = ? ORDER BY id DESC LIMIT ?",
                     (model_id, limit + 1))
    finally:
        db.close()
    rows.reverse()
    out, previous = [], None
    for row in rows:
        row = _with_diff(row, previous)
        if row["ok"]:
            previous = row
        out.append(row)
    return list(reversed(out[-limit:] if len(rows) > limit else out))


def run_check(run_id):
    db = _connect()
    try:
        check = _one(db, "SELECT * FROM deployment_checks WHERE run_id = ? ORDER BY id DESC LIMIT 1", (run_id,))
        if check is None:
            return None
        previous = _one(db, "SELECT * FROM deployment_checks WHERE id = ?", (check["previous_id"],)) \
            if check["previous_id"] else None
    except sqlite3.OperationalError:
        return None
    finally:
        db.close()
    if previous:
        previous = _with_diff(previous, None)
    return _with_diff(check, previous)


def latest_checks():
    db = _connect()
    try:
        rows = _rows(db, """SELECT m.id AS model_id, m.name AS model_name, m.model_id AS identifier,
                                   c.id, c.created_at, c.ok, c.error, c.fingerprint, c.changed, c.run_id,
                                   (SELECT COUNT(DISTINCT fingerprint) FROM deployment_checks d
                                     WHERE d.model_id = m.id AND d.ok = 1) AS fingerprints
                              FROM models m
                              LEFT JOIN deployment_checks c
                                ON c.id = (SELECT MAX(id) FROM deployment_checks WHERE model_id = m.id)
                             ORDER BY m.name""")
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()
    return rows


# --- Canaries ---------------------------------------------------------------------

def validate_canary(name, model_id, suite, interval_hours, max_concurrency, webhook_url):
    from app.benchmarking.suites import SUITE_NAMES
    from app.benchmarking.usecase_suites import is_key

    name = (name or "").strip()
    if not name or len(name) > NAME_LIMIT:
        raise MonitoringError(f"Give the canary a name of at most {NAME_LIMIT} characters")
    if suite not in SUITE_NAMES and not is_key(suite):
        raise MonitoringError("Choose a known quality suite")
    try:
        interval_hours, max_concurrency = int(interval_hours), int(max_concurrency)
    except (TypeError, ValueError) as error:
        raise MonitoringError("Interval and concurrency must be whole numbers") from error
    if not MIN_HOURS <= interval_hours <= MAX_HOURS:
        raise MonitoringError(f"The interval must be between {MIN_HOURS} and {MAX_HOURS} hours")
    if not 1 <= max_concurrency <= MAX_CONCURRENCY:
        raise MonitoringError(f"Concurrency must be between 1 and {MAX_CONCURRENCY}")
    webhook_url = (webhook_url or "").strip() or None
    if webhook_url:
        from app.services.url_guard import UnsafeURLError, validate_endpoint

        if not webhook_url.startswith("https://"):
            raise MonitoringError("Webhooks must use https")
        try:
            validate_endpoint(webhook_url)
        except UnsafeURLError as error:
            raise MonitoringError(f"Webhook URL rejected: {error}") from error
    db = _connect()
    try:
        if _one(db, "SELECT id FROM models WHERE id = ?", (model_id,)) is None:
            raise MonitoringError("Unknown model")
    finally:
        db.close()
    return {"name": name, "model_id": int(model_id), "suite": suite, "interval_hours": interval_hours,
            "max_concurrency": max_concurrency, "webhook_url": webhook_url}


def create_canary(user_id, **fields):
    values = validate_canary(**fields)
    db = _connect()
    try:
        canary_id = db.execute(
            """INSERT INTO canaries (name, model_id, suite, max_concurrency, interval_hours, enabled, next_run_at,
                                     webhook_url, created_by, created_at)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)""",
            (values["name"], values["model_id"], values["suite"], values["max_concurrency"],
             values["interval_hours"], _now().isoformat(), values["webhook_url"], user_id,
             _now().isoformat())).lastrowid
        db.commit()
        return canary_id
    finally:
        db.close()


def update_canary(canary_id, **fields):
    values = validate_canary(**fields)
    db = _connect()
    try:
        current = _one(db, "SELECT * FROM canaries WHERE id = ?", (canary_id,))
        if current is None:
            raise MonitoringError("Canary not found")
        # A different model or suite makes the old baseline meaningless.
        baseline = current["baseline_run_id"]
        if (values["model_id"], values["suite"], values["max_concurrency"]) != (
                current["model_id"], current["suite"], current["max_concurrency"]):
            baseline = None
        db.execute("""UPDATE canaries SET name = ?, model_id = ?, suite = ?, max_concurrency = ?, interval_hours = ?,
                                          webhook_url = ?, baseline_run_id = ? WHERE id = ?""",
                   (values["name"], values["model_id"], values["suite"], values["max_concurrency"],
                    values["interval_hours"], values["webhook_url"], baseline, canary_id))
        db.commit()
    finally:
        db.close()


def set_enabled(canary_id, enabled):
    db = _connect()
    try:
        db.execute("UPDATE canaries SET enabled = ?, next_run_at = COALESCE(next_run_at, ?) WHERE id = ?",
                   (1 if enabled else 0, _now().isoformat(), canary_id))
        db.commit()
    finally:
        db.close()


def delete_canary(canary_id):
    """Delete a canary; its runs stay as ordinary runs."""
    db = _connect()
    try:
        if db.execute("SELECT 1 FROM test_runs WHERE canary_id = ? AND status IN ('pending', 'running')",
                      (canary_id,)).fetchone():
            raise MonitoringError("Wait for the canary's current run to finish or stop it first")
        db.execute("UPDATE test_runs SET canary_id = NULL WHERE canary_id = ?", (canary_id,))
        db.execute("DELETE FROM canaries WHERE id = ?", (canary_id,))
        db.commit()
    finally:
        db.close()


def set_baseline(canary_id, run_id):
    db = _connect()
    try:
        run = _one(db, "SELECT status FROM test_runs WHERE id = ? AND canary_id = ?", (run_id, canary_id))
        if run is None or run["status"] != "completed":
            raise MonitoringError("Only a completed run of this canary can be its baseline")
        db.execute("UPDATE canaries SET baseline_run_id = ? WHERE id = ?", (run_id, canary_id))
        db.commit()
    finally:
        db.close()


def _create_run(db, canary):
    from app.benchmarking.vllm_telemetry import metrics_scope
    from app.services.model_settings import decoding_settings
    from app.services.run_submission import make_run_options, quality_config_for

    model = _one(db, "SELECT temperature, reasoning_effort, b300_metrics_model FROM models WHERE id = ?",
                 (canary["model_id"],))
    options = make_run_options(mode="quality", max_concurrency=canary["max_concurrency"], suite=canary["suite"])
    return db.execute(
        """INSERT INTO test_runs (model_id, status, created_by, workers, quality_config_json, run_options_json,
                                  created_at, decoding_config_json, metrics_config_json, canary_id)
           VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)""",
        (canary["model_id"], canary["created_by"], min(4, canary["max_concurrency"]),
         json.dumps(quality_config_for(options)), json.dumps(options), _now().isoformat(),
         json.dumps(decoding_settings(model["temperature"], model["reasoning_effort"], seed=0)),
         json.dumps(metrics_scope(model)), canary["id"])).lastrowid


def _busy(db, canary_id):
    return db.execute("SELECT 1 FROM test_runs WHERE canary_id = ? AND status IN ('pending', 'running')",
                      (canary_id,)).fetchone() is not None


def run_now(canary_id):
    db = _connect()
    try:
        canary = _one(db, "SELECT * FROM canaries WHERE id = ?", (canary_id,))
        if canary is None:
            raise MonitoringError("Canary not found")
        if _busy(db, canary_id):
            raise MonitoringError("This canary already has a queued or running run")
        try:
            run_id = _create_run(db, canary)
        except Exception as error:  # noqa: BLE001 - e.g. a suite that no longer exists
            raise MonitoringError(f"Could not create the canary run: {getattr(error, 'detail', error)}") from error
        db.execute("UPDATE canaries SET next_run_at = ? WHERE id = ?",
                   ((_now() + timedelta(hours=canary["interval_hours"])).isoformat(), canary_id))
        db.commit()
    finally:
        db.close()
    from app.services import run_queue

    run_queue.dispatch_next()
    return run_id


def schedule_due(now=None):
    """Queue a run for every enabled canary that is due; returns the new run ids."""
    now = now or _now()
    created = []
    db = _connect()
    try:
        due = _rows(db, "SELECT * FROM canaries WHERE enabled = 1 AND (next_run_at IS NULL OR next_run_at <= ?)",
                    (now.isoformat(),))
        for canary in due:
            # No backlog: the next slot counts from now, whether or not this one runs.
            db.execute("UPDATE canaries SET next_run_at = ? WHERE id = ?",
                       ((now + timedelta(hours=canary["interval_hours"])).isoformat(), canary["id"]))
            if _busy(db, canary["id"]):
                continue
            try:
                created.append(_create_run(db, canary))
            except Exception:  # noqa: BLE001 - one broken canary must not block the others
                logger.exception("Could not create a run for canary %d", canary["id"])
        db.commit()
    except sqlite3.OperationalError:
        return []  # Database not initialised yet.
    finally:
        db.close()
    return created


# --- Evaluation -------------------------------------------------------------------

def _comparison(baseline, run):
    from app.benchmarking.quality_report import compare_groups

    left, right = _loads(baseline["quality_json"]), _loads(run["quality_json"])
    if not left or not right:
        return {"compatible": False, "reason": "A quality report is missing"}
    return compare_groups([left], [right])


def _latency(baseline, run):
    out = {}
    for key, margin in (("ttft_p50_ms", TTFT_MARGIN_MS), ("latency_p50_ms", LATENCY_MARGIN_MS)):
        before, after = baseline.get(key), run.get(key)
        if before and after:
            out[key] = {"baseline": before, "run": after,
                        "regressed": after > LATENCY_RATIO * before and after - before >= margin}
    return out


def evaluate_run(run_id):
    """Evaluate a finished canary run once; returns its status or None."""
    db = _connect()
    created = []
    try:
        run = _one(db, """SELECT id, status, error_message, quality_json, ttft_p50_ms, latency_p50_ms, error_count,
                                 scored_questions, canary_id, canary_status, model_id
                            FROM test_runs WHERE id = ?""", (run_id,))
        if run is None or run["canary_id"] is None or run["canary_status"] is not None \
                or run["status"] in ("pending", "running"):
            return None
        canary = _one(db, "SELECT * FROM canaries WHERE id = ?", (run["canary_id"],))
        if canary is None:
            return None
        found = []  # (kind, severity, title, detail)
        detail = {"revision": REVISION, "baseline_run_id": canary["baseline_run_id"]}
        if run["status"] != "completed":
            if run["error_message"] == STOPPED:
                found.append(("stopped", "info", "Canary run stopped by an administrator", None))
            else:
                found.append(("run_failed", "alert", f"Canary run failed: {(run['error_message'] or '')[:200]}", None))
        elif not run["scored_questions"]:
            found.append(("run_failed", "alert", "Canary run completed without any scored answers", None))
        else:
            if run["error_count"]:
                found.append(("availability", "warning",
                              f"{run['error_count']} request(s) failed at the endpoint", None))
            baseline = _one(db, "SELECT * FROM test_runs WHERE id = ?", (canary["baseline_run_id"],)) \
                if canary["baseline_run_id"] else None
            if baseline is None or baseline["status"] != "completed":
                db.execute("UPDATE canaries SET baseline_run_id = ? WHERE id = ?", (run_id, canary["id"]))
                detail["baseline_run_id"] = run_id
                found.append(("baseline_set", "info", "This run is now the canary baseline", None))
            elif baseline["id"] != run_id:
                comparison = _comparison(baseline, run)
                detail["comparison"] = {k: comparison.get(k) for k in (
                    "compatible", "reason", "balanced_difference", "ci95", "p_value", "verdict", "paired",
                    "full_pass_discordance")}
                if not comparison.get("compatible"):
                    found.append(("baseline_incompatible", "warning",
                                  f"Cannot compare with the baseline: {comparison.get('reason')}", None))
                else:
                    discord = comparison.get("full_pass_discordance") or {}
                    mcnemar = discord.get("mcnemar_p")
                    worse = comparison["verdict"] == "left_higher" or (
                        mcnemar is not None and mcnemar < 0.05
                        and discord["left_only_pass"] > discord["right_only_pass"])
                    better = comparison["verdict"] == "right_higher"
                    difference = comparison["balanced_difference"] * 100
                    if worse:
                        found.append(("quality_regression", "alert",
                                      f"Quality regressed by {abs(difference):.1f} points against the baseline",
                                      None))
                    elif better:
                        found.append(("quality_improvement", "info",
                                      f"Quality improved by {difference:.1f} points against the baseline", None))
                latency = _latency(baseline, run)
                detail["latency"] = latency
                slow = [k for k, v in latency.items() if v["regressed"]]
                if slow:
                    parts = ", ".join(f"{'TTFT' if k.startswith('ttft') else 'latency'} median "
                                      f"{latency[k]['baseline']:.0f} → {latency[k]['run']:.0f} ms" for k in slow)
                    found.append(("latency_regression", "warning", f"Slower than the baseline: {parts}", None))
        check = _one(db, "SELECT id, changed FROM deployment_checks WHERE run_id = ? ORDER BY id DESC LIMIT 1",
                     (run_id,))
        if check:
            detail["check_id"] = check["id"]
            detail["deployment_changed"] = check["changed"]
        # A deployment change found by this run's snapshot is already an event; count it here.
        status = max([sev for _, sev, _, _ in found] + (["alert"] if check and check["changed"] == "deployment" else []),
                     key=lambda s: RANK[s], default="ok")
        status = "ok" if status == "info" else status
        detail["events"] = [kind for kind, *_ in found]
        db.execute("UPDATE test_runs SET canary_status = ?, canary_json = ? WHERE id = ?",
                   (status, json.dumps(detail, separators=(",", ":")), run_id))
        for kind, severity, title, extra in found:
            created.append(_add_event(db, model_id=run["model_id"], canary_id=canary["id"], run_id=run_id,
                                      kind=kind, severity=severity, title=title, detail=extra or detail))
        _retain(db, canary)
        db.commit()
    finally:
        db.close()
    _notify(created)
    return status


def evaluate_pending():
    """Evaluate finished canary runs that missed their callback (for example after a restart)."""
    db = _connect()
    try:
        ids = [r[0] for r in db.execute(
            """SELECT id FROM test_runs WHERE canary_id IS NOT NULL AND canary_status IS NULL
                AND status NOT IN ('pending', 'running') ORDER BY id""").fetchall()]
    except sqlite3.OperationalError:
        return
    finally:
        db.close()
    for run_id in ids:
        try:
            evaluate_run(run_id)
        except Exception:  # noqa: BLE001
            logger.exception("Canary evaluation failed for run %d", run_id)


def on_run_finished(run_id):
    try:
        evaluate_run(run_id)
    except Exception:  # noqa: BLE001 - never break the queue
        logger.exception("Canary evaluation failed for run %d", run_id)


def tick():
    evaluate_pending()
    schedule_due()


def _retain(db, canary):
    keep = KEEP_RUNS
    rows = _rows(db, """SELECT id, canary_status FROM test_runs WHERE canary_id = ?
                         AND status NOT IN ('pending', 'running') ORDER BY id DESC""", (canary["id"],))
    baseline = _one(db, "SELECT baseline_run_id FROM canaries WHERE id = ?", (canary["id"],))["baseline_run_id"]
    doomed = [r["id"] for r in rows[keep:]
              if r["id"] != baseline and r["canary_status"] not in ("warning", "alert")]
    for run_id in doomed:
        db.execute("DELETE FROM test_results WHERE run_id = ?", (run_id,))
        db.execute("DELETE FROM benchmark_progress WHERE run_id = ?", (run_id,))
        db.execute("DELETE FROM test_runs WHERE id = ?", (run_id,))


# --- Page queries -------------------------------------------------------------------

def canaries_overview():
    db = _connect()
    try:
        canaries = _rows(db, """SELECT c.*, m.name AS model_name FROM canaries c JOIN models m ON m.id = c.model_id
                                 ORDER BY c.name""")
        for canary in canaries:
            recent = _rows(db, """SELECT id, status, canary_status, avg_score, completed_at FROM test_runs
                                   WHERE canary_id = ? ORDER BY id DESC LIMIT 12""", (canary["id"],))
            canary["recent"] = list(reversed(recent))
            canary["last"] = recent[0] if recent else None
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()
    return canaries


def get_canary(canary_id):
    db = _connect()
    try:
        return _one(db, """SELECT c.*, m.name AS model_name, m.model_id AS identifier FROM canaries c
                            JOIN models m ON m.id = c.model_id WHERE c.id = ?""", (canary_id,))
    finally:
        db.close()


def canary_history(canary_id, limit=200):
    db = _connect()
    try:
        rows = _rows(db, """SELECT r.id, r.status, r.created_at, r.completed_at, r.avg_score, r.scored_questions,
                                   r.total_questions, r.error_count, r.ttft_p50_ms, r.latency_p50_ms,
                                   r.canary_status, r.canary_json, r.error_message,
                                   (SELECT fingerprint FROM deployment_checks d WHERE d.run_id = r.id
                                     ORDER BY d.id DESC LIMIT 1) AS fingerprint,
                                   (SELECT changed FROM deployment_checks d WHERE d.run_id = r.id
                                     ORDER BY d.id DESC LIMIT 1) AS changed
                              FROM test_runs r WHERE r.canary_id = ? ORDER BY r.id DESC LIMIT ?""",
                     (canary_id, limit))
    finally:
        db.close()
    for row in rows:
        row["canary"] = _loads(row.pop("canary_json"), {})
    return rows
