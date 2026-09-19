"""Sequential benchmark run queue and scheduler.

Owns the single active-run slot: at most one benchmark executes at any
moment. Runs are persisted (``test_runs.status='pending'`` with their full
parameters in ``run_options_json``), so the queue survives restarts and this
module only holds the transient "which run is currently active" pointer.

Every spawn of a benchmark thread goes through :func:`enqueue_run` /
:func:`dispatch_next` — never through ``benchmark_runner.start_benchmark``
directly — so the single-run invariant has exactly one choke point. As a
second line of defence a dispatch is skipped whenever the database shows a
run still in ``running`` state.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_active_run_id: int | None = None  # None = queue slot free
_TICK_SECONDS = 15.0
_tick_started = False


def _connect() -> sqlite3.Connection:
    from app.config import DATABASE_PATH

    db = sqlite3.connect(str(DATABASE_PATH), timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def active_run_id() -> int | None:
    """Return the id of the run occupying the slot, or None."""
    with _LOCK:
        return _active_run_id


def _any_running(db: sqlite3.Connection) -> bool:
    return db.execute(
        "SELECT 1 FROM test_runs WHERE status = 'running' LIMIT 1"
    ).fetchone() is not None


def _next_pending_run(db: sqlite3.Connection) -> sqlite3.Row | None:
    """Oldest startable pending run: planless runs and due plans.

    Ordering key: planless runs effectively scheduled "at creation" share the
    front of the queue by run id; runs of a plan order by their plan's
    scheduled_at (due plans only reach here), then run id.
    """
    now = datetime.now(timezone.utc).isoformat()
    return db.execute(
        """SELECT tr.* FROM test_runs tr
              LEFT JOIN run_plans p ON p.id = tr.plan_id
            WHERE tr.status = 'pending'
              AND tr.run_options_json IS NOT NULL
              AND (tr.plan_id IS NULL
                   OR (p.status = 'active'
                       AND (p.scheduled_at IS NULL OR p.scheduled_at <= ?)))
            ORDER BY p.scheduled_at IS NOT NULL, p.scheduled_at, tr.id
            LIMIT 1""",
        (now,),
    ).fetchone()


def _is_due(row: sqlite3.Row, plan: sqlite3.Row | None) -> bool:
    """Whether a pending run may start now: no plan, cancelled plans are
    excluded upstream, active plans must be unscheduled or due."""
    if plan is None:
        return True
    if plan["status"] != "active":
        return False
    due = plan["scheduled_at"]
    return due is None or due <= datetime.now(timezone.utc).isoformat()


def _try_start(run_id: int) -> bool:
    """Claim the slot and spawn the benchmark thread for a pending run.

    Returns True when the run was started. All guards live here so no caller
    can bypass the single-run invariant.
    """
    global _active_run_id
    db = _connect()
    try:
        row = db.execute(
            """SELECT tr.status, tr.run_options_json, tr.plan_id, m.* FROM test_runs tr
                  JOIN models m ON m.id = tr.model_id
                WHERE tr.id = ?""",
            (run_id,),
        ).fetchone()
        # A Stop request while the run sat in the queue already made it
        # terminal; it must never start.
        if row is None or row["status"] != "pending":
            return False
        if row["run_options_json"] is None:
            # Legacy row from before the queue existed: no parameters to
            # re-dispatch with, so fail it instead of running defaults.
            db.execute(
                """UPDATE test_runs
                      SET status = 'failed', completed_at = ?,
                          error_message = 'Queued before run parameters were persisted.'
                    WHERE id = ? AND status = 'pending'""",
                (datetime.now(timezone.utc).isoformat(), run_id),
            )
            db.commit()
            return False
        plan = None
        if row["plan_id"] is not None:
            plan = db.execute(
                "SELECT status, scheduled_at FROM run_plans WHERE id = ?",
                (row["plan_id"],),
            ).fetchone()
        if not _is_due(row, plan):
            # Scheduled for later: the tick dispatches it when due.
            return False
        model_keys = {
            "status", "run_options_json", "plan_id"}
        model = {k: row[k] for k in row.keys() if k not in model_keys}
        options = json.loads(row["run_options_json"])
    finally:
        db.close()

    with _LOCK:
        if _active_run_id is not None:
            return False
        db = _connect()
        try:
            if _any_running(db):
                # Belt and braces: the memory slot may be free while an
                # orphaned 'running' row remains; do not start a second run.
                return False
            # Re-check under the lock: a Stop in the window between the first
            # read and the claim must still win.
            still_pending = db.execute(
                "SELECT 1 FROM test_runs WHERE id = ? AND status = 'pending'",
                (run_id,),
            ).fetchone()
            if still_pending is None:
                return False
        finally:
            db.close()
        # Claim the slot while still under the lock, then spawn outside it:
        # a slow import must not hold up other dispatchers.
        _active_run_id = run_id

    options["concurrency_levels"] = tuple(options.get("concurrency_levels") or (1, 2, 4, 8))
    options["context_sizes"] = tuple(options.get("context_sizes") or ())
    try:
        _spawn_benchmark(run_id, model, options)
    except Exception as e:  # noqa: BLE001 - a spawn failure frees the slot again
        logger.exception("Could not spawn benchmark run %d", run_id)
        with _LOCK:
            if _active_run_id == run_id:
                _active_run_id = None
        # A deterministic spawn failure (corrupt stored options) would be
        # retried by every tick and pin the queue head forever, so the run is
        # terminal here and the queue moves on.
        db = _connect()
        try:
            db.execute(
                """UPDATE test_runs
                      SET status = 'failed', completed_at = ?,
                          error_message = ?
                    WHERE id = ? AND status = 'pending'""",
                (datetime.now(timezone.utc).isoformat(),
                 f"Could not start run: {e}", run_id),
            )
            db.commit()
        finally:
            db.close()
        return False
    logger.info("Started benchmark run %d from the queue", run_id)
    return True


def _spawn_benchmark(run_id: int, model: dict, options: dict) -> None:
    from app.services import benchmark_runner

    benchmark_runner.start_benchmark(run_id, model, on_finish=on_run_finished, **options)


def enqueue_run(run_id: int) -> bool:
    """Queue a run and start it immediately when the slot is free."""
    try:
        return _try_start(run_id)
    except Exception:  # noqa: BLE001 - a dispatch bug must not kill the request
        logger.exception("enqueue_run(%d) failed", run_id)
        return False


def on_run_finished(run_id: int) -> None:
    """Release the slot and dispatch the next due run. Called from the runner
    thread's finally, so it must never raise."""
    global _active_run_id
    try:
        with _LOCK:
            if _active_run_id == run_id:
                _active_run_id = None
        dispatch_next()
    except Exception:  # noqa: BLE001
        logger.exception("on_run_finished(%d) failed", run_id)


def _clear_wedged_slot() -> None:
    """Free the slot when the recorded run is already terminal in the DB.

    Covers any path that somehow bypasses the runner's finally, keeping the
    queue self-healing; normally on_run_finished has already cleared it.
    """
    global _active_run_id
    with _LOCK:
        run_id = _active_run_id
    if run_id is None:
        return
    try:
        db = _connect()
        try:
            row = db.execute(
                "SELECT status FROM test_runs WHERE id = ?", (run_id,)
            ).fetchone()
        finally:
            db.close()
        if row is None or row["status"] not in ("pending", "running"):
            logger.warning(
                "Clearing wedged queue slot: run %d is '%s'",
                run_id, row["status"] if row else "missing",
            )
            with _LOCK:
                if _active_run_id == run_id:
                    _active_run_id = None
    except Exception:  # noqa: BLE001
        logger.exception("Wedge check failed for run %d", run_id)


def dispatch_next() -> None:
    """Start the next due pending run when the queue slot is free."""
    _clear_wedged_slot()
    if active_run_id() is not None:
        return
    try:
        db = _connect()
        try:
            if _any_running(db):
                return
            row = _next_pending_run(db)
        finally:
            db.close()
        if row is not None:
            _try_start(row["id"])
    except Exception:  # noqa: BLE001
        logger.exception("dispatch_next failed")


def cancel_plan(plan_id: int) -> int:
    """Cancel a plan: fail all its still-pending runs. Returns how many."""
    db = _connect()
    try:
        db.execute(
            "UPDATE run_plans SET status = 'cancelled' WHERE id = ?", (plan_id,)
        )
        cursor = db.execute(
            """UPDATE test_runs
                  SET status = 'failed', completed_at = ?,
                      error_message = 'Plan cancelled.'
                WHERE plan_id = ? AND status = 'pending'""",
            (datetime.now(timezone.utc).isoformat(), plan_id),
        )
        db.commit()
        return cursor.rowcount
    finally:
        db.close()


def delete_plan(plan_id: int) -> None:
    """Delete a plan row, detaching and stopping its runs.

    Pending/running member runs are failed first — a run must never start or
    continue for a plan that no longer exists — and every member run's
    plan_id is cleared so finished runs keep their results without a dangling
    FK into run_plans.
    """
    now = datetime.now(timezone.utc).isoformat()
    db = _connect()
    try:
        member_ids = [
            r[0]
            for r in db.execute(
                "SELECT id FROM test_runs WHERE plan_id = ?", (plan_id,)
            ).fetchall()
        ]
        db.execute(
            """UPDATE test_runs
                  SET status = 'failed', completed_at = ?,
                      error_message = 'Plan deleted.'
                WHERE plan_id = ? AND status IN ('pending', 'running')""",
            (now, plan_id),
        )
        db.execute(
            "UPDATE test_runs SET plan_id = NULL WHERE plan_id = ?", (plan_id,)
        )
        if member_ids:
            placeholders = ", ".join("?" for _ in member_ids)
            db.execute(
                f"DELETE FROM benchmark_progress WHERE run_id IN ({placeholders})",
                tuple(member_ids),
            )
        db.execute("DELETE FROM run_plans WHERE id = ?", (plan_id,))
        db.commit()
    finally:
        db.close()


def _tick_loop() -> None:
    while True:
        time.sleep(_TICK_SECONDS)
        try:
            dispatch_next()
        except Exception:  # noqa: BLE001
            logger.exception("Queue tick failed")


def resume() -> None:
    """Called once at app startup: relaunch due pending runs and start the
    tick thread that fires future-dated plans."""
    global _tick_started
    dispatch_next()
    with _LOCK:
        if _tick_started:
            return
        _tick_started = True
    threading.Thread(target=_tick_loop, daemon=True, name="run-queue-tick").start()
