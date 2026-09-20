#!/usr/bin/env python3
"""Self-tests for the sequential run queue in app/services/run_queue.py.

Pins the behaviours that are easy to silently get wrong:

* two enqueues while one run occupies the slot - the second stays pending and
  starts when the first finishes;
* a plan scheduled in the future is not dispatched before its time and is
  dispatched after;
* a run stopped (marked failed) while queued is skipped, never started;
* cancelling a plan fails its queued runs and they are never dispatched;
* wedge recovery: if the slot's recorded run is terminal in the DB, the next
  dispatch clears the slot and moves on.

No threads and no network: ``run_queue._spawn_benchmark`` is stubbed so
dispatch decisions run synchronously, against a temporary SQLite database
carrying the real schema plus the app migrations.

Usage:
    python selftest_run_queue.py
Exit code 0 if every case behaves as specified.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
        return
    print(f"  FAIL {name}" + (f" - {detail}" if detail else ""))
    FAILURES.append(name)


# ---------------------------------------------------------------------------
# Environment: point app.config at a scratch DB before importing the modules
# under test. The queue reads and writes from short-lived connections, so the
# test connections run in autocommit mode to keep them mutually visible.
# ---------------------------------------------------------------------------

# Windows keeps the SQLite file open through pooled connections well after the
# last close, so the tempdir teardown would raise a harmless PermissionError at
# interpreter exit. Ignoring cleanup errors removes the need for the os._exit(0)
# this file used to end with -- that skipped the stdout flush, so every "ok"
# line was silently dropped whenever output was piped rather than a terminal.
_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
db_path = Path(_tmp.name) / "bench.db"


class _AutocommitConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("isolation_level", None)
        super().__init__(*args, **kwargs)


_real_connect = sqlite3.connect


def _test_connect(*args, **kwargs):
    kwargs["factory"] = _AutocommitConnection
    return _real_connect(*args, **kwargs)


import app.config as app_config  # noqa: E402

_real_data_dir, _real_db_path = app_config.DATA_DIR, app_config.DATABASE_PATH
app_config.DATA_DIR = Path(_tmp.name)
app_config.DATABASE_PATH = db_path

sqlite3.connect = _test_connect

import app.database as app_database  # noqa: E402
import app.services.run_queue as run_queue  # noqa: E402


def _db() -> sqlite3.Connection:
    db = _real_connect(str(db_path), factory=_AutocommitConnection, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def add_run(plan_id: int | None = None, *, status: str = "pending") -> int:
    with _db() as db:
        return db.execute(
            """INSERT INTO test_runs (model_id, status, run_options_json, plan_id)
               VALUES (1, ?, '{}', ?)""",
            (status, plan_id),
        ).lastrowid


def add_plan(scheduled_at: str | None, status: str = "active") -> int:
    with _db() as db:
        return db.execute(
            "INSERT INTO run_plans (name, scheduled_at, status) VALUES (NULL, ?, ?)",
            (scheduled_at, status),
        ).lastrowid


def run_status(run_id: int) -> str:
    with _db() as db:
        return db.execute(
            "SELECT status FROM test_runs WHERE id=?", (run_id,)
        ).fetchone()["status"]


def set_status(run_id: int, status: str) -> None:
    with _db() as db:
        db.execute("UPDATE test_runs SET status=? WHERE id=?", (status, run_id))


class _Recorder:
    """Namespace so the stub always sees the live list, not a snapshot."""

    started: list[int] = []


_rec = _Recorder()
_original_spawn = run_queue._spawn_benchmark
run_queue._spawn_benchmark = lambda run_id, model, options: _rec.started.append(run_id)


def finish(run_id: int, status: str = "completed") -> None:
    """Simulate the benchmark thread ending: terminal status + callback."""
    set_status(run_id, status)
    run_queue.on_run_finished(run_id)


def reset(started: list[str] | None = None) -> list[int]:
    """Clear the slot and the recording, and terminate any still-open runs."""
    with _db() as db:
        db.execute(
            "UPDATE test_runs SET status='failed' WHERE status IN ('pending','running')"
        )
    run_queue._active_run_id = None
    _rec.started.clear()
    return []


setup_done = False
if not setup_done:
    asyncio.run(app_database.init_db())
    with _db() as db:
        db.execute(
            "INSERT INTO models (name, base_url, api_key, model_id) VALUES ('m1', 'http://stub.invalid/v1', 'k', 'm1')"
        )
    setup_done = True

# --- 1. FIFO: second run waits, starts when the first finishes --------------
print("queue FIFO")
reset()
r1, r2 = add_run(), add_run()
check("first run starts", run_queue.enqueue_run(r1) and _rec.started == [r1])
check("slot is held while run 1 is active", run_queue.active_run_id() == r1)
check("second run stays queued",
      not run_queue.enqueue_run(r2) and _rec.started == [r1])
check("second run still pending in DB", run_status(r2) == "pending")
finish(r1)
check("finishing run 1 starts run 2", _rec.started == [r1, r2])

# --- 2. Future-dated plan holds, then fires ---------------------------------
print("scheduled plans")
reset()
future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
r_future = add_run(add_plan(future))
r_past = add_run(add_plan(past))
r_plain = add_run()
run_queue.dispatch_next()
check("planless runs dispatch first", _rec.started == [r_plain],
      f"started={_rec.started}")
run_queue.dispatch_next()
check("a tick never starts a second run", _rec.started == [r_plain])
finish(r_plain)
check("due plan dispatches next", _rec.started == [r_plain, r_past],
      f"started={_rec.started}")
reset()  # fails the still-pending future-plan run too; re-queue it fresh below
r_future2 = add_run(add_plan(future))
run_queue.dispatch_next()
check("future plan is not dispatchable while it is the only choice",
      _rec.started == [], f"started={_rec.started}")
check("future plan run stays pending", run_status(r_future2) == "pending")

# --- 3. Stopped queued run is skipped ---------------------------------------
print("stop while queued")
reset()
r_a, r_b = add_run(), add_run()
run_queue.enqueue_run(r_a)
set_status(r_b, "failed")  # what /runs/{id}/stop does to a queued run
finish(r_a, "failed")
check("stopped queued run never starts", _rec.started == [r_a],
      f"started={_rec.started}")

# --- 4. Plan cancellation ----------------------------------------------------
print("plan cancel")
reset()
p = add_plan(None)
r_c, r_d = add_run(p), add_run(p)
cancelled = run_queue.cancel_plan(p)
check("cancel fails the queued runs", cancelled == 2, f"cancelled={cancelled}")
check("queued runs are failed in DB",
      run_status(r_c) == "failed" and run_status(r_d) == "failed")
run_queue.dispatch_next()
check("cancelled plan runs are never dispatched", _rec.started == [],
      f"started={_rec.started}")

# --- 5. Wedge recovery --------------------------------------------------------
print("wedge recovery")
reset()
r_w = add_run(status="running")          # slot's run, already terminal below
run_queue._active_run_id = r_w
set_status(r_w, "failed")                # ...but the run is terminal in the DB
r_next = add_run()
run_queue.dispatch_next()
check("wedged slot clears and next run starts", _rec.started == [r_next],
      f"started={_rec.started}")

run_queue._spawn_benchmark = _original_spawn
# Undo the global patches: this module is picked up by `unittest discover` under
# the selftest_*.py pattern, and a leaked sqlite3.connect or scratch DATABASE_PATH
# would follow every module loaded after it.
sqlite3.connect = _real_connect
app_config.DATA_DIR, app_config.DATABASE_PATH = _real_data_dir, _real_db_path

print()
if FAILURES:
    print(f"{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
    sys.exit(1)
print("All run-queue self-tests passed.")
