"""Lossless storage migration of the configured server database before dispatch.

Offline callers must stop the application. Startup callers run after init_db
and before serving requests or resuming any benchmark/judge threads.
"""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from app.storage import DETECT_TYPES, MAGIC, LZMA_MAGIC, pack_text, unpack_text

STORAGE_REVISION = "lossless-lzma-v1"
PAYLOADS = {
    "test_results": ("prompt", "response", "quality_metadata_json"),
    "test_runs": ("quality_json", "perf_json", "quality_summary_json"),
    "performance_cells": ("result_json",),
    "usecase_suites": ("source_yaml",),
    "ab_pairs": ("prompt", "system_prompt", "reference", "answer_a", "answer_b"),
}


def inspect_database(path):
    path = Path(path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        files = {"database": path.stat().st_size}
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar = Path(str(path) + suffix)
            files[suffix[1:]] = sidecar.stat().st_size if sidecar.exists() else 0
        backups = list((path.parent / "backups").glob(f"{path.stem}-*.db.gz"))
        backup_bytes = sum(item.stat().st_size for item in backups)
        return {"bytes": files["database"], "files": files,
                "active_bytes": sum(files.values()),
                "local_backups": {"files": len(backups), "bytes": backup_bytes},
                "active_and_local_backup_bytes": sum(files.values()) + backup_bytes,
                "free_bytes": db.execute("PRAGMA freelist_count").fetchone()[0]
                * db.execute("PRAGMA page_size").fetchone()[0],
                "runs": db.execute("SELECT COUNT(*) FROM test_runs").fetchone()[0],
                "results": db.execute("SELECT COUNT(*) FROM test_results").fetchone()[0]}


def backup_database(path, backup_dir):
    """Called with the writer reserved; a separate reader copies its snapshot."""
    backup_dir = Path(backup_dir).resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backup_dir / f"{path.stem}-{stamp}.db.gz"
    # Do not leave a partial archive looking like a restorable backup.
    partial = backup.with_suffix(backup.suffix + ".partial")
    try:
        with tempfile.TemporaryDirectory(prefix="bench-backup-", dir=backup_dir) as scratch:
            snapshot = Path(scratch) / "snapshot.db"
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
                with closing(sqlite3.connect(snapshot)) as target:
                    source.backup(target)
                    check_integrity(target)
            with snapshot.open("rb") as source, partial.open("xb") as dest:
                with gzip.GzipFile(fileobj=dest, mode="wb", compresslevel=6, mtime=0) as zipped:
                    shutil.copyfileobj(source, zipped)
        partial.replace(backup)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return backup


def check_integrity(db):
    if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise RuntimeError("SQLite integrity check failed")
    if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise RuntimeError("SQLite foreign-key check failed")


def compress_payloads(db):
    """Repack one value at a time, preserving its exact decoded bytes."""
    changed = 0
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, columns in PAYLOADS.items():
        if table not in tables:
            continue
        existing = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
        for column in columns:
            if column not in existing:
                continue
            # Raw reads distinguish empty TEXT from NULL (declared-type
            # converters map empty TEXT to None), and literal marker prefixes.
            for row_id, raw, kind in db.execute(
                    f"SELECT rowid,CAST({column} AS BLOB),typeof({column}) FROM {table} "
                    f"WHERE {column} IS NOT NULL"):
                value = raw.decode("utf-8") if kind == "text" else unpack_text(raw)
                packed = pack_text(value)
                if unpack_text(packed) != value:
                    raise RuntimeError("Compression roundtrip verification failed")
                if isinstance(packed, bytes) and (len(packed) < len(raw) or kind == "text"
                                                 and (not raw or raw.startswith((MAGIC, LZMA_MAGIC)))):
                    db.execute(f"UPDATE {table} SET {column}=? WHERE rowid=?", (packed, row_id))
                    changed += 1
    return changed


@contextmanager
def maintenance_lock(path):
    """Prevent two new server processes from compacting/resuming concurrently."""
    with Path(str(path) + ".storage-maintenance.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def migrate_storage(path, *, startup=False, force=False, backup_dir=None):
    """Versioned startup migration; resume a committed-but-unvacuumed attempt.

    Pending runs are allowed only before startup dispatch. No historical scores
    are recalculated, and no records or backups are deleted.
    """
    path = Path(path).resolve()
    backup_dir = Path(backup_dir or path.parent / "backups").resolve()
    with maintenance_lock(path), closing(sqlite3.connect(
            path.as_uri() + "?mode=rw", uri=True, timeout=30, detect_types=DETECT_TYPES)) as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("""CREATE TABLE IF NOT EXISTS database_storage_migrations (
                revision TEXT PRIMARY KEY, state TEXT NOT NULL, report_json TEXT NOT NULL)""")
            previous = db.execute("SELECT state,report_json FROM database_storage_migrations WHERE revision=?",
                                  (STORAGE_REVISION,)).fetchone()
            if previous and previous[0] == "complete" and not force:
                db.rollback()
                return {"revision": STORAGE_REVISION, "skipped": True}
            statuses = ("running",) if startup else ("running", "pending")
            marks = ",".join("?" for _ in statuses)
            if db.execute(f"SELECT 1 FROM test_runs WHERE status IN ({marks}) LIMIT 1", statuses).fetchone():
                raise RuntimeError("Database storage migration requires a stopped application and idle workers")
            if db.execute("SELECT 1 FROM ab_judges WHERE status='running' LIMIT 1").fetchone():
                raise RuntimeError("Database storage migration requires idle judge workers")
            if previous and previous[0] == "compressed":
                report = json.loads(previous[1])
            else:
                before = inspect_database(path)
                # Reserve room for backup snapshot + gzip and the rewrite/WAL.
                # VACUUM also needs a temporary database; estimates are conservative.
                backup_dir.mkdir(parents=True, exist_ok=True)
                required = 3 * before["active_bytes"]
                if shutil.disk_usage(path.parent).free < required:
                    raise RuntimeError(f"Storage migration needs at least {required} free bytes on the database filesystem")
                if shutil.disk_usage(backup_dir).free < 2 * before["active_bytes"]:
                    raise RuntimeError("Insufficient free space for the database backup")
                check_integrity(db)
                backup = backup_database(path, backup_dir)
                report = {"revision": STORAGE_REVISION, "before": before,
                          "backup": str(backup), "backup_bytes": backup.stat().st_size,
                          "compressed_payloads": compress_payloads(db)}
                check_integrity(db)
                db.execute("INSERT OR REPLACE INTO database_storage_migrations VALUES (?, 'compressed', ?)",
                           (STORAGE_REVISION, json.dumps(report)))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        # Completion is recorded only after physical reclamation succeeds.
        # On interruption the committed 'compressed' state reuses its backup.
        checkpoint = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint[0]:
            raise RuntimeError("Storage migration blocked by another database connection; stop all app processes")
        db.execute("VACUUM")
        if db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
            raise RuntimeError("Storage migration could not truncate the WAL")
        check_integrity(db)
        db.execute("UPDATE database_storage_migrations SET state='complete' WHERE revision=?", (STORAGE_REVISION,))
        db.commit()
        if db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
            raise RuntimeError("Storage migration could not checkpoint its completion")
        report["after"] = inspect_database(path)
        report["active_plus_backup_bytes"] = report["after"]["active_bytes"] + report["backup_bytes"]
        return report
