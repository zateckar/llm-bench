"""Inspect or losslessly compress benchmark payloads and refresh saved scores.

Run with --apply to mutate an idle database. A compressed SQLite backup is made
under data/backups first. No model answers, criteria or timing samples are pruned.
"""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from app.storage import DETECT_TYPES, pack_text, unpack_text
from app.benchmarking.quality_report import achievement_score, rescore_report

PAYLOADS = {
    "test_results": ("prompt", "response", "quality_metadata_json"),
    "test_runs": ("quality_json", "perf_json"),
    "performance_cells": ("result_json",),
}


def inspect_database(path):
    path = Path(path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        return {"bytes": path.stat().st_size,
                "free_bytes": db.execute("PRAGMA freelist_count").fetchone()[0]
                * db.execute("PRAGMA page_size").fetchone()[0],
                "runs": db.execute("SELECT COUNT(*) FROM test_runs").fetchone()[0],
                "results": db.execute("SELECT COUNT(*) FROM test_results").fetchone()[0]}


def maintain_database(path, backup_dir=None):
    path = Path(path).resolve()
    before = inspect_database(path)
    backup_dir = Path(backup_dir or path.parent / "backups").resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=30,
                                 detect_types=DETECT_TYPES)) as db:
        # Reserve the writer slot throughout backup and migration. A dispatcher
        # cannot start a run between our idle check and the updates.
        db.execute("BEGIN IMMEDIATE")
        backup = None
        changed = 0
        rescored = []
        try:
            if db.execute("SELECT 1 FROM test_runs WHERE status IN ('running','pending') LIMIT 1").fetchone():
                raise RuntimeError("Database maintenance requires an idle run queue")
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = backup_dir / f"{path.stem}-{stamp}.db.gz"
            with tempfile.TemporaryDirectory(prefix="bench-backup-", dir=backup_dir) as scratch:
                snapshot = Path(scratch) / "snapshot.db"
                with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
                    with closing(sqlite3.connect(snapshot)) as target:
                        source.backup(target)
                with snapshot.open("rb") as source, backup.open("xb") as dest:
                    with gzip.GzipFile(fileobj=dest, mode="wb", compresslevel=6, mtime=0) as zipped:
                        shutil.copyfileobj(source, zipped)
            # Keep one canonical fractional score in the indexed result rows.
            columns = {row[1] for row in db.execute("PRAGMA table_info(test_results)")}
            if "quality_outcome" not in columns:
                db.execute("ALTER TABLE test_results ADD COLUMN quality_outcome TEXT")
            db.execute("UPDATE test_results SET quality_outcome='' WHERE quality_metadata_json IS NULL AND quality_outcome IS NULL")
            for row_id, raw in db.execute(
                "SELECT id,quality_metadata_json FROM test_results WHERE quality_metadata_json IS NOT NULL"
            ).fetchall():
                record = json.loads(raw)
                record.setdefault("evaluator_score", record.get("score", 0))
                record["score"] = achievement_score(record)
                payload = pack_text(json.dumps(record, separators=(",", ":")))
                db.execute("UPDATE test_results SET score=?,quality_metadata_json=?,quality_outcome=? WHERE id=?",
                           (record["score"], payload, record.get("outcome", ""), row_id))
            for run_id, raw in db.execute(
                "SELECT id,quality_json FROM test_runs WHERE quality_json IS NOT NULL"
            ).fetchall():
                report = json.loads(raw)
                if report.get("schema_version") != 3:
                    continue
                report = rescore_report(report)
                summary = report["summary"]
                db.execute("UPDATE test_runs SET quality_json=?,avg_score=?,weighted_score=? WHERE id=?",
                           (pack_text(json.dumps(report, separators=(",", ":"))),
                            summary["category_balanced"] or 0, summary["category_balanced"] or 0, run_id))
                rescored.append({"run_id": run_id, "achievement": summary["category_balanced"],
                                 "full_pass": summary["category_balanced_full_pass"],
                                 "scored": summary["scored"], "outcomes": summary["outcomes"]})
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table, columns in PAYLOADS.items():
                if table not in tables:
                    continue
                for column in columns:
                    for row_id, value in db.execute(
                        f"SELECT rowid,{column} FROM {table} WHERE {column} IS NOT NULL AND typeof({column})='text'"
                    ).fetchall():
                        packed = pack_text(value)
                        if unpack_text(packed) != value:
                            raise RuntimeError("Compression roundtrip verification failed")
                        if isinstance(packed, bytes):
                            db.execute(f"UPDATE {table} SET {column}=? WHERE rowid=?", (packed, row_id))
                            changed += 1
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite integrity check failed")
            db.commit()
        except BaseException:
            db.rollback()
            raise
        # VACUUM reclaims both pre-existing free pages and the compressed space.
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.execute("VACUUM")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite integrity check failed after compaction")
    return {"before": before, "after": inspect_database(path), "backup": str(backup),
            "backup_bytes": backup.stat().st_size, "compressed_payloads": changed,
            "rescored_runs": rescored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path(__file__).parent / "data" / "bench.db")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = maintain_database(args.database) if args.apply else inspect_database(args.database)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
