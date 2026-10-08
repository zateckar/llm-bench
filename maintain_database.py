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
from app.services.quality_views import refresh_projections
from app.database import READ_INDEXES
from app.services.database_maintenance import migrate_storage, inspect_database

PAYLOADS = {
    "test_results": ("prompt", "response", "quality_metadata_json"),
    "test_runs": ("quality_json", "perf_json", "quality_summary_json"),
    "performance_cells": ("result_json",),
}


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
            if "prompt_preview" not in columns:
                db.execute("ALTER TABLE test_results ADD COLUMN prompt_preview TEXT")
            run_columns = {row[1] for row in db.execute("PRAGMA table_info(test_runs)")}
            if "quality_summary_json" not in run_columns:
                db.execute("ALTER TABLE test_runs ADD COLUMN quality_summary_json TEXT")
            # execute statements individually: executescript would commit and
            # release the writer reservation before the idle-only work finishes.
            for statement in READ_INDEXES.split(";"):
                if statement.strip():
                    db.execute(statement)
            db.execute("CREATE INDEX IF NOT EXISTS idx_test_results_order ON test_results(run_id,category,question_index,id)")
            db.execute("UPDATE test_results SET quality_outcome='' WHERE quality_metadata_json IS NULL AND quality_outcome IS NULL")
            for row_id, raw in db.execute(
                "SELECT id,quality_metadata_json FROM test_results WHERE quality_metadata_json IS NOT NULL"
            ):
                record = json.loads(raw)
                record.setdefault("evaluator_score", record.get("score", 0))
                record["score"] = achievement_score(record)
                payload = pack_text(json.dumps(record, separators=(",", ":")))
                db.execute("UPDATE test_results SET score=?,quality_metadata_json=?,quality_outcome=? WHERE id=?",
                           (record["score"], payload, record.get("outcome", ""), row_id))
            for run_id, raw in db.execute(
                "SELECT id,quality_json FROM test_runs WHERE quality_json IS NOT NULL"
            ):
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
                # A rescore must also replace the matching read projection.
                db.execute("UPDATE test_runs SET quality_summary_json=NULL WHERE id=?", (run_id,))
            refresh_projections(db)
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table, columns in PAYLOADS.items():
                if table not in tables:
                    continue
                for column in columns:
                    for row_id, value, stored_bytes in db.execute(
                        f"SELECT rowid,{column},length(CAST({column} AS BLOB)) FROM {table} WHERE {column} IS NOT NULL"
                    ):
                        packed = pack_text(value)
                        if unpack_text(packed) != value:
                            raise RuntimeError("Compression roundtrip verification failed")
                        if isinstance(packed, bytes) and len(packed) < stored_bytes:
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
    parser.add_argument("--storage-only", action="store_true",
                        help="Lossless compression/VACUUM without changing scores; stop the app first")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.apply and args.storage_only:
        result = migrate_storage(args.database, force=True)
    else:
        result = maintain_database(args.database) if args.apply else inspect_database(args.database)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
