"""Production startup storage migration, rollback, and restart regressions."""

import asyncio
from contextlib import closing
import gzip
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import zlib

from app import config
from app.services.database_maintenance import migrate_storage, maintenance_lock, STORAGE_REVISION
from app.storage import DETECT_TYPES, MAGIC, LZMA_MAGIC

ROOT = Path(__file__).parent


class MigrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "server.db"
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript((ROOT / "app/schema.sql").read_text(encoding="utf-8"))
            db.execute("INSERT INTO models(id,name,base_url,api_key,model_id) VALUES(1,'fixture','','unused','fixture')")
            db.execute("INSERT INTO test_runs(id,model_id,status,avg_score,weighted_score) VALUES(1,1,'completed',0.125,0.75)")
            fixture = "".join(f"{i}: repeated long fixture data;" for i in range(5000))
            self.text = (fixture + " different field " + fixture) * 3
            self.old = MAGIC + zlib.compress(self.text.encode(), 6)
            db.execute("INSERT INTO test_results(run_id,test_id,category,prompt,response,score,passed) VALUES(1,'a','Reasoning',?,?,0.125,1)",
                       (self.text, self.old))

    def read(self):
        with closing(sqlite3.connect(self.path, detect_types=DETECT_TYPES)) as db:
            return db.execute("SELECT prompt,response,score,passed FROM test_results").fetchall()

    def test_server_database_migration_preserves_scores_and_backup_and_skips_restart(self):
        before = self.read()
        stats = migrate_storage(self.path, startup=True)
        self.assertLess(stats["after"]["bytes"], stats["before"]["bytes"])
        self.assertEqual(stats["after"]["free_bytes"], 0)
        self.assertEqual(self.read(), before)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT avg_score,weighted_score FROM test_runs").fetchone(), (0.125, 0.75))
            self.assertEqual(db.execute("SELECT state FROM database_storage_migrations").fetchone(), ("complete",))
            self.assertTrue(db.execute("SELECT response FROM test_results").fetchone()[0].startswith(LZMA_MAGIC))
        restored = self.path.parent / "restored.db"
        restored.write_bytes(gzip.decompress(Path(stats["backup"]).read_bytes()))
        with closing(sqlite3.connect(restored)) as db:
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchall(), [("ok",)])
            self.assertEqual(db.execute("SELECT response FROM test_results").fetchone()[0], self.old)
        size = self.path.stat().st_size
        self.assertTrue(migrate_storage(self.path, startup=True)["skipped"])
        self.assertEqual(self.path.stat().st_size, size)
        self.assertEqual(len(list((self.path.parent / "backups").glob("*.db.gz"))), 1)

    def test_pending_runs_wait_for_startup_migration_but_offline_requires_idle(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE test_runs SET status='pending',run_options_json='{}'")
        with self.assertRaisesRegex(RuntimeError, "idle workers"):
            migrate_storage(self.path)
        self.assertFalse((self.path.parent / "backups").exists())
        migrate_storage(self.path, startup=True)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT status,run_options_json FROM test_runs").fetchone(), ("pending", "{}"))

    def test_running_workers_block_even_startup(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE test_runs SET status='running'")
        with self.assertRaisesRegex(RuntimeError, "idle workers"):
            migrate_storage(self.path, startup=True)
        self.assertEqual(self.read()[0][0], self.text)
        self.assertFalse((self.path.parent / "backups").exists())

    def test_failed_compression_rolls_back_and_preserves_restorable_backup(self):
        before = self.read()
        with patch("app.services.database_maintenance.compress_payloads", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                migrate_storage(self.path)
        self.assertEqual(self.read(), before)
        migrate_storage(self.path)
        self.assertEqual(self.read(), before)

    def test_failed_vacuum_is_resumed_without_another_backup(self):
        real_connect = sqlite3.connect
        class InterruptedVacuum(sqlite3.Connection):
            def execute(self, sql, *args):
                if sql == "VACUUM":
                    raise RuntimeError("interrupted vacuum")
                return super().execute(sql, *args)
        def connect(*args, **kwargs):
            kwargs["factory"] = InterruptedVacuum
            return real_connect(*args, **kwargs)
        with patch("app.services.database_maintenance.sqlite3.connect", side_effect=connect):
            with self.assertRaisesRegex(RuntimeError, "interrupted vacuum"):
                migrate_storage(self.path)
        with closing(real_connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT state FROM database_storage_migrations").fetchone(), ("compressed",))
        stats = migrate_storage(self.path, startup=True)
        self.assertEqual(stats["revision"], STORAGE_REVISION)
        self.assertEqual(len(list((self.path.parent / "backups").glob("*.db.gz"))), 1)

    def test_empty_and_literal_marker_text_remain_lossless(self):
        for value in ("", "LLMBZ1\0literal", "LLMBX1\0literal"):
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("UPDATE test_results SET response=?", (value,))
            migrate_storage(self.path, force=True)
            self.assertEqual(self.read()[0][1], value)

    def test_low_disk_fails_before_backup_or_payload_changes(self):
        before = self.read()
        with patch("app.services.database_maintenance.shutil.disk_usage") as usage:
            usage.return_value.free = 0
            with self.assertRaisesRegex(RuntimeError, "free bytes"):
                migrate_storage(self.path)
        self.assertEqual(self.read(), before)
        self.assertEqual(list((self.path.parent / "backups").glob("*")), [])

    def test_concurrent_maintenance_is_rejected_before_creating_backup(self):
        with maintenance_lock(self.path):
            with self.assertRaises(OSError):
                migrate_storage(self.path, startup=True)
        self.assertFalse((self.path.parent / "backups").exists())

    def test_running_judge_is_not_compacted(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO ab_studies(id,name,run_a,run_b,label_a,label_b,suite_name,scope) VALUES(1,'fixture',1,1,'a','b','fixture','all')")
            db.execute("INSERT INTO ab_judges(study_id,model_name,model_identifier,revision,status) VALUES(1,'fixture','fixture','fixture','running')")
        with self.assertRaisesRegex(RuntimeError, "idle judge"):
            migrate_storage(self.path, startup=True)
        self.assertFalse((self.path.parent / "backups").exists())

    def test_pinned_wal_reader_blocks_completion_and_retry_reuses_backup(self):
        real_connect = sqlite3.connect
        with closing(real_connect(self.path)) as reader:
            reader.execute("PRAGMA journal_mode=WAL")
            reader.execute("BEGIN")
            reader.execute("SELECT * FROM test_results").fetchall()
            def connect(*args, **kwargs):
                kwargs["timeout"] = 0
                return real_connect(*args, **kwargs)
            with patch("app.services.database_maintenance.sqlite3.connect", side_effect=connect):
                with self.assertRaisesRegex(RuntimeError, "another database connection"):
                    migrate_storage(self.path, startup=True)
            reader.rollback()
        with closing(real_connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT state FROM database_storage_migrations").fetchone(), ("compressed",))
        migrate_storage(self.path, startup=True)
        self.assertEqual(len(list((self.path.parent / "backups").glob("*.db.gz"))), 1)

    def test_lifespan_migrates_configured_database_before_resuming_queue(self):
        from app.main import lifespan, app
        events = []
        async def init():
            events.append("init")
        def migration(path, *, startup):
            self.assertEqual(path, self.path)
            self.assertTrue(startup)
            events.append("storage")
            return {}
        async def run():
            with patch.object(config, "DATABASE_PATH", self.path), \
                    patch("app.main.init_db", new=init), \
                    patch("app.services.database_maintenance.migrate_storage", side_effect=migration), \
                    patch("app.services.run_queue.resume", side_effect=lambda: events.append("resume")):
                async with lifespan(app):
                    events.append("serve")
        asyncio.run(run())
        self.assertEqual(events, ["init", "storage", "resume", "serve"])

    def test_failed_migration_never_resumes_queue(self):
        from app.main import lifespan, app
        async def run():
            with patch("app.main.init_db", new=AsyncMock()), \
                    patch("app.services.database_maintenance.migrate_storage", side_effect=RuntimeError("blocked")), \
                    patch("app.services.run_queue.resume") as resume:
                with self.assertRaisesRegex(RuntimeError, "blocked"):
                    async with lifespan(app):
                        self.fail("must not serve")
                resume.assert_not_called()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
