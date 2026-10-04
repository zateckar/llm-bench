"""Lossless storage, maintenance, backups and fractional scoring regressions."""

import asyncio
from contextlib import closing
import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app import config
from app.database import fetch_all, fetch_one
from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.models import CriterionResult, EvaluationResult, Question, RequestMetrics, Result
from app.benchmarking.quality_report import make_report, paired_comparison, rescore_report
from app.services.benchmark_runner import _store_result
from app.storage import DETECT_TYPES, pack_text, unpack_text
from maintain_database import maintain_database

ROOT = Path(__file__).parent


def partial(ident="a", earned=1, possible=4):
    return Result(Question(ident, "Reasoning", "background " * 4000, "exact_match", ["yes"]),
                  "answer " * 4000, 0, evaluation=EvaluationResult(
                      "exact_match", criteria=[CriterionResult("content", status="fail", earned=earned,
                                                               possible=possible)]))


class ScoringTests(unittest.TestCase):
    def test_primary_score_and_paired_difference_use_achievement(self):
        left = make_report([partial()], ClientConfig("", "", "fixture"))
        right = make_report([partial(earned=3)], ClientConfig("", "", "fixture"))
        self.assertEqual(left["results"][0]["score"], 0.25)
        self.assertEqual(left["results"][0]["evaluator_score"], 0)
        self.assertFalse(left["results"][0]["passed"])
        self.assertEqual(left["summary"]["category_balanced"], 0.25)
        self.assertEqual(left["summary"]["category_balanced_full_pass"], 0)
        self.assertEqual(left["summary"]["categories"]["Reasoning"]["score"], 0.25)
        self.assertEqual(paired_comparison(left, right)["balanced_difference"], 0.5)
        self.assertEqual(rescore_report(rescore_report(left)), left)

    def test_format_failure_stays_in_denominator_and_errors_are_excluded(self):
        wrong = Result(Question("b", "Reasoning", "reply", "exact_match"), "bad", 0,
                       outcome="formatting", evaluation=EvaluationResult("exact_match", criteria=[
                           CriterionResult("format", status="fail", dimension="contract")]))
        error = Result(Question("c", "Reasoning", "reply", "exact_match"), "", 0,
                       outcome="endpoint_error", metrics=RequestMetrics(ok=False))
        report = make_report([partial(), wrong, error], ClientConfig("", "", "fixture"))
        self.assertEqual(report["summary"]["category_balanced"], 0.125)
        self.assertEqual(report["summary"]["scored"], 2)
        self.assertEqual(report["summary"]["capability_criterion_coverage"], 0.5)
        self.assertEqual(report["summary"]["missing_outcome_score_bounds"], [0.25 / 3, 1.25 / 3])


class StorageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "bench.db"
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript((ROOT / "app/schema.sql").read_text(encoding="utf-8"))
            db.execute("INSERT INTO models(id,name,base_url,api_key,model_id) VALUES(1,'fixture','','unused','fixture')")
            db.execute("INSERT INTO test_runs(id,model_id,status,started_at) VALUES(1,1,'pending','2026-10-04T11:00:00+00:00')")
        patcher = patch.object(config, "DATABASE_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_mixed_payloads_and_iso_timestamps_through_application_reads(self):
        for value in (None, "", "short", "LLMBZ1\x00literal", "Příliš žluťoučký 🐈\n" * 1000):
            self.assertEqual(unpack_text(pack_text(value)), value)
        result = partial()
        _store_result(1, 1, result)
        row = asyncio.run(fetch_one("SELECT * FROM test_results WHERE run_id=1"))
        self.assertEqual(row["prompt"], result.question.prompt)
        self.assertEqual(row["response"], result.response)
        self.assertEqual(row["score"], 0.25)
        self.assertEqual(json.loads(row["quality_metadata_json"])["evaluation"]["criterion_achievement"], 0.25)
        run = asyncio.run(fetch_one("SELECT started_at FROM test_runs WHERE id=1"))
        self.assertEqual(run["started_at"], "2026-10-04T11:00:00+00:00")
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT typeof(prompt) FROM test_results").fetchone()[0], "blob")

    def test_idle_maintenance_preserves_payloads_and_provides_restorable_backup(self):
        result = partial()
        report = make_report([result], ClientConfig("", "", "fixture"))
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE test_runs SET status='completed',quality_json=? WHERE id=1", (json.dumps(report),))
            db.execute("INSERT INTO test_results(run_id,test_id,category,prompt,response,score,quality_metadata_json) VALUES(1,'a','Reasoning',?,?,0,?)",
                       (result.question.prompt, result.response, json.dumps(report["results"][0])))
        before = asyncio.run(fetch_all("SELECT prompt,response,passed FROM test_results"))
        stats = maintain_database(self.path)
        self.assertLess(stats["after"]["bytes"], stats["before"]["bytes"])
        self.assertEqual(stats["before"]["results"], stats["after"]["results"])
        self.assertEqual(asyncio.run(fetch_all("SELECT prompt,response,passed FROM test_results")), before)
        self.assertEqual(asyncio.run(fetch_one("SELECT avg_score FROM test_runs"))["avg_score"], 0.25)
        restored = self.path.parent / "restored.db"
        restored.write_bytes(gzip.decompress(Path(stats["backup"]).read_bytes()))
        with closing(sqlite3.connect(restored, detect_types=DETECT_TYPES)) as db:
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(db.execute("SELECT response FROM test_results").fetchone()[0], result.response)
            self.assertEqual(db.execute("SELECT avg_score FROM test_runs").fetchone()[0], 0)
        maintain_database(self.path)
        self.assertEqual(asyncio.run(fetch_all("SELECT prompt,response,passed FROM test_results")), before)

    def test_active_queue_is_not_modified_or_backed_up(self):
        with self.assertRaisesRegex(RuntimeError, "idle run queue"):
            maintain_database(self.path)
        self.assertFalse((self.path.parent / "backups").exists())
        self.assertEqual(asyncio.run(fetch_one("SELECT status FROM test_runs"))["status"], "pending")


if __name__ == "__main__":
    unittest.main()
