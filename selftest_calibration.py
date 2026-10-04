"""Calibration rejects incomparable, incomplete, and duplicated evidence."""

from copy import deepcopy
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.benchmarking.calibration import analyze_runs, markdown, read_runs
from calibrate_suite import main


def run(ident=1, model=1, passed=True, *, protocol=None, outcome=None, scored=True):
    fingerprint = hashlib.sha256(b"task").hexdigest()
    suite_hash = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    row = {"id": "q1", "fingerprint": fingerprint, "category": "Coding", "family": "family",
           "scope": "capability", "passed": passed, "scored": scored,
           "outcome": outcome or ("pass" if passed else "task_failure"), "evaluation": {"contract_score": 1}}
    base_protocol = {"revision": "v1", "temperature": 0, "evaluation_schema_version": 2,
                     "evaluator_versions": {"code_exec": "6"}, "model_seed": 0,
                     "max_output_tokens": 65536, "quality_workers": 4, "client": {"revision": "v4"}}
    report = {"schema_version": 3, "suite_hash": suite_hash, "model": f"endpoint-model-{model}",
              "protocol": {**base_protocol, **(protocol or {})}, "results": [row],
              "summary": {"category_balanced": int(passed)}}
    return {"id": ident, "model_id": model, "model_name": f"model-{model}", "status": "completed",
            "started_at": "start", "completed_at": "finish", "total_questions": 1,
            "test_suite_hash": suite_hash, "quality_json": json.dumps(report)}


class CalibrationTests(unittest.TestCase):
    def test_matched_panel_and_confounds(self):
        a, b = run(), run(2, 2, False, outcome="truncation")
        report = analyze_runs([a, b])
        cohort = report["cohorts"][0]
        self.assertEqual(cohort["admission_evidence"], "preliminary")
        self.assertEqual(cohort["questions"][0]["observed_pattern"], "mixed_observed")
        self.assertEqual(cohort["questions"][0]["completion_or_format_failures"], 1)
        self.assertEqual(cohort["paired_comparisons"][0]["left_only_pass"], 1)
        self.assertIn("model-2", markdown(report))

    def test_distinct_protocols_do_not_pool(self):
        report = analyze_runs([run(), run(2, 2, protocol={"revision": "v1", "temperature": 1})])
        self.assertEqual(len(report["cohorts"]), 2)
        self.assertTrue(all(not cohort["paired_comparisons"] for cohort in report["cohorts"]))

    def test_exclusions_and_no_automatic_pruning(self):
        good = run()
        running = {**run(2), "status": "running"}
        invalid = {**run(3), "quality_json": "{"}
        missing = {**run(4), "total_questions": 2}
        report = analyze_runs([good, running, invalid, missing])
        self.assertEqual(report["included_run_ids"], [1])
        self.assertEqual(len(report["excluded_runs"]), 3)
        self.assertFalse(report["admission_policy"]["automatic_pruning"])

    def test_changed_and_duplicated_identity_inventory(self):
        good = run()
        for mutation in ("duplicate", "hash", "verdict"):
            bad = deepcopy(run(2))
            payload = json.loads(bad["quality_json"])
            if mutation == "duplicate":
                payload["results"] *= 2
                bad["total_questions"] = 2
            elif mutation == "hash":
                payload["results"][0]["fingerprint"] = "f" * 64
            else:
                payload["results"][0]["scored"] = False
            bad["quality_json"] = json.dumps(payload)
            with self.subTest(mutation=mutation):
                self.assertEqual(analyze_runs([good, bad])["included_run_ids"], [1])

    def test_unscored_rows_are_not_failures(self):
        report = analyze_runs([run(passed=False, scored=False, outcome="endpoint_error")])
        question = report["cohorts"][0]["questions"][0]
        self.assertEqual(question["observed_pattern"], "unscored")
        self.assertIsNone(question["strict_success"])

    def test_repeat_policy_and_model_balance(self):
        runs = [run(i, 1, True) for i in range(1, 4)] + [run(4, 2, False)]
        cohort = analyze_runs(runs)["cohorts"][0]
        self.assertEqual(cohort["families"][0]["model_balanced_strict_success"], 0.5)
        self.assertEqual(cohort["admission_evidence"], "preliminary")
        runs.extend([run(5, 2, False), run(6, 2, False)])
        self.assertEqual(analyze_runs(runs)["cohorts"][0]["admission_evidence"], "repeat_panel_available")
        with self.assertRaises(ValueError):
            analyze_runs([run(), run()])
        reports = [run(i, model, False, scored=False, outcome="endpoint_error")
                   for i, model in enumerate([1, 1, 1, 2, 2, 2], 1)]
        self.assertEqual(analyze_runs(reports)["cohorts"][0]["admission_evidence"], "preliminary")

    def test_sqlite_snapshot_is_read_only_and_credentials_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.db"
            with closing(sqlite3.connect(path)) as db:
                db.executescript("CREATE TABLE models(id INTEGER,name TEXT,api_key TEXT); "
                                 "CREATE TABLE test_runs(id INTEGER,model_id INTEGER,status TEXT,"
                                 "started_at TEXT,completed_at TEXT,total_questions INTEGER,"
                                 "test_suite_hash TEXT,quality_json TEXT);")
                db.execute("INSERT INTO models VALUES(1,'model','secret')")
                fixture = run()
                db.execute("INSERT INTO test_runs VALUES(?,?,?,?,?,?,?,?)", tuple(fixture[key] for key in
                           ("id", "model_id", "status", "started_at", "completed_at", "total_questions",
                            "test_suite_hash", "quality_json")))
                db.commit()
            before = path.read_bytes()
            records = read_runs(path)
            self.assertEqual(records[0]["id"], 1)
            self.assertNotIn("secret", json.dumps(records))
            self.assertEqual(before, path.read_bytes())

    def test_duplicate_model_configuration_is_one_model(self):
        original, duplicate = run(), run(2, 2)
        report = json.loads(duplicate["quality_json"])
        report["model"] = "endpoint-model-1"
        duplicate["quality_json"] = json.dumps(report)
        cohort = analyze_runs([original, duplicate])["cohorts"][0]
        self.assertEqual(cohort["distinct_models"], 1)
        self.assertEqual(cohort["repeats"][0]["runs"], 2)

    def test_cli_preserves_source_and_requires_distinct_json_markdown_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "source.json"
            database.write_bytes(b"untouched database")
            for output in (database, Path(directory) / "report.md"):
                with self.subTest(output=output), patch("calibrate_suite.read_runs") as reader, \
                        patch("sys.argv", ["calibrate_suite.py", "--database", str(database), "--output", str(output)]), \
                        patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as error:
                        main()
                    self.assertEqual(error.exception.code, 2)
                    reader.assert_not_called()
                    self.assertEqual(database.read_bytes(), b"untouched database")
            output = Path(directory) / "report.json"
            with patch("calibrate_suite.read_runs", return_value=[run()]), \
                    patch("sys.argv", ["calibrate_suite.py", "--database", str(database), "--output", str(output)]), \
                    patch("builtins.print"):
                main()
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["included_run_ids"], [1])
            self.assertTrue(output.with_suffix(".md").read_text(encoding="utf-8").startswith("# Existing-task calibration"))
            self.assertEqual(database.read_bytes(), b"untouched database")


if __name__ == "__main__":
    unittest.main()
