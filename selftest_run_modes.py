#!/usr/bin/env python3
"""Offline tests for runs that measure quality, performance or both (docs/design-run-modes.md)."""

import asyncio
from contextlib import ExitStack, closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import config as app_config
from app.benchmarking import load_test, load_workload
from app.benchmarking.models import Question
from app.benchmarking.quality_suite import provenance
from app.services import benchmark_runner as runner
from app.services import run_modes, run_submission, scorecard
from app.storage import DETECT_TYPES
from selftest_benchmark_runner import FakeClient
from selftest_load_test import FakeServer, workload

ROOT = Path(__file__).parent
MODEL = {"id": 1, "name": "fake", "base_url": "http://fake.invalid", "api_key": "unused", "model_id": "fake"}
SWEEP = {"schema_version": 4, "kind": "context_sweep", "finished": True, "efforts": [], "notes": []}


class ModeTests(unittest.TestCase):
    def test_normalise(self):
        cases = {
            ("quality", None): (True, None),
            ("performance", None): (False, "fixed"),
            ("both", None): (True, "fixed"),
            ("both", "sweep"): (True, "sweep"),
            ("both", "load"): (True, "load"),
            ("performance", "load"): (False, "load"),
            ("sweep", None): (False, "sweep"),
            ("load", "load"): (False, "load"),
        }
        for (mode, performance), expected in cases.items():
            self.assertEqual(run_modes.normalise(mode, performance), expected, (mode, performance))
        for mode, performance in (("quality", "sweep"), ("quality", "fixed"), ("sweep", "load"),
                                  ("both", "bogus"), ("bogus", None), (None, None), ("both", 3)):
            with self.assertRaises(ValueError, msg=(mode, performance)):
                run_modes.normalise(mode, performance)
        self.assertEqual(run_modes.parts({}), (True, "fixed"))
        self.assertEqual(run_modes.parts({"mode": "nonsense"}), (True, "fixed"))
        self.assertEqual(run_modes.label({"mode": "both", "performance": "load"}, "Rigorous"),
                         "Quality · Rigorous + Open-loop load")
        self.assertEqual(run_modes.label({"mode": "sweep"}), "Context & reasoning sweep")
        self.assertEqual(run_modes.label({"mode": "both", "performance": "standard"}, "Standard quality suite"),
                         "Quality · Standard quality suite + Performance · latency, context, capacity")
        self.assertEqual(run_modes.normalise("performance", "standard"), (False, "standard"))
        # Runs of retired suites keep their labels.
        self.assertEqual(run_modes.describe('{"mode": "quality", "suite": "safety-language"}'),
                         "Quality · Safety & language adherence")
        self.assertEqual(run_modes.describe('{"mode": "quality"}'), "Quality · Rigorous capability suite")
        self.assertIsNone(run_modes.describe(None))
        self.assertIsNone(run_modes.describe("not json"))

    def test_options(self):
        make = run_submission.make_run_options
        default_load = load_workload.parse_settings(load_workload.default_settings()).model_dump()
        # New runs store the suite and the standard performance test with its stage settings.
        self.assertEqual(make(), {"mode": "both", "max_concurrency": 8, "suite": "standard",
                                  "performance": "standard", "in_flight_cap": 256, "load": default_load})
        self.assertEqual(make(mode="quality", max_concurrency=3), {"mode": "quality", "max_concurrency": 3,
                                                                   "suite": "standard"})
        self.assertEqual(make(mode="performance", context_max=65536, in_flight_cap=64),
                         {"mode": "performance", "max_concurrency": 8, "performance": "standard",
                          "in_flight_cap": 64, "context_max": 65536, "load": default_load})
        # Former tests map to the standard test, keeping compatible settings.
        self.assertEqual(make(mode="both", performance="fixed"), make())
        self.assertEqual(make(mode="sweep", max_concurrency=256), make(mode="performance"))
        sweep = make(mode="both", performance="sweep", max_concurrency=16, context_max=65536)
        self.assertEqual((sweep["performance"], sweep["max_concurrency"], sweep["context_max"]), ("standard", 16, 65536))
        load = make(mode="load", max_concurrency=300, load={"preset": "chat", "rates": [1, 2]})
        self.assertEqual((load["mode"], load["performance"], load["in_flight_cap"], load["max_concurrency"],
                          load["load"]["preset"], load["load"]["rates"]),
                         ("performance", "standard", 300, 8, "chat", [1.0, 2.0]))
        for kwargs in ({"mode": "quality", "performance": "sweep"}, {"mode": "sweep", "performance": "load"},
                       {"mode": "both", "performance": "bogus"},
                       {"mode": "performance", "suite": "usecase:hr@1"},
                       {"mode": "quality", "suite": "tool-conformance"},
                       {"mode": "both", "max_concurrency": 33},
                       {"mode": "both", "in_flight_cap": 2000},
                       {"mode": "both", "context_max": 100},
                       {"mode": "both", "performance": "load", "max_concurrency": 2000}):
            with self.subTest(kwargs), self.assertRaises(HTTPException):
                make(**kwargs)

    def test_rerun_specs_are_canonical(self):
        legacy_load = load_workload.parse_settings({"preset": "chat", "rates": [1, 2]}).model_dump()
        legacy = {"mode": "load", "max_concurrency": 128, "load": legacy_load}
        spec = run_submission.spec_from_run({"model_id": 1, "run_options_json": json.dumps(legacy)})
        self.assertEqual((spec["mode"], spec["performance"], spec["in_flight_cap"], spec["load"]),
                         ("performance", "standard", 128, legacy_load))
        # A run of a retired suite reruns the standard suite.
        old = {"mode": "both", "performance": "sweep", "max_concurrency": 32, "context_max": 65536,
               "sweep_rounds": 1, "sweep_output_tokens": 8192, "suite": "tool-conformance"}
        spec = run_submission.spec_from_run({"model_id": 1, "run_options_json": json.dumps(old)})
        self.assertEqual((spec["suite"], spec["performance"], spec["context_max"], spec["max_concurrency"]),
                         ("standard", "standard", 65536, 32))
        combined = run_submission.make_run_options(mode="both", context_max=32768)
        self.assertEqual(run_submission.spec_from_run({"model_id": 1, "run_options_json": json.dumps(combined)}),
                         {"model_id": 1, **combined})


class DatabaseCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        self.path = Path(directory) / "bench.db"
        self.questions = [Question(str(i), "Reasoning", "good", "json_match", {"value": {"x": 1}}) for i in range(3)]
        for patcher in (
            patch.object(app_config, "DATABASE_PATH", self.path),
            patch("app.services.url_guard.validate_endpoint"),
            patch("app.services.monitoring.capture_for_run"),
            patch.object(runner, "load_questions", return_value=self.questions),
            patch("app.benchmarking.llm_client.ChatClient", FakeClient),
            patch.object(load_test, "ChatClient", FakeServer(slots=2, service=0.05)),
            patch.object(load_workload, "MIN_STEP_SECONDS", 1),
        ):
            self.stack.enter_context(patcher)
        scorecard.clear_caches()
        self.addCleanup(scorecard.clear_caches)
        with closing(sqlite3.connect(self.path)) as db:
            db.executescript((ROOT / "app/schema.sql").read_text(encoding="utf-8"))
            db.execute("INSERT INTO users (id, username, role) VALUES (1, 'admin', 'admin')")
            db.execute("INSERT INTO models (id, name, base_url, api_key, model_id) "
                       "VALUES (1, 'fake', 'http://fake.invalid', 'unused', 'fake')")
            db.commit()

    def sql(self, query, params=()):
        with closing(sqlite3.connect(self.path, detect_types=DETECT_TYPES)) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute(query, params).fetchall()]
            db.commit()
            return rows

    def start(self, options, run_id=1):
        self.sql("INSERT INTO test_runs (id, model_id, status, quality_config_json, run_options_json) "
                 "VALUES (?, 1, 'pending', ?, ?)", (run_id, json.dumps(provenance()), json.dumps(options)))
        finished = Mock()
        runner._run_benchmark(run_id, MODEL, on_finish=finished, **options)
        finished.assert_called_once_with(run_id)
        return self.sql("SELECT * FROM test_runs WHERE id=?", (run_id,))[0]


class SubmissionTests(DatabaseCase):
    def validate(self, specs):
        return asyncio.run(run_submission.validated_specs(json.dumps(specs)))

    def test_combined_specs(self):
        (_, config, options, repeats), = self.validate(
            [{"model_id": 1, "mode": "both", "max_concurrency": 32, "context_max": 65536, "in_flight_cap": 64,
              "load": {"preset": "chat", "rates": [1, 2]}, "repeats": 2}])
        self.assertEqual((options["performance"], options["context_max"], options["in_flight_cap"], options["suite"],
                          options["load"]["rates"], repeats), ("standard", 65536, 64, "standard", [1.0, 2.0], 2))
        self.assertEqual((config["name"], config["revision"]), ("standard", "standard-v1"))
        # Saved plans and API callers may still name a former performance test.
        (_, _, options, _), = self.validate([{"model_id": 1, "mode": "both", "performance": "load",
                                              "max_concurrency": 64, "load": {"preset": "chat", "rates": [1, 2]}}])
        self.assertEqual((options["mode"], options["performance"], options["in_flight_cap"]), ("both", "standard", 64))
        for bad in ({"mode": "quality", "context_max": 65536},
                    {"mode": "quality", "load": {"preset": "chat", "rates": [1]}},
                    {"mode": "quality", "in_flight_cap": 8},
                    {"mode": "quality", "performance": "sweep"},
                    {"mode": "quality", "suite": "safety-language"},
                    {"mode": "performance", "suite": "standard", "unknown": 1},
                    {"mode": "both", "performance": ["sweep"]}):
            with self.subTest(bad), self.assertRaises(HTTPException):
                self.validate([{"model_id": 1, **bad}])


class RunnerTests(DatabaseCase):
    """Runs queued before the consolidation still execute their former test (decision 14).

    Their stored options are written out here as they were stored then; the
    staged standard test is covered in selftest_consolidation.py."""

    def test_quality_then_sweep(self):
        seen = []

        def measure(run_id, config, sweep_config, scope=None):
            # Quality has finished and its answers are stored before the sweep starts.
            seen.append((len(self.sql("SELECT * FROM test_results")), sweep_config.context_max))
            runner._store_sweep_checkpoint(run_id, {**SWEEP, "finished": False})
            return json.dumps(SWEEP), "", 1234.0

        options = {"mode": "both", "performance": "sweep", "max_concurrency": 16, "context_max": 65536,
                   "sweep_rounds": 1, "sweep_output_tokens": 8192}
        with patch.object(runner, "_measure_context_sweep", side_effect=measure):
            run = self.start(options)
        self.assertEqual(seen, [(3, 65536)])
        self.assertEqual((run["status"], run["total_questions"], run["scored_questions"], run["passed_questions"]),
                         ("completed", 3, 3, 3), run["error_message"])
        self.assertEqual(json.loads(run["quality_json"])["summary"]["scored"], 3)
        self.assertEqual(json.loads(run["perf_json"])["kind"], "context_sweep")
        self.assertEqual(run["workers"], 4)

    def test_quality_then_open_loop(self):
        load = {"preset": "custom", "workload": workload(ttft=400), "rates": [2], "step_seconds": 1}
        options = {"mode": "both", "performance": "load", "max_concurrency": 32,
                   "load": load_workload.parse_settings(load).model_dump()}
        run = self.start(options)
        self.assertEqual((run["status"], run["scored_questions"]), ("completed", 3), run["error_message"])
        perf = json.loads(run["perf_json"])
        self.assertEqual((perf["schema_version"], perf["kind"]), (5, "open_loop"))
        self.assertGreater(perf["steps"][0]["completed"], 0)
        self.assertEqual(json.loads(run["quality_json"])["summary"]["scored"], 3)
        # The run counts as capacity evidence; its retired rigorous suite is no scorecard column.
        evidence = scorecard.model_evidence(scorecard.collect(), 1)
        self.assertEqual(evidence["quality"], {})
        self.assertTrue(evidence["capacity"])

    def test_performance_is_skipped_when_quality_fails(self):
        for q in self.questions:
            q.prompt = "error"
        with patch.object(runner, "_measure_open_loop") as measure:
            run = self.start({"mode": "both", "performance": "load", "max_concurrency": 8,
                              "load": load_workload.default_settings()})
        measure.assert_not_called()
        self.assertEqual(run["status"], "failed")
        self.assertIn("No questions could be scored", run["error_message"])
        self.assertIsNone(run["perf_json"])

    def test_performance_failure_keeps_quality_and_saved_steps(self):
        def measure(run_id, *args):
            runner._store_sweep_checkpoint(run_id, {**SWEEP, "finished": False})
            raise RuntimeError("endpoint went away")

        options = {"mode": "both", "performance": "sweep", "max_concurrency": 16, "context_max": 1048576,
                   "sweep_rounds": 1, "sweep_output_tokens": 8192}
        with patch.object(runner, "_measure_context_sweep", side_effect=measure), \
                self.assertLogs(runner.logger, "ERROR"):
            run = self.start(options)
        self.assertEqual(run["status"], "failed")
        self.assertIn("Performance measurement failed: endpoint went away", run["error_message"])
        self.assertEqual(run["scored_questions"], 3)
        self.assertIsNotNone(run["quality_json"])
        self.assertFalse(json.loads(run["perf_json"])["finished"])

    def test_performance_only_and_legacy_modes_dispatch_as_before(self):
        for options, target in (({"mode": "sweep", "max_concurrency": 16}, "_run_context_sweep"),
                                ({"mode": "performance", "performance": "sweep", "max_concurrency": 16,
                                  "context_max": 65536}, "_run_context_sweep"),
                                ({"mode": "performance", "performance": "load", "max_concurrency": 16,
                                  "load": load_workload.default_settings()}, "_run_open_loop")):
            with self.subTest(options), patch.object(runner, target) as dispatched, \
                    patch.object(runner, "run_quality") as quality:
                self.sql("DELETE FROM test_runs")
                self.start(options)
                dispatched.assert_called_once()
                quality.assert_not_called()


class PageTests(DatabaseCase):
    def test_run_list_detail_and_form(self):
        from app.routes import admin, runs

        rows = [(1, {"mode": "both", "performance": "load"}, 3, 1), (2, {"mode": "sweep"}, 0, 1),
                (3, {"mode": "quality", "suite": "safety-language"}, 46, 0), (4, None, 5, 1)]
        for run_id, options, questions, perf in rows:
            self.sql("""INSERT INTO test_runs (id, model_id, status, total_questions, perf_json, run_options_json)
                        VALUES (?, 1, 'completed', ?, ?, ?)""",
                     (run_id, questions, '{"schema_version": 3}' if perf else None,
                      json.dumps(options) if options else None))
        app = FastAPI()

        @app.middleware("http")
        async def user(request, call_next):
            request.state.user = {"id": 1, "role": "admin", "username": "admin"}
            return await call_next(request)

        for router in (admin.router, runs.router):
            app.include_router(router)
        with patch.object(runs, "get_current_user", AsyncMock(return_value={"id": 1, "role": "admin"})), \
                TestClient(app) as client:
            listing = client.get("/runs").text
            self.assertIn("Measures", listing)
            for text in ("Open-loop load", "Context &amp; reasoning sweep", "Quality · safety-language",
                         "Fixed workload"):
                self.assertIn(text, listing)
            detail = client.get("/runs/1").text
            self.assertIn("Quality · Rigorous capability suite + Open-loop load", detail)
            self.assertIn("Measures:", client.get("/runs/3").text)
            self.assertNotIn("Measures:", client.get("/runs/4").text)
            self.sql("UPDATE test_runs SET status='running' WHERE id=2")
            self.assertIn("Context &amp; reasoning sweep", client.get("/admin/run/2/progress").text)
            # Former sweep links open a performance-only run of the standard test.
            form = client.get("/admin/run?mode=sweep").text.replace("&#34;", '"')
            self.assertIn('"mode": "performance"', form)
            self.assertIn('"performance": "standard"', form)
            for text in ("> Quality</label>", "> Performance</label>", "Advanced settings", "Quality runs first",
                         "In-flight cap · capacity stage", "Override the context limit"):
                self.assertIn(text, form)
            for text in ('x-model="run.performance"', '<option value="sweep">', '<option value="load">'):
                self.assertNotIn(text, form)


if __name__ == "__main__":
    unittest.main()
