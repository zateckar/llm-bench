#!/usr/bin/env python3
"""Offline tests for one quality suite and one staged performance test (docs/design-consolidation.md)."""

import asyncio
from contextlib import ExitStack, closing
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import aiosqlite
from fastapi import FastAPI

from app import config as app_config, database
from app.benchmarking import (load_test, load_workload, open_suite, perf_sweep, quality_suite, safety_suite,
                              session_load, session_workload, staged_performance as staged, standard_suite, tool_suite)
from app.benchmarking.models import Question
from app.benchmarking.quality_report import area_summary
from app.benchmarking.suites import DEFAULT_SUITE, RETIRED_SUITES, SUITE_NAMES, get_suite
from app.services import benchmark_runner as runner, html_reports, run_submission, scorecard
from app.services.capacity import estimate_capacity, evidence
from app.services.sweep_reports import hydrate_sweep
from app.storage import DETECT_TYPES
from app.templates_config import templates
from selftest_benchmark_runner import FakeClient as QualityClient
from selftest_capacity import run as latency_run
from selftest_load_test import CONFIG as LOAD_CONFIG, FakeServer, workload
from selftest_sessions import TEST_MODEL, FakeServer as SessionServer
from selftest_sweep import FakeClient as SweepClient

ROOT = Path(__file__).parent
MODEL = {"id": 1, "name": "fake", "base_url": "http://fake.invalid", "api_key": "unused", "model_id": "fake"}


class StandardSuiteTests(unittest.TestCase):
    def test_union_areas_and_unchanged_fingerprints(self):
        questions = standard_suite.load_questions()
        self.assertEqual(len(questions), 495)
        counts = {key: sum(standard_suite.area_of(q.metadata) == key for q in questions)
                  for key in standard_suite.AREA_LABELS}
        self.assertEqual(counts, {"reasoning": 323, "tools": 96, "safety": 46, "open": 30})
        self.assertEqual(standard_suite.validate_suite(questions), [])
        # Every question keeps its metadata, so the rigorous subset hashes as before.
        reasoning = [q for q in questions if standard_suite.area_of(q.metadata) == "reasoning"]
        self.assertEqual(quality_suite.suite_hash(reasoning), "7909c6325d1abd7b")
        self.assertEqual(quality_suite.suite_hash(reasoning), quality_suite.suite_hash(quality_suite.load_questions()))
        provenance = standard_suite.provenance()
        self.assertEqual((provenance["revision"], provenance["questions"]), ("standard-v1", 495))
        self.assertEqual([(a["area"], a["revision"], a["scored"]) for a in provenance["areas"]],
                         [("reasoning", quality_suite.REVISION, True), ("tools", tool_suite.REVISION, True),
                          ("safety", safety_suite.REVISION, True), ("open", open_suite.REVISION, False)])
        provenance["areas"].clear()  # a copy: the cached inventory is unchanged
        self.assertEqual(len(standard_suite.provenance()["areas"]), 4)

    def test_registry(self):
        self.assertEqual((DEFAULT_SUITE, SUITE_NAMES), ("standard", ("standard",)))
        suite = get_suite("standard")
        self.assertEqual(suite.label, "Standard quality suite")
        execution = suite.execution(4096)
        self.assertEqual(execution["native"]["suite_revision"], "standard-v1")
        self.assertIn("native_tools", suite.client_extra())
        for name in RETIRED_SUITES:
            self.assertTrue(get_suite(name).load(), name)  # still loadable for history
        self.assertEqual(get_suite(None).name, "rigorous")

    def test_area_summary(self):
        def row(cohort, category, passed, scope="capability", outcome=None):
            return {"metadata": {"cohort": cohort}, "category": category, "family": category + str(passed),
                    "scope": scope, "scored": scope != "open_ended", "passed": passed, "score": float(passed),
                    "outcome": outcome or ("pass" if passed else "task_failure")}

        rows = [row(quality_suite.REVISION, "Math", True), row(quality_suite.REVISION, "Code", False),
                row(safety_suite.REVISION, "Confidentiality", True),
                row(open_suite.REVISION, "Writing", False, "open_ended", "recorded")]
        areas = area_summary(rows)
        self.assertEqual(list(areas), ["reasoning", "safety", "open"])
        self.assertEqual((areas["reasoning"]["score"], areas["reasoning"]["categories"]), (0.5, 2))
        self.assertEqual(areas["safety"]["full_pass_rate"], 1.0)
        self.assertEqual((areas["open"]["graded"], areas["open"]["score"], areas["open"]["recorded"]),
                         (False, None, 1))
        # One area (a retired suite or a use-case suite) has no breakdown.
        self.assertEqual(area_summary(rows[:2]), {})
        self.assertEqual(area_summary([{**r, "metadata": {}} for r in rows]), {})


class StageTests(unittest.TestCase):
    def test_context_limits_and_ladder(self):
        self.assertEqual(staged.context_limit_for(32768), (28160, "declared"))
        self.assertEqual(staged.context_limit_for(None), (131072, "default"))
        self.assertEqual(staged.context_limit_for(1024), (131072, "default"))  # too small to leave headroom
        self.assertEqual(staged.context_limit_for(2_000_000), (1_048_576, "declared"))
        self.assertEqual(staged.context_limit_for(32768, 65536), (65536, "override"))
        self.assertEqual(staged.contexts_for(28160), (256, 8192, 28160))
        self.assertEqual(staged.contexts_for(131072), (256, 8192, 32768, 65536, 131072))
        for bad in (100, 2_000_000, True, "8192"):
            with self.assertRaises(ValueError):
                staged.context_limit(bad)
        self.assertEqual(staged.context_effort("high"), "high")
        self.assertEqual(staged.context_effort(None), "default")
        self.assertEqual(staged.context_effort("turbo"), "default")

    def test_sweep_config_explicit_contexts_and_efforts(self):
        config = perf_sweep.SweepConfig(context_max=28160, max_concurrency=1, sweep_output_tokens=4096,
                                        context_lengths=(256, 8192, 28160), effort_list=("high",))
        self.assertEqual((config.contexts, config.concurrencies, config.efforts), ((256, 8192, 28160), (1,), ("high",)))
        self.assertEqual(config.requests_per_effort, 9)
        self.assertEqual(perf_sweep.SweepConfig().efforts, perf_sweep.EFFORTS)
        for kwargs in ({"context_lengths": (100,)}, {"effort_list": ("turbo",)}):
            with self.assertRaises(ValueError):
                perf_sweep.SweepConfig(**kwargs)

    def test_stages_of_staged_and_historical_reports(self):
        fixed = latency_run()["perf"]
        sweep = {"schema_version": 4, "kind": "context_sweep"}
        load = {"schema_version": 5, "kind": "open_loop"}
        self.assertEqual(staged.stages_of(fixed), {"latency": fixed})
        self.assertEqual(staged.stages_of(sweep), {"context": sweep})
        self.assertEqual(staged.stages_of(load), {"capacity": load})
        self.assertEqual(staged.stages_of({"schema_version": 2}), {})
        report = staged.new_report("fake", {})
        report["stages"] = {"latency": fixed, "context": dict(sweep, finished=True), "capacity": "broken"}
        self.assertEqual(list(staged.stages_of(report)), ["latency", "context"])
        report.update(cancelled=True, running_stage="context")
        self.assertEqual(staged.stages_of(report)["context"]["cancelled"], True)
        self.assertFalse(staged.stages_of(report)["context"]["finished"])
        self.assertTrue(staged.is_performance_report(report))


class DatabaseCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        self.path = Path(directory) / "bench.db"
        self.stack.enter_context(patch.object(app_config, "DATABASE_PATH", self.path))
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


def latency_result(ok=True):
    perf = latency_run()["perf"]
    return SimpleNamespace(concurrency=[SimpleNamespace(requests=10, errors=0 if ok else 10)], notes=["n"],
                           telemetry=None, to_dict=lambda: perf)


def load_test_report():
    """An open-loop report, as the capacity stage of standard runs before v4 stored it."""
    with patch.object(load_workload, "MIN_STEP_SECONDS", 1):
        settings = load_workload.parse_settings(
            {"preset": "custom", "workload": workload(ttft=400), "rates": [2], "step_seconds": 1})
        return load_test.run_load_test(LOAD_CONFIG, settings, 4,
                                       client_factory=FakeServer(slots=2, service=0.05)).to_dict()


class StagedRunnerTests(DatabaseCase):
    def setUp(self):
        super().setUp()
        SweepClient.instances = []
        SweepClient.handler = None
        SweepClient.active = SweepClient.peak = 0
        for patcher in (
            patch("app.services.url_guard.validate_endpoint"),
            patch("app.services.monitoring.capture_for_run", side_effect=self.snapshot),
            patch.object(perf_sweep, "SweepClient", SweepClient),
            patch("app.benchmarking.llm_client.ChatClient", QualityClient),
            patch.object(session_load, "ChatClient", SessionServer),
            patch.object(session_workload, "MIN_WARMUP_SECONDS", 1),
            patch.object(session_workload, "MIN_MEASURE_SECONDS", 1),
        ):
            self.stack.enter_context(patcher)
        SessionServer.reset()
        # The capacity stage is no longer run.
        self.open_loop = self.stack.enter_context(patch.object(load_test, "run_load_test"))
        self.addCleanup(lambda: self.open_loop.assert_not_called())
        # One short level of two users at a 2k cap keeps the users stage quick.
        self.users = {"preset": "custom", "model": TEST_MODEL, "context_caps": [2048], "start_users": 2,
                      "max_users": 2, "warmup_seconds": 1, "measure_seconds": 1, "saturation": False}
        self.declared = 32768

    def snapshot(self, run_id, model):
        self.sql("""INSERT INTO deployment_checks (model_id, run_id, revision, created_at, ok, fingerprint, snapshot_json)
                    VALUES (1, ?, 'deployment-probe-v1', '2026-10-05T00:00:00+00:00', 1, 'f1', ?)""",
                 (run_id, json.dumps({"hard": {"served": {"max_model_len": self.declared}}})))

    def start(self, options, run_id=1):
        self.sql("INSERT INTO test_runs (id, model_id, status, run_options_json) VALUES (?, 1, 'pending', ?)",
                 (run_id, json.dumps(options)))
        finished = Mock()
        runner._run_benchmark(run_id, MODEL, on_finish=finished, **options)
        finished.assert_called_once_with(run_id)
        return self.sql("SELECT * FROM test_runs WHERE id=?", (run_id,))[0]

    def options(self, **kwargs):
        return run_submission.make_run_options(**{"mode": "performance", "users": self.users, **kwargs})

    def test_three_stages_into_one_report(self):
        with patch.object(runner, "run_perf_suite", return_value=latency_result()) as latency:
            run = self.start(self.options(max_concurrency=4, context_concurrency=4, target_ttft_ms=1500))
        self.assertEqual(latency.call_args.args[1].max_concurrency, 4)
        self.assertEqual(run["status"], "completed", run["error_message"])
        perf = json.loads(run["perf_json"])
        self.assertEqual((perf["schema_version"], perf["kind"], perf["revision"]), (6, "staged", staged.REVISION))
        self.assertEqual(list(perf["stages"]), ["latency", "context", "users"])
        self.assertNotIn("capacity", perf["protocol"])
        self.assertTrue(perf["finished"])
        users = perf["stages"]["users"]
        self.assertEqual((users["schema_version"], users["kind"], users["protocol"]["context_caps"]),
                         (7, "sessions", [2048]))
        self.assertEqual(users["summary"]["headline"]["users"], 2)
        self.assertEqual(perf["protocol"]["users"]["context_caps"], [2048])
        self.assertIsNone(perf["running_stage"])
        # The context ladder ends at the declared limit minus output and framing headroom.
        context = perf["protocol"]["context"]
        self.assertEqual((context["limit"], context["limit_source"], context["contexts"], context["effort"]),
                         (28160, "declared", [256, 8192, 28160], "default"))
        self.assertEqual((context["concurrencies"], context["targets"]["ttft_p95_ms"]), ([1, 2, 4], 1500))
        sweep = perf["stages"]["context"]
        self.assertEqual((sweep["protocol"]["contexts"], sweep["protocol"]["candidate_efforts"],
                          sweep["protocol"]["revision"]), ([256, 8192, 28160], ["default"], "context-limits-v1"))
        cells = self.sql("SELECT context_tokens, concurrency FROM performance_cells ORDER BY context_tokens, concurrency")
        self.assertEqual([(c["context_tokens"], c["concurrency"]) for c in cells],
                         [(n, c) for n in (256, 8192, 28160) for c in (1, 2, 4)])
        # A context-limit override wins over the declared limit. A queued run from before v4 still
        # carries capacity settings; they are ignored.
        self.sql("DELETE FROM performance_cells")
        with patch.object(runner, "run_perf_suite", return_value=latency_result()):
            run = self.start({**self.options(context_max=8192, context_concurrency=2),
                              "load": load_workload.default_settings(), "in_flight_cap": 64}, run_id=2)
        self.assertEqual(json.loads(run["perf_json"])["protocol"]["context"]["contexts"], [256, 8192])
        self.assertEqual(list(json.loads(run["perf_json"])["stages"]), ["latency", "context", "users"])

        # One staged run is latency, context and users evidence on the scorecard.
        evidence_ = scorecard.model_evidence(scorecard.collect(), 1)
        self.assertEqual((evidence_["latency"]["run_id"], evidence_["context"]["run_id"]), (2, 2))
        self.assertEqual(evidence_["context"]["tokens"], 8192)
        self.assertEqual(evidence_["capacity"], {})
        users_key = "custom:" + users["protocol"]["user_model_hash"]
        self.assertEqual(list(evidence_["users"]), [users_key])
        self.assertEqual(evidence_["users"][users_key]["results"][0]["context_cap"], 2048)
        self.assertEqual(evidence_["users"][users_key]["run_id"], 2)

    def test_failing_stage_does_not_stop_the_next(self):
        with patch.object(runner, "run_perf_suite", return_value=latency_result()), \
                patch.object(perf_sweep, "run_sweep", side_effect=RuntimeError("sweep broke")), \
                self.assertLogs(runner.logger, "ERROR"):
            run = self.start(self.options())
        perf = json.loads(run["perf_json"])
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["error_message"], "Context stage: sweep broke")
        self.assertEqual(list(perf["stages"]), ["latency", "users"])
        self.assertEqual(perf["stage_errors"], {"context": "Context stage: sweep broke"})
        self.assertFalse(perf["finished"])

    def test_endpoint_down_stops_later_stages(self):
        with patch.object(runner, "run_perf_suite", return_value=latency_result(ok=False)), \
                patch.object(perf_sweep, "run_sweep") as sweep:
            run = self.start(self.options())
        sweep.assert_not_called()
        perf = json.loads(run["perf_json"])
        self.assertEqual((run["status"], list(perf["stages"])), ("failed", ["latency"]))
        self.assertIn("No latency-stage request completed successfully", run["error_message"])
        self.assertIn("no request succeeded", perf["notes"][0])

    def test_quality_then_stages_and_area_breakdown(self):
        questions = [Question(f"{area}-{i}", category, "good", "json_match", {"value": {"x": 1}},
                              metadata={"cohort": module.REVISION})
                     for area, category, module in (("r", "Reasoning", quality_suite), ("s", "Safety", safety_suite))
                     for i in range(2)]
        self.addCleanup(standard_suite._provenance.cache_clear)
        standard_suite._provenance.cache_clear()
        with patch.object(standard_suite, "load_questions", return_value=questions), \
                patch.object(runner, "run_perf_suite", return_value=latency_result()):
            run = self.start(run_submission.make_run_options(mode="both", users=self.users))
        self.assertEqual((run["status"], run["scored_questions"]), ("completed", 4), run["error_message"])
        self.assertEqual(json.loads(run["quality_config_json"])["revision"], "standard-v1")
        quality = json.loads(run["quality_json"])
        self.assertEqual(quality["suite"]["name"], "standard")
        self.assertEqual(set(quality["summary"]["areas"]), {"reasoning", "safety"})
        self.assertEqual(json.loads(run["perf_json"])["kind"], "staged")
        # The downloadable report shows the areas and all three stages of the same run.
        html = html_reports.render_report([asyncio.run(html_reports.load_run(1))])
        for text in ("Area · Reasoning &amp; knowledge", "Area · Safety &amp; language", "STAGE 3 OF 3",
                     "Users at SLO by context cap"):
            self.assertIn(text, html)
        self.assertNotIn('id="perf-stage-capacity"', html)


class ViewTests(DatabaseCase):
    def staged_run(self):
        perf = staged.new_report("fake", {})
        perf["stages"] = {"latency": latency_run()["perf"],
                          "context": {"schema_version": 4, "kind": "context_sweep", "finished": True,
                                      "protocol": {"revision": "context-sweep-v3", "max_output_tokens": 4096,
                                                   "temperature": 0, "reference_tokenizer": "cl100k_base",
                                                   "rounds_per_cell": 1},
                                      "efforts": [], "notes": [],
                                      "cells": [{"effort": "default", "context_tokens": 8192, "concurrency": 1,
                                                 "status": "measured", "requests": 1, "completed": 1, "errors": 0,
                                                 "incomplete": 0, "latency": {"p50_ms": 2000},
                                                 "ttft": {"p50_ms": 600}, "aggregate_tokens_per_sec": 30,
                                                 "request_tokens_per_sec": {"p50": 20}}]}}
        perf["stage_errors"] = {"capacity": "Capacity stage: no arrivals"}
        return {"id": 1, "status": "failed", "label": "#1 staged", "perf": perf}

    def test_stage_views_align_staged_and_historical_runs(self):
        historical = {**latency_run(), "id": 2, "label": "#2 fixed"}
        view = html_reports.staged_view([self.staged_run(), historical], offline=True)
        # The retired capacity stage appears only when a run measured it.
        latency, context, users = view["stages"]
        self.assertEqual((users["key"], users["view"]), ("users", None))
        self.assertEqual((latency["key"], latency["runs"], latency["missing"]), ("latency", ["#1 staged", "#2 fixed"], []))
        self.assertEqual((context["runs"], context["missing"]), (["#1 staged"], ["#2 fixed"]))
        self.assertEqual(len(latency["view"]["load_views"]), 2)
        self.assertEqual(latency["view"]["capacity_views"], [])  # computed once per run instead
        self.assertEqual([c["label"] for c in view["capacity_views"]], ["#1 staged", "#2 fixed"])
        self.assertIn("#1 staged: Capacity stage: no arrivals", view["notes"])
        html = templates.get_template("performance_report.html").render(performance=view, offline=True)
        for text in ("STAGE 1 OF 3", "Latency by concurrency", "STAGE 3 OF 3", "None of these runs measured this stage",
                     "Serving capacity", "Not measured in #2 fixed"):
            self.assertIn(text, html)
        old = self.staged_run()
        old["perf"]["stages"]["capacity"] = load_test_report()
        keys = [s["key"] for s in html_reports.staged_view([old, historical], offline=True)["stages"]]
        self.assertEqual(keys, ["latency", "context", "capacity", "users"])
        self.assertNotIn("New context sweep", html)
        empty = templates.get_template("performance_report.html").render(
            performance=html_reports.staged_view([{"label": "q", "perf": {}}]), offline=True)
        self.assertIn("No performance measurements", empty)

    def test_capacity_reads_every_stage(self):
        run = self.staged_run()
        run["perf"]["stages"]["capacity"] = {
            "schema_version": 5, "kind": "open_loop", "protocol": {"workload": {"classes": []}},
            "summary": {"sustainable_status": "not_met"}, "steps": []}
        self.assertEqual([p["source"] for p in evidence(run["perf"])], ["Fixed load", "Context sweep"])
        result = estimate_capacity({**run, "status": "completed"})
        self.assertEqual(result["arrival"]["status"], "not_met")
        self.assertIsNotNone(result["scenarios"][0]["estimate"])

    def test_hydration_fills_the_context_stage(self):
        self.sql("INSERT INTO test_runs (id, model_id, status) VALUES (1, 1, 'completed')")
        self.sql("INSERT INTO performance_cells VALUES (1, 'default', 256, 1, ?)",
                 (json.dumps({"effort": "default", "context_tokens": 256, "concurrency": 1, "requests": 2,
                              "completed": 2}),))
        perf = asyncio.run(hydrate_sweep(1, self.staged_run()["perf"]))
        self.assertEqual((len(perf["stages"]["context"]["cells"]), perf["stages"]["context"]["successful_requests"]),
                         (1, 2))
        self.assertNotIn("cells", perf["stages"]["latency"])

    def test_stop_marks_the_staged_report_cancelled(self):
        from fastapi.testclient import TestClient

        from app.routes import runs

        perf = self.staged_run()["perf"]
        perf["running_stage"] = "context"
        self.sql("INSERT INTO test_runs (id, model_id, status, perf_json) VALUES (1, 1, 'running', ?)",
                 (json.dumps(perf),))
        app = FastAPI()
        app.include_router(runs.router)
        with patch.object(runs, "require_admin", AsyncMock(return_value={"id": 1, "role": "admin"})), \
                TestClient(app) as client:
            self.assertEqual(client.post("/runs/1/stop", follow_redirects=False).status_code, 302)
        stored = json.loads(self.sql("SELECT perf_json FROM test_runs")[0]["perf_json"])
        self.assertEqual((stored["cancelled"], stored["finished"]), (True, False))
        self.assertTrue(staged.stages_of(stored)["context"]["cancelled"])


class MigrationTests(DatabaseCase):
    def test_canaries_and_gates_move_to_the_standard_suite(self):
        self.sql("INSERT INTO test_runs (id, model_id, status) VALUES (1, 1, 'completed')")
        self.sql("""INSERT INTO canaries (id, name, model_id, suite, interval_hours, baseline_run_id)
                    VALUES (1, 'old', 1, 'tool-conformance', 6, 1), (2, 'team', 1, 'usecase:hr@1', 6, 1)""")
        gates = [{"type": "quality", "suite": "rigorous", "threshold": 0.7}, {"type": "operations"},
                 {"type": "quality", "suite": "usecase:hr@1", "threshold": 0.5}]
        self.sql("INSERT INTO decision_profiles (id, name, gates_json) VALUES (1, 'p', ?), (2, 'broken', 'nope')",
                 (json.dumps(gates),))

        async def migrate():
            async with aiosqlite.connect(self.path) as db:
                await database._migrate_retired_suites(db)
                await db.commit()

        asyncio.run(migrate())
        asyncio.run(migrate())  # idempotent
        canaries = self.sql("SELECT suite, baseline_run_id FROM canaries ORDER BY id")
        self.assertEqual([(c["suite"], c["baseline_run_id"]) for c in canaries],
                         [("standard", None), ("usecase:hr@1", 1)])
        stored = json.loads(self.sql("SELECT gates_json FROM decision_profiles WHERE id=1")[0]["gates_json"])
        self.assertEqual([g.get("suite") for g in stored], ["standard", None, "usecase:hr@1"])


if __name__ == "__main__":
    unittest.main()
