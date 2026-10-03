"""Shared fixed suite, simplified forms, plans, scheduling and complete reruns."""

import asyncio
from contextlib import ExitStack
import sqlite3
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import benchmark
from app.routes import admin, plans, runs
from app.services import benchmark_runner, run_queue, run_submission
from llm_client import ClientConfig
from models import Question, RequestMetrics, Result, TokenUsage
from quality_execution import score_response
from quality_report import make_report, paired_comparison
from quality_suite import load_questions, suite_hash

MODEL = {
    "id": 1,
    "name": "fake",
    "model_id": "fake",
    "base_url": "http://fake.invalid",
    "api_key": "unused",
}


class OptionsTests(unittest.TestCase):
    def test_only_two_benchmark_settings(self):
        self.assertEqual(
            set(benchmark.parser().parse_args([]).__dict__), {"mode", "max_concurrency", "report"}
        )
        self.assertEqual(run_submission.make_run_options(), {"mode": "both", "max_concurrency": 8})
        self.assertEqual(
            set(run_submission.spec_defaults()), {"model_id", "mode", "max_concurrency"}
        )
        self.assertIs(benchmark.run_quality, benchmark_runner.run_quality)
        self.assertIs(benchmark.run_perf_suite, benchmark_runner.run_perf_suite)
        for mode, maximum in [
            ("v6", 8),
            ("quality", 0),
            ("both", 33),
            ("both", True),
            ("both", 1.5),
        ]:
            with self.assertRaises(HTTPException):
                run_submission.make_run_options(mode=mode, max_concurrency=maximum)

    def test_fixed_questions_are_reproducible_and_strict(self):
        a, b = load_questions(), load_questions()
        self.assertEqual(len(a), 275)
        self.assertEqual(suite_hash(a), suite_hash(b))
        self.assertEqual(len({q.id for q in a}), 275)
        self.assertTrue(all(q.max_tokens == 65536 and q.pass_threshold == 1 for q in a))
        self.assertEqual(sum(bool(q.interaction) for q in a), 20)
        self.assertEqual(
            [q.metadata["context_tokens"] for q in a if q.metadata.get("context_tokens")],
            [8192, 32768] * 4,
        )

    def test_browser_uses_complete_bank_and_state_gates(self):
        from app.routes.tests_browser import load_tests_from_yaml

        categories, error = load_tests_from_yaml()
        self.assertIsNone(error)
        rows = [row for items in categories.values() for row in items]
        self.assertEqual(len(rows), 275)
        self.assertEqual(sum(r["evaluator"] == "interactive_state" for r in rows), 20)

    def test_cli_uses_full_suite_and_writes_schema_three(self):
        def quality(questions, config, maximum, **kwargs):
            self.assertEqual(len(questions), 275)
            self.assertEqual(
                (config.max_tokens, config.temperature, config.seed, maximum), (65536, 0, 0, 8)
            )
            return [Result(q, "fixture", 1, outcome="pass") for q in questions], 100, ""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.md"
            with (
                patch.object(benchmark, "load_dotenv"),
                patch.dict(
                    os.environ,
                    {
                        "OPENAI_BASE_URL": "http://fake.invalid",
                        "OPENAI_MODEL": "fake",
                        "OPENAI_KEY": "unused",
                    },
                ),
                patch("sys.argv", ["benchmark.py", "--mode", "quality", "--report", str(path)]),
                patch.object(benchmark, "run_quality", side_effect=quality),
                patch.object(benchmark, "run_perf_suite") as performance,
            ):
                self.assertEqual(benchmark.main(), 0)
            data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(data["quality"]["summary"]["count"], 275)
            self.assertEqual(data["quality"]["summary"]["category_balanced"], 1)
            self.assertIn("Balanced strict success", path.read_text(encoding="utf-8"))
            performance.assert_not_called()

    def test_balanced_primary_is_strict_and_paired_metric_matches(self):
        q = Question(
            "q",
            "Code",
            "Implement f",
            "code_exec",
            [
                {"function": "f", "args": [0], "expected": 0},
                {"function": "f", "args": [1], "expected": 1},
            ],
        )
        partial = score_response(
            q, "```python\ndef f(x): return 0\n```", TokenUsage(), RequestMetrics()
        )
        config = ClientConfig("http://fake.invalid", "unused", "fake")
        a = make_report([partial], config)
        self.assertEqual(a["summary"]["category_balanced"], 0)
        self.assertEqual(a["summary"]["category_balanced_criterion_achievement"], 0.5)
        b = make_report([replace(partial, score=1, evaluation=None, outcome="pass")], config)
        self.assertEqual(paired_comparison(a, b)["balanced_difference"], 1)

    def test_submission_validation_and_browser_timezone(self):
        with patch.object(run_submission, "fetch_all", AsyncMock(return_value=[MODEL])):
            prepared = asyncio.run(
                run_submission.validated_specs(
                    '[{"model_id":"1","mode":"both","max_concurrency":5}]'
                )
            )
        self.assertEqual(prepared[0][2], {"mode": "both", "max_concurrency": 5})
        self.assertEqual(
            run_submission.parse_browser_local("2026-10-03T05:00", "-120"),
            "2026-10-03T03:00:00+00:00",
        )
        self.assertEqual(
            run_submission.parse_browser_local("2026-12-03T05:00", "-60"),
            "2026-12-03T04:00:00+00:00",
        )
        self.assertIsNone(run_submission.parse_browser_local("", ""))
        for time, offset in [("bad", "0"), ("2026-10-03T05:00", ""), ("2026-10-03T05:00", "99999")]:
            with self.assertRaises(HTTPException):
                run_submission.parse_browser_local(time, offset)


class SubmissionTests(unittest.TestCase):
    """Exercise real routes, persistence and dispatch without model requests."""

    real_dispatch = staticmethod(run_queue.dispatch_next)

    def setUp(self):
        from app import config
        from app.auth import create_session_token

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        )
        self.path = Path(directory) / "bench.db"
        self.stack.enter_context(patch.object(config, "DATABASE_PATH", self.path))
        self.stack.enter_context(patch.object(run_queue, "_active_run_id", None))
        self.dispatch = self.stack.enter_context(patch.object(run_queue, "dispatch_next"))
        self.started = []
        self.stack.enter_context(
            patch.object(run_queue, "_spawn_benchmark", side_effect=self.spawn)
        )
        db = sqlite3.connect(self.path)
        try:
            db.executescript(Path("app/schema.sql").read_text(encoding="utf-8"))
            db.execute("INSERT INTO users (id, username, role) VALUES (1, 'tester', 'admin')")
            for model_id in (1, 2):
                db.execute(
                    "INSERT INTO models (id, name, base_url, api_key, model_id) VALUES (?, ?, ?, ?, ?)",
                    (model_id, f"fake{model_id}", MODEL["base_url"], "unused", MODEL["model_id"]),
                )
            db.commit()
        finally:
            db.close()
        app = FastAPI()

        @app.middleware("http")
        async def user(request, call_next):
            request.state.user = {"id": 1, "role": "admin", "username": "tester"}
            return await call_next(request)

        for router in (admin.router, plans.router, runs.router):
            app.include_router(router)
        self.client = self.stack.enter_context(TestClient(app))
        self.client.cookies.set("session", create_session_token(1))

    def sql(self, query, params=()):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            cursor = db.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]
            db.commit()
            return rows
        finally:
            db.close()

    def spawn(self, run_id, model, options):
        self.started.append((run_id, model["id"], dict(options)))
        self.sql("UPDATE test_runs SET status = 'running' WHERE id = ?", (run_id,))

    def post(self, specs=None, *, scheduled="", url="/admin/run"):
        return self.client.post(
            url,
            data={
                "name": "Comparison",
                "scheduled_at": scheduled,
                "tz_offset": "-120",
                "runs_json": json.dumps(
                    specs or [{"model_id": 1, "mode": "quality", "max_concurrency": 3}]
                ),
            },
            follow_redirects=False,
        )

    def test_one_form_for_new_and_edit(self):
        response = self.client.get("/admin/plans/new", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["location"], "/admin/run")
        page = self.client.get("/admin/run")
        self.assertEqual(page.status_code, 200)
        self.assertIn('action="/admin/run"', page.text)
        self.assertIn("275 questions", page.text)
        for word in (
            "Question limit",
            "Quality profile",
            "SLO",
            "Shared prefix",
            "Question seeds",
            "Context sweep",
            "Create Plan",
        ):
            self.assertNotIn(word, page.text)
        self.post(scheduled="2099-10-03T05:00")
        editor = self.client.get("/admin/plans/1/edit")
        self.assertEqual(editor.status_code, 200)
        for word in ("Maximum concurrency", "Quality + performance", "Add another run"):
            self.assertIn(word, page.text)
            self.assertIn(word, editor.text)
        self.assertIn("2099-10-03T03:00:00+00:00", editor.text)
        self.assertIn("this.initialOffset = date.getTimezoneOffset()", editor.text)
        self.assertIn("new Date(this.scheduledAt).getTimezoneOffset()", editor.text)
        listing = self.client.get("/admin/plans")
        self.assertEqual(listing.status_code, 200)
        self.assertIn("Queue &amp; schedules", listing.text)
        self.assertNotIn('href="/admin/plans/new"', listing.text)
        self.sql("UPDATE test_runs SET status = 'completed' WHERE id = 1")
        self.assertIn(">Finished</span>", self.client.get("/admin/plans").text)
        history = self.client.get("/runs")
        self.assertEqual(history.status_code, 200)
        self.assertIn("Part of submission #1", history.text)
        # There is no separate plan creation endpoint.
        self.assertEqual(self.client.post("/admin/plans").status_code, 405)

    def test_immediate_and_scheduler_dispatch_identical_options(self):
        # Use the real dispatcher. Spawning is still a synchronous local fake.
        self.stack.enter_context(patch.object(run_queue, "dispatch_next", self.real_dispatch))
        for mode in ("both", "quality", "performance"):
            for maximum in (1, 3, 8, 32):
                with self.subTest(mode=mode, maximum=maximum):
                    specs = [{"model_id": 1, "mode": mode, "max_concurrency": maximum}]
                    self.assertEqual(self.post(specs).status_code, 302)
                    immediate = self.sql("SELECT * FROM test_runs ORDER BY id DESC LIMIT 1")[0]
                    self.assertEqual(immediate["status"], "running")
                    self.assertEqual(
                        self.post(specs, scheduled="2099-10-03T05:00").status_code, 302
                    )
                    future = self.sql("SELECT * FROM test_runs ORDER BY id DESC LIMIT 1")[0]
                    self.assertEqual(future["status"], "pending")
                    for key in ("workers", "quality_config_json", "run_options_json"):
                        self.assertEqual(immediate[key], future[key])
                    self.assertEqual(immediate["workers"], min(4, maximum))
                    self.sql(
                        "UPDATE test_runs SET status = 'completed' WHERE id = ?", (immediate["id"],)
                    )
                    run_queue.on_run_finished(immediate["id"])
                    self.assertEqual(
                        self.sql("SELECT status FROM test_runs WHERE id = ?", (future["id"],))[0][
                            "status"
                        ],
                        "pending",
                    )
                    self.sql(
                        "UPDATE run_plans SET scheduled_at = '2020-01-01T00:00:00+00:00' WHERE id = ?",
                        (future["plan_id"],),
                    )
                    run_queue.dispatch_next()
                    self.assertEqual(self.started[-2][2], self.started[-1][2])
                    self.assertEqual(
                        self.started[-1][2], {"mode": mode, "max_concurrency": maximum}
                    )
                    self.sql(
                        "UPDATE test_runs SET status = 'completed' WHERE id = ?", (future["id"],)
                    )
                    run_queue.on_run_finished(future["id"])

    def test_batch_order_and_one_active_run(self):
        self.stack.enter_context(patch.object(run_queue, "dispatch_next", self.real_dispatch))
        specs = [
            {"model_id": model, "mode": mode, "max_concurrency": maximum}
            for model, mode, maximum in [(2, "performance", 1), (1, "both", 5), (2, "quality", 32)]
        ]
        self.assertEqual(self.post(specs).status_code, 302)
        rows = self.sql("SELECT * FROM test_runs ORDER BY id")
        self.assertEqual([r["model_id"] for r in rows], [2, 1, 2])
        self.assertEqual([r["status"] for r in rows], ["running", "pending", "pending"])
        for row in rows:
            self.assertEqual(self.started[-1][0], row["id"])
            self.sql("UPDATE test_runs SET status = 'completed' WHERE id = ?", (row["id"],))
            run_queue.on_run_finished(row["id"])
        self.assertEqual([entry[1] for entry in self.started], [2, 1, 2])
        self.assertIsNone(run_queue.active_run_id())

    def test_validation_is_identical_for_immediate_schedule_and_edit(self):
        self.post(scheduled="2099-10-03T05:00")
        invalid = [
            "bad",
            "{}",
            "[]",
            json.dumps([None]),
            json.dumps([{"model_id": True}]),
            json.dumps([{"model_id": 999}]),
            json.dumps([{"model_id": 1, "mode": {}}]),
            json.dumps([{"model_id": 1, "max_concurrency": 0}]),
            json.dumps([{"model_id": 1, "max_concurrency": 33}]),
            json.dumps([{"model_id": 1, "max_concurrency": True}]),
            json.dumps([{"model_id": 1, "max_concurrency": 1.5}]),
            json.dumps([{"model_id": 1, "test_ids": ["old"]}]),
            json.dumps([{"model_id": 1}] * 51),
        ]
        before = self.sql("SELECT * FROM test_runs")
        for raw in invalid:
            errors = []
            for url, scheduled in [
                ("/admin/run", ""),
                ("/admin/run", "2099-10-03T05:00"),
                ("/admin/plans/1/edit", "2099-10-03T05:00"),
            ]:
                response = self.client.post(
                    url,
                    data={"runs_json": raw, "scheduled_at": scheduled, "tz_offset": "-120"},
                    follow_redirects=False,
                )
                self.assertEqual(response.status_code, 422, (raw, url))
                errors.append(response.json())
            self.assertEqual(errors[0], errors[1])
            self.assertEqual(errors[0], errors[2])
        self.assertEqual(self.sql("SELECT * FROM test_runs"), before)
        self.assertEqual(len(self.sql("SELECT * FROM run_plans")), 1)

    def test_edit_clone_and_rerun_use_same_submission_settings(self):
        self.post(scheduled="2099-10-03T05:00")
        self.assertEqual(
            self.client.post("/admin/plans/1/clone", follow_redirects=False).status_code, 302
        )
        rows = self.sql("SELECT * FROM test_runs ORDER BY id")
        for key in ("workers", "run_options_json", "quality_config_json"):
            self.assertEqual(rows[0][key], rows[1][key])
        self.sql("UPDATE test_runs SET status = 'completed', quality_json = 'saved' WHERE id = 1")
        # A past protocol's extra settings are dropped when reusing the supported options.
        self.sql(
            "UPDATE test_runs SET run_options_json = ? WHERE id = 1",
            (json.dumps({"mode": "quality", "max_concurrency": 3, "test_ids": ["old"]}),),
        )
        response = self.client.post("/runs/1/rerun", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        repeated = self.sql("SELECT * FROM test_runs ORDER BY id DESC LIMIT 1")[0]
        self.assertEqual(
            json.loads(repeated["run_options_json"]), {"mode": "quality", "max_concurrency": 3}
        )
        self.assertIsNotNone(repeated["plan_id"])
        self.assertIsNone(
            self.sql("SELECT scheduled_at FROM run_plans WHERE id = ?", (repeated["plan_id"],))[0][
                "scheduled_at"
            ]
        )
        before = len(self.sql("SELECT * FROM test_runs"))
        self.client.post(f"/runs/{repeated['id']}/rerun", follow_redirects=False)
        self.assertEqual(len(self.sql("SELECT * FROM test_runs")), before)
        self.assertEqual(
            self.post(
                [{"model_id": 2, "mode": "performance", "max_concurrency": 5}],
                scheduled="",
                url="/admin/plans/1/edit",
            ).status_code,
            302,
        )
        history = self.sql("SELECT * FROM test_runs WHERE id = 1")[0]
        self.assertIsNone(history["plan_id"])
        self.assertEqual(history["quality_json"], "saved")
        edited = self.sql("SELECT * FROM test_runs WHERE plan_id = 1")[0]
        self.assertEqual((edited["model_id"], edited["workers"]), (2, 4))
        self.assertEqual(
            json.loads(edited["run_options_json"]), {"mode": "performance", "max_concurrency": 5}
        )
        self.assertIsNone(
            self.sql("SELECT scheduled_at FROM run_plans WHERE id = 1")[0]["scheduled_at"]
        )

    def test_creation_and_replacement_roll_back_all_rows(self):
        prepared = asyncio.run(run_submission.validated_specs(json.dumps([{"model_id": 1}])))
        # Simulate a model being deleted after validation, before the transaction.
        invalid = [prepared[0], (999, prepared[0][1], prepared[0][2])]
        with self.assertRaises(sqlite3.IntegrityError):
            asyncio.run(run_submission.submit_runs("failure", 1, None, invalid))
        self.assertEqual(self.sql("SELECT * FROM run_plans"), [])
        self.assertEqual(self.sql("SELECT * FROM test_runs"), [])
        self.dispatch.assert_not_called()
        self.post([{"model_id": 1}, {"model_id": 2}], scheduled="2099-10-03T05:00")
        self.sql("UPDATE test_runs SET status = 'completed', quality_json = 'saved' WHERE id = 1")
        before_plans = self.sql("SELECT * FROM run_plans")
        before_runs = self.sql("SELECT * FROM test_runs ORDER BY id")
        self.dispatch.reset_mock()
        with self.assertRaises(sqlite3.IntegrityError):
            asyncio.run(run_submission.replace_runs(1, "failure", 1, None, invalid))
        self.assertEqual(self.sql("SELECT * FROM run_plans"), before_plans)
        self.assertEqual(self.sql("SELECT * FROM test_runs ORDER BY id"), before_runs)
        self.dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
