"""Shared fixed suite, simplified forms, plans, scheduling and complete reruns."""

import asyncio
from contextlib import ExitStack
import sqlite3
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from app.routes import admin, plans, runs
from app.services import run_queue, run_submission
from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.models import Question, RequestMetrics, TokenUsage
from app.benchmarking.quality_execution import score_response
from app.benchmarking.quality_report import make_report, paired_comparison
from app.benchmarking.quality_suite import load_questions, suite_hash

MODEL = {
    "id": 1,
    "name": "fake",
    "model_id": "fake",
    "base_url": "http://fake.invalid",
    "api_key": "unused",
}


def slim(options):
    """Stored options without the users stage's (default) user model."""
    return {key: value for key, value in options.items() if key != "users"}


# The standard performance test's stored settings (docs/design-consolidation.md).
PERF = {"performance": "standard", "context_concurrency": 256, "target_ttft_ms": 2000, "target_output_rate": 40}


class OptionsTests(unittest.TestCase):
    def test_only_two_benchmark_settings(self):
        options = run_submission.make_run_options()
        self.assertEqual(slim(options), {"mode": "both", "max_concurrency": 8, "suite": "standard", **PERF})
        self.assertEqual(options["users"]["preset"], "mixed")
        self.assertEqual(
            set(run_submission.spec_defaults()),
            {"model_id", "mode", "max_concurrency", "suite", "users", *PERF},
        )
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
        self.assertEqual(len(a), 323)
        self.assertEqual(suite_hash(a), suite_hash(b))
        self.assertEqual(len({q.id for q in a}), 323)
        self.assertTrue(all(q.max_tokens == 65536 and q.pass_threshold == 1 for q in a))
        self.assertEqual(sum(bool(q.interaction) for q in a), 28)
        self.assertEqual(
            [q.metadata["context_tokens"] for q in a if q.metadata.get("context_tokens")],
            [8192, 32768] * 4,
        )

    def test_browser_uses_complete_bank_and_state_gates(self):
        from app.routes.tests_browser import load_tests_from_yaml

        categories, error = load_tests_from_yaml()
        self.assertIsNone(error)
        rows = [row for items in categories.values() for row in items]
        # The browser shows the whole standard suite.
        self.assertEqual(len(rows), 495)
        self.assertEqual(sum(r["evaluator"] == "interactive_state" for r in rows), 20)
        self.assertEqual(sum(r["evaluator"] == "behavioral_reconstruction" for r in rows), 8)

    def test_balanced_primary_is_fractional_and_paired_metric_matches(self):
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
        self.assertEqual(a["summary"]["category_balanced"], 0.5)
        self.assertEqual(a["summary"]["category_balanced_full_pass"], 0)
        self.assertEqual(a["summary"]["category_balanced_criterion_achievement"], 0.5)
        b = make_report([replace(partial, score=1, evaluation=None, outcome="pass")], config)
        self.assertEqual(paired_comparison(a, b)["balanced_difference"], 0.5)

    def test_submission_validation_and_browser_timezone(self):
        with patch.object(run_submission, "fetch_all", AsyncMock(return_value=[MODEL])):
            prepared = asyncio.run(
                run_submission.validated_specs(
                    '[{"model_id":"1","mode":"both","max_concurrency":5}]'
                )
            )
        self.assertEqual(slim(prepared[0][2]), {"mode": "both", "max_concurrency": 5, "suite": "standard", **PERF})
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
        self.seeds = []
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
        self.seeds.append(model.get("seed", 0))
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
        self.assertIn("standard suite of 495 tasks", page.text)
        for word in (
            "Question limit",
            "Quality profile",
            "Question seeds",
            "Context sweep",
            "Create Plan",
        ):
            self.assertNotIn(word, page.text)
        # Quality and performance are chosen independently; there is one
        # performance test, and its settings are folded under Advanced.
        self.assertNotIn('<option value="load">', page.text)
        self.post(scheduled="2099-10-03T05:00")
        editor = self.client.get("/admin/plans/1/edit")
        self.assertEqual(editor.status_code, 200)
        for word in ("Maximum concurrency", "> Quality</label>", "> Performance</label>", "Advanced settings",
                     "Add another run"):
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

    def test_b300_selectors_are_frozen_at_submission_and_mapping_is_optional(self):
        self.sql("UPDATE models SET b300_metrics_model='test/model' WHERE id=1")
        with patch.dict("os.environ", {"PROMETHEUS_B300_HOST": "smbea02n01", "PROMETHEUS_TIMESTAMP_SHIFT_SECONDS": "85"}):
            self.assertEqual(self.post(scheduled="2099-10-03T05:00").status_code, 302)
        row = self.sql("SELECT * FROM test_runs")[0]
        scope = json.loads(row["metrics_config_json"])
        self.assertEqual(scope, {"hardware": "B300", "host": "smbea02n01", "job": "vllm", "model_name": "test/model", "timestamp_shift_seconds": "85"})
        self.sql("UPDATE models SET b300_metrics_model='different/model' WHERE id=1")
        self.sql("UPDATE run_plans SET scheduled_at='2020-01-01T00:00:00+00:00'")
        with patch.object(run_queue, "_spawn_benchmark") as spawn:
            with patch.dict("os.environ", {"PROMETHEUS_B300_HOST": "other-host", "PROMETHEUS_TIMESTAMP_SHIFT_SECONDS": "0"}):
                self.real_dispatch()
            self.assertEqual(spawn.call_args.args[1]["_metrics_scope"], scope)
        self.assertEqual(self.post([{"model_id": 2}], scheduled="2099-10-03T05:00").status_code, 302)
        self.assertIsNone(json.loads(self.sql("SELECT metrics_config_json FROM test_runs WHERE model_id=2")[0]["metrics_config_json"]))

    def test_model_form_persists_validates_and_can_disable_b300_mapping(self):
        form = {"name": "fake1", "model_id": "fake", "base_url": "http://fake.invalid", "b300_metrics_model": "test/model"}
        with patch("app.services.url_guard.validate_endpoint"):
            response = self.client.post("/admin/models/1/edit", data=form, follow_redirects=False)
            self.assertEqual(response.status_code, 302)
            self.assertEqual(self.sql("SELECT b300_metrics_model FROM models WHERE id=1")[0]["b300_metrics_model"], "test/model")
            invalid = self.client.post("/admin/models/1/edit", data={**form, "b300_metrics_model": "bad\nname"}, follow_redirects=False)
            self.assertEqual(invalid.status_code, 422)
            self.assertEqual(self.sql("SELECT b300_metrics_model FROM models WHERE id=1")[0]["b300_metrics_model"], "test/model")
            self.client.post("/admin/models/1/edit", data={**form, "b300_metrics_model": ""}, follow_redirects=False)
            self.assertIsNone(self.sql("SELECT b300_metrics_model FROM models WHERE id=1")[0]["b300_metrics_model"])
            # GPU indices are normalised, validated and kept when a form omits them.
            self.client.post("/admin/models/1/edit", data={**form, "b300_gpus": " 3, 0-1 "}, follow_redirects=False)
            self.assertEqual(self.sql("SELECT b300_gpus FROM models WHERE id=1")[0]["b300_gpus"], "0,1,3")
            self.assertEqual(self.client.post("/admin/models/1/edit", data={**form, "b300_gpus": "0-99"},
                                              follow_redirects=False).status_code, 422)
            self.client.post("/admin/models/1/edit", data=form, follow_redirects=False)
            self.assertEqual(self.sql("SELECT b300_gpus FROM models WHERE id=1")[0]["b300_gpus"], "0,1,3")
            with patch.dict("os.environ", {"PROMETHEUS_B300_HOST": "smbea02n01", "PROMETHEUS_TIMESTAMP_SHIFT_SECONDS": ""}):
                self.assertEqual(self.post(scheduled="2099-10-03T05:00").status_code, 302)
            scope = json.loads(self.sql("SELECT metrics_config_json FROM test_runs")[0]["metrics_config_json"])
            self.assertEqual(scope["gpus"], [0, 1, 3])

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
                        slim(self.started[-1][2]),
                        {"mode": mode, "max_concurrency": maximum,
                         **({"suite": "standard"} if mode != "performance" else {}),
                         **(PERF if mode != "quality" else {})},
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
            json.loads(repeated["run_options_json"]), {"mode": "quality", "max_concurrency": 3, "suite": "standard"}
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
            slim(json.loads(edited["run_options_json"])), {"mode": "performance", "max_concurrency": 5, **PERF}
        )
        self.assertIsNone(
            self.sql("SELECT scheduled_at FROM run_plans WHERE id = 1")[0]["scheduled_at"]
        )

    def test_repeat_groups_and_suites_round_trip(self):
        specs = [
            {"model_id": 1, "mode": "quality", "max_concurrency": 3,
             "suite": "standard", "repeats": 3},
            {"model_id": 2, "mode": "quality", "max_concurrency": 3},
        ]
        self.assertEqual(self.post(specs, scheduled="2099-10-03T05:00").status_code, 302)
        rows = self.sql("SELECT * FROM test_runs ORDER BY id")
        self.assertEqual(len(rows), 4)
        group = rows[:3]
        self.assertEqual({r["repeat_group_id"] for r in group}, {rows[0]["id"]})
        self.assertEqual([r["repeat_index"] for r in group], [0, 1, 2])
        self.assertEqual({r["repeat_count"] for r in group}, {3})
        # Repeat i samples with seed i; seed 0 stays implicit as in ordinary runs.
        self.assertNotIn("seed", json.loads(group[0]["decoding_config_json"]))
        self.assertEqual([json.loads(r["decoding_config_json"]).get("seed") for r in group[1:]], [1, 2])
        for row in group:
            self.assertEqual(json.loads(row["run_options_json"]),
                             {"mode": "quality", "max_concurrency": 3, "suite": "standard"})
            self.assertEqual(json.loads(row["quality_config_json"])["name"], "standard")
        single = rows[3]
        self.assertEqual((single["repeat_group_id"], single["repeat_index"], single["repeat_count"]),
                         (None, None, None))
        # New runs store the suite explicitly, also when it is the default.
        self.assertEqual(json.loads(single["run_options_json"]),
                         {"mode": "quality", "max_concurrency": 3, "suite": "standard"})
        self.assertEqual(json.loads(single["quality_config_json"])["name"], "standard")

        # The dispatcher hands each member its own seed and the suite option.
        self.stack.enter_context(patch.object(run_queue, "dispatch_next", self.real_dispatch))
        self.sql("UPDATE run_plans SET scheduled_at = '2020-01-01T00:00:00+00:00'")
        run_queue.dispatch_next()
        for row in rows:
            self.sql("UPDATE test_runs SET status = 'completed' WHERE id = ?", (row["id"],))
            run_queue.on_run_finished(row["id"])
        self.assertEqual(self.seeds, [0, 1, 2, 0])
        self.assertEqual([s[2].get("suite") for s in self.started], ["standard"] * 4)

        # Editing and cloning collapse the group back into one specification.
        self.assertEqual(self.client.post("/admin/plans/1/clone", follow_redirects=False).status_code, 302)
        clone = self.sql("SELECT * FROM test_runs WHERE id > 4 ORDER BY id")
        self.assertEqual(len(clone), 4)
        self.assertEqual([r["repeat_index"] for r in clone], [0, 1, 2, None])
        self.assertEqual({r["repeat_group_id"] for r in clone[:3]}, {clone[0]["id"]})
        self.assertEqual(run_submission.specs_from_runs(rows)[0]["repeats"], 3)
        self.assertEqual(len(run_submission.specs_from_runs(rows)), 2)

        # Rerunning any member reruns the whole group.
        before = len(self.sql("SELECT * FROM test_runs"))
        self.assertEqual(self.client.post(f"/runs/{rows[1]['id']}/rerun", follow_redirects=False).status_code, 302)
        added = self.sql("SELECT * FROM test_runs WHERE id > ? ORDER BY id", (before,))
        self.assertEqual([r["repeat_index"] for r in added], [0, 1, 2])

    def test_suite_and_repeat_validation(self):
        invalid = [
            [{"model_id": 1, "repeats": 0}],
            [{"model_id": 1, "repeats": 11}],
            [{"model_id": 1, "repeats": "x"}],
            [{"model_id": 1, "repeats": True}],
            [{"model_id": 1, "repeats": 1.5}],
            [{"model_id": 1, "suite": "nope"}],
            [{"model_id": 1, "mode": "performance", "suite": "tool-conformance"}],
            # Retired built-in suites are part of the standard suite now.
            [{"model_id": 1, "mode": "quality", "suite": "tool-conformance"}],
            [{"model_id": 1, "mode": "quality", "suite": "rigorous"}],
            [{"model_id": 1, "repeats": 10}] * 6,
        ]
        for specs in invalid:
            with self.subTest(specs=specs[0]):
                self.assertEqual(self.post(specs).status_code, 422)
        self.assertEqual(self.sql("SELECT * FROM test_runs"), [])
        self.assertEqual(self.post([{"model_id": 1, "repeats": 10}] * 5,
                                   scheduled="2099-10-03T05:00").status_code, 302)
        self.assertEqual(len(self.sql("SELECT * FROM test_runs")), 50)
        page = self.client.get("/admin/run")
        self.assertIn("Standard quality suite", page.text)
        self.assertNotIn("Tool-calling &amp; structured-output conformance", page.text)

    def test_creation_and_replacement_roll_back_all_rows(self):
        prepared = asyncio.run(run_submission.validated_specs(json.dumps([{"model_id": 1}])))
        # Simulate a model being deleted after validation, before the transaction.
        invalid = [prepared[0], (999, prepared[0][1], prepared[0][2])]
        with self.assertRaises(HTTPException):
            asyncio.run(run_submission.submit_runs("failure", 1, None, invalid))
        self.assertEqual(self.sql("SELECT * FROM run_plans"), [])
        self.assertEqual(self.sql("SELECT * FROM test_runs"), [])
        self.dispatch.assert_not_called()
        self.post([{"model_id": 1}, {"model_id": 2}], scheduled="2099-10-03T05:00")
        self.sql("UPDATE test_runs SET status = 'completed', quality_json = 'saved' WHERE id = 1")
        before_plans = self.sql("SELECT * FROM run_plans")
        before_runs = self.sql("SELECT * FROM test_runs ORDER BY id")
        self.dispatch.reset_mock()
        with self.assertRaises(HTTPException):
            asyncio.run(run_submission.replace_runs(1, "failure", 1, None, invalid))
        self.assertEqual(self.sql("SELECT * FROM run_plans"), before_plans)
        self.assertEqual(self.sql("SELECT * FROM test_runs ORDER BY id"), before_runs)
        self.dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
