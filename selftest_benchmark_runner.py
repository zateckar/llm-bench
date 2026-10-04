"""Offline regressions for saved answers, run finalisation, and progress reporting."""

import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from app.storage import DETECT_TYPES
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config as app_config
from app.routes import admin, runs
from app.services import benchmark_runner as runner
from app.benchmarking.llm_client import ChatClient, ClientConfig
from app.benchmarking.models import Question, RequestMetrics, TokenUsage
from app.benchmarking.quality_suite import provenance

ROOT = Path(__file__).parent


class FakeClient:
    def __init__(self, config):
        self.config = config

    def complete(self, prompt, *args, **kwargs):
        if prompt == "error":
            return (
                "[API ERROR]",
                TokenUsage(),
                RequestMetrics(
                    ok=False,
                    error="HTTP 500: Cannot connect to model server",
                    attempts=3,
                ),
            )
        return (
            '{"x":0}' if prompt == "wrong" else '{"x":1}',
            TokenUsage(10, 5),
            RequestMetrics(latency_ms=10, ttft_ms=2, finish_reason="stop"),
        )


class RunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "bench.db"
        with closing(sqlite3.connect(self.path, detect_types=DETECT_TYPES)) as db:
            db.executescript((ROOT / "app/schema.sql").read_text(encoding="utf-8"))
            db.execute(
                "INSERT INTO models(id,name,base_url,api_key,model_id) "
                "VALUES(1,'fake','http://fake.invalid','unused','fake')"
            )
            options = provenance()
            db.execute(
                "INSERT INTO test_runs(id,model_id,status,quality_config_json) "
                "VALUES(1,1,'pending',?)",
                (json.dumps(options),),
            )
            db.commit()
        self.model = dict(
            id=1,
            name="fake",
            base_url="http://fake.invalid",
            api_key="unused",
            model_id="fake",
        )
        self.questions = [
            Question(str(i), "Reasoning", "good", "json_match", {"value": {"x": 1}})
            for i in range(3)
        ]
        for patcher in (
            patch.object(app_config, "DATABASE_PATH", self.path),
            patch("app.services.url_guard.validate_endpoint"),
            patch.object(runner, "load_questions", return_value=self.questions),
            patch("app.benchmarking.llm_client.ChatClient", FakeClient),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def sql(self, query, params=()):
        with closing(sqlite3.connect(self.path, detect_types=DETECT_TYPES)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(query, params).fetchall()
            db.commit()
            return [dict(row) for row in rows]

    def set_prompts(self, prompts):
        self.questions[:] = [
            Question(str(i), "Reasoning", prompt, "json_match", {"value": {"x": 1}})
            for i, prompt in enumerate(prompts)
        ]

    def run_benchmark(self, workers=1, run_perf=False, run_context=False):
        finished = Mock()
        runner._run_benchmark(
            run_id=1,
            model=self.model,
            mode="both" if run_perf or run_context else "quality",
            max_concurrency=workers,
            on_finish=finished,
        )
        finished.assert_called_once_with(1)
        self.assertEqual(self.sql("SELECT * FROM benchmark_progress"), [])
        return self.sql("SELECT * FROM test_runs WHERE id=1")[0]

    def test_all_endpoint_errors_fail_the_run(self):
        for q in self.questions:
            q.prompt = "error"
        run = self.run_benchmark()
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["total_questions"], 3)
        self.assertEqual(run["scored_questions"], 0)
        self.assertEqual(run["error_count"], 3)
        self.assertIn("No questions could be scored", run["error_message"])
        self.assertIn("HTTP 500", run["error_message"])
        self.assertEqual(len(self.sql("SELECT * FROM test_results")), 3)
        self.assertIsNone(json.loads(run["quality_json"])["summary"]["category_balanced"])

    def test_wrong_answers_are_completed_scored_results(self):
        for q in self.questions:
            q.prompt = "wrong"
        run = self.run_benchmark()
        self.assertEqual(
            (run["status"], run["scored_questions"], run["passed_questions"]), ("completed", 3, 0)
        )
        self.assertEqual(run["error_count"], 0)

    def test_partial_endpoint_failure_keeps_scores_with_multiple_workers(self):
        self.questions[1].prompt = "error"
        run = self.run_benchmark(workers=2)
        self.assertEqual(
            (run["status"], run["scored_questions"], run["passed_questions"]), ("completed", 2, 2)
        )
        self.assertEqual(run["error_count"], 1)
        self.assertEqual(run["avg_score"], 1)

    def check_sustained_outage(self, workers):
        self.set_prompts(["error"] * 32)
        with (
            patch.object(runner, "run_perf_suite") as perf,
        ):
            run = self.run_benchmark(workers=workers, run_perf=True, run_context=True)
        recorded = sum(
            r["request_ok"] == 0
            and json.loads(r["quality_metadata_json"])["outcome"] == "endpoint_error"
            for r in self.sql("SELECT * FROM test_results")
        )
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["total_questions"], 32)
        self.assertGreaterEqual(recorded, 8)
        self.assertLessEqual(recorded, 8 + workers - 1)
        self.assertEqual(run["error_count"], recorded)
        self.assertIn("Stopped after eight consecutive endpoint failures", run["error_message"])
        perf.assert_not_called()

    def test_sustained_outage_stops_sequential_run_and_skips_probes(self):
        self.check_sustained_outage(workers=1)

    def test_sustained_outage_stops_parallel_run_and_skips_probes(self):
        self.check_sustained_outage(workers=4)

    def test_outage_after_success_preserves_the_earlier_scores(self):
        self.set_prompts(["good"] * 2 + ["error"] * 16)
        run = self.run_benchmark()
        self.assertEqual(
            (run["status"], run["scored_questions"], run["passed_questions"]), ("failed", 2, 2)
        )
        self.assertEqual(run["avg_score"], 1)
        self.assertEqual(len(self.sql("SELECT * FROM test_results")), 18)

    def test_endpoint_recovery_and_wrong_answers_reset_failure_streak(self):
        self.set_prompts(["error"] * 7 + ["wrong"] + ["error"] * 7 + ["good"])
        run = self.run_benchmark()
        self.assertEqual(
            (run["status"], run["scored_questions"], run["passed_questions"]), ("completed", 2, 1)
        )
        self.assertEqual(len(self.sql("SELECT * FROM test_results")), 16)

    def test_unsupported_context_does_not_count_as_sustained_endpoint_outage(self):
        self.set_prompts(["error"] * 7 + ["context"] + ["error"] * 7 + ["good"])
        self.questions[7].metadata["context_tokens"] = 8192
        original = FakeClient.complete

        def complete(client, prompt, *args, **kwargs):
            if prompt == "context":
                return (
                    "[API ERROR]",
                    TokenUsage(),
                    RequestMetrics(
                        ok=False,
                        error="HTTP 400 context_length_exceeded",
                    ),
                )
            return original(client, prompt, *args, **kwargs)

        with patch.object(FakeClient, "complete", complete):
            run = self.run_benchmark()
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["scored_questions"], 1)
        self.assertEqual(len(self.sql("SELECT * FROM test_results")), 16)

    def test_short_all_error_run_skips_performance_probes(self):
        self.set_prompts(["error"] * 3)
        with patch.object(runner, "run_perf_suite") as perf:
            run = self.run_benchmark(run_perf=True)
        self.assertEqual(run["status"], "failed")
        perf.assert_not_called()

    def test_failed_request_retains_time_spent_connecting_and_retrying(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                elapsed = [0.0]
                response = SimpleNamespace(
                    status_code=500,
                    text="Cannot connect to model server",
                    close=lambda: None,
                )

                def post(*args, **kwargs):
                    elapsed[0] += 5
                    return response

                def sleep(seconds):
                    elapsed[0] += seconds

                client = ChatClient(
                    ClientConfig(
                        base_url="http://fake.invalid",
                        api_key="unused",
                        model="fake",
                        stream=streaming,
                        max_retries=3,
                        retry_delay=2,
                    )
                )
                try:
                    with (
                        patch.object(client.session, "post", side_effect=post) as requests,
                        patch("app.benchmarking.llm_client.time.perf_counter", side_effect=lambda: elapsed[0]),
                        patch("app.benchmarking.llm_client.time.sleep", side_effect=sleep),
                    ):
                        _, tokens, metrics = client.complete("test")
                    self.assertFalse(metrics.ok)
                    self.assertEqual(metrics.attempts, 3)
                    self.assertEqual(metrics.latency_ms, 21000)
                    self.assertEqual(tokens.total_tokens, 0)
                    self.assertEqual(requests.call_count, 3)
                finally:
                    client.session.close()

    def test_report_failure_preserves_saved_answers_and_summary(self):
        with (
            patch.object(runner, "make_quality_report", side_effect=ValueError("report broke")),
            self.assertLogs(runner.logger, level="ERROR"),
        ):
            run = self.run_benchmark()
        self.assertEqual(
            (run["status"], run["scored_questions"], run["passed_questions"]), ("failed", 3, 3)
        )
        self.assertEqual(run["avg_score"], 1)
        self.assertEqual(run["total_prompt_tokens"], 30)
        self.assertIn("Quality report generation failed: report broke", run["error_message"])
        self.assertEqual(len(self.sql("SELECT * FROM test_results")), 3)

    def test_finalisation_failure_recovers_summary_from_database(self):
        self.questions[1].prompt = "error"
        with (
            patch.object(
                runner, "_finish_run", side_effect=sqlite3.OperationalError("finish broke")
            ),
            self.assertLogs(runner.logger, level="ERROR"),
        ):
            run = self.run_benchmark()
        self.assertEqual(
            (
                run["status"],
                run["total_questions"],
                run["scored_questions"],
                run["passed_questions"],
                run["error_count"],
            ),
            ("failed", 3, 2, 2, 1),
        )
        self.assertEqual(run["avg_score"], 1)
        self.assertEqual(run["weighted_score"], 1)
        self.assertEqual(run["total_completion_tokens"], 10)
        self.assertIn("finish broke", run["error_message"])

    def test_failed_save_does_not_count_an_unstored_answer(self):
        original = runner._store_result

        def save(run_id, index, result):
            if index == 2:
                raise sqlite3.OperationalError("save broke")
            return original(run_id, index, result)

        with (
            patch.object(runner, "_store_result", side_effect=save),
            self.assertLogs(runner.logger, level="ERROR"),
        ):
            run = self.run_benchmark()
        self.assertEqual(
            (
                run["status"],
                run["total_questions"],
                run["scored_questions"],
                run["passed_questions"],
            ),
            ("failed", 3, 1, 1),
        )
        self.assertEqual(len(self.sql("SELECT * FROM test_results")), 1)

    def test_late_failure_does_not_overwrite_a_stopped_run(self):
        self.sql(
            "UPDATE test_runs SET status='failed', error_message='Stopped by admin' WHERE id=1"
        )
        runner._mark_run_failed(1, "late failure")
        self.assertEqual(
            self.sql("SELECT error_message FROM test_runs")[0]["error_message"], "Stopped by admin"
        )

    def test_progress_stream_exposes_saved_successes_and_errors(self):
        self.questions[1].prompt = "error"
        self.run_benchmark()
        self.sql("UPDATE test_runs SET status='running' WHERE id=1")
        runner._update_progress(1, "Reasoning: 2", 3, 3, "Processed 3/3 questions")

        async def read_progress():
            request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
            with patch.object(admin, "_admin_required", return_value={"role": "admin"}):
                response = await admin.admin_run_stream(request, 1)
            try:
                event = await anext(response.body_iterator)
                data = json.loads(event.split("data: ", 1)[1])
                self.assertEqual(data["recorded_questions"], 3)
                self.assertEqual(data["scored_questions"], 2)
                self.assertEqual(data["request_errors"], 1)
                self.assertIn("HTTP 500", data["last_request_error"])
            finally:
                await response.body_iterator.aclose()

        asyncio.run(read_progress())

    def test_cancelled_questions_do_not_count_as_request_errors(self):
        self.set_prompts(["error"] * 20)
        run = self.run_benchmark()
        self.assertEqual(run["error_count"], 8)
        self.assertEqual(len(self.sql("SELECT * FROM test_results")), 20)

        async def read_done():
            request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
            with patch.object(admin, "_admin_required", return_value={"role": "admin"}):
                response = await admin.admin_run_stream(request, 1)
            try:
                event = await anext(response.body_iterator)
                data = json.loads(event.split("data: ", 1)[1])
                self.assertEqual(data["request_errors"], 8)
                self.assertIn("HTTP 500", data["last_request_error"])
            finally:
                await response.body_iterator.aclose()

        asyncio.run(read_done())

    def test_done_stream_includes_failure_reason_and_counts(self):
        for q in self.questions:
            q.prompt = "error"
        self.run_benchmark()

        async def read_done():
            request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
            with patch.object(admin, "_admin_required", return_value={"role": "admin"}):
                response = await admin.admin_run_stream(request, 1)
            try:
                event = await anext(response.body_iterator)
                self.assertTrue(event.startswith("event: done"))
                data = json.loads(event.split("data: ", 1)[1])
                self.assertEqual(data["request_errors"], 3)
                self.assertEqual(data["scored_questions"], 0)
                self.assertIn("HTTP 500", data["error_message"])
            finally:
                await response.body_iterator.aclose()

        asyncio.run(read_done())

    def test_historical_all_error_run_displays_no_score_and_endpoint_reason(self):
        for q in self.questions:
            q.prompt = "error"
        self.run_benchmark()
        # Older runners called these runs completed. Render them correctly
        # without mutating their stored status or grading.
        self.sql("UPDATE test_runs SET status='completed', error_message='' WHERE id=1")
        app = FastAPI()
        app.include_router(runs.router)
        with (
            TestClient(app) as client,
            patch.object(
                runs,
                "get_current_user",
                AsyncMock(return_value={"role": "admin"}),
            ),
        ):
            detail = client.get("/runs/1")
            listing = client.get("/runs")
        self.assertEqual(detail.status_code, 200)
        self.assertIn("No quality score is available", detail.text)
        self.assertIn("Cannot connect to model server", detail.text)
        self.assertIn("No scored answers", listing.text)
        self.assertIn("n/a", listing.text)
        self.assertNotIn("0/3", listing.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
