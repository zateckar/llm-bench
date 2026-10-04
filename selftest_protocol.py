"""Offline budget, recovery, configuration, migration and queue regressions."""

import asyncio
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from app.storage import DETECT_TYPES
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config as app_config
from app.benchmarking.llm_client import ChatClient, ClientConfig, client_protocol
from app.benchmarking import perf
from app.benchmarking.models import Question, RequestMetrics, TokenUsage
from app.benchmarking.quality_execution import execute_question, run_quality
from app.benchmarking.quality_report import make_report, paired_comparison
from app.database import _apply_migrations, get_db
from app.routes import admin
from app.services import benchmark_runner, run_queue, run_submission
from selftest_client import Response, event
from selftest_perf import FakeClient as PerfClient

ROOT = Path(__file__).parent
CONFIG = ClientConfig("http://fake.invalid", "unused", "fake", max_tokens=64,
                      retry_transport_errors=False)
QUESTION = Question("q", "Reasoning", "Return JSON with x=1", "json_match", {"value": {"x": 1}})


def answer(text, tokens=8, finish="stop", ok=True):
    return text, TokenUsage(10, tokens), RequestMetrics(
        latency_ms=10, ttft_ms=2, finish_reason=finish, ok=ok,
        error=None if ok else "HTTP 500", streamed=True, stream_chunks=10, stream_span_ms=9,
    )


class ScriptClient:
    def __init__(self, responses):
        self.config = CONFIG
        self.responses = iter(responses)
        self.calls = []

    def complete(self, prompt, system_prompt, **kwargs):
        self.calls.append(([{"role": "user", "content": prompt}], kwargs))
        return next(self.responses)

    def complete_messages(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return next(self.responses)


class RecoveryTests(unittest.TestCase):
    def test_real_client_recovers_and_applies_model_settings_to_both_payloads(self):
        config = replace(CONFIG, temperature=1, reasoning_effort="high")
        responses = [
            Response([event(reasoning="work", finish="length"),
                      {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 56}}, b"[DONE]"]),
            Response([event('{"x":1}', finish="stop"),
                      {"choices": [], "usage": {"prompt_tokens": 15, "completion_tokens": 8}}, b"[DONE]"]),
        ]
        with patch("requests.Session.post", side_effect=responses) as post:
            results, _, _ = run_quality([QUESTION], config, 1)
        self.assertTrue(results[0].passed)
        self.assertEqual(results[0].tokens.completion_tokens, 64)
        self.assertEqual(results[0].tokens.prompt_tokens, 25)
        payloads = [c.kwargs["json"] for c in post.call_args_list]
        self.assertEqual([p["max_tokens"] for p in payloads], [56, 8])
        self.assertTrue(all(p["temperature"] == 1 and p["reasoning_effort"] == "high" for p in payloads))
        self.assertEqual(payloads[1]["messages"][1]["content"], "<think>work</think>")

    def test_length_recovers_with_one_shared_budget_and_full_history(self):
        first = "<think>unfinished reasoning</think>"
        client = ScriptClient([answer(first, 56, "length"), answer('{"x":1}')])
        result = execute_question(QUESTION, client)
        self.assertTrue(result.passed)
        self.assertEqual([c[1]["max_tokens"] for c in client.calls], [56, 8])
        self.assertEqual(client.calls[1][0][1]["content"], first)
        self.assertEqual((result.tokens.prompt_tokens, result.tokens.completion_tokens), (20, 64))
        self.assertEqual((result.metrics.latency_ms, result.metrics.ttft_ms, result.metrics.attempts), (20, 2, 2))
        self.assertIsNone(result.metrics.stream_span_ms)
        self.assertTrue(result.diagnostics["recovery"]["recovered"])
        self.assertFalse(result.diagnostics["recovery"]["initial_passed"])
        report = make_report([result], CONFIG)
        self.assertEqual(report["protocol"]["max_output_tokens"], 64)
        self.assertEqual(report["protocol"]["execution"]["total_output_budget"], 64)
        self.assertEqual(report["protocol"]["execution"]["final_answer_reserve"], 8)
        self.assertEqual(report["summary"]["recovery"]["model_calls"], 2)
        self.assertEqual(report["summary"]["recovery"]["recovered"], 1)
        old = {**report, "protocol": {**report["protocol"]}}
        old["protocol"].pop("execution")
        self.assertFalse(paired_comparison(old, report)["compatible"])
        self.assertFalse(paired_comparison(report, make_report([result], replace(CONFIG, temperature=1)))["compatible"])
        self.assertFalse(paired_comparison(report, make_report([result], replace(CONFIG, reasoning_effort="high")))["compatible"])

    def test_reasoning_only_recovers_even_with_stop(self):
        client = ScriptClient([answer("<think>work</think>"), answer('{"x":1}')])
        result = execute_question(QUESTION, client)
        self.assertTrue(result.passed)
        self.assertEqual(result.diagnostics["recovery"]["initial_outcome"], "missing_answer")

    def test_complete_wrong_and_empty_responses_do_not_get_oracle_retries(self):
        for text in ('{"x":1}', '{"x":0}', ""):
            client = ScriptClient([answer(text)])
            result = execute_question(QUESTION, client)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(result.passed, text == '{"x":1}')
            self.assertFalse(result.diagnostics["recovery"]["attempted"])

    def test_still_truncated_stops_after_two_and_retains_both_outputs(self):
        client = ScriptClient([answer("<think>a</think>", 56, "length"), answer("<think>b</think>", 8, "length")])
        result = execute_question(QUESTION, client)
        self.assertEqual(result.outcome, "truncation")
        self.assertTrue(result.is_scored)
        self.assertFalse(result.passed)
        self.assertEqual([c["response"] for c in result.diagnostics["quality_requests"]], ["<think>a</think>", "<think>b</think>"])

    def test_second_output_is_a_replacement_and_no_answer_key_is_sent(self):
        q = replace(QUESTION, expected={"value": {"x": "SECRET_EXPECTED"}})
        client = ScriptClient([answer('{"x":', 56, "length"), answer('1}')])
        result = execute_question(q, client)
        self.assertFalse(result.passed)
        self.assertEqual(result.response, "1}")
        self.assertNotIn("SECRET_EXPECTED", json.dumps(client.calls))

    def test_cancel_before_recovery_does_not_start_a_second_call(self):
        client = ScriptClient([answer("<think>a</think>", 56, "length")])
        result = execute_question(QUESTION, client, cancelled=lambda: bool(client.calls))
        self.assertEqual(result.outcome, "cancelled")
        self.assertFalse(result.is_scored)
        self.assertEqual(len(client.calls), 1)

    def test_failures_and_budget_violations_preserve_cost_and_stop(self):
        for responses in (
            [answer("error", 0, ok=False)],
            [answer("<think>a</think>", 56, "length"), answer("error", 0, ok=False)],
            [answer("bad", 57, "length")],
        ):
            client = ScriptClient(responses)
            result = execute_question(QUESTION, client)
            self.assertFalse(result.is_scored)
            self.assertEqual(result.tokens.prompt_tokens, 10 * len(responses))
            self.assertEqual(len(client.calls), len(responses))

    def test_saved_recovery_errors_hide_endpoints_but_keep_successful_text(self):
        first = "<think>Source https://example.com/legitimate</think>"
        failure = answer("[API ERROR: https://private.invalid/completions]", 0, ok=False)
        failure[2].error = "HTTP 500: https://private.invalid/completions"
        result = execute_question(QUESTION, ScriptClient([answer(first, 56, "length"), failure]))
        benchmark_runner._sanitize_result_errors(result)
        self.assertEqual(result.diagnostics["quality_requests"][0]["response"], first)
        self.assertNotIn("private.invalid", json.dumps(make_report([result], CONFIG)))

    def test_small_allocations_and_missing_usage_stay_within_requested_budget(self):
        for budget in (1, 7, 8, 32, 65536):
            client = ScriptClient([answer("<think>a</think>", 0, "length"), answer('{"x":1}', 0)])
            client.config = replace(CONFIG, max_tokens=budget)
            execute_question(QUESTION, client)
            self.assertLessEqual(sum(c[1]["max_tokens"] for c in client.calls), budget)
            self.assertEqual(len(client.calls), 1 if budget < 8 else 2)

    def test_recovery_stays_in_the_question_worker_and_runs_in_parallel(self):
        barrier = threading.Barrier(4)
        lock = threading.Lock()
        active = peak = calls = 0

        class ParallelClient:
            def __init__(self, config):
                self.config, self.session = config, Mock()

            def complete(self, *args, **kwargs):
                nonlocal active, peak, calls
                with lock:
                    active += 1
                    calls += 1
                    peak = max(peak, active)
                barrier.wait(timeout=5)
                with lock:
                    active -= 1
                return answer("<think>a</think>", 56, "length")

            def complete_messages(self, *args, **kwargs):
                nonlocal calls
                with lock:
                    calls += 1
                return answer('{"x":1}')

        with patch("app.benchmarking.llm_client.ChatClient", ParallelClient):
            results, _, _ = run_quality([replace(QUESTION, id=str(i)) for i in range(8)], CONFIG, 8)
        self.assertEqual(peak, 4)
        self.assertEqual(calls, 16)
        self.assertTrue(all(r.passed for r in results))


class ModelSettingsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "bench.db"
        with closing(sqlite3.connect(self.path, detect_types=DETECT_TYPES)) as db:
            db.executescript((ROOT / "app/schema.sql").read_text(encoding="utf-8"))
            db.execute("INSERT INTO users(id,username,role) VALUES(1,'admin','admin')")
            db.commit()
        patcher = patch.object(app_config, "DATABASE_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        app = FastAPI()

        @app.middleware("http")
        async def user(request, call_next):
            request.state.user = {"id": 1, "role": "admin", "username": "admin"}
            return await call_next(request)

        app.include_router(admin.router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        guard = patch("app.services.url_guard.validate_endpoint")
        guard.start()
        self.addCleanup(guard.stop)

    def sql(self, query):
        with closing(sqlite3.connect(self.path, detect_types=DETECT_TYPES)) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute(query)]
            db.commit()
            return rows

    def create(self, **settings):
        return self.client.post("/admin/models", data={
            "name": "fake", "base_url": "http://fake.invalid", "api_key": "secret",
            "model_id": "fake", **settings,
        }, follow_redirects=False)

    def test_create_edit_validate_and_render_settings_without_showing_key(self):
        self.assertEqual(self.create(temperature="1", reasoning_effort="xhigh").status_code, 302)
        model = self.sql("SELECT * FROM models")[0]
        config = benchmark_runner._build_client_config(model)
        self.assertEqual((config.temperature, config.reasoning_effort), (1, "xhigh"))
        edit = self.client.get("/admin/models/1/edit")
        self.assertIn('name="temperature"', edit.text)
        self.assertIn('value="xhigh" selected', edit.text)
        self.assertNotIn('value="secret"', edit.text)
        response = self.client.post("/admin/models/1/edit", data={
            "name": "changed", "base_url": "http://fake.invalid", "api_key": "",
            "model_id": "fake", "temperature": "0.7", "reasoning_effort": "",
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        model = self.sql("SELECT * FROM models")[0]
        self.assertEqual((model["temperature"], model["reasoning_effort"], model["api_key"]), (0.7, None, "secret"))
        for value in ("nan", "inf", "-1", "2.1", "abc"):
            self.assertEqual(self.create(temperature=value).status_code, 422)
        self.assertEqual(self.create(reasoning_effort="invalid").status_code, 422)
        self.assertEqual(len(self.sql("SELECT * FROM models")), 1)

    def test_submitted_settings_survive_later_model_edits(self):
        self.create(temperature="1", reasoning_effort="high")
        prepared = [(1, {}, run_submission.make_run_options(mode="quality"))]
        with patch.object(run_queue, "dispatch_next"):
            asyncio.run(run_submission.submit_runs("queued", 1, None, prepared))
        self.sql("UPDATE models SET temperature=0, reasoning_effort='low' WHERE id=1")
        with patch.object(run_queue, "_active_run_id", None), patch.object(run_queue, "_spawn_benchmark") as spawn:
            self.assertTrue(run_queue._try_start(1))
        config = benchmark_runner._build_client_config(spawn.call_args.args[1])
        self.assertEqual((config.temperature, config.reasoning_effort), (1, "high"))

    def test_corrupt_saved_settings_fail_without_pinning_queue(self):
        self.create()
        with patch.object(run_queue, "dispatch_next"):
            asyncio.run(run_submission.submit_runs("queued", 1, None, [
                (1, {}, run_submission.make_run_options(mode="quality")),
            ]))
        self.sql("UPDATE test_runs SET decoding_config_json='[]' WHERE id=1")
        with patch.object(run_queue, "_active_run_id", None), patch.object(run_queue, "_spawn_benchmark") as spawn:
            self.assertFalse(run_queue._try_start(1))
            self.assertIsNone(run_queue.active_run_id())
            spawn.assert_not_called()
        self.assertEqual(self.sql("SELECT status FROM test_runs")[0]["status"], "failed")

    def test_missing_fields_on_edit_keep_settings_and_new_form_defaults_are_independent(self):
        self.create(temperature="1", reasoning_effort="high")
        page = self.client.get("/admin/models")
        self.assertIn('value="0" class="input-field"', page.text)
        response = self.client.post("/admin/models/1/edit", data={
            "name": "changed", "base_url": "http://fake.invalid", "model_id": "fake",
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        model = self.sql("SELECT * FROM models")[0]
        self.assertEqual((model["temperature"], model["reasoning_effort"]), (1, "high"))

    def test_performance_warmup_and_measurements_keep_selected_decoding(self):
        PerfClient.instances = []
        config = replace(CONFIG, temperature=1, reasoning_effort="high")
        with patch.object(perf, "ChatClient", PerfClient):
            report = perf.run_perf_suite(config, perf.PerfConfig(2))
        self.assertEqual((report.protocol["temperature"], report.protocol["reasoning_effort"]), (1, "high"))
        self.assertTrue(all(c.config.temperature == 1 and c.config.reasoning_effort == "high" for c in PerfClient.instances))

    def test_idempotent_migrations_preserve_existing_models_and_runs(self):
        self.create()
        self.sql("ALTER TABLE models DROP COLUMN temperature")
        self.sql("ALTER TABLE models DROP COLUMN reasoning_effort")
        self.sql("ALTER TABLE test_runs DROP COLUMN decoding_config_json")

        async def migrate():
            db = await get_db()
            try:
                await _apply_migrations(db)
                await _apply_migrations(db)
                await db.commit()
            finally:
                await db.close()

        asyncio.run(migrate())
        model = self.sql("SELECT * FROM models")[0]
        self.assertEqual((model["temperature"], model["reasoning_effort"]), (0, None))

    def test_wire_settings_and_no_regeneration_after_transport_failure(self):
        config = replace(CONFIG, temperature=1, reasoning_effort="high", max_retries=3)
        client = ChatClient(config)
        self.addCleanup(client.session.close)
        response = Response([event("answer", finish="stop"), b"[DONE]"])
        with patch.object(client.session, "post", return_value=response) as post:
            client.complete("q")
        payload = post.call_args.kwargs["json"]
        self.assertEqual((payload["temperature"], payload["reasoning_effort"]), (1, "high"))
        self.assertEqual(client_protocol(config)["reasoning_effort"], "high")
        default_client = ChatClient(CONFIG)
        self.addCleanup(default_client.session.close)
        self.assertNotIn("reasoning_effort", default_client._payload([], 8, False))
        with patch.object(client.session, "post", side_effect=ConnectionError("broken")) as post:
            _, _, metrics = client.complete("q")
        self.assertFalse(metrics.ok)
        self.assertEqual(post.call_count, 1)
        # Explicit rejection of a compatibility field can still negotiate
        # before generation. It does not silently remove reasoning effort.
        rejected = Response(status=400, text='{"error":"unsupported stream_options"}')
        response = Response([event("answer", finish="stop"), b"[DONE]"])
        with patch.object(client.session, "post", side_effect=[rejected, response]) as post:
            _, _, metrics = client.complete("q")
        self.assertTrue(metrics.ok)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "high")
        rejected = Response(status=400, text='{"error":"unsupported reasoning_effort"}')
        with patch.object(client.session, "post", return_value=rejected) as post:
            _, _, metrics = client.complete("q")
        self.assertFalse(metrics.ok)
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
