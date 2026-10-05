"""Sweep protocol, controlled clients, incremental storage and report regressions."""

import asyncio
from contextlib import closing
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import config as app_config
from app.benchmarking import perf_sweep as sweep
from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.models import RequestMetrics, TokenUsage
from app.routes import runs
from app.services import benchmark_runner as runner, html_reports, run_submission
from selftest_reports import AssetParser, fixture_database

CONFIG = ClientConfig("http://controlled.invalid/v1", "unused", "fixture", timeout=1)


class FakeClient:
    instances = []
    handler = None
    active = peak = 0
    lock = threading.Lock()

    def __init__(self, config, omitted=()):
        self.config = config
        self.omitted = set(omitted)
        self.session = Mock()
        self.streaming_supported = True
        self._stream_usage_supported = True
        self.calls = []
        self.instances.append(self)

    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        with self.lock:
            FakeClient.active += 1
            FakeClient.peak = max(FakeClient.peak, FakeClient.active)
        try:
            time.sleep(0.001)
            if FakeClient.handler:
                return FakeClient.handler(self, prompt, kwargs)
            metrics = RequestMetrics(
                latency_ms=100,
                ttft_ms=25,
                completion_tokens=20,
                prompt_tokens=256,
                streamed=True,
                stream_chunks=3,
                stream_span_ms=50,
                finish_reason="stop",
            )
            return "1,2,3", TokenUsage(256, 20), metrics
        finally:
            with self.lock:
                FakeClient.active -= 1


class SweepProtocolTests(unittest.TestCase):
    def setUp(self):
        FakeClient.instances = []
        FakeClient.handler = None
        FakeClient.active = FakeClient.peak = 0

    def test_exact_requested_grid_and_budget(self):
        config = sweep.SweepConfig()
        self.assertEqual(config.contexts, (256, 32768, *range(65536, 1048577, 65536)))
        self.assertEqual(config.concurrencies, (1, *range(16, 257, 16)))
        self.assertEqual(len(config.contexts) * len(config.concurrencies), 306)
        self.assertEqual(config.requests_per_effort, 78390)
        self.assertEqual(len(config.contexts) * len(config.concurrencies) * len(sweep.EFFORTS) * 2, 4896)
        for maximum in (257, 32767, 32768, 65535, 1048575):
            bounded = sweep.SweepConfig(maximum, 255)
            self.assertEqual(bounded.contexts[-1], maximum)
            self.assertEqual(bounded.concurrencies[-1], 255)
            self.assertLessEqual(len(bounded.contexts) * len(bounded.concurrencies) * 16, 5000)
        self.assertEqual(sweep.SweepConfig(256, 3).concurrencies, (1, 3))
        for kwargs in (
            {"context_max": 255},
            {"context_max": 1048577},
            {"max_concurrency": 257},
            {"sweep_rounds": 0},
            {"sweep_rounds": True},
            {"sweep_output_tokens": 65537},
        ):
            with self.assertRaises(ValueError):
                sweep.SweepConfig(**kwargs)

    def test_long_prompts_have_exact_reference_lengths_and_unique_prefixes(self):
        for length in (256, 16384, 1048576):
            bank = sweep.PromptBank(length)
            for index in (1, 255, 1025):
                prompt = bank.prompt(index)
                self.assertEqual(len(bank.encoding.encode(prompt)), length)
                self.assertTrue(prompt.endswith("with no prose."))
            self.assertNotEqual(bank.prompt(0), bank.prompt(1))

    def test_rates_truncation_and_missing_first_token(self):
        def mixed(client, prompt, kwargs):
            index = int(prompt.split(". Background")[0].rsplit("-", 1)[1])
            metrics = RequestMetrics(
                latency_ms=100,
                ttft_ms=25,
                completion_tokens=20,
                prompt_tokens=250,
                streamed=True,
                stream_chunks=3,
                stream_span_ms=50,
                finish_reason="length" if index % 2 else "stop",
            )
            return "1,2,3", TokenUsage(), metrics

        FakeClient.handler = mixed
        clients = [FakeClient(CONFIG) for _ in range(8)]
        cell = sweep.measure_cell(
            clients, 8, sweep.PromptBank(256), sweep.SweepConfig(256, 8, 2), lambda: False
        )
        self.assertEqual((cell["requests"], cell["completed"], cell["incomplete"]), (16, 8, 8))
        self.assertEqual(cell["latency"]["count"], 8)
        self.assertEqual(cell["ttft"]["count"], 16)
        self.assertEqual(cell["request_tokens_per_sec"]["p50"], 200)
        self.assertEqual(cell["generation_tokens_per_sec"]["p50"], 380)
        self.assertNotIn("p50_ms", cell["request_tokens_per_sec"])
        self.assertLessEqual(FakeClient.peak, 8)
        self.assertGreater(FakeClient.peak, 1)
        FakeClient.handler = lambda *_: (
            "1",
            TokenUsage(),
            RequestMetrics(latency_ms=10, finish_reason="stop"),
        )
        cell = sweep.measure_cell(
            clients, 1, sweep.PromptBank(256), sweep.SweepConfig(256, 1), lambda: False
        )
        self.assertIsNone(cell["ttft"]["p50_ms"])
        self.assertIsNone(cell["generation_tokens_per_sec"]["p50"])

    def test_capability_probes_context_rejection_and_single_attempt_cells(self):
        def endpoint(client, prompt, kwargs):
            if client.config.reasoning_effort not in {None, "low"}:
                return (
                    "",
                    TokenUsage(),
                    RequestMetrics(
                        ok=False,
                        error='HTTP 400: {"error":{"param":"reasoning_effort","code":"unsupported_value"}}',
                    ),
                )
            length = len(sweep.PromptBank(256).encoding.encode(prompt))
            if length >= 16384:
                return (
                    "",
                    TokenUsage(),
                    RequestMetrics(
                        ok=False, error="HTTP 400: context_length_exceeded", latency_ms=5
                    ),
                )
            return (
                "1",
                TokenUsage(),
                RequestMetrics(
                    latency_ms=10, ttft_ms=2, completion_tokens=20, finish_reason="stop"
                ),
            )

        FakeClient.handler = endpoint
        snapshots, cells = [], []
        with patch.object(sweep, "SweepClient", FakeClient):
            report = sweep.run_sweep(
                CONFIG,
                sweep.SweepConfig(65536, 8),
                on_cell=cells.append,
                checkpoint=lambda data: snapshots.append(json.loads(json.dumps(data))),
                retain_cells=False,
            )
        self.assertEqual(
            [e["effort"] for e in report.efforts if e["status"] == "accepted"], ["default", "low"]
        )
        self.assertEqual(len(cells), 12)
        self.assertEqual(report.resolved_cells, 48)
        self.assertEqual(report.cells, [])
        self.assertTrue(report.finished)
        for effort in ("default", "low"):
            group = [c for c in cells if c["effort"] == effort]
            self.assertEqual(
                [c["status"] for c in group],
                [
                    "measured",
                    "measured",
                    "unsupported_context",
                    "skipped_context",
                    "skipped_context",
                    "skipped_context",
                ],
            )
        attempts = [k["retries"] for c in FakeClient.instances for _, k in c.calls]
        self.assertEqual(attempts.count(3), 8)  # one capability probe per candidate
        self.assertEqual(
            attempts.count(1), sum(c["requests"] + c.get("cache_reuse", {}).get("requests", 0) for c in cells) + 16
        )  # two sets of seven worker warmups plus two cache primes
        self.assertTrue(
            all(k["retries"] == 1 for c in FakeClient.instances for _, k in c.calls[1:])
        )
        self.assertTrue(all(c.session.close.call_count == 1 for c in FakeClient.instances))
        self.assertGreater(len(snapshots), 8)

    def test_timeout_does_not_become_a_context_limit_and_stop_closes_sessions(self):
        def endpoint(client, prompt, kwargs):
            if kwargs["retries"] == 1 and len(prompt) < 2000 and "-100" not in prompt:
                return (
                    "",
                    TokenUsage(),
                    RequestMetrics(ok=False, error="HTTP 504: upstream timeout", latency_ms=5),
                )
            return (
                "1",
                TokenUsage(),
                RequestMetrics(latency_ms=10, completion_tokens=20, finish_reason="stop"),
            )

        FakeClient.handler = endpoint
        with (
            patch.object(sweep, "SweepClient", FakeClient),
            patch.object(sweep, "EFFORTS", ("default",)),
        ):
            report = sweep.run_sweep(CONFIG, sweep.SweepConfig(16384, 8))
        self.assertEqual(
            [c["status"] for c in report.cells],
            ["failed", "skipped_failure", "measured", "measured"],
        )
        self.assertFalse(sweep.context_rejection("HTTP 413: upload too large"))
        self.assertFalse(sweep.context_rejection("timeout during context prefill"))
        FakeClient.handler = None
        stopped = False

        def saved(cell):
            nonlocal stopped
            stopped = True

        with patch.object(sweep, "SweepClient", FakeClient):
            report = sweep.run_sweep(
                CONFIG, sweep.SweepConfig(16384, 8), on_cell=saved, cancelled=lambda: stopped
            )
        self.assertTrue(report.cancelled)
        self.assertFalse(report.finished)
        self.assertEqual(len(report.cells), 1)
        self.assertTrue(all(c.session.close.call_count == 1 for c in FakeClient.instances))

    def failure_sweep(self, *, bad_contexts=(), bad_loads=(), outage=False):
        original_measure = sweep.measure_cell

        def measure(clients, concurrency, bank, config, cancelled, *cache_args):
            if bank.context_tokens in bad_contexts or concurrency in bad_loads:
                return {"context_tokens": bank.context_tokens, "concurrency": concurrency,
                        "status": "failed", "requests": concurrency, "completed": 0,
                        "errors": concurrency, "incomplete": 0, "error": "HTTP 503: overloaded"}
            return original_measure(clients, concurrency, bank, config, cancelled, *cache_args)

        if outage:
            def handler(client, prompt, kwargs):
                if "-900" in prompt:
                    return "", TokenUsage(), RequestMetrics(ok=False, error="HTTP 500: Cannot connect to host")
                return "1", TokenUsage(), RequestMetrics(latency_ms=10, finish_reason="stop")
            FakeClient.handler = handler
        with (patch.object(sweep, "SweepClient", FakeClient),
              patch.object(sweep, "EFFORTS", ("default",)),
              patch.object(sweep, "measure_cell", measure)):
            return sweep.run_sweep(CONFIG, sweep.SweepConfig(262144, 80))

    def test_repeated_context_failures_require_healthy_control_before_ceiling(self):
        report = self.failure_sweep(bad_contexts={32768, 65536, 131072})
        self.assertTrue(report.finished)
        self.assertEqual(report.efforts[0]["limits"]["context"]["at"], 32768)
        first_workers = [c for c in report.cells if c["concurrency"] == 1]
        self.assertEqual([c["status"] for c in first_workers],
                         ["measured", "failed", "failed", "failed", "skipped_context", "skipped_context"])
        self.assertTrue(first_workers[3]["limit_control"]["ok"])
        self.assertEqual(first_workers[4]["requests"], 0)

    def test_context_success_resets_streak_and_single_failure_does_not_set_ceiling(self):
        report = self.failure_sweep(bad_contexts={32768, 65536, 196608, 262144})
        self.assertEqual(report.efforts[0]["limits"], {})
        cell = next(c for c in report.cells if c["context_tokens"] == 131072 and c["concurrency"] == 1)
        self.assertEqual(cell["status"], "measured")

    def test_three_failed_loads_skip_larger_loads_and_preserve_single_worker_contexts(self):
        report = self.failure_sweep(bad_loads={16, 32, 48})
        self.assertEqual(report.efforts[0]["limits"]["concurrency"]["at"], 16)
        first_row = [c for c in report.cells if c["context_tokens"] == 256]
        self.assertEqual([c["status"] for c in first_row],
                         ["measured", "failed", "failed", "failed", "skipped_failure", "skipped_failure"])
        later = [c for c in report.cells if c["context_tokens"] == 32768]
        self.assertEqual(later[0]["status"], "measured")
        self.assertTrue(all(c["requests"] == 0 for c in later[1:]))

    def test_load_success_resets_streak(self):
        report = self.failure_sweep(bad_loads={16, 32, 64, 80})
        self.assertNotIn("concurrency", report.efforts[0]["limits"])
        self.assertTrue(all(c["status"] == "measured" for c in report.cells if c["concurrency"] == 48))

    def test_failed_control_stops_outage_without_inventing_context_limit(self):
        report = self.failure_sweep(bad_contexts={32768, 65536, 131072}, outage=True)
        self.assertFalse(report.finished)
        self.assertFalse(report.cancelled)
        self.assertIn("Cannot connect to host", report.stop_error)
        self.assertEqual(report.efforts[0]["limits"], {})
        self.assertFalse(any(c["context_tokens"] > 131072 for c in report.cells))
        self.assertTrue(all(c.session.close.call_count == 1 for c in FakeClient.instances))

    def test_only_explicit_sampling_rejections_remove_fields(self):
        def endpoint(client, prompt, kwargs):
            if "temperature" not in client.omitted:
                return (
                    "",
                    TokenUsage(),
                    RequestMetrics(
                        ok=False,
                        error='HTTP 400: {"error":{"param":"temperature","code":"unsupported_parameter"}}',
                    ),
                )
            return "1", TokenUsage(), RequestMetrics(latency_ms=10, finish_reason="stop")

        FakeClient.handler = endpoint
        with (
            patch.object(sweep, "SweepClient", FakeClient),
            patch.object(sweep, "EFFORTS", ("high",)),
        ):
            report = sweep.run_sweep(CONFIG, sweep.SweepConfig(256, 1))
        self.assertEqual(report.efforts[0]["omitted_fields"], ["temperature"])
        self.assertEqual(FakeClient.instances[0].config.reasoning_effort, "high")
        client = sweep.SweepClient(
            ClientConfig("", "", "fixture", reasoning_effort="max"), {"temperature", "seed"}
        )
        payload = client._payload([{"role": "user", "content": "1"}], 256, True)
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertNotIn("temperature", payload)
        client.session.close()
        self.assertFalse(
            sweep.rejects("HTTP 500: upstream outage; reasoning_effort=high", "reasoning_effort")
        )

    def test_load_dependent_context_rejection_still_tests_longer_contexts_at_one_worker(self):
        measure = sweep.measure_cell

        def capacity(clients, concurrency, bank, config, cancelled, *cache_args):
            if concurrency == 8:
                return {
                    "context_tokens": bank.context_tokens,
                    "concurrency": 8,
                    "status": "unsupported_context",
                    "requests": 8,
                    "completed": 0,
                    "errors": 8,
                    "incomplete": 0,
                    "error": "context_length_exceeded",
                }
            return measure(clients, concurrency, bank, config, cancelled, *cache_args)

        with (
            patch.object(sweep, "SweepClient", FakeClient),
            patch.object(sweep, "EFFORTS", ("default",)),
            patch.object(sweep, "measure_cell", capacity),
        ):
            report = sweep.run_sweep(CONFIG, sweep.SweepConfig(16384, 8))
        self.assertEqual(
            [c["status"] for c in report.cells],
            ["measured", "unsupported_context", "measured", "unsupported_context"],
        )


class SweepHTTPTests(unittest.TestCase):
    def test_actual_client_handles_all_previously_failed_input_lengths(self):
        lengths = (32768, 49152, 65536, 81920, 98304, 114688, 131072)
        observed = []
        encoding = sweep.PromptBank(256).encoding

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                count = len(encoding.encode(payload["messages"][0]["content"]))
                observed.append(count)
                events = [
                    {"choices": [{"index": 0, "delta": {"content": "1,2,3"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": count, "completion_tokens": 5}},
                    {"choices": [{"index": 0, "delta": {}}], "usage": {"completion_tokens": 5}},
                ]
                body = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = sweep.SweepClient(ClientConfig(f"http://127.0.0.1:{server.server_port}", "unused", "fixture", timeout=2))
        try:
            for length in lengths:
                text, usage, metrics = client.complete(sweep.PromptBank(length).prompt(0), retries=1)
                self.assertTrue(metrics.ok, (length, metrics.error))
                self.assertEqual(text, "1,2,3")
                self.assertEqual(usage.prompt_tokens, length)
            self.assertEqual(observed, list(lengths))
        finally:
            client.session.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_actual_http_negotiates_fields_without_dropping_requested_effort(self):
        payloads = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payloads.append(payload)
                rejected = next(
                    (key for key in ("max_tokens", "temperature", "seed") if key in payload), None
                )
                if not rejected and payload.get("stream"):
                    rejected = "stream"
                if rejected:
                    response = {"error": {"param": rejected, "code": "unsupported_parameter"}}
                    status = 400
                else:
                    response = {
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "1,2,3"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 256, "completion_tokens": 5},
                    }
                    status = 200
                body = json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = ClientConfig(
                f"http://127.0.0.1:{server.server_port}/v1", "unused", "fixture", timeout=1
            )
            with patch.object(sweep, "EFFORTS", ("high",)):
                report = sweep.run_sweep(config, sweep.SweepConfig(256, 1, 1, 256))
            self.assertEqual(report.efforts[0]["status"], "accepted")
            self.assertEqual(report.efforts[0]["output_token_parameter"], "max_completion_tokens")
            self.assertFalse(report.efforts[0]["streaming"])
            self.assertEqual(report.cells[0]["completed"], 1)
            self.assertIsNone(report.cells[0]["ttft"]["p50_ms"])
            self.assertTrue(all(p["reasoning_effort"] == "high" for p in payloads))
            self.assertEqual(payloads[-1]["max_completion_tokens"], 256)
            self.assertNotIn("temperature", payloads[-1])
            self.assertNotIn("seed", payloads[-1])
            self.assertEqual(len(payloads), 8)  # probes, cold call, cache prime, warm call
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class SweepStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "sweep.db"
        fixture_database(self.path)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO users(id,username,role) VALUES(1,'fixture','admin')")
            db.execute("DELETE FROM test_results WHERE run_id=1")
            db.execute(
                "UPDATE test_runs SET status='pending',perf_json=NULL,quality_json=NULL,total_questions=0,scored_questions=0,error_message=NULL WHERE id=1"
            )
        self.db_patch = patch.object(app_config, "DATABASE_PATH", self.path)
        self.db_patch.start()
        FakeClient.instances = []
        FakeClient.handler = None
        FakeClient.active = FakeClient.peak = 0

    def tearDown(self):
        self.db_patch.stop()
        self.directory.cleanup()

    def test_runner_persists_hydrated_json_offline_and_online_views(self):
        with (
            patch.object(sweep, "SweepClient", FakeClient),
            patch.object(sweep, "EFFORTS", ("default", "low")),
            patch("app.services.url_guard.validate_endpoint"),
        ):
            runner._run_benchmark_impl(
                1,
                {"base_url": CONFIG.base_url, "api_key": "unused", "model_id": "fixture"},
                "sweep",
                8,
                context_max=16384,
                sweep_rounds=1,
                sweep_output_tokens=256,
            )
        run = asyncio.run(html_reports.load_run(1))
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["total_questions"], 0)
        self.assertEqual(len(run["perf"]["cells"]), 8)
        view = html_reports.performance_view([run])["sweep_views"][0]
        self.assertEqual(view["measured"], 8)
        self.assertEqual(html_reports.performance_view([run])["notes"], [])
        self.assertEqual(view["data"]["cells"][0]["values"]["tokens_per_sec"], 200)
        text = html_reports.render_report([run])
        parser = AssetParser()
        parser.feed(text)
        self.assertEqual(parser.scripts, 0)
        self.assertEqual(parser.external, [])
        self.assertIn("All exact measurements", text)
        self.assertIn("200.0 tok/s", text)
        self.assertIn(">200.0 tok/s</text>", text)
        self.assertNotIn("Quality & coverage", text)
        self.assertNotIn("Category/family-balanced strict success", text)
        self.assertNotIn("Question results", text)
        app = FastAPI()
        app.include_router(runs.router)
        with (
            patch.object(runs, "get_current_user", AsyncMock(return_value={"id": 1})),
            TestClient(app) as client,
        ):
            response = client.get("/runs/1/performance.json")
            self.assertEqual(len(response.json()["cells"]), 8)
            self.assertEqual(response.headers["cache-control"], "no-store")
            response = client.get("/runs/1")
            self.assertEqual(response.status_code, 200)
            self.assertIn('aria-label="Measurement"', response.text)
            self.assertIn("sweepExplorer", response.text)
        # Rerunning a former sweep runs the standard test and keeps its context maximum.
        legacy = {"mode": "sweep", "max_concurrency": 256, "context_max": 1048576,
                  "sweep_rounds": 1, "sweep_output_tokens": 8192}
        options = run_submission.spec_from_run({"model_id": 1, "run_options_json": json.dumps(legacy)})
        self.assertEqual((options["mode"], options["performance"], options["context_max"], options["max_concurrency"]),
                         ("performance", "standard", 1048576, 8))
        for kwargs in ({"context_max": 1048577}, {"context_max": 255}, {"context_max": True}):
            with self.assertRaises(HTTPException):
                run_submission.make_run_options(mode="performance", **kwargs)

    def test_stop_guards_checkpoints_cells_and_retains_saved_results(self):
        report = sweep.SweepReport("fixture", {"contexts": [256], "concurrencies": [1]}).to_dict()
        secret_error = "https://user:password@private.invalid/path?api_key=DO_NOT_EXPORT_API_KEY"
        report["stop_error"] = secret_error
        report["notes"] = [secret_error]
        report["efforts"] = [{"effort": "default", "limits": {"context": {"reason": secret_error}}}]
        cell = {
            "effort": "default",
            "context_tokens": 256,
            "concurrency": 1,
            "status": "failed",
            "requests": 1,
            "completed": 0,
            "error": "https://user:password@private.invalid/path?api_key=DO_NOT_EXPORT_API_KEY",
            "limit_control": {"ok": False, "error": secret_error},
        }
        runner._store_sweep_checkpoint(1, report)
        runner._store_sweep_cell(1, cell)
        app = FastAPI()
        app.include_router(runs.router)
        with (
            patch.object(runs, "require_admin", AsyncMock(return_value={"id": 1, "role": "admin"})),
            TestClient(app) as client,
        ):
            response = client.post("/runs/1/stop", follow_redirects=False)
            self.assertEqual(response.status_code, 302)
        runner._store_sweep_cell(1, {**cell, "completed": 99})
        runner._store_sweep_checkpoint(1, {**report, "finished": True})
        run = asyncio.run(html_reports.load_run(1))
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["perf"]["cells"][0]["completed"], 0)
        self.assertFalse(run["perf"]["finished"])
        self.assertTrue(run["perf"]["cancelled"])
        self.assertNotIn("DO_NOT_EXPORT_API_KEY", json.dumps(run["perf"]))
        self.assertNotIn("password", json.dumps(run["perf"]))
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("DELETE FROM test_runs WHERE id=1")
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM performance_cells WHERE run_id=1").fetchone()[0], 0
            )

    def test_endpoint_stop_is_failed_even_after_successful_measurements(self):
        report = sweep.SweepReport("fixture", {"contexts": [256], "concurrencies": [1]},
                                   successful_requests=2, stop_error="Control failed: Cannot connect to host")
        cell = {"effort": "default", "context_tokens": 256, "concurrency": 1,
                "status": "measured", "requests": 1, "completed": 1}
        runner._store_sweep_cell(1, cell)
        with patch.object(sweep, "run_sweep", return_value=report):
            runner._run_context_sweep(1, CONFIG, sweep.SweepConfig(256, 1))
        run = asyncio.run(html_reports.load_run(1))
        self.assertEqual(run["status"], "failed")
        self.assertIn("Cannot connect to host", run["error_message"])
        self.assertEqual(run["perf"]["cells"][0]["completed"], 1)
        self.assertFalse(run["perf"]["finished"])

    def test_submission_and_queue_preserve_full_sweep_options(self):
        from app.services import run_queue

        options = run_submission.make_run_options(mode="sweep", max_concurrency=256, sweep_rounds=2)
        prepared = asyncio.run(
            run_submission.validated_specs(json.dumps([{"model_id": 1, **options}]))
        )
        with patch.object(run_queue, "dispatch_next"):
            _, ids = asyncio.run(run_submission.submit_runs("sweep", 1, None, prepared))
        with (
            patch.object(run_queue, "_active_run_id", None),
            patch.object(run_queue, "_spawn_benchmark") as spawn,
        ):
            self.assertTrue(run_queue._try_start(ids[0]))
            self.assertEqual(spawn.call_args.args[2], options)
        with self.assertRaises(HTTPException):
            asyncio.run(
                run_submission.validated_specs(
                    json.dumps([{"model_id": 1, "mode": "quality", "context_max": 16384}])
                )
            )


if __name__ == "__main__":
    unittest.main()
