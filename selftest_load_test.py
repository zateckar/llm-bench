#!/usr/bin/env python3
"""Offline tests for open-loop load tests: workloads, schedules, dispatch, SLOs, reports."""

from contextlib import ExitStack, closing
import json
from pathlib import Path
import sqlite3
import statistics
import tempfile
import threading
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.benchmarking import load_test, load_workload
from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.load_test import Record, analyse_step, run_load_test, summarise
from app.benchmarking.load_workload import Arrival, PromptBuilder, parse_settings, schedule
from app.benchmarking.models import RequestMetrics, TokenUsage
from app.services.capacity import CapacityAssumptions, estimate_capacity
from app.storage import DETECT_TYPES

CONFIG = ClientConfig("http://fake.invalid", "k", "fake")


def workload(ttft=400, tpot=None, e2e=None, classes=None):
    return {"classes": classes or [{"name": "t", "weight": 1, "input": [[200, 300, 1]], "output": [[5, 10, 1]],
                                    "shared_prefix": 0, "slo": {"ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e}}]}


def settings(rates, step=2, **kw):
    slo = {k: kw.pop(k) for k in ("ttft", "tpot", "e2e") if k in kw}
    return parse_settings({"preset": "custom", "workload": kw.pop("workload", None) or workload(**slo),
                           "rates": rates, "step_seconds": step, **kw})


class FakeServer:
    """A server with a fixed number of slots: queueing appears as first-token delay."""

    def __init__(self, slots=8, service=0.02, error=False, stream=True, warmup_ok=True):
        self.slots = threading.Semaphore(slots)
        self.service, self.error, self.stream, self.warmup_ok = service, error, stream, warmup_ok
        self.lock = threading.Lock()
        self.sent, self.systems, self.prompts = [], set(), []

    def __call__(self, config):
        return FakeClient(self, config)


class FakeClient:
    def __init__(self, server, config):
        self.server, self.config = server, config
        self.streaming_supported = server.stream
        self._stream_usage_supported = True
        self.session = Mock()

    def complete(self, prompt, system_prompt=None, *, max_tokens=None, retries=None, stream=None):
        s = self.server
        warmup = prompt.startswith("Warmup")
        with s.lock:
            if not warmup:
                s.sent.append(time.perf_counter())
                s.systems.add(system_prompt)
                s.prompts.append(prompt[:60])
        if (s.error and not warmup) or (warmup and not s.warmup_ok):
            return "[API ERROR]", TokenUsage(), RequestMetrics(ok=False, error="HTTP 500 boom at http://secret.host/x", latency_ms=1)
        start = time.perf_counter()
        with s.slots:
            time.sleep(s.service)
            ttft = (time.perf_counter() - start) * 1000
            time.sleep(0.005)
        latency = (time.perf_counter() - start) * 1000
        return "1, 2, 3", TokenUsage(250, max_tokens), RequestMetrics(
            latency_ms=latency, ttft_ms=ttft, completion_tokens=max_tokens, prompt_tokens=250, streamed=True,
            stream_chunks=max_tokens, stream_span_ms=max_tokens * 3.0, finish_reason="length",
            cached_tokens=100, cached_tokens_reported=True)


class WorkloadTests(unittest.TestCase):
    def test_presets_and_custom_detection(self):
        for preset in ("chat", "agent", "rag", "mixed"):
            parsed = parse_settings({"preset": preset, "rates": [1]})
            self.assertEqual(parsed.preset, preset)
        mixed = parse_settings({"preset": "mixed", "rates": [1]})
        self.assertEqual([c.name for c in mixed.workload.classes], ["chat", "agent"])
        self.assertAlmostEqual(mixed.workload.shares()[0], 0.8)
        edited = mixed.workload.model_dump()
        edited["classes"][0]["weight"] = 70
        self.assertEqual(parse_settings({"preset": "mixed", "workload": edited, "rates": [1]}).preset, "custom")
        self.assertNotEqual(parse_settings({"preset": "custom", "workload": edited, "rates": [1]}).workload.fingerprint(),
                            mixed.workload.fingerprint())

    def test_rejections(self):
        base = workload()["classes"][0]
        cases = {
            "shared prefix plus 128": {"workload": {"classes": [{**base, "shared_prefix": 150}]}},
            "at least 32": {"workload": {"classes": [{**base, "shared_prefix": 10}]}},
            "unique": {"workload": {"classes": [base, base]}},
            "[min, max, weight]": {"workload": {"classes": [{**base, "input": [[200, 300]]}]}},
            "min <= max": {"workload": {"classes": [{**base, "output": [[10, 5, 1]]}]}},
            "lowercase": {"workload": {"classes": [{**base, "name": "Chat"}]}},
            "strictly increasing": {"rates": [2, 1]},
            "between 0.01": {"rates": [0]},
            "Step duration": {"step_seconds": 5},
            "500,000": {"rates": [400, 500], "step_seconds": 1000},
            "12 hours": {"rates": [0.1, 0.2, 0.3, 0.4], "step_seconds": 14_400},
            "Extra inputs": {"colour": "red"},
            "Unknown workload preset": {"preset": "nightly"},
        }
        for needle, change in cases.items():
            with self.subTest(needle):
                raw = {"preset": "custom", "workload": workload(), "rates": [1], **change}
                with self.assertRaises(ValueError) as caught:
                    parse_settings(raw)
                self.assertIn(needle, load_workload.error_text(caught.exception))

    def test_schedules_are_reproducible_and_shaped(self):
        mixed = parse_settings({"preset": "mixed", "rates": [5], "step_seconds": 3600})
        first = schedule(mixed, 0)
        self.assertEqual(first, schedule(mixed, 0))
        self.assertNotEqual(first, schedule(mixed.model_copy(update={"seed": 1}), 0))
        gaps = [b.offset - a.offset for a, b in zip(first, first[1:])]
        self.assertAlmostEqual(statistics.mean(gaps), 0.2, delta=0.01)
        self.assertAlmostEqual(statistics.stdev(gaps) / statistics.mean(gaps), 1.0, delta=0.05)
        self.assertAlmostEqual(sum(a.klass == 0 for a in first) / len(first), 0.8, delta=0.02)
        for a in first:
            cls = mixed.workload.classes[a.klass]
            self.assertTrue(any(lo <= a.input_tokens <= hi for lo, hi, _ in cls.input))
            self.assertTrue(any(lo <= a.output_tokens <= hi for lo, hi, _ in cls.output))
        bursty = mixed.model_copy(update={"arrival": "gamma", "burstiness": 2.0})
        gaps = [b.offset - a.offset for a, b in zip(*(lambda s: (s, s[1:]))(schedule(bursty, 0)))]
        self.assertAlmostEqual(statistics.mean(gaps), 0.2, delta=0.015)
        self.assertAlmostEqual(statistics.stdev(gaps) / statistics.mean(gaps), 2.0, delta=0.15)

    def test_prompts_have_exact_reference_lengths_and_shared_prefixes(self):
        mixed = parse_settings({"preset": "mixed", "rates": [1]})
        builder = PromptBuilder(mixed.workload, "n0nce")
        seen = set()
        for klass, tokens in ((0, 428), (0, 7999), (1, 6128), (1, 40001)):
            system, user = builder.build(Arrival(0, klass, tokens, 10), f"0-{tokens}")
            self.assertEqual(builder.count(system) + builder.count(user), tokens)
            self.assertEqual(builder.count(system), mixed.workload.classes[klass].shared_prefix)
            self.assertIn("Count upward", user)
            seen.add(system)
        self.assertEqual(len(seen), 2)
        plain = PromptBuilder(settings([1], step=10).workload, "x")
        system, user = plain.build(Arrival(0, 0, 256, 5), "1")
        self.assertIsNone(system)
        self.assertEqual(plain.count(user), 256)


def record(offset, metrics=None, lag=1.0, klass=0, output=10):
    return Record(Arrival(offset, klass, 250, output), metrics, lag if metrics else None)


def ok(ttft=100.0, latency=200.0, tokens=10, span=None, **kw):
    return RequestMetrics(latency_ms=latency, ttft_ms=ttft, completion_tokens=tokens, prompt_tokens=250, streamed=True,
                          stream_chunks=tokens, stream_span_ms=tokens * 5.0 if span is None else span,
                          finish_reason="length", **kw)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(load_workload, "MIN_STEP_SECONDS", 1)
        patcher.start()
        self.addCleanup(patcher.stop)

    def analyse(self, s, records, index=0, **kw):
        return analyse_step(s, index, records, duration=kw.pop("duration", float(s.step_seconds)),
                            drain_seconds=1.0, started_at=0, ended_at=1, max_in_flight=3, **kw)

    def test_slo_checks_include_dispatch_lag(self):
        slo = settings([1], ttft=300, tpot=10, e2e=1000).workload.classes[0].slo
        self.assertEqual(record(0, ok(ttft=250), lag=10).misses(slo), [])
        self.assertEqual(record(0, ok(ttft=250), lag=60).misses(slo), ["ttft"])
        self.assertEqual(record(0, ok(latency=950), lag=60).misses(slo), ["e2e"])
        self.assertEqual(record(0, ok(span=200)).misses(slo), ["tpot"])
        self.assertEqual(record(0, ok(span=5)).misses(slo), ["tpot_unmeasured"])  # burst delivery
        self.assertEqual(record(0, ok(ttft=None)).misses(slo), ["ttft_unmeasured"])
        self.assertEqual(record(0, None).misses(slo), ["dropped"])
        self.assertEqual(record(0, RequestMetrics(ok=False, error="x")).misses(slo), ["error"])
        self.assertEqual(record(0, ok(tokens=0)).misses(slo), ["incomplete"])
        filtered = ok()
        filtered.finish_reason = "content_filter"
        self.assertEqual(record(0, filtered).misses(slo), ["incomplete"])

    def test_attainment_goodput_and_class_rule(self):
        two = workload(classes=[
            {"name": "chat", "weight": 9, "input": [[200, 300, 1]], "output": [[5, 10, 1]], "shared_prefix": 0, "slo": {"ttft_ms": 300}},
            {"name": "agent", "weight": 1, "input": [[200, 300, 1]], "output": [[5, 10, 1]], "shared_prefix": 0, "slo": {"ttft_ms": 300}},
        ])
        s = parse_settings({"preset": "custom", "workload": two, "rates": [10], "step_seconds": 2, "attainment": 0.9})
        records = [record(i * 0.1, ok()) for i in range(19)] + [record(1.95, ok(ttft=900), klass=1)]
        step = self.analyse(s, records)
        self.assertAlmostEqual(step["attainment"], 0.95)
        self.assertAlmostEqual(step["goodput_rps"], 9.5)
        self.assertTrue(step["meets_target"])
        self.assertFalse(step["classes"]["agent"]["meets_target"])
        self.assertFalse(step["passed"])  # every class must reach the target
        self.assertTrue(step["conclusive_failure"])
        self.assertEqual(step["failure_reasons"], {"ttft": 1})
        self.assertEqual(step["mean_in_flight"], 20 * 0.2 / 3)

    def test_client_limits_make_failures_inconclusive(self):
        s = settings([10], ttft=300)
        dropped = [record(i * 0.1, ok()) for i in range(15)] + [record(1.6 + i * 0.1, None) for i in range(4)]
        step = self.analyse(s, dropped)
        self.assertTrue(step["client_limited"])
        self.assertFalse(step["passed"])
        self.assertFalse(step["conclusive_failure"])  # sent requests all met the SLO
        overloaded = [record(i * 0.1, ok(ttft=900)) for i in range(15)] + [record(1.6, None)]
        self.assertTrue(self.analyse(s, overloaded)["conclusive_failure"])
        lagging = [record(i * 0.1, ok(ttft=50), lag=400) for i in range(10)]
        step = self.analyse(s, lagging)
        self.assertTrue(step["client_limited"])
        self.assertFalse(step["conclusive_failure"])

    def test_sustainable_rate_status(self):
        def step(rate, passed, conclusive=True, cancelled=False):
            return {"offered_rate": rate, "passed": passed, "conclusive_failure": conclusive and not passed,
                    "cancelled": cancelled, "goodput_rps": rate}
        self.assertEqual(summarise([step(1, True), step(2, True), step(4, False)])["sustainable_status"], "established")
        self.assertEqual(summarise([step(1, True), step(2, True), step(4, False)])["first_failed_rate"], 4)
        lower = summarise([step(1, True), step(2, False, conclusive=False)])
        self.assertEqual((lower["sustainable_status"], lower["sustainable_rate"], lower["inconclusive_rates"]), ("lower_bound", 1, [2]))
        self.assertEqual(summarise([step(1, False), step(2, False)])["sustainable_status"], "not_met")
        self.assertEqual(summarise([step(1, True, cancelled=True)])["sustainable_status"], "not_measured")
        self.assertTrue(summarise([step(1, False), step(2, True), step(4, False)])["non_monotonic"])

    def test_windows_detect_drift_and_samples_are_thinned(self):
        s = settings([2], step=60, ttft=1000)
        records = [record(t / 2, ok(ttft=100 if t < 80 else 600)) for t in range(120)]
        step = self.analyse(s, records, sample_limit=25)
        self.assertEqual(len(step["windows"]), 6)
        self.assertEqual(sum(w["arrivals"] for w in step["windows"]), 120)
        self.assertTrue(step["drift"]["degrading"])
        self.assertGreater(step["drift"]["ttft_ratio"], 5)
        self.assertEqual((len(step["samples"]["t"]), step["samples_total"]), (25, 120))
        self.assertEqual(step["samples"]["t"], sorted(step["samples"]["t"]))
        steady = self.analyse(s, [record(t / 2, ok()) for t in range(120)])
        self.assertFalse(steady["drift"]["degrading"])
        # Doubling far below the SLO keeps the step stable; the ratio stays visible.
        small = self.analyse(s, [record(t / 2, ok(ttft=100 if t < 80 else 200)) for t in range(120)])
        self.assertGreater(small["drift"]["ttft_ratio"], 1.5)
        self.assertFalse(small["drift"]["degrading"])
        self.assertIsNone(self.analyse(settings([2], step=30), records[:60])["drift"])


class EngineTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(load_workload, "MIN_STEP_SECONDS", 1)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_test(self, server, s, cap=64, **kw):
        return run_load_test(CONFIG, s, cap, client_factory=server, **kw)

    def test_arrivals_are_not_delayed_by_a_slow_server(self):
        server = FakeServer(slots=1, service=0.25)
        s = settings([8], step=2, ttft=400)
        report = self.run_test(server, s).to_dict()
        step = report["steps"][0]
        expected = schedule(s, 0)
        self.assertEqual(step["arrivals"], len(expected))
        self.assertEqual(step["dropped"], 0)
        # Send times follow the schedule, not completions (closed loop would serialize them).
        gaps = [b - a for a, b in zip(server.sent, server.sent[1:])]
        self.assertLess(max(gaps), 1.0)
        self.assertLess(step["dispatch_lag"]["p99_ms"], 100)
        self.assertGreater(step["max_in_flight"], 3)
        self.assertGreater(step["ttft"]["p95_ms"], 1000)  # server queueing is visible
        self.assertFalse(step["passed"])
        self.assertTrue(step["conclusive_failure"])
        self.assertEqual(report["summary"]["sustainable_status"], "not_met")

    def test_cap_drops_arrivals_instead_of_delaying_them(self):
        server = FakeServer(slots=1, service=0.3)
        report = self.run_test(server, settings([10], step=2, ttft=5000), cap=2).to_dict()
        step = report["steps"][0]
        self.assertGreater(step["dropped"], 5)
        self.assertEqual(step["max_in_flight"], 2)
        self.assertEqual(step["dispatched"] + step["dropped"], step["arrivals"])
        self.assertTrue(step["client_limited"])
        self.assertFalse(step["passed"])
        self.assertIn("client-limited", " ".join(report["notes"]))

    def test_ladder_establishes_a_sustainable_rate(self):
        server = FakeServer(slots=2, service=0.1)
        checkpoints = []
        report = self.run_test(server, settings([2, 60], step=2, ttft=400), on_step=lambda r: checkpoints.append(len(r.steps)))
        data = report.to_dict()
        self.assertEqual(checkpoints, [1, 2])
        self.assertEqual([s["passed"] for s in data["steps"]], [True, False])
        self.assertEqual(data["summary"]["sustainable_rate"], 2)
        self.assertEqual(data["summary"]["sustainable_status"], "established")
        self.assertEqual(data["protocol"]["revision"], "open-loop-v1")
        self.assertEqual(data["protocol"]["workload_hash"], settings([1]).workload.fingerprint())
        first = data["steps"][0]
        self.assertAlmostEqual(first["cache_metrics"]["cache_hit_fraction"], 0.4)
        self.assertEqual(first["output_tokens_delivered"], sum(first["samples"]["out"]))
        self.assertEqual(len(server.systems), 1)
        json.dumps(data)

    def test_lower_bound_when_every_rate_passes(self):
        data = self.run_test(FakeServer(slots=8, service=0.01), settings([1, 4], step=1, ttft=500)).to_dict()
        self.assertEqual([s["passed"] for s in data["steps"]], [True, True])
        self.assertEqual((data["summary"]["sustainable_rate"], data["summary"]["sustainable_status"]), (4, "lower_bound"))

    def test_consecutive_failures_stop_the_ladder(self):
        data = self.run_test(FakeServer(), settings([2, 3, 4, 5], step=1, ttft=1)).to_dict()
        self.assertEqual(len(data["steps"]), 2)
        self.assertIn("2 consecutive steps", data["stop_reason"])
        self.assertEqual(data["summary"]["sustainable_status"], "not_met")

    def test_endpoint_failures(self):
        failing = self.run_test(FakeServer(error=True), settings([2, 4], step=1)).to_dict()
        self.assertEqual(len(failing["steps"]), 1)
        self.assertIn("No request completed", failing["stop_reason"])
        self.assertEqual(failing["steps"][0]["failure_reasons"]["error"], failing["steps"][0]["arrivals"])
        self.assertIn("HTTP 500", failing["steps"][0]["error_examples"][0])
        self.assertIn("did not answer warmup", self.run_test(FakeServer(warmup_ok=False), settings([1], step=1)).stop_reason)
        self.assertIn("does not stream", self.run_test(FakeServer(stream=False), settings([1], step=1)).stop_reason)

    def test_cancellation_keeps_a_partial_step_out_of_the_result(self):
        started = time.perf_counter()
        report = self.run_test(FakeServer(), settings([5, 10], step=30, ttft=500),
                               cancelled=lambda: time.perf_counter() - started > 1.5).to_dict()
        self.assertLess(time.perf_counter() - started, 10)
        self.assertTrue(report["cancelled"])
        self.assertEqual(len(report["steps"]), 1)
        self.assertTrue(report["steps"][0]["cancelled"])
        self.assertLess(report["steps"][0]["arrivals"], 30)
        self.assertEqual(report["summary"]["sustainable_status"], "not_measured")


class SubmissionAndReportTests(unittest.TestCase):
    def setUp(self):
        from app import config
        from app.routes import admin, compare, plans, runs
        from app.services import benchmark_runner, run_queue

        self.runner = benchmark_runner
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        self.path = Path(directory) / "bench.db"
        for patcher in (
            patch.object(config, "DATABASE_PATH", self.path),
            patch.object(run_queue, "dispatch_next"),
            patch.object(load_workload, "MIN_STEP_SECONDS", 1),
            patch("app.services.url_guard.validate_endpoint"),
            patch.object(load_test, "ChatClient", FakeServer(slots=2, service=0.05)),
            patch.object(runs, "get_current_user", AsyncMock(return_value={"id": 1, "role": "admin"})),
            patch.object(compare, "get_current_user", AsyncMock(return_value={"id": 1, "role": "admin"})),
        ):
            self.stack.enter_context(patcher)
        with closing(sqlite3.connect(self.path)) as db:
            db.executescript(Path("app/schema.sql").read_text(encoding="utf-8"))
            db.execute("INSERT INTO users (id, username, role) VALUES (1, 'admin', 'admin')")
            db.execute("INSERT INTO models (id, name, base_url, api_key, model_id) VALUES (1, 'm', 'http://fake.invalid', 'k', 'fake')")
            db.commit()
        app = FastAPI()

        @app.middleware("http")
        async def user(request, call_next):
            request.state.user = {"id": 1, "role": "admin", "username": "admin"}
            return await call_next(request)

        for router in (admin.router, plans.router, runs.router, compare.router):
            app.include_router(router)
        self.client = self.stack.enter_context(TestClient(app))

    def sql(self, query, params=()):
        with closing(sqlite3.connect(self.path, detect_types=DETECT_TYPES)) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute(query, params).fetchall()]
            db.commit()
            return rows

    def submit(self, spec):
        return self.client.post("/admin/run", data={"runs_json": json.dumps([spec]), "scheduled_at": "2099-01-01T00:00",
                                                    "tz_offset": "0"}, follow_redirects=False)

    def test_options_round_trip_and_validation(self):
        from app.services.run_submission import make_run_options, spec_from_run

        # The standard test no longer runs open-loop load (standard-performance-v4): the legacy mode
        # "load" maps to it and its load settings are accepted and ignored.
        self.assertEqual(make_run_options(mode="load", max_concurrency=256), make_run_options(mode="performance"))
        load = {"preset": "chat", "rates": [0.5, 1], "step_seconds": 60, "arrival": "gamma", "burstiness": 3}
        posted = self.submit({"model_id": 1, "mode": "load", "max_concurrency": 128, "load": load,
                              "in_flight_cap": 64})
        self.assertEqual(posted.status_code, 302)
        options = json.loads(self.sql("SELECT run_options_json FROM test_runs")[0]["run_options_json"])
        self.assertEqual((options["mode"], options["performance"], options["max_concurrency"],
                          "load" in options, "in_flight_cap" in options),
                         ("performance", "standard", 8, False, False))
        spec = spec_from_run({"model_id": 1, "run_options_json": json.dumps(options)})
        self.assertNotIn("load", spec)
        response = self.submit({"model_id": 1, "mode": "quality", "max_concurrency": 8, "load": load})
        self.assertEqual(response.status_code, 422)
        self.assertIn("require a performance run", response.text)
        self.assertNotIn("Capacity stage workload", self.client.get("/admin/run").text)

    def run_load(self, run_id, rates=(2, 60), ttft=400):
        load = {"preset": "custom", "workload": workload(ttft=ttft), "rates": list(rates), "step_seconds": 2}
        options = {"mode": "load", "max_concurrency": 32, "load": parse_settings(load).model_dump()}
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("INSERT INTO test_runs (id, model_id, status, run_options_json) VALUES (?, 1, 'pending', ?)",
                       (run_id, json.dumps(options)))
            db.commit()
        finished = Mock()
        model = {"id": 1, "name": "m", "base_url": "http://fake.invalid", "api_key": "k", "model_id": "fake"}
        self.runner._run_benchmark(run_id, model, "load", 32, finished, load=options["load"])
        finished.assert_called_once_with(run_id)
        return self.sql("SELECT * FROM test_runs WHERE id=?", (run_id,))[0]

    def test_runner_reports_capacity_and_comparison(self):
        run = self.run_load(1)
        self.assertEqual(run["status"], "completed", run["error_message"])
        perf = json.loads(run["perf_json"])
        self.assertEqual((perf["schema_version"], perf["kind"]), (5, "open_loop"))
        self.assertEqual(perf["summary"]["sustainable_status"], "established")
        self.assertEqual(self.sql("SELECT * FROM benchmark_progress"), [])
        page = self.client.get("/runs/1").text
        for text in ("Open-loop load", "Sustainable arrival rate", "Results by offered rate", "Request classes", "SLO attainment by offered rate"):
            self.assertIn(text, page)
        self.assertEqual(self.client.get("/runs/1/performance.json").json()["capacity_estimate"]["arrival"]["status"], "established")
        capacity = self.client.get("/runs/1/capacity").text
        self.assertIn("Open-loop arrival capacity", capacity)
        self.assertIn("Established", capacity)
        report = self.client.get("/runs/1/report.html").text
        self.assertIn("Results by offered rate", report)
        self.assertNotIn("<script", report.split("Open-loop load", 1)[1].split("</section>", 1)[0])

        self.run_load(2)
        self.run_load(3, rates=(1, 2), ttft=500)  # different SLO: not comparable
        compare = self.client.get("/compare?runs=1,2,3").text
        self.assertIn("Open-loop comparison", compare)
        self.assertIn("Not comparable with these runs", compare)
        self.assertIn("#3", compare.split("Not comparable with these runs", 1)[1][:200])

    def test_failed_endpoint_fails_the_run_without_leaking_urls(self):
        with patch.object(load_test, "ChatClient", FakeServer(error=True)):
            run = self.run_load(1, rates=(2,))
        self.assertEqual(run["status"], "failed")
        self.assertIn("No request completed", run["error_message"])
        self.assertNotIn("secret.host", run["perf_json"])

    def test_capacity_assumptions_apply_to_the_measured_rate(self):
        perf = {"schema_version": 5, "kind": "open_loop", "protocol": {"workload": parse_settings({"preset": "mixed", "rates": [1]}).workload.model_dump(),
                "attainment_target": 0.95, "workload_hash": "h", "preset": "mixed"},
                "summary": {"sustainable_status": "lower_bound", "sustainable_rate": 4, "goodput_at_sustainable": 3.9},
                "steps": [{"offered_rate": 4, "passed": True, "completed": 100, "incomplete": 0, "input_tokens_reported": 400_000,
                           "output_tokens_delivered": 30_000, "classes": {"chat": {"attainment": 0.97}, "agent": {"attainment": 0.96}}}]}
        result = estimate_capacity({"status": "completed", "perf": perf}, CapacityAssumptions(headroom_percent=25))
        arrival = result["arrival"]
        self.assertEqual(arrival["estimate"]["requests_per_second"], 3)
        self.assertEqual((arrival["mean_input"], arrival["mean_output"]), (4000, 300))
        # 3 req/s × 300 tokens × 60 s split 80/20 over 300 and 2,000 tokens/min/user.
        self.assertEqual(arrival["estimate"]["active_users"], int(3 * 0.8 * 60 * 300 / 300 + 3 * 0.2 * 60 * 300 / 2000))
        self.assertEqual(result["maximum_capacity"]["status"], "lower_bound")
        self.assertEqual(result["revision"], "serving-capacity-v3")
        self.assertIsNone(estimate_capacity({"status": "completed", "perf": {"schema_version": 3, "protocol": {"revision": "x"}}})["arrival"])


if __name__ == "__main__":
    unittest.main()
