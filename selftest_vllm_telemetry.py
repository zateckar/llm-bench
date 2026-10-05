"""B300 selector isolation, failure safety, frozen telemetry and report correlation."""

import json
import unittest
from unittest.mock import Mock, patch

import requests

from app.benchmarking.vllm_telemetry import Settings, VllmTelemetry, metrics_model, metrics_scope, queries
from app.services.telemetry_views import telemetry_view
from app.services.html_reports import staged_view
from app.templates_config import templates

SCOPE = {"hardware": "B300", "host": "smbea02n01", "job": "vllm", "model_name": "test/model"}
SETTINGS = Settings("https://metrics.example/v1/query", "subscription-secret", "authorization-secret")


def response(payload, status=200):
    result = Mock(status_code=status)
    result.iter_content.return_value = iter([json.dumps(payload).encode()])
    return result


def fake_session(*, offset=0, missing=None, failure=None):
    session = Mock()

    def get(url, **kwargs):
        query = kwargs["params"]["query"]
        if url.endswith("/query"):
            return response({"status": "success", "data": {"resultType": "vector", "result": [
                {"metric": {"instance": "server:8009"}, "value": [1000+offset, "0"]}]}})
        if failure and failure in query:
            return response({"status": "error", "error": SETTINGS.subscription_key}, 401)
        values = [[1000, "1"], [1005, "NaN"], [1010, "2"], [1015, "3"]]
        if missing and missing in query:
            values = [[1000, "NaN"]]
        return response({"status": "success", "data": {"resultType": "matrix", "result": [
            {"metric": {"instance": "server:8009"}, "values": values}]}})

    session.get.side_effect = get
    return session


def capture(session):
    collector = VllmTelemetry(SCOPE, settings=SETTINGS, session=session)
    with patch("app.benchmarking.vllm_telemetry.time.time", side_effect=[1000, 1000, 1000, 1000, 1060]):
        collector.begin()
        return collector.finish()


class CollectorTests(unittest.TestCase):
    def test_exact_b300_scope_and_escaped_model_labels(self):
        definitions, selector = queries({**SCOPE, "model_name": 'model"},job="dcgm-exporter'})
        self.assertIn('model_name="model\\\"},job=\\\"dcgm-exporter"', selector)
        self.assertTrue(all('Hostname="smbea02n01",job="vllm"' in d["query"] for d in definitions))
        with self.assertRaises(ValueError):
            queries({**SCOPE, "hardware": "MI300A"})
        for value in (True, 1, "bad\nname", "x"*513):
            with self.assertRaises(ValueError):
                metrics_model(value)
        self.assertIsNone(metrics_scope({}))
        with patch.dict("os.environ", {"PROMETHEUS_B300_HOST": "smbea02n01", "PROMETHEUS_TIMESTAMP_SHIFT_SECONDS": ""}):
            self.assertEqual(metrics_scope({"b300_metrics_model": " test/model "}), SCOPE)

    def test_authorization_bounded_history_and_missing_values(self):
        session = fake_session()
        data = capture(session)
        self.assertEqual(data["status"], "collected")
        self.assertEqual(data["window"], {"start": 880, "end": 1060, "measurement_start": 1000, "step_seconds": 5, "baseline_seconds": 120})
        self.assertIsNone(data["metrics"][0]["series"][0]["values"][1][1])
        self.assertEqual(len(data["metrics"]), 12)
        for call in session.get.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(kwargs["headers"]["Authorization"], "Basic authorization-secret")
            self.assertFalse(kwargs["allow_redirects"])
            self.assertTrue(kwargs["stream"])
        session.close.assert_called_once()
        encoded = json.dumps(data, allow_nan=False)
        self.assertNotIn(SETTINGS.subscription_key, encoded)
        self.assertNotIn(SETTINGS.authorization, encoded)

    def test_partial_failure_does_not_invent_zero_or_leak_gateway_body(self):
        data = capture(fake_session(missing="kv_cache", failure="request_decode_time"))
        self.assertEqual(data["status"], "partial")
        status = {m["id"]: m["status"] for m in data["metrics"]}
        self.assertEqual(status["kv_cache"], "missing")
        self.assertEqual(status["decode"], "error")
        self.assertNotIn("subscription-secret", json.dumps(data))

    def test_setup_failure_and_cancellation_close_without_retry(self):
        session = Mock()
        session.get.side_effect = requests.ConnectionError("authorization-secret")
        collector = VllmTelemetry(SCOPE, settings=SETTINGS, session=session)
        collector.begin()
        self.assertEqual(collector.finish()["status"], "unavailable")
        session.get.assert_called_once()
        session.close.assert_called_once()
        self.assertNotIn("authorization-secret", json.dumps(collector.data))
        session = fake_session()
        collector = VllmTelemetry(SCOPE, settings=SETTINGS, session=session, cancelled=lambda: True)
        collector.begin()
        self.assertEqual(collector.finish()["status"], "cancelled")
        self.assertEqual(session.get.call_count, 1)
        session.close.assert_called_once()

    def test_redirects_and_oversized_responses_are_not_followed_or_saved(self):
        for result in (response({}, 302), Mock(status_code=200, iter_content=Mock(return_value=iter([b"x"*(8*1024*1024+1)])))):
            session = Mock(get=Mock(return_value=result))
            collector = VllmTelemetry(SCOPE, settings=SETTINGS, session=session)
            collector.begin()
            self.assertEqual(collector.finish()["status"], "unavailable")
            result.close.assert_called_once()
            session.get.assert_called_once()

    def test_malformed_series_fails_only_that_metric(self):
        session = fake_session()
        original = session.get.side_effect

        def get(url, **kwargs):
            if "kv_cache" in kwargs["params"]["query"]:
                return response({"status": "success", "data": {"resultType": "matrix", "result": ["invalid"]}})
            return original(url, **kwargs)

        session.get.side_effect = get
        self.assertEqual(capture(session)["status"], "partial")

    def test_configured_shift_changes_query_window_and_preserves_source_samples(self):
        session = fake_session(offset=-85)
        original = session.get.side_effect

        def get(url, **kwargs):
            if url.endswith("/query"):
                return original(url, **kwargs)
            return response({"status": "success", "data": {"resultType": "matrix", "result": [
                {"metric": {"instance": "server:8009"}, "values": [[915, "1"], [920, "2"], [925, "3"]]}]}})

        session.get.side_effect = get
        collector = VllmTelemetry({**SCOPE, "timestamp_shift_seconds": "85"}, settings=SETTINGS, session=session)
        with patch("app.benchmarking.vllm_telemetry.time.time", side_effect=[1000, 1000, 1000, 1000, 1060]):
            collector.begin()
            data = collector.finish()
        self.assertEqual(session.get.call_args_list[0].kwargs["params"]["time"], 915)
        self.assertEqual(session.get.call_args_list[1].kwargs["params"]["start"], 795)
        self.assertEqual(session.get.call_args_list[1].kwargs["params"]["end"], 975)
        self.assertEqual(data["metrics"][0]["series"][0]["values"][0][0], 915)
        self.assertEqual(data["latest_aligned_sample_timestamp"], 1010)
        view = telemetry_view({"perf": {"telemetry": data, "concurrency": [
            {"concurrency": 4, "started_at": 1000, "ended_at": 1020}]}})
        self.assertEqual(view["rows"][0]["samples"], 3)
        self.assertIn("Queueing observed", view["rows"][0]["signals"])
        self.assertEqual(view["charts"][0]["series"][0]["points"][0][0], 0)

    def test_invalid_or_disagreeing_shift_does_not_enable_phase_diagnosis(self):
        for shift in ("NaN", "inf", "601", "invalid"):
            session = fake_session()
            collector = VllmTelemetry({**SCOPE, "timestamp_shift_seconds": shift}, settings=SETTINGS, session=session)
            collector.begin()
            self.assertEqual(collector.finish()["status"], "unavailable")
            session.get.assert_not_called()
        from app.benchmarking.vllm_telemetry import alignment_shift
        self.assertEqual(alignment_shift({"api_clock_offset_seconds": -120, "alignment": {"timestamp_shift_seconds": 85}}), (85, False))


class ReportTests(unittest.TestCase):
    def run_data(self, offset=0):
        telemetry = capture(fake_session(offset=offset))
        return {"label": "B300 fixture", "perf": {"schema_version": 3, "telemetry": telemetry, "concurrency": [
            {"concurrency": 2, "requests": 4, "errors": 0, "started_at": 1000, "ended_at": 1020,
             "output_tokens_per_sec": 20, "latency": {"p95_ms": 500}, "ttft": {"p95_ms": 100}}]}}

    def test_phase_summary_and_svg_survive_offline_rendering(self):
        run = self.run_data()
        view = telemetry_view(run)
        self.assertEqual(view["rows"][0]["samples"], 3)
        self.assertIn("Queueing observed", view["rows"][0]["signals"])
        context = staged_view([run], offline=True)
        html = templates.get_template("performance_report.html").render(performance=context, offline=True)
        self.assertIn("B300 vLLM timeline", html)
        self.assertIn("Server and benchmark output", html)
        self.assertIn("<svg", html)
        self.assertIn("All traffic", html)
        self.assertNotIn("subscription-secret", html)
        self.assertNotIn("authorization-secret", html)

    def test_clock_disagreement_disables_phase_diagnosis(self):
        view = telemetry_view(self.run_data(offset=-85))
        row = view["rows"][0]
        self.assertIn("Clock alignment uncertain", row["signals"])
        self.assertNotIn("Queueing observed", row["signals"])
        self.assertEqual(row["waiting"], "n/a")

    def test_common_time_axis_preserves_empty_tail_and_hides_empty_legends(self):
        run = self.run_data()
        run["perf"]["concurrency"] = []
        view = telemetry_view(run)
        for chart in view["charts"]:
            self.assertIn(">-2</text>", chart["svg"])
            self.assertIn(">-1.5</text>", chart["svg"])
            self.assertTrue(all("Benchmark" not in s["label"] for s in chart["series"]))

    def test_legacy_reports_and_sweep_phase_timestamps(self):
        self.assertIsNone(telemetry_view({"perf": {"schema_version": 3}}))
        run = self.run_data()
        point = run["perf"]["concurrency"][0]
        run["perf"] = {"schema_version": 4, "kind": "context_sweep", "telemetry": run["perf"]["telemetry"],
                       "cells": [{**point, "context_tokens": 8192, "effort": "low", "cache_reuse": point}]}
        view = telemetry_view(run)
        self.assertEqual(view["phase_count"], 2)
        self.assertTrue(view["rows"][0]["label"].startswith("Cold"))
        self.assertTrue(view["rows"][1]["label"].startswith("Warm"))

    def test_sweep_collector_is_persisted_without_changing_measured_duration(self):
        from app.services import benchmark_runner as runner
        from app.benchmarking.perf_sweep import SweepConfig, SweepReport
        from app.benchmarking.llm_client import ClientConfig
        collector = Mock(data={"status": "collecting", "scope": SCOPE})
        collector.finish.return_value = {"status": "collected", "scope": SCOPE}
        report = SweepReport("fake", {}, successful_requests=1)

        def sweep(*args, **kwargs):
            kwargs["checkpoint"](report.to_dict())
            return report

        with (patch.object(runner, "_begin_telemetry", return_value=collector),
              patch.object(runner, "_run_is_active", return_value=True),
              patch.object(runner, "_store_sweep_checkpoint") as checkpoint,
              patch.object(runner, "_summarise_and_finish") as finish,
              patch("app.benchmarking.perf_sweep.run_sweep", side_effect=sweep),
              patch.object(runner.time, "perf_counter", side_effect=[100, 120])):
            runner._run_context_sweep(1, ClientConfig("", "", "fake"), SweepConfig(256, 1), SCOPE)
        self.assertEqual(checkpoint.call_args.args[1]["telemetry"]["status"], "collecting")
        self.assertEqual(finish.call_args.args[4], 20000)
        self.assertEqual(json.loads(finish.call_args.args[5])["telemetry"]["status"], "collected")
        collector.finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
