"""Budget, stream progress and performance calculation regressions; no endpoint calls."""

from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

from app.benchmarking import perf
from app.benchmarking.llm_client import ChatClient, ClientConfig, _StreamDeadlineExceeded, _response_lines
from app.benchmarking.models import RequestMetrics
from app.services.benchmark_runner import _build_client_config
from app.services.html_reports import performance_view, quality_timing_view
from selftest_client import Response, event


CONFIG = ClientConfig("http://fake.invalid", "unused", "fake", max_retries=3)


class TelemetryTests(unittest.TestCase):
    def test_deadline_preserves_progress_without_restarting_generation(self):
        client = ChatClient(replace(CONFIG, stream_deadline=1800))
        response = Response()

        def payloads(_response, _started, deadline):
            self.assertEqual(deadline, 1800)
            import json
            yield json.dumps(event(reasoning="working"))
            yield json.dumps(event(content="partial"))
            raise _StreamDeadlineExceeded("stream deadline exceeded after 1800s")

        try:
            with (patch.object(client.session, "post", return_value=response) as post,
                  patch("app.benchmarking.llm_client._sse_payloads", side_effect=payloads),
                  patch.object(client, "_sleep_backoff") as backoff):
                text, usage, metrics = client.complete("test")
            self.assertFalse(metrics.ok)
            self.assertEqual(metrics.attempts, 1)
            self.assertIn("deadline exceeded", text)
            self.assertEqual(usage.completion_tokens, 0)
            post.assert_called_once()
            backoff.assert_not_called()
            response.close.assert_called_once()
            progress = metrics.attempt_diagnostics[0]
            self.assertEqual((progress["stream_chunks"], progress["reasoning_chars"], progress["content_chars"]), (2, 7, 7))
            self.assertIsNotNone(progress["ttft_ms"])
            self.assertIsNotNone(progress["last_delivery_ms"])
        finally:
            client.session.close()

    def test_eof_after_budget_is_still_a_deadline_failure(self):
        response = Response()
        with patch("app.benchmarking.llm_client.time.perf_counter", return_value=5):
            with self.assertRaises(_StreamDeadlineExceeded):
                list(_response_lines(response, 0, 4))

    def test_separate_budgets_and_validation(self):
        cfg = _build_client_config({"base_url": "http://fake.invalid", "api_key": "unused", "model_id": "fake"})
        self.assertEqual((cfg.timeout, cfg.stream_deadline_seconds), (180, 1800))
        self.assertEqual(CONFIG.stream_deadline_seconds, 360)
        for bad in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                replace(CONFIG, stream_deadline=bad)

    def test_output_time_proxy_excludes_ineligible_samples(self):
        m = RequestMetrics(streamed=True, stream_chunks=3, stream_span_ms=990, completion_tokens=100)
        self.assertEqual(m.output_token_time_ms, 10)
        for changes in ({"ok": False}, {"streamed": False}, {"completion_tokens": 1},
                        {"completion_tokens_estimated": True}, {"stream_chunks": 1},
                        {"stream_span_ms": 2}, {"stream_span_ms": None}):
            self.assertIsNone(replace(m, **changes).output_token_time_ms)

    def test_load_samples_preserve_failures_and_wall_time_denominator(self):
        good = RequestMetrics(latency_ms=50, ttft_ms=10, completion_tokens=100,
                              streamed=True, stream_chunks=3, stream_span_ms=990,
                              chunk_gaps_ms=[400, 590])
        bad = RequestMetrics(ok=False, latency_ms=500, error="failure")
        with (patch.object(perf, "_probe", side_effect=[good, bad, good]),
              patch.object(perf.time, "perf_counter", side_effect=[0, 1])):
            point = perf._measure_level([Mock()], 1, ["a", "b", "c"], lambda: False)
        data = point.to_dict()
        self.assertEqual((point.requests, point.errors, point.requests_per_sec, point.output_tokens_per_sec), (3, 1, 2, 200))
        self.assertEqual(data["latency"]["count"], 2)
        self.assertEqual(data["output_length"]["mean"], 100)
        self.assertEqual(data["output_token_time"]["p50_ms"], 10)
        self.assertEqual(data["chunk_gap"]["count"], 4)
        self.assertEqual(len(data["samples"]), 3)
        view = performance_view([{"label": "fixture", "perf": {"schema_version": 3, "concurrency": [data]}}])
        self.assertEqual(len(view["distributions"]), 3)

    def test_quality_errors_are_separate_from_successful_percentiles(self):
        rows = [
            {"category": "Terminal Algorithms", "request_ok": 0, "latency_ms": 360000,
             "detail": "stream deadline exceeded", "ttft_ms": None},
            {"category": "Code", "request_ok": 1, "latency_ms": 1000, "ttft_ms": 50},
        ]
        view = quality_timing_view({"label": "run", "results": rows})
        self.assertEqual(view["deadline_count"], 1)
        self.assertEqual(view["latency_p50"], "1,000.0 ms")
        self.assertEqual(view["success_rate"], "50.0%")
        self.assertEqual(len(view["series"]), 3)


if __name__ == "__main__":
    unittest.main()
