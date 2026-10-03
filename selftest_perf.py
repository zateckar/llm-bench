"""Fixed performance protocol and streaming accounting; no network calls."""

import json
import threading
import time
import unittest
from unittest.mock import Mock, patch
import tiktoken
import perf
from llm_client import ChatClient, ClientConfig
from models import ConcurrencyPoint, LatencyStats, RequestMetrics, TokenUsage

CONFIG = ClientConfig("http://fake.invalid", "unused", "fake")


class FakeClient:
    instances = []

    def __init__(self, config):
        self.config = config
        self.session = Mock()
        self.streaming_supported = True
        self._stream_usage_supported = True
        self.calls = []
        self.instances.append(self)

    def complete(self, prompt, **kwargs):
        self.calls.append(kwargs)
        time.sleep(0.001)
        metrics = RequestMetrics(
            latency_ms=2,
            ttft_ms=1,
            prompt_tokens=1024,
            completion_tokens=80,
            streamed=True,
            stream_chunks=1,
            stream_span_ms=0,
            completion_tokens_estimated=True,
        )
        return "answer", TokenUsage(1024, 80), metrics


class PerfTests(unittest.TestCase):
    def setUp(self):
        FakeClient.instances = []

    def test_grid_and_validation(self):
        self.assertEqual(perf.PerfConfig(8).levels, (1, 2, 4, 8))
        self.assertEqual(perf.PerfConfig(3).levels, (1, 2, 3))
        self.assertEqual(perf.PerfConfig(1).levels, (1,))
        for bad in (0, 33, True, 1.5, "8"):
            with self.assertRaises(ValueError):
                perf.PerfConfig(bad)

    def test_prompt_exact_length_and_unique_requests(self):
        enc = tiktoken.get_encoding("cl100k_base")
        prompts = [perf._prompt("a" * 32, i) for i in (0, 100, 99999)]
        self.assertTrue(all(len(enc.encode(p)) == 1024 for p in prompts))
        self.assertEqual(len(set(prompts)), 3)
        self.assertTrue(all(p.endswith("with no prose.") for p in prompts))

    def test_reused_warm_sessions_and_single_attempt_measurements(self):
        progress = Mock()
        with patch.object(perf, "ChatClient", FakeClient):
            report = perf.run_perf_suite(CONFIG, perf.PerfConfig(3), progress)
        self.assertEqual(len(FakeClient.instances), 3)
        self.assertEqual([p.requests for p in report.concurrency], [24, 24, 24])
        self.assertEqual([p.errors for p in report.concurrency], [0, 0, 0])
        self.assertEqual(sum(len(c.calls) for c in FakeClient.instances), 76)
        self.assertEqual([k["retries"] for k in FakeClient.instances[0].calls[:2]], [3, 3])
        self.assertTrue(
            all(
                k["retries"] == 1
                for c in FakeClient.instances
                for k in c.calls[2 if c is FakeClient.instances[0] else 0 :]
            )
        )
        for c in FakeClient.instances:
            c.session.close.assert_called_once()
        self.assertEqual(progress.call_args.args[1:], (76, 76))
        self.assertTrue(
            all(
                p.estimated_token_requests == 24 and p.burst_delivery_requests == 24
                for p in report.concurrency
            )
        )
        self.assertEqual(report.to_dict()["schema_version"], 3)
        self.assertNotIn("capacity", json.dumps(report.to_dict()))

    def test_failures_stay_in_wall_clock_denominator(self):
        p = ConcurrencyPoint(
            2,
            10,
            4,
            1000,
            LatencyStats.from_samples([50] * 6),
            LatencyStats(),
            600,
            1024,
            failure_latency=LatencyStats.from_samples([100] * 4),
        )
        self.assertEqual(p.requests_per_sec, 6)
        self.assertEqual(p.output_tokens_per_sec, 600)
        self.assertEqual(p.error_rate, 0.4)
        self.assertEqual(p.to_dict()["failure_latency"]["count"], 4)

    def test_bounded_concurrency_and_all_failed_level_stop(self):
        active = peak = 0
        lock = threading.Lock()

        def complete(client, prompt, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.002)
            with lock:
                active -= 1
            ok = kwargs["retries"] == 3 or len(client.calls) == 0
            client.calls.append(kwargs)
            return (
                "",
                TokenUsage(),
                RequestMetrics(ok=ok, latency_ms=2, error=None if ok else "timeout"),
            )

        with (
            patch.object(perf, "ChatClient", FakeClient),
            patch.object(FakeClient, "complete", complete),
        ):
            report = perf.run_perf_suite(CONFIG, perf.PerfConfig(3))
        self.assertLessEqual(peak, 3)
        self.assertEqual(len(report.concurrency), 1)
        self.assertEqual(report.concurrency[0].errors, 24)
        self.assertIn("All requests failed", report.notes[-1])

    def test_cancellation_closes_sessions(self):
        def progress(phase, n, total):
            stopped[0] = True

        stopped = [False]
        with patch.object(perf, "ChatClient", FakeClient):
            report = perf.run_perf_suite(
                CONFIG, perf.PerfConfig(), progress, cancelled=lambda: stopped[0]
            )
        self.assertTrue(report.cancelled)
        self.assertEqual(report.concurrency, [])
        FakeClient.instances[0].session.close.assert_called_once()

    def test_progress_does_not_inflate_timed_throughput(self):
        clock = [0.0]

        def progress(*args):
            clock[0] += 10

        with (
            patch.object(perf, "ChatClient", FakeClient),
            patch.object(perf.time, "perf_counter", side_effect=lambda: clock[0]),
        ):
            report = perf.run_perf_suite(CONFIG, perf.PerfConfig(1), progress)
        self.assertEqual(report.concurrency[0].wall_ms, 0)

    def test_stream_chunks_are_not_token_counts(self):
        class Response:
            status_code = 200

            def iter_lines(self):
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {"content": " ".join(str(i) for i in range(80))},
                                    "finish_reason": "stop",
                                }
                            ]
                        }
                    )
                ).encode()
                yield b"data: [DONE]"

            def close(self):
                pass

        client = ChatClient(CONFIG)
        try:
            with patch.object(client.session, "post", return_value=Response()):
                text, tokens, metrics = client.complete("test")
            self.assertTrue(tokens.completion_tokens_estimated)
            self.assertGreater(tokens.completion_tokens, 1)
            self.assertEqual(metrics.stream_chunks, 1)
            self.assertTrue(metrics.burst_delivery)
            self.assertEqual(metrics.completion_tokens, tokens.completion_tokens)
            self.assertEqual(metrics.ttft_ms is None, False)
            self.assertTrue(text)
        finally:
            client.session.close()


if __name__ == "__main__":
    unittest.main()
