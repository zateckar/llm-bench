"""Regression tests for failures exposed by the newest recorded runs."""

import unittest

from app.benchmarking.models import RequestMetrics
from app.benchmarking.perf_sweep import SweepConfig, Targets, judge
from app.benchmarking.session_load import analyse_level
from selftest_sessions import record, settings


class MeasurementAuditTests(unittest.TestCase):
    def test_multi_event_buffering_does_not_establish_decode_speed(self):
        normal = RequestMetrics(streamed=True, completion_tokens=100, stream_chunks=50,
                                stream_span_ms=1000, first_delivery_fraction=.01,
                                chunk_gaps_ms=[20] * 49)
        self.assertIsNotNone(normal.output_token_time_ms)
        normal.first_delivery_fraction = .6
        self.assertTrue(normal.burst_delivery)
        self.assertIsNone(normal.output_token_time_ms)
        normal.first_delivery_fraction = .01
        normal.chunk_gaps_ms = [.01] * 48 + [999]
        self.assertTrue(normal.burst_delivery)
        self.assertIsNone(normal.output_token_time_ms)

    def test_p95_target_requires_complete_delivery_coverage(self):
        cell = {"requests": 24, "completed": 24,
                "ttft": {"count": 24, "p95_ms": 100},
                "output_token_time": {"count": 2, "p95_ms": .9}}
        self.assertEqual(judge(cell, None, Targets()), (None, ""))
        cell["output_token_time"]["count"] = 24
        self.assertEqual(judge(cell, None, Targets()), (True, ""))
        config = SweepConfig(min_samples=20)
        self.assertEqual([c * config.rounds_for(c) for c in (1, 2, 4, 8, 32)], [20, 20, 20, 24, 32])

    def test_session_rates_include_the_drain_for_the_counted_requests(self):
        s = settings(warmup_seconds=60, measure_seconds=180)
        rows = [record(k, 2) for k in (0, 1) for _ in range(20)]
        for r in rows:
            r.finished = 6
        level = analyse_level(s.model, s, 4096, 10, rows, window=(1, 3), started_at=0, ended_at=6)
        self.assertTrue(level["passed"])
        self.assertEqual((level["rate_seconds"], level["drain_seconds"]), (5, 3))
        self.assertEqual(level["output_tokens_per_sec"], 800 / 5)
        self.assertEqual(level["requests_per_sec"], 40 / 5)

    def test_missing_class_or_incomplete_warmup_cannot_pass(self):
        s = settings(warmup_seconds=60, measure_seconds=180)
        chat_only = [record(0, 2) for _ in range(20)]
        kwargs = dict(window=(1, 3), started_at=0, ended_at=4)
        level = analyse_level(s.model, s, 4096, 10, chat_only, **kwargs)
        self.assertFalse(level["passed"])
        both = [*chat_only, *(record(1, 2) for _ in range(20))]
        level = analyse_level(s.model, s, 4096, 10, both, **kwargs, warmup_complete=False)
        self.assertFalse(level["passed"])


if __name__ == "__main__":
    unittest.main()
