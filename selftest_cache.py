"""Cache telemetry coverage, rate proxies and paired prefix workloads."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from app.benchmarking.cache_metrics import cache_metrics
from app.benchmarking.llm_client import ClientConfig, _parse_usage
from app.benchmarking.models import RequestMetrics
from app.benchmarking.perf import PerfConfig, run_perf_suite
from app.benchmarking.perf_sweep import PromptBank
from app.services.html_reports import performance_view
from selftest_perf import FakeClient


class CacheTests(unittest.TestCase):
    def test_unknown_and_explicit_zero_cache_usage_are_distinct(self):
        for raw, expected in [({}, False), ({"prompt_cache_hit_tokens": 0}, True),
                              ({"prompt_tokens_details": {"cached_tokens": 0}}, True),
                              ({"prompt_cache_hit_tokens": -1}, False),
                              ({"prompt_cache_hit_tokens": True}, False),
                              ({"prompt_tokens": 10, "prompt_cache_hit_tokens": 11}, False)]:
            self.assertEqual(_parse_usage(raw).cached_tokens_reported, expected)
        self.assertIsNone(cache_metrics([RequestMetrics(prompt_tokens=100)], 100)["cache_hit_fraction"])
        self.assertEqual(cache_metrics([RequestMetrics(prompt_tokens=100, cached_tokens_reported=True)], 100)["cache_hit_fraction"], 0)

    def test_rate_denominators_and_invalid_samples(self):
        good = RequestMetrics(prompt_tokens=1000, cached_tokens=800, cached_tokens_reported=True,
                              ttft_ms=100, streamed=True, completion_tokens=101,
                              stream_chunks=5, stream_span_ms=1000)
        data = cache_metrics([good, replace(good, ok=False)], 2000)
        self.assertEqual(data["cache_hit_fraction"], 0.8)
        self.assertEqual(data["uncached_prefill_tokens_per_sec"]["p50"], 2000)
        self.assertEqual(data["effective_prompt_tokens_per_sec"]["p50"], 10000)
        self.assertEqual(data["decode_tokens_per_sec"]["p50"], 100)
        self.assertEqual(data["cached_tokens_per_wall_sec"], 400)
        for bad in [replace(good, ttft_ms=None), replace(good, prompt_tokens_estimated=True),
                    replace(good, streamed=False), replace(good, attempts=2)]:
            self.assertEqual(cache_metrics([bad], 100)["uncached_prefill_tokens_per_sec"]["count"], 0)
        for bad in [replace(good, stream_chunks=1), replace(good, completion_tokens_estimated=True)]:
            self.assertEqual(cache_metrics([bad], 100)["decode_tokens_per_sec"]["count"], 0)

    def test_shared_prefix_keeps_suffix_unique_and_exact_length(self):
        for size in (256, 8192, 32768):
            bank = PromptBank(size)
            warm = [bank.prompt(i, "warm") for i in (0, 1, 100001)]
            self.assertTrue(all(len(bank.encoding.encode(p)) == size for p in warm))
            self.assertEqual(len(set(warm)), 3)
            self.assertEqual(warm[0][:size], warm[1][:size])
            self.assertNotEqual(bank.prompt(0)[:100], bank.prompt(1)[:100])
            self.assertNotEqual(warm[0][:100], bank.prompt(0)[:100])

    def test_fixed_report_and_html_view_include_both_modes_at_same_load(self):
        FakeClient.instances = []
        with patch("app.benchmarking.perf.ChatClient", FakeClient):
            report = run_perf_suite(ClientConfig("", "", "fixture"), PerfConfig(1)).to_dict()
        self.assertEqual([p["context_tokens"] for p in report["cache_reuse"]], [8192, 32768])
        for pair in report["cache_reuse"]:
            self.assertTrue(pair["priming_ok"])
            self.assertEqual(pair["cold"]["requests"], pair["warm"]["requests"])
            self.assertEqual(pair["cold"]["concurrency"], pair["warm"]["concurrency"])
        view = performance_view([{"label": "fixture", "perf": report}])
        self.assertEqual([r["mode"] for r in view["cache_rows"]], ["cold", "warm", "cold", "warm"])
        self.assertTrue(all(r["hit"] == "n/a" for r in view["cache_rows"]))


if __name__ == "__main__":
    unittest.main()
