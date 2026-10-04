"""Capacity arithmetic, workload isolation, telemetry alignment and authenticated exports."""

import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.routes import runs
from app.services.capacity import REVISION, CapacityAssumptions, daily_budget, estimate_capacity
from app.services.html_reports import performance_view
from app.templates_config import templates


def point(concurrency=4, context=1024, rate=200):
    return {"concurrency": concurrency, "context_tokens": context, "status": "measured",
            "requests": 100, "completed": 100, "errors": 0, "wall_ms": 128000,
            "output_tokens": 25600, "prompt_tokens": context*100,
            "output_tokens_per_sec": rate,
            "latency": {"p95_ms": 8000, "mean_ms": 4000}, "ttft": {"p95_ms": 1000},
            "output_token_time": {"count": 100, "p95_ms": 25}}


def run():
    return {"id": 1, "status": "completed", "label": "#1 fake", "perf": {
        "schema_version": 3, "protocol": {"revision": "performance-v8", "input_reference_tokens": 1024},
        "concurrency": [point()], "cache_reuse": []}}


class CapacityTests(unittest.TestCase):
    def test_daily_volume_users_and_slots_have_distinct_units(self):
        result = estimate_capacity(run())
        chat = result["scenarios"][0]
        # 200 output/s x 70% = 140; busy distribution effective duration = 22,500s.
        self.assertAlmostEqual(chat["estimate"]["output_tokens_per_second"], 140)
        self.assertAlmostEqual(chat["estimate"]["daily_output_tokens"], 140*22500*0.98)
        self.assertAlmostEqual(chat["estimate"]["flat_daily_output_tokens"], 140*86400*0.98)
        self.assertEqual(chat["estimate"]["active_users"], 28)
        self.assertEqual(chat["evidence"]["concurrency"], 4)
        self.assertAlmostEqual(chat["mean_inflight"], 140/256*4)
        self.assertEqual(chat["estimate"]["daily_total_tokens"], chat["estimate"]["daily_input_tokens"]+chat["estimate"]["daily_output_tokens"])
        self.assertIsNone(result["scenarios"][1]["estimate"])
        self.assertIsNone(result["mix"]["estimate"])

    def test_peak_distribution_checks_both_busy_and_quiet_periods(self):
        cfg = CapacityAssumptions(busy_hours=4, busy_traffic_percent=10, peak_factor=1, availability_percent=100)
        self.assertAlmostEqual(daily_budget(1, cfg), 20*3600/0.9)
        cfg = cfg.model_copy(update={"peak_factor": 20})
        self.assertAlmostEqual(daily_budget(1, cfg), 4*3600/0.1/20)

    def test_quality_usage_is_visible_without_becoming_a_chat_capacity_limit(self):
        data = run()
        data.update(total_questions=315, total_completion_tokens=2034027,
                    duration_ms=6654087.368356995, workers=4,
                    created_at="2026-10-04T02:20:33+00:00", completed_at="2026-10-04T11:27:00+00:00")
        result = estimate_capacity(data)
        observed = result["observations"][0]
        self.assertEqual(observed["output_tokens"], 2034027)
        self.assertAlmostEqual(observed["seconds"]/3600, 1.8483576023)
        self.assertAlmostEqual(observed["output_tokens_per_second"], 305.6808375665)
        self.assertGreater(observed["flat_24h_output_tokens"], 26_000_000)
        self.assertIsNone(result["maximum_capacity"]["output_tokens_per_second"])
        self.assertEqual(result["maximum_capacity"]["status"], "not_established")
        self.assertIn("ceiling reached", result["scenarios"][0]["limit"])
        # The unrelated quality mix does not inflate the matching chat workload.
        self.assertEqual(result["scenarios"][0]["estimate"], estimate_capacity(run())["scenarios"][0]["estimate"])
        html = templates.get_template("capacity_report.html").render(capacity_views=[result], offline=True)
        self.assertIn("26.41M", html)
        self.assertIn("Traffic budgets at tested rates", html)
        self.assertNotIn("confidence intervals", html)

    def test_quality_only_runs_show_observed_rates_but_not_workload_budgets(self):
        data = {"id": 1, "status": "completed", "total_questions": 100,
                "total_completion_tokens": 3600000, "duration_ms": 3600000, "workers": 4}
        result = estimate_capacity(data)
        self.assertEqual(result["observations"][0]["output_tokens_per_second"], 1000)
        self.assertIsNone(result["scenarios"][0]["estimate"])
        self.assertIsNone(result["scenarios"][1]["estimate"])
        for duration in (0, None, float("nan")):
            data["duration_ms"] = duration
            self.assertEqual(estimate_capacity(data)["observations"], [])

    def test_fastest_point_that_fails_latency_does_not_set_capacity(self):
        data = run()
        slow = point(concurrency=32, rate=800)
        slow["ttft"]["p95_ms"] = 9000
        data["perf"]["concurrency"].append(slow)
        chat = estimate_capacity(data)["scenarios"][0]
        self.assertEqual(chat["evidence"]["concurrency"], 4)
        self.assertIn("First-token latency exceeds target", chat["rejections"])
        self.assertIn("Higher tested concurrency", chat["limit"])

    def test_missing_burst_estimated_or_incomplete_streaming_evidence_blocks_estimates(self):
        for updates in ({"burst_delivery_requests": 1}, {"estimated_token_requests": 1},
                        {"estimated_prompt_requests": 1}, {"completed": 98},
                        {"output_token_time": {"count": 99, "p95_ms": 25}},
                        {"output_token_time": {"count": 100, "p95_ms": 1000}},
                        {"requests": 3, "completed": 3}, {"completed": 101}):
            data = run()
            data["perf"]["concurrency"][0].update(updates)
            self.assertIsNone(estimate_capacity(data)["scenarios"][0]["estimate"], updates)

    def test_reasoning_cache_and_context_are_not_pooled(self):
        data = run()
        cold = point(context=32768, rate=100)
        warm = point(context=32768, rate=500)
        data["perf"]["cache_reuse"] = [{"context_tokens": 32768, "cold": cold, "warm": warm, "priming_ok": True}]
        result = estimate_capacity(data)
        self.assertAlmostEqual(result["scenarios"][1]["estimate"]["output_tokens_per_second"], 70)
        self.assertIsNone(estimate_capacity(data, CapacityAssumptions(effort="high"))["scenarios"][1]["estimate"])
        warm_result = estimate_capacity(data, CapacityAssumptions(cache_mode="warm"))
        # Request throughput bounds output throughput; no assuming 500 tokens/s from one count.
        self.assertAlmostEqual(warm_result["scenarios"][1]["estimate"]["output_tokens_per_second"], 140)
        data["perf"]["cache_reuse"][0]["priming_ok"] = False
        self.assertIsNone(estimate_capacity(data, CapacityAssumptions(cache_mode="warm"))["scenarios"][1]["estimate"])

    def test_mix_shares_one_capacity_budget_and_uses_request_weights(self):
        data = run()
        agent = point(context=32768, rate=100)
        data["perf"]["cache_reuse"] = [{"context_tokens": 32768, "cold": agent}]
        result = estimate_capacity(data)
        chat_rate = result["scenarios"][0]["estimate"]["requests_per_second"]
        agent_rate = result["scenarios"][1]["estimate"]["requests_per_second"]
        mix_rate = result["mix"]["estimate"]["requests_per_second"]
        self.assertAlmostEqual(0.8*mix_rate/chat_rate+0.2*mix_rate/agent_rate, 1)
        self.assertLess(result["mix"]["estimate"]["output_tokens_per_second"], 140)
        cfg = CapacityAssumptions(chat_mix_percent=100)
        self.assertAlmostEqual(estimate_capacity(run(), cfg)["mix"]["estimate"]["requests_per_second"], chat_rate)

    def test_unfinished_cancelled_and_unversioned_results_do_not_claim_capacity(self):
        for status in ("pending", "running"):
            data = run()
            data["status"] = status
            self.assertIsNone(estimate_capacity(data)["scenarios"][0]["estimate"])
        data = run()
        data["perf"]["cancelled"] = True
        self.assertIsNone(estimate_capacity(data)["scenarios"][0]["estimate"])
        data = run()
        data["perf"]["protocol"] = {}
        self.assertIsNone(estimate_capacity(data)["scenarios"][0]["estimate"])

    def test_monitoring_is_context_not_additional_measured_capacity(self):
        data = run()
        data["perf"]["telemetry"] = {"status": "collected", "api_clock_offset_seconds": -85,
            "alignment": {"timestamp_shift_seconds": 85}, "window": {"measurement_start": 1000},
            "metrics": [{"id": "output", "series": [{"values": [[900, 1000], [905, 1000], [910, 1000]]}]},
                        {"id": "waiting", "series": [{"values": [[915, 1], [920, 2], [925, 3]]}]}]}
        result = estimate_capacity(data)
        self.assertEqual(result["monitoring"]["baseline_output"], 1000)
        self.assertEqual(len(result["monitoring"]["warnings"]), 1)
        self.assertAlmostEqual(result["scenarios"][0]["estimate"]["output_tokens_per_second"], 140)
        data["perf"]["telemetry"]["alignment"]["timestamp_shift_seconds"] = 0
        self.assertIsNone(estimate_capacity(data)["monitoring"]["baseline_output"])

    def test_sweep_complete_counts_and_provider_lengths_are_supported(self):
        data = run()
        cell = point(context=32768, rate=100)
        cell.update(effort="default", output_tokens={"mean": 256}, prompt_tokens={"mean": 33000}, aggregate_tokens_per_sec=100)
        data["perf"] = {"schema_version": 4, "kind": "context_sweep", "protocol": {"revision": "context-sweep-v2"}, "cells": [cell]}
        result = estimate_capacity(data)
        self.assertIsNotNone(result["scenarios"][1]["estimate"])
        self.assertEqual(result["scenarios"][1]["evidence"]["mean_input"], 33000)

    def test_assumptions_reject_invalid_and_nonfinite_values(self):
        for kwargs in ({"peak_factor": 0}, {"busy_hours": 24}, {"busy_traffic_percent": 100},
                       {"headroom_percent": float("nan")}, {"chat_context": -1}, {"cache_mode": "unknown"},
                       {"chat_user_tokens_minute": 1e-310}):
            with self.assertRaises(ValidationError):
                CapacityAssumptions(**kwargs)


class CapacityRouteTests(unittest.TestCase):
    def setUp(self):
        self.data = run()
        self.row = {"id": 1, "status": "completed", "model_name": "fake", "perf_json": json.dumps(self.data["perf"])}
        app = FastAPI()
        app.include_router(runs.router)
        self.client = TestClient(app)

    def request(self, path, authenticated=True):
        with (patch.object(runs, "get_current_user", AsyncMock(return_value={"role": "user"} if authenticated else None)),
              patch.object(runs, "fetch_one", AsyncMock(return_value=copy.deepcopy(self.row))),
              patch("app.services.sweep_reports.hydrate_sweep", AsyncMock(side_effect=lambda _, perf: perf))):
            return self.client.get(path, follow_redirects=False)

    def test_auth_validation_custom_exports_and_benchmark_json(self):
        self.assertEqual(self.request("/runs/1/capacity", False).status_code, 302)
        self.assertEqual(self.request("/runs/1/capacity?peak_factor=0").status_code, 422)
        html = self.request("/runs/1/capacity?headroom_percent=50")
        self.assertEqual(html.status_code, 200)
        self.assertIn('value="50.0"', html.text)
        payload = self.request("/runs/1/capacity.json?headroom_percent=50").json()
        self.assertAlmostEqual(payload["scenarios"][0]["estimate"]["output_tokens_per_second"], 100)
        offline = self.request("/runs/1/capacity.html?headroom_percent=50")
        self.assertEqual(offline.status_code, 200)
        self.assertNotIn("<script", offline.text)
        self.assertNotIn("<form", offline.text)
        self.assertIn("50.0% headroom", offline.text)
        benchmark = self.request("/runs/1/performance.json").json()
        self.assertEqual(benchmark["capacity_estimate"]["revision"], REVISION)

    def test_capacity_is_embedded_in_shared_performance_report(self):
        view = performance_view([self.data], offline=True)
        html = templates.get_template("performance_report.html").render(performance=view, offline=True)
        self.assertIn("Serving capacity", html)
        self.assertIn("Active users @ tested rate", html)
        self.assertIn("Long-context agents", html)
        self.assertNotIn("/capacity\"", html)


if __name__ == "__main__":
    unittest.main()
