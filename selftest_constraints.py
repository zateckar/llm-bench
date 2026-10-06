#!/usr/bin/env python3
"""Offline tests for the constraint diagnosis of users-stage levels (app/benchmarking/constraints.py)."""

import unittest

from app.benchmarking import constraints
from app.benchmarking.constraints import diagnose, kv_capacity_tokens, window_signals


def signals(running_at_ceiling=0.0, aligned=True, **values):
    """Window signals from ``key=(mean, max)`` pairs."""
    out = {"metrics": {k: {"mean": mean, "max": peak, "min": mean, "samples": 10} for k, (mean, peak) in values.items()},
           "aligned": aligned}
    if running_at_ceiling:
        out["running_at_ceiling"] = running_at_ceiling
    return out


def level(factor="ttft", *, errors=0, requests=100, reuse=0.9, client_hit=None, output=1000.0, prompt=1000.0,
          client_limited=False, hard_limit="", working=None):
    return {"users": 32, "passed": False, "cancelled": False, "client_limited": client_limited,
            "requests": requests, "hard_limit": hard_limit,
            "classes": {"chat": {"errors": errors, "incomplete": 0}},
            "limiting": {"factor": factor, "label": factor, "detail": ""},
            "expected_prefix_reuse": reuse, "cache_metrics": {"cache_hit_fraction": client_hit},
            "output_tokens_per_sec": output, "input_tokens_per_sec": prompt,
            "working_set_tokens": {"mean": working, "max": working} if working else None,
            "dispatch_lag": {"p99_ms": 250}}


CONFIG = [{"instance": "server:8000", "block_size": "16", "num_gpu_blocks": "10000", "cache_dtype": "auto",
           "enable_prefix_caching": "True", "num_cpu_blocks": "0"}]
QUIET = {"waiting": (0, 0), "running": (20, 24), "kv_cache": (0.4, 0.5), "prefix_hit": (0.88, 0.9),
         "output": (1000, 1100), "input": (1000, 1100)}


def titles(result):
    return [r["title"] for r in result["remedies"]]


class WindowTests(unittest.TestCase):
    def test_window_signals_combine_series_and_skip_the_rolling_minute(self):
        def metric(key, kind, combine, rows):
            return {"id": key, "kind": kind, "combine": combine, "status": "collected",
                    "series": [{"values": values} for values in rows]}

        telemetry = {"alignment": {"timestamp_shift_seconds": 10}, "api_clock_offset_seconds": -10, "metrics": [
            metric("kv_cache", "gauge", "max", [[[90, 0.5], [95, 0.93], [400, 1.0]], [[95, 0.7]]]),
            metric("waiting", "gauge", "sum", [[[95, 1], [250, 0]], [[95, 2], [250, 0]]]),
            metric("running", "gauge", "sum", [[[95, 8], [250, 4]]]),
            # A rate: samples within the first minute of a long window still describe the level before.
            metric("output", "rate", "sum", [[[95, 10], [200, 100], [250, 120]]]),
            {"id": "sm_active", "status": "missing", "series": []}]}
        result = window_signals(telemetry, 100, 300)
        self.assertTrue(result["aligned"])
        self.assertEqual(result["metrics"]["kv_cache"]["max"], 0.93)
        self.assertEqual(result["metrics"]["waiting"]["max"], 3)
        self.assertEqual(result["metrics"]["output"]["mean"], 110)
        self.assertNotIn("sm_active", result["metrics"])
        self.assertEqual(result["running_at_ceiling"], 0.5)
        self.assertIsNone(window_signals({"metrics": []}, 0, 1))
        self.assertIsNone(window_signals(telemetry, None, 1))
        self.assertFalse(window_signals({**telemetry, "api_clock_offset_seconds": 100}, 100, 300)["aligned"])

    def test_kv_capacity_from_cache_config(self):
        self.assertEqual(kv_capacity_tokens(CONFIG * 2), 320_000)
        self.assertIsNone(kv_capacity_tokens([{"block_size": "16"}]))
        self.assertIsNone(kv_capacity_tokens(None))


class DiagnosisTests(unittest.TestCase):
    def test_full_kv_cache(self):
        result = diagnose(level(), signals(**{**QUIET, "kv_cache": (0.9, 0.97), "preemptions": (0.1, 0.5),
                                                "waiting": (3, 9)}), CONFIG)
        self.assertEqual((result["constraint"], result["basis"]), ("kv_capacity", "server"))
        self.assertEqual(titles(result)[0], "FP8 KV cache")
        self.assertIn("KV cache offloading", titles(result))
        self.assertTrue(any("Preemptions" in e for e in result["evidence"]))
        fp8 = diagnose(level(), signals(**{**QUIET, "kv_cache": (0.9, 0.97)}),
                       [{**CONFIG[0], "cache_dtype": "fp8_e4m3", "num_cpu_blocks": "4096"}])
        self.assertNotIn("FP8 KV cache", titles(fp8))
        self.assertNotIn("KV cache offloading", titles(fp8))

    def test_evicted_histories_and_cache_misses(self):
        evicted = diagnose(level(working=400_000), signals(**{**QUIET, "kv_cache": (0.7, 0.85), "prefix_hit": (0.2, 0.3)}),
                           CONFIG)
        self.assertEqual(evicted["constraint"], "cache_eviction")
        self.assertEqual(titles(evicted)[0], "KV cache offloading")
        self.assertTrue(any("vs KV cache capacity 160,000 tokens" in e for e in evicted["evidence"]))
        missed = diagnose(level(), signals(**{**QUIET, "prefix_hit": (0.02, 0.03)}),
                          [{**CONFIG[0], "enable_prefix_caching": "False"}])
        self.assertEqual(missed["constraint"], "cache_miss")
        self.assertEqual(titles(missed)[0], "Enable prefix caching")
        # Sessions that allow little reuse cannot show lost histories.
        self.assertNotEqual(diagnose(level(reuse=0.1), signals(**{**QUIET, "prefix_hit": (0.0, 0.0)}))["constraint"],
                            "cache_miss")

    def test_queueing_kinds(self):
        slots = diagnose(level(), signals(running_at_ceiling=0.8, **{**QUIET, "waiting": (4, 10), "running": (60, 64)}))
        self.assertEqual(slots["constraint"], "batch_slots")
        self.assertEqual(titles(slots)[0], "Raise --max-num-seqs")
        busy = diagnose(level(), signals(**{**QUIET, "waiting": (4, 10), "sm_active": (0.9, 0.95)}))
        self.assertEqual(busy["constraint"], "compute")
        idle = diagnose(level(), signals(**{**QUIET, "waiting": (4, 10), "sm_active": (0.2, 0.3)}))
        self.assertEqual(idle["constraint"], "scheduler")
        unknown = diagnose(level(), signals(**{**QUIET, "waiting": (4, 10)}))
        self.assertEqual(unknown["constraint"], "queue")
        heavy = diagnose(level(output=100, prompt=50_000), signals(**{**QUIET, "waiting": (4, 10), "output": (100, 100),
                                                                    "input": (50_000, 60_000), "prefix_hit": (0.85, 0.9)}))
        self.assertEqual(heavy["constraint"], "prefill")

    def test_prefill_and_decode(self):
        prefill = diagnose(level(), signals(**{**QUIET, "input": (40_000, 40_000), "output": (500, 500),
                                                "prefix_hit": (0.85, 0.9)}))
        self.assertEqual(prefill["constraint"], "prefill")
        self.assertEqual(titles(prefill)[0], "Disaggregate prefill and decode")
        decode = diagnose(level("tpot"), signals(**{**QUIET, "input": (200, 200), "output": (2000, 2000)}))
        self.assertEqual(decode["constraint"], "decode")
        self.assertEqual(titles(decode)[0], "Speculative decoding")
        interference = diagnose(level("tpot"), signals(**{**QUIET, "input": (80_000, 80_000), "output": (1000, 1000),
                                                          "prefix_hit": (0.85, 0.9)}))
        self.assertEqual(interference["constraint"], "prefill_interference")
        self.assertEqual(titles(interference)[0], "Disaggregate prefill and decode")
        self.assertTrue(any("prompt-heavy" in e for e in interference["evidence"]))

    def test_client_basis_errors_and_alignment(self):
        self.assertEqual(diagnose(level())["constraint"], "prefill_or_queue")
        self.assertEqual(diagnose(level())["basis"], "client")
        self.assertEqual(diagnose(level("tpot", output=1000, prompt=100))["constraint"], "decode")
        # The client's cached-token counts reveal lost histories without server metrics.
        self.assertEqual(diagnose(level(client_hit=0.1))["constraint"], "cache_miss")
        self.assertEqual(diagnose(level(client_limited=True))["constraint"], "client")
        errors = diagnose(level("error", errors=50), signals(**QUIET))
        self.assertEqual(errors["constraint"], "errors")
        # Errors while the KV cache is full are a symptom: the cache is named, errors are also present.
        overloaded = diagnose(level("error", errors=50), signals(**{**QUIET, "kv_cache": (0.95, 0.99)}))
        self.assertEqual(overloaded["constraint"], "kv_capacity")
        self.assertIn("Request errors", [a["label"] for a in overloaded["also"]])
        unaligned = diagnose(level(), signals(aligned=False, **{**QUIET, "kv_cache": (0.95, 0.99)}))
        self.assertEqual((unaligned["basis"], unaligned["constraint"]), ("client", "prefill_or_queue"))

    def test_every_constraint_has_label_summary_and_remedies(self):
        for key in constraints.LABELS:
            self.assertIn(key, constraints.SUMMARY)
            self.assertIsInstance(constraints.remedies(key), list)
        self.assertTrue(constraints.needs_diagnosis({"passed": False}))
        self.assertTrue(constraints.needs_diagnosis({"passed": True, "hard_limit": "x"}))
        self.assertFalse(constraints.needs_diagnosis({"passed": True}))
        self.assertFalse(constraints.needs_diagnosis({"passed": False, "cancelled": True}))


if __name__ == "__main__":
    unittest.main()
