#!/usr/bin/env python3
"""Offline tests for the users-at-SLO session engine (docs/design-session-capacity.md)."""

from contextlib import ExitStack
import random
import threading
import time
import unittest
from unittest.mock import Mock, patch

from app.benchmarking import session_load, session_workload as sw
from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.models import RequestMetrics, TokenUsage
from app.benchmarking.session_load import Record, analyse_level, cap_result, next_users

CONFIG = ClientConfig("http://users.invalid/v1", "unused", "fixture", timeout=5)
SLO = {"ttft_ms": 1000, "output_tokens_per_sec": 40}
TEST_MODEL = {"classes": [
    {"name": "chat", "weight": 40, "system_tokens": 64, "first_input": [[48, 100, 1]], "turn_input": [[48, 80, 1]],
     "output": [[10, 20, 1]], "think_seconds": [[0.001, 0.002, 1]], "turns": [[2, 4, 1]], "slo": SLO},
    {"name": "agent", "weight": 60, "system_tokens": 256, "first_input": [[100, 200, 1]],
     "turn_input": [[100, 400, 1]], "output": [[10, 20, 1]], "think_seconds": [[0.001, 0.002, 1]],
     "turns": [[5, 40, 1]], "slo": SLO},
]}


def settings(**kwargs):
    return sw.parse_settings({"preset": "custom", "model": TEST_MODEL, "warmup_seconds": 1, "measure_seconds": 2,
                              "start_users": 16, "max_users": 64, "saturation": False, **kwargs})


class FakeServer:
    """Output slows to 25 tok/s once more than ``limit`` requests are in flight; beyond ``hard``
    in flight it rejects requests as overloaded."""

    lock = threading.Lock()
    active = peak = 0
    limit = 20
    hard = None
    down = False
    contexts = []

    def __init__(self, config):
        self.config = config
        self.session = Mock()
        self.streaming_supported = True
        self._stream_usage_supported = True

    @classmethod
    def reset(cls, limit=20, down=False, hard=None):
        cls.active = cls.peak = 0
        cls.limit, cls.down, cls.hard, cls.contexts = limit, down, hard, []

    def complete(self, prompt, system=None, **kwargs):
        return "OK", TokenUsage(), RequestMetrics(latency_ms=5, ttft_ms=2, completion_tokens=2, finish_reason="stop",
                                                  streamed=True, stream_chunks=2, stream_span_ms=20)

    def complete_messages(self, messages, max_tokens=None, retries=None):
        cls = type(self)
        with cls.lock:
            cls.active += 1
            cls.peak = max(cls.peak, cls.active)
            load = cls.active
        try:
            time.sleep(0.02)
            tokens = sum(len(m["content"]) for m in messages) // 4
            cls.contexts.append(tokens)
            if cls.down:
                return "", TokenUsage(), RequestMetrics(ok=False, error="HTTP 503: down", latency_ms=20)
            if cls.hard and load > cls.hard:
                return "", TokenUsage(), RequestMetrics(ok=False, error="HTTP 503: overloaded", latency_ms=20)
            tpot = 10.0 if load <= cls.limit else 40.0
            return ("1, 2, 3", TokenUsage(tokens, max_tokens),
                    RequestMetrics(latency_ms=20 + tpot * max_tokens, ttft_ms=1 + tokens / 100,
                                   completion_tokens=max_tokens, prompt_tokens=tokens, finish_reason="length",
                                   streamed=True, stream_chunks=max_tokens, stream_span_ms=tpot * (max_tokens - 1)))
        finally:
            with cls.lock:
                cls.active -= 1


class ShortLevels(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for patcher in (patch.object(sw, "MIN_WARMUP_SECONDS", 1), patch.object(sw, "MIN_MEASURE_SECONDS", 1),
                        patch.object(session_load, "EARLY_MIN_SECONDS", 0.5)):
            stack.enter_context(patcher)
        FakeServer.reset()


class SettingsTests(ShortLevels):
    def test_presets_and_validation(self):
        mixed = sw.parse_settings(sw.default_settings())
        self.assertEqual((mixed.preset, [c.name for c in mixed.model.classes], mixed.model.shares()),
                         ("mixed", ["chat", "agent"], [0.4, 0.6]))
        self.assertEqual(mixed.model.classes[1].slo.tpot_ms, 25)
        self.assertEqual(sw.parse_settings({"preset": "agent"}).model.classes[0].name, "agent")
        changed = sw.parse_settings({**sw.default_settings(), "model": TEST_MODEL})
        self.assertEqual(changed.preset, "custom")
        for bad in ({"context_caps": [65536, 32768]}, {"context_caps": [512]}, {"start_users": 300, "max_users": 200},
                    {"measure_seconds": 0}, {"model": {"classes": []}},
                    {"model": {"classes": [{**TEST_MODEL["classes"][0], "turns": [[0, 2, 1]]}]}},
                    {"model": {"classes": [{**TEST_MODEL["classes"][0], "first_input": [[10, 20, 1]]}]}},
                    {"preset": "bogus"}):
            with self.subTest(bad), self.assertRaises(ValueError):
                sw.parse_settings({**sw.default_settings(), **bad})
        with self.assertRaises(ValueError):
            sw.check_fits(mixed.model, 4096)  # an agent's 8k system prompt does not fit
        self.assertEqual(sw.caps_for(mixed, 257_536), (32_768, 131_072, 257_536))
        self.assertEqual(sw.caps_for(mixed, 28_160), (28_160,))
        self.assertEqual(sw.caps_for(sw.parse_settings({**sw.default_settings(), "context_caps": [65536, 262144]}),
                                     131_072), (65_536,))

    def test_assignment_keeps_the_mix_in_every_prefix(self):
        model = sw.parse_settings(sw.default_settings()).model
        self.assertEqual(model.assign(10).count(0), 4)
        self.assertEqual(model.assign(7).count(1), 4)
        assigned = model.assign(100)
        for n in (5, 10, 50):
            self.assertLessEqual(abs(assigned[:n].count(0) - 0.4 * n), 1)

    def test_plans_fit_the_cap_and_repeat_with_the_seed(self):
        model = sw.parse_settings(sw.default_settings()).model
        for cap in (32_768, 131_072):
            for user in range(20):
                plan = sw.plan_session(model, 1, cap, sw.user_rng(0, cap, 20, user))
                used = model.classes[1].system_tokens + sw.MESSAGE_FRAMING + sum(
                    t.input_tokens + t.output_tokens + 2 * sw.MESSAGE_FRAMING for t in plan.turns)
                self.assertLessEqual(used, cap)
        a = sw.starting_plan(model, 1, 131_072, sw.user_rng(3, 131_072, 8, 5))
        b = sw.starting_plan(model, 1, 131_072, sw.user_rng(3, 131_072, 8, 5))
        self.assertEqual(a, b)
        self.assertLess(a.start, len(a.turns))
        # Agent sessions in progress are long: most are well into their history.
        starts = [sw.starting_plan(model, 1, 262_144, random.Random(i)) for i in range(200)]
        self.assertGreater(sum(p.start for p in starts) / len(starts), 10)

    def test_prompts_have_exact_reference_lengths(self):
        model = sw.parse_settings(sw.default_settings()).model
        prompts = sw.SessionPrompts(model, "nonce")
        self.assertEqual(prompts.count(prompts.system(1)[0]["content"]), 8000)
        for n in (48, 333, 16000):
            self.assertEqual(prompts.count(prompts.user("0-1-0", 7, n)), n)
            self.assertEqual(prompts.count(prompts.assistant(n)), n)
        plan = sw.SessionPlan(1, (sw.Turn(500, 50, 1), sw.Turn(700, 60, 1), sw.Turn(900, 70, 1)), 2)
        history = prompts.history(plan, "0-1-0")
        self.assertEqual([m["role"] for m in history], ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual(sum(prompts.count(m["content"]) for m in history), 8000 + 500 + 50 + 700 + 60)


class SearchTests(unittest.TestCase):
    def search(self, capacity, *, start=8, maximum=256, failed=(), prior=None):
        passed, fails, tried = [], list(failed), []
        while True:
            users = prior if prior and not tried else next_users(passed, fails, start=start, maximum=maximum,
                                                                 resolution=0.1)
            if users is None:
                return tried, max(passed, default=0)
            tried.append(users)
            (passed if users <= capacity else fails).append(users)

    def test_doubling_then_bisection(self):
        self.assertEqual(self.search(52), ([8, 16, 32, 64, 48, 56, 52], 52))
        self.assertEqual(self.search(1000, maximum=100), ([8, 16, 32, 64, 100], 100))
        self.assertEqual(self.search(0), ([8, 4, 2, 1], 0))
        self.assertEqual(self.search(5), ([8, 4, 6, 5], 5))

    def test_a_larger_cap_starts_at_the_previous_result_and_never_retries_failures(self):
        tried, users = self.search(40, failed=[56], prior=52)
        self.assertEqual(tried[0], 52)
        self.assertTrue(all(u < 56 for u in tried))
        self.assertTrue(36 <= users <= 40, (tried, users))
        tried, users = self.search(60, failed=[56], prior=52)
        self.assertEqual((tried, users), ([52], 52))

    def test_cap_results(self):
        def level(users, passed, **extra):
            return {"users": users, "passed": passed, "cancelled": False, "client_limited": False,
                    "limiting": None if passed else {"factor": "tpot", "label": "x", "detail": ""},
                    "attainment": 1.0, "requests": 10, "context_tokens": {}, "output_tokens_per_sec": 1,
                    "classes": {}, **extra}

        entry = {"levels": [level(8, True), level(16, False)]}
        self.assertEqual({k: cap_result(entry)[k] for k in ("users", "status", "first_failed", "inherited")},
                         {"users": 8, "status": "established", "first_failed": 16, "inherited": False})
        self.assertEqual(cap_result({"levels": [level(64, True)]})["status"], "lower_bound")
        self.assertEqual(cap_result({"levels": [level(1, False)]})["status"], "not_met")
        inherited = cap_result({"levels": [level(52, True)], "inherited_failure": 56})
        self.assertEqual((inherited["status"], inherited["first_failed"], inherited["inherited"],
                          inherited["limiting"]["factor"]), ("established", 56, True, "inherited"))
        self.assertEqual(cap_result({"levels": [], "inferred_not_met": True, "inherited_failure": 1})["status"],
                         "not_met")
        self.assertEqual(cap_result({"levels": []})["status"], "not_measured")
        limited = cap_result({"levels": [level(8, True), level(16, False, client_limited=True)]})
        self.assertEqual((limited["status"], limited["client_limited"]), ("lower_bound", True))


def record(klass, sent, *, ttft=100.0, tpot=10.0, ok=True, first=False, lag=0.0, context=1000):
    metrics = RequestMetrics(ok=ok, error="" if ok else "HTTP 500: boom", ttft_ms=ttft, completion_tokens=20,
                             prompt_tokens=context, finish_reason="length", streamed=True, stream_chunks=20,
                             stream_span_ms=tpot * 19, latency_ms=ttft + tpot * 19)
    return Record(0, klass, sent, lag, context, 20, 1, metrics, first)


class AnalysisTests(ShortLevels):
    def test_level_attainment_misses_and_limiting_factor(self):
        s = settings()
        records = [record(0, 1.5 + i / 100) for i in range(20)]
        records += [record(1, 1.5 + i / 100, ttft=960, lag=50) for i in range(18)]  # 1,010 ms with lag
        records += [record(1, 1.6, ok=False), record(1, 1.7)]
        records += [record(1, 0.5, ttft=99_999), record(1, 1.8, ttft=99_999, first=True), record(1, 9.0, tpot=99)]
        level = analyse_level(s.model, s, 4096, 10, records, window=(1.0, 3.0), started_at=0, ended_at=4)
        chat, agent = level["classes"]["chat"], level["classes"]["agent"]
        self.assertEqual((chat["requests"], chat["attainment"], chat["users"]), (20, 1.0, 4))
        self.assertEqual((agent["requests"], agent["slo_met"], agent["errors"]), (20, 1, 1))
        self.assertEqual(agent["failure_reasons"], {"ttft": 18, "error": 1})
        self.assertFalse(level["passed"])
        self.assertEqual(level["limiting"]["factor"], "ttft")
        self.assertIn("agent: 5% met the SLO; first token p95 1.0 s vs 1 s", level["limiting"]["detail"])
        self.assertTrue(level["client_limited"] is False)
        self.assertAlmostEqual(agent["output_speed"]["p05"], 100)
        self.assertEqual(level["samples"]["c"].count(1), 20)
        self.assertEqual((level["window_started_at"], level["window_ended_at"]), (1.0, 3.0))
        self.assertEqual(level["expected_prefix_reuse"], 0)
        self.assertEqual(level["ttft"]["p50_ms"], 100)
        slow = [record(k, 1.5, tpot=30) for k in (0, 1) for _ in range(20)]
        level = analyse_level(s.model, s, 4096, 10, slow, window=(1.0, 3.0), started_at=0, ended_at=4)
        self.assertEqual(level["limiting"]["factor"], "tpot")
        lagging = [record(k, 1.5, lag=500, ttft=1) for k in (0, 1) for _ in range(20)]
        level = analyse_level(s.model, s, 4096, 10, lagging, window=(1.0, 3.0), started_at=0, ended_at=4)
        self.assertEqual((level["client_limited"], level["limiting"]["factor"]), (True, "client"))

    def test_expected_reuse_counts_the_history_before_each_turn(self):
        s = settings()
        records = [record(0, 1.5, context=1000) for _ in range(10)]
        for r in records:
            r.reusable = 800
        level = analyse_level(s.model, s, 4096, 4, records, window=(1.0, 3.0), started_at=100, ended_at=104,
                              working_set={"mean": 5000, "max": 6000})
        self.assertAlmostEqual(level["expected_prefix_reuse"], 0.8)
        self.assertEqual(level["working_set_tokens"]["max"], 6000)
        self.assertEqual((level["window_started_at"], level["window_ended_at"]), (101.0, 103.0))


def level_with(users, *, output=100.0, ttft_p50=100.0, ttft_p95=500.0, speed=50.0, errors=0, requests=100,
               phase="slo", passed=False, **extra):
    """A minimal analysed level for hard-limit and saturation rules."""
    klass = {"requests": requests, "errors": errors, "incomplete": 0, "slo": SLO,
             "ttft": {"p95_ms": ttft_p95}, "output_speed": {"p50": speed, "p05": speed}}
    return {"users": users, "passed": passed, "cancelled": False, "client_limited": False, "requests": requests,
            "classes": {"chat": klass}, "output_tokens_per_sec": output, "ttft": {"p50_ms": ttft_p50},
            "warmup_complete": True, "phase": phase, "early_stop": "", "attainment": 0.5,
            "limiting": None if passed else {"factor": "ttft", "label": "x", "detail": ""},
            "context_tokens": {}, **extra}


class SaturationRuleTests(unittest.TestCase):
    def test_hard_limits(self):
        hard = session_load.hard_limit
        self.assertEqual(hard(level_with(8)), "")
        self.assertIn("failed or were incomplete", hard(level_with(8, errors=11)))
        self.assertEqual(hard(level_with(8, errors=10)), "")
        self.assertIn("over 10× the 1 s target", hard(level_with(8, ttft_p95=10_001)))
        self.assertIn("under a 10th of the 40 tok/s", hard(level_with(8, speed=3.9)))
        self.assertIn("first answer", hard(level_with(8, warmup_complete=False)))
        self.assertEqual(hard(level_with(8, requests=0)), "No request was measured")
        # Throughput plateau: barely more output with more users while first tokens grow.
        previous = level_with(16, output=1000, ttft_p50=200)
        self.assertIn("Throughput stopped growing", hard(level_with(32, output=1050, ttft_p50=300), previous))
        self.assertEqual(hard(level_with(32, output=1200, ttft_p50=300), previous), "")
        self.assertEqual(hard(level_with(32, output=1050, ttft_p50=250), previous), "")
        self.assertEqual(hard(level_with(8, cancelled=True, errors=50)), "")
        self.assertEqual(hard(level_with(64, phase="saturation", early_stop="Stopped early: x")), "Stopped early: x")

    def test_saturation_search_doubles_until_a_limit(self):
        entry = {"levels": [level_with(8, passed=True), level_with(16, output=200)], "saturation_search": True}
        session_load.judge_limits(entry)
        self.assertEqual(session_load.next_saturation(entry, 1024), 32)
        self.assertEqual(session_load.next_saturation(entry, 20), 20)
        self.assertIsNone(session_load.next_saturation(entry, 16))
        self.assertEqual(session_load.saturation_result(entry)["status"], "not_reached")
        # A smaller cap was exhausted at 30 users: counts at or above it are not measured.
        self.assertIsNone(session_load.next_saturation({**entry, "inherited_saturation": 30}, 1024))
        inherited = session_load.saturation_result({**entry, "inherited_saturation": 30})
        self.assertEqual((inherited["status"], inherited["users"], inherited["below"]), ("inherited", 30, 16))
        entry["levels"].append(level_with(32, output=210, ttft_p50=400, phase="saturation"))
        session_load.judge_limits(entry)
        self.assertIn("Throughput stopped growing", entry["levels"][-1]["hard_limit"])
        self.assertIsNone(session_load.next_saturation(entry, 1024))
        result = session_load.saturation_result(entry)
        self.assertEqual((result["status"], result["users"], result["below"], result["peak_users"]),
                         ("reached", 32, 16, 32))
        self.assertIsNone(session_load.saturation_result({"levels": []}))
        limited = {"levels": [level_with(8, passed=True), level_with(16, client_limited=True)], "saturation_search": True}
        session_load.judge_limits(limited)
        self.assertIsNone(session_load.next_saturation(limited, 1024))
        self.assertEqual(session_load.saturation_result(limited)["status"], "client_limited")


class EngineTests(ShortLevels):
    def run_test(self, **kwargs):
        levels = []
        report = session_load.run_session_test(CONFIG, settings(**kwargs), 8192, client_factory=FakeServer,
                                               on_level=lambda r: levels.append(len(r.caps[-1]["levels"])))
        return report, levels

    def test_search_finds_the_users_the_fake_server_serves(self):
        report, saved = self.run_test(context_caps=[2048, 8192])
        data = report.to_dict()
        self.assertEqual((data["schema_version"], data["kind"], data["protocol"]["revision"]), (7, "sessions", "sessions-v2"))
        first, second = data["caps"]
        tried = [lv["users"] for lv in first["levels"]]
        self.assertEqual(tried[:2], [16, 32])
        result = first["result"]
        self.assertEqual(result["status"], "established", tried)
        self.assertTrue(16 <= result["users"] < 32, tried)
        self.assertEqual(result["limiting"]["factor"], "tpot")
        self.assertLessEqual(FakeServer.peak, 64)
        # The previous result is a starting hint; the larger cap measures its own failure.
        self.assertEqual(second["levels"][0]["users"], result["users"])
        self.assertTrue(any(lv["users"] >= result["first_failed"] for lv in second["levels"]))
        self.assertIsNone(second["inherited_failure"])
        self.assertEqual(data["summary"]["headline"]["context_cap"], 8192)
        self.assertEqual(len(saved), len(first["levels"]) + len(second["levels"]))
        # Sessions grow: agents at the larger cap send longer histories.
        self.assertGreater(second["levels"][0]["context_tokens"]["p95"], first["levels"][0]["context_tokens"]["p95"])
        self.assertLessEqual(max(FakeServer.contexts), 8192)
        failing = next(lv for lv in first["levels"] if not lv["passed"])
        self.assertLess(failing["attainment"], settings().attainment)
        # Without telemetry the constraint comes from the client: output speed misses on a prompt-heavy load.
        self.assertEqual(result["constraint"]["basis"], "client")
        self.assertIn(result["constraint"]["constraint"], ("decode", "prefill_interference"))
        self.assertIsNone(first["saturation"])
        # Histories make most of each prompt reusable; the working set is the users' histories.
        self.assertGreater(failing["expected_prefix_reuse"], 0.5)
        self.assertGreater(failing["working_set_tokens"]["mean"], 0)

    def test_saturation_search_stops_at_the_hard_limit(self):
        FakeServer.reset(limit=20, hard=40)
        report, _ = self.run_test(context_caps=[2048, 8192], saturation=True, saturation_max_users=256)
        first, second = report.to_dict()["caps"]
        tried = [(lv["users"], lv["phase"]) for lv in first["levels"]]
        saturation = first["saturation"]
        self.assertEqual(saturation["status"], "reached", tried)
        self.assertEqual(saturation["users"], 64, tried)
        self.assertIn("failed or were incomplete", saturation["reason"])
        self.assertEqual(saturation["constraint"]["constraint"], "errors")
        self.assertEqual([u for u, phase in tried if phase == "saturation"], [64])
        self.assertLessEqual(FakeServer.peak, 128)
        # The larger cap independently measures its saturation limit.
        self.assertIsNone(second["inherited_saturation"])
        self.assertTrue(any(lv["phase"] == "saturation" for lv in second["levels"]))
        self.assertEqual(second["saturation"]["status"], "reached")
        self.assertLessEqual(second["saturation"]["users"], 128)
        results = report.to_dict()["summary"]["results"]
        self.assertEqual(results[0]["saturation"]["users"], 64)

        # With telemetry over the level windows, the diagnosis moves to the server's signals.
        # Dense samples: test levels stop early after well under a second.
        start = int(first["levels"][0]["started_at"]) - 5
        stamps = [start + i / 20 for i in range((int(time.time()) + 5 - start) * 20)]
        telemetry = {"alignment": {"timestamp_shift_seconds": 0}, "api_clock_offset_seconds": 0,
                     "cache_config": [{"block_size": "16", "num_gpu_blocks": "100", "cache_dtype": "auto"}],
                     "metrics": [{"id": "kv_cache", "kind": "gauge", "combine": "max", "status": "collected",
                                  "series": [{"values": [[t, 0.97] for t in stamps]}]}]}
        report.attach_telemetry(telemetry)
        data = report.to_dict()
        first = data["caps"][0]
        self.assertEqual(first["saturation"]["constraint"]["constraint"], "kv_capacity")
        self.assertEqual(first["saturation"]["constraint"]["basis"], "server")
        self.assertIn("Request errors", [a["label"] for a in first["saturation"]["constraint"]["also"]])
        self.assertEqual(first["result"]["constraint"]["constraint"], "kv_capacity")
        self.assertEqual(first["levels"][0]["server"]["metrics"]["kv_cache"]["max"], 0.97)

        from app.services.session_views import session_views
        from app.templates_config import templates
        views = session_views([{"label": "#1", "perf": data}])
        view = views["views"][0]
        self.assertEqual(view["results"][0]["saturation"], "64")
        self.assertEqual(view["diagnoses"][0]["items"][0]["label"], "KV cache capacity")
        html = templates.env.get_template("performance_sessions.html").render(performance={"session_views": views})
        for text in ("What limits this deployment", "Exhausted at", "FP8 KV cache", "Saturation",
                     "Output throughput by simulated users", "KV max 97.0%"):
            self.assertIn(text, html)

    def test_endpoint_down_stops_the_stage(self):
        FakeServer.reset(down=True)
        report, _ = self.run_test(context_caps=[2048, 8192])
        self.assertIn("No request completed with 16 users", report.stop_reason)
        self.assertEqual(len(report.caps), 1)

    def test_cancel_keeps_finished_levels(self):
        calls = []

        def cancelled():
            calls.append(1)
            return len(calls) > 4  # two warmup checks, then about a second into the first level

        report = session_load.run_session_test(CONFIG, settings(context_caps=[2048]), 8192,
                                               client_factory=FakeServer, cancelled=cancelled)
        self.assertTrue(report.cancelled)
        self.assertTrue(report.caps[0]["levels"][-1]["cancelled"])
        self.assertNotEqual(report.caps[0]["result"]["status"], "established")

    def test_view_renders_results_and_levels(self):
        from app.services.session_views import session_views
        from app.templates_config import templates

        report, _ = self.run_test(context_caps=[2048], max_users=16)
        views = session_views([{"label": "#1", "perf": report.to_dict()}, {"label": "#2", "perf": report.to_dict()}])
        view = views["views"][0]
        self.assertEqual(view["results"][0]["users"], "≥ 16")
        self.assertEqual(view["results"][0]["status"], "Lower bound")
        self.assertEqual(views["comparison"]["rows"][0]["cells"][0]["users"], "≥ 16")
        html = templates.env.get_template("performance_sessions.html").render(performance={"session_views": views})
        for text in ("Users at SLO", "Users at SLO by context cap", "Measured levels", "User model",
                     "Users at SLO comparison", "2k"):
            self.assertIn(text, html)


if __name__ == "__main__":
    unittest.main()
