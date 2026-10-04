"""Independent finite-domain oracles, adaptive observations, and repair mutants."""

from copy import deepcopy
from fractions import Fraction
import itertools
import json
import unittest
from unittest.mock import patch

from app.benchmarking.interactive_tasks import run_interaction
from app.benchmarking.models import RequestMetrics, TokenUsage
from app.benchmarking.quality_execution import score_response
from app.benchmarking.reconstruction_cases import (
    make_reconstruction_tasks, ReconstructionEnvironment, replacement_source,
)

PROFILES = tuple(itertools.product((False, True), repeat=3))


def family_from_prompt(q):
    return "quantizer" if "Implement quantize(" in q.prompt else "cache"


def domain_from_prompt(q):
    marker = "DOMAIN=" if family_from_prompt(q) == "quantizer" else "ALPHABET="
    return json.JSONDecoder().raw_decode(q.prompt.split(marker, 1)[1])[0]


def input_domain(q):
    data = domain_from_prompt(q)
    if family_from_prompt(q) == "quantizer":
        return [list(values) for values in itertools.product(
            *(range(data[key][0], data[key][1] + 1) for key in ("value", "divisor", "offset", "cap")))]
    return [[list(command) for command in trace] for length in range(5)
            for trace in itertools.product(data, repeat=length)]


def oracle(family, profile, value):
    """No generator/reference interpreter or emitted replacement is called."""
    if family == "quantizer":
        x, divisor, adjustment, ceiling = value
        amount = Fraction(x + adjustment if profile[1] else x, divisor)
        integral = int(amount)
        if profile[0] and amount < integral:
            integral -= 1
        result = integral + (0 if profile[1] else adjustment)
        lower = abs(result) if profile[2] else max(-ceiling, result)
        return min(ceiling, lower)
    deadlines, durations, payloads, priorities = {}, {}, {}, {}
    now, serial, outputs = 0, 0, []
    for operation in value:
        serial += 1
        kind = operation[0]
        if kind == "tick":
            now += operation[1]
        alive = {key for key, deadline in deadlines.items()
                 if deadline >= now} if profile[0] else {
                     key for key, deadline in deadlines.items() if deadline > now}
        deadlines = {key: deadlines[key] for key in alive}
        if kind == "put":
            _, key, payload, duration = operation
            if key not in deadlines and len(deadlines) == 2:
                del deadlines[min(deadlines, key=lambda key: priorities[key])]
            deadlines[key], durations[key] = now + duration, duration
            payloads[key], priorities[key] = payload, serial
            outputs.append(None)
        elif kind == "get":
            key = operation[1]
            exists = key in deadlines
            outputs.append(payloads[key] if exists else None)
            if exists and profile[1]:
                deadlines[key] = now + durations[key]
            if exists and profile[2]:
                priorities[key] = serial
        else:
            outputs.append(now)
    return [outputs, [[key, payloads[key], deadlines[key]] for key in sorted(deadlines)]]


class ObservingAgent:
    """Learns only from the emitted prompt and observed probe outputs."""
    def __init__(self, q):
        self.q, self.probes = q, []
        self.family = family_from_prompt(q)
        self.domain = input_domain(q)

    def complete_messages(self, messages, **kwargs):
        replies = [json.loads(message["content"].removeprefix("Simulated tool result: "))
                   for message in messages[2:] if message["role"] == "user"]
        if replies and replies[-1].get("accepted"):
            action = {"done": True}
        else:
            possible = [profile for profile in PROFILES if all(
                oracle(self.family, profile, value) == reply["output"]
                for value, reply in zip(self.probes, replies))]
            if len(possible) == 1:
                action = {"tool": "submit", "args": {"code": replacement_source(self.family, possible[0])}}
            else:
                def split_cost(value):
                    counts = {}
                    for profile in possible:
                        label = json.dumps(oracle(self.family, profile, value))
                        counts[label] = counts.get(label, 0) + 1
                    return max(counts.values())

                probe = min(self.domain, key=split_cost)
                self.probes.append(probe)
                action = {"tool": "probe", "args": {"input": probe}}
        return json.dumps(action), TokenUsage(10, 5), RequestMetrics(finish_reason="stop", latency_ms=1)


class ScriptClient:
    def __init__(self, actions):
        self.actions = iter(actions)

    def complete_messages(self, *args, **kwargs):
        return json.dumps(next(self.actions)), TokenUsage(10, 5), RequestMetrics(finish_reason="stop")


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = [q for seed in (19, 23) for variant in (0, 1)
                         for q in make_reconstruction_tasks(seed, variant)]

    def test_exhaustive_oracle_and_fixture_groups(self):
        count = 0
        for q in self.questions:
            family, profile = family_from_prompt(q), tuple(q.interaction["profile"])
            domain = input_domain(q)
            self.assertEqual(len(q.expected), len(domain))
            self.assertEqual(len({fixture["id"] for fixture in q.expected}), len(domain))
            self.assertEqual(len(q.rubric), len(domain))
            baseline = (False, False, False) if family == "quantizer" else (False, False, True)
            for value, fixture in zip(domain, q.expected):
                self.assertEqual(fixture["args"], value if family == "quantizer" else [value])
                expected = oracle(family, profile, value)
                self.assertEqual(fixture["expected"], expected, (q.id, value))
                group = "regression" if expected == oracle(family, baseline, value) else "deviation"
                self.assertTrue(fixture["id"].startswith(group))
                count += 1
            for group in ("regression", "deviation"):
                weights = [item["weight"] for item in q.rubric if item["group"] == group]
                self.assertTrue(weights)
                self.assertAlmostEqual(sum(weights), 0.5)
            self.assertEqual(sum(a != b for a, b in zip(profile, baseline)) >= 2, True)
        self.assertEqual(count, 13544)

    def test_all_policy_pairs_observationally_identifiable(self):
        for q in self.questions[:2]:
            family, domain = family_from_prompt(q), input_domain(q)
            signatures = {profile: json.dumps([oracle(family, profile, value) for value in domain]) for profile in PROFILES}
            self.assertEqual(len(set(signatures.values())), 8)

    def test_prompt_only_adaptive_agent_and_frozen_feedback(self):
        profile_sets = {}
        for q in self.questions:
            agent = ObservingAgent(q)
            result = run_interaction(q, agent)
            self.assertTrue(result.passed, (q.id, result.detail))
            self.assertLessEqual(len(agent.probes), 12)
            self.assertEqual(result.evaluation.evaluator, "behavioral_reconstruction")
            self.assertEqual(result.diagnostics["probe_count"], len(agent.probes))
            self.assertEqual(result.diagnostics["unnecessary_calls"], 0)
            self.assertEqual(result.evaluation.criterion_achievement, 1)
            replies = [row["content"] for row in result.diagnostics["transcript"] if row["role"] == "tool"]
            self.assertFalse(any("expected" in row or "profile" in row for row in replies))
            self.assertEqual(set(replies[-1]), {"accepted", "next"})
            profile_sets.setdefault(family_from_prompt(q), set()).add(tuple(q.interaction["profile"]))
        self.assertEqual([len(profiles) for profiles in profile_sets.values()], [4, 4])

    def test_baseline_and_each_single_policy_mutant_fail(self):
        failures = 0
        for q in self.questions:
            family, correct = family_from_prompt(q), tuple(q.interaction["profile"])
            for profile in PROFILES:
                if profile == correct:
                    continue
                result = score_response(q, replacement_source(family, profile), TokenUsage(), RequestMetrics(finish_reason="stop"))
                self.assertFalse(result.passed, (q.id, profile))
                self.assertEqual(result.outcome, "task_failure")
                failures += 1
        self.assertEqual(failures, 56)

    def test_probe_types_submission_and_completion_gates(self):
        for q in self.questions[:2]:
            env = ReconstructionEnvironment(q)
            good = input_domain(q)[0]
            original = deepcopy(good)
            env.call("probe", {"input": good})
            self.assertEqual(good, original)
            bad = [True, 2, 0, 1] if env.family == "quantizer" else [["tick", True]]
            self.assertIn("error", env.call("probe", {"input": bad}))
            self.assertFalse(env.verdict(grade=False)["success"])
            source = replacement_source(env.family, q.interaction["profile"])
            no_probe = run_interaction(q, ScriptClient([{"tool": "submit", "args": {"code": source}}, {"done": True}]))
            self.assertFalse(no_probe.passed)
            self.assertEqual(no_probe.diagnostics["probe_count"], 0)
            with patch("app.benchmarking.quality_execution.score_response") as grader:
                cancelled = run_interaction(q, ScriptClient([]), cancelled=lambda: True)
                self.assertEqual(cancelled.outcome, "cancelled")
                grader.assert_not_called()
            empty = run_interaction(q, ScriptClient([{"done": True}]))
            self.assertFalse(empty.passed)

    def test_no_grading_before_done_or_on_transport_failure(self):
        q = self.questions[0]
        env = ReconstructionEnvironment(q)
        env.call("submit", {"code": "def quantize(*args): return 0"})
        with patch("app.benchmarking.quality_execution.score_response") as grader:
            env.verdict(grade=False)
            grader.assert_not_called()
        self.assertIn("error", env.call("probe", {"input": [0, 2, 0, 1]}))
        self.assertIn("action_after_submission", env.violations)

    def test_infrastructure_error_remains_unscored(self):
        q = self.questions[0]
        source = replacement_source("quantizer", q.interaction["profile"])
        actions = [{"tool": "probe", "args": {"input": [0, 2, 0, 1]}},
                   {"tool": "submit", "args": {"code": source}}, {"done": True}]
        with patch("app.benchmarking.code_runner.run_code_tests", return_value={"infrastructure_error": True, "error": "unavailable"}):
            result = run_interaction(q, ScriptClient(actions))
        self.assertEqual(result.outcome, "evaluator_error")
        self.assertFalse(result.is_scored)

    def test_probe_limit_and_no_submission_feedback(self):
        q = self.questions[0]
        env = ReconstructionEnvironment(q)
        for _ in range(12):
            self.assertIn("output", env.call("probe", {"input": [0, 2, 0, 1]}))
        self.assertIn("error", env.call("probe", {"input": [0, 2, 0, 1]}))
        self.assertEqual(len(env.probes), 12)
        self.assertEqual(env.verdict(grade=False)["unnecessary_calls"], 11)
        # A broken program is accepted as source and receives no grade early.
        self.assertTrue(env.call("submit", {"code": "def quantize(*args): return 999"})["accepted"])
        self.assertIn("error", env.call("submit", {"code": "def quantize(*args): return 0"}))

    def test_invalid_source_and_input_mutation_do_not_pass(self):
        q = self.questions[0]
        actions = [{"tool": "probe", "args": {"input": [0, 2, 0, 1]}},
                   {"tool": "submit", "args": {"code": "this is not Python"}}, {"done": True}]
        result = run_interaction(q, ScriptClient(actions))
        self.assertFalse(result.passed)
        self.assertEqual(result.outcome, "task_failure")
        self.assertEqual(result.evaluation.contract_score, 0)
        q = self.questions[1]
        source = replacement_source("cache", q.interaction["profile"])
        source = source.replace("    return [results,", "    if operations:\n        operations[0].append('changed')\n    return [results,")
        result = score_response(q, source, TokenUsage(), RequestMetrics(finish_reason="stop"))
        self.assertFalse(result.passed)
        self.assertTrue(any(c.reason_code == "input_mutation" for c in result.evaluation.criteria))


if __name__ == "__main__":
    unittest.main()
