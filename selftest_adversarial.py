"""Independent algorithms, negative controls and evaluator-failure regressions."""

from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import heapq
import itertools
import json
import unittest
from unittest.mock import patch

from app.benchmarking.adversarial_cases import load_adversarial_questions
from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.models import Question, RequestMetrics, TokenUsage
from app.benchmarking.quality_execution import score_response
from app.benchmarking.quality_report import make_report, paired_comparison
from app.benchmarking.quality_suite import load_questions
from app.benchmarking.rigorous_cases import schema
from selftest_specialists import corruptions
from app.benchmarking.test_loader import SuiteError, _parse_question
from validate_suite import Report, check_json_match


def score(q, answer):
    return score_response(q, answer, TokenUsage(), RequestMetrics(finish_reason="stop"))


def subsets(items):
    return (subset for size in range(len(items) + 1) for subset in itertools.combinations(items, size))


class AdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = [q for seed in (19, 23) for variant in (0, 1)
                         for q in load_adversarial_questions(seed, variant)]

    def test_answers_and_corruptions(self):
        rejected = 0
        for q in self.questions:
            answer = q.expected["value"]
            good = score(q, json.dumps(answer))
            self.assertTrue(good.passed, (q.id, good.detail))
            self.assertAlmostEqual(good.evaluation.criterion_achievement, 1)
            self.assertFalse(score(q, "```json\n" + json.dumps(answer) + "\n```").passed)
            for wrong in corruptions(answer):
                self.assertFalse(score(q, json.dumps(wrong)).passed, (q.id, wrong))
                rejected += 1
            for key in answer:
                wrong = deepcopy(answer)
                del wrong[key]
                result = score(q, json.dumps(wrong))
                self.assertFalse(result.passed)
                self.assertLess(result.evaluation.criterion_achievement, 1)
        print(f"Rejected {rejected} corrupted v11 answers")

    def test_whole_suite_combined_structure_failures_stay_scored(self):
        for q in load_questions():
            if q.evaluator != "json_match" or q.interaction:
                continue
            for response in ('{}', '{"__unexpected__":true}'):
                result = score(q, response)
                self.assertTrue(result.is_scored, (q.id, result.detail))
                self.assertFalse(result.passed, q.id)
                ids = [c.criterion_id for c in result.evaluation.criteria]
                self.assertEqual(len(ids), len(set(ids)), q.id)

    def test_contracts_do_not_reveal_empty_arrays_or_disconnect_answers(self):
        self.assertEqual(schema([]), schema([1]))
        self.assertEqual(schema([]), schema([["T1", "T2"]]))
        contracts = defaultdict(list)
        for q in self.questions:
            line = q.prompt.split("Output contract (types, not answer values): ")[1].splitlines()[0]
            contracts[q.metadata["family"]].append(json.loads(line))
        for family, values in contracts.items():
            self.assertTrue(all(v == values[0] for v in values), family)

    def test_adaptive_budget_feasibility_oracle(self):
        # Feasibility at a budget, not the generator's cost-minimization DP.
        for q in self.questions:
            if "adaptive-minimax" not in q.id:
                continue
            data, want = q.metadata["oracle_input"], q.expected["value"]
            worlds, sensors = data["worlds"], data["sensors"]
            @lru_cache(None)
            def feasible(indices, available, budget):
                if budget < 0:
                    return False
                if budget < 0:
                    return False
                if len({worlds[i]["label"] for i in indices}) == 1:
                    return True
                for sensor in available:
                    cost = sensors[sensor]["cost"]
                    if cost > budget:
                        continue
                    buckets = defaultdict(list)
                    for i in indices:
                        buckets[worlds[i]["results"][sensor]].append(i)
                    if len(buckets) < 2:
                        continue
                    if all(feasible(tuple(bucket), tuple(s for s in available if s != sensor), budget - cost)
                           for bucket in buckets.values()):
                        return True
                return False
            all_indices, all_sensors = tuple(range(len(worlds))), tuple(range(len(sensors)))
            costs = sum(s["cost"] for s in sensors)
            minimum = next(b for b in range(costs + 1) if feasible(all_indices, all_sensors, b))
            self.assertEqual(want["adaptive_cost"], minimum)
            # Reconstruct the canonical tree with budget-feasibility queries.
            def policy(indices, available):
                labels = {worlds[i]["label"] for i in indices}
                if len(labels) == 1:
                    return next(iter(labels))
                budget = next(b for b in range(costs + 1) if feasible(indices, available, b))
                for sensor in available:
                    buckets = defaultdict(list)
                    for i in indices:
                        buckets[worlds[i]["results"][sensor]].append(i)
                    remainder = tuple(s for s in available if s != sensor)
                    if len(buckets) > 1 and all(feasible(tuple(v), remainder, budget - sensors[sensor]["cost"])
                                               for v in buckets.values()):
                        return sensors[sensor]["id"] + "{" + ",".join(
                            f"{outcome}:" + policy(tuple(buckets[outcome]), remainder)
                            for outcome in sorted(buckets)) + "}"
                self.fail("No optimal feasible query")
            self.assertEqual(want["policy"], policy(all_indices, all_sensors))
            fixed = []
            for selected in subsets(all_sensors):
                observations = defaultdict(set)
                for world in worlds:
                    observations[tuple(world["results"][s] for s in selected)].add(world["label"])
                if all(len(labels) == 1 for labels in observations.values()):
                    fixed.append((sum(sensors[s]["cost"] for s in selected), [sensors[s]["id"] for s in selected]))
            self.assertEqual((want["fixed_cost"], want["fixed_sensors"]), min(fixed))
            self.assertEqual(want["adaptivity_gain"], want["fixed_cost"] - minimum)

    def test_network_heap_oracle(self):
        for q in self.questions:
            if "network-interdiction" not in q.id:
                continue
            edges, want = q.metadata["oracle_input"]["edges"], q.expected["value"]
            def shortest(removed):
                adjacency = defaultdict(list)
                for e in edges:
                    if e["id"] not in removed:
                        adjacency[e["from"]].append((e["to"], e["cost"]))
                queue, visited = [(0, ["s"])], set()
                while queue:
                    cost, path = heapq.heappop(queue)
                    node = path[-1]
                    if node in visited:
                        continue
                    visited.add(node)
                    if node == "t":
                        return cost, path
                    for child, distance in adjacency[node]:
                        heapq.heappush(queue, (cost + distance, path + [child]))
                return None, []
            self.assertEqual(shortest(set()), (want["baseline_cost"], want["baseline_path"]))
            trials = [(list(s), shortest(set(s))) for s in subsets([e["id"] for e in edges]) if len(s) <= 2]
            def rank(trial):
                failed, (cost, _) = trial
                return (cost is not None, -(cost or 0), len(failed), failed)
            worst, (cost, path) = min(trials, key=rank)
            self.assertEqual((worst, cost, path), (want["worst_failure"], want["worst_cost"], want["worst_path"]))
            disconnected = [failed for failed, (cost, _) in trials if cost is None]
            cuts = [s for s in disconnected if not any(set(t) < set(s) for t in disconnected)]
            self.assertEqual(want["minimal_cuts"], sorted(cuts))
            repairs = [(cost, e, p) for e in worst for cost, p in [shortest(set(worst) - {e})] if cost is not None]
            self.assertEqual((want["restored_cost"], want["restore_edge"], want["restored_path"]),
                             min(repairs) if repairs else (None, None, []))

    def test_history_pairwise_and_serial_execution_oracle(self):
        classifications = set()
        for q in self.questions:
            if "transaction-consistency" not in q.id:
                continue
            history, want = q.metadata["oracle_input"]["history"], q.expected["value"]
            transactions = sorted({event["tx"] for event in history})
            commit = {e["tx"]: i for i, e in enumerate(history) if e["op"] == "c"}
            positions = {e["id"]: i for i, e in enumerate(history)}
            edges = set()
            # Group by resource and consider every ordered cross-tx pair.
            for resource in {e.get("key") for e in history} - {None}:
                events = [e for e in history if e.get("key") == resource]
                for a in events:
                    for b in events:
                        if (positions[a["id"]] < positions[b["id"]] and a["tx"] != b["tx"]
                            and (a["op"] == "w" or b["op"] == "w")):
                            edges.add((a["tx"], b["tx"]))
            reads = []
            for i, event in enumerate(history):
                if event["op"] == "r":
                    writes = [e for e in history[:i] if e["op"] == "w" and e["key"] == event["key"]]
                    reads.append([event["id"], writes[-1]["tx"] if writes else None])
            final = {key: next(e["tx"] for e in reversed(history) if e["op"] == "w" and e["key"] == key)
                     for key in {e["key"] for e in history if e["op"] == "w"}}
            conflict_orders, view_orders = [], []
            for order in itertools.permutations(transactions):
                if all(order.index(a) < order.index(b) for a, b in edges):
                    conflict_orders.append(list(order))
                serial = [e for tx in order for e in history if e["tx"] == tx]
                sources, last = {}, {}
                for e in serial:
                    if e["op"] == "r":
                        sources[e["id"]] = last.get(e["key"])
                    elif e["op"] == "w":
                        last[e["key"]] = e["tx"]
                if sources == dict(reads) and last == final:
                    view_orders.append(list(order))
            self.assertEqual(want["reads_from"], reads)
            self.assertEqual(want["final_writers"], final)
            self.assertEqual(want["conflict_edges"], [list(e) for e in sorted(edges)])
            self.assertEqual(want["conflict_orders"], conflict_orders)
            self.assertEqual(want["view_orders"], view_orders)
            external = [(next(e for e in history if e["id"] == ident), writer) for ident, writer in reads
                        if writer and writer != next(e["tx"] for e in history if e["id"] == ident)]
            self.assertEqual(want["recoverable"], all(commit[w] < commit[e["tx"]] for e, w in external))
            self.assertEqual(want["cascadeless"], all(commit[w] < positions[e["id"]] for e, w in external))
            strict = all(not any(e["op"] != "c" and e["tx"] != writer["tx"] and e.get("key") == writer["key"]
                                 for e in history[i + 1:commit[writer["tx"]]])
                         for i, writer in enumerate(history) if writer["op"] == "w")
            self.assertEqual(want["strict"], strict)
            classifications.add((bool(conflict_orders), bool(view_orders), want["recoverable"], strict))
        self.assertEqual(len(classifications), 4)

    def test_provenance_antichain_oracle(self):
        # Propagate minimal provenance sets through rules, without the
        # generator's enumeration/closure over every subset of base facts.
        for q in self.questions:
            if "grounded-proof-repair" not in q.id:
                continue
            data, want = q.metadata["oracle_input"], q.expected["value"]
            supports = defaultdict(set)
            for fact in data["facts"]:
                supports[fact["atom"]].add(frozenset([fact["id"]]))
            changed = True
            while changed:
                changed = False
                for rule in data["rules"]:
                    options = itertools.product(*(supports[p] for p in rule["if"]))
                    candidates = supports[rule["then"]] | {frozenset().union(*parts) for parts in options}
                    minimal = {s for s in candidates if not any(t < s for t in candidates)}
                    if minimal != supports[rule["then"]]:
                        supports[rule["then"]] = minimal
                        changed = True
            positive, negative = supports["+" + data["target"]], supports["-" + data["target"]]
            self.assertEqual(want["positive_supports"], sorted(sorted(s) for s in positive))
            self.assertEqual(want["negative_supports"], sorted(sorted(s) for s in negative))
            status = {(True, True): "both", (True, False): "true_only",
                      (False, True): "false_only", (False, False): "neither"}
            self.assertEqual(want["status"], status[bool(positive), bool(negative)])
            self.assertEqual(want["unfounded_cycle_derived"], any(supports[a] for a in data["unseeded_atoms"]))
            repairs = [sorted(s) for s in subsets([f["id"] for f in data["facts"]])
                       if all(set(s) & p for p in positive) and any(not set(s) & n for n in negative)]
            best_size = min(map(len, repairs)) if repairs else None
            self.assertEqual(want["minimum_retractions"], sorted(s for s in repairs if len(s) == best_size))

    def test_client_settings_prevent_invalid_paired_comparisons(self):
        q = self.questions[0]
        result = score(q, json.dumps(q.expected["value"]))
        config = ClientConfig("http://fake.invalid", "secret", "model")
        report = make_report([result], config)
        self.assertNotIn("secret", json.dumps(report))
        other = make_report([result], ClientConfig("http://fake.invalid", "secret", "model", timeout=240))
        self.assertFalse(paired_comparison(report, other)["compatible"])
        self.assertFalse(paired_comparison(report, make_report([result], config, max_concurrency=1))["compatible"])


class DiagnosticValidationTests(unittest.TestCase):
    def setUp(self):
        self.q = Question("q", "Test", "Return JSON", "json_match", {"value": {"a": 1}})

    def test_nonfinite_or_out_of_range_scores_are_unscored(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with patch.dict("app.benchmarking.quality_execution.EVALUATORS", json_match=lambda *a, **k: (value, "ok")):
                result = score(self.q, '{"a":1}')
            self.assertEqual(result.outcome, "evaluator_error")
            self.assertFalse(result.is_scored)

    def test_invalid_criteria_are_evaluator_errors(self):
        for criteria in ([{"id": "a", "earned": float("nan")}],
                         [{"id": "a", "status": "invalid"}], [{"id": "a"}, {"id": "a"}],
                         [{"id": "a", "earned": 2, "possible": 1}],
                         [{"id": "a", "status": "pass", "earned": 0}], ["invalid"], "invalid"):
            def evaluator(*args, _diagnostics, **kwargs):
                _diagnostics["criteria"] = criteria
                return 1, "ok"
            with patch.dict("app.benchmarking.quality_execution.EVALUATORS", json_match=evaluator):
                result = score(self.q, '{"a":1}')
            self.assertEqual(result.outcome, "evaluator_error", criteria)
            self.assertFalse(result.is_scored)

    def test_invalid_contract_score_is_an_evaluator_error(self):
        def evaluator(*args, _diagnostics, **kwargs):
            _diagnostics["contract_score"] = float("nan")
            return 1, "ok"
        with patch.dict("app.benchmarking.quality_execution.EVALUATORS", json_match=evaluator):
            result = score(self.q, '{"a":1}')
        self.assertEqual(result.outcome, "evaluator_error")

    def test_invalid_evaluator_detail_is_an_evaluator_error(self):
        with patch.dict("app.benchmarking.quality_execution.EVALUATORS", json_match=lambda *a, **k: (1, None)):
            result = score(self.q, '{"a":1}')
        self.assertEqual(result.outcome, "evaluator_error")
        self.assertFalse(result.is_scored)

    def test_missing_mandatory_rubric_credit_stays_in_denominator(self):
        self.q.rubric = [{"id": "json:a"}, {"id": "required-evidence"}]
        result = score(self.q, '{"a":1}')
        self.assertFalse(result.passed)
        self.assertEqual(result.evaluation.criterion_achievement, 0.5)

    def test_literal_root_field_does_not_block_unrelated_content(self):
        self.q.expected = {"value": {"root": {"x": 1}, "answer": 2}, "strict_json": True}
        for response in ('{"root":[],"answer":2}', '{"answer":2}', '{"answer":2,"extra":0}'):
            result = score(self.q, response)
            self.assertTrue(result.is_scored, result.detail)
            self.assertFalse(result.passed)
            self.assertEqual(result.evaluation.criterion_achievement, 1 / (3 if 'extra' in response else 2))
            self.assertEqual(next(c for c in result.evaluation.criteria if c.criterion_id == "json:answer").status, "pass")
            ids = [c.criterion_id for c in result.evaluation.criteria]
            self.assertEqual(len(ids), len(set(ids)))

    def test_root_scalar_with_extra_key_retains_content_credit(self):
        self.q.expected = {"value": {"root": 1, "answer": 2}, "strict_json": True}
        result = score(self.q, '{"root":1,"answer":2,"extra":0}')
        self.assertTrue(result.is_scored)
        self.assertFalse(result.passed)
        self.assertEqual(result.evaluation.criterion_achievement, 2 / 3)
        self.assertEqual(result.evaluation.contract_score, 0)

    def test_authoring_rejects_invalid_metadata_and_optional_fields(self):
        base = {"id": "q", "category": "Test", "prompt": "Return JSON", "evaluator": "json_match",
                "expected": {"value": {"answer": 1}}}
        for bad in ({"pass_threshold": True}, {"system_prompt": ["not text"]},
                    {"metadata": {"family": []}}, {"source": {}}, {"description": False}):
            with self.assertRaises(SuiteError):
                _parse_question({**base, **bad}, "fixture")

    def test_json_answer_keys_are_possible_and_have_unambiguous_paths(self):
        for value in ({"x": float("nan")}, {"x": float("inf")}, {1: "x"},
                      {"a.b": 1}, {"": 1}, {"x": object()}):
            report = Report()
            check_json_match(report, "fixture", {"value": value})
            self.assertTrue(report.errors, value)


if __name__ == "__main__":
    unittest.main()
