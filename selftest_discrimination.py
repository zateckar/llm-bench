"""Independent exhaustive oracles and adversarial controls for the new ladder."""

import copy
import itertools
import json
import unittest

from app.benchmarking import discrimination_cases as cases
from app.benchmarking.evaluators import EVALUATORS
from app.benchmarking.models import RequestMetrics, TokenUsage, Result
from app.benchmarking.quality_execution import score_response
from app.benchmarking.quality_report import result_record, summarize


def schedule(data):
    feasible = []
    for p in itertools.product(range(data["horizon"]), repeat=len(data["jobs"])):
        if any(p[i] < j["release"] or any(p[k] >= p[i] for k in j["after"])
               for i, j in enumerate(data["jobs"])):
            continue
        if any(sum(j["units"] for j, t in zip(data["jobs"], p) if t == slot) > data["capacity"]
               for slot in range(data["horizon"])):
            continue
        loss = sum(j["weight"] * max(0, p[i] + 1 - j["due"]) ** 2 for i, j in enumerate(data["jobs"]))
        feasible.append((loss, p))
    if not feasible:
        return {"minimum_cost": None, "optimal_count": 0, "slots": None}
    best, p = min(feasible)
    return {"minimum_cost": best, "optimal_count": sum(loss == best for loss, _ in feasible), "slots": list(p)}


def diagnosis(data):
    # Bottom-up table over explicit sets instead of recursive bit-mask DP.
    costs, roots = {}, {}
    for size in range(1, len(data["states"]) + 1):
        for states in itertools.combinations(data["states"], size):
            options = []
            if size == 1:
                costs[states] = 0
                continue
            for t in data["tests"]:
                branches = [tuple(s for s in states if t["positive"][s] == bit) for bit in (0, 1)]
                if any(not b or b == states or costs[b] is None for b in branches):
                    continue
                options.append((t["cost"] + max(costs[b] for b in branches), t["id"]))
            costs[states] = min((cost for cost, _ in options), default=None)
            roots[states] = sorted(t for cost, t in options if cost == costs[states])
    full = tuple(data["states"])
    cost = costs[full]
    return {"worst_cost": cost, "optimal_first_tests": roots.get(full, []),
            "within_budget": cost is not None and cost <= data["budget"]}


def language(data):
    accepted = []
    for word in itertools.product(data["alphabet"], repeat=data["length"]):
        state, total = data["initial"], 0
        for i, symbol in enumerate(word, 1):
            state, delta = data["transitions"][state][symbol]
            total += i * delta
        if state in data["accepting"] and total % data["modulus"] == data["checksum"]:
            accepted.append("".join(word))
    return {"count": len(accepted), "first": min(accepted, default=None), "last": max(accepted, default=None)}


def portfolio(data):
    feasible = []
    for size in range(len(data["items"]) + 1):
        for subset in itertools.combinations(range(len(data["items"])), size):
            chosen = set(subset)
            cost = sum(data["items"][i]["cost"] for i in chosen)
            if cost > data["budget"]:
                continue
            if any(a in chosen and b in chosen for a, b in data["conflicts"]):
                continue
            if any(b in chosen and a not in chosen for a, b in data["requires"]):
                continue
            profits = [sum(data["items"][i]["profit"][s] for i in chosen) for s in range(3)]
            feasible.append((subset, cost, profits))
    ideal = [max(row[2][s] for row in feasible) for s in range(3)]
    ranked = []
    for subset, cost, profits in feasible:
        regrets = [ideal[s] - profits[s] for s in range(3)]
        ranked.append((max(regrets), sum(regrets), cost, subset, profits, regrets))
    _, _, cost, subset, profits, regrets = min(ranked)
    return {"ideal_profits": ideal, "chosen": list(subset), "spend": cost, "profits": profits, "regrets": regrets}


def route(data):
    # Enumerate actual walks and every coupon position, independently of the
    # reference's state pruning and relaxation.
    candidates = []
    def visit(path, tolls):
        if path[-1] == data["end"] and all(i in path for i in range(data["required_mask"].bit_length())
                                           if data["required_mask"] & (1 << i)):
            for coupon in range(-1, len(tolls)):
                cost = sum(tolls) if coupon < 0 else sum(tolls) - tolls[coupon] + tolls[coupon] // 2
                candidates.append((cost, len(tolls), path, coupon))
        if len(tolls) == data["max_hops"]:
            return
        for a, b, toll in data["edges"]:
            if a == path[-1]:
                visit([*path, b], [*tolls, toll])
    visit([data["start"]], [])
    if not candidates:
        return None
    cost, hops, path, coupon = min(candidates)
    return {"cost": cost, "hops": hops, "path": path, "coupon": coupon}


def assignment(data):
    # Explicit bijection search, then select lexically among regret-optimal
    # assignments. No permutation helper or reference objective is reused.
    all_rows = []
    def visit(p, remaining):
        if not remaining:
            sums = [sum(m[i][col] for i, col in enumerate(p)) for m in data["matrices"]]
            all_rows.append((p, sums))
        for col in remaining:
            visit([*p, col], remaining - {col})
    visit([], set(range(len(data["matrices"][0]))))
    ideal = [min(row[s] for _, row in all_rows) for s in range(3)]
    max_regret = min(max(row[s] - ideal[s] for s in range(3)) for _, row in all_rows)
    left = [(p, row) for p, row in all_rows if max(row[s] - ideal[s] for s in range(3)) == max_regret]
    summed = min(sum(row[s] - ideal[s] for s in range(3)) for _, row in left)
    p, costs = min((p, row) for p, row in left if sum(row[s] - ideal[s] for s in range(3)) == summed)
    return {"assignment": p, "costs": costs, "regrets": [a - b for a, b in zip(costs, ideal)], "ideals": ideal}


ORACLES = {"resource-schedule": schedule, "adaptive-diagnosis": diagnosis,
           "checksum-language": language, "robust-portfolio": portfolio,
           "coupon-route": route, "robust-assignment": assignment}


class LadderTests(unittest.TestCase):
    def test_independent_oracles_and_corruptions(self):
        for seed, variant in itertools.product((19, 23), range(2)):
            for q in cases.make_discrimination_tasks(seed, variant):
                family = q.metadata["family"].removeprefix("Q16-")
                if q.evaluator == "code_exec":
                    for fixture in q.expected:
                        self.assertEqual(fixture["expected"], ORACLES[family](fixture["args"][0]), q.id)
                    if family == "coupon-route":
                        self.assertEqual(sum(f["expected"] is not None for f in q.expected), 27)
                    continue
                data = json.loads(q.prompt.split("\nINPUT=", 1)[1].split("\n", 1)[0])
                expected = ORACLES[family](data)
                self.assertEqual(expected, q.expected["value"], q.id)
                if family == "adaptive-diagnosis":
                    self.assertGreaterEqual(expected["worst_cost"], 6 + 2 * variant)
                correct = score_response(q, json.dumps(expected), TokenUsage(), RequestMetrics())
                self.assertTrue(correct.passed, (q.id, correct.detail))
                for field in expected:
                    altered = copy.deepcopy(expected)
                    altered.pop(field)
                    result = score_response(q, json.dumps(altered), TokenUsage(), RequestMetrics())
                    self.assertFalse(result.passed, (q.id, field))
                    self.assertLess(result.achievement_score, 1, (q.id, field))
                    altered = copy.deepcopy(expected)
                    value = altered[field]
                    altered[field] = ([*value, "extra"] if isinstance(value, list) else
                                      not value if isinstance(value, bool) else "wrong")
                    result = score_response(q, json.dumps(altered), TokenUsage(), RequestMetrics())
                    self.assertFalse(result.passed, (q.id, field))
                    self.assertLess(result.achievement_score, 1, (q.id, field))

    def test_code_reference_and_near_misses_execute(self):
        import inspect
        for q in [q for seed, variant in itertools.product((19, 23), range(2))
                  for q in cases.make_discrimination_tasks(seed, variant)]:
            if q.evaluator != "code_exec":
                continue
            name = "route_answer" if "coupon-route" in q.id else "assignment_answer"
            source = "import itertools\n" + inspect.getsource(getattr(cases, name)).replace(f"def {name}(", "def solve(")
            score, detail = EVALUATORS["code_exec"](source, q.expected, rubric=q.rubric)
            self.assertEqual(score, 1, (q.id, detail))
            for bad in ("def solve(data): return None", "def solve(data): return {}",
                        source.replace("toll // 2", "toll") if name == "route_answer" else
                        source.replace("max(regrets), sum(regrets)", "max(row), sum(row)")):
                score, detail = EVALUATORS["code_exec"](bad, q.expected)
                self.assertLess(score, 1, (q.id, detail))

    def test_adaptive_information_is_not_greedy_test_cost(self):
        data = {"states": [0, 1, 2, 3], "budget": 3, "tests": [
            {"id": "cheap", "cost": 1, "positive": [0, 0, 0, 1]},
            {"id": "split", "cost": 2, "positive": [0, 0, 1, 1]},
            {"id": "cross", "cost": 2, "positive": [0, 1, 0, 1]}]}
        self.assertEqual(cases.diagnosis_answer(data), diagnosis(data))
        self.assertFalse(cases.diagnosis_answer(data)["within_budget"])

    def test_ladder_coverage_and_full_success_are_separate(self):
        from app.templates_config import templates

        rows = []
        for seed, variant in itertools.product((19, 23), range(2)):
            for q in cases.make_discrimination_tasks(seed, variant):
                result = Result(q, "partial answer", .75, outcome="content_mismatch",
                                metrics=RequestMetrics(ok=not variant))
                rows.append(result_record(result))
        ladder = summarize(rows)["challenge"]
        self.assertEqual(ladder["tiers"]["core"],
                         {"planned": 12, "scored": 12, "achievement": .75, "full_pass": 0})
        self.assertEqual(ladder["tiers"]["extended"],
                         {"planned": 12, "scored": 0, "achievement": None, "full_pass": None})
        html = templates.get_template("quality_macros.html").module.capability_ladder(ladder)
        for text in ("capability-ladder-v1", "Full task success", "75.0%", "0.0%", "0/12", "n/a"):
            self.assertIn(text, html)


if __name__ == "__main__":
    unittest.main()
