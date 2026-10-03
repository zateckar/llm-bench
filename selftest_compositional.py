"""Independent prompt-derived v12 oracles, contrast checks and corruptions."""

from copy import deepcopy
import itertools
import json
import unittest

from compositional_cases import load_compositional_questions
from models import RequestMetrics, TokenUsage
from quality_execution import score_response
from selftest_specialists import corruptions


def score(q, answer):
    return score_response(q, json.dumps(answer), TokenUsage(), RequestMetrics(finish_reason="stop"))


def prompt_data(q):
    # Recompute from exactly what the model sees; don't trust oracle metadata.
    return json.loads(q.prompt.split("\nINPUT=", 1)[1].split("\n\nOutput contract", 1)[0])


def permutation_oracle(ops, capacity):
    """Full permutations, real-time check, then a separate queue interpreter."""
    valid = []
    for permutation in itertools.permutations(ops):
        if any(later["end"] < earlier["start"]
               for i, earlier in enumerate(permutation) for later in permutation[i + 1:]):
            continue
        contents = []
        for op in permutation:
            if op["kind"] == "take":
                observed = contents[0] if contents else None
                contents = contents[1:]
            elif len(contents) == capacity:
                observed = False
            else:
                observed = True
                contents += [op["value"]]
            if (observed != op["result"]) or (type(observed) is not type(op["result"])):
                break
        else:
            valid.append(([op["id"] for op in permutation], contents))
    return sorted(valid)


def knowledge_oracle(data):
    worlds = {w["id"]: w for w in data["worlds"]}
    relation = {a: {(i, j) for i, w in worlds.items() for j, v in worlds.items() if w[a] == v[a]}
                for a in "ABC"}

    def holds(formula, world, active):
        op = formula[0]
        if op == "atom":
            return worlds[world][formula[1]]
        if op == "not":
            return not holds(formula[1], world, active)
        if op == "and":
            return holds(formula[1], world, active) and holds(formula[2], world, active)
        if op == "K":
            return all(holds(formula[2], other, active) for other in active
                       if (world, other) in relation[formula[1]])
        raise AssertionError(op)

    active, stages = sorted(worlds), []
    for formula in data["announcements"]:
        active = [w for w in active if holds(formula, w, active)]
        stages.append(active)
    # All-pairs min-plus closure; the production solver uses single-source BFS.
    distance = {(i, j): 0 if i == j else 1 if (i, j) in relation["A"] | relation["B"] else 1000
                for i in active for j in active}
    for k in active:
        for i in active:
            for j in active:
                distance[i, j] = min(distance[i, j], distance[i, k] + distance[k, j])
    false = [w for w in active if not worlds[w]["p"]]
    remaining = {i: min((distance[i, j] for j in false), default=1000) for i in active}
    paths = []
    for i in active:
        path = []
        if remaining[i] < 1000:
            path = [i]
            while remaining[path[-1]]:
                path.append(min(j for j in active
                                if distance[path[-1], j] == 1 and remaining[j] == remaining[path[-1]] - 1))
        paths.append(path)
    return {"survivors_after_each": stages,
            "probe_worlds": [[w for w in active if holds(f, w, active)] for f in data["probes"]],
            "everyone_AB_p": [w for w in active if all(holds(["K", a, ["atom", "p"]], w, active) for a in "AB")],
            "common_AB_p": [w for w in active if remaining[w] == 1000],
            "counterexample_paths": paths}


class CompositionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = [q for seed in (19, 23) for variant in (0, 1)
                         for q in load_compositional_questions(seed, variant)]

    def test_queue_answers_from_permutation_oracle(self):
        for q in self.questions:
            if "queue-linearizability" not in q.id:
                continue
            data = prompt_data(q)
            ops, capacity = data["operations"], data["capacity"]
            orders = permutation_oracle(ops, capacity)
            repairs = []
            for size in range(len(ops) + 1):
                for removed in itertools.combinations(sorted(op["id"] for op in ops), size):
                    retained = [op for op in ops if op["id"] not in removed]
                    valid = permutation_oracle(retained, capacity)
                    if valid:
                        repairs.append({"removed": list(removed), "order": valid[0][0],
                                        "final_queue": valid[0][1], "count": len(valid)})
                if repairs:
                    break
            expected = {"linearizable": bool(orders), "linearization_count": len(orders),
                        "canonical_order": orders[0][0] if orders else [],
                        "possible_final_queues": [list(s) for s in sorted({tuple(s) for _, s in orders})],
                        "minimum_removals": size, "repairs": repairs}
            self.assertEqual(q.expected["value"], expected, q.id)

    def test_knowledge_answers_from_relation_and_distance_oracle(self):
        for q in self.questions:
            if "public-knowledge" in q.id:
                self.assertEqual(q.expected["value"], knowledge_oracle(prompt_data(q)), q.id)

    def test_correct_answers_and_individual_corruptions(self):
        rejected = 0
        for q in self.questions:
            answer = q.expected["value"]
            good = score(q, answer)
            self.assertTrue(good.passed, (q.id, good.detail))
            self.assertAlmostEqual(good.evaluation.criterion_achievement, 1)
            for wrong in corruptions(answer):
                result = score(q, wrong)
                self.assertTrue(result.is_scored, (q.id, result.detail))
                self.assertFalse(result.passed, (q.id, wrong))
                rejected += 1
            for key in answer:
                wrong = deepcopy(answer)
                del wrong[key]
                result = score(q, wrong)
                self.assertFalse(result.passed)
                # Each top-level requirement contributes exactly one equal share.
                self.assertAlmostEqual(result.evaluation.criterion_achievement, 1 - 1 / len(answer))
        print(f"Rejected {rejected} corrupted v12 answers")

    def test_contrasts_and_no_contract_leakage(self):
        contracts = {}
        for q in self.questions:
            contract = q.prompt.split("Output contract (types, not answer values): ")[1].splitlines()[0]
            family = q.metadata["family"]
            self.assertEqual(contract, contracts.setdefault(family, contract))
            answer = q.expected["value"]
            if "public-knowledge" in q.id:
                self.assertNotEqual(answer["everyone_AB_p"], answer["common_AB_p"])
                self.assertGreaterEqual(max(map(len, answer["counterexample_paths"])), 3)
                wrong = deepcopy(answer)
                wrong["common_AB_p"] = wrong["everyone_AB_p"]
                self.assertFalse(score(q, wrong).passed)
                stages = answer["survivors_after_each"]
                self.assertGreater(len(prompt_data(q)["worlds"]), len(stages[0]))
                self.assertGreater(len(stages[0]), len(stages[1]))
            else:
                self.assertEqual(answer["linearizable"], q.metadata["variant"] == 0)
                if not answer["linearizable"]:
                    self.assertGreater(answer["minimum_removals"], 1)
                    self.assertGreater(len(answer["repairs"]), 1)
                    wrong = deepcopy(answer)
                    wrong["repairs"] = wrong["repairs"][:1]
                    self.assertFalse(score(q, wrong).passed)

    def test_queue_pairs_change_exactly_one_observed_result(self):
        for seed in (19, 23):
            pair = [prompt_data(q) for q in self.questions
                    if q.metadata['seed'] == seed and 'queue-linearizability' in q.id]
            self.assertEqual(pair[0]['capacity'], pair[1]['capacity'])
            differences = []
            for a, b in zip(pair[0]['operations'], pair[1]['operations']):
                for key in a:
                    if a[key] != b[key]:
                        differences.append((a['id'], key))
            self.assertEqual(len(differences), 1)
            self.assertEqual(differences[0][1], 'result')


if __name__ == "__main__":
    unittest.main()
