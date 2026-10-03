"""Independent prompt-derived v14 oracles and repeated-review regressions."""

from copy import deepcopy
from fractions import Fraction
import itertools
import json
import sqlite3
import unittest

from evaluators import values_equal
from models import Question, RequestMetrics, TokenUsage
from quality_execution import score_response
from quality_suite import fingerprint
from reasoning_cases import load_reasoning_questions
from rigorous_cases import balanced_rubric, load_new_questions
from selftest_specialists import corruptions
from validate_suite import Report, check_json_match


def score(q, answer):
    return score_response(q, json.dumps(answer), TokenUsage(), RequestMetrics(finish_reason="stop"))


def data_from(q):
    return json.loads(q.prompt.split("\nINPUT=", 1)[1].split("\n\nOutput contract", 1)[0])


def probability_oracle(data):
    # Integer-weight table over the FULL past/future joint distribution; no
    # posterior-predictive shortcut or generator helper is reused here.
    mass = [Fraction(0) for _ in range(8)]
    for z in range(2):
        ws = data["weights"][z]
        totals = [0] * 8
        for a, b, c, d, e in itertools.product(range(1, 6), repeat=5):
            if a + b + c < data["threshold"] or a == c or (a + 2 * b) % data["audit_modulus"]:
                continue
            weight = ws[a - 1] * ws[b - 1] * ws[c - 1] * ws[d - 1] * ws[e - 1]
            facts = [1, z, a != b and b != c, a + b + c,
                     d + e >= data["future_threshold"], d * e, d, e]
            for i, fact in enumerate(facts):
                totals[i] += weight * fact
        for i, total in enumerate(totals):
            mass[i] += Fraction(total * data["prior_weights"][z],
                                sum(data["prior_weights"]) * sum(ws) ** 5)
    values = [mass[0]] + [m / mass[0] for m in mass[1:5]] + [
        mass[5] / mass[0] - mass[6] * mass[7] / mass[0] ** 2]
    return dict(zip(("report_probability", "posterior_regime_1", "all_distinct_given_report",
                     "expected_sum_given_report", "future_sum_probability", "future_covariance"),
                    ([v.numerator, v.denominator] for v in values)))


def retrieval_oracle(data):
    db = sqlite3.connect(":memory:")
    try:
        db.execute("CREATE TABLE records(id TEXT, key TEXT, revision INT, active INT, kind TEXT, value TEXT)")
        db.executemany("INSERT INTO records VALUES (:id,:key,:revision,:active,:kind,:value)", data["records"])
        selected = {row[1]: row for row in db.execute(
            "SELECT id,key,revision,active,kind,value FROM "
            "(SELECT *,row_number() OVER (PARTITION BY key ORDER BY revision DESC,id DESC) AS n "
            "FROM records) WHERE n=1")}

        def visit(key, visited):
            if key in visited:
                return [], [], "cycle", None
            row = selected.get(key)
            if row is None:
                return [key], [], "missing", None
            if not row[3]:
                return [key], [row[0]], "deleted", None
            if row[4] == "result":
                return [key], [row[0]], "resolved", row[5]
            path, evidence, status, result = visit(row[5], visited | {key})
            return [key] + path, [row[0]] + evidence, status, result

        return {"queries": [dict(zip(("path", "evidence_ids", "status", "result"), visit(k, set())))
                            for k in data["starts"]]}
    finally:
        db.close()


def grid_oracle(data):
    # Assign entity labels TO positions, pruning completed relations at each
    # group. The generator instead assigns positions to entities exhaustively.
    def valid(clue, assignment):
        if clue["a"] not in assignment or clue["b"] not in assignment:
            return True
        difference = assignment[clue["b"]] - assignment[clue["a"]]
        if clue["op"] == "before":
            return difference > 0
        if clue["op"] == "next":
            return difference == 1
        if clue["op"] == "same":
            return difference == 0
        if clue["op"] == "adjacent":
            return difference in (-1, 1)
        if clue["op"] == "apart":
            return difference < -1 or difference > 1
        raise AssertionError(clue)

    output = []
    labels = sorted(sum(data["groups"], []))
    for removed in data["omit"]:
        clues = [c for c in data["clues"] if c["id"] not in removed]
        partial = [{}]
        for group in data["groups"]:
            expanded = []
            for known in partial:
                for order in itertools.permutations(group):
                    candidate = {**known, **{label: position for position, label in enumerate(order, 1)}}
                    if all(valid(c, candidate) for c in clues):
                        expanded.append(candidate)
            partial = expanded
        rows = sorted([w[k] for k in labels] for w in partial)
        output.append({"count": len(rows),
                       "domains": {k: sorted({w[k] for w in partial}) for k in labels},
                       "first_two": rows[:2]})
    return {"scenarios": output}


def code_oracle(q, data):
    source = "def run(xs):" + q.prompt.split("def run(xs):", 1)[1].split("\nThe input domain", 1)[0]
    # This is our trusted emitted task source, never a model submission.
    namespace = {}
    exec(source, namespace)
    original = namespace["run"]
    exec(source.replace("rows = [a, a, a.copy()]", "rows = [a, a.copy(), a.copy()]"), namespace)
    mutant = namespace["run"]
    domain = sorted(map(list, itertools.product(data["alphabet"], repeat=4)))
    outputs = [original(x) for x in domain]
    differing = [x for x, output in zip(domain, outputs) if output != mutant(x)]
    inverse = []
    for target in data["targets"]:
        solutions = [x for x, output in zip(domain, outputs) if output == target]
        inverse.append({"count": len(solutions), "first_two": solutions[:2], "last_two": solutions[-2:]})
    first = differing[0]
    return {"forward": [original(x) for x in data["probes"]], "inverse": inverse,
            "distinguishing_count": len(differing), "first_distinguishing_input": first,
            "first_distinguishing_outputs": [original(first), mutant(first)]}


class ReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = []
        for seed in (19, 23):
            for variant in (0, 1):
                cls.questions.extend(load_reasoning_questions(seed, variant))
            cls.questions.extend(q for q in load_new_questions(split="evaluation", variants=2, seed=seed)
                                 if q.metadata["family"] in ("Q9-selective-report", "Q9-revision-multihop"))

    def test_answers_against_independent_prompt_oracles(self):
        for q in self.questions:
            data = data_from(q)
            family = q.metadata["family"]
            if family == "Q9-selective-report":
                expected = probability_oracle(data)
                self.assertNotEqual(expected["future_covariance"], [0, 1])
                self.assertNotEqual(Fraction(*expected["posterior_regime_1"]),
                                    Fraction(data["prior_weights"][1], sum(data["prior_weights"])))
            elif family == "Q9-revision-multihop":
                expected = retrieval_oracle(data)
                statuses = {r["status"] for r in expected["queries"]}
                self.assertTrue({"missing", "deleted", "cycle"} <= statuses)
                self.assertEqual("resolved" in statuses, q.metadata["variant"] == 0)
                self.assertEqual(len({r["id"] for r in data["records"]}), len(data["records"]))
                self.assertTrue(all(r["kind"] in {"link", "result"} for r in data["records"]))
                self.assertTrue(all(r["id"].startswith("R") for r in data["records"]))
            elif family == "Q14-grid-ambiguity":
                expected = grid_oracle(data)
                counts = [s["count"] for s in expected["scenarios"]]
                self.assertEqual(counts[0], 1)
                self.assertGreater(counts[1], 1)
                self.assertGreaterEqual(counts[2], counts[1])
            else:
                expected = code_oracle(q, data)
                self.assertTrue(any(x["count"] == 0 for x in expected["inverse"]))
                self.assertTrue(any(x["count"] > 2 for x in expected["inverse"]))
                self.assertGreater(expected["distinguishing_count"], 0)
            self.assertEqual(q.expected["value"], expected, q.id)
            result = score(q, expected)
            self.assertTrue(result.passed, (q.id, result.detail))
            self.assertEqual(result.evaluation.contract_score, 1)
            self.assertAlmostEqual(result.evaluation.criterion_achievement, 1)

    def test_reproducibility_and_family_contracts(self):
        for seed in (19, 23):
            for variant in (0, 1):
                fresh = load_reasoning_questions(seed, variant)
                stored = [q for q in self.questions if q.id in {x.id for x in fresh}]
                self.assertEqual([fingerprint(q) for q in fresh], [fingerprint(q) for q in stored])
        for family in {q.metadata["family"] for q in self.questions}:
            questions = [q for q in self.questions if q.metadata["family"] == family]
            contracts = [q.prompt.split("Output contract (types, not answer values): ")[1]
                         for q in questions]
            self.assertEqual(len(set(contracts)), 1, family)

    def test_retrieval_record_order_and_dominated_rows_do_not_change_answer(self):
        from reasoning_cases import resolve_records
        for q in self.questions:
            if q.metadata["family"] != "Q9-revision-multihop":
                continue
            data = data_from(q)
            # Reverse ties as well as the entire packet; lexical ID precedence
            # cannot silently fall back to first/last row order.
            data["records"].reverse()
            original = q.expected["value"]
            self.assertEqual(resolve_records(data), original)
            self.assertEqual(retrieval_oracle(data), original)
            old = {**data["records"][0], "id": "R0000", "revision": 0, "active": True}
            data["records"].append(old)
            self.assertEqual(resolve_records(data), original)

    def test_near_misses_and_structure_failures_are_scored(self):
        rejected = 0
        for q in self.questions:
            for wrong in corruptions(q.expected["value"]):
                result = score(q, wrong)
                self.assertTrue(result.is_scored, (q.id, result.detail))
                self.assertFalse(result.passed, (q.id, wrong))
                rejected += 1
            for key in q.expected["value"]:
                wrong = deepcopy(q.expected["value"])
                del wrong[key]
                result = score(q, wrong)
                self.assertFalse(result.passed)
                self.assertEqual(result.evaluation.contract_score, 0)
            for text in ("{}", "null", "[]", json.dumps(q.expected["value"]) + " trailing",
                         "```json\n" + json.dumps(q.expected["value"]) + "\n```"):
                result = score_response(q, text, TokenUsage(), RequestMetrics())
                self.assertTrue(result.is_scored)
                self.assertFalse(result.passed)
        print(f"Rejected {rejected} corrupted v14 answers")

    def test_atomic_credit_is_per_quantity_including_structural_errors(self):
        q = next(q for q in self.questions if q.metadata["family"] == "Q9-selective-report")
        for value in ([999, 1], [999], [], [True, 1], "wrong", None, [1, 2, 3]):
            wrong = {**q.expected["value"], "report_probability": value}
            result = score(q, wrong)
            self.assertTrue(result.is_scored, result.detail)
            self.assertFalse(result.passed)
            self.assertAlmostEqual(result.evaluation.criterion_achievement, 5 / 6, msg=str(value))
        wrong = deepcopy(q.expected["value"])
        wrong["report_probability"][0] += 1
        self.assertAlmostEqual(score(q, wrong).evaluation.criterion_achievement, 5 / 6)
        wrong = deepcopy(q.expected["value"])
        del wrong["report_probability"]
        self.assertAlmostEqual(score(q, wrong).evaluation.criterion_achievement, 5 / 6)

    def test_integer_contracts_survive_atomic_grouping(self):
        def float_substitutions(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    for changed in float_substitutions(child):
                        yield {**value, key: changed}
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    for changed in float_substitutions(child):
                        yield value[:i] + [changed] + value[i + 1:]
            elif type(value) is int:
                yield float(value)

        rejected = 0
        for q in self.questions:
            for wrong in float_substitutions(q.expected["value"]):
                result = score(q, wrong)
                self.assertTrue(result.is_scored, (q.id, result.detail))
                self.assertFalse(result.passed, q.id)
                self.assertEqual(result.evaluation.contract_score, 0)
                self.assertLess(result.evaluation.criterion_achievement, 1)
                rejected += 1
        print(f"Rejected {rejected} v14 integer-to-float substitutions")

    def test_atomic_nested_answers_and_invalid_authoring(self):
        answer = {"rows": [{"ratio": [1, 2], "value": 3}], "empty": []}
        paths = ["rows[0].ratio", "empty"]
        q = Question("atomic", "Reasoning", "Return exact JSON", "json_match",
                     {"value": answer, "atomic_paths": paths},
                     rubric=balanced_rubric(answer, atomic_paths=paths))
        for rows, achievement in (([], 0.5), ([{"ratio": [1, 2], "value": 4}], 0.75),
                                  ([{"ratio": [1, 3], "value": 3}], 0.75)):
            result = score(q, {"rows": rows, "empty": []})
            self.assertAlmostEqual(result.evaluation.criterion_achievement, achievement)
        report = Report()
        check_json_match(report, "ok", q.expected)
        self.assertEqual(report.errors, [])
        for invalid in ("rows", ["rows", "rows[0]"], ["rows[00]"], [""], [False], [{}],
                        ["missing"], ["rows[0].value"], ["empty", "empty"]):
            report = Report()
            check_json_match(report, "bad", {"value": answer, "atomic_paths": invalid})
            self.assertTrue(report.errors, invalid)

    def test_atomic_root_field_is_distinct_from_document_contract(self):
        answer = {"root": [1, 2], "count": 3}
        q = Question("root", "Reasoning", "Return exact JSON", "json_match",
                     {"value": answer, "atomic_paths": ["root"]},
                     rubric=balanced_rubric(answer, atomic_paths=["root"]))
        result = score(q, {**answer, "extra": True})
        self.assertFalse(result.passed)
        self.assertEqual(result.evaluation.contract_score, 0)
        self.assertEqual(result.evaluation.criterion_achievement, 1)
        result = score(q, [])
        self.assertFalse(result.passed)
        self.assertEqual(result.evaluation.criterion_achievement, 0)

    def test_additional_generation_seeds_have_independent_answers(self):
        for seed in (0, 7, 101):
            for variant in (0, 1):
                for q in load_reasoning_questions(seed, variant):
                    data = data_from(q)
                    answer = grid_oracle(data) if q.metadata["family"] == "Q14-grid-ambiguity" else code_oracle(q, data)
                    self.assertEqual(q.expected["value"], answer, q.id)

    def test_complete_bank_json_keys_satisfy_evaluator_and_rubric(self):
        from quality_suite import load_questions
        checked = 0
        for q in load_questions():
            if q.evaluator != "json_match" or q.interaction:
                continue
            result = score(q, q.expected["value"])
            self.assertTrue(result.passed, (q.id, result.detail))
            self.assertAlmostEqual(result.evaluation.criterion_achievement, 1, msg=q.id)
            checked += 1
        self.assertEqual(checked, 264)

    def test_boolean_mapping_keys_cannot_impersonate_numbers(self):
        self.assertFalse(values_equal({True: "x"}, {1: "x"}))
        self.assertFalse(values_equal({"nested": {False: [3]}}, {"nested": {0: [3]}}))
        self.assertTrue(values_equal({1.0: "x"}, {1: "x"}))
        self.assertTrue(values_equal({False: "x"}, {False: "x"}))
        self.assertFalse(values_equal({1.00000001: "x"}, {1.0: "x"}))
        q = Question("map", "Code Generation", "Return integer-keyed map", "code_exec",
                     [{"function": "f", "expected": {1: "x"}}])
        for source, passes in (("def f(): return {True: 'x'}", False),
                               ("def f(): return {1: 'x'}", True)):
            result = score_response(q, source, TokenUsage(), RequestMetrics())
            self.assertTrue(result.is_scored)
            self.assertEqual(result.passed, passes)


if __name__ == "__main__":
    unittest.main()
