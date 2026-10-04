"""Independent v10 oracles and regressions for issues found during critical review."""

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
import itertools
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.benchmarking.context_cases import FAMILIES, context_question
from app.benchmarking.evaluators import extract_json
from app.benchmarking.frontier_cases import UNICODE_REFERENCE, load_frontier_questions
from app.benchmarking.interactive_tasks import Environment, make_tasks, run_interaction
from app.benchmarking.models import Question, RequestMetrics, TokenUsage
from app.benchmarking.quality_execution import score_response
from app.benchmarking.quality_suite import load_questions, suite_hash
from app.benchmarking.rigorous_cases import balanced_rubric
from selftest_specialists import corruptions
from app.benchmarking.test_loader import (
    SuiteError,
    _parse_question,
    compute_test_suite_hash,
    load_all_tests,
    load_yaml_tests,
)


def score(q, text):
    return score_response(q, text, TokenUsage(), RequestMetrics(finish_reason="stop"))


class FrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = [
            q
            for seed in (19, 23)
            for variant in (0, 1)
            for q in load_frontier_questions(seed, variant)
        ]

    def test_portfolios_against_combination_search(self):
        for q in self.questions:
            if q.metadata["family"] != "Q10-robust-portfolio":
                continue
            data = q.metadata["oracle_input"]
            feasible = []
            for count in range(3, 7):
                for subset in itertools.combinations(data["projects"], count):
                    ids = sorted(p["id"] for p in subset)
                    cost = sum(p["cost"] for p in subset)
                    if cost > data["budget"] or any(
                        a in ids and b not in ids for a, b in data["dependencies"]
                    ):
                        continue
                    if any(a in ids and b in ids for a, b in data["exclusions"]):
                        continue
                    returns = list(map(sum, zip(*(p["returns"] for p in subset))))
                    feasible.append((min(returns), sum(returns), cost, ids, returns))
            feasible.sort(key=lambda r: (-r[0], -r[1], r[2], r[3]))
            best, second = feasible[:2]
            self.assertEqual(
                q.expected["value"],
                {
                    "selected": best[3],
                    "scenario_returns": best[4],
                    "worst_return": best[0],
                    "total_return": best[1],
                    "cost": best[2],
                    "runner_up": second[3],
                    "worst_return_gap": best[0] - second[0],
                },
                q.id,
            )

    def test_causal_probabilities_against_weighted_contingency_table(self):
        for q in self.questions:
            if q.metadata["family"] != "Q10-causal-counterfactual":
                continue
            priors = [Fraction(*p) for p in q.metadata["oracle_input"]["priors"]]
            # Enumerate integer masks; truth-table formulas differ from the builder.
            table = defaultdict(Fraction)
            for mask in range(16):
                bits = [(mask >> i) & 1 for i in range(4)]
                u, v, w, z = bits
                a = int(u != v)
                b = max(a, w)
                y0 = int((u == 1 and z == 0) or (a == 1 and z == 1))
                y1 = int((u == 0 and z == 0) or (a == 1 and z == 1))
                weight = Fraction(1)
                for bit, p in zip(bits, priors):
                    weight *= p if bit else 1 - p
                table[(b, y1 if b else y0, y1, y0)] += weight
            probabilities = {
                "observational": sum(p for (b, y, _, _), p in table.items() if b and y)
                / sum(p for (b, _, _, _), p in table.items() if b),
                "interventional": sum(p for (_, _, y, _), p in table.items() if y),
                "untreated": sum(p for (_, _, _, y), p in table.items() if y),
                "benefit_probability": sum(p for (_, _, y1, y0), p in table.items() if y1 > y0),
                "harm_probability": sum(p for (_, _, y1, y0), p in table.items() if y0 > y1),
            }
            probabilities["average_effect"] = (
                probabilities["interventional"] - probabilities["untreated"]
            )
            self.assertEqual(
                q.expected["value"],
                {k: [v.numerator, v.denominator] for k, v in probabilities.items()},
                q.id,
            )
            self.assertNotEqual(probabilities["observational"], probabilities["interventional"])

    def test_sql_answer_keys_against_sqlite(self):
        for q in self.questions:
            if q.metadata["family"] != "Q10-sql-null-cardinality":
                continue
            data = q.metadata["oracle_input"]
            db = sqlite3.connect(":memory:")
            try:
                db.executescript(
                    "CREATE TABLE customers(id,tenant); CREATE TABLE orders(id,customer,status,amount); CREATE TABLE tags(order_id,tag);"
                )
                db.executemany("INSERT INTO customers VALUES (?,?)", data["customers"])
                db.executemany("INSERT INTO orders VALUES (?,?,?,?)", data["orders"])
                db.executemany("INSERT INTO tags VALUES (?,?)", data["tags"])
                rows = []
                for cid, count, orders, amounts, total in db.execute(
                    """SELECT c.id,COUNT(*),COUNT(DISTINCT o.id),COUNT(o.amount),SUM(o.amount)
                       FROM customers c LEFT JOIN orders o ON o.customer=c.id AND o.status='paid'
                       LEFT JOIN tags t ON t.order_id=o.id WHERE c.tenant='red'
                       GROUP BY c.id ORDER BY c.id"""
                ):
                    tags = [
                        r[0]
                        for r in db.execute(
                            """SELECT DISTINCT t.tag FROM orders o JOIN tags t
                       ON t.order_id=o.id WHERE o.customer=? AND o.status='paid' AND t.tag IS NOT NULL ORDER BY t.tag""",
                            (cid,),
                        )
                    ]
                    rows.append(
                        {
                            "customer": cid,
                            "joined_rows": count,
                            "orders": orders,
                            "nonnull_amounts": amounts,
                            "sum_amount": total,
                            "tags": tags,
                        }
                    )
                total = db.execute("""SELECT SUM(o.amount) FROM orders o JOIN customers c
                   ON c.id=o.customer WHERE c.tenant='red' AND o.status='paid'""").fetchone()[0]
                null_rows = list(db.execute("SELECT id FROM customers WHERE id NOT IN ('z',NULL)"))
                self.assertEqual(
                    q.expected["value"],
                    {"rows": rows, "not_in_rows": null_rows, "distinct_order_total": total},
                    q.id,
                )
            finally:
                db.close()

    def test_weighted_repairs_against_clause_subset_feasibility(self):
        for q in self.questions:
            if q.metadata["family"] != "Q10-weighted-policy-repair":
                continue
            data = q.metadata["oracle_input"]

            def matches(clause, assignment):
                return any(assignment[int(i)] == bool(v) for i, v in clause)

            worlds = [
                bits
                for bits in itertools.product((False, True), repeat=7)
                if all(matches(c, bits) for c in data["required"])
            ]
            found = []
            for mask in range(1 << len(data["optional"])):
                removed = [row for i, row in enumerate(data["optional"]) if mask & (1 << i)]
                remaining = [row for i, row in enumerate(data["optional"]) if not mask & (1 << i)]
                solutions = [
                    bits for bits in worlds if all(matches(c["clause"], bits) for c in remaining)
                ]
                if solutions:
                    found.append(
                        (
                            sum(c["weight"] for c in removed),
                            len(removed),
                            sorted(c["id"] for c in removed),
                            [int(bit) for bit in min(solutions)],
                            len(solutions),
                        )
                    )
            best = min(found)
            self.assertEqual(
                q.expected["value"],
                {
                    "removed": best[2],
                    "cost": best[0],
                    "assignment": best[3],
                    "remaining_solutions": best[4],
                },
                q.id,
            )

    def test_vector_frontier_against_pairwise_partial_order(self):
        for q in self.questions:
            if q.metadata["family"] != "Q10-causal-register":
                continue
            approved = [r for r in q.metadata["oracle_input"]["records"] if r["approved"]]
            dominated = set()
            for a, b in itertools.permutations(approved, 2):
                difference = [x - y for x, y in zip(a["clock"], b["clock"])]
                if min(difference) >= 0 and max(difference) > 0:
                    dominated.add(b["id"])
            heads = [r for r in approved if r["id"] not in dominated]
            values = {r["value"] for r in heads}
            self.assertEqual(
                q.expected["value"],
                {
                    "heads": sorted(r["id"] for r in heads),
                    "live_values": sorted(v for v in values if v is not None),
                    "has_tombstone": None in values,
                    "conflict": len(values) > 1,
                    "dominated": sorted(dominated),
                },
                q.id,
            )

    def test_unicode_reference_and_plausible_mutants(self):
        mutants = [
            UNICODE_REFERENCE.replace(".casefold()", ".lower()"),
            UNICODE_REFERENCE.replace("order > chosen[key][0]", "order[0] > chosen[key][0][0]"),
            UNICODE_REFERENCE.replace(
                "if key not in chosen", "if deleted: continue\n        if key not in chosen"
            ),
        ]
        for q in self.questions:
            if q.evaluator != "code_exec":
                continue
            self.assertTrue(score(q, "```python\n" + UNICODE_REFERENCE + "\n```").passed, q.id)
            for mutant in mutants:
                self.assertFalse(score(q, "```python\n" + mutant + "\n```").passed, q.id)

    def test_all_new_answer_leaves_omissions_and_markdown(self):
        for q in self.questions:
            if q.evaluator != "json_match":
                continue
            answer = q.expected["value"]
            self.assertTrue(score(q, json.dumps(answer)).passed, q.id)
            self.assertFalse(score(q, "```json\n" + json.dumps(answer) + "\n```").passed, q.id)
            for wrong in corruptions(answer):
                self.assertFalse(score(q, json.dumps(wrong)).passed, (q.id, wrong))

    def test_context_answers_from_emitted_archive_not_builder_metadata(self):
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        for seed in (19, 23):
            for family in FAMILIES:
                answers = []
                for size in (8192, 32768):
                    q = context_question(size, seed, family)
                    self.assertEqual(len(encoding.encode(q.prompt)), size)
                    rows = [
                        json.loads(line[7:])
                        for line in q.prompt.splitlines()
                        if line.startswith("RECORD ")
                    ]
                    self.assertEqual(len({r["record"] for r in rows}), len(rows))
                    sites = [
                        r
                        for r in rows
                        if r["kind"] == "site" and r["temperature"] < -10 and r["capacity"] >= 50
                    ]
                    self.assertEqual(len(sites), 1)
                    site_record = sites[0]
                    site = site_record["site"]
                    if family == "relational-revisions":
                        alias = max(
                            (
                                r
                                for r in rows
                                if r["kind"] == "alias" and r["site"] == site and r["approved"]
                            ),
                            key=lambda r: r["revision"],
                        )
                        quota = max(
                            (
                                r
                                for r in rows
                                if r["kind"] == "quota"
                                and r["route"] == alias["route"]
                                and r["approved"]
                                and r["effective_from"] <= 50
                                and r["known_at"] <= 55
                            ),
                            key=lambda r: r["revision"],
                        )
                        factor = next(
                            r for r in rows if r["kind"] == "factor" and r["site"] == site
                        )
                        answer = {
                            "site": site,
                            "route": alias["route"],
                            "quota": quota["quota"],
                            "multiplier": factor["multiplier"],
                            "total": quota["quota"] * factor["multiplier"],
                            "evidence": sorted(
                                r["record"] for r in (site_record, alias, quota, factor)
                            ),
                        }
                    else:
                        selected = {}
                        for row in sorted(rows, key=lambda r: r.get("revision", 0)):
                            if (
                                row["kind"] == "shipment"
                                and row["site"] == site
                                and row["approved"]
                                and row["effective_from"] <= 50
                                and row["known_at"] <= 55
                            ):
                                selected[row["shipment"]] = row
                        live = [r for r in selected.values() if not r["deleted"]]
                        answer = {
                            "site": site,
                            "shipments": sorted(r["shipment"] for r in live),
                            "count": len(live),
                            "totals": [
                                [item, sum(r["quantity"] for r in live if r["item"] == item)]
                                for item in ("amber", "blue", "green")
                            ],
                            "net_quantity": sum(r["quantity"] for r in live),
                            "evidence": sorted(
                                [site_record["record"]] + [r["record"] for r in selected.values()]
                            ),
                        }
                    self.assertEqual(q.expected["value"], answer, q.id)
                    self.assertTrue(score(q, json.dumps(answer)).passed)
                    for wrong in corruptions(answer):
                        self.assertFalse(score(q, json.dumps(wrong)).passed, (q.id, wrong))
                    positions = [
                        q.prompt.index('"record":"' + r["record"] + '"') / len(q.prompt)
                        for r in q.metadata["core_records"]
                    ]
                    self.assertGreater(max(positions) - min(positions), 0.75)
                    answers.append(answer)
                self.assertEqual(answers[0], answers[1])


class ReviewRegressionTests(unittest.TestCase):
    def test_split_bank_and_recursive_loader(self):
        files = list(Path("tests/questions").rglob("*.yaml"))
        self.assertEqual(len(files), 171)
        self.assertTrue(all(len(load_yaml_tests(p)) == 1 for p in files))
        self.assertLess(max(len(p.read_text(encoding="utf-8").splitlines()) for p in files), 1600)
        self.assertEqual(len(load_all_tests()), 171)
        self.assertEqual(suite_hash(load_questions()), suite_hash(load_questions()))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "nested" / "q.yaml"
            p.parent.mkdir()
            p.write_text(
                "- id: q\n  category: Test\n  prompt: Return x\n  evaluator: exact_match\n  expected: x\n",
                encoding="utf-8",
            )
            self.assertEqual(len(load_all_tests(directory)), 1)
            old = compute_test_suite_hash(directory)
            p.write_text(
                p.read_text(encoding="utf-8").replace("expected: x", "expected: y"),
                encoding="utf-8",
            )
            self.assertNotEqual(old, compute_test_suite_hash(directory))
            p.write_text(p.read_text(encoding="utf-8") + "  expected: z\n", encoding="utf-8")
            with self.assertRaises(SuiteError):
                load_yaml_tests(p)

    def test_loader_rejects_nonfinite_weights_and_ambiguous_settings(self):
        base = {
            "id": "q",
            "category": "Test",
            "prompt": "Return JSON",
            "evaluator": "json_match",
            "expected": {"value": {"x": 1}},
        }
        bad = [
            {"prompt": {}},
            {"weight": float("nan")},
            {"weight": float("inf")},
            {"max_tokens": 1.5},
            {"max_tokens": True},
            {"keywords": ["x"]},
            {"rubric": [{"id": "json:x", "weight": float("inf")}]},
        ]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(SuiteError):
                _parse_question({**base, **change}, "fixture")

    def test_short_arrays_keep_precise_credit_and_cardinality_gate(self):
        q = Question(
            "q",
            "Test",
            "Return JSON",
            "json_match",
            {"value": {"items": [7, 8, 9]}},
            rubric=balanced_rubric({"items": [7, 8, 9]}),
        )
        result = score(q, '{"items":[7,8]}')
        self.assertFalse(result.passed)
        self.assertEqual(result.evaluation.contract_score, 0)
        self.assertAlmostEqual(result.evaluation.criterion_achievement, 2 / 3)
        self.assertEqual(
            next(c for c in result.evaluation.criteria if c.criterion_id == "json:items[2]").earned,
            0,
        )
        self.assertFalse(score(q, '{"items":[7,8,9,10]}').passed)

    def test_dependency_credit_is_blocked_transitively(self):
        q = Question(
            "q",
            "Test",
            "Return JSON",
            "json_match",
            {"value": {"a": 1, "b": 2, "c": 3}},
            rubric=[
                {"id": "json:c", "depends_on": ["json:b"]},
                {"id": "json:b", "depends_on": ["json:a"]},
                {"id": "json:a"},
            ],
        )
        result = score(q, '{"a":0,"b":2,"c":3}')
        self.assertFalse(result.passed)
        self.assertEqual(result.evaluation.criterion_achievement, 0)
        for ident in ("json:b", "json:c"):
            criterion = next(c for c in result.evaluation.criteria if c.criterion_id == ident)
            self.assertEqual(criterion.reason_code, "blocked_by_dependency")
            self.assertEqual(criterion.earned, 0)

    def test_code_mutation_and_forbidden_import_are_real_failures(self):
        q = Question(
            "q",
            "Test",
            "Implement f without input mutations",
            "code_exec",
            [{"function": "f", "args": [[1, 2]], "expected": 3, "preserve_inputs": True}],
        )
        result = score(
            q,
            "```python\ndef f(values):\n    result=sum(values)\n    values[0]=True\n    return result\n```",
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.evaluation.criteria[0].reason_code, "input_mutation")
        self.assertEqual(result.evaluation.criterion_achievement, 0)
        self.assertTrue(score(q, "```python\ndef f(values): return sum(values)\n```").passed)
        self.assertFalse(
            score(q, "```python\nimport requests\ndef f(values): return sum(values)\n```").passed
        )

    def test_overflow_and_duplicate_json_are_invalid(self):
        for text in ['{"x":1e999}', '{"x":NaN}', '{"x":1,"x":1}']:
            self.assertIsNotNone(extract_json(text, strict=True)[1])
        self.assertIsNotNone(
            extract_json('```json\n{"x":1}\n```', strict=True, allow_fence=False)[1]
        )

    def test_scalar_type_mismatch_is_contract_failure(self):
        q = Question("q", "Test", "Return a numeric count", "json_match", {"value": {"count": 1}})
        for response in ('{"count":true}', '{"count":"1"}', '{"count":null}'):
            result = score(q, response)
            self.assertFalse(result.passed)
            self.assertEqual(result.evaluation.contract_score, 0)
            self.assertEqual(result.evaluation.criteria[1].reason_code, "type_mismatch")
        self.assertTrue(score(q, '{"count":1.0}').passed)

    def test_interactive_completion_and_authorization_are_separate(self):
        q = replace(
            make_tasks(19, 0)[2], metadata={**make_tasks(19, 0)[2].metadata, "max_turns": 1}
        )

        class Client:
            def __init__(self, text):
                self.text = text

            def complete_messages(self, *args, **kwargs):
                return self.text, TokenUsage(1, 1), RequestMetrics(finish_reason="stop")

        verdict = {"success": True, "state_success": True, "violations": [], "unnecessary_calls": 0}
        with patch.object(Environment, "verdict", return_value=verdict):
            exhausted = run_interaction(q, Client('{"tool":"list_objects","args":{"cursor":null}}'))
            self.assertFalse(exhausted.passed)
            self.assertEqual(exhausted.evaluation.contract_score, 0)
            self.assertEqual(
                next(
                    c for c in exhausted.evaluation.criteria if c.criterion_id == "protocol"
                ).status,
                "fail",
            )
            done = run_interaction(q, Client('{"done":true}'))
            self.assertTrue(done.passed)
            self.assertEqual(done.evaluation.evaluator, "interactive_state")
            self.assertEqual(done.evaluation.evaluator_version, "2")
        with patch.object(
            Environment,
            "verdict",
            return_value={**verdict, "success": False, "violations": ["unauthorized_delete"]},
        ):
            unsafe = run_interaction(q, Client('{"done":true}'))
            self.assertFalse(unsafe.passed)
            self.assertEqual(
                next(
                    c for c in unsafe.evaluation.criteria if c.criterion_id == "final-state"
                ).status,
                "pass",
            )
            self.assertEqual(
                next(
                    c for c in unsafe.evaluation.criteria if c.criterion_id == "authorization"
                ).status,
                "fail",
            )


if __name__ == "__main__":
    unittest.main()
