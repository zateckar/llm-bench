"""Regression tests for database-discovered defects and all reliability-v4 items."""

from copy import deepcopy
from dataclasses import replace
import json
import tarfile
import unittest

from evaluators import EVALUATORS, eval_json_match, eval_numeric_match, values_equal
from llm_client import ChatClient, ClientConfig, _extract_message_text
from models import RequestMetrics, TokenUsage
from quality_execution import score_response
from quality_suite import (
    DEFAULT_MAX_OUTPUT_TOKENS, MAX_MAX_OUTPUT_TOKENS, QualityConfig, assemble_questions,
)
from reliability_cases import CODE, CREATIVE_IDEALS, build_cases
from test_loader import load_all_tests, _parse_question


class ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = {q.id: q for q in load_all_tests()}

    def grade(self, id, text):
        q = self.questions[id]
        return EVALUATORS[q.evaluator](text, q.expected)[0]

    def test_authoring_matches_shipped_questions(self):
        for raw in build_cases():
            q = _parse_question(raw, raw["id"])
            self.assertEqual(q, self.questions[q.id])
        categories = {raw["category"] for raw in build_cases()}
        for category in categories:
            self.assertGreaterEqual(
                sum(
                    q.category == category and q.id.startswith("RV4-")
                    for q in self.questions.values()
                ),
                2,
            )

    def test_every_structured_field_is_required_and_checked(self):
        def mutants(value):
            if isinstance(value, dict):
                for key in value:
                    missing = deepcopy(value)
                    del missing[key]
                    yield missing
                    for bad in mutants(value[key]):
                        yield {**value, key: bad}
                yield {**value, "unsolicited_claim": "true"}
            elif isinstance(value, list):
                yield value + ["invented"]
                for i, item in enumerate(value):
                    for bad in mutants(item):
                        yield value[:i] + [bad] + value[i + 1 :]
            elif isinstance(value, bool):
                yield not value
                yield int(value)
            elif isinstance(value, (int, float)):
                yield value + 1
            elif value is None:
                yield "unknown"
                yield 0
            else:
                yield value + " wrong"

        for q in self.questions.values():
            if not q.id.startswith("RV4-") or q.evaluator != "json_match":
                continue
            answer = q.expected["value"]
            with self.subTest(id=q.id):
                self.assertEqual(self.grade(q.id, json.dumps(answer)), 1)
                for mutant in mutants(answer):
                    self.assertEqual(self.grade(q.id, json.dumps(mutant)), 0, mutant)
                self.assertEqual(
                    self.grade(q.id, json.dumps(answer) + "\nActually that is false."), 0
                )

    def test_all_code_references_and_wrong_implementations(self):
        # Independently worked examples, rather than only checking an oracle
        # against fixtures produced by that same oracle.
        examples = {
            "RV4-CG-01": ([[[2, 5], [0, 2], [4, 8], [9, 9], [12, 10]]], [[0, 8]]),
            "RV4-CG-02": (
                [[["x", 2, "a"], ["x", 1, None], ["y", 3, None], ["x", 2, "b"]]],
                {"x": "b"},
            ),
            "RV4-AC-01": ([[-4, -2, -2], 1, 2], [-2, 1, 2]),
            "RV4-AC-02": ([[["a", 0, 2, 3], ["b", 2, 4, 3], ["c", 0, 4, 6]]], [6, ["a", "b"]]),
            "RV4-TA-01": ([[-5, -2], -3], [1, 2]),
            "RV4-TA-02": ([["a", "b", "c"], [["a", "c"], ["a", "c"]]], ["a", "b", "c"]),
            "RV4-TD-01": ([[[0, 2], [0, 3], [3, 4], [4, -2]], 3], [2, 5, 4, 2]),
            "RV4-TD-02": ([{"x": 5}, [["p", "x", 2], ["q", "x", -7], ["p", "y", 99]]], {"x": 0}),
        }
        for id, (fn, code) in CODE.items():
            with self.subTest(id=id):
                scope = {}
                exec(code, scope)
                args, expected = examples[id]
                self.assertEqual(scope[fn](*args), expected)
                self.assertEqual(self.grade(id, code), 1)
                self.assertLess(self.grade(id, f"def {fn}(*args): return None"), 1)
        rolling_code = CODE["RV4-TD-01"][1]
        self.assertLess(
            self.grade("RV4-TD-01", rolling_code.replace("t-width < s", "t-width <= s")), 1
        )
        self.assertLess(self.grade("RV4-TD-01", rolling_code.replace("events[:i+1]", "events")), 1)
        for id, text in CREATIVE_IDEALS.items():
            self.assertEqual(self.grade(id, text), 1)
            self.assertLess(self.grade(id, text.replace("\n", " ", 1)), 1)

    def test_independent_combinatorial_keys(self):
        # Direct hand-enumerated conditional sample space: two equal faces x
        # and singleton y, x != y, repeated value has three possible positions.
        pairs = [(x, y) for x in range(1, 7) for y in range(1, 7) if x != y and 2 * x + y >= 12]
        from fractions import Fraction

        p = Fraction(sum(x == 6 or y == 6 for x, y in pairs), len(pairs))
        self.assertEqual(
            self.questions["RV4-MR-01"].expected["value"],
            {
                "numerator": p.numerator,
                "denominator": p.denominator,
                "condition_outcomes": 3 * len(pairs),
            },
        )
        # Four chosen cycle vertices with at least one unchosen between each:
        # cycle independent-set count n/(n-k) * C(n-k,k) = 25.
        self.assertEqual(
            self.questions["RV4-MR-02"].expected["value"],
            {"labeled": 25, "rotations": 3, "dihedral": 3},
        )
        self.assertEqual(
            self.questions["RV4-AU-02"].expected["value"],
            {"selected": ["A", "B", "E"], "cost": 9, "benefit": 18},
        )
        self.assertEqual(
            self.questions["RV4-SA-01"].expected["value"],
            {"accepted": [0, 1, 2, 10, 11, 12], "denied": [3, 13]},
        )

    def test_numeric_and_json_false_passes(self):
        self.assertFalse(values_equal(2**64, 2**64 + 1))
        self.assertFalse(values_equal(float(2**64), 2**64 + 1, rel=0, abs_tol=0))
        self.assertEqual(eval_numeric_match("299792459", 299792458)[0], 0)
        self.assertEqual(
            eval_numeric_match("6.022 × 10²³", {"value": 6.022e23, "relative": 0.0005})[0], 1
        )
        spec = {"value": {"n": 2**64 + 1, "tiny": 6.62607015e-34}, "strict_json": True}
        self.assertEqual(eval_json_match(json.dumps(spec["value"]), spec)[0], 1)
        for bad in [dict(n=2**64, tiny=6.62607015e-34), dict(n=2**64 + 1, tiny=0)]:
            self.assertEqual(eval_json_match(json.dumps(bad), spec)[0], 0)
        for text in [
            '{"x":0,"x":1}',
            '{"x":NaN}',
            '{"x":1}\n{"x":2}',
            'Earlier {"x":1} but the actual answer is 2',
        ]:
            self.assertEqual(eval_json_match(text, {"value": {"x": 1}, "strict_json": True})[0], 0)

    def test_real_database_regressions(self):
        self.assertEqual(self.grade("FK-21", "Tungsten: 3,420 degrees C."), 1)
        self.assertEqual(
            self.grade(
                "CL-07",
                "1. O(n log n)\n2. O(n^(log_2 7))\n3. O(n)\n4. O(n)\n5. O(n)\n6. O(n alpha(n))",
            ),
            1,
        )
        self.assertEqual(
            self.grade(
                "IF-09",
                "Yesterday we sought cold brook stones under low clouds, then crept through dense spruce boughs to home; dusk held hush, owls spoke, dogs slept, embers glowed, yet hope kept us cool for tomorrow",
            ),
            1,
        )
        self.assertEqual(self.grade("FK-26", "ANSWERS: 241.8, 24.6"), 0)
        self.assertEqual(self.grade("FK-26", "ANSWERS: -241.8, 24.6"), 1)
        self.assertEqual(self.questions["TA-13"].expected[-1]["expected"], 2)
        self.assertEqual(self.grade("FK-23", "10973732"), 1)
        self.assertEqual(self.grade("FK-23", "10973731"), 0)
        q = self.questions["TF-09"]
        for fixture in q.expected:
            self.assertEqual(tarfile.nti(bytes(fixture["args"][0])), fixture["expected"])
        self.assertEqual(
            self.grade(
                "TF-09",
                """def parse_tar_size(field: bytes) -> int:
    if field[0] == 128:
        return int.from_bytes(field[1:], 'big')
    return int(field.strip(b' \\0') or b'0', 8)
""",
            ),
            1,
        )

    def test_reasoning_never_becomes_a_final_answer(self):
        text = _extract_message_text([{"message": {"reasoning_content": '{"x":1}'}}])
        q = replace(self.questions["RV4-IF-01"], expected={"value": {"x": 1}})
        r = score_response(q, text, TokenUsage(), RequestMetrics())
        self.assertEqual((r.score, r.outcome), (0, "missing_answer"))
        r = score_response(q, text, TokenUsage(), RequestMetrics(finish_reason="length"))
        self.assertEqual((r.score, r.outcome), (0, "truncation"))

        class Response:
            status_code = 200

            def iter_lines(self):
                yield b'data: {"choices":[{"delta":{"reasoning_content":"42"}}]}'
                yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
                yield b"data: [DONE]"

            def close(self):
                pass

        client = ChatClient(ClientConfig("https://fake.invalid", "unused", "fake"))
        client.session.post = lambda *a, **k: Response()
        content, _, _ = client.complete("reply")
        self.assertEqual(content, "<think>42</think>")

    def test_uniform_budget_and_provenance(self):
        base = list(self.questions.values())
        default = assemble_questions(base, QualityConfig())
        # Interactive tasks included: one JSON action still costs a full round
        # of reasoning, so their per-turn cap follows the run's setting too.
        self.assertTrue(all(q.max_tokens == DEFAULT_MAX_OUTPUT_TOKENS for q in default))
        custom = assemble_questions(base, QualityConfig(max_output_tokens=32768))
        self.assertTrue(all(q.max_tokens == 32768 for q in custom))
        self.assertTrue(any(q.interaction for q in default))
        from quality_suite import suite_hash

        # The cap is a run knob, not part of the question: changing it must not
        # fork the suite hash, or every cap change would strand past runs in
        # their own incomparable cohort.
        self.assertEqual(suite_hash(default), suite_hash(custom))
        with self.assertRaises(ValueError):
            QualityConfig(max_output_tokens=True)
        with self.assertRaises(ValueError):
            QualityConfig(max_output_tokens=MAX_MAX_OUTPUT_TOKENS + 1)
        QualityConfig(max_output_tokens=MAX_MAX_OUTPUT_TOKENS)

    def test_heuristic_passes_do_not_raise_capability(self):
        from quality_report import result_record, summarize
        from models import Result

        questions = assemble_questions(
            [self.questions["FK-05"], self.questions["RV4-FK-01"]],
            QualityConfig(generated=False, interactive=False),
        )
        rows = [
            result_record(Result(q, "", score, metrics=RequestMetrics()))
            for q, score in zip(questions, [1, 0])
        ]
        summary = summarize(rows)
        self.assertEqual(summary["category_balanced"], 0)
        self.assertEqual(summary["category_balanced_pass_rate"], 0)
        self.assertEqual(summary["heuristic_mean"], 1)
        self.assertEqual(summary["all_item_mean"], 0.5)
        self.assertEqual(summary["categories"]["Factual Knowledge"]["count"], 1)

    def test_budget_survives_web_and_plan_round_trip(self):
        from app.routes.admin import make_quality_config
        from app.routes.plans import _spec_to_kwargs, _QUALITY_FIELDS
        from fastapi import HTTPException

        spec = dict(
            static_only="",
            suite_seeds="1729",
            suite_split="evaluation",
            variants=2,
            quality_context_sizes="",
            input_price="",
            output_price="",
            quality_max_tokens=32768,
        )
        config = make_quality_config(**_spec_to_kwargs(spec, _QUALITY_FIELDS))
        self.assertEqual(config["max_output_tokens"], 32768)
        with self.assertRaises(HTTPException):
            make_quality_config(**{**spec, "quality_max_tokens": 0})
        from benchmark import parse_args

        self.assertEqual(parse_args(["--quality-max-tokens", "32768"]).quality_max_tokens, 32768)

    def test_audit_preserves_original_grades_and_skips_changed_contracts(self):
        from audit_results import audit_database
        from pathlib import Path
        import sqlite3
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.db"
            with sqlite3.connect(path) as db:
                db.executescript((Path(__file__).parent / "app/schema.sql").read_text())
                db.execute(
                    "INSERT INTO models(id,name,base_url,api_key,model_id) VALUES(1,'fake','private','secret','fake')"
                )
                db.execute("INSERT INTO test_runs(id,model_id,status) VALUES(1,1,'completed')")
                q = self.questions["FK-21"]
                for id, prompt in [(1, q.prompt), (2, "different task")]:
                    db.execute(
                        "INSERT INTO test_results(id,run_id,test_id,category,prompt,response,score,evaluator) "
                        "VALUES(?,1,?,?,?,?,0,?)",
                        (id, q.id, q.category, prompt, "Tungsten: 3,420 degrees C.", q.evaluator),
                    )
            db.close()
            before = path.read_bytes()
            report = audit_database(path, [1], replay=True)
            items = report["runs"][0]["items"]
            self.assertEqual(items[0]["score"], 0)
            self.assertEqual(items[0]["replay"]["score"], 1)
            self.assertIn("replay_skipped", items[1])
            self.assertEqual(before, path.read_bytes())
            self.assertNotIn("secret", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
