"""Oracle, contract and adversarial regression tests for rigorous-v10."""

from copy import deepcopy
from dataclasses import replace
import json
import unittest

from llm_client import ClientConfig
from models import Question, RequestMetrics, TokenUsage
from quality_execution import score_response
from quality_report import make_report, paired_comparison, result_record, summarize
from quality_suite import load_questions, fingerprint
from rigorous_cases import CODE_REFERENCES, balanced_rubric, load_new_questions, schema
from validate_suite import Report, validate_question


def score(q, answer, metrics=None):
    return score_response(q, answer, TokenUsage(), metrics or RequestMetrics(finish_reason="stop"))


class RigorousTests(unittest.TestCase):
    def test_fixed_suite_covers_every_category(self):
        qs = load_questions()
        self.assertEqual(len(qs), 275)
        self.assertEqual(len({q.category for q in qs}), 30)
        self.assertEqual(len({q.id for q in qs}), len(qs))
        self.assertFalse(any(q.id.startswith("H7-") for q in qs))
        self.assertTrue(all(q.metadata.get("cohort") == "rigorous-v10" for q in qs))
        validation = Report()
        for q in qs:
            if not q.interaction:
                validate_question(validation, q)
            if q.evaluator == "code_exec":
                self.assertNotIn("Return one JSON object", q.prompt)
        self.assertEqual(validation.errors, [])
        self.assertEqual(validation.warnings, [])

    def test_seeded_splits_and_instances_are_reproducible(self):
        a = load_new_questions(split="evaluation", variants=2, seed=19)
        b = load_new_questions(split="evaluation", variants=2, seed=19)
        c = load_new_questions(split="development", variants=2, seed=19)
        d = load_new_questions(split="evaluation", variants=2, seed=23)
        self.assertEqual([fingerprint(q) for q in a], [fingerprint(q) for q in b])
        self.assertFalse(set(fingerprint(q) for q in a) & set(fingerprint(q) for q in c + d))
        combined = load_new_questions(seed=19, variants=2) + load_new_questions(seed=23, variants=2)
        self.assertEqual(len(combined), 52)
        self.assertEqual(len({q.id for q in combined}), 52)

    def test_supported_seed_boundaries_and_maximum_variant_count(self):
        for seed in (0, 19, 23, 1729, 2**32 - 1):
            questions = load_new_questions(split="evaluation", variants=10, seed=seed)
            self.assertEqual(len(questions), 130)
            self.assertEqual(len({q.id for q in questions}), 130)

    def test_all_new_json_keys_and_single_field_corruptions(self):
        for split in ("development", "evaluation"):
            for q in load_new_questions(split=split, variants=2):
                if q.evaluator != "json_match":
                    continue
                expected = q.expected["value"]
                with self.subTest(q=q.id):
                    ideal = score(q, json.dumps(expected))
                    self.assertTrue(ideal.passed)
                    self.assertAlmostEqual(ideal.evaluation.criterion_achievement, 1)
                    for key in expected:
                        bad = deepcopy(expected)
                        del bad[key]
                        result = score(q, json.dumps(bad))
                        self.assertFalse(result.passed)
                        self.assertEqual(result.evaluation.contract_score, 0)
                        # Missing a large requirement cannot disappear from the
                        # diagnostic denominator, even if every other field is right.
                        self.assertLess(result.evaluation.criterion_achievement, 1)
                        bad = deepcopy(expected)
                        value = bad[key]
                        bad[key] = (
                            not value
                            if isinstance(value, bool)
                            else value + 1
                            if isinstance(value, int)
                            else "corrupt"
                        )
                        self.assertFalse(score(q, json.dumps(bad)).passed)
                    self.assertFalse(
                        score(q, json.dumps({**expected, "invented_fact": True})).passed
                    )

    def test_code_references_pass_in_real_subprocess(self):
        for split in ("development", "evaluation"):
            for q in load_new_questions(split=split, variants=2):
                if q.evaluator == "code_exec":
                    function = q.expected[0]["function"]
                    result = score(q, "```python\n" + CODE_REFERENCES[function] + "\n```")
                    self.assertTrue(result.passed, (q.id, result.detail))
                    self.assertEqual(result.score, 1)
                    for group in ("boundary", "generated-combination"):
                        self.assertAlmostEqual(
                            sum(item["weight"] for item in q.rubric if item["group"] == group), 0.5
                        )
                    wrong = "```python\ndef " + function + "(*args): return {}\n```"
                    self.assertFalse(score(q, wrong).passed)

    def test_near_miss_programs_fail_boundary_regressions(self):
        fixes = {
            "decode_fields": CODE_REFERENCES["decode_fields"].replace(
                "row.append(field)\n    rows.append(row)",
                "row.append(field)\n    if field or row[:-1]: rows.append(row)",
            ),
            "dependency_layers": CODE_REFERENCES["dependency_layers"].replace(
                "prerequisites[b].add(a)", "prerequisites[b].update([a] if a in known else [])"
            ),
            "settle": CODE_REFERENCES["settle"].replace("balances = before", "pass"),
        }
        for q in load_new_questions():
            if q.evaluator == "code_exec":
                function = q.expected[0]["function"]
                result = score(q, "```python\n" + fixes[function] + "\n```")
                self.assertFalse(result.passed, (q.id, result.detail))
                self.assertGreater(result.score, 0)
                self.assertLess(result.score, 1)

    def test_requirements_are_balanced_and_missing_structures_are_not_free(self):
        expected = {"answer": 7, "evidence": list(range(100))}
        q = Question(
            "weighted",
            "Reasoning",
            "return explicit JSON",
            "json_match",
            {"value": expected},
            rubric=balanced_rubric(expected),
        )
        result = score(q, json.dumps({"answer": 0, "evidence": list(range(100))}))
        self.assertFalse(result.passed)
        self.assertAlmostEqual(result.evaluation.criterion_achievement, 0.5)
        missing = score(q, '{"answer":7}')
        self.assertAlmostEqual(missing.evaluation.criterion_achievement, 0.5)
        self.assertEqual(missing.evaluation.contract_score, 0)
        legacy = score(replace(q, rubric=None), '{"answer":7}')
        self.assertAlmostEqual(legacy.evaluation.criterion_achievement, 1 / 101)

    def test_code_fixture_exception_is_zero_achievement(self):
        q = Question(
            "crash",
            "Code Generation",
            "implement f",
            "code_exec",
            [
                {"function": "f", "args": [1], "expected": 1},
                {"function": "f", "args": [0], "expected": 0},
            ],
        )
        result = score(q, "```python\ndef f(x): return 1 // x\n```")
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.5)
        self.assertEqual(result.evaluation.criterion_achievement, 0.5)
        self.assertEqual(result.evaluation.criteria[1].reason_code, "fixture_error")

    def test_missing_outcomes_show_bounds_instead_of_inflating_confidence(self):
        q = Question(
            "x",
            "Reasoning",
            "explicit JSON",
            "json_match",
            {"value": {"x": 1}},
            metadata={"family": "one"},
        )
        good = score(q, '{"x":1}')
        missing = score(
            replace(q, id="y", metadata={"family": "two"}),
            "",
            RequestMetrics(ok=False, error="timeout"),
        )
        report = summarize([result_record(good), result_record(missing)])
        self.assertEqual(report["category_balanced"], 1)
        self.assertEqual(report["planned_capability_count"], 2)
        self.assertEqual(report["excluded_capability_count"], 1)
        self.assertEqual(report["missing_outcome_score_bounds"], [0.5, 1])

    def test_evaluator_revision_is_part_of_comparison_protocol(self):
        q = Question("x", "Reasoning", "explicit JSON", "json_match", {"value": {"x": 1}})
        report = make_report(
            [score(q, '{"x":1}')], ClientConfig("http://fake.invalid", "unused", "fake")
        )
        other = deepcopy(report)
        other["protocol"]["evaluator_versions"]["json_match"] = "old"
        self.assertFalse(paired_comparison(report, other)["compatible"])

    def test_database_discovered_keyword_and_count_regressions(self):
        from evaluators import eval_contains_keywords, eval_format_check

        expected = {"all": ["mozambique"], "groups": [["ak-47", "ak47", "kalashnikov"]]}
        self.assertEqual(eval_contains_keywords("Mozambique: AK\u201147", expected)[0], 1)
        self.assertEqual(eval_contains_keywords("Mozambique: AK\u201148", expected)[0], 0)
        rubric = {"checks": [{"type": "count_occurrences", "pattern": r"\d+%", "min_count": 2}]}
        self.assertEqual(eval_format_check("20% 35% 60%", rubric)[0], 1)
        self.assertEqual(eval_format_check("20% 35%", rubric)[0], 1)
        self.assertEqual(eval_format_check("20%", rubric)[0], 0)
        exact = {"checks": [{"type": "count_occurrences", "pattern": r"\d+%", "count": 2}]}
        self.assertEqual(eval_format_check("20% 35% 60%", exact)[0], 0)

    def test_revised_statistics_prompt_accepts_at_least_two_supplied_figures(self):
        q = next(q for q in load_questions() if q.id == "CW-04")
        paragraphs = [
            "hypothetical 20% 35% 60% " + " ".join(["example"] * 30),
            " ".join(["sample"] * 34),
            " ".join(["illustration"] * 34),
        ]
        answer = "\n\n".join(paragraphs)
        self.assertTrue(score(q, answer).passed)
        self.assertFalse(score(q, answer.replace("hypothetical", "confirmed")).passed)
        self.assertFalse(score(q, answer.replace("35%", "some").replace("60%", "some")).passed)

    def test_invalid_occurrence_bounds_are_rejected(self):
        for check in (
            {"count": 2, "min_count": 1},
            {"min_count": 3, "max_count": 2},
            {"min_count": True},
        ):
            q = Question(
                "bad",
                "Creative Writing",
                "write measurable constraints",
                "format_check",
                {"checks": [{"type": "count_occurrences", "pattern": "abc", **check}]},
            )
            report = Report()
            validate_question(report, q)
            self.assertTrue(report.errors)

    def test_schema_does_not_leak_answer_values_or_result_array_lengths(self):
        self.assertEqual(schema({"result": [103, 209]}), schema({"result": [11]}))


if __name__ == "__main__":
    unittest.main()
