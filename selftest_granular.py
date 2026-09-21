#!/usr/bin/env python3
"""Regression tests for criterion-level evaluation diagnostics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_client import ClientConfig
from models import Question, RequestMetrics, TokenUsage
from quality_execution import score_response
from quality_report import make_report, paired_comparison, result_record
from quality_suite import QualityConfig
from test_loader import SuiteError, load_yaml_tests


def score(question: Question, response: str):
    return score_response(
        question,
        response,
        TokenUsage(),
        RequestMetrics(finish_reason="stop"),
    )


class GranularEvaluationTests(unittest.TestCase):
    def test_json_reports_each_leaf_without_changing_full_pass(self):
        question = Question(
            "json",
            "Reasoning",
            "return JSON",
            "json_match",
            {"value": {"answer": 7, "trace": ["a", "b"]}, "strict_json": True},
        )
        result = score(question, '{"answer": 7, "trace": ["a", "wrong"]}')

        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.evaluation.contract_score, 1.0)
        self.assertAlmostEqual(result.evaluation.criterion_achievement, 2 / 3)
        failed = [c for c in result.evaluation.criteria if c.status == "fail"]
        self.assertEqual([c.criterion_id for c in failed], ["json:trace[1]"])

    def test_json_parse_failure_is_contract_failure_and_not_content_credit(self):
        question = Question(
            "json",
            "Reasoning",
            "return JSON",
            "json_match",
            {"value": {"answer": 7}, "strict_json": True},
        )
        result = score(question, "answer: 7")

        self.assertFalse(result.passed)
        self.assertEqual(result.evaluation.contract_score, 0.0)
        self.assertEqual(result.evaluation.criteria[0].criterion_id, "json-document")
        self.assertEqual(result.evaluation.criteria[0].reason_code, "invalid_json")
        self.assertIsNone(result.evaluation.criterion_achievement)

    def test_json_structure_failure_blocks_child_content_credit(self):
        question = Question(
            "json",
            "Reasoning",
            "return JSON",
            "json_match",
            {"value": {"phenotypes": {"dark": [1, 2], "light": [3, 4]}}},
        )
        result = score(question, '{"phenotypes": [1, 2, 3]}')

        self.assertFalse(result.passed)
        children = [c for c in result.evaluation.criteria if c.criterion_id.startswith("json:phenotypes.")]
        self.assertTrue(children)
        self.assertTrue(all(c.status == "not_evaluated" for c in children))
        self.assertIsNone(result.evaluation.criterion_achievement)

    def test_json_missing_parent_blocks_descendant_content_credit(self):
        question = Question(
            "json",
            "Reasoning",
            "return JSON",
            "json_match",
            {"value": {"record": {"id": 7, "kind": "x"}}},
        )
        result = score(question, "{}")

        descendants = [
            c for c in result.evaluation.criteria if c.criterion_id.startswith("json:record.")
        ]
        self.assertTrue(descendants)
        self.assertTrue(all(c.status == "not_evaluated" for c in descendants))
        self.assertEqual(result.evaluation.contract_score, 0.0)

    def test_code_exec_attaches_stable_fixture_criteria(self):
        question = Question(
            "code",
            "Code Generation",
            "implement f",
            "code_exec",
            {
                "tests": [
                    {"id": "normal", "function": "f", "args": [1], "expected": 2},
                    {"id": "boundary", "function": "f", "args": [0], "expected": 99},
                ]
            },
        )
        result = score(question, "```python\ndef f(x): return x + 1\n```")

        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.5)
        self.assertEqual(
            [(c.criterion_id, c.status) for c in result.evaluation.criteria],
            [("normal", "pass"), ("boundary", "fail")],
        )

    def test_explicit_critical_gate_overrides_partial_threshold(self):
        question = Question(
            "code",
            "Code Generation",
            "implement f",
            "code_exec",
            {
                "tests": [
                    {"id": "normal", "function": "f", "args": [1], "expected": 2},
                    {"id": "boundary", "function": "f", "args": [0], "expected": 99},
                ]
            },
            pass_threshold=0.5,
            rubric=[{"id": "boundary", "critical": True}],
        )
        result = score(question, "```python\ndef f(x): return x + 1\n```")

        self.assertEqual(result.score, 0.5)
        self.assertFalse(result.passed)
        self.assertEqual(result.outcome, "task_failure")

    def test_legacy_evaluator_gets_explicit_fallback_criterion(self):
        question = Question("exact", "Knowledge", "answer", "exact_match", ["yes"])
        result = score(question, "yes")

        self.assertTrue(result.passed)
        self.assertEqual([c.criterion_id for c in result.evaluation.criteria], ["task"])
        self.assertEqual(result.evaluation.criteria[0].reason_code, "full_match")

    def test_unemitted_rubric_criterion_is_not_silently_passed(self):
        question = Question(
            "exact",
            "Knowledge",
            "answer",
            "exact_match",
            ["yes"],
            rubric=[{"id": "answer", "critical": True}],
        )
        result = score(question, "yes")

        self.assertFalse(result.passed)
        missing = next(c for c in result.evaluation.criteria if c.criterion_id == "answer")
        self.assertEqual(missing.status, "not_evaluated")
        self.assertEqual(missing.reason_code, "rubric_criterion_unavailable")

    def test_report_schema_and_old_report_compatibility(self):
        question = Question("exact", "Knowledge", "answer", "exact_match", ["yes"])
        result = score(question, "yes")
        report = make_report(
            [result],
            QualityConfig(),
            ClientConfig("https://example.invalid", "unused", "model"),
        )

        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(result_record(result)["evaluation"]["criterion_count"], 1)
        old = json.loads(json.dumps(report))
        old["schema_version"] = 1
        self.assertTrue(paired_comparison(old, report)["compatible"])

    def test_rubric_contract_is_validated_and_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rubric.yaml"
            path.write_text(
                """
- id: R-1
  category: Code Generation
  prompt: implement f
  evaluator: code_exec
  expected:
    tests:
      - id: boundary
        function: f
        args: [0]
        expected: 1
  rubric:
    - id: boundary
      dimension: boundary_behavior
      weight: 2
      critical: true
""",
                encoding="utf-8",
            )
            question = load_yaml_tests(path)[0]
            result = score(question, "```python\ndef f(x): return x + 1\n```")
            criterion = result.evaluation.criteria[0]
            self.assertEqual(criterion.dimension, "boundary_behavior")
            self.assertEqual(criterion.possible, 2.0)
            self.assertTrue(criterion.critical)

            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "      critical: true", "      critical: true\n    - id: boundary"
                ),
                encoding="utf-8",
            )
            with self.assertRaises(SuiteError):
                load_yaml_tests(path)


if __name__ == "__main__":
    unittest.main()
