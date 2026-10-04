#!/usr/bin/env python3
"""Regression tests for criterion-level evaluation diagnostics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.models import Question, RequestMetrics, TokenUsage
from app.benchmarking.quality_execution import score_response
from app.benchmarking.quality_report import make_report, paired_comparison, result_record
from app.benchmarking.test_loader import SuiteError, load_yaml_tests


def score(question: Question, response: str):
    return score_response(
        question,
        response,
        TokenUsage(),
        RequestMetrics(finish_reason="stop"),
    )


class GranularEvaluationTests(unittest.TestCase):
    def test_reasoning_envelopes_do_not_rewrite_answer_data(self):
        from app.benchmarking.evaluators import strip_think_blocks
        self.assertEqual(strip_think_blocks('<think>unfinished answer: 42'), '')
        self.assertEqual(strip_think_blocks('<think>x</think><reasoning>y</reasoning>42'), '42')
        for literal in ('<think>secret</think>', '</think>', '<think>unfinished'):
            q = Question('literal', 'Code Generation', 'return literal', 'code_exec',
                         [{'function': 'f', 'expected': literal}])
            response = f'```python\ndef f():\n    return {literal!r}\n```'
            self.assertTrue(score(q, response).passed, literal)
            jq = Question('json', 'Reasoning', 'return literal', 'json_match',
                          {'value': {'text': literal}, 'strict_json': True})
            answer = json.dumps({'text': literal})
            self.assertTrue(score(jq, answer).passed)
            self.assertTrue(score(jq, '<think>work</think>' + answer).passed)
        q = Question('reasoning-only', 'Reasoning', 'return JSON', 'json_match',
                     {'value': {'answer': 42}, 'strict_json': True})
        result = score(q, '<think>{"answer":42}')
        self.assertFalse(result.passed)
        self.assertEqual(result.outcome, 'missing_answer')
        q.expected['allow_fence'] = False
        result = score(q, '```json\n{"answer":42}\n```')
        self.assertFalse(result.passed)
        self.assertEqual(result.outcome, 'formatting')

    def test_unordered_tolerance_finds_a_complete_matching(self):
        from app.benchmarking.evaluators import values_equal
        # 1.5 can match either target; 1.0 can only match 1.0.
        for got in ([1.5, 1.0], [1.0, 1.5]):
            for want in ([1.0, 2.0], [2.0, 1.0]):
                self.assertTrue(values_equal(got, want, rel=0, abs_tol=0.6, unordered=True))
        self.assertFalse(values_equal([1.0, 1.0], [1.0, 2.0], rel=0, abs_tol=0.6, unordered=True))

    def test_code_objects_cannot_impersonate_values_or_reorder_set_members(self):
        q = Question('object', 'Code Generation', 'return list', 'code_exec',
                     [{'function': 'f', 'expected': [1, 2]}])
        forged = "class Pretend:\n    def __repr__(self): return '[1, 2]'\ndef f(): return Pretend()"
        self.assertFalse(score(q, forged).passed)
        q.expected = [{'function': 'f', 'expected': [[1, 2], [3, 4]]}]
        self.assertTrue(score(q, 'def f():\n    return {(3, 4), (1, 2)}').passed)
        self.assertFalse(score(q, 'def f():\n    return {(2, 1), (3, 4)}').passed)

    def test_aborted_code_cannot_forge_sandbox_output(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        q = Question('abort', 'Code Generation', 'return one', 'code_exec',
                     [{'function': 'f', 'expected': 1}])
        forged = '{"results":[{"ok":true,"value":{"kind":"json","json":1}}]}'
        for code in (f'print({forged!r})\nraise SystemExit(0)\ndef f(): return 0',
                     f'def f():\n    print({forged!r})\n    raise SystemExit(0)'):
            result = score(q, code)
            self.assertTrue(result.is_scored)
            self.assertFalse(result.passed)
        with patch('app.benchmarking.code_runner.subprocess.run', return_value=SimpleNamespace(
                returncode=1, stdout=forged, stderr='child failed')):
            self.assertFalse(score(q, 'def f():\n    return 1').passed)
        with patch('app.benchmarking.code_runner.subprocess.run', side_effect=OSError('cannot launch interpreter')):
            result = score(q, 'def f():\n    return 1')
            self.assertEqual(result.outcome, 'evaluator_error')
            self.assertFalse(result.is_scored)

    def test_deep_invalid_json_is_a_scored_format_failure(self):
        import sys
        q = Question('deep', 'Reasoning', 'return JSON', 'json_match',
                     {'value': {'answer': 1}, 'strict_json': True})
        depth = sys.getrecursionlimit() + 100
        result = score(q, '[' * depth + '1' + ']' * depth)
        self.assertTrue(result.is_scored)
        self.assertEqual(result.outcome, 'formatting')
        self.assertFalse(result.passed)
        q.expected = {'value': {'answer': '["' * 1000}, 'strict_json': True}
        self.assertTrue(score(q, json.dumps(q.expected['value'])).passed)

    def test_raw_single_line_function_is_graded_as_written(self):
        q = Question('one-line', 'Code Generation', 'implement f', 'code_exec',
                     [{'function': 'f', 'args': [2], 'expected': 3}])
        self.assertTrue(score(q, 'def f(x): return x + 1').passed)
        self.assertFalse(score(q, 'def f(x): return x - 1').passed)

    def test_revised_writing_constraints_reject_near_misses(self):
        from app.benchmarking.quality_suite import load_questions
        from app.benchmarking.evaluators import eval_format_check
        from validate_suite import Report, check_format_check
        questions = {q.id: q for q in load_questions() if q.category == 'Creative Writing'}
        first = 'Night silent bird wings\nRain taps old stones\nDawn opens iron gates\nHome calls lost echoes'
        second = 'I found a silver key\nYou heard one distant bell\nWe crossed that narrow bridge\nThey raised our amber lantern'
        for qid, answer in [('RV4-CW-01', first), ('RV4-CW-02', second)]:
            self.assertTrue(score(questions[qid], answer).passed)
            self.assertFalse(score(questions[qid], answer.replace('\n', '\n\n', 1)).passed)
        self.assertFalse(score(questions['RV4-CW-01'], first.replace('old', 'SILENT,')).passed)
        self.assertFalse(score(questions['RV4-CW-02'], second.replace('one', '“SILVER”')).passed)
        spec = {'checks': [{'type': 'count_occurrences', 'pattern': r'\d+%', 'distinct': True, 'min_count': 2}]}
        self.assertEqual(eval_format_check('20% 35%', spec)[0], 1)
        self.assertEqual(eval_format_check('20% 20%', spec)[0], 0)
        bad = Report()
        check_format_check(bad, 'bad-distinct', {'checks': [{**spec['checks'][0], 'distinct': 'yes'}]})
        self.assertTrue(bad.errors)
        q = questions['CW-04']
        answer = '\n\n'.join(['hypothetical 20% 20% ' + ' '.join(['example'] * 31),
                              ' '.join(['sample'] * 34), ' '.join(['illustration'] * 34)])
        self.assertFalse(score(q, answer).passed)
        self.assertTrue(score(q, answer.replace('20% 20%', '20% 35%')).passed)

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
        children = [
            c for c in result.evaluation.criteria if c.criterion_id.startswith("json:phenotypes.")
        ]
        self.assertTrue(children)
        self.assertTrue(all(c.status == "not_evaluated" for c in children))
        self.assertEqual(result.evaluation.criterion_achievement, 0.0)

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
            ClientConfig("https://example.invalid", "unused", "model"),
        )

        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(result_record(result)["evaluation"]["criterion_count"], 1)
        old = json.loads(json.dumps(report))
        old["schema_version"] = 1
        self.assertFalse(paired_comparison(old, report)["compatible"])

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
