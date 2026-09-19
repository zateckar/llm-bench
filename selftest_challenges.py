#!/usr/bin/env python3
"""Check challenge answer keys, correct solutions, and near-miss rejection offline.

Run with: python selftest_challenges.py
No endpoint, API credentials, or second model is needed. Reasoning answers are
derived by exhaustive search/exact arithmetic or reviewed policy traces. Each
JSON leaf is independently corrupted to catch graders ignoring part of an answer.
Code solutions run in the same subprocess sandbox used by the benchmark, along
with a plausible broken implementation for each task.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

from challenge_oracles import CODE, REASONING
from evaluators import EVALUATORS
from test_loader import load_all_tests


def corruptions(value):
    """Yield one wrong leaf at a time, preserving the rest of the answer."""
    if isinstance(value, dict):
        for key, child in value.items():
            for wrong in corruptions(child):
                yield {**value, key: wrong}
        for key in value:
            yield {k: v for k, v in value.items() if k != key}
        yield {**value, "unexpected_field": True}
    elif isinstance(value, list):
        for i, child in enumerate(value):
            for wrong in corruptions(child):
                yield value[:i] + [wrong] + value[i + 1 :]
        if value:
            yield value[:-1]
            if value[::-1] != value:
                yield value[::-1]
        else:
            yield ["unexpected"]
    elif isinstance(value, bool):
        yield not value
    elif isinstance(value, (int, float)):
        yield value + 1
    elif value is None:
        yield "null"
    else:
        yield value + "-wrong"


# Each mutation violates a specific prompt requirement, rather than merely
# returning an obviously empty answer. Fixture coverage must catch every one.
MUTATIONS = {
    "AC2-01": ("leave < closing", "leave <= closing"),
    "AC2-02": ("if out and out[-1][1] == a and out[-1][2] == count:", "if False:"),
    "AC2-03": ('"blocked": sorted(remaining)', '"blocked": []'),
    "AC2-04": ("(v[0], v[1])", "(v[0], -v[1])"),
    "AC2-05": ("candidate < best", "candidate[0] <= best[0]"),
    "AC2-06": ("(profit == best[0] and ids < best[1])", "False"),
    "AC2-07": ("rejected.append(event_id)", "rejected.append(event_id); seen.remove(event_id)"),
    "AC2-08": (
        'tokens.append(("literal", pattern[i]))',
        'tokens.append(("wild" if pattern[i] in "*?" else "literal", pattern[i]))',
    ),
}


def main():
    questions = {
        q.id: q
        for q in load_all_tests(Path(__file__).parent / "tests")
        if q.source == "challenge-v2"
    }
    assert set(questions) == set(REASONING) | set(CODE), "Oracle coverage differs from suite"
    checks = 0
    for ident, oracle in REASONING.items():
        q = questions[ident]
        assert q.evaluator == "json_match" and q.pass_threshold == 1.0, ident
        assert q.expected.get("mode") == "exact" and not q.expected.get("ignore_keys"), ident
        answer = oracle()
        assert answer == q.expected["value"], f"{ident}: stored answer differs from derivation"
        evaluate = EVALUATORS[q.evaluator]
        # Key order, formatting, and a model's think block must not cause false failures.
        for response in [
            json.dumps(answer),
            "```json\n" + json.dumps(answer, indent=2, sort_keys=True) + "\n```",
            "<think>Working through the constraints.</think>\n" + json.dumps(answer),
        ]:
            score, detail = evaluate(response, q.expected)
            assert score == 1, f"{ident}: correct answer rejected: {detail}"
            checks += 1
        for wrong in corruptions(answer):
            score, detail = evaluate(json.dumps(wrong), q.expected)
            assert score < q.pass_threshold, f"{ident}: corrupted answer passed: {wrong}; {detail}"
            checks += 1
        for response in ["I would analyze the constraints and verify the result.", "{}", "[]"]:
            score, detail = evaluate(response, q.expected)
            assert score < q.pass_threshold, f"{ident}: empty/generic answer passed: {detail}"
            checks += 1

    for ident, oracle in CODE.items():
        q = questions[ident]
        assert q.evaluator == "code_exec" and q.pass_threshold == 1.0, ident
        assert len(q.expected) >= 7, f"{ident}: insufficient fixture coverage"
        for fixture in q.expected:
            assert oracle(*copy.deepcopy(fixture["args"])) == fixture["expected"], (
                f"{ident}: fixture differs from oracle: {fixture}"
            )
            checks += 1
        source = inspect.getsource(oracle)
        evaluate = EVALUATORS[q.evaluator]
        score, detail = evaluate("```python\n" + source + "\n```", q.expected)
        assert score == 1, f"{ident}: reference solution rejected: {detail}"
        old, new = MUTATIONS[ident]
        assert old in source, f"{ident}: stale mutation"
        score, detail = evaluate("```python\n" + source.replace(old, new) + "\n```", q.expected)
        assert score < q.pass_threshold, f"{ident}: flawed solution passed: {detail}"
        checks += 2
    print(f"All {checks} challenge checks passed across {len(questions)} questions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
