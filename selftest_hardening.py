#!/usr/bin/env python3
"""Offline checks for the hardening-v7 candidate profile."""

from __future__ import annotations

import copy
import json
import unittest

from difficulty_hardening import HARDENING_FAMILIES, build_backlog, validate_backlog
from evaluators import eval_code_exec, eval_json_match
from hardening_cases import build_cases, load_questions, verify_cases
from interactive_tasks import Environment, make_tasks
from quality_suite import QualityConfig, assemble_questions, suite_hash
from test_loader import load_all_tests, load_profile_tests


def _mutate_leaf(value):
    """Return a copy with one scalar changed, preserving its JSON shape."""
    changed = False

    def visit(node):
        nonlocal changed
        if changed:
            return node
        if isinstance(node, dict):
            return {key: visit(child) for key, child in node.items()}
        if isinstance(node, list):
            return [visit(child) for child in node]
        changed = True
        if isinstance(node, bool):
            return not node
        if isinstance(node, (int, float)):
            return node + 1
        if node is None:
            return "changed"
        return str(node) + "-changed"

    result = visit(copy.deepcopy(value))
    if not changed:  # pragma: no cover - every accepted answer has a leaf
        raise AssertionError("cannot mutate an answer without a scalar leaf")
    return result


FOLD_RECORDS_CODE = """```python
def fold_records(records):
    seen = set()
    balances = {}
    for row in records:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        balances[row["account"]] = balances.get(row["account"], 0) + (
            row["amount"] if row["op"] == "credit" else -row["amount"]
        )
    return [{"account": account, "balance": balances[account]} for account in sorted(balances)]
```"""


BEST_ROUTE_CODE = """```python
def best_route(graph, start, target):
    best = None

    def visit(node, cost, path):
        nonlocal best
        if node == target:
            candidate = (cost, tuple(path))
            if best is None or candidate < best:
                best = candidate
            return
        for source, dest, edge_cost, enabled in graph:
            if source == node and enabled and dest not in path:
                visit(dest, cost + edge_cost, path + [dest])

    visit(start, 0, [start])
    if best is None:
        return {"cost": None, "path": []}
    return {"cost": best[0], "path": list(best[1])}
```"""


class HardeningTests(unittest.TestCase):
    def test_every_category_has_two_distinct_candidate_families(self):
        validate_backlog()
        self.assertEqual(len(HARDENING_FAMILIES), 29)
        self.assertTrue(all(len(families) >= 2 for families in HARDENING_FAMILIES.values()))

    def test_calibration_recommendations_mark_priority(self):
        calibration = {
            "cohorts": {
                "x": {
                    "reports": 5,
                    "recommendations": [{"category": "Translation", "action": "add_families"}],
                }
            }
        }
        backlog = build_backlog(calibration)
        translation = next(row for row in backlog["categories"] if row["category"] == "Translation")
        self.assertEqual(translation["priority"], "high")
        self.assertEqual(len(translation["families"]), 2)

    def test_generated_bank_has_independent_oracles_and_unique_ids(self):
        self.assertEqual(verify_cases(), {"cases": 116, "families": 58, "variants": 2})
        cases = build_cases()
        self.assertEqual(len(cases), 116)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertEqual(len(load_questions()), 112)  # interactive cases use the simulator

        for case in cases:
            if case["category"] == "Interactive Tool Use":
                continue
            if case["evaluator"] == "json_match":
                answer = json.dumps(case["expected"]["value"], ensure_ascii=False)
                self.assertEqual(eval_json_match(answer, case["expected"])[0], 1.0, case["id"])
                mutated = json.dumps(_mutate_leaf(case["expected"]["value"]), ensure_ascii=False)
                self.assertEqual(eval_json_match(mutated, case["expected"])[0], 0.0, case["id"])

    def test_code_fixtures_run_in_the_sandbox(self):
        for case in build_cases():
            if case["evaluator"] != "code_exec":
                continue
            function = case["expected"][0]["function"]
            code = FOLD_RECORDS_CODE if function == "fold_records" else BEST_ROUTE_CODE
            score, detail = eval_code_exec(code, case["expected"])
            self.assertEqual(score, 1.0, f"{case['id']}: {detail}")

    def test_profile_assembles_static_and_interactive_hardening(self):
        base = load_profile_tests("hardening-only", split="development")
        config = QualityConfig(
            profile="hardening-only",
            generated=True,
            interactive=True,
            strengthen_code=False,
            context_sizes=(),
            variants=2,
            seeds=(1729,),
        )
        questions = assemble_questions(base, config)
        h7 = [question for question in questions if question.id.startswith("H7-")]
        self.assertEqual(len(h7), 116)
        self.assertEqual(len({question.metadata["family"] for question in h7}), 58)
        self.assertTrue(all(question.metadata.get("cohort") == "hardening-v7" for question in h7))

    def test_frozen_v6_hash_is_unchanged(self):
        questions = assemble_questions(load_all_tests(), QualityConfig())
        self.assertEqual(suite_hash(questions), "921ffa63e20a612e")

    def test_language_translation_bank_covers_all_six_directions(self):
        directions = set()
        for case in build_cases():
            if case["category"] == "Language Translations":
                directions.update(tuple(row["direction"]) for row in case["expected"]["value"]["meanings"])
        self.assertEqual(directions, {("cs", "en"), ("en", "cs"), ("cs", "de"), ("de", "cs"), ("en", "de"), ("de", "en")})

    def test_interactive_hardening_environments_require_replanning(self):
        config = QualityConfig(profile="hardening-only", generated=False, variants=1)
        task = next(task for task in make_tasks(config, 1729, 0) if task.metadata["family"] == "H7-resource-races")
        env = Environment(task.interaction)
        for spec in task.interaction["resources"]:
            initial = env.call("get_document", {"id": spec["id"]})
            self.assertEqual(initial["etag"], "e1")
            stale = env.call(
                "put_document",
                {
                    "id": spec["id"],
                    "if_match": "e1",
                    "body": {"owner": "Mira", "labels": ["blue", "red"], "limit": spec["limit"]},
                },
            )
            self.assertEqual(stale["status"], 412)
            refreshed = env.call("get_document", {"id": spec["id"]})
            self.assertTrue(
                env.call(
                    "put_document",
                    {
                        "id": spec["id"],
                        "if_match": refreshed["etag"],
                        "body": {
                            "owner": "Nora",
                            "labels": sorted(["blue", spec["concurrent_label"], "red"]),
                            "limit": spec["limit"],
                        },
                    },
                )["ok"]
            )
        self.assertTrue(env.verdict()["success"])

        permissions = Environment({"kind": "permission", "target_resource": "r7", "tenant": "tenant-h7", "now": 60})
        first = permissions.call("list_permissions", {"cursor": None})
        self.assertEqual(first["next"], "page2")
        second = permissions.call("list_permissions", {"cursor": "page2"})
        target = next(row for row in second["rows"] if row["resource"] == "r7")
        self.assertTrue(
            permissions.call(
                "set_permission",
                {"resource": "r7", "scopes": ["read", "write"], "if_version": target["version"]},
            )["ok"]
        )
        self.assertTrue(permissions.verdict()["success"])

    def test_backlog_is_explicitly_not_default_suite_content(self):
        self.assertIn("not accepted benchmark items", build_backlog()["note"])


if __name__ == "__main__":
    unittest.main()
