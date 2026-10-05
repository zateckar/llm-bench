"""Repeated-run statistics: estimators, paired tests and group membership."""

from copy import deepcopy
from itertools import combinations
import hashlib
import unittest

from app.benchmarking.paired_stats import (
    hierarchical_bootstrap,
    holm,
    mcnemar_exact,
    pass_at_k,
    pass_hat_k,
    sign_flip_test,
    t_interval_95,
    verdict,
)
from app.benchmarking.quality_report import (
    bootstrap,
    clusters,
    compare_groups,
    comparable_protocol,
    paired_comparison,
)
from app.benchmarking.repeat_stats import group_summary

PROTOCOL = {"revision": "v1", "temperature": 0.7, "model_seed": 0,
            "client": {"revision": "c", "model_seed": 0}}


def report(scores, *, seed=0, suite="s1", protocol=None, passes=None):
    """Synthetic schema-3 report: ``scores`` maps (category, family, task) to achievement."""
    rows = []
    for (category, family, task), score in sorted(scores.items()):
        passed = score >= 1 if passes is None else passes[(category, family, task)]
        rows.append({
            "id": task, "fingerprint": hashlib.sha256(task.encode()).hexdigest(),
            "category": category, "family": family, "scope": "capability",
            "scored": score is not None, "passed": bool(passed) and score is not None,
            "outcome": "pass" if passed else "task_failure",
            "score": score or 0.0, "evaluation": None,
        })
    base = deepcopy(protocol or PROTOCOL)
    base["model_seed"] = seed
    base["client"]["model_seed"] = seed
    return {"schema_version": 3, "suite_hash": suite, "protocol": base, "results": rows,
            "summary": {"category_balanced": sum(r["score"] for r in rows) / len(rows),
                        "category_balanced_full_pass": None, "scored": len(rows),
                        "total": len(rows), "categories": {}}}


def grid(value_for):
    return {(c, f"{c}-f{f}", f"{c}-f{f}-t{t}"): value_for(c, f, t)
            for c in ("A", "B") for f in range(6) for t in range(2)}


class EstimatorTests(unittest.TestCase):
    def test_pass_k_matches_enumeration(self):
        for n in range(1, 7):
            for c in range(n + 1):
                attempts = [1] * c + [0] * (n - c)
                for k in range(1, n + 1):
                    draws = list(combinations(range(n), k))
                    every = sum(all(attempts[i] for i in d) for d in draws) / len(draws)
                    some = sum(any(attempts[i] for i in d) for d in draws) / len(draws)
                    self.assertAlmostEqual(pass_hat_k(n, c, k), every)
                    self.assertAlmostEqual(pass_at_k(n, c, k), some)
        with self.assertRaises(ValueError):
            pass_hat_k(2, 3, 1)

    def test_t_interval_and_holm(self):
        self.assertIsNone(t_interval_95([0.5, 0.6]))
        low, high = t_interval_95([0.5, 0.6, 0.7])
        self.assertAlmostEqual((low + high) / 2, 0.6)
        self.assertAlmostEqual(high - 0.6, 4.303 * 0.1 / 3 ** 0.5)
        self.assertEqual(holm([0.01, None, 0.04, 0.03]), [0.03, None, 0.06, 0.06])

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 6), 2 / 64)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)

    def test_sign_flip_null_and_effect(self):
        self.assertEqual(sign_flip_test({"A": [0.0, 0.0]}), 1.0)
        balanced_noise = {"A": [0.1, -0.1, 0.2, -0.2], "B": [0.05, -0.05, 0.3, -0.3]}
        self.assertGreater(sign_flip_test(balanced_noise), 0.5)
        effect = {"A": [0.2] * 8, "B": [0.3] * 8}
        p = sign_flip_test(effect)
        self.assertLess(p, 0.001)
        self.assertEqual(p, sign_flip_test(effect))

    def test_single_repeat_bootstrap_reproduces_family_bootstrap(self):
        tasks = {"A": {"f1": [([0.0], [1.0])], "f2": [([0.5], [0.5]), ([1.0], [0.0])]},
                 "B": {"g1": [([0.2], [0.4])], "g2": [([0.0], [0.0])]}}
        rows = [{"category": c, "family": f, "score": right[0] - left[0]}
                for c, fams in tasks.items() for f, pairs in fams.items() for left, right in pairs]
        self.assertEqual(hierarchical_bootstrap(tasks), bootstrap(clusters(rows)))

    def test_verdict_requires_interval_and_p(self):
        self.assertEqual(verdict(0.1, [0.02, 0.2], 0.01), "right_higher")
        self.assertEqual(verdict(-0.1, [-0.2, -0.01], 0.01), "left_higher")
        self.assertEqual(verdict(0.1, [-0.01, 0.2], 0.01), "no_detectable_difference")
        self.assertEqual(verdict(0.1, [0.02, 0.2], 0.2), "no_detectable_difference")
        self.assertEqual(verdict(None, None, None), "unavailable")


class ComparisonTests(unittest.TestCase):
    def test_seed_alone_does_not_block_pairing(self):
        left = report(grid(lambda c, f, t: 0.5))
        right = report(grid(lambda c, f, t: 0.5), seed=3)
        self.assertEqual(comparable_protocol(left["protocol"]), comparable_protocol(right["protocol"]))
        result = paired_comparison(left, right)
        self.assertTrue(result["compatible"])
        self.assertFalse(result["same_seed"])
        self.assertEqual(result["model_seeds"], [0, 3])
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["verdict"], "no_detectable_difference")

    def test_other_protocol_changes_still_block(self):
        other = deepcopy(PROTOCOL)
        other["temperature"] = 0
        result = paired_comparison(report(grid(lambda *a: 1)), report(grid(lambda *a: 1), protocol=other))
        self.assertFalse(result["compatible"])

    def test_planted_effect_is_detected_across_groups(self):
        lefts = [report(grid(lambda c, f, t, s=s: 0.4 + 0.05 * ((f + t + s) % 3)), seed=s) for s in range(3)]
        rights = [report(grid(lambda c, f, t, s=s: 0.6 + 0.05 * ((f + t + s) % 3)), seed=s) for s in range(3)]
        result = compare_groups(lefts, rights)
        self.assertTrue(result["compatible"])
        self.assertAlmostEqual(result["balanced_difference"], 0.2)
        self.assertGreater(result["ci95"][0], 0)
        self.assertLess(result["p_value"], 0.001)
        self.assertEqual(result["verdict"], "right_higher")
        self.assertIsNone(result["full_pass_discordance"])
        self.assertEqual((result["left_runs"], result["right_runs"]), (3, 3))

    def test_single_runs_report_mcnemar(self):
        left = report(grid(lambda c, f, t: 1.0 if f < 3 else 0.0))
        right = report(grid(lambda c, f, t: 1.0))
        result = paired_comparison(left, right)
        discordance = result["full_pass_discordance"]
        self.assertEqual(discordance["right_only_pass"], 12)
        self.assertEqual(discordance["left_only_pass"], 0)
        self.assertLess(discordance["mcnemar_p"], 0.001)


class GroupSummaryTests(unittest.TestCase):
    def entries(self, reports, statuses=None):
        return [{"run_id": i + 1, "status": (statuses or {}).get(i + 1, "completed"), "report": r}
                for i, r in enumerate(reports)]

    def test_flaky_tasks_pass_k_and_consistency(self):
        def scores(seed):
            return {("A", "fa", "stable"): 1.0, ("A", "fb", "never"): 0.0,
                    ("B", "fc", "flaky"): 1.0 if seed != 1 else 0.0}
        summary = group_summary(self.entries([report(scores(s), seed=s) for s in range(3)]))
        self.assertTrue(summary["available"])
        self.assertEqual(summary["usable_runs"], 3)
        self.assertEqual([t["id"] for t in summary["flaky_tasks"]], ["flaky"])
        self.assertEqual(summary["flaky_tasks"][0]["passes"], 2)
        self.assertAlmostEqual(summary["consistent_pass_fraction"], 2 / 3)
        curve = {row["k"]: row for row in summary["pass_k"]}
        # Category A: stable 1, never 0 -> 0.5; category B: flaky pass^k = C(2,k)/C(3,k).
        self.assertAlmostEqual(curve[1]["pass_hat_k"], (0.5 + 2 / 3) / 2)
        self.assertAlmostEqual(curve[3]["pass_hat_k"], (0.5 + 0) / 2)
        self.assertAlmostEqual(curve[3]["pass_at_k"], (0.5 + 1) / 2)
        self.assertIsNotNone(summary["score_spread"]["ci95"])
        self.assertEqual([r["model_seed"] for r in summary["runs"]], [0, 1, 2])

    def test_unusable_members_are_excluded_with_reasons(self):
        other = deepcopy(PROTOCOL)
        other["temperature"] = 0
        data = grid(lambda *a: 1.0)
        reports = [report(data), report(data, seed=1), report(data, seed=2, suite="s2"),
                   report(data, seed=3, protocol=other), report(data, seed=4), {}]
        summary = group_summary(self.entries(reports, {5: "failed"}))
        self.assertEqual(summary["usable_runs"], 2)
        self.assertEqual(summary["excluded_runs"], [
            {"run_id": 3, "reason": "different_suite"},
            {"run_id": 4, "reason": "different_protocol"},
            {"run_id": 5, "reason": "not_completed"},
            {"run_id": 6, "reason": "missing_quality_report"},
        ])
        self.assertIsNone(summary["score_spread"]["ci95"])

    def test_empty_group(self):
        self.assertFalse(group_summary(self.entries([{}]))["available"])


if __name__ == "__main__":
    unittest.main()
