#!/usr/bin/env python3
"""Tests for empirical difficulty calibration."""

from __future__ import annotations

import unittest

from difficulty_calibration import calibrate_reports


def report(run_id, rows, *, suite="suite", budget=65536):
    return {
        "run_id": run_id,
        "suite_hash": suite,
        "protocol": {"quality_max_output_tokens": budget},
        "results": rows,
    }


def row(fingerprint, item_id, category, family, passed, score=1.0):
    return {
        "fingerprint": fingerprint,
        "id": item_id,
        "category": category,
        "family": family,
        "scope": "capability",
        "scored": True,
        "passed": passed,
        "score": score,
        "metadata": {"difficulty": "expert"},
    }


class CalibrationTests(unittest.TestCase):
    def test_cohorts_and_family_saturation_are_separate(self):
        rows_a = [
            row("a", "A", "Logical Reasoning", "policy", True),
            row("b", "B", "Logical Reasoning", "policy", True),
            row("c", "C", "Translation", "scope", False, 0.0),
        ]
        rows_b = [
            row("a", "A", "Logical Reasoning", "policy", True),
            row("b", "B", "Logical Reasoning", "policy", False, 0.0),
            row("c", "C", "Translation", "scope", False, 0.0),
        ]
        result = calibrate_reports([
            report(1, rows_a),
            report(2, rows_b),
            report(3, rows_a, budget=16384),
        ], min_families=2)

        self.assertEqual(len(result["cohorts"]), 2)
        first = next(c for c in result["cohorts"].values() if c["reports"] == 2)
        self.assertEqual(first["fully_covered_questions"], 3)
        self.assertEqual(first["categories"]["Logical Reasoning"]["families"], 1)
        self.assertTrue(any(r["action"] == "add_families" for r in first["recommendations"]))
        self.assertTrue(any(r["action"] == "replace_saturated_items" for r in first["recommendations"]))

    def test_missing_questions_are_not_counted_as_failures(self):
        result = calibrate_reports([
            report(1, [row("a", "A", "Security", "threat", True)]),
            report(2, []),
        ])
        cohort = next(iter(result["cohorts"].values()))
        self.assertEqual(cohort["fully_covered_questions"], 0)
        self.assertEqual(cohort["categories"], {})

    def test_invalid_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_reports([], saturation_rate=0.2, floor_rate=0.8)
        with self.assertRaises(ValueError):
            calibrate_reports([], min_families=0)


if __name__ == "__main__":
    unittest.main()
