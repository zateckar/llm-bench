#!/usr/bin/env python3
"""Regression checks for the hardening profile manifest and release gate."""

from __future__ import annotations

import unittest

from hardening_release import build_manifest, evaluate_release


class HardeningReleaseTests(unittest.TestCase):
    def test_manifest_is_complete_and_oracle_verified(self):
        manifest = build_manifest()
        self.assertEqual(manifest["raw_candidate_count"], 116)
        self.assertEqual(manifest["static_count"], 112)
        self.assertEqual(manifest["interactive_count"], 4)
        self.assertEqual(manifest["profile_count"], 116)
        self.assertEqual(manifest["category_count"], 29)
        self.assertEqual(manifest["family_count"], 58)
        self.assertEqual(manifest["oracle_check"]["cases"], 116)
        self.assertTrue(manifest["manifest_sha256"])

    def test_no_endpoint_is_explicitly_pending(self):
        result = evaluate_release()
        self.assertEqual(result["status"], "pending_model_pilot")
        self.assertFalse(result["release_ready"])
        self.assertIn("matched hardening quality reports", result["blockers"][0])

    def test_three_matched_reports_can_enter_provisional_band(self):
        manifest = build_manifest()
        reports = []
        for run_id in range(3):
            reports.append(
                {
                    "run_id": run_id,
                    "suite_hash": manifest["suite_hash"],
                    "quality_config": {
                        "profile": "hardening-only",
                        "split": manifest["split"],
                        "variants": manifest["variants"],
                        "seeds": [manifest["seed"]],
                        "generated": False,
                        "interactive": True,
                        "strengthen_code": False,
                        "context_sizes": [],
                        "max_output_tokens": 65536,
                    },
                    "protocol": {
                        "quality_profile": "hardening-only",
                        "revision": "quality-v6",
                        "temperature": 0,
                        "model_seed": 0,
                        "default_max_tokens": 65536,
                        "quality_max_output_tokens": 65536,
                    },
                    "results": [
                        {
                            "id": f"{family}-development-v01",
                            "category": manifest["categories"][index // 2],
                            "family": family,
                            "metadata": {"cohort": "hardening-v7"},
                            "scored": True,
                            "passed": (run_id + index) % 2 == 0,
                            "score": 1.0 if (run_id + index) % 2 == 0 else 0.0,
                            "outcome": "pass" if (run_id + index) % 2 == 0 else "task_failure",
                        }
                        for index, family in enumerate(manifest["families"])
                    ],
                }
            )
        result = evaluate_release(reports)
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["release_ready"])
        self.assertEqual(result["pilot"]["categories_observed"], 29)
        self.assertEqual(len(result["pilot"]["selected_families"]), 58)


if __name__ == "__main__":
    unittest.main()
