#!/usr/bin/env python3
"""Regression checks for deterministic historical review queue construction."""

from __future__ import annotations

import unittest

from review_queue import build_review_queue


class ReviewQueueTests(unittest.TestCase):
    def test_nonpasses_runtime_failures_and_success_samples_are_separated(self):
        audit = {
            "audit_version": "test",
            "snapshot_started_at": "now",
            "runs": [
                {
                    "run_id": 1,
                    "items": [
                        {"id": "bad", "category": "C", "evaluator": "json_match", "scope": "capability", "scored": True, "passed": False, "score": 0, "outcome": "task_failure", "detail": "wrong"},
                        {"id": "loop", "category": "C", "evaluator": "json_match", "scope": "capability", "scored": False, "passed": False, "score": 0, "outcome": "repetition", "detail": "loop"},
                        {"id": "good", "category": "C", "evaluator": "json_match", "scope": "capability", "scored": True, "passed": True, "score": 1, "outcome": "pass", "detail": "ok"},
                    ],
                }
            ],
        }
        queue = build_review_queue(audit, success_samples_per_group=1)
        self.assertEqual(queue["summary"]["capability_nonpasses"], 1)
        self.assertEqual(queue["summary"]["runtime_failures"], 1)
        self.assertEqual(queue["summary"]["success_samples"], 1)
        self.assertEqual({row["id"] for row in queue["items"]}, {"bad", "loop", "good"})


if __name__ == "__main__":
    unittest.main()
