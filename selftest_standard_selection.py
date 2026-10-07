"""Frozen selection, strict evidence, coverage and archive regressions."""

import copy
from dataclasses import replace
import json
import unittest

from app.benchmarking import quality_suite, standard_selection, standard_suite
from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.models import Question, RequestMetrics, Result
from app.benchmarking.quality_report import make_report
from select_standard_questions import propose


def question(qid, evaluator="json_match"):
    return Question(qid, "Test", "Return one JSON document", evaluator,
                    {"value": {"answer": 1}} if evaluator == "json_match" else None,
                    metadata={"cohort": quality_suite.REVISION, "family": qid})


def panel():
    questions = [question(qid) for qid in ("passed", "failed", "missing", "changed", "outage", "new")]
    questions.append(question("open", "open_ended"))
    runs = []
    for model in range(3):
        results = []
        for q in questions:
            if q.id == "new" or q.id == "missing" and model == 1:
                continue
            if q.id == "changed" and model == 2:
                q = replace(q, expected={"value": {"answer": 2}})
            failed, outage = q.id == "failed" and model == 2, q.id == "outage" and model == 1
            results.append(Result(q, "response", 0 if failed or outage else 1,
                                  metrics=RequestMetrics(ok=not outage),
                                  outcome="recorded" if q.evaluator == "open_ended" else
                                  "endpoint_error" if outage else "task_failure" if failed else "pass"))
        report = make_report(results, ClientConfig("http://fake.invalid/v1", "unused", f"model-{model}"))
        runs.append({"id": model + 1, "model_id": model + 1, "status": "completed",
                     "total_questions": len(results), "test_suite_hash": report["suite_hash"],
                     "quality_json": json.dumps(report)})
    return questions, runs


class SelectionTests(unittest.TestCase):
    def test_only_unchanged_unanimous_full_passes_are_removed(self):
        questions, runs = panel()
        proposed = propose(questions, runs)
        self.assertEqual([row["id"] for row in proposed["removed"]], ["passed"])
        self.assertEqual(proposed["retained_reasons"],
                         {"failure_or_unscored": 2, "new_or_missing": 2, "changed_identity": 1, "ungraded": 1})
        self.assertEqual([q.id for q in standard_selection.select_questions(questions, proposed)],
                         [q.id for q in questions if q.id != "passed"])

    def test_corrupt_panel_and_changed_source_are_rejected(self):
        questions, runs = panel()
        with self.assertRaisesRegex(ValueError, "unique"):
            propose(questions, [*runs, runs[0]])
        with self.assertRaisesRegex(ValueError, "three"):
            propose(questions, runs[:2])
        bad = copy.deepcopy(runs)
        bad[0]["status"] = "running"
        with self.assertRaisesRegex(ValueError, "run_not_completed"):
            propose(questions, bad)
        bad = copy.deepcopy(runs)
        for run in bad:
            report = json.loads(run["quality_json"])
            report["model"] = "same-model"
            run["quality_json"] = json.dumps(report)
        with self.assertRaisesRegex(ValueError, "recorded model"):
            propose(questions, bad)
        proposed = propose(questions, runs)
        changed = [replace(q, prompt="changed") if q.id == "passed" else q for q in questions]
        with self.assertRaisesRegex(ValueError, "source bank changed"):
            standard_selection.select_questions(changed, proposed)
        wrong_counts = copy.deepcopy(proposed)
        wrong_counts["retained_questions"] += 1
        with self.assertRaisesRegex(ValueError, "counts"):
            standard_selection.select_questions(questions, wrong_counts)
        proposed["removed"].append(proposed["removed"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            standard_selection.select_questions(questions, proposed)

    def test_frozen_inventory_keeps_new_ladder_and_archive(self):
        all_questions = standard_suite.load_all_questions()
        selected = standard_suite.load_questions()
        self.assertEqual((len(all_questions), len(selected)), (519, 339))
        self.assertEqual(quality_suite.suite_hash(selected), "0f69edef339e5ad3")
        self.assertEqual(sum(q.metadata.get("challenge") == "capability-ladder-v1" for q in selected), 24)
        self.assertEqual(sum(q.evaluator == "open_ended" for q in selected), 30)
        self.assertEqual(len(standard_selection.provenance()["dropped_categories"]), 6)
        provenance = standard_selection.provenance()
        provenance["panel"].clear()
        self.assertEqual(len(standard_selection.provenance()["panel"]), 4)
        from app.templates_config import templates

        html = templates.get_template("quality_macros.html").module.selection_note(standard_selection.provenance())
        for text in ("180 questions", "Confidentiality", "provisional", "retained tasks"):
            self.assertIn(text, html)


if __name__ == "__main__":
    unittest.main()
