#!/usr/bin/env python3
"""Offline tests for open-ended questions, blind A/B studies, the pairwise judge and statistics."""

import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import random
import re
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config as app_config
from app.benchmarking import ab_stats, open_suite, pairwise_judge
from app.benchmarking.evaluators import EVALUATOR_VERSIONS
from app.benchmarking.models import Question, RequestMetrics, TokenUsage
from app.benchmarking.quality_execution import score_response
from app.benchmarking.quality_report import result_record, summarize
from app.benchmarking.quality_suite import suite_hash
from app.benchmarking.suite_checks import Report, validate_question
from app.benchmarking.suites import get_suite
from app.benchmarking.usecase_suites import UsecaseSuiteError, parse_suite
from app.services import ab_studies
from app.services import benchmark_runner as runner
from app.services.ab_studies import StudyError
from app.storage import DETECT_TYPES

ROOT = Path(__file__).parent
OPEN_HASH = "0a7491b1367eba4c"


def open_question(**fields):
    return Question(**{"id": "o1", "category": "Writing", "prompt": "Write a note.", "evaluator": "open_ended",
                       "expected": {"criteria": ["Short"]}, "metadata": {"scope": "open_ended"}, **fields})


def ok(finish="stop"):
    return RequestMetrics(latency_ms=5, ttft_ms=1, finish_reason=finish)


class OpenEndedTests(unittest.TestCase):
    def test_recorded_not_scored(self):
        result = score_response(open_question(), "A fine note.", TokenUsage(5, 3), ok())
        self.assertEqual((result.outcome, result.is_scored, result.passed), ("recorded", False, False))
        self.assertIsNone(result.evaluation)
        for response, finish, outcome in (("<think>hm</think>", "stop", "missing_answer"),
                                          ("cut", "length", "truncation")):
            failed = score_response(open_question(), response, TokenUsage(), ok(finish))
            self.assertEqual((failed.outcome, failed.is_scored), (outcome, False))
        self.assertNotIn("open_ended", EVALUATOR_VERSIONS)

    def test_summary_keeps_open_ended_apart(self):
        scored = Question("c1", "Math", "2+2", "exact_match", "4", metadata={"scope": "capability"})
        rows = [result_record(score_response(scored, "4", TokenUsage(), ok())),
                result_record(score_response(open_question(), "Note.", TokenUsage(), ok())),
                result_record(score_response(open_question(id="o2"), "", TokenUsage(), ok()))]
        summary = summarize(rows)
        self.assertEqual((summary["scored"], summary["passes"], summary["category_balanced"]), (1, 1, 1.0))
        self.assertEqual(list(summary["categories"]), ["Math"])
        self.assertEqual(summary["open_ended"], {"count": 2, "recorded": 1, "categories": ["Writing"],
                                                 "outcomes": {"recorded": 1, "missing_answer": 1}})
        self.assertNotIn("open_ended", summarize(rows[:1]))

    def test_builtin_suite(self):
        questions = get_suite("assistant-open").load()
        self.assertEqual(len(questions), 30)
        self.assertEqual(suite_hash(questions), OPEN_HASH)
        self.assertEqual(suite_hash(open_suite.load_questions()), OPEN_HASH)
        self.assertEqual({q.metadata["lang"] for q in questions}, {"en", "cs", "de"})
        by_category = {}
        for q in questions:
            by_category[q.category] = by_category.get(q.category, 0) + 1
            self.assertEqual((q.evaluator, q.metadata["scope"], q.system_prompt), ("open_ended", "open_ended", open_suite.SYSTEM))
            self.assertTrue(q.id.endswith(q.metadata["lang"]))
        self.assertEqual(set(by_category.values()), {5})
        report = Report()
        for q in questions:
            validate_question(report, q)
        self.assertEqual((report.errors, report.warnings), ([], []))

    def test_usecase_suites_accept_open_ended(self):
        doc = {"suite": {"slug": "team-open", "name": "Team"}, "questions": [
            {"id": "w1", "category": "Writing", "prompt": "Write.", "evaluator": "open_ended",
             "expected": {"criteria": ["Polite"], "reference": "Dear team"}}]}
        q = parse_suite(json.dumps(doc)).questions[0]
        self.assertEqual(q.metadata["scope"], "open_ended")
        for expected, needle in (({"criteria": []}, "criteria"), ({"criteria": ["x"], "notes": 1}, "mapping"),
                                 ({"criteria": ["x"], "reference": ""}, "reference")):
            doc["questions"][0]["expected"] = expected
            with self.assertRaises(UsecaseSuiteError) as caught:
                parse_suite(json.dumps(doc))
            self.assertTrue(any(needle in p for p in caught.exception.problems), caught.exception.problems)


# --- Database fixtures -------------------------------------------------------------

class FakeModelClient:
    """alpha answers briefly behind a reasoning block; beta answers at length and fails one request."""

    def __init__(self, config):
        self.config = config

    def complete(self, prompt, system_prompt=None, **_kwargs):
        if self.config.model == "beta" and "Kubernetes" in prompt:
            return "[API ERROR]", TokenUsage(), RequestMetrics(ok=False, error="HTTP 500", attempts=1)
        text = ("<think>pondering</think>Short answer." if self.config.model == "alpha"
                else "A longer, more detailed answer with steps and examples.")
        return text, TokenUsage(10, 5), ok()


ANSWER = re.compile(r"<answer_(\d)>\n(.*?)\n</answer_\1>", re.S)


class FakeJudge:
    """Prefers the detailed answer; returns junk for the 8D request while ``fail`` is set."""

    fail = False
    calls = 0
    on_call = None

    def __init__(self, config):
        self.config = config
        self.session = Mock()

    def complete(self, prompt, system_prompt=None, **_kwargs):
        type(self).calls += 1
        if FakeJudge.on_call:
            FakeJudge.on_call()
        if FakeJudge.fail and "8D" in prompt:
            return "I cannot decide.", TokenUsage(), ok()
        answers = dict(ANSWER.findall(prompt))
        winner = "1" if "detailed" in answers["1"] else "2"
        return f'<think>compare</think>```json\n{{"reason": "More complete.", "winner": "{winner}"}}\n```', TokenUsage(), ok()


class DatabaseCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        self.path = Path(directory) / "bench.db"
        self.stack.enter_context(patch.object(app_config, "DATABASE_PATH", self.path))
        self.stack.enter_context(patch("app.services.url_guard.validate_endpoint"))
        self.stack.enter_context(patch("app.benchmarking.llm_client.ChatClient", FakeModelClient))
        db = sqlite3.connect(self.path)
        db.executescript((ROOT / "app/schema.sql").read_text(encoding="utf-8"))
        db.execute("INSERT INTO users (id, username, role) VALUES (1, 'admin', 'admin'), (2, 'ann', 'user'), (3, 'bob', 'user')")
        for i, name in ((1, "alpha"), (2, "beta"), (3, "judge")):
            db.execute("INSERT INTO models (id, name, base_url, api_key, model_id) VALUES (?, ?, 'http://x.invalid', 'k', ?)",
                       (i, name, name))
        db.executemany("INSERT INTO test_runs (id, model_id, status) VALUES (?, ?, 'pending')", [(1, 1), (2, 2)])
        db.commit()
        db.close()
        FakeJudge.fail, FakeJudge.calls, FakeJudge.on_call = False, 0, None
        for run_id, model in ((1, "alpha"), (2, "beta")):
            runner._run_benchmark(run_id, {"id": run_id, "name": model, "base_url": "http://x.invalid",
                                           "api_key": "k", "model_id": model},
                                  "quality", 2, None, suite="assistant-open")

    def sql(self, query, params=()):
        db = sqlite3.connect(self.path, detect_types=DETECT_TYPES)
        db.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in db.execute(query, params).fetchall()]
            db.commit()
            return rows
        finally:
            db.close()

    def judge(self, study_id, fail=False, workers=1):
        FakeJudge.fail = fail
        model = self.sql("SELECT * FROM models WHERE id = 3")[0]
        return ab_studies.start_judge(study_id, model, spawn=lambda judge_id, config: pairwise_judge.run_judge(
            judge_id, config, client_factory=FakeJudge, workers=workers))


class StudyTests(DatabaseCase):
    def test_open_ended_only_runs_complete(self):
        for run_id, recorded in ((1, 30), (2, 29)):
            run = self.sql("SELECT * FROM test_runs WHERE id = ?", (run_id,))[0]
            self.assertEqual(run["status"], "completed", run["error_message"])
            summary = json.loads(run["quality_json"])["summary"]
            self.assertEqual((summary["scored"], summary["open_ended"]["recorded"]), (0, recorded))
            self.assertEqual(json.loads(run["quality_config_json"])["name"], "assistant-open")

    def test_create_pairs_by_fingerprint(self):
        study_id = ab_studies.create_study("", 1, 2, "open_ended", 1)
        study = ab_studies.get_study(study_id)
        self.assertEqual((study["label_a"], study["label_b"], study["suite_name"]),
                         ("alpha · run #1", "beta · run #2", "assistant-open"))
        self.assertEqual(study["name"], "alpha · run #1 vs beta · run #2")
        pairs = ab_studies.study_pairs(study_id)
        self.assertEqual(len(pairs), 29)
        self.assertFalse(any("Kubernetes" in p["prompt"] for p in pairs))
        self.assertEqual(pairs, sorted(pairs, key=lambda p: (p["category"], p["question_id"])))
        first = pairs[0]
        self.assertEqual((first["answer_a"], first["system_prompt"]), ("Short answer.", open_suite.SYSTEM))
        self.assertTrue(first["criteria"])
        self.assertTrue(any(p["reference"] for p in pairs))
        self.assertEqual(len(ab_studies.study_pairs(ab_studies.create_study("x", 1, 2, "all", 1))), 29)
        with patch.object(ab_studies, "MAX_PAIRS", 5):
            capped = [p["question_id"] for p in ab_studies.study_pairs(ab_studies.create_study("c", 2, 1, "all", 1))]
            again = [p["question_id"] for p in ab_studies.study_pairs(ab_studies.create_study("c", 2, 1, "all", 1))]
        self.assertEqual((len(capped), capped), (5, again))

    def test_creation_rules(self):
        self.sql("INSERT INTO test_runs (id, model_id, status, quality_config_json) VALUES (3, 1, 'completed', ?)",
                 (json.dumps({"name": "tool-conformance"}),))
        self.sql("INSERT INTO test_runs (id, model_id, status) VALUES (4, 1, 'running')")
        for args, needle in (((1, 1, "all"), "two different"), ((1, 3, "all"), "different suites"),
                             ((1, 4, "all"), "not completed"), ((1, 9, "all"), "does not exist"),
                             ((1, 2, "bogus"), "scope")):
            with self.subTest(needle), self.assertRaisesRegex(StudyError, needle):
                ab_studies.create_study("", *args, 1)
        self.assertEqual(self.sql("SELECT COUNT(*) AS n FROM ab_studies")[0]["n"], 0)

    def test_votes_are_blind_balanced_and_final(self):
        study_id = ab_studies.create_study("", 1, 2, "open_ended", 1)
        rng = random.Random(3)
        pair, left = ab_studies.next_pair(study_id, 2, rng=rng)
        self.assertEqual(ab_studies.record_vote(study_id, pair["id"], 2, "left", left), left)
        with self.assertRaisesRegex(StudyError, "already voted"):
            ab_studies.record_vote(study_id, pair["id"], 2, "right", left)
        # Bob gets an unvoted pair first; Ann never sees her pair again.
        for _ in range(20):
            self.assertNotEqual(ab_studies.next_pair(study_id, 3, rng=rng)[0]["id"], pair["id"])
            self.assertNotEqual(ab_studies.next_pair(study_id, 2, rng=rng)[0]["id"], pair["id"])
        self.assertEqual(ab_studies.record_vote(study_id, pair["id"], 3, "right", "b"), "a")
        self.assertEqual(ab_studies.record_vote(study_id, pair["id"] + 1, 3, "both_bad", "a", " meh "), "both_bad")
        skipped = ab_studies.next_pair(study_id, 3, skip=pair["id"] + 2, rng=rng)[0]
        self.assertNotEqual(skipped["id"], pair["id"] + 2)
        self.assertEqual(ab_studies.vote_progress(study_id, 3), (2, 29))
        for args, needle in (((pair["id"] + 3, 3, "maybe", "a"), "Choose"), ((pair["id"] + 3, 3, "left", "c"), "order"),
                             ((pair["id"] + 3, 3, "left", "a", "x" * 501), "500"), ((10_000, 3, "left", "a"), "belong")):
            with self.subTest(needle), self.assertRaisesRegex(StudyError, needle):
                ab_studies.record_vote(study_id, *args)
        ab_studies.set_status(study_id, "closed")
        with self.assertRaisesRegex(StudyError, "closed"):
            ab_studies.record_vote(study_id, pair["id"] + 3, 3, "left", "a")
        self.assertEqual([v["comment"] for v in ab_studies.votes(study_id)], [None, None, "meh"])
        # Deleted users' votes stay, anonymously.
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("DELETE FROM users WHERE id = 3")
        db.commit()
        db.close()
        self.assertEqual(len(ab_studies.votes(study_id)), 3)
        self.assertEqual(self.sql("SELECT COUNT(*) AS n FROM ab_votes WHERE user_id IS NULL")[0]["n"], 2)

    def test_judge_progress_errors_resume_and_cancel(self):
        study_id = ab_studies.create_study("", 1, 2, "open_ended", 1)
        judge_id = self.judge(study_id, fail=True)
        judge = ab_studies.judges(study_id)[0]
        self.assertEqual((judge["status"], judge["done"], judge["errors"], judge["total"]), ("completed", 28, 1, 29))
        self.assertIn("start the judge again", judge["error"])
        row = self.sql("SELECT * FROM ab_judgments WHERE verdict IS NOT NULL LIMIT 1")[0]
        self.assertEqual((row["first"], row["second"], row["verdict"], row["consistent"]), ("b", "b", "b", 1))
        self.assertEqual(row["reason_first"], "More complete.")
        calls = FakeJudge.calls
        self.assertEqual(self.judge(study_id), judge_id)  # resume re-judges only the failed pair
        self.assertEqual(FakeJudge.calls - calls, 2)
        judge = ab_studies.judges(study_id)[0]
        self.assertEqual((judge["status"], judge["done"], judge["errors"], judge["error"]), ("completed", 29, 0, None))
        self.assertEqual(len(ab_studies.judges(study_id)), 1)

        other = ab_studies.create_study("", 1, 2, "open_ended", 1)
        FakeJudge.on_call = lambda: ab_studies.cancel_judge(other, judge_id + 1)
        self.judge(other)
        cancelled = ab_studies.judges(other)[0]
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertLess(cancelled["done"], 29)
        FakeJudge.on_call = None

    def test_consecutive_judge_errors_stop_and_restart_interrupts(self):
        study_id = ab_studies.create_study("", 1, 2, "open_ended", 1)
        with patch.object(FakeJudge, "complete", return_value=("", TokenUsage(), RequestMetrics(ok=False, error="HTTP 503"))):
            self.judge(study_id)
        judge = ab_studies.judges(study_id)[0]
        self.assertEqual((judge["status"], judge["errors"], judge["done"]), ("failed", pairwise_judge.STOP_AFTER_ERRORS, 0))
        self.assertIn("consecutive judge errors", judge["error"])
        with self.assertRaisesRegex(StudyError, "already running"):
            self.sql("UPDATE ab_judges SET status = 'running'")
            self.judge(study_id)
        from app import database
        asyncio.run(database.init_db())
        self.assertEqual(ab_studies.judges(study_id)[0]["status"], "interrupted")
        with self.assertRaisesRegex(StudyError, "Cancel"):
            self.sql("UPDATE ab_judges SET status = 'running'")
            ab_studies.delete_study(study_id)

    def test_results_combine_people_and_judges(self):
        study_id = ab_studies.create_study("", 1, 2, "open_ended", 1)
        self.judge(study_id)
        pairs = ab_studies.study_pairs(study_id)
        for pair in pairs[:12]:
            ab_studies.record_vote(study_id, pair["id"], 2, "right", "a")  # B
        ab_studies.record_vote(study_id, pairs[12]["id"], 2, "left", "a")  # A
        summary = ab_studies.results(study_id)
        people, judge = summary["people"], summary["judges"][0]
        self.assertEqual((people["b"], people["a"], people["votes"]), (12, 1, 13))
        self.assertEqual(judge["preference"]["verdict"], "b")
        self.assertEqual(judge["preference"]["preference_b"], 1.0)
        self.assertEqual(judge["diagnostics"]["position_consistency"], 1.0)
        self.assertEqual(judge["diagnostics"]["longer_answer_win_rate"], 1.0)
        self.assertEqual((judge["agreement"]["pairs"], judge["agreement"]["agreement"]), (13, 12 / 13))
        exported = ab_studies.export(study_id)
        self.assertEqual(len(exported["pairs"]), 29)
        self.assertEqual(exported["pairs"][0]["judgments"][0]["verdict"], "b")
        self.assertNotIn("user_id", json.dumps(exported))


class JudgeUnitTests(unittest.TestCase):
    PAIR = {"prompt": "Write a memo.", "system_prompt": "Be brief.", "criteria": ["Polite"], "reference": "Memo",
            "answer_a": "AAA", "answer_b": "BBB"}

    def test_prompt(self):
        ab = pairwise_judge.judge_prompt(self.PAIR, "ab")
        ba = pairwise_judge.judge_prompt(self.PAIR, "ba")
        for text in ("Be brief.", "Write a memo.", "- Polite", "Memo", '"winner"'):
            self.assertIn(text, ab)
        self.assertIn("<answer_1>\nAAA\n</answer_1>", ab)
        self.assertIn("<answer_1>\nBBB\n</answer_1>", ba)
        plain = pairwise_judge.judge_prompt({**self.PAIR, "criteria": None, "reference": None, "system_prompt": None}, "ab")
        self.assertIn(pairwise_judge.DEFAULT_CRITERIA[1], plain)
        self.assertNotIn("Reference", plain)
        long = pairwise_judge.judge_prompt({**self.PAIR, "answer_b": "x" * 30_000}, "ab")
        self.assertIn("Answer 2 was truncated to 24,000 characters", long)
        self.assertNotIn("x" * 24_001, long)

    def test_parse_and_combine(self):
        for text, winner in (('{"reason": "r", "winner": "1"}', "1"), ('```json\n{"winner": 2}\n```', "2"),
                             ('<think>x</think>{"winner": "Answer 2", "reason": "y"}', "2"),
                             ('Verdict: {"winner": "TIE"}', "tie")):
            self.assertEqual(pairwise_judge.parse_verdict(text)[0], winner, text)
        for text in ("no json", '{"winner": "3"}', '{"reason": "x"}', '["1"]'):
            with self.assertRaises(ValueError):
                pairwise_judge.parse_verdict(text)
        self.assertEqual([pairwise_judge.to_side(w, o) for w, o in (("1", "ab"), ("2", "ab"), ("1", "ba"), ("2", "ba"), ("tie", "ba"))],
                         ["a", "b", "b", "a", "tie"])
        for first, second, expected in (("a", "a", ("a", True)), ("a", "b", ("tie", False)),
                                        ("a", "tie", ("a", False)), ("tie", "b", ("b", False)), ("tie", "tie", ("tie", True))):
            self.assertEqual(pairwise_judge.combine(first, second), expected)

    def test_judge_pair_failures(self):
        client = Mock()
        client.complete.side_effect = [('{"winner": "1"}', TokenUsage(), ok()), ("junk", TokenUsage(), ok())]
        out = pairwise_judge.judge_pair(client, {**self.PAIR, "id": 1})
        self.assertEqual((out["first"], out["verdict"]), ("a", None))
        self.assertIn("Unusable judge output (ba)", out["error"])
        client.complete.side_effect = [("", TokenUsage(), RequestMetrics(ok=False, error="timeout"))]
        self.assertIn("timeout", pairwise_judge.judge_pair(client, self.PAIR)["error"])


class StatsTests(unittest.TestCase):
    def items(self, values):
        return [{"pair_id": i, "category": c, "family": f"{c}{i}", "value": v} for i, (c, v) in enumerate(values)]

    def test_sign_test(self):
        self.assertAlmostEqual(ab_stats.sign_test(8, 2), 2 * (1 + 10 + 45) / 1024)
        self.assertEqual((ab_stats.sign_test(5, 5), ab_stats.sign_test(0, 0)), (1.0, None))

    def test_preference(self):
        clear = ab_stats.preference(self.items([("x", 1)] * 5 + [("y", 1)] * 5))
        self.assertEqual((clear["verdict"], clear["preference_b"], clear["ci95"]), ("b", 1.0, [1.0, 1.0]))
        self.assertAlmostEqual(clear["p_value"], 2 / 1024)
        mixed = ab_stats.preference(self.items([("x", 1), ("x", 0), ("y", 1), ("y", 1), ("y", 0.5)]))
        self.assertAlmostEqual(mixed["preference_b"], (0.5 + 2.5 / 3) / 2)
        self.assertEqual((mixed["a"], mixed["b"], mixed["tie"], mixed["verdict"]), (1, 3, 1, "unclear"))
        self.assertEqual(mixed["categories"]["y"], {"pairs": 3, "preference_b": 2.5 / 3, "a": 0, "b": 2, "tie": 1})
        self.assertEqual(ab_stats.preference([])["verdict"], "no_data")
        towards_a = ab_stats.preference(self.items([("x", 0)] * 12))
        self.assertEqual(towards_a["verdict"], "a")

    def test_votes_kappa_and_length_bias(self):
        pairs = [{"id": i, "category": "c", "family": str(i)} for i in range(4)]
        votes = [{"pair_id": 0, "verdict": "a"}, {"pair_id": 0, "verdict": "b"},
                 {"pair_id": 1, "verdict": "b"}, {"pair_id": 1, "verdict": "b"}, {"pair_id": 1, "verdict": "a"},
                 {"pair_id": 2, "verdict": "both_bad"}]
        human = {i["pair_id"]: i for i in ab_stats.human_items(pairs, votes)}
        self.assertEqual((human[0]["value"], human[0]["label"]), (0.5, "tie"))
        self.assertAlmostEqual(human[1]["value"], 2 / 3)
        self.assertEqual((human[1]["label"], human[2]["label"], 3 in human), ("b", "tie", False))
        judge = [{"pair_id": i, "label": label} for i, label in enumerate("aabb")]
        people = [{"pair_id": i, "label": label} for i, label in enumerate("abbb")]
        self.assertEqual(ab_stats.agreement(judge, people), {"pairs": 4, "agreement": 0.75, "kappa": 0.5})
        self.assertEqual(ab_stats.agreement(judge, [])["kappa"], None)
        diag = ab_stats.judge_diagnostics(
            [{"pair_id": 0, "label": "a", "consistent": True}, {"pair_id": 1, "label": "b", "consistent": False},
             {"pair_id": 2, "label": "tie", "consistent": True}, {"pair_id": 3, "label": "a", "consistent": True}],
            {0: (10, 5), 1: (10, 5), 2: (1, 2), 3: (4, 4)})
        self.assertEqual(diag, {"position_consistency": 0.75, "decisive_with_length_difference": 2,
                                "longer_answer_win_rate": 0.5})


class PageTests(DatabaseCase):
    def setUp(self):
        super().setUp()
        from app.routes import compare, studies

        self.role = "admin"

        def current():
            return {"id": 1 if self.role == "admin" else 2, "role": self.role, "username": self.role}

        async def lookup(_request):
            return None if self.role == "anonymous" else current()

        self.stack.enter_context(patch.object(studies, "get_current_user", AsyncMock(side_effect=lookup)))
        self.stack.enter_context(patch.object(compare, "get_current_user", AsyncMock(side_effect=lookup)))
        self.stack.enter_context(patch.object(pairwise_judge, "start_judge_thread"))
        app = FastAPI()

        @app.middleware("http")
        async def user(request, call_next):
            request.state.user = None if self.role == "anonymous" else current()
            return await call_next(request)

        app.include_router(studies.router)
        app.include_router(compare.router)
        self.client = self.stack.enter_context(TestClient(app))

    def test_admin_flow_and_blind_voting(self):
        compare_page = self.client.get("/compare?runs=1,2").text
        self.assertIn("/studies?run_a=1&amp;run_b=2", compare_page)
        form = self.client.get("/studies?run_a=1&run_b=2").text
        self.assertIn("Create a study", form)
        self.assertIn('value="1" selected', form)
        bad = self.client.post("/studies", data={"run_a": 1, "run_b": 1, "scope": "open_ended"})
        self.assertEqual(bad.status_code, 422)
        self.assertIn("two different runs", bad.text)
        created = self.client.post("/studies", data={"run_a": 1, "run_b": 2, "scope": "open_ended", "name": "Pilot"},
                                   follow_redirects=False)
        self.assertEqual((created.status_code, created.headers["location"]), (303, "/studies/1"))
        detail = self.client.get("/studies/1").text
        self.assertIn("Pilot", detail)
        self.assertIn("No verdicts yet", detail)
        self.assertIn("This model wrote one of the answers", detail)
        self.assertEqual(self.client.post("/studies/1/judges", data={"model_id": 3}, follow_redirects=False).status_code, 303)
        self.assertEqual(ab_studies.judges(1)[0]["status"], "running")
        self.assertIn('http-equiv="refresh"', self.client.get("/studies/1").text)
        self.client.post(f"/studies/1/judges/{ab_studies.judges(1)[0]['id']}/cancel")
        self.assertEqual(ab_studies.judges(1)[0]["status"], "cancelled")

        self.role = "user"
        vote = self.client.get("/studies/1/vote")
        self.assertEqual(vote.status_code, 200)
        for hidden in ("alpha", "beta", "run #"):
            self.assertNotIn(hidden, vote.text)
        pair_id = int(re.search(r'name="pair_id" value="(\d+)"', vote.text)[1])
        shown_left = re.search(r'name="shown_left" value="(a|b)"', vote.text)[1]
        posted = self.client.post("/studies/1/vote", data={"pair_id": pair_id, "choice": "left", "shown_left": shown_left,
                                                          "comment": "<b>clear</b>"}, follow_redirects=False)
        self.assertEqual((posted.status_code, posted.headers["location"]), (303, "/studies/1/vote"))
        again = self.client.post("/studies/1/vote", data={"pair_id": pair_id, "choice": "left", "shown_left": shown_left},
                                 follow_redirects=False)
        self.assertIn("error=You%20have%20already%20voted", again.headers["location"])
        self.assertIn("You have voted on 1 of 29", self.client.get("/studies/1/vote").text)
        pair_page = self.client.get(f"/studies/1/pairs/{pair_id}").text
        self.assertIn("&lt;b&gt;clear&lt;/b&gt;", pair_page)
        self.assertIn("alpha · run #1", pair_page)
        export = self.client.get("/studies/1/export.json").json()
        self.assertEqual(sum(len(p["votes"]) for p in export["pairs"]), 1)
        for path, data in (("/studies", {"run_a": 1, "run_b": 2}), ("/studies/1/judges", {"model_id": 3}),
                           ("/studies/1/status", {"status": "closed"}), ("/studies/1/delete", {})):
            self.assertEqual(self.client.post(path, data=data, follow_redirects=False).status_code, 403, path)
        self.assertNotIn("Create a study", self.client.get("/studies").text)
        self.assertNotIn("Create blind A/B study", self.client.get("/compare?runs=1,2").text)

        self.role = "admin"
        self.client.post("/studies/1/status", data={"status": "closed"})
        self.assertIn("closed for voting", self.client.get("/studies/1/vote").text)
        self.assertEqual(self.client.post("/studies/1/delete", follow_redirects=False).headers["location"], "/studies")
        self.assertEqual(self.sql("SELECT COUNT(*) AS n FROM ab_votes")[0]["n"], 0)
        self.assertEqual(self.client.get("/studies/1").status_code, 404)

    def test_login_required(self):
        self.role = "anonymous"
        for path in ("/studies", "/studies/1", "/studies/1/vote"):
            response = self.client.get(path, follow_redirects=False)
            self.assertEqual((response.status_code, response.headers["location"]), (303, "/login"), path)


if __name__ == "__main__":
    unittest.main()
