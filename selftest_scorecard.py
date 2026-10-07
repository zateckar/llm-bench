#!/usr/bin/env python3
"""Offline tests for model scorecards, decision profiles and decision records."""

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config as app_config
from app.benchmarking import decision_gates
from app.benchmarking.decision_gates import GateError
from app.services import scorecard
from app.services.scorecard import ScorecardError
from app.storage import DETECT_TYPES, pack_text
from selftest_monitoring import report

ROOT = Path(__file__).parent
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
PRESETS = {"chat", "agent", "rag", "mixed"}


def closed_loop(ttft_p95=400.0, output=40.0, levels=(1, 4)):
    return {"schema_version": 3, "protocol": {"input_reference_tokens": 1000, "max_output_tokens": 256},
            "concurrency": [{"concurrency": c, "requests": 10, "errors": 0, "output_tokens_per_sec": output * c,
                             "ttft": {"p50_ms": ttft_p95 / 2, "p95_ms": ttft_p95 * c},
                             "latency": {"p95_ms": 3000.0 * c}, "request_tokens_per_sec": {"p50": output / c}}
                            for c in levels]}


def open_loop(rate=2.0, status="established", preset="chat"):
    return {"schema_version": 5, "kind": "open_loop",
            "protocol": {"preset": preset, "workload_hash": "ab" * 8, "attainment_target": 0.95},
            "summary": {"sustainable_rate": rate, "sustainable_status": status, "goodput_at_sustainable": rate}}


EVIDENCE = {
    "quality": {"rigorous": {"run_id": 1, "freshness": "current", "score": 0.72, "score_ci95": [0.65, 0.79],
                             "full_pass": 0.5, "full_pass_ci95": None,
                             "categories": {"Code": {"score": 0.9, "full_pass": 0.8}}}},
    "capacity": {"chat": {"run_id": 2, "freshness": "current", "rate": 2.0, "status": "lower_bound"}},
    "latency": {"run_id": 3, "freshness": "stale", "levels": {1: {"ttft_p95": 800.0, "latency_p95": 3000.0,
                                                                  "request_output": 35.0}}},
    "context": None,
    "declared_context": {"tokens": 32768, "freshness": "current"},
    "operations": {"open_alerts": 0, "canaries": [{"name": "c", "status": "ok"}]},
}


class GateTests(unittest.TestCase):
    def validate(self, gates):
        return decision_gates.validate(gates, suite_exists=lambda s: s in {"rigorous", "tool-conformance"},
                                       presets=PRESETS)

    def test_validation(self):
        gates = self.validate([
            {"type": "quality", "suite": "rigorous", "threshold": "0.7", "category": ""},
            {"type": "capacity", "workload": "custom:" + "ab" * 8, "threshold": 1},
            {"type": "latency", "metric": "ttft_p95", "concurrency": "4", "threshold": 2000},
            {"type": "context", "threshold": 32768.0}, {"type": "operations", "ignored": 1}])
        self.assertEqual(gates[0], {"type": "quality", "suite": "rigorous", "metric": "score", "category": None,
                                    "threshold": 0.7})
        self.assertEqual((gates[2]["concurrency"], gates[3]["threshold"], gates[4]), (4, 32768, {"type": "operations"}))
        for gates, needle in (([], "at least one"), ([{"type": "x"}], "unknown gate"),
                              ([{"type": "quality", "suite": "assistant-open", "threshold": 0.5}], "suite"),
                              ([{"type": "quality", "suite": "rigorous", "threshold": 70}], "between"),
                              ([{"type": "quality", "suite": "rigorous", "threshold": True}], "number"),
                              ([{"type": "capacity", "workload": "custom:zz", "threshold": 1}], "workload"),
                              ([{"type": "latency", "metric": "ttft_p95", "concurrency": 1.5, "threshold": 1}], "whole"),
                              ([{"type": "latency", "metric": "p50", "threshold": 1}], "metric"),
                              ([{"type": "context", "threshold": float("nan")}], "between"),
                              ([{"type": "operations"}] * 21, "at most")):
            with self.subTest(needle), self.assertRaisesRegex(GateError, needle):
                self.validate(gates)
        self.assertEqual(decision_gates.describe(gates_q := self.validate(
            [{"type": "quality", "suite": "rigorous", "category": "Code", "metric": "full_pass", "threshold": 0.8}])[0],
            {"rigorous": "Rigorous"}), "Rigorous · Code: full pass ≥ 80%")
        self.assertEqual(gates_q["category"], "Code")
        self.assertEqual(decision_gates.describe({"type": "latency", "metric": "request_output", "concurrency": 4,
                                                  "threshold": 30}), "per-request output p50 at concurrency 4 ≥ 30 tok/s")

    def test_evaluation_paths(self):
        def one(gate, evidence=EVIDENCE):
            return decision_gates.evaluate([gate], evidence)["results"][0]

        passed = one({"type": "quality", "suite": "rigorous", "metric": "score", "threshold": 0.7})
        self.assertEqual((passed["status"], passed["uncertain"], passed["run_id"]), ("pass", True, 1))
        self.assertFalse(one({"type": "quality", "suite": "rigorous", "metric": "score", "threshold": 0.6})["uncertain"])
        self.assertEqual(one({"type": "quality", "suite": "rigorous", "metric": "full_pass", "threshold": 0.6})["status"], "fail")
        self.assertEqual(one({"type": "quality", "suite": "rigorous", "metric": "score", "category": "Code",
                              "threshold": 0.85})["status"], "pass")
        self.assertEqual(one({"type": "quality", "suite": "rigorous", "metric": "score", "category": "Nope",
                              "threshold": 0.5})["status"], "missing")
        self.assertEqual(one({"type": "quality", "suite": "tool-conformance", "metric": "score",
                              "threshold": 0.5})["status"], "missing")
        enough = one({"type": "capacity", "workload": "chat", "threshold": 2})
        self.assertEqual((enough["status"], enough["uncertain"]), ("pass", False))
        bound = one({"type": "capacity", "workload": "chat", "threshold": 3})
        self.assertEqual((bound["status"], bound["uncertain"]), ("fail", True))
        none = one({"type": "capacity", "workload": "chat", "threshold": 1},
                   {"capacity": {"chat": {"rate": None, "status": "not_met", "freshness": "current"}}})
        self.assertEqual((none["status"], none["note"]), ("fail", "No offered rate met the SLO"))
        stale = one({"type": "latency", "metric": "ttft_p95", "concurrency": 1, "threshold": 1000})
        self.assertEqual((stale["status"], stale["outcome"]), ("stale", "pass"))
        self.assertEqual(one({"type": "latency", "metric": "ttft_p95", "concurrency": 8, "threshold": 1})["status"], "missing")
        speed = one({"type": "latency", "metric": "request_output", "concurrency": 1, "threshold": 40},
                    {"latency": {"freshness": "current", "levels": {"1": {"request_output": 35.0}}}})
        self.assertEqual(speed["status"], "fail")
        declared = one({"type": "context", "threshold": 32768})
        self.assertEqual((declared["status"], declared["note"]), ("pass", "Context declared by the deployment"))
        measured = one({"type": "context", "threshold": 65536},
                       {**EVIDENCE, "context": {"tokens": 65536, "freshness": "unverified", "run_id": 9}})
        self.assertEqual((measured["status"], measured["note"], measured["run_id"]), ("pass", "Context measured", 9))
        self.assertEqual(one({"type": "context", "threshold": 1}, {})["status"], "missing")
        self.assertEqual(one({"type": "operations"})["status"], "pass")
        self.assertEqual(one({"type": "operations"}, {"operations": {"open_alerts": 2}})["status"], "fail")
        self.assertEqual(one({"type": "operations"}, {"operations": {"canaries": [{"name": "c", "status": "warning"}]}})["status"], "fail")
        self.assertEqual(one({"type": "operations"}, {"operations": {"canaries": [{"name": "c", "status": None}]}})["status"], "missing")

        status = lambda *s: decision_gates.verdict([{"status": x} for x in s])  # noqa: E731
        self.assertEqual((status("pass"), status("pass", "stale"), status("missing", "fail"), status()),
                         ("meets", "incomplete", "fails", "incomplete"))

    def test_users_gate(self):
        gate = decision_gates.validate([{"type": "users", "user_model": "mixed", "context": "131072", "threshold": 50}],
                                       suite_exists=lambda s: False, presets=PRESETS, user_presets={"mixed"})[0]
        self.assertEqual(gate, {"type": "users", "user_model": "mixed", "context": 131072, "threshold": 50})
        self.assertEqual(decision_gates.describe(gate, user_labels={"mixed": "Mixed"}),
                         "Users at SLO on Mixed with sessions up to 131,072 tokens ≥ 50")
        for raw, needle in (({"user_model": "bogus"}, "user-model"), ({"threshold": 1.5}, "whole"),
                            ({"context": 0}, "between")):
            with self.subTest(needle), self.assertRaisesRegex(GateError, needle):
                decision_gates.validate([{**gate, **raw}], suite_exists=lambda s: False, presets=PRESETS,
                                        user_presets={"mixed"})
        results = [{"context_cap": 32768, "users": 90, "status": "established"},
                   {"context_cap": 131072, "users": 60, "status": "established"},
                   {"context_cap": 262144, "users": 40, "status": "lower_bound"}]
        evidence = {"users": {"mixed": {"results": results, "freshness": "current", "run_id": 7}}}

        def one(context, threshold, item=evidence):
            return decision_gates.evaluate([{**gate, "context": context, "threshold": threshold}], item)["results"][0]

        # Evidence must measure this cap, without inheriting a failure from another cap.
        self.assertEqual([(r["status"], r["value"]) for r in (one(131072, 50), one(131072, 70), one(32768, 70))],
                         [("pass", 60), ("fail", 60), ("pass", 90)])
        bound = one(262144, 50)
        self.assertEqual((bound["status"], bound["uncertain"], bound["run_id"]), ("fail", True, 7))
        self.assertEqual(one(500_000, 1)["status"], "missing")
        self.assertEqual(one(100_000, 50)["status"], "missing")
        self.assertEqual(one(200_000, 50)["status"], "missing")
        self.assertEqual(one(1, 1, {})["status"], "missing")
        not_met = {"users": {"mixed": {"results": [{"context_cap": 8192, "users": 0, "status": "not_met"}]}}}
        self.assertEqual((one(8192, 1, not_met)["status"], one(8192, 1, not_met)["value"]), ("fail", 0))
        not_met["users"]["mixed"]["results"][0]["inherited"] = True
        self.assertEqual(one(8192, 1, not_met)["status"], "missing")
        adapted = scorecard._sessions({"summary": {"results": not_met["users"]["mixed"]["results"]}})
        self.assertTrue(adapted["results"][0]["inherited"])

    def test_verdicts(self):
        status = lambda *s: decision_gates.verdict([{"status": x} for x in s])  # noqa: E731
        self.assertEqual((status("pass"), status("pass", "stale"), status("missing", "fail"), status()),
                         ("meets", "incomplete", "fails", "incomplete"))


class DatabaseCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        self.path = Path(directory) / "bench.db"
        self.stack.enter_context(patch.object(app_config, "DATABASE_PATH", self.path))
        scorecard.clear_caches()
        self.addCleanup(scorecard.clear_caches)
        db = sqlite3.connect(self.path)
        db.executescript((ROOT / "app/schema.sql").read_text(encoding="utf-8"))
        db.execute("INSERT INTO users (id, username, role) VALUES (1, 'admin', 'admin'), (2, 'viewer', 'user')")
        db.executemany("INSERT INTO models (id, name, base_url, api_key, model_id) VALUES (?, ?, 'https://h/v1', 'k', ?)",
                       [(1, "alpha", "alpha-7b"), (2, "beta", "beta-8b"), (3, "gamma", "gamma-9b")])
        db.commit()
        db.close()

    def sql(self, query, params=()):
        db = sqlite3.connect(self.path, detect_types=DETECT_TYPES)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            rows = [dict(r) for r in db.execute(query, params).fetchall()]
            db.commit()
            return rows
        finally:
            db.close()

    def add_run(self, model_id, quality=None, *, options=None, perf=None, status="completed", days_ago=1,
                group=None, canary=None, scored=None):
        when = (NOW - timedelta(days=days_ago)).isoformat()
        db = sqlite3.connect(self.path)
        run_id = db.execute(
            """INSERT INTO test_runs (model_id, status, created_at, started_at, completed_at, quality_json, perf_json,
                                      run_options_json, scored_questions, repeat_group_id, canary_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model_id, status, when, when, when, pack_text(json.dumps(quality)) if quality else None,
             pack_text(json.dumps(perf)) if perf else None,
             json.dumps(options or {"mode": "quality", "suite": "standard"}),
             scored if scored is not None else (quality["summary"]["scored"] if quality else 0),
             group, canary)).lastrowid
        if group == 0:
            db.execute("UPDATE test_runs SET repeat_group_id = id WHERE id = ?", (run_id,))
        db.commit()
        db.close()
        return run_id

    def check(self, model_id, fingerprint, run_id=None, changed=None, days_ago=0, max_len=None):
        snapshot = {"hard": {"served": {"max_model_len": max_len}} if max_len else {}}
        self.sql("""INSERT INTO deployment_checks (model_id, run_id, revision, created_at, ok, fingerprint,
                                                   snapshot_json, changed)
                    VALUES (?, ?, 'deployment-probe-v1', ?, 1, ?, ?, ?)""",
                 (model_id, run_id, (NOW - timedelta(days=days_ago)).isoformat(), fingerprint,
                  json.dumps(snapshot), changed))

    def collect(self):
        return scorecard.collect(now=NOW)

    def evidence(self, model_id):
        return scorecard.model_evidence(self.collect(), model_id)


def half(c, f):
    return f < 5


class EvidenceTests(DatabaseCase):
    def test_latest_not_best_and_exclusions(self):
        self.add_run(1, report(True), days_ago=5)
        latest = self.add_run(1, report(half), days_ago=3)
        self.add_run(1, report(True), status="failed", days_ago=2)
        self.add_run(1, report(True), scored=0, days_ago=2)
        # Runs of retired built-in suites, including historical runs without a suite, are not columns.
        self.add_run(1, report(True), options={"mode": "quality", "suite": "assistant-open"})
        self.add_run(1, report(True), options={"mode": "quality", "suite": "tool-conformance"})
        self.add_run(1, report(True), options={"mode": "quality"})
        e = self.evidence(1)
        self.assertEqual(set(e["quality"]), {"standard"})
        q = e["quality"]["standard"]
        self.assertEqual((q["run_id"], q["score"], q["runs"]), (latest, 0.5, 1))
        self.assertEqual(q["categories"]["Cat0"], {"score": 0.5, "full_pass": 0.5})
        self.assertEqual(q["freshness"], "unverified")
        self.assertEqual(self.collect()["suites"], [{"key": "standard", "label": "Standard quality suite"}])

    def test_repeat_group(self):
        first = self.add_run(2, report(True), group=0, days_ago=3)
        self.add_run(2, report(half), group=first, days_ago=2)
        self.add_run(2, None, group=first, status="failed", days_ago=1)
        q = self.evidence(2)["quality"]["standard"]
        self.assertEqual((q["group_id"], q["runs"], q["score"]), (first, 2, 0.75))
        self.assertIsNotNone(q["score_ci95"])
        self.assertEqual(q["run_id"], first + 1)

    def test_freshness_and_age(self):
        current = self.add_run(1, report(True), days_ago=2)
        self.check(1, "f1", run_id=current, days_ago=2)
        self.assertEqual(self.evidence(1)["quality"]["standard"]["freshness"], "current")
        self.check(1, "f2", changed="deployment", days_ago=1)
        self.assertEqual(self.evidence(1)["quality"]["standard"]["freshness"], "stale")
        self.check(1, "f1", changed="deployment")  # rolled back: the run's deployment is served again
        e = self.evidence(1)
        self.assertEqual((e["quality"]["standard"]["freshness"], e["fingerprint"]), ("current", "f1"))

        old = self.add_run(2, report(True), days_ago=200)
        self.check(2, "g2", changed="deployment", days_ago=100)
        q = self.evidence(2)["quality"]["standard"]
        self.assertEqual((q["run_id"], q["freshness"], q["old"], q["age_days"]), (old, "stale", True, 200))
        self.add_run(3, report(True), days_ago=120)
        q = self.evidence(3)["quality"]["standard"]
        self.assertEqual((q["freshness"], q["old"]), ("unverified", True))

    def test_performance_evidence(self):
        self.add_run(1, perf=closed_loop(ttft_p95=900), options={"mode": "performance"}, days_ago=4)
        latest = self.add_run(1, perf=closed_loop(), options={"mode": "performance"}, days_ago=3)
        self.add_run(1, perf=open_loop(1.0), options={"mode": "load"}, days_ago=3)
        chat = self.add_run(1, perf=open_loop(4.0, "lower_bound"), options={"mode": "load"}, days_ago=2)
        agent = self.add_run(1, perf=open_loop(None, "not_met", preset="custom"), options={"mode": "load"})
        sweep = self.add_run(1, perf={"schema_version": 4, "kind": "context_sweep"}, options={"mode": "sweep"})
        for context, status in ((8192, "measured"), (65536, "measured"), (131072, "failed")):
            self.sql("INSERT INTO performance_cells VALUES (?, 'default', ?, 1, ?)",
                     (sweep, context, json.dumps({"status": status})))
        self.check(1, "f1", max_len=131072)
        data = self.collect()
        e = scorecard.model_evidence(data, 1)
        self.assertEqual(e["latency"]["run_id"], latest)
        self.assertEqual(e["latency"]["levels"][1]["ttft_p95"], 400.0)
        self.assertEqual(e["latency"]["levels"][4]["request_output"], 10.0)
        self.assertEqual((e["capacity"]["chat"]["run_id"], e["capacity"]["chat"]["rate"]), (chat, 4.0))
        custom = e["capacity"]["custom:" + "ab" * 8]
        self.assertEqual((custom["run_id"], custom["status"]), (agent, "not_met"))
        self.assertEqual((e["context"]["tokens"], e["context"]["source"]), (65536, "measured"))
        self.assertEqual(e["declared_context"]["tokens"], 131072)
        self.assertEqual([w["label"] for w in data["workloads"]], ["Chat", "Custom workload abababab"])
        exported = json.loads(json.dumps(scorecard.public(data)))
        self.assertNotIn("reports", json.dumps(exported))

    def test_operations(self):
        self.sql("""INSERT INTO canaries (id, name, model_id, suite, interval_hours, enabled)
                    VALUES (1, 'watch', 1, 'tool-conformance', 6, 1), (2, 'paused', 1, 'rigorous', 6, 0)""")
        self.add_run(1, report(True), canary=1)
        e = self.evidence(1)
        self.assertEqual([(c["name"], c["status"]) for c in e["operations"]["canaries"]], [("watch", None)])
        self.sql("UPDATE test_runs SET canary_status = 'warning'")
        self.sql("""INSERT INTO monitor_events (model_id, kind, severity, title) VALUES
                    (1, 'deployment_changed', 'alert', 't'), (1, 'availability', 'warning', 't'),
                    (2, 'baseline_set', 'info', 't')""")
        ops = self.evidence(1)["operations"]
        self.assertEqual((ops["open_alerts"], ops["open_warnings"], ops["canaries"][0]["status"]), (1, 1, "warning"))
        self.assertEqual(self.evidence(2)["operations"]["open_alerts"], 0)


class TieTests(DatabaseCase):
    def test_leader_ties_and_incompatible(self):
        self.add_run(1, report(True))
        self.add_run(2, report(lambda c, f: not (c == 0 and f == 0)))
        self.add_run(3, report(False))
        marks = scorecard.ties(self.collect(), "standard")
        self.assertEqual(marks[1], {"mark": "leader"})
        self.assertEqual(marks[2]["mark"], "tied")
        self.assertEqual(marks[3]["mark"], "below")
        self.assertEqual(marks[3]["adjustment"], "Holm")
        self.assertLess(marks[3]["difference"], 0)

        self.add_run(3, report(True, temperature=0.7))
        marks = scorecard.ties(self.collect(), "standard")
        self.assertEqual((marks[1]["mark"], marks[3]["mark"]), ("leader", "not_comparable"))
        self.assertIn("decoding", marks[3]["reason"])
        self.check(1, "new", changed="deployment")
        marks = scorecard.ties(self.collect(), "standard")
        self.assertEqual(marks[1], {"mark": "stale"})
        self.assertEqual(marks[3]["mark"], "leader")  # 1.0 achievement, alphabetically after the stale leader
        self.assertEqual(scorecard.ties(self.collect(), "tool-conformance"), {})


class ProfileTests(DatabaseCase):
    GATES = [{"type": "quality", "suite": "standard", "threshold": 0.7}, {"type": "operations"}]

    def test_profiles_and_decisions(self):
        for fields, needle in (({"name": ""}, "name"), ({"gates": "nope"}, "JSON"),
                               ({"gates": [{"type": "quality", "suite": "usecase:hr@1", "threshold": 0.5}]}, "suite"),
                               ({"description": "x" * 3000}, "description")):
            with self.subTest(needle), self.assertRaisesRegex(ScorecardError, needle):
                scorecard.create_profile(1, **{"name": "Chat", "description": "", "gates": self.GATES, **fields})
        profile_id = scorecard.create_profile(1, name=" Chat ", description="HR assistant", gates=json.dumps(self.GATES))
        profile = scorecard.get_profile(profile_id)
        self.assertEqual((profile["name"], profile["gates"][0]["metric"]), ("Chat", "score"))
        run_id = self.add_run(1, report(True))
        self.add_run(2, report(half))
        self.check(1, "f1", run_id=run_id)
        rows = scorecard.evaluate_profile(profile, self.collect())
        self.assertEqual([(r["model"]["name"], r["evaluation"]["verdict"]) for r in rows],
                         [("alpha", "meets"), ("gamma", "incomplete"), ("beta", "fails")])
        self.assertEqual(scorecard.gate_labels(profile), ["Standard quality suite: achievement ≥ 70%",
                                                          "No open monitoring alert and every canary ok"])
        overview = scorecard.profile_overview(self.collect())
        self.assertEqual((overview[0]["counts"], overview[0]["meeting"]),
                         ({"meets": 1, "incomplete": 1, "fails": 1}, ["alpha"]))

        with self.assertRaisesRegex(ScorecardError, "note"):
            scorecard.record_decision(profile_id, 1, decision="approved", note=" ", user_id=1)
        with self.assertRaisesRegex(ScorecardError, "Choose"):
            scorecard.record_decision(profile_id, 1, decision="maybe", note="x", user_id=1)
        scorecard.record_decision(profile_id, 1, decision="conditional", note="Pilot only", user_id=1)
        scorecard.record_decision(profile_id, 1, decision="approved", note="Pilot went well", user_id=1)
        self.add_run(1, report(False))  # later evidence must not rewrite the records
        self.check(1, "f2", changed="deployment")
        records = scorecard.decisions(profile_id=profile_id)
        self.assertEqual([(r["decision"], r["superseded"], r["drifted"]) for r in records],
                         [("approved", False, True), ("conditional", True, True)])
        snapshot = records[0]["evaluation"]
        self.assertEqual((snapshot["verdict"], snapshot["results"][0]["run_id"], snapshot["results"][0]["value"]),
                         ("meets", run_id, 1.0))
        self.assertEqual((records[0]["fingerprint"], records[0]["revision"]), ("f1", decision_gates.REVISION))
        self.assertEqual(snapshot["requirements"][0], "Standard quality suite: achievement ≥ 70%")
        self.assertEqual(len(scorecard.decisions(model_id=2)), 0)

        scorecard.update_profile(profile_id, name="Chat v2", description="", gates=self.GATES[:1])
        self.assertEqual(len(scorecard.get_profile(profile_id)["gates"]), 1)
        with self.assertRaisesRegex(ScorecardError, "not found"):
            scorecard.update_profile(999, name="x", description="", gates=self.GATES)
        self.assertEqual(scorecard.delete_profile(profile_id), 2)
        self.assertEqual(self.sql("SELECT COUNT(*) AS n FROM decision_records")[0]["n"], 0)


class PageTests(DatabaseCase):
    def setUp(self):
        super().setUp()
        from app.routes import dashboard, scorecard as scorecard_routes

        self.role = "admin"

        def current():
            return None if self.role == "anonymous" else {"id": 1 if self.role == "admin" else 2,
                                                          "role": self.role, "username": self.role}

        for module in (scorecard_routes, dashboard):
            self.stack.enter_context(patch.object(module, "get_current_user", AsyncMock(side_effect=lambda _r: current())))
        app = FastAPI()

        @app.middleware("http")
        async def user(request, call_next):
            request.state.user = current()
            return await call_next(request)

        for module in (scorecard_routes, dashboard):
            app.include_router(module.router)
        self.client = self.stack.enter_context(TestClient(app))

    def test_pages_and_access(self):
        self.add_run(1, report(True))
        self.add_run(2, report(half))
        self.add_run(1, perf=closed_loop(), options={"mode": "performance"})
        self.add_run(1, perf=open_loop(), options={"mode": "load"})
        self.add_run(1, perf={"schema_version": 7, "kind": "sessions",
                              "protocol": {"preset": "mixed", "user_model_hash": "cd" * 8, "attainment_target": 0.95},
                              "summary": {"results": [
                                  {"context_cap": 131072, "users": 60, "status": "established", "first_failed": 66,
                                   "limiting": {"label": "KV cache full: session histories no longer fit"}},
                                  {"context_cap": 262144, "users": None, "status": "not_measured"}]}},
                     options={"mode": "performance"})
        page = self.client.get("/scorecard").text
        for text in ("Standard quality suite", "Chat", "100.0%", "50.0%", "400 ms", "Create one", "unverified"):
            self.assertIn(text, page)
        ties = self.client.get("/scorecard/ties", params={"suite": "standard"}).json()
        self.assertEqual((ties["1"]["mark"], ties["2"]["mark"]), ("leader", "below"))

        created = self.client.post("/scorecard/profiles", data={"name": "Coding agents", "description": "",
                                                                "gates": json.dumps(ProfileTests.GATES)},
                                   follow_redirects=False)
        self.assertEqual((created.status_code, created.headers["location"]), (303, "/scorecard/profiles/1"))
        bad = self.client.post("/scorecard/profiles", data={"name": "x", "gates": "[]"}, follow_redirects=False)
        self.assertIn("error=A%20profile%20needs", bad.headers["location"])
        profile_page = self.client.get("/scorecard/profiles/1").text
        for text in ("Coding agents", "Standard quality suite: achievement ≥ 70%", "meets", "does not meet",
                     "Record a decision", "Edit profile"):
            self.assertIn(text, profile_page)
        decided = self.client.post("/scorecard/profiles/1/decisions",
                                   data={"model_id": 1, "decision": "approved", "note": "Best fit"},
                                   follow_redirects=False)
        self.assertEqual(decided.headers["location"], "/scorecard/profiles/1#decisions")
        self.assertIn("Best fit", self.client.get("/scorecard/profiles/1").text)
        model_page = self.client.get("/scorecard/models/1").text
        for text in ("Coding agents", "Best fit", "Latency", "Chat", "No studies involve this model",
                     "Mixed · 40% chat, 60% agents", "131,072 tokens:", "60", "KV cache full"):
            self.assertIn(text, model_page)
        self.assertNotIn("262,144 tokens:", model_page)
        self.assertIn("user_models", self.client.get("/scorecard/profiles").text)  # gate editor options
        self.assertEqual(self.client.get("/scorecard/models/99").status_code, 404)
        exported = self.client.get("/scorecard.json").json()
        self.assertEqual(exported["profiles"][0]["models"][0]["evaluation"]["verdict"], "meets")
        self.assertEqual(exported["ties"]["standard"]["1"]["mark"], "leader")
        profile_json = self.client.get("/scorecard/profiles/1/export.json").json()
        self.assertEqual((profile_json["kind"], profile_json["decisions"][0]["note"]), ("decision_profile", "Best fit"))
        dashboard = self.client.get("/dashboard").text
        self.assertIn("Decision profiles", dashboard)
        self.assertIn("met by alpha", dashboard)
        self.client.post("/scorecard/profiles/1/edit", data={"name": "Agents", "description": "",
                                                             "gates": json.dumps([{"type": "operations"}])})
        self.assertEqual(scorecard.get_profile(1)["name"], "Agents")

        self.role = "viewer"
        self.assertNotIn("Record a decision", self.client.get("/scorecard/profiles/1").text)
        self.assertNotIn("New profile", self.client.get("/scorecard/profiles").text)
        for path, data in (("/scorecard/profiles", {"name": "x", "gates": "[]"}),
                           ("/scorecard/profiles/1/edit", {"name": "x"}), ("/scorecard/profiles/1/delete", {}),
                           ("/scorecard/profiles/1/decisions", {"model_id": 1, "decision": "approved", "note": "x"})):
            self.assertEqual(self.client.post(path, data=data, follow_redirects=False).status_code, 403, path)

        self.role = "admin"
        self.assertIn("1 decision record", self.client.get("/scorecard/profiles/1").text)
        self.client.post("/scorecard/profiles/1/delete")
        self.assertEqual(self.client.get("/scorecard/profiles/1").status_code, 404)
        self.role = "anonymous"
        response = self.client.get("/scorecard", follow_redirects=False)
        self.assertEqual((response.status_code, response.headers["location"]), (303, "/login"))


if __name__ == "__main__":
    unittest.main()
