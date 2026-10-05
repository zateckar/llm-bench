#!/usr/bin/env python3
"""Offline tests for deployment fingerprints, canaries and monitoring events."""

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
from app.benchmarking import deployment_probe
from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.models import Question, RequestMetrics, TokenUsage
from app.benchmarking.quality_execution import score_response
from app.benchmarking.quality_report import make_report
from app.services import benchmark_runner as runner
from app.services import monitoring
from app.services.monitoring import MonitoringError
from app.storage import DETECT_TYPES

ROOT = Path(__file__).parent


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    """A vLLM-like server whose details the tests change."""

    served = 0

    def __init__(self, max_len=32768, reject_tools=True, version="0.11.0", template_extra=0, down=False):
        self.max_len, self.reject_tools, self.version = max_len, reject_tools, version
        self.template_extra, self.down = template_extra, down
        self.calls = []
        self.created = 0

    def request(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append((method, url, json))
        if self.down:
            raise ConnectionError("connection refused")
        self.created += 1
        if url.endswith("/v1/models"):
            return FakeResponse(200, {"data": [{"id": "other"}, {"id": "prod-model", "root": "/weights/prod",
                                                "max_model_len": self.max_len, "owned_by": "vllm",
                                                "created": self.created}]}, {"server": "uvicorn"})
        if url.endswith("/version"):
            return FakeResponse(200, {"version": self.version})
        if json.get("tools") and self.reject_tools:
            return FakeResponse(400, {"error": "tool parser not enabled"})
        tokens = len(str(json["messages"])) // 4 + self.template_extra
        FakeSession.served += 1
        content = f"2, 3, 5, 7, 11, 13, 17, 19 (variant {FakeSession.served})"
        return FakeResponse(200, {"model": "prod-model", "system_fingerprint": None,
                                  "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": tokens}})

    def close(self):
        pass


def snap(**kwargs):
    return deployment_probe.capture("https://llm.example:8443/v1/", "key", "prod-model",
                                    session=FakeSession(**kwargs))


class ProbeTests(unittest.TestCase):
    def test_snapshot_contents_and_stable_fingerprint(self):
        first, second = snap(), snap()
        self.assertTrue(first["ok"], first)
        hard = first["hard"]
        self.assertEqual(hard["endpoint"], "https://llm.example:8443/v1")
        self.assertEqual(hard["served"], {"listed": True, "id": "prod-model", "root": "/weights/prod",
                                          "max_model_len": 32768, "owned_by": "vllm"})
        self.assertEqual((hard["version"], hard["server_header"], hard["response_model"]), ("0.11.0", "uvicorn", ["prod-model"]))
        self.assertEqual(hard["probes"]["tools"], {"status": 400})
        self.assertEqual(set(hard["probes"]), set(deployment_probe.PROBES))
        self.assertIsInstance(hard["probes"]["multilingual"]["prompt_tokens"], int)
        self.assertNotEqual(first["behaviour"]["text"], second["behaviour"]["text"])
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(len(first["fingerprint"]), 16)

    def test_diffs_and_classification(self):
        base = snap()
        changed = snap(max_len=131072, template_extra=3)
        changes = deployment_probe.diff(base, changed)
        fields = [c["field"] for c in changes]
        self.assertIn("served.max_model_len", fields)
        self.assertIn("probes.plain.prompt_tokens", fields)
        self.assertEqual(deployment_probe.classify(changes), "deployment")
        self.assertEqual(deployment_probe.summary(changes),
                         "context limit 32,768 → 131,072; chat template or tokenizer (4 of 5 probes count differently)")
        moved = json.loads(json.dumps(base))
        moved["hard"]["endpoint"] = "https://other.example/v1"
        self.assertEqual(deployment_probe.classify(deployment_probe.diff(base, moved)), "configuration")
        self.assertIsNone(deployment_probe.classify(deployment_probe.diff(base, snap())))

    def test_failures(self):
        session = FakeSession(down=True)
        down = deployment_probe.capture("http://h/v1", "k", "prod-model", session=session)
        self.assertEqual((down["ok"], len(session.calls)), (False, 1))
        self.assertIn("unreachable", down["error"])
        self.assertNotIn("fingerprint", down)

        class Rejecting(FakeSession):
            def request(self, method, url, headers=None, json=None, timeout=None):
                if method == "POST":
                    return FakeResponse(401, {"error": "bad key"})
                return super().request(method, url, headers, json, timeout)

        rejected = deployment_probe.capture("http://h/v1", "k", "prod-model", session=Rejecting())
        self.assertFalse(rejected["ok"])
        self.assertIn("HTTP 401", rejected["error"])
        self.assertEqual(deployment_probe.origin_of("http://h:8000/v1/"), "http://h:8000")
        self.assertEqual(deployment_probe.sanitize_endpoint("https://user:pw@h:8443/v1/?key=x"), "https://h:8443/v1")
        missing = deployment_probe.capture("http://h/v1", "k", "absent", session=FakeSession())
        self.assertEqual(missing["hard"]["served"], {"listed": False})


# --- Database fixtures -------------------------------------------------------------

def report(passed, categories=4, families=10, temperature=0):
    results = []
    for c in range(categories):
        for f in range(families):
            q = Question(f"q{c}-{f}", f"Cat{c}", f"Say YES ({c}/{f})", "exact_match", "YES",
                         metadata={"family": f"fam{c}-{f}", "scope": "capability"})
            ok = passed if isinstance(passed, bool) else passed(c, f)
            results.append(score_response(q, "YES" if ok else "NO", TokenUsage(5, 2),
                                          RequestMetrics(latency_ms=10, finish_reason="stop")))
    return make_report(results, ClientConfig("http://x.invalid", "k", "prod-model", temperature=temperature))


class DatabaseCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        self.path = Path(directory) / "bench.db"
        self.stack.enter_context(patch.object(app_config, "DATABASE_PATH", self.path))
        self.stack.enter_context(patch("app.services.url_guard.validate_endpoint"))
        self.dispatch = self.stack.enter_context(patch("app.services.run_queue.dispatch_next"))
        self.webhooks = []
        self.webhook_error = None

        def post(url, payload):
            self.webhooks.append((url, payload))
            return self.webhook_error

        self.stack.enter_context(patch.object(monitoring, "_post_webhook", side_effect=post))
        self.snapshots = [snap()]
        self.stack.enter_context(patch.object(monitoring, "capture", side_effect=lambda model: self.snapshots[0]))
        db = sqlite3.connect(self.path)
        db.executescript((ROOT / "app/schema.sql").read_text(encoding="utf-8"))
        db.execute("INSERT INTO users (id, username, role) VALUES (1, 'admin', 'admin'), (2, 'viewer', 'user')")
        db.execute("INSERT INTO models (id, name, base_url, api_key, model_id) VALUES (1, 'prod', 'https://llm.example/v1', 'k', 'prod-model')")
        db.commit()
        db.close()
        self.model = {"id": 1, "name": "prod", "base_url": "https://llm.example/v1", "api_key": "k", "model_id": "prod-model"}

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

    def canary(self, **fields):
        values = {"name": "Prod canary", "model_id": 1, "suite": "standard", "interval_hours": 6,
                  "max_concurrency": 4, "webhook_url": "https://hooks.example/T1", **fields}
        return monitoring.create_canary(1, **values)

    def finished_run(self, canary_id, quality, status="completed", ttft=500.0, latency=2000.0, errors=0,
                     message=None, scored=None):
        db = sqlite3.connect(self.path)
        run_id = db.execute(
            """INSERT INTO test_runs (model_id, status, quality_json, ttft_p50_ms, latency_p50_ms, error_count,
                                      error_message, scored_questions, avg_score, canary_id, completed_at)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (status, json.dumps(quality) if quality else None, ttft, latency, errors, message,
             scored if scored is not None else (quality["summary"]["scored"] if quality else 0),
             quality["summary"]["category_balanced"] if quality else 0, canary_id,
             datetime.now(timezone.utc).isoformat())).lastrowid
        db.commit()
        db.close()
        return run_id

    def events(self):
        return self.sql("SELECT kind, severity, run_id, canary_id, notified FROM monitor_events ORDER BY id")


class SnapshotTests(DatabaseCase):
    def test_change_events(self):
        first = monitoring.record_check(self.model, snap())
        second = monitoring.record_check(self.model, snap())
        self.assertEqual(self.events(), [])
        checks = self.sql("SELECT id, changed, previous_id FROM deployment_checks ORDER BY id")
        self.assertEqual([(c["changed"], c["previous_id"]) for c in checks], [(None, None), (None, first)])
        monitoring.record_check(self.model, {"revision": "x", "ok": False, "error": "down", "hard": {}})
        monitoring.record_check(self.model, snap(version="0.12.0"))
        events = self.events()
        self.assertEqual([(e["kind"], e["severity"]) for e in events], [("deployment_changed", "alert")])
        self.assertEqual(self.webhooks, [])  # no canary for this model yet
        latest = self.sql("SELECT previous_id, changed FROM deployment_checks ORDER BY id DESC LIMIT 1")[0]
        self.assertEqual((latest["previous_id"], latest["changed"]), (second, "deployment"))
        detail = monitoring.events()[0]["detail"]
        self.assertEqual(detail["changes"], [{"field": "version", "before": "0.11.0", "after": "0.12.0"}])
        self.canary()
        moved = json.loads(json.dumps(snap(version="0.12.0")))
        moved["hard"]["endpoint"] = "https://new.example/v1"
        monitoring.record_check(self.model, moved)
        self.assertEqual(self.events()[-1]["kind"], "configuration_changed")
        self.assertEqual(self.webhooks, [])  # info events are not posted
        timeline = monitoring.model_checks(1)
        self.assertEqual(timeline[0]["changes"][0]["field"], "endpoint")
        self.assertEqual(timeline[-1]["changes"], [])

    def test_runner_records_snapshot_and_survives_failures(self):
        from selftest_benchmark_runner import FakeClient

        db = sqlite3.connect(self.path)
        db.execute("INSERT INTO test_runs (id, model_id, status) VALUES (1, 1, 'pending'), (2, 1, 'pending')")
        db.commit()
        db.close()
        questions = [Question(str(i), "Reasoning", "good", "json_match", {"value": {"x": 1}}) for i in range(2)]
        with patch.object(runner, "load_questions", return_value=questions), \
                patch("app.benchmarking.llm_client.ChatClient", FakeClient):
            runner._run_benchmark(1, self.model, "quality", 1, None)
            with patch.object(monitoring, "capture", side_effect=RuntimeError("probe bug")):
                runner._run_benchmark(2, self.model, "quality", 1, None)
        runs = self.sql("SELECT id, status FROM test_runs ORDER BY id")
        self.assertEqual([r["status"] for r in runs], ["completed", "completed"])
        checks = self.sql("SELECT run_id, ok, fingerprint FROM deployment_checks")
        self.assertEqual([(c["run_id"], c["ok"]) for c in checks], [(1, 1)])
        self.assertEqual(monitoring.run_check(1)["fingerprint"], checks[0]["fingerprint"])
        self.assertIsNone(monitoring.run_check(2))


class CanaryTests(DatabaseCase):
    def test_validation(self):
        for fields, needle in (({"interval_hours": 0}, "between"), ({"suite": "nope"}, "suite"),
                               ({"webhook_url": "http://hooks.example"}, "https"), ({"model_id": 9}, "Unknown model"),
                               ({"name": " "}, "name"), ({"max_concurrency": 99}, "Concurrency"),
                               ({"interval_hours": "x"}, "whole")):
            with self.subTest(needle), self.assertRaisesRegex(MonitoringError, needle):
                self.canary(**fields)

    def test_scheduling_without_backlog(self):
        canary_id = self.canary()
        now = datetime.now(timezone.utc) + timedelta(seconds=1)
        created = monitoring.schedule_due(now)
        self.assertEqual(len(created), 1)
        run = self.sql("SELECT * FROM test_runs WHERE id = ?", (created[0],))[0]
        self.assertEqual((run["status"], run["canary_id"], run["workers"]), ("pending", canary_id, 4))
        self.assertEqual(json.loads(run["run_options_json"]), {"mode": "quality", "max_concurrency": 4,
                                                               "suite": "standard"})
        self.assertEqual(json.loads(run["quality_config_json"])["name"], "standard")
        self.assertEqual(json.loads(run["decoding_config_json"]), {"temperature": 0.0, "reasoning_effort": None})
        next_at = self.sql("SELECT next_run_at FROM canaries")[0]["next_run_at"]
        self.assertEqual(next_at, (now + timedelta(hours=6)).isoformat())
        self.assertEqual(monitoring.schedule_due(now + timedelta(hours=1)), [])
        later = now + timedelta(hours=20)  # due, but the previous run is still pending
        self.assertEqual(monitoring.schedule_due(later), [])
        self.assertEqual(self.sql("SELECT next_run_at FROM canaries")[0]["next_run_at"],
                         (later + timedelta(hours=6)).isoformat())
        with self.assertRaisesRegex(MonitoringError, "already has"):
            monitoring.run_now(canary_id)
        self.sql("UPDATE test_runs SET status = 'completed'")
        monitoring.set_enabled(canary_id, False)
        self.assertEqual(monitoring.schedule_due(later + timedelta(days=30)), [])
        self.assertEqual(len(self.sql("SELECT id FROM test_runs WHERE canary_id = ?", (canary_id,))), 1)
        run_id = monitoring.run_now(canary_id)
        self.dispatch.assert_called()
        self.assertEqual(self.sql("SELECT canary_id FROM test_runs WHERE id = ?", (run_id,))[0]["canary_id"], canary_id)

    def test_evaluation(self):
        canary_id = self.canary()
        good, bad = report(True), report(False)
        baseline = self.finished_run(canary_id, good)
        self.assertEqual(monitoring.evaluate_run(baseline), "ok")
        self.assertEqual(self.sql("SELECT baseline_run_id FROM canaries")[0]["baseline_run_id"], baseline)
        self.assertIsNone(monitoring.evaluate_run(baseline))  # evaluated once

        same = self.finished_run(canary_id, good)
        self.assertEqual(monitoring.evaluate_run(same), "ok")
        detail = json.loads(self.sql("SELECT canary_json FROM test_runs WHERE id = ?", (same,))[0]["canary_json"])
        self.assertEqual((detail["comparison"]["verdict"], detail["events"]), ("no_detectable_difference", []))

        regressed = self.finished_run(canary_id, bad)
        self.assertEqual(monitoring.evaluate_run(regressed), "alert")
        self.assertEqual(self.events()[-1]["kind"], "quality_regression")
        self.assertEqual(self.events()[-1]["notified"], "sent")
        self.assertIn("ALERT · prod: Quality regressed by 100.0 points", self.webhooks[-1][1]["text"])
        self.assertEqual(self.webhooks[-1][0], "https://hooks.example/T1")

        improved = self.finished_run(canary_id, good)
        self.sql("UPDATE canaries SET baseline_run_id = ?", (regressed,))
        self.assertEqual(monitoring.evaluate_run(improved), "ok")
        self.assertEqual(self.events()[-1]["kind"], "quality_improvement")
        self.sql("UPDATE canaries SET baseline_run_id = ?", (baseline,))

        slow = self.finished_run(canary_id, good, ttft=2000, errors=2)
        self.webhook_error = "HTTP 500"
        self.assertEqual(monitoring.evaluate_run(slow), "warning")
        self.assertEqual([e["kind"] for e in self.events()[-2:]], ["availability", "latency_regression"])
        self.assertEqual(self.events()[-1]["notified"], "HTTP 500")
        self.webhook_error = None
        small = self.finished_run(canary_id, good, ttft=700)  # 1.4x and +200 ms: within tolerance
        self.assertEqual(monitoring.evaluate_run(small), "ok")

        other = json.loads(json.dumps(good))
        other["suite"]["name"] = "other"
        incompatible = self.finished_run(canary_id, other)
        self.assertEqual(monitoring.evaluate_run(incompatible), "warning")
        self.assertEqual(self.events()[-1]["kind"], "baseline_incompatible")

        failed = self.finished_run(canary_id, None, status="failed", message="HTTP 503 everywhere")
        self.assertEqual(monitoring.evaluate_run(failed), "alert")
        self.assertEqual(self.events()[-1]["kind"], "run_failed")
        stopped = self.finished_run(canary_id, None, status="failed", message=monitoring.STOPPED)
        self.assertEqual(monitoring.evaluate_run(stopped), "ok")
        empty = self.finished_run(canary_id, good, scored=0)
        self.assertEqual(monitoring.evaluate_run(empty), "alert")

        changed = self.finished_run(canary_id, good)
        monitoring.record_check(self.model, snap())
        monitoring.record_check(self.model, snap(max_len=8192), run_id=changed)
        self.assertEqual(monitoring.evaluate_run(changed), "alert")
        change_event = self.sql("SELECT canary_id FROM monitor_events WHERE kind = 'deployment_changed'")[0]
        self.assertEqual(change_event["canary_id"], canary_id)

    def test_baselines_pending_evaluation_and_retention(self):
        canary_id = self.canary()
        with self.assertRaisesRegex(MonitoringError, "completed run of this canary"):
            monitoring.set_baseline(canary_id, 999)
        good = report(True)
        alert = self.finished_run(canary_id, report(False))
        runs = [self.finished_run(canary_id, good) for _ in range(6)]
        monitoring.set_baseline(canary_id, runs[0])
        with patch.object(monitoring, "KEEP_RUNS", 3):
            monitoring.evaluate_pending()
        remaining = [r["id"] for r in self.sql("SELECT id FROM test_runs ORDER BY id")]
        # The first run became an alert against the baseline; the baseline and newest three stay.
        self.assertEqual(remaining, [alert, runs[0], *runs[3:]])
        self.assertTrue(all(r["canary_status"] for r in self.sql("SELECT canary_status FROM test_runs")))
        self.assertEqual(self.sql("SELECT baseline_run_id FROM canaries")[0]["baseline_run_id"], runs[0])

        update = {"name": "Renamed", "model_id": 1, "suite": "standard", "interval_hours": 12,
                  "max_concurrency": 4, "webhook_url": ""}
        monitoring.update_canary(canary_id, **update)
        canary = self.sql("SELECT * FROM canaries")[0]
        self.assertEqual((canary["name"], canary["baseline_run_id"], canary["webhook_url"]), ("Renamed", runs[0], None))
        monitoring.update_canary(canary_id, **{**update, "max_concurrency": 2})
        self.assertIsNone(self.sql("SELECT baseline_run_id FROM canaries")[0]["baseline_run_id"])
        monitoring.delete_canary(canary_id)
        self.assertEqual(self.sql("SELECT COUNT(*) AS n FROM test_runs WHERE canary_id IS NOT NULL")[0]["n"], 0)
        self.assertEqual(len(self.sql("SELECT id FROM test_runs")), 5)

    def test_queue_hooks(self):
        from app.services import run_queue

        canary_id = self.canary()
        run_id = self.finished_run(canary_id, report(True))
        with patch.object(run_queue, "dispatch_next"):
            run_queue.on_run_finished(run_id)
        self.assertEqual(self.sql("SELECT canary_status FROM test_runs")[0]["canary_status"], "ok")
        with patch.object(monitoring, "evaluate_run", side_effect=RuntimeError("boom")), \
                patch.object(run_queue, "dispatch_next") as dispatch:
            run_queue.on_run_finished(run_id)  # never raises
        dispatch.assert_called_once()


class PageTests(DatabaseCase):
    def setUp(self):
        super().setUp()
        from app.routes import dashboard, monitoring as monitoring_routes, runs

        self.role = "admin"

        def current():
            return None if self.role == "anonymous" else {"id": 1 if self.role == "admin" else 2,
                                                          "role": self.role, "username": self.role}

        for module in (monitoring_routes, runs, dashboard):
            self.stack.enter_context(patch.object(module, "get_current_user", AsyncMock(side_effect=lambda _r: current())))
        app = FastAPI()

        @app.middleware("http")
        async def user(request, call_next):
            request.state.user = current()
            return await call_next(request)

        for module in (monitoring_routes, runs, dashboard):
            app.include_router(module.router)
        self.client = self.stack.enter_context(TestClient(app))

    def test_monitoring_pages(self):
        created = self.client.post("/monitoring/canaries", data={"name": "Prod canary", "model_id": 1,
                                                                 "suite": "standard", "interval_hours": 6,
                                                                 "max_concurrency": 4, "webhook_url": ""},
                                   follow_redirects=False)
        self.assertEqual((created.status_code, created.headers["location"]), (303, "/monitoring/canaries/1"))
        bad = self.client.post("/monitoring/canaries", data={"name": "x", "model_id": 1, "suite": "nope"},
                               follow_redirects=False)
        self.assertIn("error=Choose%20a%20known", bad.headers["location"])
        baseline = self.finished_run(1, report(True))
        monitoring.record_check(self.model, snap(), run_id=baseline)
        monitoring.evaluate_run(baseline)
        regressed = self.finished_run(1, report(False))
        monitoring.record_check(self.model, snap(max_len=8192), run_id=regressed)
        monitoring.evaluate_run(regressed)

        page = self.client.get("/monitoring").text
        for text in ("Needs attention", "Quality regressed", "Deployment changed: context limit 32,768 → 8,192",
                     "Prod canary", "New canary", "Check now"):
            self.assertIn(text, page)
        self.assertIn("2 unacknowledged alerts", self.client.get("/dashboard").text)
        canary_page = self.client.get("/monitoring/canaries/1").text
        self.assertIn("Set as baseline", canary_page)
        self.assertIn("-100.0", canary_page)
        detail = self.client.get(f"/runs/{regressed}").text
        self.assertIn("changed since the previous snapshot", detail)
        self.assertIn("served.max_model_len", detail)
        self.assertIn("Canary run of", detail)
        self.assertIn("canary", self.client.get("/runs").text)
        model_page = self.client.get("/monitoring/models/1").text
        self.assertIn("Timeline", model_page)
        self.assertIn("deployment changed", model_page)

        self.client.post("/monitoring/canaries/1/baseline", data={"run_id": regressed})
        self.assertEqual(self.sql("SELECT baseline_run_id FROM canaries")[0]["baseline_run_id"], regressed)
        self.assertEqual(self.client.post("/monitoring/canaries/1/run", follow_redirects=False).status_code, 303)
        self.assertEqual(self.sql("SELECT COUNT(*) AS n FROM test_runs WHERE status = 'pending'")[0]["n"], 1)
        busy = self.client.post("/monitoring/canaries/1/run", follow_redirects=False)
        self.assertIn("error=", busy.headers["location"])
        self.client.post("/monitoring/canaries/1/enabled", data={"enabled": 0})
        self.assertEqual(self.sql("SELECT enabled FROM canaries")[0]["enabled"], 0)
        self.client.post("/monitoring/models/1/check")
        self.assertEqual(self.sql("SELECT COUNT(*) AS n FROM deployment_checks WHERE run_id IS NULL")[0]["n"], 1)

        self.role = "viewer"
        page = self.client.get("/monitoring").text
        self.assertNotIn("New canary", page)
        self.assertNotIn("Acknowledge", page)
        for path, data in (("/monitoring/canaries", {"name": "x", "model_id": 1, "suite": "rigorous"}),
                           ("/monitoring/canaries/1/run", {}), ("/monitoring/canaries/1/delete", {}),
                           ("/monitoring/events/ack", {"event_id": 1}), ("/monitoring/models/1/check", {})):
            self.assertEqual(self.client.post(path, data=data, follow_redirects=False).status_code, 403, path)

        self.role = "admin"
        open_ids = [e["id"] for e in monitoring.events(open_only=True)]
        self.client.post("/monitoring/events/ack", data={"event_id": open_ids})
        self.assertEqual(monitoring.open_counts(), {"alert": 0, "warning": 0})
        self.assertNotIn("unacknowledged", self.client.get("/dashboard").text)
        self.assertIn("acknowledged by admin", self.client.get("/monitoring").text)
        self.sql("UPDATE test_runs SET status = 'completed' WHERE status = 'pending'")
        self.assertEqual(self.client.post("/monitoring/canaries/1/delete", follow_redirects=False).headers["location"],
                         "/monitoring")
        self.assertEqual(self.client.get("/monitoring/canaries/1").status_code, 404)

        self.role = "anonymous"
        response = self.client.get("/monitoring", follow_redirects=False)
        self.assertEqual((response.status_code, response.headers["location"]), (303, "/login"))


if __name__ == "__main__":
    unittest.main()
