#!/usr/bin/env python3
"""Offline tests for uploaded use-case suites: parsing, versions, runs, reports."""

from contextlib import ExitStack
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.benchmarking import usecase_suites
from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.models import RequestMetrics, TokenUsage
from app.benchmarking.quality_execution import execute_question, score_response
from app.benchmarking.quality_report import compare_groups, make_report
from app.benchmarking.suites import get_suite
from app.benchmarking.usecase_suites import UsecaseSuiteError, parse_suite
from app.storage import DETECT_TYPES
from selftest_tool_conformance import PlanClient, reply

EXAMPLE = Path("app/benchmarking/usecase_example.yaml").read_text(encoding="utf-8")
KEY = "usecase:example-service-desk@1"

ANSWERS = {
    "triage-printer-cs": "PRINTING",
    "triage-vpn-de": "NETWORK",
    "invoice-total-cs": "12100",
    "policy-laptop-replacement": "Yes, it is due: developer laptops are replaced after 36 months.",
}


def minimal(questions, slug="team-suite"):
    return json.dumps({"suite": {"slug": slug, "name": "Team"}, "questions": questions})


def question(**fields):
    return {"id": "q1", "category": "C", "prompt": "Say the queue name.", "evaluator": "exact_match",
            "expected": "PRINTING", **fields}


def answer(q):
    if q.request:
        return execute_question(q, PlanClient([reply(json.dumps(q.expected["json"]["value"]))]))
    return score_response(q, ANSWERS[q.id], TokenUsage(10, 5), RequestMetrics(latency_ms=5))


class ParserTests(unittest.TestCase):
    def test_example_document(self):
        parsed = parse_suite(EXAMPLE)
        self.assertEqual(parsed.slug, "example-service-desk")
        self.assertEqual(parsed.warnings, [])
        self.assertEqual(parsed.suite_hash, parse_suite(EXAMPLE).suite_hash)
        by_id = {q.id: q for q in parsed.questions}
        self.assertEqual(len(by_id), 5)
        for q in parsed.questions:
            self.assertEqual(q.max_tokens, 65536)
            self.assertEqual(q.metadata["scope"], "capability")
            self.assertEqual(q.metadata["cohort"], "usecase:example-service-desk")
            self.assertTrue(q.system_prompt.startswith("You are the internal service desk"))
        native = by_id["invoice-header-en"]
        self.assertEqual(native.evaluator, "native_structured_output")
        self.assertEqual(native.request["response_format"]["type"], "json_schema")
        self.assertEqual(native.expected["json"]["allow_fence"], False)
        self.assertEqual(native.metadata["protocol"], "native-tools-v1")
        self.assertEqual(by_id["triage-vpn-de"].metadata["family"], "triage-vpn")

    def test_question_level_settings_override_suite_defaults(self):
        doc = json.loads(minimal([question(system_prompt="Own prompt", max_tokens=512)]))
        doc["suite"]["system_prompt"] = "Default"
        q = parse_suite(json.dumps(doc)).questions[0]
        self.assertEqual((q.system_prompt, q.max_tokens, q.metadata["family"]), ("Own prompt", 512, "q1"))

    def test_rejections_are_specific(self):
        schema = {"type": "object", "additionalProperties": False, "required": ["total"],
                  "properties": {"total": {"type": "number"}}}
        cases = {
            "not allowed in use-case suites": minimal([question(evaluator="code_exec", expected=[
                {"function": "f", "args": [], "expected": 1}])]),
            "suite.slug": minimal([question()], slug="Bad_Slug"),
            "duplicate question id": minimal([question(), question()]),
            "unknown field": minimal([question(colour="red")]),
            "matches the empty string": minimal([question(evaluator="regex_all", expected=["a*"])]),
            "coin flip": minimal([question(expected="yes")]),
            "violates the schema": minimal([question(evaluator="json_match", expected={"value": {"total": "x"}},
                                                     response_format={"type": "json_schema", "json_schema": {
                                                         "name": "t", "schema": schema}})]),
            "requires the json_match": minimal([question(response_format={"type": "json_object"})]),
            "is not supported": minimal([question(evaluator="json_match", expected={"value": {"total": 1}},
                                                  response_format={"type": "json_schema", "json_schema": {
                                                      "name": "t", "schema": {**schema, "properties": {
                                                          "total": {"type": "number", "multipleOf": 1}}}}})]),
            "exactly the keys": json.dumps({"suite": {"slug": "ab", "name": "x"}, "questions": [], "x": 1}),
            "YAML error": "suite: {slug: ab}\nsuite: {slug: cd}\nquestions: []\n",
            "larger than": "#" * (usecase_suites.MAX_BYTES + 1),
            "non-empty list": minimal([]),
        }
        for needle, text in cases.items():
            with self.subTest(needle):
                with self.assertRaises(UsecaseSuiteError) as caught:
                    parse_suite(text)
                self.assertTrue(any(needle in p for p in caught.exception.problems), caught.exception.problems)

    def test_warnings_do_not_block(self):
        parsed = parse_suite(minimal([question(evaluator="mcq", expected="B")]))
        self.assertTrue(any("options" in w for w in parsed.warnings))


class SuiteRouteTests(unittest.TestCase):
    def setUp(self):
        from app import config
        from app.routes import admin, plans, runs, suites
        from app.services import run_queue

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        self.path = Path(directory) / "bench.db"
        self.stack.enter_context(patch.object(config, "DATABASE_PATH", self.path))
        self.stack.enter_context(patch.object(run_queue, "dispatch_next"))
        db = sqlite3.connect(self.path)
        db.executescript(Path("app/schema.sql").read_text(encoding="utf-8"))
        db.execute("INSERT INTO users (id, username, role) VALUES (1, 'admin', 'admin'), (2, 'viewer', 'user')")
        db.execute("INSERT INTO models (id, name, base_url, api_key, model_id) VALUES (1, 'm', 'http://x.invalid', 'k', 'm')")
        db.commit()
        db.close()
        self.role = "admin"
        app = FastAPI()

        @app.middleware("http")
        async def user(request, call_next):
            request.state.user = {"id": 1 if self.role == "admin" else 2, "role": self.role, "username": self.role}
            return await call_next(request)

        for router in (admin.router, plans.router, runs.router, suites.router):
            app.include_router(router)
        self.client = self.stack.enter_context(TestClient(app))

    def sql(self, query, params=()):
        db = sqlite3.connect(self.path, detect_types=DETECT_TYPES)
        db.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in db.execute(query, params).fetchall()]
            db.commit()
            return rows
        finally:
            db.close()

    def upload(self, text, action="save", file=False):
        if file:
            return self.client.post("/admin/suites", data={"action": action},
                                    files={"file": ("suite.yaml", text.encode("utf-8"), "text/yaml")},
                                    follow_redirects=False)
        return self.client.post("/admin/suites", data={"action": action, "yaml_text": text}, follow_redirects=False)

    def test_versions_check_archive_and_downloads(self):
        checked = self.upload(EXAMPLE, "check")
        self.assertEqual(checked.status_code, 200)
        self.assertIn("would be saved as version 1", checked.text)
        self.assertEqual(self.sql("SELECT * FROM usecase_suites"), [])
        saved = self.upload(EXAMPLE)
        self.assertEqual((saved.status_code, saved.headers["location"]), (302, "/admin/suites/example-service-desk?saved=1"))
        row = self.sql("SELECT * FROM usecase_suites")[0]
        self.assertEqual((row["version"], row["question_count"], row["source_yaml"]), (1, 5, EXAMPLE))
        same = self.upload(EXAMPLE)
        self.assertEqual(same.status_code, 422)
        self.assertIn("already has exactly these questions", same.text)
        changed = EXAMPLE.replace("Reply with the queue name only.", "Reply with the queue name only, in capitals.", 1)
        self.assertEqual(self.upload(changed, file=True).status_code, 302)
        self.assertEqual([r["version"] for r in self.sql("SELECT version FROM usecase_suites ORDER BY version")], [1, 2])

        listing = self.client.get("/admin/suites")
        self.assertIn("Example - service desk assistant", listing.text)
        self.assertIn("2 versions", listing.text)
        detail = self.client.get("/admin/suites/example-service-desk?version=1")
        self.assertIn("triage-printer-cs", detail.text)
        self.assertIn("response_format json_schema", detail.text)
        self.assertIn(KEY, detail.text)
        self.assertEqual(self.client.get("/admin/suites/example-service-desk/1.yaml").text, EXAMPLE)
        self.assertEqual(self.client.get("/admin/suites/example-service-desk/9.yaml").status_code, 404)
        self.assertIn("slug: example-service-desk", self.client.get("/admin/suites/example.yaml").text)

        form = self.client.get("/admin/run").text
        self.assertIn("usecase:example-service-desk@2", form)
        self.assertNotIn(KEY, form)
        self.client.post("/admin/suites/example-service-desk/archive", data={"archived": 1})
        self.assertNotIn("usecase:example-service-desk@2", self.client.get("/admin/run").text)
        self.client.post("/admin/suites/example-service-desk/archive", data={"archived": 0})
        self.assertIn("usecase:example-service-desk@2", self.client.get("/admin/run").text)

    def test_invalid_upload_and_access(self):
        bad = self.upload(minimal([question(evaluator="code_exec", expected=[])]))
        self.assertEqual(bad.status_code, 422)
        self.assertIn("not accepted", bad.text)
        self.assertEqual(self.sql("SELECT * FROM usecase_suites"), [])
        self.role = "user"
        for response in (self.client.get("/admin/suites", follow_redirects=False), self.upload(EXAMPLE)):
            self.assertEqual(response.status_code, 302)
            self.assertEqual(response.headers["location"], "/login")
        self.assertEqual(self.sql("SELECT * FROM usecase_suites"), [])

    def test_runs_pin_versions_and_the_runner_loads_them(self):
        self.upload(EXAMPLE)
        spec = {"model_id": 1, "mode": "quality", "max_concurrency": 2, "suite": KEY}
        posted = self.client.post("/admin/run", data={"runs_json": json.dumps([spec]),
                                                      "scheduled_at": "2099-01-01T00:00", "tz_offset": "0"},
                                  follow_redirects=False)
        self.assertEqual(posted.status_code, 302)
        run = self.sql("SELECT * FROM test_runs")[0]
        self.assertEqual(json.loads(run["run_options_json"])["suite"], KEY)
        config = json.loads(run["quality_config_json"])
        self.assertEqual((config["name"], config["version"], config["revision"]),
                         ("usecase:example-service-desk", 1, "usecase-v1"))
        for bad in ("usecase:example-service-desk@9", "usecase:BAD@1", "usecase:example-service-desk"):
            response = self.client.post("/admin/run", data={"runs_json": json.dumps([{**spec, "suite": bad}])},
                                        follow_redirects=False)
            self.assertEqual(response.status_code, 422, bad)
        # A newer upload never changes the pinned run; editing keeps v1 selectable.
        self.upload(EXAMPLE.replace("Total due", "Amount due"))
        self.assertIn(KEY, self.client.get("/admin/plans/1/edit").text)
        self.client.post(f"/runs/{run['id']}/rerun")  # pending runs cannot be rerun
        self.sql("UPDATE test_runs SET status='completed' WHERE id=?", (run["id"],))
        self.client.post(f"/runs/{run['id']}/rerun", follow_redirects=False)
        rerun = self.sql("SELECT run_options_json FROM test_runs ORDER BY id DESC LIMIT 1")[0]
        self.assertEqual(json.loads(rerun["run_options_json"])["suite"], KEY)

        suite = get_suite(KEY)
        self.assertEqual(suite.name, "usecase:example-service-desk")
        self.assertEqual(len(suite.load()), 5)
        self.assertEqual(suite.provenance()["label"], "Example - service desk assistant v1")
        self.sql("UPDATE usecase_suites SET suite_hash='tampered' WHERE version=1")
        with self.assertRaisesRegex(ValueError, "stored fingerprint"):
            get_suite(KEY)

    def test_reports_and_cross_version_comparison(self):
        self.upload(EXAMPLE)
        self.upload(EXAMPLE.replace("Toner jsme už vyměnili", "Toner byl vyměněn").replace(
            "toner jsme už vyměnili", "toner byl vyměněn"))
        config = ClientConfig("http://x.invalid", "k", "m")
        reports = []
        for version in (1, 2):
            suite = get_suite(f"usecase:example-service-desk@{version}")
            results = [answer(q) for q in suite.load()]
            self.assertTrue(all(r.passed for r in results), [(r.question.id, r.detail) for r in results])
            reports.append(make_report(results, config, suite=suite))
        first = reports[0]
        self.assertEqual(first["suite"]["name"], "usecase:example-service-desk")
        self.assertEqual(first["protocol"]["revision"], "usecase-v1")
        self.assertEqual(first["protocol"]["execution"]["revision"], "bounded-quality-v1")
        self.assertEqual(first["summary"]["passes"], 5)
        self.assertIn("native", first["summary"])
        comparison = compare_groups([reports[0]], [reports[1]])
        self.assertTrue(comparison["compatible"], comparison)
        self.assertEqual((comparison["paired"], comparison["left_unmatched"], comparison["right_unmatched"]), (4, 1, 1))
        rigorous_like = {**first, "suite": {"revision": "rigorous-v15"}}
        self.assertFalse(compare_groups([first], [rigorous_like])["compatible"])


if __name__ == "__main__":
    unittest.main()
