"""Portable report and comparison regressions; fake data, no model/server calls."""

import asyncio
from contextlib import closing
from html.parser import HTMLParser
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config as app_config
from app.routes import compare, dashboard, runs
from app.services import html_reports as reports
from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.models import Question, RequestMetrics, Result, TokenUsage
from app.benchmarking.quality_report import make_report

ROOT = Path(__file__).parent


def fixture_database(path):
    db = sqlite3.connect(path)
    db.executescript((ROOT / "app/schema.sql").read_text(encoding="utf-8"))
    db.execute(
        "INSERT INTO models(id,name,base_url,api_key,model_id) VALUES(1,?,?,?,?)",
        ("Orion 32B", "https://private.invalid", "DO_NOT_EXPORT_API_KEY", "orion-32b"),
    )
    db.execute(
        "INSERT INTO models(id,name,base_url,api_key,model_id) VALUES(2,?,?,?,?)",
        ("Atlas 70B", "https://secret.invalid", "DO_NOT_EXPORT_API_KEY", "atlas-70b"),
    )
    categories = [
        "Advanced Coding",
        "Logical Reasoning",
        "Mathematical Reasoning",
        "Reading Comprehension",
        "Tool Using",
        "Instruction Following",
    ]
    for rid in (1, 2, 3):
        factor = 1 if rid == 1 else 1.65
        load = []
        for level, tps, ttft in [(1, 35, 240), (2, 64, 310), (4, 112, 640), (8, 130, 2300)]:
            load.append(
                {
                    "concurrency": level,
                    "requests": 32,
                    "errors": 0,
                    "error_rate": 0,
                    "output_tokens_per_sec": tps * factor,
                    "requests_per_sec": tps * factor / 300,
                    "latency": {"p95_ms": 4200 * level / factor},
                    "ttft": {"p95_ms": ttft / factor},
                    "wall_ms": 10000,
                    "estimated_token_requests": 0,
                    "burst_delivery_requests": 0,
                }
            )
        perf = {
            "schema_version": 3,
            "protocol": {
                "revision": "performance-v3",
                "input_reference_tokens": 1024,
                "max_output_tokens": 256,
            },
            "peak_output_tokens_per_sec": 130 * factor,
            "peak_requests_per_sec": 0.43 * factor,
            "concurrency": load,
            "notes": [],
        }
        results = []
        for i, category in enumerate(categories):
            q = Question(
                id=f"Q{i}",
                category=category,
                prompt="Return the answer as JSON.",
                evaluator="json_match",
                expected={"value": {"answer": 42}},
            )
            score = int(bool((i + rid) % 4) and not (rid == 1 and i == 5))
            results.append(
                Result(
                    q,
                    '{"answer":42}',
                    score,
                    "Correct" if score else "Incorrect answer",
                    tokens=TokenUsage(prompt_tokens=200, completion_tokens=50),
                    metrics=RequestMetrics(ok=True, latency_ms=1450 / factor, ttft_ms=150 / factor),
                    outcome="pass" if score else "task_failure",
                )
            )
        quality = make_report(
            results,
            ClientConfig("http://fake.invalid", "unused", "fake"),
        )
        if rid == 3:
            quality = None
            perf = None
        db.execute(
            """INSERT INTO test_runs(id,model_id,status,total_questions,scored_questions,passed_questions,
                      avg_score,weighted_score,workers,duration_ms,latency_p50_ms,latency_p95_ms,latency_p99_ms,
                      ttft_p50_ms,ttft_p95_ms,output_tokens_per_sec,total_prompt_tokens,total_completion_tokens,
                      test_suite_hash,quality_json,perf_json,started_at,completed_at)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rid,
                1 if rid != 2 else 2,
                "completed",
                6,
                6,
                sum(r.passed for r in results),
                sum(r.score for r in results) / len(results),
                sum(r.score for r in results) / len(results),
                1,
                55000,
                1450 / factor,
                3300 / factor,
                4100 / factor,
                150 / factor,
                360 / factor,
                0 if rid == 3 else 34 * factor,
                24000,
                9000,
                "fixture-suite",
                json.dumps(quality),
                json.dumps(perf),
                "2026-09-19 12:00:00",
                "2026-09-19 12:04:00",
            ),
        )
        for i, result in enumerate(results):
            db.execute(
                """INSERT INTO test_results(run_id,test_id,category,question_index,score,passed,quality_scored,
                          request_ok,detail,prompt,response,evaluator,latency_ms,ttft_ms,prompt_tokens,completion_tokens)
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rid,
                    result.question.id,
                    result.question.category,
                    i,
                    result.score,
                    int(result.passed),
                    None if rid == 3 else 1,
                    1,
                    result.detail,
                    result.question.prompt,
                    result.response,
                    result.question.evaluator,
                    1450 / factor,
                    150 / factor,
                    200,
                    50,
                ),
            )
    db.commit()
    db.close()


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external = []
        self.scripts = 0
        self.svg = 0

    def handle_starttag(self, tag, attrs):
        self.scripts += tag == "script"
        self.svg += tag == "svg"
        for key, value in attrs:
            if key in {"src", "href", "srcset"} and value and not value.startswith(("#", "data:")):
                self.external.append(value)


class ResultSectionParser(HTMLParser):
    """Read visible text within the result sections, including nested diagnostics."""

    def __init__(self):
        super().__init__()
        self.stack = []
        self.text = {"Quality results": [], "Performance results": []}
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"style", "script"}:
            self.ignored += 1
        if tag == "section":
            self.stack.append(dict(attrs).get("aria-label"))

    def handle_endtag(self, tag):
        if tag in {"style", "script"}:
            self.ignored -= 1
        if tag == "section":
            self.stack.pop()

    def handle_data(self, data):
        if not self.ignored:
            for label in self.text:
                if label in self.stack:
                    self.text[label].append(data)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "reports.db"
        fixture_database(self.path)
        self.db_patch = patch.object(app_config, "DATABASE_PATH", self.path)
        self.db_patch.start()
        app = FastAPI()
        app.include_router(runs.router)
        app.include_router(compare.router)
        app.include_router(dashboard.router)
        self.client = TestClient(app)
        self.auth = [
            patch.object(module, "get_current_user", new=AsyncMock(return_value={"id": 1}))
            for module in (runs, compare, dashboard)
        ]
        for p in self.auth:
            p.start()

    def tearDown(self):
        self.client.close()
        for p in self.auth:
            p.stop()
        self.db_patch.stop()
        self.directory.cleanup()

    def test_downloads_are_offline_and_include_data(self):
        for url, filename in [
            ("/runs/1/report.html", "run-1.html"),
            ("/compare/report.html?runs=1&runs=2", "comparison-1-2.html"),
        ]:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.headers["content-disposition"], f'attachment; filename="{filename}"'
            )
            self.assertIn("text/html", response.headers["content-type"])
            self.assertEqual(response.headers["cache-control"], "no-store")
            parser = AssetParser()
            parser.feed(response.text)
            self.assertEqual(parser.external, [])
            self.assertEqual(parser.scripts, 0)
            self.assertGreaterEqual(parser.svg, 5)
            self.assertIn("Estimated tokens (requests)", response.text)
            self.assertIn("Return the answer as JSON.", response.text)
            self.assertNotIn("DO_NOT_EXPORT_API_KEY", response.text)
            self.assertNotIn("private.invalid", response.text)
        self.assertIn("identical tasks", response.text)
        self.assertRegex(response.text, r"p = (&lt; 0\.001|[01]\.\d{3})")

    def test_performance_json_auth_missing_and_secret_exclusion(self):
        response = self.client.get("/runs/1/performance.json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["schema_version"], 3)
        self.assertNotIn("DO_NOT_EXPORT_API_KEY", response.text)
        self.assertIn("run-1.performance.json", response.headers["content-disposition"])
        self.assertEqual(self.client.get("/runs/3/performance.json").status_code, 404)
        with patch.object(runs, "get_current_user", AsyncMock(return_value=None)):
            self.assertEqual(self.client.get("/runs/1/performance.json", follow_redirects=False).status_code, 302)

    def test_challenge_cohort_and_interactive_turns_render_safely(self):
        from app.benchmarking.quality_suite import load_questions

        questions = [q for q in load_questions() if q.id in {"H5-CL-worlds-01", "S6-policy-01"}]
        quality = make_report(
            [Result(q, "{}", 0, outcome="task_failure") for q in questions],
            ClientConfig("https://fake.invalid", "unused", "fake"),
        )
        meta = {
            "metadata": {"protocol": "json-actions-v1"},
            "diagnostics": {
                "transcript": [
                    {"role": "assistant", "content": '{"tool":"read","args":{}}'},
                    {"role": "tool", "content": {"value": 7}},
                    {"role": "assistant", "content": "<script>bad()</script>"},
                ]
            },
        }
        with closing(sqlite3.connect(self.path)) as db:
            db.execute(
                "UPDATE test_runs SET quality_json=?, perf_json=NULL WHERE id=1",
                (json.dumps(quality),),
            )
            db.execute(
                "UPDATE test_results SET quality_metadata_json=? WHERE run_id=1 AND test_id='Q0'",
                (json.dumps(meta),),
            )
            db.commit()
        for url in ("/runs/1", "/runs/1/report.html"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertIn("achievement", response.text)
            rendered = response.text
            if url == "/runs/1":
                with closing(sqlite3.connect(self.path)) as db:
                    ids = [r[0] for r in db.execute("SELECT id FROM test_results WHERE run_id=1")]
                rendered += "".join(self.client.get(f"/runs/1/results/{rid}").text for rid in ids)
            self.assertIn("Interactive transcript", rendered)
            self.assertIn("Turn 2", rendered)
            self.assertIn("&lt;script&gt;bad()&lt;/script&gt;", rendered)
            self.assertNotIn("<script>bad()</script>", rendered)

        for url in ("/compare?runs=1&runs=2", "/compare/report.html?runs=1&runs=2"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Criterion achievement", response.text)

    def test_native_tool_results_render_wire_diagnostics(self):
        from app.benchmarking.quality_execution import execute_question
        from app.benchmarking.quality_report import result_record
        from app.benchmarking.suites import get_suite
        from selftest_tool_conformance import BY_ID, PlanClient, oracle_plan, reply

        stream = BY_ID["T1-typed-arguments-s19-v01-stream"]
        blocking = BY_ID["T1-typed-arguments-s19-v01-blocking"]
        named = BY_ID["T2-tool-choice-named-s19-v01"]
        none = BY_ID["T2-tool-choice-none-s19-v01"]
        results = [
            execute_question(stream, PlanClient(oracle_plan(stream))),
            execute_question(blocking, PlanClient(oracle_plan(blocking))),
            execute_question(named, PlanClient([], rejected="HTTP 400: tool_choice requires --tool-call-parser")),
            execute_question(none, PlanClient([reply('[TOOL_CALLS] <script>bad()</script>')])),
        ]
        quality = make_report(results, ClientConfig("https://fake.invalid", "unused", "fake"),
                              suite=get_suite("tool-conformance"))
        native = quality["summary"]["native"]
        self.assertEqual(native["feature_rejected"], 1)
        self.assertEqual(native["rejection_status"], {"400": 1})
        self.assertEqual(native["markup_leaks"], {"mistral": 1})
        self.assertEqual(native["transports"]["blocking"], {"tasks": 1, "passes": 1, "contract_failures": 0})
        ordinary = Result(Question("q", "Code", "p", "exact_match", "x"), "x", 1, outcome="pass")
        rigorous = make_report([ordinary], ClientConfig("https://fake.invalid", "unused", "fake"))
        self.assertNotIn("native", rigorous["summary"])
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE test_runs SET quality_json=? WHERE id=1", (json.dumps(quality),))
            db.execute("DELETE FROM test_results WHERE run_id=1")
            for index, result in enumerate(results):
                record = result_record(result)
                db.execute(
                    """INSERT INTO test_results(run_id,test_id,category,question_index,score,passed,quality_scored,
                              request_ok,detail,prompt,response,evaluator,quality_metadata_json,quality_outcome)
                              VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (result.question.id, result.question.category, index, result.achievement_score,
                     int(result.passed), int(result.is_scored), int(result.metrics.ok), result.detail,
                     result.question.prompt, result.response, result.question.evaluator,
                     json.dumps(record), record["outcome"]),
                )
            db.commit()
        for url in ("/runs/1", "/runs/1/report.html"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Tool-calling wire diagnostics", response.text)
            rendered = response.text
            if url == "/runs/1":
                with closing(sqlite3.connect(self.path)) as db:
                    ids = [r[0] for r in db.execute("SELECT id FROM test_results WHERE run_id=1")]
                rendered += "".join(self.client.get(f"/runs/1/results/{rid}").text for rid in ids)
            self.assertIn("Native tool-call transcript", rendered)
            self.assertIn("tool call create_calendar_event", rendered)
            self.assertIn("request rejected (HTTP 400)", rendered)
            self.assertIn("&lt;script&gt;bad()&lt;/script&gt;", rendered)
            self.assertNotIn("<script>bad()</script>", rendered)
            self.assertNotIn("Infrastructure Errors", response.text)

    def test_repeat_group_page_badges_and_group_comparison(self):
        def report(seed, strength):
            results = []
            for c in range(4):
                for f in range(3):
                    q = Question(f"G{c}-{f}", f"Category {c}", "Return JSON.", "json_match",
                                 {"value": {"answer": 42}}, metadata={"family": f"fam-{c}-{f}"})
                    passed = (c * 3 + f + seed) % 10 < strength
                    results.append(Result(q, "{}", int(passed), outcome="pass" if passed else "task_failure"))
            return make_report(results, ClientConfig("https://fake.invalid", "unused", "fake", seed=seed))

        with closing(sqlite3.connect(self.path)) as db:
            for group, model, strength in ((10, 1, 4), (20, 2, 8)):
                for index in range(3):
                    db.execute(
                        """INSERT INTO test_runs(id,model_id,status,quality_json,repeat_group_id,repeat_index,
                                  repeat_count,run_options_json) VALUES(?,?,?,?,?,?,3,?)""",
                        (group + index, model, "completed" if index < 2 or group == 20 else "failed",
                         json.dumps(report(index, strength)), group, index,
                         json.dumps({"mode": "quality", "max_concurrency": 4})),
                    )
            db.commit()
        page = self.client.get("/runs/groups/10?vs=20&vs=10&vs=999")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Run-to-run variability", page.text)
        self.assertIn("2/3 runs usable", page.text)
        self.assertIn("Run #12 excluded: not completed", page.text)
        self.assertIn("pass^k", page.text)
        self.assertIn("Unstable tasks", page.text)
        self.assertIn("Group #20 (Atlas 70B)", page.text)
        self.assertIn("across 2 vs 3 runs", page.text)
        self.assertIn("right is higher", page.text)
        self.assertEqual(page.context["selected_ids"], [20])
        self.assertEqual(self.client.get("/runs/groups/999").status_code, 404)
        listing = self.client.get("/runs")
        self.assertIn('href="/runs/groups/10"', listing.text)
        self.assertIn("R2/3", listing.text)
        self.assertIn("group summary", self.client.get("/runs/11").text)

    def test_stale_binary_summary_is_recomputed_from_saved_criteria(self):
        with closing(sqlite3.connect(self.path)) as db:
            raw = db.execute("SELECT quality_json FROM test_runs WHERE id=1").fetchone()[0]
            report = json.loads(raw)
            report["summary"]["category_balanced"] = 0.123
            db.execute(
                "UPDATE test_runs SET quality_json=?, avg_score=.123 WHERE id=1",
                (json.dumps(report),),
            )
            db.commit()
        run = asyncio.run(reports.load_run(1))
        self.assertAlmostEqual(run["avg_score"], 2 / 3)
        self.assertIn("66.7%", self.client.get("/compare?runs=1&runs=2").text)

    def test_performance_values_and_online_parity(self):
        selected = [asyncio.run(reports.load_run(i)) for i in (1, 2)]
        view = reports.performance_view(selected)
        row = next(r for r in view["suite_rows"] if r["label"] == "Peak aggregate output")
        self.assertEqual(row["values"], ["130.0 tok/s", "214.5 tok/s"])
        online = self.client.get("/compare?runs=1&runs=2")
        self.assertEqual(online.status_code, 200)
        for row in view["suite_rows"]:
            for value in row["values"]:
                self.assertIn(value, online.text)
        self.assertIn("Delivered output under load", online.text)
        self.assertIn('aria-label="Delivered output under load"', reports.render_report(selected))
        self.assertIn("/compare/report.html?runs=1&amp;runs=2", online.text)

    def test_performance_comparison_aligns_runs_and_missing_loads(self):
        selected = [asyncio.run(reports.load_run(i)) for i in (2, 1)]
        selected[0]["perf"]["concurrency"] = selected[0]["perf"]["concurrency"][1:]
        comparison = reports.performance_view(selected)["comparison"]
        self.assertEqual([r["label"] for r in comparison["runs"]], [r["label"] for r in selected])
        workload = comparison["workloads"][0]
        self.assertEqual(workload["concurrencies"], [1, 2, 4, 8])
        first = workload["rows"][0]
        self.assertEqual((first["context"], first["concurrency"]), (1024, 1))
        self.assertIsNone(first["points"][0])
        self.assertEqual(first["points"][1]["formatted"]["aggregate_output"], "35.0 tok/s")
        shared = workload["rows"][1]["points"]
        self.assertEqual([p["values"]["aggregate_output"] for p in shared], [105.6, 64])
        for url in ("/compare?runs=2&runs=1", "/compare/report.html?runs=2&runs=1"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertIn('aria-label="Side-by-side performance comparison"', response.text)
            if "report.html" in url:
                self.assertIn("35.0 tok/s", response.text)
            else:
                self.assertIn("performanceComparison", response.text)
        self.assertNotIn('aria-label="Side-by-side performance comparison"', self.client.get("/runs/1").text)

    def test_performance_comparison_keeps_budgets_and_sampling_settings_distinct(self):
        selected = [asyncio.run(reports.load_run(i)) for i in (1, 2)]
        for field, value in (("max_output_tokens", 8192), ("temperature", 1),
                             ("reasoning_effort", "high"), ("reference_tokenizer", "other")):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(selected))
                changed[1]["perf"]["protocol"][field] = value
                workloads = reports.performance_view(changed)["comparison"]["workloads"]
                self.assertEqual(len(workloads), 2)
                self.assertTrue(all(row["points"][1] is None for row in workloads[0]["rows"]))
                self.assertTrue(all(row["points"][0] is None for row in workloads[1]["rows"]))

    def test_sweep_comparison_matches_contexts_prefixes_and_revisions(self):
        selected = [asyncio.run(reports.load_run(i)) for i in (1, 2)]
        cold = {"effort": "default", "context_tokens": 32768, "concurrency": 1,
                "status": "measured", "requests": 1, "completed": 1, "errors": 0, "incomplete": 0,
                "latency": {"p50_ms": 2000}, "ttft": {"p50_ms": 600},
                "aggregate_tokens_per_sec": 30, "request_tokens_per_sec": {"p50": 20}}
        for index, run in enumerate(selected):
            run["perf"] = {"schema_version": 4, "kind": "context_sweep", "finished": index == 0,
                           "protocol": {"revision": f"context-sweep-v{index + 2}",
                                        "max_output_tokens": 8192, "temperature": 0,
                                        "reference_tokenizer": "cl100k_base", "rounds_per_cell": index + 1},
                           "cells": [json.loads(json.dumps(cold))]}
        selected[0]["perf"]["cells"][0]["cache_reuse"] = {**cold, "ttft": {"p50_ms": 80}}
        selected[1]["perf"]["cells"].append({**cold, "context_tokens": 65536, "status": "failed",
                                             "requests": 1, "completed": 0, "errors": 1,
                                             "ttft": {}, "latency": {}, "aggregate_tokens_per_sec": 0,
                                             "error": "EngineCore <script>alert(1)</script>"})
        comparison = reports.performance_view(selected)["comparison"]
        self.assertEqual(len(comparison["workloads"]), 2)
        shared = comparison["workloads"][0]
        self.assertEqual(shared["contexts"], [32768, 65536])
        self.assertEqual(shared["rounds"], [1, 2])
        self.assertEqual([p["formatted"]["ttft_p50"] for p in shared["rows"][0]["points"]], ["600 ms", "600 ms"])
        self.assertIsNone(shared["rows"][1]["points"][0])
        failed = shared["rows"][1]["points"][1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["formatted"]["aggregate_output"], "0.0 tok/s")
        self.assertEqual(failed["formatted"]["ttft_p50"], "n/a")
        warm = comparison["workloads"][1]["rows"][0]["points"]
        self.assertEqual(warm[0]["formatted"]["ttft_p50"], "80 ms")
        self.assertIsNone(warm[1])
        self.assertTrue(comparison["runs"][1]["partial"])
        html = reports.render_report(selected)
        self.assertIn("These runs use different performance revisions", html)
        self.assertIn("Not measured for this workload", html)
        self.assertIn("EngineCore &lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<script>alert(1)</script>", html)
        parser = AssetParser()
        parser.feed(html)
        self.assertEqual(parser.scripts, 0)
        self.assertEqual(parser.external, [])

    def test_prefix_cache_comparison_retains_cold_warm_rates_and_missing_run(self):
        selected = [asyncio.run(reports.load_run(i)) for i in (1, 2, 3)]
        point = {"requests": 4, "errors": 0, "output_tokens_per_sec": 42,
                 "ttft": {"p50_ms": 10}, "cache_metrics": {"cache_hit_fraction": 0,
                 "uncached_prefill_tokens_per_sec": {"p50": 123}}}
        selected[0]["perf"]["cache_reuse"] = [{"context_tokens": 8192, "concurrency": 1,
                                               "cold": point, "warm": {**point, "ttft": {"p50_ms": 5}}}]
        workloads = reports.performance_view(selected)["comparison"]["workloads"]
        cache = [w for w in workloads if w["label"].startswith("Prefix cache")]
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache[0]["rows"][0]["points"][0]["formatted"]["cache_hit"], "0.0%")
        self.assertEqual(cache[1]["rows"][0]["points"][0]["formatted"]["ttft_p50"], "5 ms")
        self.assertTrue(all(p is None for w in cache for p in w["rows"][0]["points"][1:]))
        self.assertIsNone(reports.performance_view([selected[0]])["comparison"])

    def assert_results_separated(self, html, *, quality=True, performance=True):
        parser = ResultSectionParser()
        parser.feed(html)
        quality_text = " ".join(parser.text["Quality results"])
        performance_text = " ".join(parser.text["Performance results"])
        if quality:
            self.assertIn("Criterion achievement", quality_text)
            self.assertIn("Quality-task timing diagnostics", quality_text)
        if performance:
            self.assertIn("Latency by concurrency", performance_text)
            self.assertIn("130.0 tok/s", performance_text)
        else:
            self.assertIn("No performance measurements", performance_text)
        self.assertNotIn("tok/s", quality_text)
        self.assertNotIn("Peak aggregate output", quality_text)
        self.assertNotIn("Latency by concurrency", quality_text)
        self.assertNotIn("Quality-task", performance_text)
        self.assertNotIn("Slowest categories", performance_text)
        self.assertNotIn("Advanced Coding", performance_text)
        self.assertNotIn("Criterion achievement", performance_text)
        self.assertNotIn("Successful completion", html)

    def test_quality_and_performance_sections_are_separate_on_all_reports(self):
        for url in ("/runs/1", "/compare?runs=1&runs=2", "/runs/1/report.html",
                    "/compare/report.html?runs=1&runs=2"):
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assert_results_separated(response.text)
        html = self.client.get("/runs/1/report.html").text
        quality_cards = html.split('aria-label="Quality results"')[0]
        self.assertNotIn("Peak output under load", quality_cards)
        self.assertNotIn("Question latency p50", quality_cards)
        self.assertIn("Full task success", quality_cards)

    def test_quality_only_run_keeps_timings_out_of_empty_performance_section(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE test_runs SET perf_json=NULL WHERE id=1")
            db.commit()
        for url in ("/runs/1", "/runs/1/report.html"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assert_results_separated(response.text, performance=False)

    def test_performance_only_run_and_comparison_have_no_quality_charts(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("DELETE FROM test_results WHERE run_id IN (1,2)")
            db.execute("UPDATE test_runs SET quality_json=NULL,total_questions=0 WHERE id IN (1,2)")
            db.commit()
        for url in ("/runs/1", "/compare?runs=1&runs=2", "/runs/1/report.html",
                    "/compare/report.html?runs=1&runs=2"):
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assert_results_separated(response.text, quality=False)
                self.assertNotIn("Quality-task timing diagnostics", response.text)
                self.assertNotIn("compareRadar", response.text)
                self.assertNotIn("Return the answer as JSON.", response.text)
                if "report.html" not in url:
                    self.assertIn("tab: 'performance'", response.text)

    def test_performance_view_is_independent_of_quality_results(self):
        run = asyncio.run(reports.load_run(1))
        baseline = reports.performance_view([run])
        self.assertNotIn("request_views", baseline)
        run["results"] = [{"category": "POISON", "request_ok": 1, "latency_ms": 999999999}]
        run["quality"] = {"summary": {"category_balanced": 0}}
        self.assertEqual(reports.performance_view([run]), baseline)
        run["perf"] = {}
        self.assertEqual(reports.performance_view([run])["load_views"], [])

    def test_mixed_comparison_keeps_quality_only_and_performance_only_runs_separate(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE test_runs SET perf_json=NULL WHERE id=1")
            db.execute("DELETE FROM test_results WHERE run_id=2")
            db.execute("UPDATE test_runs SET quality_json=NULL,total_questions=0 WHERE id=2")
            db.commit()
        for url in ("/compare?runs=1&runs=2", "/compare/report.html?runs=1&runs=2"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            parser = ResultSectionParser()
            parser.feed(response.text)
            quality_text = " ".join(parser.text["Quality results"])
            performance_text = " ".join(parser.text["Performance results"])
            self.assertIn("Quality-task timing diagnostics", quality_text)
            self.assertNotIn("tok/s", quality_text)
            self.assertIn("214.5 tok/s", performance_text)
            self.assertNotIn("Quality-task", performance_text)
            self.assertNotIn("Advanced Coding", performance_text)

    def test_run_lists_do_not_present_quality_timings_as_performance(self):
        for url in ("/runs", "/dashboard"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("tok/s", response.text)
            self.assertNotIn("Quality output rate", response.text)
            self.assertNotIn("p50 / p95", response.text)

    def test_dashboard_keeps_scores_when_newest_run_is_not_completed(self):
        for status in ("running", "pending", "failed"):
            with self.subTest(status=status):
                with closing(sqlite3.connect(self.path)) as db:
                    db.execute("UPDATE test_runs SET status=? WHERE id=3", (status,))
                    db.commit()
                response = self.client.get("/dashboard")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.context["last_run"]["id"], 2)
                self.assertEqual(response.context["recent_runs"][0]["status"], status)
                self.assertIn('id="radarChart"', response.text)

    def test_dashboard_finds_quality_run_beyond_recent_runs(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE test_results SET quality_scored=0 WHERE run_id=3")
            # More than a page of newer active, failed, and performance-only runs.
            db.executemany(
                "INSERT INTO test_runs(id,model_id,status,total_questions) VALUES(?,1,?,0)",
                [(rid, ("running", "pending", "failed", "completed")[rid % 4])
                 for rid in range(4, 16)],
            )
            quality = json.loads(db.execute(
                "SELECT quality_json FROM test_runs WHERE id=2"
            ).fetchone()[0])
            db.commit()
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="radarChart"', response.text)
        self.assertEqual(response.context["last_run"]["id"], 2)
        self.assertEqual(len(response.context["recent_runs"]), 10)
        self.assertTrue(all(run["id"] > 2 for run in response.context["recent_runs"]))
        for category in response.context["last_run_categories"]:
            self.assertEqual(category["avg_score"],
                             quality["summary"]["categories"][category["category"]]["score"])
        self.assertIn('href="/runs/2"', response.text)
        self.assertIn("Run #2", response.text)
        self.assertNotIn("No completed runs", response.text)

    def test_dashboard_last_run_uses_the_rigorous_suite_only(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE test_runs SET run_options_json=? WHERE id=3",
                       (json.dumps({"mode": "quality", "max_concurrency": 4, "suite": "tool-conformance"}),))
            db.execute("UPDATE test_runs SET run_options_json='not json' WHERE id=2")
            db.commit()
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["last_run"]["id"], 2)

    def test_dashboard_legacy_scores_include_zero_and_exclude_unscored_answers(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE test_results SET score=0,quality_scored=NULL WHERE run_id=3")
            db.execute("UPDATE test_results SET category='Legacy' WHERE run_id=3")
            db.execute("UPDATE test_results SET score=1,quality_scored=0 WHERE run_id=3 AND test_id='Q0'")
            db.commit()
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["last_run"]["id"], 3)
        category, = response.context["last_run_categories"]
        self.assertEqual(category["avg_score"], 0)
        self.assertEqual(category["scored"], 5)
        self.assertIn('id="radarChart"', response.text)

    def test_dashboard_empty_state_requires_completed_quality_scores(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE test_runs SET status='failed' WHERE id=1")
            db.execute("DELETE FROM test_results WHERE run_id=2")
            db.execute("UPDATE test_runs SET total_questions=0,quality_json=NULL WHERE id=2")
            db.execute("UPDATE test_results SET quality_scored=0 WHERE run_id=3")
            db.commit()
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context["last_run"])
        self.assertEqual(response.context["last_run_categories"], [])
        self.assertNotIn('id="radarChart"', response.text)
        self.assertIn("No completed runs with quality scores yet", response.text)

    def test_legacy_zero_missing_partial_and_invalid_json(self):
        run = asyncio.run(reports.load_run(3))
        text = reports.render_report([run])
        self.assertIn("No performance measurements", text)
        self.assertIn("n/a", text)
        run_legacy = {**run, "perf": {"schema_version": 2, "concurrency": []}}
        self.assertIn("no performance report for the current protocol", reports.render_report([run_legacy]))
        run.update(status="running", quality={}, perf=reports.object_json("{broken"))
        self.assertIn("partial snapshot", reports.render_report([run]))
        self.assertEqual(reports.object_json("[1,2]"), {})
        self.assertEqual(reports.fmt(float("nan")), "n/a")
        self.assertEqual(reports.fmt(None), "n/a")
        self.assertEqual(reports.fmt(0, "%"), "0.0%")
        self.assertEqual(reports.fmt(0.02, "req/s"), "0.020 req/s")

    def test_excluded_answers_and_safe_untrusted_text(self):
        payload = '</pre><script>alert("XSS")</script><img src="https://bad.invalid">'
        with closing(sqlite3.connect(self.path)) as db:
            db.execute(
                "UPDATE test_results SET quality_scored=0,request_ok=0,response=?,detail=? WHERE run_id=1",
                (payload, payload),
            )
            db.execute("UPDATE models SET name=? WHERE id=1", (payload,))
            db.commit()
        run = asyncio.run(reports.load_run(1))
        self.assertEqual(run["scored_questions"], 0)
        self.assertTrue(all(c["avg_score"] is None for c in run["categories"].values()))
        text = reports.render_report([run])
        parser = AssetParser()
        parser.feed(text)
        self.assertEqual(parser.scripts, 0)
        self.assertEqual(parser.external, [])
        self.assertNotIn(payload, text)
        self.assertIn("&lt;script&gt;", text)
        context = reports.comparison_context([run, asyncio.run(reports.load_run(2))])
        self.assertIsNone(context["comparison_data"][0]["results"][0]["run_1_score"])
        online = self.client.get("/compare?runs=1&runs=2")
        self.assertNotIn(payload, online.text)

    def test_auth_not_found_and_selection(self):
        self.assertEqual(self.client.get("/runs/999/report.html").status_code, 404)
        self.assertEqual(
            self.client.get("/runs/99999999999999999999999/report.html").status_code, 404
        )
        self.assertEqual(self.client.get("/compare/report.html?runs=1&runs=999").status_code, 404)
        self.assertEqual(self.client.get("/compare/report.html?runs=1&runs=1").status_code, 422)
        self.assertEqual(self.client.get("/compare/report.html?runs=garbage").status_code, 422)
        response = self.client.get("/compare/report.html?runs=2,1&runs=2")
        self.assertEqual(response.status_code, 200)
        self.assertIn("comparison-2-1.html", response.headers["content-disposition"])
        for module in (runs, compare):
            module.get_current_user.return_value = None
        for url in ("/runs/1/report.html", "/compare/report.html?runs=1&runs=2", "/compare"):
            self.assertEqual(
                self.client.get(url, follow_redirects=False).headers["location"], "/login"
            )

    def test_svg_uses_numeric_axes_and_breaks_missing_segments(self):
        chart = reports.line_chart(
            [{"label": "x", "color": reports.COLORS[0], "points": [(1, 0), (2, None), (8, 10)]}],
            "Example",
            "Concurrency",
            "ms",
        )
        self.assertIn("Example", chart)
        self.assertEqual(chart.count("<circle"), 2)
        self.assertNotIn('stroke-width="2.5"', chart)
        self.assertIsNone(reports.line_chart([], "Empty", "X", "ms"))

    def test_peak_cards_use_one_observed_load_and_include_failed_requests(self):
        run = {"label": "fixture", "perf": {"schema_version": 3, "protocol": {
            "input_reference_tokens": 4096, "max_output_tokens": 512,
            "attempts_per_request": 1, "revision": "fixture-v1"}, "concurrency": [
                {"concurrency": 1, "requests": 10, "errors": 0,
                 "output_tokens_per_sec": 30, "latency": {"p95_ms": 1200},
                 "ttft": {"p95_ms": 200}},
                {"concurrency": 4, "requests": 20, "errors": 2,
                 "output_tokens_per_sec": 100, "latency": {"p95_ms": 8000},
                 "ttft": {"p95_ms": 2200}},
                {"concurrency": 8, "requests": 20, "errors": 10,
                 "output_tokens_per_sec": 80, "latency": {"p95_ms": 20000},
                 "ttft": {"p95_ms": 7000}},
            ]}}
        view = reports.performance_view([run])
        summary = view["load_views"][0]
        self.assertEqual(summary["peak_concurrency"], 4)
        self.assertEqual([m["value"] for m in summary["metrics"]],
                         ["100.0 tok/s", "8.00 s", "2.20 s", "76.0%"])
        self.assertEqual(summary["metrics"][3]["detail"], "38 / 50 timed requests · 12 errors")
        self.assertEqual(summary["protocol"]["input"], "4,096")
        self.assertEqual(summary["protocol"]["output"], "512")
        self.assertEqual([r["concurrency"] for r in view["load_rows"] if r["peak"]], ["4"])

    def test_failed_and_missing_measurements_do_not_invent_a_peak(self):
        failed = {"label": "failed", "perf": {"schema_version": 3, "cancelled": True,
            "concurrency": [{"concurrency": 1, "requests": 24, "errors": 24,
                             "output_tokens_per_sec": 0}]}}
        view = reports.performance_view([failed])
        summary = view["load_views"][0]
        self.assertIsNone(summary["peak_concurrency"])
        self.assertEqual([m["value"] for m in summary["metrics"]],
                         ["n/a", "n/a", "n/a", "0.0%"])
        self.assertFalse(view["load_rows"][0]["peak"])
        self.assertIn("stopped early", " ".join(view["notes"]))
        failed["perf"]["concurrency"][0].pop("errors")
        self.assertEqual(reports.performance_view([failed])["load_views"][0]["metrics"][3]["value"], "n/a")
        self.assertEqual(self.client.get("/runs/3").status_code, 200)
        self.assertIn("No performance measurements", self.client.get("/runs/3").text)

    def test_quality_tail_and_error_categories_remain_visible(self):
        rows = [
            {"category": "Fast", "request_ok": 1, "latency_ms": 900, "ttft_ms": 90},
            {"category": "Slow", "request_ok": 1, "latency_ms": 120000, "ttft_ms": 100},
            {"category": "Slow", "request_ok": 0, "latency_ms": 900000,
             "detail": "stream deadline exceeded"},
            {"category": "Failed only", "request_ok": 0, "latency_ms": 950000},
        ]
        view = reports.quality_timing_view({"label": "fixture", "results": rows})
        self.assertEqual(view["metrics"][0]["value"], "50.0%")
        self.assertEqual(view["error_count"], 2)
        self.assertEqual(view["deadline_count"], 1)
        self.assertEqual([r["category"] for r in view["slowest"]], ["Slow", "Fast"])
        self.assertEqual(view["slowest"][0]["time_p95"], "2.00 min")
        self.assertEqual(len(view["rows"]), 3)
        self.assertIn("Request error", view["chart"])
        self.assertNotIn("First delivery:", view["chart"])
        self.assertIn("First delivery:", view["delivery_chart"])

    def test_timing_units_and_percentile_axes(self):
        self.assertEqual([reports.duration(v) for v in (None, 0, 125, 1760, 152900)],
                         ["n/a", "0 ms", "125 ms", "1.76 s", "2.55 min"])
        series = [{"label": "fixture", "color": reports.COLORS[0], "points": [(50, 1000), (100, 800000)]}]
        chart = reports.line_chart(series, "Tail", "Percentile", "ms", percentile=True)
        self.assertIn(">min</text>", chart)
        self.assertIn(">0</text>", chart)
        self.assertIn(">25</text>", chart)
        self.assertIn(">100</text>", chart)
        self.assertNotIn(">800k</text>", chart)
        single = reports.line_chart([{**series[0], "points": [(100, 1000)]}],
                                    "Single", "Percentile", "ms", percentile=True)
        self.assertIn('cx="538"', single)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--preview":
        directory = Path(sys.argv[2])
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.db"
            fixture_database(path)
            with patch.object(app_config, "DATABASE_PATH", path):
                selected = [asyncio.run(reports.load_run(i)) for i in (1, 2)]
                (directory / "comparison.html").write_text(
                    reports.render_report(selected), encoding="utf-8"
                )
                (directory / "run.html").write_text(
                    reports.render_report(selected[:1]), encoding="utf-8"
                )
        print(directory)
    else:
        unittest.main(verbosity=2)
