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
        self.assertIn("identical capability questions scored by both", response.text)

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
            self.assertIn("Interactive transcript", response.text)
            self.assertIn("Turn 2", response.text)
            self.assertIn("&lt;script&gt;bad()&lt;/script&gt;", response.text)
            self.assertNotIn("<script>bad()</script>", response.text)

        for url in ("/compare?runs=1&runs=2", "/compare/report.html?runs=1&runs=2"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Criterion achievement", response.text)

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

    def assert_results_separated(self, html, *, quality=True, performance=True):
        parser = ResultSectionParser()
        parser.feed(html)
        quality_text = " ".join(parser.text["Quality results"])
        performance_text = " ".join(parser.text["Performance results"])
        if quality:
            self.assertIn("Criterion achievement", quality_text)
            self.assertIn("Quality-task timing diagnostics", quality_text)
        if performance:
            self.assertIn("Fixed-workload load test", performance_text)
            self.assertIn("130.0 tok/s", performance_text)
        else:
            self.assertIn("No fixed-workload measurements", performance_text)
        self.assertNotIn("tok/s", quality_text)
        self.assertNotIn("Peak aggregate output", quality_text)
        self.assertNotIn("Fixed-workload load test", quality_text)
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

    def test_legacy_zero_missing_partial_and_invalid_json(self):
        run = asyncio.run(reports.load_run(3))
        text = reports.render_report([run])
        self.assertIn("no performance report for the current protocol", text)
        self.assertIn("n/a", text)
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
        self.assertIn("No fixed-workload measurements", self.client.get("/runs/3").text)

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
