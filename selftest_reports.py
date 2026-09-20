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
from app.routes import compare, runs
from app.services import html_reports as reports
from llm_client import ClientConfig
from models import Question, RequestMetrics, Result, TokenUsage
from quality_report import make_report
from quality_suite import QualityConfig

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
                    "stream_tps": {"p50_ms": tps * factor / level},
                }
            )
        perf = {
            "endpoint": "https://private.invalid",
            "streaming": True,
            "decode_tokens_per_sec": 35 * factor,
            "prefill_tokens_per_sec": 9200 * factor,
            "long_output_tokens_per_sec": 32 * factor,
            "peak_output_tokens_per_sec": 130 * factor,
            "peak_requests_per_sec": 0.43 * factor,
            "saturation_concurrency": 4,
            "scaling_efficiency": 0.46,
            "serial_latency": {"p50_ms": 1800 / factor, "p95_ms": 2350 / factor},
            "serial_ttft": {"p50_ms": 220 / factor, "p95_ms": 290 / factor},
            "slo_capacity": 4,
            "capacity_users": 118 * factor,
            "slo_ttft_p95_ms": 2000,
            "slo_stream_tps_p50": 15,
            "slo_error_rate": 0.01,
            "requests_per_user_hour": 60,
            "cache_probe": {"ttft_speedup": 2.4, "cache_hit_ratio": 0.9, "prefill_gain": 2.1},
            "concurrency": load,
            "context_sweep": [
                {
                    "context_tokens": size,
                    "concurrency": 1,
                    "ttft": {"p50_ms": size / 8 / factor},
                    "warm_ttft": {"p50_ms": size / 20 / factor},
                    "error_rate": 0,
                    "prompt_tokens_per_sec": 7800 * factor,
                    "output_tokens_per_sec": 33 * factor,
                    "skipped": size == 131072 and rid == 1,
                    "skip_reason": "Context limit" if size == 131072 and rid == 1 else "",
                }
                for size in (8192, 32768, 131072)
            ],
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
            QualityConfig(input_price=1, output_price=2),
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
        self.client = TestClient(app)
        self.auth = [
            patch.object(module, "get_current_user", new=AsyncMock(return_value={"id": 1}))
            for module in (runs, compare)
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
            self.assertIn("Concurrency measurements", response.text)
            self.assertIn("Context limit", response.text)
            self.assertIn("Return the answer as JSON.", response.text)
            self.assertNotIn("DO_NOT_EXPORT_API_KEY", response.text)
            self.assertNotIn("private.invalid", response.text)
        self.assertIn("identical capability questions scored by both", response.text)

    def test_challenge_cohort_and_interactive_turns_render_safely(self):
        from quality_suite import assemble_questions
        from test_loader import load_all_tests

        q = next(q for q in load_all_tests() if q.id == "H5-CL-worlds-01")
        q = assemble_questions([q], QualityConfig(generated=False, interactive=False))[0]
        quality = make_report([Result(q, "{}", 0, outcome="task_failure")],
                              QualityConfig(), ClientConfig("https://fake.invalid", "unused", "fake"))
        meta = {"metadata": {"protocol": "json-actions-v1"}, "diagnostics": {"transcript": [
            {"role": "assistant", "content": '{"tool":"read","args":{}}'},
            {"role": "tool", "content": {"value": 7}},
            {"role": "assistant", "content": "<script>bad()</script>"},
        ]}}
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE test_runs SET quality_json=?, perf_json=NULL WHERE id=1", (json.dumps(quality),))
            db.execute("UPDATE test_results SET quality_metadata_json=? WHERE run_id=1 AND test_id='Q0'",
                       (json.dumps(meta),))
            db.commit()
        for url in ("/runs/1", "/runs/1/report.html"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Ceiling-v5 challenge subset", response.text)
            self.assertIn("Interactive transcript", response.text)
            self.assertIn("Turn 2", response.text)
            self.assertIn("&lt;script&gt;bad()&lt;/script&gt;", response.text)
            self.assertNotIn("<script>bad()</script>", response.text)

    def test_performance_values_and_online_parity(self):
        selected = [asyncio.run(reports.load_run(i)) for i in (1, 2)]
        view = reports.performance_view(selected)
        row = next(r for r in view["suite_rows"] if r["label"] == "Peak aggregate output")
        self.assertEqual(row["values"], ["130.0 tok/s", "214.5 tok/s"])
        row = next(r for r in view["suite_rows"] if r["label"] == "Single-stream decode")
        self.assertEqual(row["values"], ["35.0 tok/s", "57.8 tok/s"])
        skipped = next(r for r in view["context_rows"] if r["status"] == "Context limit")
        self.assertEqual(skipped["values"][2:], ["n/a"] * 5)
        online = self.client.get("/compare?runs=1&runs=2")
        self.assertEqual(online.status_code, 200)
        for row in view["suite_rows"]:
            for value in row["values"]:
                self.assertIn(value, online.text)
        self.assertIn("Throughput under load", online.text)
        self.assertIn("/compare/report.html?runs=1&amp;runs=2", online.text)

    def test_legacy_zero_missing_partial_and_invalid_json(self):
        run = asyncio.run(reports.load_run(3))
        text = reports.render_report([run])
        self.assertIn("0.0 tok/s", text)
        self.assertIn("dedicated performance suite not recorded", text)
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
