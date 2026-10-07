"""Projection isolation, lossless compression, lazy answers and bounded lists."""

from contextlib import closing
import json
import sqlite3
import unittest
import zlib
from unittest.mock import AsyncMock, patch
from starlette.requests import Request

from app.benchmarking.quality_report import paired_comparison, rescore_report
from app.routes import runs
from app.services.quality_views import projection_json, quality_projection, backfill_projections
from app.storage import MAGIC, DETECT_TYPES, pack_text, unpack_text, cancel_performance
import selftest_reports as report_fixtures


class CompressionTests(unittest.TestCase):
    def test_dense_charts_keep_all_vertices_and_gaps_with_bounded_svg_nodes(self):
        from app.services.html_reports import line_chart
        from html.parser import HTMLParser

        class Paths(HTMLParser):
            def __init__(self):
                super().__init__()
                self.paths = []
            def handle_starttag(self, tag, attrs):
                if tag == "path":
                    attrs = dict(attrs)
                    if attrs.get("stroke-width") == "2.5":
                        self.paths.append(attrs["d"])
        points = [(i, i % 37) for i in range(1000)]
        points[500] = (500, None)
        chart = line_chart([{"points": points, "color": "#123", "label": "dense"}], "Dense", "X", "ms")
        paths = Paths()
        paths.feed(chart)
        self.assertEqual(len(paths.paths), 1)
        self.assertEqual(paths.paths[0].count("M"), 2)
        self.assertEqual(paths.paths[0].count("L"), 997)
        self.assertLessEqual(chart.count("<circle"), 8)

    def test_authentication_is_reused_within_request_but_revocation_is_seen_on_next(self):
        import asyncio
        from app import auth

        token = auth.create_session_token(1, 0)
        def request():
            return Request({"type": "http", "headers": [(b"cookie", f"session={token}".encode())]})
        async def check():
            fetch = AsyncMock(return_value={"id": 1, "token_version": 0})
            with patch.object(auth, "fetch_one", fetch):
                first = request()
                self.assertIsNotNone(await auth.get_current_user(first))
                self.assertIsNotNone(await auth.get_current_user(first))
                self.assertEqual(fetch.await_count, 1)
                fetch.return_value = {"id": 1, "token_version": 1}
                self.assertIsNone(await auth.get_current_user(request()))
                self.assertEqual(fetch.await_count, 2)
        asyncio.run(check())

    def test_old_and_new_payloads_roundtrip_and_large_repeated_records_shrink(self):
        fixture = "".join(f"{i}: repeated long fixture data;" for i in range(5000))
        raw = (fixture + " different field " + fixture) * 3
        old = MAGIC + zlib.compress(raw.encode(), 6)
        packed = pack_text(raw)
        self.assertEqual(unpack_text(old), raw)
        self.assertEqual(unpack_text(packed), raw)
        self.assertLess(len(packed), len(old) / 2)
        self.assertEqual(unpack_text(pack_text("žluťoučký 🐈" * 10000)), "žluťoučký 🐈" * 10000)

    def test_cancellation_reads_compressed_reports(self):
        report = {"schema_version": 6, "kind": "staged", "finished": True,
                  "cancelled": False, "samples": "sample " * 20000}
        result = json.loads(unpack_text(cancel_performance(pack_text(json.dumps(report)))))
        self.assertEqual(result, {**report, "finished": False, "cancelled": True})


class DatabasePerformanceTests(unittest.TestCase):
    setUp = report_fixtures.ReportTests.setUp
    tearDown = report_fixtures.ReportTests.tearDown
    def test_projection_matches_full_comparison_and_does_not_read_audit_payloads(self):
        with closing(sqlite3.connect(self.path, detect_types=DETECT_TYPES)) as db, db:
            reports = [rescore_report(json.loads(db.execute(
                "SELECT quality_json FROM test_runs WHERE id=?", (rid,)).fetchone()[0])) for rid in (1, 2)]
            projections = [quality_projection(report, []) for report in reports]
            self.assertEqual(paired_comparison(*reports), paired_comparison(*projections))
            for rid, report in zip((1, 2), reports):
                db.execute("UPDATE test_runs SET quality_summary_json=?,quality_json=? WHERE id=?",
                           (projection_json(report, []), b"LLMBZ1\x00broken-audit", rid))
            db.execute("UPDATE test_results SET prompt_preview='Preview',prompt=?,response=?,quality_metadata_json=?",
                       (b"LLMBZ1\x00broken-audit",) * 3)
        for url in ("/dashboard", "/runs/1", "/compare", "/compare?runs=1&runs=2",
                    "/runs/1/answers?category=Advanced%20Coding"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("broken-audit", response.text)
        self.assertNotIn("Return the answer as JSON", self.client.get("/runs/1").text)

    def test_lazy_answer_is_run_scoped_authenticated_and_escaped(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            rid = db.execute("SELECT id FROM test_results WHERE run_id=1 LIMIT 1").fetchone()[0]
            db.execute("UPDATE test_results SET response='<script>secret()</script>' WHERE id=?", (rid,))
        response = self.client.get(f"/runs/1/results/{rid}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIn("&lt;script&gt;secret()&lt;/script&gt;", response.text)
        self.assertNotIn("<script>secret()</script>", response.text)
        self.assertEqual(self.client.get(f"/runs/2/results/{rid}").status_code, 404)
        with patch.object(runs, "get_current_user", AsyncMock(return_value=None)):
            self.assertEqual(self.client.get(f"/runs/1/results/{rid}", follow_redirects=False).status_code, 302)

    def test_backfill_is_idempotent_and_lists_are_paged(self):
        backfill_projections(self.path)
        with closing(sqlite3.connect(self.path, detect_types=DETECT_TYPES)) as db, db:
            before = db.execute("SELECT quality_summary_json FROM test_runs WHERE id=1").fetchone()[0]
            self.assertIsNotNone(before)
            self.assertEqual(db.execute("SELECT prompt_preview FROM test_results LIMIT 1").fetchone()[0],
                             "Return the answer as JSON.")
            db.executemany("INSERT INTO test_runs(model_id,status) VALUES(1,'completed')", [()] * 55)
        backfill_projections(self.path)
        with closing(sqlite3.connect(self.path, detect_types=DETECT_TYPES)) as db:
            self.assertEqual(db.execute("SELECT quality_summary_json FROM test_runs WHERE id=1").fetchone()[0], before)
        first, second = self.client.get("/runs"), self.client.get("/runs?page=2")
        self.assertEqual(len(first.context["runs"]), 50)
        self.assertEqual(len(second.context["runs"]), 8)
        self.assertFalse({r["id"] for r in first.context["runs"]} & {r["id"] for r in second.context["runs"]})
        self.assertEqual(self.client.get("/runs?page=0").status_code, 422)


if __name__ == "__main__":
    unittest.main()
