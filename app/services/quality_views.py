"""Persist small read projections alongside the complete quality audit report."""

import json
import sqlite3
from contextlib import closing

from app.benchmarking.quality_report import SCORING_REVISION, rescore_report
from app.services.language_views import language_report
from app.storage import pack_text
from app.storage import DETECT_TYPES
from app.benchmarking.language_id import NAMES

PROJECTION_REVISION = 1


def quality_projection(report, results):
    keep = ("id", "fingerprint", "category", "family", "scope", "score",
            "evaluator_score", "scored", "passed", "outcome")
    projected = {k: v for k, v in report.items() if k != "results"}
    projected["results"] = []
    for row in report.get("results", []):
        slim = {k: row.get(k) for k in keep}
        evaluation = row.get("evaluation")
        slim["evaluation"] = ({"criterion_achievement": evaluation.get("criterion_achievement")}
                              if evaluation is not None else None)
        projected["results"].append(slim)
    projected["answer_languages"] = language_report(report, results)
    projected["projection_revision"] = PROJECTION_REVISION
    return projected


def projection_json(report, results):
    return pack_text(json.dumps(quality_projection(report, results), separators=(",", ":")))


async def load_quality_view(run_id, raw_projection):
    from app.database import fetch_all, fetch_one

    try:
        view = json.loads(raw_projection or "null")
    except (TypeError, ValueError):
        view = None
    if (isinstance(view, dict) and view.get("projection_revision") == PROJECTION_REVISION
            and view.get("scoring_revision") == SCORING_REVISION):
        return view
    row = await fetch_one("SELECT quality_json FROM test_runs WHERE id=?", (run_id,))
    try:
        report = json.loads((row or {}).get("quality_json") or "null")
    except (TypeError, ValueError):
        return None
    if not isinstance(report, dict) or report.get("schema_version") != 3:
        return None
    from starlette.concurrency import run_in_threadpool

    report = await run_in_threadpool(rescore_report, report)
    # Legacy/test databases still work before the projection is backfilled.
    ids = [r["id"] for r in report["results"] if any(
        r.get("metadata", {}).get(key) in NAMES
        for key in ("answer_language", "language", "lang"))]
    responses = []
    for start in range(0, len(ids), 100):
        batch = ids[start:start + 100]
        marks = ",".join("?" for _ in batch)
        responses.extend(await fetch_all(
            f"SELECT test_id,response FROM test_results WHERE run_id=? AND test_id IN ({marks})",
            (run_id, *batch)))
    return await run_in_threadpool(quality_projection, report, responses)


def backfill_projections(path):
    with closing(sqlite3.connect(str(path), detect_types=DETECT_TYPES, timeout=30)) as db, db:
        refresh_projections(db)


def refresh_projections(db):
    """Bound memory to one report; safe inside maintenance's writer transaction."""
    for row_id, prompt in db.execute(
            "SELECT id,prompt FROM test_results WHERE prompt_preview IS NULL"):
        db.execute("UPDATE test_results SET prompt_preview=? WHERE id=?", ((prompt or "")[:80], row_id))
    for run_id, raw in db.execute("""SELECT id,quality_json FROM test_runs
            WHERE quality_summary_json IS NULL AND quality_json IS NOT NULL"""):
        try:
            report = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(report, dict) or report.get("schema_version") != 3:
            continue
        report = rescore_report(report)
        # Only declared-language answers are needed for the persisted diagnostic.
        wanted = {r["id"] for r in report["results"] if any(
            r.get("metadata", {}).get(k) in NAMES for k in ("answer_language", "language", "lang"))}
        responses = []
        wanted = sorted(wanted)
        for start in range(0, len(wanted), 100):
            batch = wanted[start:start + 100]
            marks = ",".join("?" for _ in batch)
            responses.extend({"test_id": ident, "response": response}
                             for ident, response in db.execute(
                                 f"SELECT test_id,response FROM test_results WHERE run_id=? AND test_id IN ({marks})",
                                 (run_id, *batch)))
        db.execute("UPDATE test_runs SET quality_summary_json=? WHERE id=?",
                   (projection_json(report, responses), run_id))
