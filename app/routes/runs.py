"""Run detail routes."""

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.auth import get_current_user, require_admin
from app.database import fetch_all, fetch_one, execute
from app.templates_config import templates

logger = logging.getLogger(__name__)

router = APIRouter()


def _is_evaluator_error(detail: str) -> bool:
    """Check if a result detail indicates an evaluator error."""
    return detail.startswith("Evaluator error:") or detail.startswith("Unknown evaluator:")


def _parse_perf(raw: str | None) -> dict | None:
    """Decode the stored performance report, tolerating a partial/failed write."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Could not decode perf_json for a run; ignoring it")
        return None
    return data if isinstance(data, dict) else None


def _context_points(perf: dict | None) -> list[dict]:
    """Return the context-scalability entries stored alongside the perf report.

    Sorted by (size, level): a context x concurrency grid stores several
    points per size and the table groups them visually by size.
    """
    if not perf:
        return []
    points = perf.get("context_sweep")
    if isinstance(points, list):
        rows = [p for p in points if isinstance(p, dict)]
        return sorted(
            rows, key=lambda p: (p.get("context_tokens") or 0, p.get("concurrency") or 0)
        )
    return []


@router.get("/runs")
async def runs_list(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    # `evaluator_error_count` is deliberately distinct from test_runs.error_count:
    # the former counts suite/evaluator bugs, the latter counts transport failures.
    runs = await fetch_all(
        """SELECT tr.*, m.name as model_name, m.model_id,
                  (SELECT COUNT(*) FROM test_results
                    WHERE run_id = tr.id
                      AND (detail LIKE 'Evaluator error:%' OR detail LIKE 'Unknown evaluator:%')
                  ) as evaluator_error_count
           FROM test_runs tr
           JOIN models m ON tr.model_id = m.id
           ORDER BY tr.id DESC"""
    )

    return templates.TemplateResponse(
        request, "runs_list.html",
        {"runs": runs},
    )


@router.get("/runs/{run_id}")
async def run_detail(request: Request, run_id: int):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    run = await fetch_one(
        """SELECT tr.*, m.name as model_name, m.model_id
           FROM test_runs tr
           JOIN models m ON tr.model_id = m.id
           WHERE tr.id = ?""",
        (run_id,),
    )
    if not run:
        return RedirectResponse(url="/runs", status_code=302)

    categories = await fetch_all(
        """SELECT category,
                  COUNT(*) as total,
                  SUM(request_ok) as scored,
                  SUM(passed) as passed,
                  AVG(CASE WHEN request_ok = 1 THEN score END) as avg_score,
                  SUM(CASE WHEN request_ok = 1 THEN score * weight END) /
                      NULLIF(SUM(CASE WHEN request_ok = 1 THEN weight END), 0) as weighted_score,
                  AVG(latency_ms) as avg_latency_ms,
                  MAX(latency_ms) as max_latency_ms
           FROM test_results WHERE run_id = ?
           GROUP BY category ORDER BY category""",
        (run_id,),
    )

    results = await fetch_all(
        """SELECT * FROM test_results WHERE run_id = ?
           ORDER BY category, question_index""",
        (run_id,),
    )

    results_by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in results_by_category:
            results_by_category[cat] = []
        results_by_category[cat].append(r)

    # Evaluator/suite bugs and transport failures are different problems from a
    # wrong answer, and are surfaced separately so they are not read as quality.
    evaluator_errors = await fetch_all(
        """SELECT * FROM test_results
           WHERE run_id = ? AND (detail LIKE 'Evaluator error:%' OR detail LIKE 'Unknown evaluator:%')
           ORDER BY category, question_index""",
        (run_id,),
    )
    transport_errors = await fetch_all(
        """SELECT * FROM test_results
           WHERE run_id = ? AND request_ok = 0
           ORDER BY category, question_index""",
        (run_id,),
    )

    difficulty_rows = await fetch_all(
        """SELECT difficulty,
                  COUNT(*) as total,
                  SUM(request_ok) as scored,
                  SUM(passed) as passed,
                  AVG(CASE WHEN request_ok = 1 THEN score END) as avg_score
           FROM test_results WHERE run_id = ?
           GROUP BY difficulty""",
        (run_id,),
    )
    tier_order = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
    difficulty_rows.sort(key=lambda r: tier_order.get(r["difficulty"], 9))

    perf_data = _parse_perf(run.get("perf_json"))
    return templates.TemplateResponse(
        request, "run_detail.html",
        {
            "run": run,
            "categories": categories,
            "results_by_category": results_by_category,
            "evaluator_errors": evaluator_errors,
            "transport_errors": transport_errors,
            "difficulty_rows": difficulty_rows,
            "perf": perf_data,
            "context_points": _context_points(perf_data),
        },
    )


@router.post("/runs/{run_id}/delete")
async def delete_run(request: Request, run_id: int):
    try:
        await require_admin(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=302)

    await execute("DELETE FROM test_results WHERE run_id = ?", (run_id,))
    await execute("DELETE FROM benchmark_progress WHERE run_id = ?", (run_id,))
    await execute("DELETE FROM test_runs WHERE id = ?", (run_id,))

    return RedirectResponse(url="/runs", status_code=302)


@router.post("/runs/{run_id}/rerun-failed")
async def rerun_failed(request: Request, run_id: int):
    """Create a new run with only the failed questions from a previous run."""
    try:
        user = await require_admin(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=302)

    # Get the original run
    original_run = await fetch_one(
        """SELECT tr.*, m.name as model_name, m.model_id
           FROM test_runs tr JOIN models m ON tr.model_id = m.id
           WHERE tr.id = ?""",
        (run_id,),
    )
    if not original_run:
        return RedirectResponse(url="/runs", status_code=302)

    # Re-run only the questions that failed for an infrastructure reason - an
    # evaluator crash or a transport error - never questions the model got wrong.
    error_results = await fetch_all(
        """SELECT DISTINCT test_id FROM test_results
           WHERE run_id = ?
             AND (request_ok = 0
                  OR detail LIKE 'Evaluator error:%'
                  OR detail LIKE 'Unknown evaluator:%')""",
        (run_id,),
    )

    if not error_results:
        return RedirectResponse(url=f"/runs/{run_id}", status_code=302)

    error_test_ids = [r["test_id"] for r in error_results]

    # Create new run
    new_run_id = await execute(
        "INSERT INTO test_runs (model_id, status, created_by) VALUES (?, 'pending', ?)",
        (original_run["model_id"], user["id"]),
    )

    # Start benchmark with only errored questions
    model = await fetch_one("SELECT * FROM models WHERE id = ?", (original_run["model_id"],))
    from app.services.benchmark_runner import start_benchmark
    start_benchmark(new_run_id, model, test_ids=error_test_ids)

    return RedirectResponse(url=f"/admin/run/{new_run_id}/progress", status_code=302)
