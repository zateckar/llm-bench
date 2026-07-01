"""Run detail routes."""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from app.auth import get_current_user, require_admin
from app.database import fetch_all, fetch_one, execute
from app.templates_config import templates

router = APIRouter()


def _is_evaluator_error(detail: str) -> bool:
    """Check if a result detail indicates an evaluator error."""
    return detail.startswith("Evaluator error:") or detail.startswith("Unknown evaluator:")


@router.get("/runs")
async def runs_list(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    runs = await fetch_all(
        """SELECT tr.*, m.name as model_name, m.model_id,
                  (SELECT COUNT(*) FROM test_results WHERE run_id = tr.id AND (detail LIKE 'Evaluator error:%' OR detail LIKE 'Unknown evaluator:%')) as error_count
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
        """SELECT tr.*, m.name as model_name, m.model_id, m.base_url
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
                  SUM(CASE WHEN score >= 0.5 THEN 1 ELSE 0 END) as passed,
                  AVG(score) as avg_score
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

    # Find evaluator errors only (not score failures)
    evaluator_errors = await fetch_all(
        """SELECT * FROM test_results
           WHERE run_id = ? AND (detail LIKE 'Evaluator error:%' OR detail LIKE 'Unknown evaluator:%')
           ORDER BY category, question_index""",
        (run_id,),
    )

    return templates.TemplateResponse(
        request, "run_detail.html",
        {
            "run": run,
            "categories": categories,
            "results_by_category": results_by_category,
            "evaluator_errors": evaluator_errors,
        },
    )


@router.post("/runs/{run_id}/delete")
async def delete_run(request: Request, run_id: int):
    try:
        user = await require_admin(request)
    except Exception:
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
    except Exception:
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

    # Find evaluator/system errors only (not quality/score failures)
    error_results = await fetch_all(
        """SELECT test_id FROM test_results
           WHERE run_id = ? AND (detail LIKE 'Evaluator error:%' OR detail LIKE 'Unknown evaluator:%')""",
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
