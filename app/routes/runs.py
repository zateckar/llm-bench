"""Run detail routes."""

import json
import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse

from app.auth import get_current_user, require_admin
from app.database import execute_transaction, fetch_all, fetch_one
from app.templates_config import templates
from app.services.capacity import CapacityAssumptions, estimate_capacity

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/runs/{run_id}/report.html")
async def run_report_download(request: Request, run_id: int):
    if not await get_current_user(request):
        return RedirectResponse(url="/login", status_code=302)
    from app.services.html_reports import load_run, render_report

    run = await load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return HTMLResponse(
        render_report([run]),
        headers={
            "Content-Disposition": f'attachment; filename="run-{run_id}.html"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/runs/{run_id}/quality.json")
async def quality_download(request: Request, run_id: int):
    if not await get_current_user(request):
        return RedirectResponse(url="/login", status_code=302)
    row = await fetch_one("SELECT quality_json FROM test_runs WHERE id=?", (run_id,))
    if not row or not row.get("quality_json"):
        raise HTTPException(status_code=404, detail="No quality report for this run")
    from app.benchmarking.quality_report import rescore_report
    return JSONResponse(
        rescore_report(json.loads(row["quality_json"])),
        headers={"Content-Disposition": f'attachment; filename="run-{run_id}.quality.json"'},
    )


def _parse_perf(raw: str | None) -> dict | None:
    try:
        data = json.loads(raw or "null")
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) and (data.get("schema_version") == 3 or
            data.get("schema_version") == 4 and data.get("kind") == "context_sweep") else None


@router.get("/runs/{run_id}/performance.json")
async def performance_download(request: Request, run_id: int):
    if not await get_current_user(request):
        return RedirectResponse(url="/login", status_code=302)
    row = await fetch_one("""SELECT perf_json,status,duration_ms,workers,total_questions,
        total_completion_tokens FROM test_runs WHERE id=?""", (run_id,))
    data = _parse_perf(row.get("perf_json")) if row else None
    if data is None:
        raise HTTPException(status_code=404, detail="No performance report for this run")
    from app.services.sweep_reports import hydrate_sweep
    data = await hydrate_sweep(run_id, data)
    data["capacity_estimate"] = estimate_capacity({**row, "id": run_id, "perf": data})
    return JSONResponse(data, headers={
        "Content-Disposition": f'attachment; filename="run-{run_id}.performance.json"',
        "Cache-Control": "no-store",
    })


@router.get("/runs/{run_id}/capacity")
@router.get("/runs/{run_id}/capacity.json")
@router.get("/runs/{run_id}/capacity.html")
async def capacity_results(request: Request, run_id: int, assumptions: Annotated[CapacityAssumptions, Query()]):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    if not 0 < run_id <= 2**63-1:
        raise HTTPException(status_code=404, detail="Run not found")
    run = await fetch_one("""SELECT tr.id,tr.status,tr.perf_json,tr.duration_ms,tr.workers,
        tr.total_questions,tr.total_completion_tokens,m.name AS model_name
        FROM test_runs tr JOIN models m ON m.id=tr.model_id WHERE tr.id=?""", (run_id,))
    perf = _parse_perf(run.get("perf_json")) if run else None
    if perf is None:
        raise HTTPException(status_code=404, detail="No performance report for this run")
    from app.services.sweep_reports import hydrate_sweep
    run["perf"] = await hydrate_sweep(run_id, perf)
    run["label"] = f"#{run_id} · {run['model_name']}"
    capacity = estimate_capacity(run, assumptions)
    if request.url.path.endswith(".json"):
        return JSONResponse(capacity, headers={"Content-Disposition": f'attachment; filename="run-{run_id}.capacity.json"',
                                               "Cache-Control": "no-store"})
    offline = request.url.path.endswith(".html")
    return templates.TemplateResponse(request, "capacity_download.html" if offline else "capacity.html", {
        "user": user, "run_id": run_id, "capacity_views": [capacity], "offline": offline,
        "assumptions": assumptions.model_dump(), "query_suffix": "?"+str(request.query_params) if request.query_params else "",
    }, headers={"Cache-Control": "no-store", **({"Content-Disposition": f'attachment; filename="run-{run_id}.capacity.html"'} if offline else {})})


@router.get("/runs")
async def runs_list(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    # `evaluator_error_count` is deliberately distinct from test_runs.error_count:
    # the former counts suite/evaluator bugs, the latter counts transport failures.
    runs = await fetch_all(
        """SELECT tr.id,tr.status,tr.total_questions,tr.scored_questions,tr.passed_questions,
                  tr.avg_score,tr.error_count,tr.workers,tr.latency_p50_ms,tr.latency_p95_ms,
                  tr.output_tokens_per_sec,tr.test_suite_hash,tr.created_at,tr.plan_id,
                  (tr.perf_json IS NOT NULL) AS perf_json,
                  m.name as model_name, m.model_id AS provider_model_id,
                  (SELECT COUNT(*) FROM test_results
                    WHERE run_id = tr.id
                      AND (detail LIKE 'Evaluator error:%' OR detail LIKE 'Unknown evaluator:%')
                  ) as evaluator_error_count
           FROM test_runs tr
           JOIN models m ON tr.model_id = m.id
           ORDER BY tr.id DESC"""
    )

    return templates.TemplateResponse(
        request,
        "runs_list.html",
        {"runs": runs},
    )


@router.get("/runs/{run_id}")
async def run_detail(request: Request, run_id: int):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    run = await fetch_one(
        """SELECT tr.*, m.name as model_name, m.model_id AS model_identifier
           FROM test_runs tr
           JOIN models m ON tr.model_id = m.id
           WHERE tr.id = ?""",
        (run_id,),
    )
    if not run:
        return RedirectResponse(url="/runs", status_code=302)
    run["model_id"] = run.pop("model_identifier")

    categories = await fetch_all(
        """SELECT category,
                  COUNT(*) as total,
                  SUM(COALESCE(quality_scored, request_ok)) as scored,
                  SUM(passed) as passed,
                  AVG(CASE WHEN COALESCE(quality_scored, request_ok) = 1 THEN score END) as avg_score,
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
    from app.services.html_reports import interactive_turns

    for r in results:
        r["interactive_turns"] = interactive_turns(r)
        try:
            metadata = json.loads(r.get("quality_metadata_json") or "{}")
        except (TypeError, ValueError):
            metadata = {}
        r["evaluation"] = metadata.get("evaluation")
        from app.benchmarking.quality_report import achievement_score
        r["score"] = achievement_score({**r, "evaluation": r["evaluation"]})
        r["attempt_diagnostics"] = metadata.get("metrics", {}).get("attempt_diagnostics", [])
        r["quality_requests"] = metadata.get("diagnostics", {}).get("quality_requests", [])
        r["quality_requests"] = metadata.get("diagnostics", {}).get("quality_requests", [])
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
    transport_errors = [r for r in results if not r["request_ok"]]

    perf_data = _parse_perf(run.get("perf_json"))
    from app.services.sweep_reports import hydrate_sweep
    perf_data = await hydrate_sweep(run_id, perf_data)
    try:
        quality_data = json.loads(run.get("quality_json") or "null")
    except (ValueError, TypeError):
        quality_data = None
    if quality_data and quality_data.get("schema_version") == 3:
        from app.benchmarking.quality_report import rescore_report
        quality_data = rescore_report(quality_data)
        run["avg_score"] = quality_data["summary"]["category_balanced"] or 0.0
        for category in categories:
            summary = quality_data["summary"]["categories"].get(category["category"])
            if summary:
                category["avg_score"] = summary["score"]
    from app.services.html_reports import performance_view, quality_timing_view

    label = f"#{run_id} · {run['model_name']}"
    performance = performance_view([{"label": label, "perf": perf_data or {}}])
    quality_timings = [quality_timing_view({"label": label, "results": results})] if results else []
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "run": run,
            "quality": quality_data,
            "categories": categories,
            "results_by_category": results_by_category,
            "evaluator_errors": evaluator_errors,
            "transport_errors": transport_errors,
            "perf": perf_data,
            "performance": performance,
            "quality_timings": quality_timings,
        },
    )


@router.post("/runs/{run_id}/stop")
async def stop_run(request: Request, run_id: int):
    """Stop an active run and make it removable.

    The worker checks the terminal status before starting further requests, and
    its final update is conditional so a late worker cannot turn a stopped run
    back into ``completed`` or ``running``.
    """
    try:
        await require_admin(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=302)

    run = await fetch_one("SELECT status FROM test_runs WHERE id = ?", (run_id,))
    if not run:
        return RedirectResponse(url="/runs", status_code=302)

    if run["status"] in ("running", "pending"):
        await execute_transaction(
            [
                (
                    """UPDATE test_runs
                          SET status = 'failed', completed_at = CURRENT_TIMESTAMP,
                              error_message = 'Stopped by administrator.'
                        WHERE id = ? AND status IN ('running', 'pending')""",
                    (run_id,),
                ),
                (
                    """UPDATE test_runs SET perf_json = CASE WHEN json_valid(perf_json)
                        THEN CASE WHEN json_extract(perf_json,'$.schema_version')=4
                                  AND json_extract(perf_json,'$.kind')='context_sweep'
                             THEN json_set(perf_json,'$.cancelled',json('true'),'$.finished',json('false'))
                             ELSE perf_json END ELSE perf_json END
                        WHERE id=? AND status='failed' AND error_message='Stopped by administrator.'""",
                    (run_id,),
                ),
                ("DELETE FROM benchmark_progress WHERE run_id = ?", (run_id,)),
            ]
        )
        logger.info("Stopped benchmark run %d", run_id)

    return RedirectResponse(url=f"/runs/{run_id}", status_code=302)


@router.post("/runs/{run_id}/delete")
async def delete_run(request: Request, run_id: int):
    try:
        await require_admin(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=302)

    run = await fetch_one("SELECT status FROM test_runs WHERE id = ?", (run_id,))
    if not run:
        return RedirectResponse(url="/runs", status_code=302)
    # Deleting an in-flight run orphans its background thread: the runner keeps
    # writing results into the now-missing rows and hits FK errors repeatedly.
    if run["status"] in ("running", "pending"):
        logger.warning("Refused to delete run %d while its status is '%s'", run_id, run["status"])
        return RedirectResponse(url=f"/runs/{run_id}?error=running", status_code=302)

    # One transaction: the child tables have no ON DELETE CASCADE, so three
    # separate auto-committing deletes could leave orphans on failure midway.
    await execute_transaction(
        [
            ("DELETE FROM test_results WHERE run_id = ?", (run_id,)),
            ("DELETE FROM benchmark_progress WHERE run_id = ?", (run_id,)),
            ("DELETE FROM test_runs WHERE id = ?", (run_id,)),
        ]
    )

    return RedirectResponse(url="/runs", status_code=302)


@router.post("/runs/{run_id}/rerun")
async def rerun(request: Request, run_id: int):
    """Queue the complete current suite with the model and run mode."""
    try:
        user = await require_admin(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=302)
    original = await fetch_one("SELECT * FROM test_runs WHERE id = ?", (run_id,))
    if not original:
        return RedirectResponse(url="/runs", status_code=302)
    if original["status"] in ("running", "pending"):
        return RedirectResponse(url=f"/runs/{run_id}?error=rerun-running", status_code=302)
    from app.services import run_submission

    prepared = await run_submission.validated_specs(
        json.dumps([run_submission.spec_from_run(original)])
    )
    _, run_ids = await run_submission.submit_runs(None, user["id"], None, prepared)
    return RedirectResponse(url=f"/admin/run/{run_ids[0]}/progress", status_code=302)
