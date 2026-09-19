"""Compare benchmark runs and download a portable report."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth import get_current_user
from app.database import fetch_all
from app.services.html_reports import comparison_context, load_run, performance_view, render_report
from app.templates_config import templates

router = APIRouter()


def selected_run_ids(request):
    # Accept both the multi-select form and comma-separated links.
    ids = []
    for value in request.query_params.getlist("runs"):
        for item in value.split(","):
            item = item.strip()
            if (
                item.isascii()
                and item.isdigit()
                and len(item) <= 19
                and 0 < int(item) <= 2**63 - 1
                and int(item) not in ids
            ):
                ids.append(int(item))
    return ids


@router.get("/compare/report.html")
async def comparison_download(request: Request):
    if not await get_current_user(request):
        return RedirectResponse(url="/login", status_code=302)
    ids = selected_run_ids(request)
    if len(ids) < 2:
        raise HTTPException(status_code=422, detail="Select at least two distinct runs")
    selected = []
    for run_id in ids:
        run = await load_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        selected.append(run)
    return HTMLResponse(
        render_report(selected),
        headers={
            "Content-Disposition": 'attachment; filename="comparison-'
            + "-".join(map(str, ids))
            + '.html"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/compare")
async def compare_page(request: Request):
    if not await get_current_user(request):
        return RedirectResponse(url="/login", status_code=302)
    completed_runs = await fetch_all(
        """SELECT tr.*, m.name as model_name, m.model_id
           FROM test_runs tr JOIN models m ON tr.model_id = m.id
           WHERE tr.status = 'completed' ORDER BY tr.id DESC"""
    )
    selected_runs = []
    for run_id in selected_run_ids(request):
        run = await load_run(run_id)
        if run:
            selected_runs.append(run)
    context = comparison_context(selected_runs)
    return templates.TemplateResponse(
        request,
        "compare.html",
        {
            "completed_runs": completed_runs,
            "selected_runs": selected_runs,
            "selected_ids": [str(run["id"]) for run in selected_runs],
            "performance": performance_view(selected_runs),
            **context,
        },
    )
