"""Run plan routes: group several run configurations and schedule them."""

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import RedirectResponse

from app.auth import require_admin
from app.database import fetch_all, fetch_one
from app.templates_config import templates
from app.services import run_submission

logger = logging.getLogger(__name__)

router = APIRouter()


def _admin_or_redirect(request: Request):
    user = getattr(request.state, "user", None)
    if not user or user["role"] != "admin":
        return RedirectResponse(url="/login", status_code=302)
    return user


def _local_display(utc_iso: str | None) -> str:
    """Render a stored UTC timestamp in the server's local timezone."""
    if not utc_iso:
        return ""
    try:
        dt = datetime.fromisoformat(utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return utc_iso[:16]


@router.get("/admin/plans")
async def plans_list(request: Request):
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user
    plans = await fetch_all(
        """SELECT p.*,
                  (SELECT COUNT(*) FROM test_runs r WHERE r.plan_id = p.id) AS total,
                  (SELECT COUNT(*) FROM test_runs r WHERE r.plan_id = p.id AND r.status = 'pending') AS pending,
                  (SELECT COUNT(*) FROM test_runs r WHERE r.plan_id = p.id AND r.status = 'running') AS running,
                  (SELECT COUNT(*) FROM test_runs r WHERE r.plan_id = p.id AND r.status = 'completed') AS completed,
                  (SELECT COUNT(*) FROM test_runs r WHERE r.plan_id = p.id AND r.status = 'failed') AS failed
           FROM run_plans p ORDER BY p.id DESC"""
    )
    for plan in plans:
        plan["scheduled_local"] = _local_display(plan.get("scheduled_at"))
    return templates.TemplateResponse(
        request,
        "admin/plans.html",
        {
            "plans": plans,
            "server_now_local": _local_display(datetime.now(timezone.utc).isoformat()),
        },
    )


@router.get("/admin/plans/new")
async def plan_new_page(request: Request):
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user
    return RedirectResponse(url="/admin/run", status_code=302)


@router.get("/admin/plans/{plan_id}/edit")
async def plan_edit_page(request: Request, plan_id: int):
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user
    plan = await fetch_one("SELECT * FROM run_plans WHERE id = ?", (plan_id,))
    if not plan:
        return RedirectResponse(url="/admin/plans", status_code=302)
    runs = await fetch_all(
        "SELECT model_id, run_options_json FROM test_runs WHERE plan_id = ? ORDER BY id",
        (plan_id,),
    )
    models = await fetch_all("SELECT * FROM models ORDER BY name")
    return templates.TemplateResponse(
        request,
        "admin/run_test.html",
        run_submission.form_context(
            models,
            specs=[run_submission.spec_from_run(run) for run in runs],
            name=plan.get("name") or "",
            scheduled=plan.get("scheduled_at"),
            action=f"/admin/plans/{plan_id}/edit",
            heading=f"Edit submission #{plan_id}",
            submit_label="Save changes",
        ),
    )


@router.post("/admin/plans/{plan_id}/edit")
async def plan_update(
    request: Request,
    plan_id: int,
    name: str = Form(""),
    scheduled_at: str = Form(""),
    tz_offset: str = Form(""),
    runs_json: str = Form(...),
):
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user
    plan = await fetch_one("SELECT status FROM run_plans WHERE id = ?", (plan_id,))
    if not plan:
        return RedirectResponse(url="/admin/plans", status_code=302)
    prepared = await run_submission.validated_specs(runs_json)
    scheduled = run_submission.parse_browser_local(scheduled_at, tz_offset)
    await run_submission.replace_runs(plan_id, name, user["id"], scheduled, prepared)
    return RedirectResponse(url="/admin/plans", status_code=302)


@router.post("/admin/plans/{plan_id}/clone")
async def plan_clone(request: Request, plan_id: int):
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user
    plan = await fetch_one("SELECT name, scheduled_at FROM run_plans WHERE id = ?", (plan_id,))
    if not plan:
        return RedirectResponse(url="/admin/plans", status_code=302)
    runs = await fetch_all(
        "SELECT model_id, run_options_json FROM test_runs WHERE plan_id = ? ORDER BY id",
        (plan_id,),
    )
    if not runs:
        return RedirectResponse(url="/admin/plans", status_code=302)
    prepared = await run_submission.validated_specs(
        json.dumps([run_submission.spec_from_run(run) for run in runs])
    )
    await run_submission.submit_runs(
        f"{plan['name']} (copy)" if plan["name"] else None,
        user["id"],
        plan["scheduled_at"],
        prepared,
    )
    return RedirectResponse(url="/admin/plans", status_code=302)


@router.post("/admin/plans/{plan_id}/cancel")
async def plan_cancel(request: Request, plan_id: int):
    try:
        await require_admin(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=302)

    plan = await fetch_one("SELECT status FROM run_plans WHERE id = ?", (plan_id,))
    if plan and plan["status"] == "active":
        from app.services import run_queue

        cancelled = run_queue.cancel_plan(plan_id)
        logger.info("Cancelled plan %d (%d queued runs failed)", plan_id, cancelled)
    return RedirectResponse(url="/admin/plans", status_code=302)


@router.post("/admin/plans/{plan_id}/delete")
async def plan_delete(request: Request, plan_id: int):
    """Delete a plan: stop and detach its runs, keep their results."""
    try:
        await require_admin(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=302)

    plan = await fetch_one("SELECT id FROM run_plans WHERE id = ?", (plan_id,))
    if plan:
        from app.services import run_queue

        run_queue.delete_plan(plan_id)
        logger.info("Deleted plan %d", plan_id)
    return RedirectResponse(url="/admin/plans", status_code=302)
