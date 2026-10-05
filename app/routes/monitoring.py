"""Monitoring pages: deployment fingerprints, canaries and events."""

from urllib.parse import quote

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.auth import get_current_user
from app.database import fetch_all
from app.services import monitoring
from app.services.monitoring import MonitoringError
from app.services.url_guard import UnsafeURLError
from app.templates_config import templates

router = APIRouter()


async def _user(request):
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user


async def _admin(request):
    user = await _user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Administrators only")
    return user


def _back(url, error=None):
    return RedirectResponse(url=f"{url}?error={quote(str(error))}" if error else url, status_code=303)


async def _suite_choices():
    from app.benchmarking import usecase_suites
    from app.benchmarking.suites import suite_choices

    choices = [{"name": c["name"], "label": c["label"]} for c in suite_choices()]
    rows = await run_in_threadpool(usecase_suites.latest_versions)
    return choices + [{"name": c["name"], "label": c["label"]} for c in map(usecase_suites.choice, rows)]


@router.get("/monitoring")
async def monitoring_page(request: Request, error: str | None = None):
    user = await _user(request)
    is_admin = user["role"] == "admin"
    context = {
        "open_events": await run_in_threadpool(lambda: monitoring.events(open_only=True, limit=200)),
        "recent_events": await run_in_threadpool(lambda: monitoring.events(limit=50)),
        "canaries": await run_in_threadpool(monitoring.canaries_overview),
        "checks": await run_in_threadpool(monitoring.latest_checks),
        "is_admin": is_admin, "error": error,
        "limits": {"min": monitoring.MIN_HOURS, "max": monitoring.MAX_HOURS, "concurrency": monitoring.MAX_CONCURRENCY},
    }
    if is_admin:
        context["models"] = await fetch_all("SELECT id, name, model_id FROM models ORDER BY name")
        context["suites"] = await _suite_choices()
    return templates.TemplateResponse(request, "monitoring.html", context)


@router.post("/monitoring/events/ack")
async def acknowledge(request: Request):
    user = await _admin(request)
    form = await request.form()
    ids = [value for value in form.getlist("event_id") if str(value).isdigit()]
    await run_in_threadpool(monitoring.acknowledge, ids, user["id"])
    return _back(form.get("next") if str(form.get("next", "")).startswith("/monitoring") else "/monitoring")


@router.post("/monitoring/canaries")
async def create_canary(request: Request, name: str = Form(""), model_id: int = Form(...), suite: str = Form(...),
                        interval_hours: str = Form("24"), max_concurrency: str = Form("4"),
                        webhook_url: str = Form("")):
    user = await _admin(request)
    try:
        canary_id = await run_in_threadpool(lambda: monitoring.create_canary(
            user["id"], name=name, model_id=model_id, suite=suite, interval_hours=interval_hours,
            max_concurrency=max_concurrency, webhook_url=webhook_url))
    except MonitoringError as error:
        return _back("/monitoring", error)
    return RedirectResponse(url=f"/monitoring/canaries/{canary_id}", status_code=303)


async def _canary(canary_id):
    canary = await run_in_threadpool(monitoring.get_canary, canary_id)
    if canary is None:
        raise HTTPException(status_code=404, detail="Canary not found")
    return canary


@router.get("/monitoring/canaries/{canary_id}")
async def canary_page(request: Request, canary_id: int, error: str | None = None):
    user = await _user(request)
    canary = await _canary(canary_id)
    is_admin = user["role"] == "admin"
    context = {
        "canary": canary, "history": await run_in_threadpool(monitoring.canary_history, canary_id),
        "events": await run_in_threadpool(lambda: monitoring.events(canary_id=canary_id, limit=50)),
        "is_admin": is_admin, "error": error,
        "limits": {"min": monitoring.MIN_HOURS, "max": monitoring.MAX_HOURS, "concurrency": monitoring.MAX_CONCURRENCY},
    }
    if is_admin:
        context["models"] = await fetch_all("SELECT id, name, model_id FROM models ORDER BY name")
        context["suites"] = await _suite_choices()
    return templates.TemplateResponse(request, "monitoring_canary.html", context)


@router.post("/monitoring/canaries/{canary_id}/edit")
async def edit_canary(request: Request, canary_id: int, name: str = Form(""), model_id: int = Form(...),
                      suite: str = Form(...), interval_hours: str = Form("24"), max_concurrency: str = Form("4"),
                      webhook_url: str = Form("")):
    await _admin(request)
    await _canary(canary_id)
    try:
        await run_in_threadpool(lambda: monitoring.update_canary(
            canary_id, name=name, model_id=model_id, suite=suite, interval_hours=interval_hours,
            max_concurrency=max_concurrency, webhook_url=webhook_url))
    except MonitoringError as error:
        return _back(f"/monitoring/canaries/{canary_id}", error)
    return _back(f"/monitoring/canaries/{canary_id}")


@router.post("/monitoring/canaries/{canary_id}/enabled")
async def enable_canary(request: Request, canary_id: int, enabled: int = Form(...)):
    await _admin(request)
    await _canary(canary_id)
    await run_in_threadpool(monitoring.set_enabled, canary_id, bool(enabled))
    return _back(f"/monitoring/canaries/{canary_id}")


@router.post("/monitoring/canaries/{canary_id}/run")
async def run_canary(request: Request, canary_id: int):
    await _admin(request)
    await _canary(canary_id)
    try:
        await run_in_threadpool(monitoring.run_now, canary_id)
    except MonitoringError as error:
        return _back(f"/monitoring/canaries/{canary_id}", error)
    return _back(f"/monitoring/canaries/{canary_id}")


@router.post("/monitoring/canaries/{canary_id}/baseline")
async def set_baseline(request: Request, canary_id: int, run_id: int = Form(...)):
    await _admin(request)
    await _canary(canary_id)
    try:
        await run_in_threadpool(monitoring.set_baseline, canary_id, run_id)
    except MonitoringError as error:
        return _back(f"/monitoring/canaries/{canary_id}", error)
    return _back(f"/monitoring/canaries/{canary_id}")


@router.post("/monitoring/canaries/{canary_id}/delete")
async def delete_canary(request: Request, canary_id: int):
    await _admin(request)
    await _canary(canary_id)
    try:
        await run_in_threadpool(monitoring.delete_canary, canary_id)
    except MonitoringError as error:
        return _back(f"/monitoring/canaries/{canary_id}", error)
    return _back("/monitoring")


@router.get("/monitoring/models/{model_id}")
async def model_page(request: Request, model_id: int, error: str | None = None):
    user = await _user(request)
    model = await fetch_all("SELECT id, name, model_id FROM models WHERE id = ?", (model_id,))
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return templates.TemplateResponse(request, "monitoring_model.html", {
        "model": model[0], "checks": await run_in_threadpool(monitoring.model_checks, model_id),
        "events": await run_in_threadpool(lambda: monitoring.events(model_id=model_id, limit=50)),
        "is_admin": user["role"] == "admin", "error": error,
    })


@router.post("/monitoring/models/{model_id}/check")
async def check_model(request: Request, model_id: int):
    await _admin(request)
    try:
        await run_in_threadpool(monitoring.check_model, model_id)
    except (MonitoringError, UnsafeURLError) as error:
        return _back(f"/monitoring/models/{model_id}", error)
    return _back(f"/monitoring/models/{model_id}")
