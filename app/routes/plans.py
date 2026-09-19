"""Run plan routes: group several run configurations and schedule them."""

import json
import logging

from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import RedirectResponse

from app.auth import require_admin
from app.database import fetch_all, fetch_one
from app.templates_config import templates

logger = logging.getLogger(__name__)

router = APIRouter()


def _admin_or_redirect(request: Request):
    user = getattr(request.state, "user", None)
    if not user or user["role"] != "admin":
        return RedirectResponse(url="/login", status_code=302)
    return user


# Quality fields of the run form, JSON-serialisable per plan run. Kept in one
# place so the builder template and the POST handler agree on the schema.
# _QUALITY_FIELDS keys map straight onto make_quality_config(**kwargs).
_QUALITY_FIELDS = (
    "suite_seeds", "suite_split", "variants", "static_only",
    "quality_context_sizes", "input_price", "output_price",
)
_OPTION_FIELDS = (
    "category", "difficulty", "limit", "workers", "run_perf", "concurrency",
    "run_context", "context_sizes", "context_concurrency", "workload_mix",
    "shared_prefix", "slo_ttft_ms", "slo_tps", "slo_errors", "req_per_user_h",
)
_BOOL_SPEC_FIELDS = {"static_only", "run_perf", "run_context", "shared_prefix"}


def _spec_to_kwargs(spec: dict, fields: tuple[str, ...]) -> dict:
    """Convert one plan-run spec dict into the stringly kwargs the shared
    `make_run_options`/`make_quality_config` helpers expect."""
    kwargs: dict = {}
    for key in fields:
        value = spec.get(key)
        if key in _BOOL_SPEC_FIELDS:
            kwargs[key] = "1" if value else ""
        elif key in ("limit", "workers", "variants"):
            try:
                default = 0 if key == "limit" else 1
                kwargs[key] = int(value) if value not in (None, "") else default
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"Invalid {key} in plan run") from exc
        else:
            kwargs[key] = str(value if value is not None else "")
    return kwargs


def _plan_spec_defaults() -> dict:
    """One empty run spec as the builder's Alpine.js `addRun()` template."""
    return {
        "model_id": None, "label": "",
        "category": "", "difficulty": "", "limit": 0, "workers": 1,
        "run_perf": False, "concurrency": "1,2,4,8",
        "run_context": False, "context_sizes": "", "context_concurrency": "4",
        "workload_mix": "uniform", "shared_prefix": False,
        "slo_ttft_ms": "", "slo_tps": "", "slo_errors": "", "req_per_user_h": "",
        "suite_seeds": "1729", "suite_split": "development", "variants": 1,
        "static_only": False, "quality_context_sizes": "",
        "input_price": "", "output_price": "",
    }


def _parse_plan_runs(raw: str) -> list[dict]:
    """Decode and structurally validate the builder's runs_json payload.

    The browser sends whatever the Alpine bindings produced; model ids are
    accepted as ints or numeric strings and normalised to ints here so a
    stringly-typed select binding cannot 422 an otherwise valid plan.
    """
    try:
        specs = json.loads(raw or "")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid plan payload") from exc
    if not isinstance(specs, list) or not specs or len(specs) > 50:
        raise HTTPException(status_code=422, detail="A plan needs 1..50 runs")
    for i, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            raise HTTPException(status_code=422, detail=f"Run {i} is malformed")
        model_id = spec.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            try:
                model_id = int(model_id)
            except ValueError:
                model_id = None
        if isinstance(model_id, bool) or not isinstance(model_id, int):
            raise HTTPException(
                status_code=422,
                detail=f"Run {i} has no model selected",
            )
        spec["model_id"] = model_id
    return specs


def _local_display(utc_iso: str | None) -> str:
    """Render a stored UTC timestamp in the server's local timezone."""
    if not utc_iso:
        return ""
    from datetime import datetime, timezone

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
    return templates.TemplateResponse(request, "admin/plans.html", {"plans": plans})


@router.get("/admin/plans/new")
async def plan_new_page(request: Request):
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user
    models = await fetch_all("SELECT * FROM models ORDER BY name")
    return templates.TemplateResponse(
        request, "admin/plan_new.html",
        {"models": models, "spec_defaults_json": json.dumps(_plan_spec_defaults())},
    )


@router.post("/admin/plans")
async def plan_create(
    request: Request,
    name: str = Form(""),
    scheduled_at: str = Form(""),
    runs_json: str = Form(""),
):
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user

    from app.routes.admin import (
        _parse_scheduled_at, make_quality_config, make_run_options,
    )

    specs = _parse_plan_runs(runs_json)
    models = {m["id"] for m in await fetch_all("SELECT id FROM models")}
    for spec in specs:
        if spec["model_id"] not in models:
            raise HTTPException(status_code=422, detail="Unknown model in plan")

    scheduled = _parse_scheduled_at(scheduled_at)

    # Validate every spec before inserting anything: a plan row with a partial
    # set of runs (from a mid-loop failure) is worse than no plan at all.
    prepared: list[tuple[int, dict, dict]] = []
    for spec in specs:
        quality_config = make_quality_config(**_spec_to_kwargs(spec, _QUALITY_FIELDS))
        run_options = make_run_options(**_spec_to_kwargs(spec, _OPTION_FIELDS))
        prepared.append((spec["model_id"], quality_config, run_options))

    await _insert_plan_with_runs(
        (name or "").strip() or None, user["id"], scheduled, prepared,
    )

    from app.services import run_queue

    run_queue.dispatch_next()
    return RedirectResponse(url="/admin/plans", status_code=302)


async def _insert_plan_with_runs(
    plan_name: str | None, user_id: int, scheduled: str | None,
    prepared: list[tuple[int, dict, dict]],
) -> int:
    """Insert the plan row and all its runs in one transaction, returning the
    plan id. execute_transaction cannot hand back lastrowid, hence this."""
    from app.database import get_db

    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO run_plans (name, created_by, scheduled_at) VALUES (?, ?, ?)",
            (plan_name, user_id, scheduled),
        )
        plan_id = cursor.lastrowid
        for model_id, quality_config, run_options in prepared:
            await db.execute(
                """INSERT INTO test_runs
                       (model_id, status, created_by, workers, quality_config_json, run_options_json, plan_id)
                   VALUES (?, 'pending', ?, ?, ?, ?, ?)""",
                (model_id, user_id, run_options["workers"],
                 json.dumps(quality_config), json.dumps(run_options), plan_id),
            )
        await db.commit()
        return plan_id
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


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
