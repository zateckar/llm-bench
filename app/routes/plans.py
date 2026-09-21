"""Run plan routes: group several run configurations and schedule them."""

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import RedirectResponse

from app.auth import require_admin
from app.database import fetch_all, fetch_one
from app.templates_config import templates
from quality_suite import DEFAULT_MAX_OUTPUT_TOKENS, MAX_MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)

# One source of truth for the output cap, so the plan builder cannot drift from
# the suite default the CLI and the run form use.
DEFAULT_MAX_TOKENS = DEFAULT_MAX_OUTPUT_TOKENS
MAX_MAX_TOKENS = MAX_MAX_OUTPUT_TOKENS

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
    "quality_context_sizes", "input_price", "output_price", "quality_max_tokens",
    "quality_profile",
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
        elif key in ("limit", "workers", "variants", "quality_max_tokens"):
            try:
                default = (
                    DEFAULT_MAX_TOKENS if key == "quality_max_tokens"
                    else 0 if key == "limit" else 1
                )
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
        "quality_max_tokens": DEFAULT_MAX_TOKENS,
        "quality_profile": "v6",
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
    try:
        dt = datetime.fromisoformat(utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return utc_iso[:16]


def _run_row_to_spec(run: dict) -> dict:
    """Rehydrate one plan-run DB row into the builder's run-spec shape for
    pre-filling the edit/clone form."""
    opts = json.loads(run.get("run_options_json") or "{}")
    qconf = json.loads(run.get("quality_config_json") or "{}")
    def s(v):  # comma-join list/tuple, str-or-empty otherwise
        return ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else ("" if v is None else str(v))
    spec = _plan_spec_defaults()
    spec.update({
        "model_id": run["model_id"],
        "category": opts.get("category") or "",
        "difficulty": opts.get("difficulty") or "",
        "limit": opts.get("limit") or 0,
        "workers": opts.get("workers") or 1,
        "run_perf": bool(opts.get("run_perf")),
        "concurrency": s(opts.get("concurrency_levels")) or "1,2,4,8",
        "run_context": bool(opts.get("run_context")),
        "context_sizes": s(opts.get("context_sizes")),
        "context_concurrency": s(opts.get("context_concurrency")) or "4",
        "workload_mix": opts.get("workload_mix") or "uniform",
        "shared_prefix": bool(opts.get("shared_prefix")),
        "slo_ttft_ms": s(opts.get("slo_ttft_ms")),
        "slo_tps": s(opts.get("slo_tps")),
        "slo_errors": s(opts.get("slo_errors")),
        "req_per_user_h": s(opts.get("req_per_user_h")),
        "suite_seeds": s(qconf.get("seeds")) or "1729",
        "suite_split": qconf.get("split") or "development",
        "variants": qconf.get("variants") or 1,
        "static_only": not (qconf.get("generated", True)
                            or qconf.get("interactive", True)
                            or qconf.get("strengthen_code", True)),
        "quality_context_sizes": s(qconf.get("context_sizes")),
        "quality_max_tokens": qconf.get("max_output_tokens", DEFAULT_MAX_TOKENS),
        "quality_profile": qconf.get("profile", "v6"),
        "input_price": s(qconf.get("input_price")),
        "output_price": s(qconf.get("output_price")),
    })
    return spec


async def _validated_specs(runs_json: str) -> list[tuple[int, dict, dict]]:
    """Parse + model-check + build configs for every spec; 422s, no DB writes."""
    from app.routes.admin import make_quality_config, make_run_options

    specs = _parse_plan_runs(runs_json)
    models = {m["id"] for m in await fetch_all("SELECT id FROM models")}
    for spec in specs:
        if spec["model_id"] not in models:
            raise HTTPException(status_code=422, detail="Unknown model in plan")
    prepared: list[tuple[int, dict, dict]] = []
    for spec in specs:
        prepared.append((
            spec["model_id"],
            make_quality_config(**_spec_to_kwargs(spec, _QUALITY_FIELDS)),
            make_run_options(**_spec_to_kwargs(spec, _OPTION_FIELDS)),
        ))
    return prepared


def _browser_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    return templates.TemplateResponse(request, "admin/plans.html", {
        "plans": plans, "server_now_local": _local_display(_browser_now_iso()),
    })


@router.get("/admin/plans/new")
async def plan_new_page(request: Request):
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user
    models = await fetch_all("SELECT * FROM models ORDER BY name")
    return templates.TemplateResponse(
        request, "admin/plan_new.html",
        {
            "models": models,
            "spec_defaults_json": json.dumps(_plan_spec_defaults()),
            "max_max_tokens": MAX_MAX_TOKENS,
            "prefill_json": "null", "plan_name": "", "plan_scheduled_input": "",
            "action": "/admin/plans", "heading": "New Run Plan",
            "browser_now_iso": _browser_now_iso(),
        },
    )


@router.get("/admin/plans/{plan_id}/edit")
async def plan_edit_page(request: Request, plan_id: int):
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user
    plan = await fetch_one("SELECT * FROM run_plans WHERE id = ?", (plan_id,))
    if not plan:
        return RedirectResponse(url="/admin/plans", status_code=302)
    from app.routes.admin import utc_to_local_input

    runs = await fetch_all(
        "SELECT model_id, run_options_json, quality_config_json FROM test_runs WHERE plan_id = ? ORDER BY id",
        (plan_id,),
    )
    # Editing a plan edits its template; runs that already executed are history
    # and only the shape is re-usable, so every stored run becomes an editable
    # spec regardless of status.
    specs = [_run_row_to_spec(r) for r in runs] or [_plan_spec_defaults()]
    models = await fetch_all("SELECT * FROM models ORDER BY name")
    return templates.TemplateResponse(
        request, "admin/plan_new.html",
        {
            "models": models,
            "spec_defaults_json": json.dumps(_plan_spec_defaults()),
            "max_max_tokens": MAX_MAX_TOKENS,
            "prefill_json": json.dumps(specs),
            "plan_name": plan.get("name") or "",
            "plan_scheduled_input": utc_to_local_input(plan.get("scheduled_at")),
            "action": f"/admin/plans/{plan_id}/edit",
            "heading": f"Edit Plan #{plan_id}",
            "browser_now_iso": _browser_now_iso(),
        },
    )


@router.post("/admin/plans")
async def plan_create(
    request: Request,
    name: str = Form(""),
    scheduled_at: str = Form(""),
    tz_offset: str = Form(""),
    runs_json: str = Form(""),
):
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user

    from app.routes.admin import parse_browser_local

    prepared = await _validated_specs(runs_json)
    scheduled = parse_browser_local(scheduled_at, tz_offset)

    await _insert_plan_with_runs(
        (name or "").strip() or None, user["id"], scheduled, prepared,
    )

    from app.services import run_queue

    run_queue.dispatch_next()
    return RedirectResponse(url="/admin/plans", status_code=302)


@router.post("/admin/plans/{plan_id}/edit")
async def plan_update(
    request: Request,
    plan_id: int,
    name: str = Form(""),
    scheduled_at: str = Form(""),
    tz_offset: str = Form(""),
    runs_json: str = Form(""),
):
    """Replace a plan's schedule and its still-queued runs.

    Runs that already started or finished are history and keep their results;
    they are only detached from the plan. Pending runs are replaced by the
    edited spec atomically.
    """
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user

    plan = await fetch_one("SELECT status FROM run_plans WHERE id = ?", (plan_id,))
    if not plan:
        return RedirectResponse(url="/admin/plans", status_code=302)

    from app.routes.admin import parse_browser_local
    from app.database import get_db

    prepared = await _validated_specs(runs_json)
    scheduled = parse_browser_local(scheduled_at, tz_offset)

    db = await get_db()
    try:
        await db.execute(
            "UPDATE run_plans SET name = ?, scheduled_at = ?, status = 'active' WHERE id = ?",
            ((name or "").strip() or None, scheduled, plan_id),
        )
        # Detach runs already underway/finished; replace the queued ones.
        await db.execute(
            "UPDATE test_runs SET plan_id = NULL WHERE plan_id = ? AND status != 'pending'",
            (plan_id,),
        )
        await db.execute(
            """DELETE FROM test_runs WHERE plan_id = ? AND status = 'pending'""",
            (plan_id,),
        )
        for model_id, quality_config, run_options in prepared:
            await db.execute(
                """INSERT INTO test_runs
                       (model_id, status, created_by, workers, quality_config_json, run_options_json, plan_id, created_at)
                   VALUES (?, 'pending', ?, ?, ?, ?, ?, ?)""",
                (model_id, user["id"], run_options["workers"],
                 json.dumps(quality_config), json.dumps(run_options), plan_id, datetime.now(timezone.utc).isoformat()),
            )
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()

    from app.services import run_queue

    run_queue.dispatch_next()
    return RedirectResponse(url="/admin/plans", status_code=302)


@router.post("/admin/plans/{plan_id}/clone")
async def plan_clone(request: Request, plan_id: int):
    """Clone a plan as a fresh pending plan with the same run specs."""
    user = _admin_or_redirect(request)
    if isinstance(user, RedirectResponse):
        return user

    plan = await fetch_one("SELECT name, scheduled_at FROM run_plans WHERE id = ?", (plan_id,))
    if not plan:
        return RedirectResponse(url="/admin/plans", status_code=302)
    runs = await fetch_all(
        "SELECT model_id, run_options_json, quality_config_json FROM test_runs WHERE plan_id = ? ORDER BY id",
        (plan_id,),
    )
    if not runs:
        return RedirectResponse(url="/admin/plans", status_code=302)

    from app.database import get_db

    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO run_plans (name, created_by, scheduled_at) VALUES (?, ?, ?)",
            (f"{plan['name']} (copy)" if plan["name"] else None, user["id"],
             plan["scheduled_at"]),
        )
        new_plan_id = cursor.lastrowid
        for r in runs:
            opts = json.loads(r["run_options_json"] or "{}")
            await db.execute(
                """INSERT INTO test_runs
                       (model_id, status, created_by, workers, quality_config_json, run_options_json, plan_id, created_at)
                   VALUES (?, 'pending', ?, ?, ?, ?, ?, ?)""",
                (r["model_id"], user["id"], opts.get("workers") or 1,
                 r["quality_config_json"], r["run_options_json"], new_plan_id, datetime.now(timezone.utc).isoformat()),
            )
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()

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
                       (model_id, status, created_by, workers, quality_config_json, run_options_json, plan_id, created_at)
                   VALUES (?, 'pending', ?, ?, ?, ?, ?, ?)""",
                (model_id, user_id, run_options["workers"],
                 json.dumps(quality_config), json.dumps(run_options), plan_id, datetime.now(timezone.utc).isoformat()),
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
