"""Admin routes: run tests, manage models, manage users."""

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse, StreamingResponse
import json
import asyncio
import sqlite3

from app.auth import hash_password, set_session_cookie
from app.database import fetch_all, fetch_one, execute
from app.templates_config import templates
from app.services import run_submission
from app.services.run_modes import describe
from app.services.model_settings import decoding_settings
from app.benchmarking.vllm_telemetry import gpu_indices, gpu_text, metrics_model
from fastapi import HTTPException

router = APIRouter()


def _admin_required(request: Request):
    """Check if current user is admin. Returns user or RedirectResponse."""
    user = getattr(request.state, "user", None)
    if not user or user["role"] != "admin":
        return RedirectResponse(url="/login", status_code=302)
    return user


def _metrics_model(value):
    try:
        return metrics_model(value)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _gpus(value):
    try:
        return gpu_text(gpu_indices(value))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


# --- Admin: Run Tests ---


@router.get("/admin/run")
async def admin_run_page(request: Request):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user
    models = await fetch_all("SELECT * FROM models ORDER BY name")
    specs = None
    # Links may ask for a performance-only run; the former sweep and load links do too.
    if request.query_params.get("mode") in ("performance", "sweep", "load"):
        specs = [{"model_id": None, **run_submission.make_run_options(mode="performance")}]
    return templates.TemplateResponse(
        request, "admin/run_test.html", run_submission.form_context(models, specs=specs)
    )


@router.post("/admin/run")
async def admin_start_run(
    request: Request,
    name: str = Form(""),
    scheduled_at: str = Form(""),
    tz_offset: str = Form(""),
    runs_json: str = Form(...),
):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user
    prepared = await run_submission.validated_specs(runs_json)
    scheduled = run_submission.parse_browser_local(scheduled_at, tz_offset)
    _, run_ids = await run_submission.submit_runs(name, user["id"], scheduled, prepared)
    if scheduled is None and len(run_ids) == 1:
        return RedirectResponse(url=f"/admin/run/{run_ids[0]}/progress", status_code=302)
    return RedirectResponse(url="/admin/plans", status_code=302)


@router.get("/admin/run/{run_id}/progress")
async def admin_run_progress(request: Request, run_id: int):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    run = await fetch_one("SELECT * FROM test_runs WHERE id = ?", (run_id,))
    if not run:
        return RedirectResponse(url="/admin/run", status_code=302)

    return templates.TemplateResponse(
        request,
        "admin/run_progress.html",
        {"run": run, "measures": describe(run.get("run_options_json"))},
    )


@router.get("/admin/run/{run_id}/stream")
async def admin_run_stream(request: Request, run_id: int):
    """SSE endpoint for real-time progress."""
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    async def event_generator():
        while True:
            # Stop streaming if the client has gone away, otherwise this loop
            # (and its DB queries) would run forever, leaking a worker.
            if await request.is_disconnected():
                break

            progress = await fetch_one(
                "SELECT * FROM benchmark_progress WHERE run_id = ?", (run_id,)
            )
            run = await fetch_one(
                "SELECT status, error_message FROM test_runs WHERE id = ?", (run_id,)
            )
            quality_counts = {}
            if (progress and (progress["phase"] or "quality") == "quality") or (
                run and run["status"] in ("completed", "failed")
            ):
                quality_counts = await fetch_one(
                    """SELECT COUNT(*) AS recorded_questions,
                              COALESCE(SUM(COALESCE(quality_scored, request_ok)), 0)
                                  AS scored_questions,
                              COALESCE(SUM(request_ok = 0 AND
                                  COALESCE(quality_outcome, '')
                                    != 'cancelled'), 0) AS request_errors,
                              (SELECT detail FROM test_results
                                WHERE run_id = ? AND request_ok = 0
                                  AND COALESCE(quality_outcome, '') != 'cancelled'
                                ORDER BY id DESC LIMIT 1) AS last_request_error
                         FROM test_results WHERE run_id = ?""",
                    (run_id, run_id),
                )

            if progress:
                data = {
                    "current_test": progress["current_test"] or "",
                    "current_index": progress["current_index"],
                    "total": progress["total"],
                    "status_message": progress["status_message"] or "",
                    # The run has two phases with independent totals, so the UI
                    # needs to know which one the numbers belong to.
                    "phase": progress["phase"] or "quality",
                    **quality_counts,
                }
                yield f"event: progress\ndata: {json.dumps(data)}\n\n"
            elif run and run["status"] == "pending":
                yield f"event: progress\ndata: {json.dumps({'status_message': 'Queued; waiting for an available slot.'})}\n\n"

            if run and run["status"] in ("completed", "failed"):
                data = {
                    "status": run["status"],
                    "error_message": run["error_message"] or "",
                    **quality_counts,
                }
                yield f"event: done\ndata: {json.dumps(data)}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Admin: Models ---


@router.get("/admin/models")
async def admin_models_page(request: Request):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    models = await fetch_all("SELECT * FROM models ORDER BY name")
    return templates.TemplateResponse(
        request,
        "admin/models.html",
        {"models": models},
    )


@router.post("/admin/models")
async def admin_create_model(
    request: Request,
    name: str = Form(...),
    base_url: str = Form(...),
    api_key: str = Form(...),
    model_id: str = Form(...),
    description: str = Form(""),
    temperature: str = Form("0"),
    reasoning_effort: str = Form(""),
    b300_metrics_model: str = Form(""),
    b300_gpus: str = Form(""),
):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    settings = decoding_settings(temperature, reasoning_effort)
    metric_name = _metrics_model(b300_metrics_model)
    gpus = _gpus(b300_gpus)
    from app.services.url_guard import validate_endpoint, UnsafeURLError

    try:
        validate_endpoint(base_url)
    except UnsafeURLError as e:
        models = await fetch_all("SELECT * FROM models ORDER BY name")
        return templates.TemplateResponse(
            request,
            "admin/models.html",
            {"models": models, "error": f"Invalid base URL: {e}"},
            status_code=400,
        )

    await execute(
        """INSERT INTO models
           (name, base_url, api_key, model_id, description, temperature, reasoning_effort, b300_metrics_model,
            b300_gpus)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (name, base_url, api_key, model_id, description, settings["temperature"], settings["reasoning_effort"],
         metric_name, gpus),
    )
    return RedirectResponse(url="/admin/models", status_code=302)


@router.get("/admin/models/{model_id}/edit")
async def admin_edit_model_page(request: Request, model_id: int):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user
    model = await fetch_one("SELECT * FROM models WHERE id = ?", (model_id,))
    if not model:
        return RedirectResponse(url="/admin/models", status_code=302)
    return templates.TemplateResponse(request, "admin/model_edit.html", {"model": model})


@router.post("/admin/models/{model_id}/edit")
async def admin_edit_model(
    request: Request,
    model_id: int,
    name: str = Form(...),
    base_url: str = Form(...),
    api_key: str = Form(""),
    model_id_str: str = Form("", alias="model_id"),
    description: str = Form(""),
    temperature: str | None = Form(None),
    reasoning_effort: str | None = Form(None),
    b300_metrics_model: str | None = Form(None),
    b300_gpus: str | None = Form(None),
):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    model = await fetch_one("SELECT * FROM models WHERE id = ?", (model_id,))
    if not model:
        return RedirectResponse(url="/admin/models", status_code=302)

    posted = await request.form()
    metric_name = _metrics_model(posted.get("b300_metrics_model", model.get("b300_metrics_model")))
    gpus = _gpus(posted.get("b300_gpus", model.get("b300_gpus")))
    settings = decoding_settings(
        posted.get("temperature", model["temperature"]),
        posted.get("reasoning_effort", model["reasoning_effort"]),
    )
    from app.services.url_guard import validate_endpoint, UnsafeURLError

    try:
        validate_endpoint(base_url)
    except UnsafeURLError as e:
        return templates.TemplateResponse(
            request,
            "admin/model_edit.html",
            {
                "model": dict(
                    model,
                    name=name,
                    base_url=base_url,
                    model_id=model_id_str,
                    description=description,
                    b300_metrics_model=metric_name,
                    b300_gpus=gpus,
                    **settings,
                ),
                "error": f"Invalid base URL: {e}",
            },
            status_code=400,
        )

    # Blank API key keeps the stored one, so the secret is never round-tripped
    # into the page or required for an unrelated field edit.
    new_key = api_key if api_key else model["api_key"]
    new_model_id = model_id_str or model["model_id"]
    await execute(
        """UPDATE models SET name = ?, base_url = ?, api_key = ?, model_id = ?, description = ?,
                            temperature = ?, reasoning_effort = ?, b300_metrics_model = ?, b300_gpus = ?
           WHERE id = ?""",
        (name, base_url, new_key, new_model_id, description,
         settings["temperature"], settings["reasoning_effort"], metric_name, gpus, model_id),
    )
    return RedirectResponse(url="/admin/models", status_code=302)


@router.post("/admin/models/{model_id}/delete")
async def admin_delete_model(request: Request, model_id: int):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    try:
        await execute("DELETE FROM models WHERE id = ?", (model_id,))
    except sqlite3.IntegrityError:
        models = await fetch_all("SELECT * FROM models ORDER BY name")
        return templates.TemplateResponse(
            request,
            "admin/models.html",
            {"models": models, "error": "Cannot delete model: it is referenced by existing runs."},
            status_code=400,
        )
    return RedirectResponse(url="/admin/models", status_code=302)


# --- Admin: Users ---


@router.get("/admin/users")
async def admin_users_page(request: Request):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    users = await fetch_all(
        "SELECT id, username, email, role, oidc_sub, display_name, created_at FROM users ORDER BY username"
    )
    return templates.TemplateResponse(
        request,
        "admin/users.html",
        {"users": users},
    )


@router.post("/admin/users")
async def admin_create_user(
    request: Request,
    username: str = Form(...),
    email: str = Form(""),
    password: str = Form(...),
    role: str = Form("user"),
):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    if len(password) < 8:
        users = await fetch_all(
            "SELECT id, username, email, role, oidc_sub, display_name, created_at FROM users ORDER BY username"
        )
        return templates.TemplateResponse(
            request,
            "admin/users.html",
            {"users": users, "error": "Password must be at least 8 characters."},
            status_code=400,
        )

    try:
        password_hash = hash_password(password)
    except ValueError as exc:
        users = await fetch_all(
            "SELECT id, username, email, role, oidc_sub, display_name, created_at FROM users ORDER BY username"
        )
        return templates.TemplateResponse(
            request,
            "admin/users.html",
            {"users": users, "error": str(exc)},
            status_code=400,
        )
    await execute(
        "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
        (username, email, password_hash, role),
    )
    return RedirectResponse(url="/admin/users", status_code=302)


@router.post("/admin/users/{user_id}/update")
async def admin_update_user(
    request: Request,
    user_id: int,
    email: str = Form(""),
    role: str = Form("user"),
    password: str = Form(""),
):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    if role != "admin":
        target = await fetch_one("SELECT role FROM users WHERE id = ?", (user_id,))
        if target and target["role"] == "admin":
            if user_id == user["id"]:
                error = "You cannot remove your own admin role."
            else:
                admins = await fetch_one("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'")
                error = (
                    None
                    if admins and admins["n"] > 1
                    else "Cannot demote the last remaining admin."
                )
            if error:
                users = await fetch_all(
                    "SELECT id, username, email, role, oidc_sub, display_name, created_at FROM users ORDER BY username"
                )
                return templates.TemplateResponse(
                    request,
                    "admin/users.html",
                    {"users": users, "error": error},
                    status_code=400,
                )

    if password:
        if len(password) < 8:
            users = await fetch_all(
                "SELECT id, username, email, role, oidc_sub, display_name, created_at FROM users ORDER BY username"
            )
            return templates.TemplateResponse(
                request,
                "admin/users.html",
                {"users": users, "error": "Password must be at least 8 characters."},
                status_code=400,
            )
        try:
            password_hash = hash_password(password)
        except ValueError as exc:
            users = await fetch_all(
                "SELECT id, username, email, role, oidc_sub, display_name, created_at FROM users ORDER BY username"
            )
            return templates.TemplateResponse(
                request,
                "admin/users.html",
                {"users": users, "error": str(exc)},
                status_code=400,
            )
        await execute(
            "UPDATE users SET email = ?, role = ?, password_hash = ?, token_version = token_version + 1 WHERE id = ?",
            (email, role, password_hash, user_id),
        )
    else:
        await execute(
            "UPDATE users SET email = ?, role = ? WHERE id = ?",
            (email, role, user_id),
        )
    return RedirectResponse(url="/admin/users", status_code=302)


@router.post("/admin/users/{user_id}/delete")
async def admin_delete_user(request: Request, user_id: int):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    if user_id == user["id"]:
        return RedirectResponse(url="/admin/users", status_code=302)

    try:
        await execute("DELETE FROM users WHERE id = ?", (user_id,))
    except sqlite3.IntegrityError:
        users = await fetch_all(
            "SELECT id, username, email, role, oidc_sub, display_name, created_at FROM users ORDER BY username"
        )
        return templates.TemplateResponse(
            request,
            "admin/users.html",
            {"users": users, "error": "Cannot delete user: they are referenced by existing runs."},
            status_code=400,
        )
    return RedirectResponse(url="/admin/users", status_code=302)


# --- Profile ---


@router.get("/admin/profile")
async def profile_page(request: Request):
    user = getattr(request.state, "user", None)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    return templates.TemplateResponse(request, "admin/profile.html", {})


@router.post("/admin/profile")
async def update_profile(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
):
    from app.auth import verify_password

    user = getattr(request.state, "user", None)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    if not user["password_hash"]:
        # OIDC-created account with no local password to verify against.
        return RedirectResponse(url="/admin/profile?error=3", status_code=302)

    if not verify_password(current_password, user["password_hash"]):
        return RedirectResponse(url="/admin/profile?error=1", status_code=302)

    if len(new_password) < 8:
        return RedirectResponse(url="/admin/profile?error=2", status_code=302)

    try:
        password_hash = hash_password(new_password)
    except ValueError:
        return RedirectResponse(url="/admin/profile?error=4", status_code=302)
    # Bumping token_version invalidates every other session of this user,
    # including the cookie that carries this very request.
    await execute(
        "UPDATE users SET password_hash = ?, token_version = token_version + 1 WHERE id = ?",
        (password_hash, user["id"]),
    )
    response = RedirectResponse(url="/admin/profile?success=1", status_code=302)
    # Re-issue a fresh cookie so the legitimate user stays logged in.
    set_session_cookie(response, user["id"], user["token_version"] + 1)
    return response
