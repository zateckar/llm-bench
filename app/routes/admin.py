"""Admin routes: run tests, manage models, manage users."""

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse, StreamingResponse
import json
import asyncio
import sqlite3
from datetime import datetime, timezone

from app.auth import hash_password, set_session_cookie
from app.database import fetch_all, fetch_one, execute
from app.templates_config import templates

router = APIRouter()


def _admin_required(request: Request):
    """Check if current user is admin. Returns user or RedirectResponse."""
    user = getattr(request.state, "user", None)
    if not user or user["role"] != "admin":
        return RedirectResponse(url="/login", status_code=302)
    return user


# --- Admin: Run Tests ---

MAX_WORKERS = 16
DEFAULT_CONCURRENCY_LEVELS = (1, 2, 4, 8)


def _parse_concurrency(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated concurrency list, falling back to the default.

    The 1..MAX_CONCURRENCY bound lives in perf.py so the web form, the CLI and the
    suite itself cannot disagree about what is measurable.
    """
    from perf import normalise_levels

    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    accepted, _dropped = normalise_levels(parts)
    return accepted or DEFAULT_CONCURRENCY_LEVELS


def _parse_context_sizes(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated list of context sizes in tokens.

    Kept intentionally simple: only strictly positive integers larger than one
    token are accepted; the probe sizes are validated again by the sweep itself.
    """
    sizes: set[int] = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value >= 1000:
            sizes.add(value)
    return tuple(sorted(sizes))


def _parse_context_concurrency(raw) -> tuple[int, ...]:
    """Parse the context-sweep concurrency field into a level list.

    Accepts a comma-separated list (a context x concurrency grid) or a bare
    int for backward compatibility with older callers. Each level is clamped
    to 1..64, matching the form's documented limits; unparseable input falls
    back to the long-standing default.
    """
    raw = str(raw if raw not in (None, "") else 4)
    levels: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            levels.add(int(part))
        except ValueError:
            continue
    if not levels:
        return (4,)
    return tuple(sorted(max(1, min(64, level)) for level in levels))


@router.get("/admin/run")
async def admin_run_page(request: Request):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    models = await fetch_all("SELECT * FROM models ORDER BY name")

    # Offer exactly the categories and difficulty tiers the suite actually
    # contains, rather than a hand-maintained list that drifts out of date.
    categories: list[str] = []
    difficulties: list[str] = []
    question_count = 0
    suite_error = None
    try:
        from app.config import TESTS_DIR
        from test_loader import load_all_tests

        questions = load_all_tests(TESTS_DIR)
        question_count = len(questions)
        categories = sorted({q.category for q in questions} | {'Interactive Tool Use','Long Context Quality'})
        order = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
        difficulties = sorted({q.difficulty for q in questions}, key=lambda d: order.get(d, 9))
    except Exception as e:  # noqa: BLE001 - surfaced in the UI instead of a 500
        suite_error = str(e)

    import perf
    from perf import MAX_CONCURRENCY

    return templates.TemplateResponse(
        request, "admin/run_test.html",
        {
            "models": models,
            "categories": categories,
            "difficulties": difficulties,
            "suite_error": suite_error,
            "question_count": question_count,
            "max_concurrency": MAX_CONCURRENCY,
            # Doubling sweeps: each shows where throughput stops scaling at a
            # different order of magnitude, so the operator picks by endpoint size.
            "concurrency_presets": [
                "1,2,4,8",
                "1,4,16,64",
                "1,8,32,128",
                "1,16,64,256",
            ],
            "default_context_sizes": ",".join(str(s) for s in perf.DEFAULT_CONTEXT_SIZES),
            "slo_defaults": {
                "ttft_ms": perf.PerfConfig().slo_ttft_p95_ms,
                "tps": perf.PerfConfig().slo_stream_tps_p50,
                "errors_pct": perf.PerfConfig().slo_error_rate * 100,
                "req_per_user_h": perf.PerfConfig().requests_per_user_hour,
            },
        },
    )


def _slo_float(raw: str, scale: float = 1.0) -> float | None:
    """Parse one SLO threshold from the form; empty/invalid -> None so the
    PerfConfig default applies instead of a silent 0 disabling the check."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value < 0:
        return None
    return value * scale


def make_run_options(
    *,
    category: str, difficulty: str, limit: int, workers: int,
    run_perf: str, concurrency: str, run_context: str,
    context_sizes: str, context_concurrency: str, workload_mix: str,
    shared_prefix: str, slo_ttft_ms: str, slo_tps: str,
    slo_errors: str, req_per_user_h: str,
) -> dict:
    """Translate one run-form field set into start_benchmark() kwargs.

    Serialisable as JSON so a pending run can be re-dispatched after a
    restart exactly as submitted.
    """
    return {
        "category": category or None,
        "limit": limit or None,
        "difficulty": difficulty or None,
        "workers": max(1, min(MAX_WORKERS, int(workers or 1))),
        "run_perf": bool(run_perf),
        "concurrency_levels": list(_parse_concurrency(concurrency)),
        "run_context": bool(run_context),
        "context_sizes": list(_parse_context_sizes(context_sizes)),
        "context_concurrency": list(_parse_context_concurrency(context_concurrency)),
        "workload_mix": workload_mix if workload_mix in ("uniform", "mixed") else "uniform",
        "shared_prefix": bool(shared_prefix),
        "slo_ttft_ms": _slo_float(slo_ttft_ms),
        "slo_tps": _slo_float(slo_tps),
        "slo_errors": _slo_float(slo_errors, scale=0.01),
        "req_per_user_h": _slo_float(req_per_user_h),
    }


def make_quality_config(
    *,
    static_only: str, suite_seeds: str, suite_split: str, variants: int,
    quality_context_sizes: str, input_price: str, output_price: str,
) -> dict:
    """Build the QualityConfig for one run form, raising HTTP 422 on bad input."""
    from dataclasses import asdict

    from fastapi import HTTPException
    from quality_suite import QualityConfig, parse_ints

    try:
        config = QualityConfig(generated=not bool(static_only), interactive=not bool(static_only),
            strengthen_code=not bool(static_only), seeds=parse_ints(suite_seeds), split=suite_split, variants=variants,
            context_sizes=parse_ints(quality_context_sizes),
            input_price=float(input_price) if input_price.strip() else None,
            output_price=float(output_price) if output_price.strip() else None)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return asdict(config)


def parse_browser_local(raw: str, tz_offset_min: str) -> str | None:
    """Convert the browser's `datetime-local` wall time to a UTC ISO string.

    ``tz_offset_min`` is the browser's ``getTimezoneOffset()`` (minutes behind
    UTC, so browser-local + offset = UTC). Falls back to server-local when the
    offset is missing/invalid (older forms), never silently accepting garbage.
    A past time is due immediately.
    """
    from datetime import timedelta

    from fastapi import HTTPException

    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        naive = datetime.strptime(raw, "%Y-%m-%dT%H:%M")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid schedule time: {raw!r}") from exc
    try:
        offset = int(tz_offset_min)
    except (TypeError, ValueError):
        offset = None
    if offset is None:
        return naive.astimezone().astimezone(timezone.utc).isoformat()
    utc = naive + timedelta(minutes=offset)  # local + getTimezoneOffset = UTC
    return utc.replace(tzinfo=timezone.utc).isoformat()


def utc_to_local_input(utc_iso: str | None) -> str:
    """Render a stored UTC ISO for a datetime-local input, in server-local time.

    The builder also ships the browser a 'now' for the offset correction, so
    the input is pre-filled consistently with how the browser will read it.
    """
    if not utc_iso:
        return ""
    try:
        dt = datetime.fromisoformat(utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%dT%H:%M")
    except ValueError:
        return ""


def _parse_scheduled_at(raw: str) -> str | None:
    """Parse a datetime-local value into UTC ISO, treating it as server-local.

    Used by the single-run /admin/run form, which has no tz-offset field; plan
    forms use parse_browser_local instead. Malformed input is a 422.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        local = datetime.strptime(raw, "%Y-%m-%dT%H:%M")
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail=f"Invalid schedule time: {raw!r}") from exc
    return local.astimezone().astimezone(timezone.utc).isoformat()

@router.post("/admin/run")
async def admin_start_run(
    request: Request,
    model_id: int = Form(...),
    category: str = Form(""),
    difficulty: str = Form(""),
    limit: int = Form(0),
    workers: int = Form(1),
    run_perf: str = Form(""),
    concurrency: str = Form("1,2,4,8"),
    run_context: str = Form(""),
    context_sizes: str = Form(""),
    context_concurrency: str = Form("4"),
    workload_mix: str = Form("uniform"),
    shared_prefix: str = Form(""),
    slo_ttft_ms: str = Form(""),
    slo_tps: str = Form(""),
    slo_errors: str = Form(""),
    req_per_user_h: str = Form(""),
    suite_seeds: str = Form("1729"),
    suite_split: str = Form("development"),
    variants: int = Form(1),
    static_only: str = Form(""),
    quality_context_sizes: str = Form(""),
    input_price: str = Form(""),
    output_price: str = Form(""),
    scheduled_at: str = Form(""),
):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    model = await fetch_one("SELECT * FROM models WHERE id = ?", (model_id,))
    if not model:
        return RedirectResponse(url="/admin/run", status_code=302)

    quality_config = make_quality_config(
        static_only=static_only, suite_seeds=suite_seeds, suite_split=suite_split,
        variants=variants, quality_context_sizes=quality_context_sizes,
        input_price=input_price, output_price=output_price)
    run_options = make_run_options(
        category=category, difficulty=difficulty, limit=limit, workers=workers,
        run_perf=run_perf, concurrency=concurrency, run_context=run_context,
        context_sizes=context_sizes, context_concurrency=context_concurrency,
        workload_mix=workload_mix, shared_prefix=shared_prefix,
        slo_ttft_ms=slo_ttft_ms, slo_tps=slo_tps, slo_errors=slo_errors,
        req_per_user_h=req_per_user_h)

    scheduled = _parse_scheduled_at(scheduled_at)
    plan_id = None
    if scheduled:
        plan_id = await execute(
            "INSERT INTO run_plans (name, created_by, scheduled_at) VALUES (NULL, ?, ?)",
            (user["id"], scheduled),
        )

    run_id = await execute(
        """INSERT INTO test_runs
               (model_id, status, created_by, workers, run_options_json, plan_id, quality_config_json)
           VALUES (?, 'pending', ?, ?, ?, ?, ?)""",
        (model_id, user["id"], run_options["workers"],
         json.dumps(run_options), plan_id, json.dumps(quality_config)),
    )

    # Always go through the queue dispatcher: at most one run executes at a
    # time, so a submission made while another run is active waits its turn.
    from app.services import run_queue

    started = run_queue.enqueue_run(run_id)
    if started:
        return RedirectResponse(url=f"/admin/run/{run_id}/progress", status_code=302)
    if scheduled:
        return RedirectResponse(url="/admin/plans", status_code=302)
    return RedirectResponse(url="/runs", status_code=302)


@router.get("/admin/run/{run_id}/progress")
async def admin_run_progress(request: Request, run_id: int):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    run = await fetch_one("SELECT * FROM test_runs WHERE id = ?", (run_id,))
    if not run:
        return RedirectResponse(url="/admin/run", status_code=302)

    return templates.TemplateResponse(
        request, "admin/run_progress.html",
        {"run": run},
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
            run = await fetch_one("SELECT status FROM test_runs WHERE id = ?", (run_id,))

            if progress:
                data = {
                    "current_test": progress["current_test"] or "",
                    "current_index": progress["current_index"],
                    "total": progress["total"],
                    "status_message": progress["status_message"] or "",
                    # The run has two phases with independent totals, so the UI
                    # needs to know which one the numbers belong to.
                    "phase": progress["phase"] or "quality",
                }
                yield f"event: progress\ndata: {json.dumps(data)}\n\n"
            elif run and run["status"] == "pending":
                yield f"event: progress\ndata: {json.dumps({'status_message': 'Starting benchmark...'})}\n\n"

            if run and run["status"] in ("completed", "failed"):
                yield f"event: done\ndata: {json.dumps({'status': run['status']})}\n\n"
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
        request, "admin/models.html",
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
):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

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
        "INSERT INTO models (name, base_url, api_key, model_id, description) VALUES (?, ?, ?, ?, ?)",
        (name, base_url, api_key, model_id, description),
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
):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    model = await fetch_one("SELECT * FROM models WHERE id = ?", (model_id,))
    if not model:
        return RedirectResponse(url="/admin/models", status_code=302)

    from app.services.url_guard import validate_endpoint, UnsafeURLError

    try:
        validate_endpoint(base_url)
    except UnsafeURLError as e:
        return templates.TemplateResponse(
            request, "admin/model_edit.html",
            {"model": dict(model, name=name, base_url=base_url,
                           model_id=model_id_str, description=description),
             "error": f"Invalid base URL: {e}"},
            status_code=400,
        )

    # Blank API key keeps the stored one, so the secret is never round-tripped
    # into the page or required for an unrelated field edit.
    new_key = api_key if api_key else model["api_key"]
    new_model_id = model_id_str or model["model_id"]
    await execute(
        """UPDATE models SET name = ?, base_url = ?, api_key = ?, model_id = ?, description = ?
           WHERE id = ?""",
        (name, base_url, new_key, new_model_id, description, model_id),
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
        request, "admin/users.html",
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
                admins = await fetch_one(
                    "SELECT COUNT(*) AS n FROM users WHERE role = 'admin'"
                )
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

    return templates.TemplateResponse(
        request, "admin/profile.html", {}
    )


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
