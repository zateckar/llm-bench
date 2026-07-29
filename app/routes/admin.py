"""Admin routes: run tests, manage models, manage users."""

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse, StreamingResponse
import json
import asyncio

from app.auth import hash_password
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
        categories = sorted({q.category for q in questions})
        order = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
        difficulties = sorted({q.difficulty for q in questions}, key=lambda d: order.get(d, 9))
    except Exception as e:  # noqa: BLE001 - surfaced in the UI instead of a 500
        suite_error = str(e)

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
        },
    )


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
):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    model = await fetch_one("SELECT * FROM models WHERE id = ?", (model_id,))
    if not model:
        return RedirectResponse(url="/admin/run", status_code=302)

    workers = max(1, min(MAX_WORKERS, workers))

    run_id = await execute(
        "INSERT INTO test_runs (model_id, status, created_by, workers) VALUES (?, 'pending', ?, ?)",
        (model_id, user["id"], workers),
    )

    from app.services.benchmark_runner import start_benchmark

    start_benchmark(
        run_id,
        model,
        category=category or None,
        limit=limit or None,
        difficulty=difficulty or None,
        workers=workers,
        run_perf=bool(run_perf),
        concurrency_levels=_parse_concurrency(concurrency),
    )

    return RedirectResponse(url=f"/admin/run/{run_id}/progress", status_code=302)


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


@router.post("/admin/models/{model_id}/delete")
async def admin_delete_model(request: Request, model_id: int):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user

    await execute("DELETE FROM models WHERE id = ?", (model_id,))
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

    password_hash = hash_password(password)
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

    if password:
        password_hash = hash_password(password)
        await execute(
            "UPDATE users SET email = ?, role = ?, password_hash = ? WHERE id = ?",
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

    await execute("DELETE FROM users WHERE id = ?", (user_id,))
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

    if not verify_password(current_password, user["password_hash"]):
        return RedirectResponse(url="/admin/profile?error=1", status_code=302)

    password_hash = hash_password(new_password)
    await execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user["id"]))
    return RedirectResponse(url="/admin/profile?success=1", status_code=302)
