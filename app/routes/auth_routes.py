"""Authentication routes: login, logout, OIDC."""

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse

from app.auth import (
    hash_password,
    verify_password,
    set_session_cookie,
    clear_session_cookie,
    get_current_user,
)
from app.database import fetch_one
from app.templates_config import templates

router = APIRouter()

# Verified instead of a real user's hash when the username does not exist, so
# the response time stays uniform and does not leak which usernames exist.
DUMMY_PASSWORD_HASH = hash_password("not-the-password")


@router.get("/login")
async def login_page(request: Request):
    user = await get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    error = request.query_params.get("error")
    return templates.TemplateResponse(
        request, "auth/login.html", {"error": error}
    )


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user = await fetch_one("SELECT * FROM users WHERE username = ?", (username,))
    password_hash = user["password_hash"] if user else DUMMY_PASSWORD_HASH
    if not verify_password(password, password_hash) or not user:
        return RedirectResponse(url="/login?error=1", status_code=302)

    response = RedirectResponse(url="/dashboard", status_code=302)
    set_session_cookie(response, user["id"], user["token_version"])
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    clear_session_cookie(response)
    return response
