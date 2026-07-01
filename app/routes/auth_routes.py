"""Authentication routes: login, logout, OIDC."""

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse

from app.auth import (
    verify_password,
    create_session_token,
    set_session_cookie,
    clear_session_cookie,
    get_current_user,
)
from app.database import fetch_one
from app.templates_config import templates

router = APIRouter()


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
    if not user or not verify_password(password, user["password_hash"]):
        return RedirectResponse(url="/login?error=1", status_code=302)

    response = RedirectResponse(url="/dashboard", status_code=302)
    set_session_cookie(response, user["id"])
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    clear_session_cookie(response)
    return response
