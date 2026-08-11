"""Authentication: local login, session management, OIDC."""

import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request, HTTPException
from fastapi.responses import RedirectResponse

from app.config import SECRET_KEY, COOKIE_SECURE
from app.database import fetch_one

serializer = URLSafeTimedSerializer(SECRET_KEY)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_session_token(user_id: int, token_version: int = 0) -> str:
    return serializer.dumps({"user_id": user_id, "token_version": token_version})


def decode_full_session_token(token: str) -> dict | None:
    """Decode a session token, or None when invalid or expired.

    Tokens issued before token_version existed lack the key; treat them as 0.
    """
    try:
        data = serializer.loads(token, max_age=86400 * 7)  # 7 days
    except (BadSignature, SignatureExpired):
        return None
    if not data.get("user_id"):
        return None
    data.setdefault("token_version", 0)
    return data


def decode_session_token(token: str) -> int | None:
    data = decode_full_session_token(token)
    return data["user_id"] if data else None


async def get_current_user(request: Request) -> dict | None:
    """Get current user from session cookie, or None if not logged in.

    The token's token_version must match the current one in the database; a
    password change bumps it so sessions issued before the change are rejected.
    """
    token = request.cookies.get("session")
    if not token:
        return None
    data = decode_full_session_token(token)
    if not data:
        return None
    user = await fetch_one("SELECT * FROM users WHERE id = ?", (data["user_id"],))
    if not user or data["token_version"] != user["token_version"]:
        return None
    return user


async def require_auth(request: Request) -> dict:
    """Require authentication, raise redirect if not logged in."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    return user


async def require_admin(request: Request) -> dict:
    """Require admin role."""
    user = await require_auth(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def login_redirect() -> RedirectResponse:
    """Create a redirect response to login page."""
    return RedirectResponse(url="/login", status_code=302)


def set_session_cookie(response: RedirectResponse, user_id: int, token_version: int = 0) -> None:
    """Set session cookie on response."""
    token = create_session_token(user_id, token_version)
    response.set_cookie(
        "session",
        token,
        max_age=86400 * 7,
        httponly=True,
        samesite="strict",
        secure=COOKIE_SECURE,
    )


def clear_session_cookie(response: RedirectResponse) -> None:
    """Clear session cookie. Attributes must match set_cookie for browsers
    to treat it as the same cookie."""
    response.delete_cookie(
        "session",
        path="/",
        samesite="strict",
        secure=COOKIE_SECURE,
    )
