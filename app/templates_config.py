"""Shared Jinja2 templates instance for all routes."""

from pathlib import Path
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


def _user_context(request: Request) -> dict:
    """Jinja2 context processor — injects `user` into every template."""
    return {"user": getattr(request.state, "user", None)}


templates = Jinja2Templates(
    directory=Path(__file__).parent / "templates",
    context_processors=[_user_context],
)
