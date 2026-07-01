"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, PlainTextResponse
from pathlib import Path

from app.config import HOST, PORT
from app.database import init_db
from app.auth import get_current_user
from app.templates_config import templates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="LLM Bench", docs_url=None, redoc_url=None, lifespan=lifespan)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


_SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


@app.middleware("http")
async def csrf_protect(request: Request, call_next):
    """Origin-based CSRF protection for state-changing requests.

    Combined with the SameSite=strict session cookie, this rejects cross-site
    form submissions: a forged POST from another origin will either omit the
    cookie (SameSite) or carry a mismatched Origin/Referer (checked here).
    """
    if request.method not in _SAFE_METHODS:
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")
        source = origin or referer
        if source is not None:
            host = urlparse(source).netloc
            if host and host != request.url.netloc:
                return PlainTextResponse("CSRF check failed: origin mismatch", status_code=403)
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.middleware("http")
async def add_user_to_context(request: Request, call_next):
    request.state.user = await get_current_user(request)
    return await call_next(request)


from app.routes import auth_routes, dashboard, runs, compare, tests_browser, admin

app.include_router(auth_routes.router)
app.include_router(dashboard.router)
app.include_router(runs.router)
app.include_router(compare.router)
app.include_router(tests_browser.router)
app.include_router(admin.router)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/dashboard", status_code=302)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
