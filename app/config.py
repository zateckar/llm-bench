"""Application configuration from environment variables."""

import os
import secrets
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TESTS_DIR = BASE_DIR / "tests"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{DATA_DIR / 'bench.db'}")
DATABASE_PATH = DATA_DIR / "bench.db"

# Environment: set ENVIRONMENT=production to enforce secure-by-default checks.
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
IS_PRODUCTION = ENVIRONMENT in ("production", "prod")

# Auth
_DEFAULT_SECRET = "dev-secret-change-in-production"
_DEFAULT_ADMIN_PASSWORD = "changeme"

SECRET_KEY = os.getenv("SECRET_KEY", "")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")

if IS_PRODUCTION:
    # Refuse to run with missing or default secrets in production: a known
    # SECRET_KEY lets anyone forge admin sessions, and a default admin password
    # is an instant account takeover.
    problems = []
    if not SECRET_KEY or SECRET_KEY == _DEFAULT_SECRET:
        problems.append("SECRET_KEY must be set to a strong, unique value")
    if not ADMIN_PASSWORD or ADMIN_PASSWORD == _DEFAULT_ADMIN_PASSWORD:
        problems.append("ADMIN_PASSWORD must be set to a strong, unique value")
    if problems:
        sys.stderr.write(
            "FATAL: insecure configuration in production:\n  - "
            + "\n  - ".join(problems)
            + "\n"
        )
        raise SystemExit(1)
else:
    # Development convenience: generate an ephemeral secret if none is provided,
    # and fall back to the well-known dev password. Sessions won't survive a
    # restart, which is fine for local dev and avoids shipping a fixed secret.
    if not SECRET_KEY:
        SECRET_KEY = secrets.token_urlsafe(32)
    if not ADMIN_PASSWORD:
        ADMIN_PASSWORD = _DEFAULT_ADMIN_PASSWORD

# OIDC (Keycloak)
OIDC_ENABLED = os.getenv("OIDC_ENABLED", "false").lower() == "true"
OIDC_ISSUER_URL = os.getenv("OIDC_ISSUER_URL", "")
OIDC_CLIENT_ID = os.getenv("OIDC_CLIENT_ID", "llm-bench")
OIDC_CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET", "")
OIDC_REDIRECT_URI = os.getenv("OIDC_REDIRECT_URI", "http://localhost:8000/auth/oidc/callback")

# Server. Bind to localhost by default so the app isn't exposed on all
# interfaces unless the operator explicitly opts in via HOST.
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Cookies: mark the session cookie Secure in production so it is never sent over
# plain HTTP. Allow override for local HTTPS testing.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true" if IS_PRODUCTION else "false").lower() == "true"
