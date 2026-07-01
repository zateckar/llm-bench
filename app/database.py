"""SQLite database connection and helpers."""

import aiosqlite
from pathlib import Path

from app.config import DATABASE_PATH, DATA_DIR

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def get_db() -> aiosqlite.Connection:
    """Get a database connection."""
    db = await aiosqlite.connect(str(DATABASE_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    """Initialize database schema and seed admin user."""
    DATA_DIR.mkdir(exist_ok=True)
    db = await get_db()
    try:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        await db.executescript(schema)

        # Migration: add test_suite_hash column if missing
        try:
            await db.execute("ALTER TABLE test_runs ADD COLUMN test_suite_hash TEXT")
        except Exception:
            pass  # Column already exists

        # Migration: add token columns if missing
        try:
            await db.execute("ALTER TABLE test_runs ADD COLUMN total_prompt_tokens INTEGER DEFAULT 0")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE test_runs ADD COLUMN total_completion_tokens INTEGER DEFAULT 0")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE test_results ADD COLUMN prompt_tokens INTEGER DEFAULT 0")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE test_results ADD COLUMN completion_tokens INTEGER DEFAULT 0")
        except Exception:
            pass

        # Seed admin user if not exists
        from app.config import ADMIN_USERNAME, ADMIN_PASSWORD, ADMIN_EMAIL
        import bcrypt

        cursor = await db.execute(
            "SELECT id FROM users WHERE username = ?", (ADMIN_USERNAME,)
        )
        existing = await cursor.fetchone()
        if not existing:
            password_hash = bcrypt.hashpw(
                ADMIN_PASSWORD.encode(), bcrypt.gensalt()
            ).decode()
            await db.execute(
                "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                (ADMIN_USERNAME, ADMIN_EMAIL, password_hash, "admin"),
            )
            print(f"Created admin user: {ADMIN_USERNAME}")

        await db.commit()
    finally:
        await db.close()


async def fetch_one(query: str, params: tuple = ()) -> dict | None:
    """Fetch a single row as dict."""
    db = await get_db()
    try:
        cursor = await db.execute(query, params)
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def fetch_all(query: str, params: tuple = ()) -> list[dict]:
    """Fetch all rows as list of dicts."""
    db = await get_db()
    try:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


async def execute(query: str, params: tuple = ()) -> int:
    """Execute a write query, return lastrowid."""
    db = await get_db()
    try:
        cursor = await db.execute(query, params)
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def execute_many(query: str, params_list: list[tuple]) -> None:
    """Execute a write query with many param sets."""
    db = await get_db()
    try:
        await db.executemany(query, params_list)
        await db.commit()
    finally:
        await db.close()
