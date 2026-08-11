"""SQLite database connection and helpers."""

import aiosqlite
from pathlib import Path

from app.config import DATABASE_PATH, DATA_DIR

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Columns added after the initial release. `CREATE TABLE IF NOT EXISTS` in
# schema.sql does not alter an existing table, so every added column needs an
# idempotent ALTER here as well. Listed as (table, column, definition).
MIGRATIONS: list[tuple[str, str, str]] = [
    ("test_runs", "test_suite_hash", "TEXT"),
    ("test_runs", "total_prompt_tokens", "INTEGER DEFAULT 0"),
    ("test_runs", "total_completion_tokens", "INTEGER DEFAULT 0"),
    ("test_runs", "weighted_score", "REAL DEFAULT 0.0"),
    ("test_runs", "scored_questions", "INTEGER DEFAULT 0"),
    ("test_runs", "error_count", "INTEGER DEFAULT 0"),
    ("test_runs", "workers", "INTEGER DEFAULT 1"),
    ("test_runs", "duration_ms", "REAL DEFAULT 0.0"),
    ("test_runs", "latency_p50_ms", "REAL"),
    ("test_runs", "latency_p95_ms", "REAL"),
    ("test_runs", "latency_p99_ms", "REAL"),
    ("test_runs", "ttft_p50_ms", "REAL"),
    ("test_runs", "ttft_p95_ms", "REAL"),
    ("test_runs", "output_tokens_per_sec", "REAL"),
    ("test_runs", "perf_json", "TEXT"),
    ("test_results", "prompt_tokens", "INTEGER DEFAULT 0"),
    ("test_results", "completion_tokens", "INTEGER DEFAULT 0"),
    ("test_results", "passed", "INTEGER DEFAULT 0"),
    ("test_results", "pass_threshold", "REAL DEFAULT 1.0"),
    ("test_results", "difficulty", "TEXT DEFAULT 'medium'"),
    ("test_results", "weight", "REAL DEFAULT 1.0"),
    ("test_results", "latency_ms", "REAL"),
    ("test_results", "ttft_ms", "REAL"),
    ("test_results", "request_ok", "INTEGER DEFAULT 1"),
    ("benchmark_progress", "phase", "TEXT DEFAULT 'quality'"),
]


async def _apply_migrations(db: aiosqlite.Connection) -> None:
    """Add any missing columns, backfilling `passed` only when it is added."""
    for table, column, definition in MIGRATIONS:
        try:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except Exception:
            continue  # Column already exists.

        # Rows written before `passed` existed default to 0, which would report
        # every historical run as a total failure. Backfill them with the old
        # 0.5 rule so old runs stay readable. Only valid on rows from before the
        # column existed, so this must run only when the ALTER actually
        # happened above; on later startups it would retroactively flip
        # current-runner rows that legitimately failed a higher threshold.
        if table == "test_results" and column == "passed":
            await db.execute(
                """UPDATE test_results
                      SET passed = CASE WHEN score >= 0.5 THEN 1 ELSE 0 END,
                          pass_threshold = 0.5
                    WHERE passed = 0 AND score >= 0.5"""
            )


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

        await _apply_migrations(db)

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
