#!/usr/bin/env python3
"""CLI tool for managing the LLM Bench application."""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


async def init_db():
    """Initialize the database schema."""
    from app.database import init_db as _init_db
    await _init_db()
    print("Database initialized successfully.")


async def create_admin(username: str, password: str, email: str = ""):
    """Create an admin user."""
    from app.auth import hash_password
    from app.database import get_db

    try:
        password_hash = hash_password(password)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return

    db = await get_db()
    try:
        cursor = await db.execute("SELECT id FROM users WHERE username = ?", (username,))
        existing = await cursor.fetchone()
        if existing:
            print(f"User '{username}' already exists.")
            return

        await db.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, "admin"),
        )
        await db.commit()
        print(f"Admin user '{username}' created successfully.")
    finally:
        await db.close()


async def list_users():
    """List all users."""
    from app.database import fetch_all
    users = await fetch_all("SELECT id, username, email, role, created_at FROM users ORDER BY username")
    if not users:
        print("No users found.")
        return

    print(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Role':<10} {'Created'}")
    print("-" * 85)
    for u in users:
        print(f"{u['id']:<5} {u['username']:<20} {(u['email'] or '-'):<30} {u['role']:<10} {u['created_at'][:10] if u['created_at'] else '-'}")


async def list_models():
    """List all models."""
    from app.database import fetch_all
    models = await fetch_all("SELECT id, name, model_id, base_url FROM models ORDER BY name")
    if not models:
        print("No models configured.")
        return

    print(f"{'ID':<5} {'Name':<25} {'Model ID':<30} {'Base URL'}")
    print("-" * 90)
    for m in models:
        print(f"{m['id']:<5} {m['name']:<25} {m['model_id']:<30} {m['base_url'][:40]}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage.py init-db          Initialize database")
        print("  python manage.py create-admin     Create admin user (password via prompt or stdin)")
        print("  python manage.py list-users       List all users")
        print("  python manage.py list-models      List all models")
        return

    command = sys.argv[1]

    if command == "init-db":
        asyncio.run(init_db())
    elif command == "create-admin":
        if len(sys.argv) < 3:
            print("Usage: python manage.py create-admin <username> [email]")
            print("  Password is prompted interactively, or read from stdin when piped in.")
            return
        import getpass
        username = sys.argv[2]
        email = sys.argv[3] if len(sys.argv) > 3 else ""
        if sys.stdin.isatty():
            # Read interactively so the password never lands in shell history or
            # the process listing.
            password = getpass.getpass("Password: ")
        else:
            # Non-interactive (piped) invocation: read the password from stdin.
            password = sys.stdin.readline().strip()
        if not password:
            print("ERROR: password must not be empty")
            return
        asyncio.run(create_admin(username, password, email))
    elif command == "list-users":
        asyncio.run(list_users())
    elif command == "list-models":
        asyncio.run(list_models())
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
