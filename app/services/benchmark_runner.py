"""Background benchmark runner."""

import logging
import sys
import time
import threading
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path so we can import existing modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests

from app.database import get_db, execute, fetch_one

logger = logging.getLogger(__name__)


def start_benchmark(run_id: int, model: dict, category: str = None, limit: int = None, test_ids: list[str] = None):
    """Start benchmark in background thread."""
    thread = threading.Thread(
        target=_run_benchmark, args=(run_id, model, category, limit, test_ids), daemon=True
    )
    thread.start()


def _update_progress(run_id: int, current_test: str, index: int, total: int, message: str = ""):
    """Update progress in DB (synchronous version for thread)."""
    import sqlite3
    from app.config import DATABASE_PATH

    db = sqlite3.connect(str(DATABASE_PATH))
    try:
        db.execute(
            """INSERT OR REPLACE INTO benchmark_progress (run_id, current_test, current_index, total, status_message)
               VALUES (?, ?, ?, ?, ?)""",
            (run_id, current_test, index, total, message),
        )
        db.commit()
    finally:
        db.close()


def _run_benchmark(run_id: int, model: dict, category: str = None, limit: int = None, test_ids: list[str] = None):
    """Execute benchmark in background thread."""
    import sqlite3
    from app.config import DATABASE_PATH, TESTS_DIR
    from test_loader import load_all_tests

    db = sqlite3.connect(str(DATABASE_PATH))
    try:
        # Mark as running
        db.execute(
            "UPDATE test_runs SET status='running', started_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), run_id),
        )
        db.commit()
    finally:
        db.close()

    questions = []
    run_total = 0
    completed_count = 0
    passed = 0
    total_score = 0.0

    try:
        # Load tests and compute suite hash
        from test_loader import compute_test_suite_hash
        questions = load_all_tests(TESTS_DIR)
        test_suite_hash = compute_test_suite_hash(TESTS_DIR)

        # Store the hash with the run
        db2 = sqlite3.connect(str(DATABASE_PATH))
        try:
            db2.execute("UPDATE test_runs SET test_suite_hash=? WHERE id=?", (test_suite_hash, run_id))
            db2.commit()
        finally:
            db2.close()

        if test_ids:
            questions = [q for q in questions if q.id in test_ids]
        elif category:
            questions = [q for q in questions if q.category.lower() == category.lower()]
        if limit and not test_ids:
            questions = questions[:limit]

        if not questions:
            _finish_run(run_id, 0, 0, 0.0, "No questions found")
            return

        run_total = len(questions)
        _update_progress(run_id, "", 1, run_total, f"Loaded {run_total} questions")

        # Import evaluators
        from evaluators import EVALUATORS

        for i, q in enumerate(questions, 1):
            _update_progress(run_id, f"{q.category}: {q.id}", i, run_total, f"Running test {i}/{run_total}")

            # Call LLM
            response, prompt_tokens, completion_tokens = _call_llm(
                q.prompt,
                q.system_prompt,
                model["base_url"],
                model["api_key"],
                model["model_id"],
            )

            # Evaluate
            evaluator = EVALUATORS.get(q.evaluator)
            if evaluator:
                try:
                    score, detail = evaluator(response, q.expected)
                except Exception as e:
                    logger.warning("Evaluator error for %s: %s", q.id, e)
                    score, detail = 0.0, f"Evaluator error: {e}"
            else:
                score, detail = 0.0, f"Unknown evaluator: {q.evaluator}"

            # Store result
            _store_result(run_id, q, response, score, detail, i, prompt_tokens, completion_tokens)

            if score >= 0.5:
                passed += 1
            total_score += score
            completed_count = i

        avg = total_score / run_total if run_total > 0 else 0.0
        _finish_run(run_id, run_total, passed, avg)

    except Exception as e:
        logger.exception("Benchmark run %d failed", run_id)
        avg = total_score / completed_count if completed_count > 0 else 0.0
        _finish_run(run_id, run_total or len(questions), passed, avg, str(e))


def _call_llm(prompt: str, system_prompt: str | None, base_url: str, api_key: str, model_id: str) -> tuple[str, int, int]:
    """Call LLM API. Returns (response_text, prompt_tokens, completion_tokens)."""
    from app.services.url_guard import validate_endpoint, UnsafeURLError

    try:
        validate_endpoint(base_url)
    except UnsafeURLError as e:
        return f"[API ERROR: blocked endpoint: {e}]", 0, 0

    base_url = base_url.rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 4096,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(3):
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"].get("content")
            if content is None:
                content = data["choices"][0]["message"].get("reasoning", "")
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            return (content.strip() if content else ""), prompt_tokens, completion_tokens
        except requests.exceptions.HTTPError:
            if resp.status_code == 429:
                time.sleep(2.0 * (2 ** attempt))
                continue
            if attempt < 2:
                time.sleep(2.0 * (2 ** attempt))
            else:
                return f"[API ERROR: HTTP {resp.status_code}]", 0, 0
        except Exception as e:
            if attempt < 2:
                time.sleep(2.0 * (2 ** attempt))
            else:
                return f"[API ERROR: {e}]", 0, 0
    return "[API ERROR: max retries exceeded]", 0, 0


def _store_result(run_id, question, response, score, detail, index, prompt_tokens=0, completion_tokens=0):
    """Store a single result in the database."""
    import sqlite3
    from app.config import DATABASE_PATH

    db = sqlite3.connect(str(DATABASE_PATH))
    try:
        db.execute(
            """INSERT INTO test_results
               (run_id, test_id, category, prompt, response, score, detail, evaluator, question_index, prompt_tokens, completion_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, question.id, question.category, question.prompt, response, score, detail, question.evaluator, index, prompt_tokens, completion_tokens),
        )
        db.commit()
    finally:
        db.close()


def _finish_run(run_id: int, total: int, passed: int, avg: float, error: str = ""):
    """Mark run as completed or failed."""
    import sqlite3
    from app.config import DATABASE_PATH

    status = "failed" if error else "completed"
    db = sqlite3.connect(str(DATABASE_PATH))
    try:
        # Calculate aggregate token usage from results
        cursor = db.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0) FROM test_results WHERE run_id = ?",
            (run_id,),
        )
        total_prompt, total_completion = cursor.fetchone()

        db.execute(
            """UPDATE test_runs
               SET status=?, completed_at=?, total_questions=?, passed_questions=?, avg_score=?, error_message=?,
                   total_prompt_tokens=?, total_completion_tokens=?
               WHERE id=?""",
            (status, datetime.now(timezone.utc).isoformat(), total, passed, avg, error,
             total_prompt, total_completion, run_id),
        )
        # Clean up progress
        db.execute("DELETE FROM benchmark_progress WHERE run_id=?", (run_id,))
        db.commit()
    finally:
        db.close()
