"""Background benchmark runner for the web app.

Runs in a daemon thread and writes progress and results straight to SQLite. It
shares the request layer (``llm_client.ChatClient``) and the scoring rules
(``models.Result``) with the CLI runner, so the two cannot drift apart in how
latency is measured or how a pass is decided.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# Add the project root to sys.path so the shared benchmark modules import cleanly
# whether the app is started from the repo root or from inside a container.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llm_client import ChatClient, ClientConfig  # noqa: E402
from models import LatencyStats, Question, Result  # noqa: E402

logger = logging.getLogger(__name__)

# Generation settings for web-initiated runs. Deterministic decoding keeps runs
# comparable; the perf suite deliberately re-measures with its own settings.
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
REQUEST_TIMEOUT = 180.0
MAX_WORKERS = 16


def start_benchmark(
    run_id: int,
    model: dict,
    category: str | None = None,
    limit: int | None = None,
    test_ids: list[str] | None = None,
    difficulty: str | None = None,
    workers: int = 1,
    run_perf: bool = False,
    concurrency_levels: tuple[int, ...] = (1, 2, 4, 8),
    run_context: bool = False,
    context_sizes: tuple[int, ...] = (),
    context_concurrency: int | tuple[int, ...] = 4,
) -> None:
    """Start a benchmark run in a background thread."""
    thread = threading.Thread(
        target=_run_benchmark,
        args=(run_id, model, category, limit, test_ids, difficulty, workers,
              run_perf, concurrency_levels, run_context, context_sizes,
              context_concurrency),
        daemon=True,
    )
    thread.start()


def _coerce_context_levels(context_concurrency: int | tuple[int, ...]) -> tuple[int, ...]:
    """Coerce the context-sweep concurrency argument into a level tuple.

    Older callers pass a bare int, the run form now passes the parsed list;
    both end up here, sorted and de-duplicated.
    """
    if isinstance(context_concurrency, (list, tuple)):
        levels = [int(level) for level in context_concurrency]
    else:
        levels = [int(context_concurrency)]
    return tuple(sorted({max(1, level) for level in levels}))


# ---------------------------------------------------------------------------
# Database helpers (each opens its own short-lived connection; sqlite3
# connections must not be shared across threads)
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    from app.config import DATABASE_PATH

    db = sqlite3.connect(str(DATABASE_PATH), timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def _update_progress(
    run_id: int, current_test: str, index: int, total: int,
    message: str = "", phase: str = "quality",
) -> None:
    db = _connect()
    try:
        db.execute(
            """INSERT OR REPLACE INTO benchmark_progress
                   (run_id, current_test, current_index, total, status_message, phase)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, current_test, index, total, message, phase),
        )
        db.commit()
    finally:
        db.close()


def _store_result(run_id: int, index: int, result: Result) -> None:
    q = result.question
    db = _connect()
    try:
        db.execute(
            """INSERT INTO test_results
                   (run_id, test_id, category, prompt, response, score, detail, evaluator,
                    question_index, prompt_tokens, completion_tokens, passed, pass_threshold,
                    difficulty, weight, latency_ms, ttft_ms, request_ok)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, q.id, q.category, q.prompt, result.response, result.score,
                result.detail, q.evaluator, index,
                result.tokens.prompt_tokens, result.tokens.completion_tokens,
                1 if result.passed and not result.is_transport_error else 0,
                q.pass_threshold, q.difficulty, q.effective_weight,
                result.metrics.latency_ms or None, result.metrics.ttft_ms,
                0 if result.is_transport_error else 1,
            ),
        )
        db.commit()
    finally:
        db.close()


def _finish_run(
    run_id: int,
    *,
    total: int,
    scored: int,
    passed: int,
    avg: float,
    weighted: float,
    errors: int,
    duration_ms: float,
    workers: int,
    latency: LatencyStats,
    ttft: LatencyStats,
    throughput: float | None,
    perf_json: str | None,
    error_message: str = "",
) -> None:
    status = "failed" if error_message else "completed"
    db = _connect()
    try:
        cursor = db.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0) "
            "FROM test_results WHERE run_id = ?",
            (run_id,),
        )
        total_prompt, total_completion = cursor.fetchone()

        db.execute(
            """UPDATE test_runs
                  SET status = ?, completed_at = ?, total_questions = ?, scored_questions = ?,
                      passed_questions = ?, avg_score = ?, weighted_score = ?, error_count = ?,
                      error_message = ?, total_prompt_tokens = ?, total_completion_tokens = ?,
                      duration_ms = ?, workers = ?,
                      latency_p50_ms = ?, latency_p95_ms = ?, latency_p99_ms = ?,
                      ttft_p50_ms = ?, ttft_p95_ms = ?, output_tokens_per_sec = ?,
                      perf_json = ?
                WHERE id = ?""",
            (
                status, datetime.now(timezone.utc).isoformat(), total, scored, passed,
                avg, weighted, errors, error_message, total_prompt, total_completion,
                duration_ms, workers,
                latency.p50, latency.p95, latency.p99, ttft.p50, ttft.p95, throughput,
                perf_json, run_id,
            ),
        )
        db.execute("DELETE FROM benchmark_progress WHERE run_id = ?", (run_id,))
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------


_URL_RE = re.compile(r"https?://\S+")


def _sanitize_error_detail(error: str) -> str:
    """Strip endpoint URLs/hosts from an error before it is persisted.

    Client exceptions embed the request URL (and sometimes an upstream error
    body that quotes it back), and the run detail page renders stored details
    to any authenticated user, so URLs never go into the database.
    """
    return _URL_RE.sub("<endpoint>", error)


def _build_client_config(model: dict) -> ClientConfig:
    return ClientConfig(
        base_url=model["base_url"],
        api_key=model["api_key"],
        model=model["model_id"],
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        timeout=REQUEST_TIMEOUT,
        stream=True,
    )


def _run_benchmark(
    run_id: int,
    model: dict,
    category: str | None,
    limit: int | None,
    test_ids: list[str] | None,
    difficulty: str | None,
    workers: int,
    run_perf: bool,
    concurrency_levels: tuple[int, ...],
    run_context: bool,
    context_sizes: tuple[int, ...],
    context_concurrency: int,
) -> None:
    from app.config import TESTS_DIR
    from app.services.url_guard import UnsafeURLError, validate_endpoint
    from evaluators import EVALUATORS
    from test_loader import SuiteError, compute_test_suite_hash, load_all_tests

    workers = max(1, min(MAX_WORKERS, workers))

    db = _connect()
    try:
        db.execute(
            "UPDATE test_runs SET status='running', started_at=?, workers=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), workers, run_id),
        )
        db.commit()
    finally:
        db.close()

    empty = LatencyStats()

    def fail(message: str) -> None:
        _finish_run(
            run_id, total=0, scored=0, passed=0, avg=0.0, weighted=0.0, errors=0,
            duration_ms=0.0, workers=workers, latency=empty, ttft=empty,
            throughput=None, perf_json=None, error_message=message,
        )

    # Validate the endpoint once, up front: a blocked URL should fail the run
    # immediately rather than producing N identical per-question errors.
    try:
        validate_endpoint(model["base_url"])
    except UnsafeURLError as e:
        fail(f"Blocked endpoint: {e}")
        return

    try:
        questions: list[Question] = load_all_tests(TESTS_DIR)
    except SuiteError as e:
        fail(f"Test suite is invalid: {e}")
        return
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to load the test suite for run %d", run_id)
        fail(f"Failed to load the test suite: {e}")
        return

    suite_hash = compute_test_suite_hash(TESTS_DIR)
    db = _connect()
    try:
        db.execute("UPDATE test_runs SET test_suite_hash=? WHERE id=?", (suite_hash, run_id))
        db.commit()
    finally:
        db.close()

    if test_ids:
        wanted = set(test_ids)
        questions = [q for q in questions if q.id in wanted]
    else:
        if category:
            questions = [q for q in questions if q.category.lower() == category.lower()]
        if difficulty:
            questions = [q for q in questions if q.difficulty == difficulty.lower()]
        if limit:
            questions = questions[:limit]

    if not questions:
        fail("No questions matched the selected filters")
        return

    # A crash from here on must not leave the run in 'running' forever (the
    # SSE progress loop would never terminate), so anything unexpected that
    # escapes an inner guard fails the run loudly.
    try:
        total = len(questions)
        config = _build_client_config(model)
        results: list[Result | None] = [None] * total
        completed = 0
        progress_lock = threading.Lock()

        _update_progress(run_id, "", 0, total, f"Loaded {total} questions", "quality")

        def evaluate(q: Question, response: str) -> tuple[float, str]:
            evaluator = EVALUATORS.get(q.evaluator)
            if evaluator is None:
                return 0.0, f"Unknown evaluator: {q.evaluator}"
            try:
                score, detail = evaluator(response, q.expected)
            except Exception as e:  # noqa: BLE001
                logger.warning("Evaluator error for %s: %s", q.id, e)
                return 0.0, f"Evaluator error: {e}"
            return max(0.0, min(1.0, float(score))), detail

        def work(index: int, q: Question, client: ChatClient) -> None:
            nonlocal completed
            response, tokens, metrics = client.complete(
                q.prompt, q.system_prompt, max_tokens=q.max_tokens
            )
            if metrics.ok:
                score, detail = evaluate(q, response)
            else:
                logger.warning(
                    "Request failed for %s in run %d: %s", q.id, run_id, metrics.error
                )
                score = 0.0
                detail = f"Request failed: {_sanitize_error_detail(metrics.error)}"

            result = Result(
                question=q, response=response, score=score, detail=detail,
                tokens=tokens, metrics=metrics,
            )
            results[index] = result
            _store_result(run_id, index + 1, result)

            with progress_lock:
                completed += 1
                _update_progress(
                    run_id, f"{q.category}: {q.id}", completed, total,
                    f"Ran {completed}/{total} questions", "quality",
                )

        started = time.perf_counter()
        try:
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    # One client per worker: requests.Session is not documented as
                    # thread-safe, and sharing it would add contention to the very
                    # latency numbers being recorded.
                    local = threading.local()

                    def job(idx: int, question: Question) -> None:
                        client = getattr(local, "client", None)
                        if client is None:
                            client = ChatClient(config)
                            local.client = client
                        work(idx, question, client)

                    futures = [pool.submit(job, i, q) for i, q in enumerate(questions)]
                    # job() writes the result row itself, so a worker exception
                    # means a missing row; surface it by failing the run below.
                    for future in futures:
                        future.result()
            else:
                client = ChatClient(config)
                for i, q in enumerate(questions):
                    work(i, q, client)
        except Exception as e:  # noqa: BLE001
            logger.exception("Benchmark run %d failed", run_id)
            _summarise_and_finish(
                run_id, results, total, workers,
                (time.perf_counter() - started) * 1000.0, None, str(e),
            )
            return

        duration_ms = (time.perf_counter() - started) * 1000.0

        perf_json: str | None = None
        if run_perf:
            perf_json = _run_perf_phase(run_id, config, concurrency_levels)

        if run_context and context_sizes:
            perf_json = _merge_context_phase(
                run_id, config, context_sizes, context_concurrency, perf_json
            )

        _summarise_and_finish(run_id, results, total, workers, duration_ms, perf_json, "")
    except Exception as e:  # noqa: BLE001
        logger.exception("Benchmark run %d failed", run_id)
        fail(str(e))


def _merge_context_phase(
    run_id: int,
    config: ClientConfig,
    context_sizes: tuple[int, ...],
    context_concurrency: int | tuple[int, ...],
    perf_json: str | None,
) -> str | None:
    """Run the context-length sweep and merge the results into perf_json."""
    from perf import ContextSweepConfig, run_context_sweep

    levels = _coerce_context_levels(context_concurrency)
    sweep = ContextSweepConfig(
        context_sizes=tuple(sorted(set(context_sizes))),
        # A single level keeps the pre-grid sweep exactly; only several levels
        # turn it into a context x concurrency grid.
        concurrency_per_size=levels[0],
        concurrency_levels=levels if len(levels) > 1 else (),
    )
    expected = sweep.total_requests()
    _update_progress(run_id, "", 0, expected, "Starting context scalability sweep", "context")

    def progress(phase: str, done: int, total: int) -> None:
        _update_progress(run_id, phase, done, total, f"Context sweep: {phase}", "context")

    try:
        points = run_context_sweep(config, sweep, progress=progress)
    except Exception:  # noqa: BLE001
        logger.exception("Context sweep failed for run %d", run_id)
        points = None

    base: dict = {}
    if perf_json:
        try:
            parsed = json.loads(perf_json)
            if isinstance(parsed, dict):
                base = parsed
        except json.JSONDecodeError:
            logger.warning("perf_json for run %d was not valid JSON; replacing", run_id)

    if points is None:
        base.setdefault("context_sweep_error", "context sweep failed; see logs")
        return json.dumps(base)

    base["context_sweep"] = [p.to_dict() for p in points]
    return json.dumps(base)


def _run_perf_phase(
    run_id: int, config: ClientConfig, concurrency_levels: tuple[int, ...]
) -> str | None:
    """Run the performance suite, reporting progress. Never fails the whole run."""
    from perf import PerfConfig, run_perf_suite

    perf_config = PerfConfig(concurrency_levels=tuple(sorted(set(concurrency_levels))))
    expected = perf_config.total_requests()
    _update_progress(run_id, "", 0, expected, "Starting performance suite", "perf")

    def progress(phase: str, done: int, total: int) -> None:
        _update_progress(run_id, phase, done, total, f"Performance: {phase}", "perf")

    try:
        report = run_perf_suite(config, perf_config, progress=progress)
    except Exception as e:  # noqa: BLE001
        logger.exception("Performance suite failed for run %d", run_id)
        # A perf failure must not discard the quality results already stored.
        return json.dumps({"error": str(e)})
    return json.dumps(report.to_dict())


def _summarise_and_finish(
    run_id: int,
    results: list[Result | None],
    total: int,
    workers: int,
    duration_ms: float,
    perf_json: str | None,
    error_message: str,
) -> None:
    done = [r for r in results if r is not None]
    scored = [r for r in done if not r.is_transport_error]
    passed = sum(1 for r in scored if r.passed)
    errors = len(done) - len(scored)

    avg = sum(r.score for r in scored) / len(scored) if scored else 0.0
    total_weight = sum(r.question.effective_weight for r in scored)
    weighted = (
        sum(r.score * r.question.effective_weight for r in scored) / total_weight
        if total_weight else 0.0
    )

    latency = LatencyStats.from_samples(
        [r.metrics.latency_ms for r in scored if r.metrics.latency_ms]
    )
    ttft = LatencyStats.from_samples(
        [r.metrics.ttft_ms for r in scored if r.metrics.ttft_ms is not None]
    )
    output_tokens = sum(r.tokens.completion_tokens for r in scored)
    throughput = output_tokens / (duration_ms / 1000.0) if duration_ms else None

    _finish_run(
        run_id,
        total=total,
        scored=len(scored),
        passed=passed,
        avg=avg,
        weighted=weighted,
        errors=errors,
        duration_ms=duration_ms,
        workers=workers,
        latency=latency,
        ttft=ttft,
        throughput=throughput,
        perf_json=perf_json,
        error_message=error_message,
    )
