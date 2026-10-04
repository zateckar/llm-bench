"""Application benchmark pipeline and SQLite lifecycle."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Callable

from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.models import LatencyStats, Result
from app.benchmarking.perf import DEFAULT_MAX_CONCURRENCY, PerfConfig, run_perf_suite
from app.benchmarking.quality_execution import run_quality
from app.benchmarking.quality_report import make_report as make_quality_report, result_record, summarize
from app.benchmarking.quality_suite import MAX_OUTPUT_TOKENS, QUALITY_WORKERS, load_questions, provenance, suite_hash

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 180.0


def start_benchmark(
    run_id, model, mode="both", max_concurrency=DEFAULT_MAX_CONCURRENCY, on_finish=None
):
    threading.Thread(
        target=_run_benchmark, args=(run_id, model, mode, max_concurrency, on_finish), daemon=True
    ).start()


def _connect() -> sqlite3.Connection:
    from app.config import DATABASE_PATH

    db = sqlite3.connect(str(DATABASE_PATH), timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def _update_progress(
    run_id: int,
    current_test: str,
    index: int,
    total: int,
    message: str = "",
    phase: str = "quality",
) -> None:
    db = _connect()
    try:
        db.execute(
            """INSERT OR REPLACE INTO benchmark_progress
                   (run_id, current_test, current_index, total, status_message, phase)
               SELECT ?, ?, ?, ?, ?, ?
                WHERE EXISTS (
                    SELECT 1 FROM test_runs
                     WHERE id = ? AND status IN ('pending', 'running')
                )""",
            (run_id, current_test, index, total, message, phase, run_id),
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
                    difficulty, weight, latency_ms, ttft_ms, request_ok, quality_metadata_json, quality_scored)
               SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                WHERE EXISTS (
                    SELECT 1 FROM test_runs
                     WHERE id = ? AND status IN ('pending', 'running')
                )""",
            (
                run_id,
                q.id,
                q.category,
                q.prompt,
                result.response,
                result.score,
                result.detail,
                q.evaluator,
                index,
                result.tokens.prompt_tokens,
                result.tokens.completion_tokens,
                1 if result.passed and result.is_scored else 0,
                q.pass_threshold,
                q.difficulty,
                q.effective_weight,
                result.metrics.latency_ms or None,
                result.metrics.ttft_ms,
                1 if result.metrics.ok else 0,
                json.dumps(result_record(result)),
                int(result.is_scored),
                run_id,
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
                WHERE id = ? AND status IN ('pending', 'running')""",
            (
                status,
                datetime.now(timezone.utc).isoformat(),
                total,
                scored,
                passed,
                avg,
                weighted,
                errors,
                error_message,
                total_prompt,
                total_completion,
                duration_ms,
                workers,
                latency.p50,
                latency.p95,
                latency.p99,
                ttft.p50,
                ttft.p95,
                throughput,
                perf_json,
                run_id,
            ),
        )
        db.execute("DELETE FROM benchmark_progress WHERE run_id = ?", (run_id,))
        db.commit()
    finally:
        db.close()


def _mark_run_failed(run_id: int, message: str, workers: int = 1) -> None:
    """Best-effort terminal update used when normal finalisation itself fails.

    This deliberately updates only active rows. A stop request or a previous
    finalisation must win over a late exception from the background thread.
    """
    db = _connect()
    try:
        # Answers are committed individually. Recover their summary even if
        # report generation or the normal finalisation failed afterwards.
        db.execute("BEGIN IMMEDIATE")
        recorded, scored, passed, avg, weighted, errors, prompt, completion = db.execute(
            """SELECT COUNT(*),
                      COALESCE(SUM(COALESCE(quality_scored, request_ok)), 0),
                      COALESCE(SUM(passed), 0),
                      COALESCE(AVG(CASE WHEN COALESCE(quality_scored, request_ok) = 1
                                        THEN score END), 0),
                      COALESCE(SUM(CASE WHEN COALESCE(quality_scored, request_ok) = 1
                                        THEN score * weight END) /
                               NULLIF(SUM(CASE WHEN COALESCE(quality_scored, request_ok) = 1
                                               THEN weight END), 0), 0),
                      COUNT(*) - COALESCE(SUM(COALESCE(quality_scored, request_ok)), 0),
                      COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0)
                 FROM test_results WHERE run_id = ?""",
            (run_id,),
        ).fetchone()
        db.execute(
            """UPDATE test_runs
                  SET status = 'failed', completed_at = ?,
                      error_message = CASE
                          WHEN error_message IS NULL OR error_message = ?
                          THEN ?
                          ELSE error_message
                      END,
                      workers = COALESCE(workers, ?),
                      total_questions = MAX(COALESCE(total_questions, 0), ?), scored_questions = ?,
                      passed_questions = ?, avg_score = ?, weighted_score = ?, error_count = ?,
                      total_prompt_tokens = ?, total_completion_tokens = ?
                WHERE id = ? AND status IN ('pending', 'running')""",
            (
                datetime.now(timezone.utc).isoformat(),
                "",
                message,
                workers,
                recorded,
                scored,
                passed,
                avg,
                weighted,
                errors,
                prompt,
                completion,
                run_id,
            ),
        )
        db.execute("DELETE FROM benchmark_progress WHERE run_id = ?", (run_id,))
        db.commit()
    finally:
        db.close()


def _safe_mark_run_failed(run_id: int, message: str, workers: int = 1) -> None:
    """Never let an exception in the failure path escape the runner thread."""
    try:
        _mark_run_failed(run_id, _sanitize_error_detail(message), workers)
    except Exception:  # noqa: BLE001
        # There is no reliable database action left to take, but the exception
        # is logged so an operator can distinguish a DB outage from a run error.
        logger.exception("Could not mark benchmark run %d as failed", run_id)


def _run_is_active(run_id: int) -> bool:
    """Return whether a worker may still write to the run."""
    db = _connect()
    try:
        row = db.execute(
            "SELECT 1 FROM test_runs WHERE id = ? AND status IN ('pending', 'running')",
            (run_id,),
        ).fetchone()
        return row is not None
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
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0,
        seed=0,
        timeout=REQUEST_TIMEOUT,
        stream_deadline=1800.0,
        stream=True,
    )


def _run_benchmark(run_id, model, mode, max_concurrency, on_finish: Callable[[int], None] | None):
    workers = min(QUALITY_WORKERS, max_concurrency)
    try:
        _run_benchmark_impl(run_id, model, mode, max_concurrency)
    except Exception as error:
        logger.exception("Benchmark run %d failed", run_id)
        _safe_mark_run_failed(run_id, str(error), workers)
    finally:
        if on_finish is not None:
            try:
                on_finish(run_id)
            except Exception:
                logger.exception("Queue callback failed for run %d", run_id)


def _run_benchmark_impl(run_id, model, mode, max_concurrency):
    from app.services.url_guard import validate_endpoint

    if mode not in {"both", "quality", "performance"}:
        raise ValueError("Mode must be both, quality, or performance")
    perf_config = PerfConfig(max_concurrency)
    workers = min(QUALITY_WORKERS, max_concurrency)
    db = _connect()
    try:
        cursor = db.execute(
            "UPDATE test_runs SET status='running',started_at=?,workers=? WHERE id=? AND status IN ('pending','running')",
            (datetime.now(timezone.utc).isoformat(), workers, run_id),
        )
        db.commit()
    finally:
        db.close()
    if cursor.rowcount != 1:
        return
    validate_endpoint(model["base_url"])
    config = _build_client_config(model)
    questions = load_questions() if mode != "performance" else []
    db = _connect()
    try:
        db.execute(
            "UPDATE test_runs SET test_suite_hash=?,total_questions=?,quality_config_json=? WHERE id=? AND status='running'",
            (
                suite_hash(questions) if questions else None,
                len(questions),
                json.dumps(provenance()),
                run_id,
            ),
        )
        db.commit()
    finally:
        db.close()
    results, elapsed, message = [], 0.0, ""
    if questions:
        processed = 0

        def stored(index, result):
            nonlocal processed
            if not _run_is_active(run_id):
                return
            result.detail = _sanitize_error_detail(result.detail)
            _store_result(run_id, index + 1, result)
            processed += 1
            _update_progress(
                run_id,
                f"{result.question.category}: {result.question.id}",
                processed,
                len(questions),
                f"Recorded {processed}/{len(questions)} outcomes",
                "quality",
            )

        _update_progress(
            run_id, "", 0, len(questions), f"Loaded {len(questions)} rigorous questions", "quality"
        )
        results, elapsed, message = run_quality(
            questions,
            config,
            max_concurrency,
            on_result=stored,
            cancelled=lambda: not _run_is_active(run_id),
        )
        if not _run_is_active(run_id):
            return
        if not any(r.is_scored for r in results):
            message = message or f"No questions could be scored ({len(results)} outcomes)."
            if results:
                message += " " + results[0].detail
    perf_json = None
    if mode != "quality" and not message and _run_is_active(run_id):
        started = time.perf_counter()
        try:
            report = run_perf_suite(
                config,
                perf_config,
                progress=lambda phase, n, total: _update_progress(
                    run_id, phase, n, total, f"{phase}: {n}/{total}", "perf"
                ),
                cancelled=lambda: not _run_is_active(run_id),
            )
            perf_json = json.dumps(report.to_dict())
            if not any(p.requests > p.errors for p in report.concurrency):
                message = "No performance requests completed successfully. " + " ".join(
                    report.notes
                )
        except Exception as error:
            logger.exception("Performance measurement failed for run %d", run_id)
            message = f"Performance measurement failed: {error}"
        if not questions:
            elapsed = (time.perf_counter() - started) * 1000
    if _run_is_active(run_id):
        _summarise_and_finish(
            run_id, results, len(questions), workers, elapsed, perf_json, message, config
        )


def _summarise_and_finish(
    run_id, results, total, workers, duration_ms, perf_json, error_message, client_config=None
):
    done = [r for r in results if r is not None]
    scored = [r for r in done if r.is_scored]
    passed = sum(r.passed for r in scored)
    avg = summarize([result_record(r) for r in done])["category_balanced"] or 0.0
    if done:
        try:
            if client_config is None:
                db = _connect()
                try:
                    model = db.execute(
                        "SELECT m.model_id FROM test_runs r JOIN models m ON r.model_id=m.id WHERE r.id=?",
                        (run_id,),
                    ).fetchone()[0]
                finally:
                    db.close()
                client_config = ClientConfig(
                    "", "", model, max_tokens=MAX_OUTPUT_TOKENS, temperature=0, seed=0
                )
            report = make_quality_report(done, client_config, max_concurrency=workers)
            avg = report["summary"]["category_balanced"] or 0.0
            db = _connect()
            try:
                db.execute(
                    "UPDATE test_runs SET quality_json=? WHERE id=? AND status IN ('pending','running')",
                    (json.dumps(report), run_id),
                )
                db.commit()
            finally:
                db.close()
        except Exception as error:
            logger.exception("Quality report generation failed for run %d", run_id)
            error_message = f"{error_message} Quality report generation failed: {error}".strip()
    output_tokens = sum(r.tokens.completion_tokens for r in scored)
    _finish_run(
        run_id,
        total=total,
        scored=len(scored),
        passed=passed,
        avg=avg,
        weighted=avg,
        errors=sum(r.outcome in {"endpoint_error", "unsupported_context"} for r in done),
        duration_ms=duration_ms,
        workers=workers,
        latency=LatencyStats.from_samples([r.metrics.latency_ms for r in scored]),
        ttft=LatencyStats.from_samples(
            [r.metrics.ttft_ms for r in scored if r.metrics.ttft_ms is not None]
        ),
        throughput=output_tokens * 1000 / duration_ms if duration_ms and done else None,
        perf_json=perf_json,
        error_message=_sanitize_error_detail(error_message),
    )
