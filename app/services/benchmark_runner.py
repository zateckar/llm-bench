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
from app.benchmarking.suites import get_suite
from app.storage import DETECT_TYPES, pack_text
from app.services.quality_views import projection_json

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 180.0


def start_benchmark(
    run_id, model, mode="both", max_concurrency=DEFAULT_MAX_CONCURRENCY, on_finish=None,
    **sweep_options,
):
    threading.Thread(
        target=_run_benchmark, args=(run_id, model, mode, max_concurrency, on_finish),
        kwargs=sweep_options, daemon=True
    ).start()


def _connect() -> sqlite3.Connection:
    from app.config import DATABASE_PATH

    db = sqlite3.connect(str(DATABASE_PATH), timeout=30, detect_types=DETECT_TYPES)
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
    record = result_record(result)
    db = _connect()
    try:
        db.execute(
            """INSERT INTO test_results
                   (run_id, test_id, category, prompt, response, score, detail, evaluator, prompt_preview,
                    question_index, prompt_tokens, completion_tokens, passed, pass_threshold,
                    difficulty, weight, latency_ms, ttft_ms, request_ok, quality_metadata_json, quality_scored, quality_outcome)
               SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                WHERE EXISTS (
                    SELECT 1 FROM test_runs
                     WHERE id = ? AND status IN ('pending', 'running')
                )""",
            (
                run_id,
                q.id,
                q.category,
                pack_text(q.prompt),
                pack_text(result.response),
                result.achievement_score,
                result.detail,
                q.evaluator,
                q.prompt[:80],
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
                pack_text(json.dumps(record, separators=(",", ":"))),
                int(result.is_scored),
                record["outcome"],
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
                      perf_json = COALESCE(?, perf_json)
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
                pack_text(perf_json),
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


def _sanitize_result_errors(result: Result) -> None:
    """Keep endpoint error strings out of saved recovery transcripts."""
    result.detail = _sanitize_error_detail(result.detail)
    if result.metrics.error:
        result.metrics.error = _sanitize_error_detail(result.metrics.error)
    if not result.metrics.ok:
        result.response = _sanitize_error_detail(result.response)

    def scrub(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"error", "detail"} and isinstance(item, str):
                    value[key] = _sanitize_error_detail(item)
                else:
                    scrub(item)
            if (isinstance(value.get("metrics"), dict)
                    and not value["metrics"].get("ok", True)
                    and isinstance(value.get("response"), str)):
                value["response"] = _sanitize_error_detail(value["response"])
        elif isinstance(value, list):
            for item in value:
                scrub(item)

    scrub(result.diagnostics)
    scrub(result.metrics.attempt_diagnostics)
    if result.evaluation:
        for criterion in result.evaluation.criteria:
            scrub(criterion.evidence)


def suite_provenance(suite_def) -> dict:
    """Provenance stored with a run; the rigorous shape stays historical."""
    if suite_def.name == "rigorous":
        return provenance()
    return {**suite_def.provenance(), "name": suite_def.name}


def _build_client_config(model: dict) -> ClientConfig:
    return ClientConfig(
        base_url=model["base_url"],
        api_key=model["api_key"],
        model=model["model_id"],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=model.get("temperature", 0),
        reasoning_effort=model.get("reasoning_effort"),
        # Repeat i of a group uses seed i; single runs and repeat 0 use seed 0.
        seed=model.get("seed", 0),
        timeout=REQUEST_TIMEOUT,
        stream_deadline=1800.0,
        stream=True,
    )


def _run_benchmark(run_id, model, mode, max_concurrency, on_finish: Callable[[int], None] | None, **sweep_options):
    workers = min(QUALITY_WORKERS, max_concurrency)
    try:
        _run_benchmark_impl(run_id, model, mode, max_concurrency, **sweep_options)
    except Exception as error:
        logger.exception("Benchmark run %d failed", run_id)
        _safe_mark_run_failed(run_id, str(error), workers)
    finally:
        if on_finish is not None:
            try:
                on_finish(run_id)
            except Exception:
                logger.exception("Queue callback failed for run %d", run_id)


def _run_benchmark_impl(run_id, model, mode, max_concurrency, suite=None, load=None, performance=None,
                        in_flight_cap=None, context_max=None, context_concurrency=None, target_ttft_ms=None,
                        target_output_rate=None, users=None, **sweep_options):
    from app.services.run_modes import normalise
    from app.services.url_guard import validate_endpoint

    # Quality runs first, then the performance test, into the same run. New
    # runs use the staged standard test; queued runs of a former test
    # ("fixed", "sweep", "load") still run that test.
    quality, kind = normalise(mode, performance)
    suite_def = get_suite(suite)
    perf_config = PerfConfig(max_concurrency) if kind in ("fixed", "standard") or kind is None else None
    if kind == "load":
        from app.benchmarking.load_workload import MAX_IN_FLIGHT, default_settings, parse_settings
        load_settings = parse_settings(load if load is not None else default_settings())
        # The former load test stored its in-flight cap as max_concurrency.
        in_flight_cap = max_concurrency
        if not 1 <= in_flight_cap <= MAX_IN_FLIGHT:
            raise ValueError(f"The in-flight cap must be between 1 and {MAX_IN_FLIGHT}")
    if kind == "standard":
        # Standard runs queued before v4 may carry the dropped capacity stage's
        # ``load`` and ``in_flight_cap``; they are ignored.
        from app.benchmarking import staged_performance as staged
        if context_max is not None:
            staged.context_limit(context_max)
        # Runs queued before the limit search use its defaults.
        from app.benchmarking import session_workload
        limits = (staged.context_concurrency(staged.DEFAULT_CONTEXT_CONCURRENCY if context_concurrency is None
                                             else context_concurrency),
                  staged.targets(target_ttft_ms, target_output_rate),
                  session_workload.parse_settings(users if users is not None
                                                  else session_workload.default_settings()))
    if kind == "sweep":
        from app.benchmarking.perf_sweep import MAX_CONTEXT, SweepConfig
        sweep_config = SweepConfig(max_concurrency=max_concurrency,
                                   context_max=MAX_CONTEXT if context_max is None else context_max,
                                   **sweep_options)
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
    if model.get("id") is not None:
        # Record what is being measured before the first measured request.
        from app.services import monitoring

        monitoring.capture_for_run(run_id, model)
    if not quality and kind == "sweep":
        _run_context_sweep(run_id, config, sweep_config, model.get("_metrics_scope"))
        return
    if not quality and kind == "load":
        _run_open_loop(run_id, config, load_settings, in_flight_cap, model.get("_metrics_scope"))
        return
    if not quality and kind == "standard":
        measured = _measure_staged(run_id, config, perf_config, context_max, model.get("_metrics_scope"), limits)
        if _run_is_active(run_id):
            perf_json, message, elapsed = measured
            _summarise_and_finish(run_id, [], 0, workers, elapsed, perf_json, message, config)
        return
    if not quality:
        questions = []
    elif suite_def.name == "rigorous":
        questions = load_questions()
    else:
        questions = suite_def.load()
    db = _connect()
    try:
        db.execute(
            "UPDATE test_runs SET test_suite_hash=?,total_questions=?,quality_config_json=? WHERE id=? AND status='running'",
            (
                suite_hash(questions) if questions else None,
                len(questions),
                json.dumps(suite_provenance(suite_def)),
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
            _sanitize_result_errors(result)
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
            run_id, "", 0, len(questions),
            f"Loaded {len(questions)} {'rigorous' if suite_def.name == 'rigorous' else suite_def.name} questions",
            "quality",
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
        if not any(r.is_scored or r.outcome == "recorded" for r in results):
            message = message or f"No questions could be scored ({len(results)} outcomes)."
            if results:
                message += " " + results[0].detail
    perf_json = None
    if kind in ("standard", "sweep", "load") and not message and _run_is_active(run_id):
        try:
            if kind == "standard":
                perf_json, message, _ = _measure_staged(run_id, config, perf_config, context_max,
                                                        model.get("_metrics_scope"), limits)
            elif kind == "sweep":
                perf_json, message, _ = _measure_context_sweep(run_id, config, sweep_config,
                                                               model.get("_metrics_scope"))
            else:
                perf_json, message, _ = _measure_open_loop(run_id, config, load_settings, in_flight_cap,
                                                           model.get("_metrics_scope"))
        except Exception as error:
            # Keep the quality results; the performance part failed.
            logger.exception("Performance measurement failed for run %d", run_id)
            message = f"Performance measurement failed: {error}"
    elif kind == "fixed" and not message and _run_is_active(run_id):
        telemetry = _begin_telemetry(model.get("_metrics_scope"), run_id)
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
            measurement_elapsed = (time.perf_counter() - started) * 1000
            if telemetry:
                report.telemetry = telemetry.finish()
            perf_json = json.dumps(report.to_dict())
            if not any(p.requests > p.errors for p in report.concurrency):
                message = "No performance requests completed successfully. " + " ".join(
                    report.notes
                )
        except Exception as error:
            if telemetry:
                telemetry.session.close()
            logger.exception("Performance measurement failed for run %d", run_id)
            message = f"Performance measurement failed: {error}"
        if not questions:
            elapsed = measurement_elapsed if perf_json else (time.perf_counter() - started) * 1000
    if _run_is_active(run_id):
        _summarise_and_finish(
            run_id, results, len(questions), workers, elapsed, perf_json, message, config,
            suite=suite_def,
        )


def _sanitize_sweep_errors(value):
    """Control probes and inferred-limit explanations also carry provider errors."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"error", "reason", "stop_error", "stop_reason"} and isinstance(item, str):
                value[key] = _sanitize_error_detail(item)
            elif key in {"notes", "error_examples", "user_failures"} and isinstance(item, list):
                value[key] = [_sanitize_error_detail(note) if isinstance(note, str) else note
                              for note in item]
            else:
                _sanitize_sweep_errors(item)
    elif isinstance(value, list):
        for item in value:
            _sanitize_sweep_errors(item)


def _store_sweep_checkpoint(run_id, report):
    _sanitize_sweep_errors(report)
    db = _connect()
    try:
        db.execute("UPDATE test_runs SET perf_json=? WHERE id=? AND status IN ('pending','running')",
                   (pack_text(json.dumps(report, separators=(",", ":"))), run_id))
        db.commit()
    finally:
        db.close()


def _store_sweep_cell(run_id, cell):
    _sanitize_sweep_errors(cell)
    db = _connect()
    try:
        db.execute("""INSERT OR REPLACE INTO performance_cells
                   (run_id,effort,context_tokens,concurrency,result_json)
                   SELECT ?,?,?,?,? WHERE EXISTS
                   (SELECT 1 FROM test_runs WHERE id=? AND status IN ('pending','running'))""",
                   (run_id, cell["effort"], cell["context_tokens"], cell["concurrency"], pack_text(json.dumps(cell, separators=(",", ":"))), run_id))
        db.commit()
    finally:
        db.close()


def _begin_telemetry(scope, run_id):
    if not scope:
        return None
    from app.benchmarking.vllm_telemetry import VllmTelemetry
    collector = VllmTelemetry(scope, cancelled=lambda: not _run_is_active(run_id))
    collector.begin()
    return collector


def _run_context_sweep(run_id, client_config, sweep_config, metrics_scope=None):
    measured = _measure_context_sweep(run_id, client_config, sweep_config, metrics_scope)
    if _run_is_active(run_id):
        perf_json, message, elapsed = measured
        _summarise_and_finish(run_id, [], 0, sweep_config.max_concurrency, elapsed, perf_json, message,
                              client_config)


def _measure_context_sweep(run_id, client_config, sweep_config, metrics_scope=None):
    """Run the sweep; returns ``(perf_json, error message, elapsed ms)``."""
    from app.benchmarking.perf_sweep import run_sweep

    telemetry = _begin_telemetry(metrics_scope, run_id)

    def checkpoint(data):
        if telemetry:
            data["telemetry"] = telemetry.data
        _store_sweep_checkpoint(run_id, data)

    started = time.perf_counter()
    try:
        report = run_sweep(
            client_config, sweep_config, retain_cells=False,
            on_cell=lambda cell: _store_sweep_cell(run_id, cell),
            checkpoint=checkpoint,
            progress=lambda label, n, total: _update_progress(run_id, label, n, total, f"Resolved {n:,}/{total:,} cells", "sweep"),
            cancelled=lambda: not _run_is_active(run_id),
        )
        measurement_elapsed = (time.perf_counter() - started) * 1000
        payload = report.to_dict()
        _sanitize_sweep_errors(payload)
        if telemetry:
            payload["telemetry"] = telemetry.finish()
    finally:
        if telemetry:
            telemetry.session.close()
    message = report.stop_error or ("" if report.successful_requests else "No complete answers were measured in the context sweep.")
    return json.dumps(payload), message, measurement_elapsed


def _run_open_loop(run_id, client_config, settings, in_flight_cap, metrics_scope=None):
    measured = _measure_open_loop(run_id, client_config, settings, in_flight_cap, metrics_scope)
    if _run_is_active(run_id):
        perf_json, message, elapsed = measured
        _summarise_and_finish(run_id, [], 0, in_flight_cap, elapsed, perf_json, message, client_config)


def _measure_open_loop(run_id, client_config, settings, in_flight_cap, metrics_scope=None):
    """Run the load test; returns ``(perf_json, error message, elapsed ms)``."""
    from app.benchmarking.load_test import run_load_test

    telemetry = _begin_telemetry(metrics_scope, run_id)
    started = time.perf_counter()
    try:
        report = run_load_test(
            client_config, settings, in_flight_cap,
            progress=lambda label, n, total: _update_progress(
                run_id, label, n, total, f"Dispatched {n:,}/{total:,} expected arrivals", "load"),
            cancelled=lambda: not _run_is_active(run_id),
            # Long ladders and soaks keep finished steps if the process stops.
            on_step=lambda report: _store_sweep_checkpoint(run_id, report.to_dict()),
        )
        measurement_elapsed = (time.perf_counter() - started) * 1000
        if telemetry:
            report.telemetry = telemetry.finish()
        payload = report.to_dict()
        _sanitize_sweep_errors(payload)
    finally:
        if telemetry:
            telemetry.session.close()
    completed = sum(step["completed"] for step in report.steps)
    message = "" if completed else (report.stop_reason or "No open-loop request completed.")
    return json.dumps(payload, separators=(",", ":")), message, measurement_elapsed


def _declared_context(run_id):
    """Context limit the deployment declared in this run's snapshot, or None."""
    db = _connect()
    try:
        row = db.execute("""SELECT snapshot_json FROM deployment_checks WHERE run_id=? AND ok=1
                            ORDER BY id DESC LIMIT 1""", (run_id,)).fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        db.close()
    try:
        snapshot = json.loads(row[0]) if row else {}
    except (TypeError, ValueError):
        return None
    served = ((snapshot if isinstance(snapshot, dict) else {}).get("hard") or {}).get("served") or {}
    value = served.get("max_model_len") if isinstance(served, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _measure_staged(run_id, client_config, perf_config, context_max=None, metrics_scope=None, limits=None):
    """Run the standard performance test's stages into one report.

    Returns ``(perf_json, error message, elapsed ms)``. The report is saved
    after every stage and while the context and users stages progress. A
    failing stage does not stop the next one unless nothing succeeded in it
    (docs/design-consolidation.md)."""
    from app.benchmarking import session_workload, staged_performance as staged
    from app.benchmarking.perf_sweep import SweepConfig, run_sweep
    from app.benchmarking.session_load import run_session_test

    limit, source = staged.context_limit_for(_declared_context(run_id), context_max)
    effort = staged.context_effort(client_config.reasoning_effort)
    concurrency, targets, user_settings = limits or (staged.DEFAULT_CONTEXT_CONCURRENCY, staged.targets(),
                                                     session_workload.parse_settings(
                                                         session_workload.default_settings()))
    sweep_config = SweepConfig(context_max=limit, max_concurrency=concurrency, sweep_rounds=1, min_samples=20,
                               sweep_output_tokens=staged.CONTEXT_OUTPUT_TOKENS,
                               context_lengths=staged.contexts_for(limit), effort_list=(effort,), targets=targets)
    report = staged.new_report(client_config.model, {
        "latency": {"levels": list(perf_config.levels), "max_concurrency": perf_config.max_concurrency},
        "context": {"contexts": list(sweep_config.contexts), "limit": limit, "limit_source": source,
                    "effort": effort, "concurrencies": list(sweep_config.concurrencies),
                    "minimum_samples_per_cell": sweep_config.min_samples,
                    "targets": targets.to_dict(), "max_output_tokens": staged.CONTEXT_OUTPUT_TOKENS},
        "users": {"preset": user_settings.preset, "user_model_hash": user_settings.model.fingerprint(),
                  "context_caps": list(session_workload.caps_for(user_settings, limit)),
                  "max_users": user_settings.max_users,
                  "saturation_max_users": user_settings.saturation_max_users if user_settings.saturation else None},
    })
    cancelled = lambda: not _run_is_active(run_id)  # noqa: E731

    def save():
        _store_sweep_checkpoint(run_id, report)

    def progress(stage, phase, unit):
        label = staged.STAGE_LABELS[stage]
        return lambda item, n, total: _update_progress(
            run_id, f"{label} · {item}", n, total, f"{label} stage · {unit} {n:,}/{total:,}", phase)

    def latency(telemetry):
        result = run_perf_suite(client_config, perf_config, progress=progress("latency", "perf", "requests"),
                                cancelled=cancelled)
        if telemetry:
            result.telemetry = telemetry.finish()
        ok = any(p.requests > p.errors for p in result.concurrency)
        return result.to_dict(), ok, "" if ok else " ".join(
            ["No latency-stage request completed successfully.", *result.notes])

    def context(telemetry):
        def checkpoint(data):
            if telemetry:
                data["telemetry"] = telemetry.data
            report["stages"]["context"] = data
            save()

        result = run_sweep(client_config, sweep_config, retain_cells=False,
                           on_cell=lambda cell: _store_sweep_cell(run_id, cell), checkpoint=checkpoint,
                           progress=progress("context", "sweep", "resolved cells"), cancelled=cancelled)
        data = result.to_dict()
        if telemetry:
            data["telemetry"] = telemetry.finish()
        ok = bool(result.successful_requests)
        return data, ok, result.stop_error or ("" if ok else "No complete answers were measured in the context stage.")

    def users(telemetry):
        def on_level(result):
            report["stages"]["users"] = result.to_dict()
            save()

        result = run_session_test(client_config, user_settings, limit,
                                  progress=progress("users", "users", "levels"),
                                  cancelled=cancelled, on_level=on_level)
        if telemetry:
            result.attach_telemetry(telemetry.finish())
        ok = any(level.get("all_completed") for cap in result.caps for level in cap["levels"])
        return result.to_dict(), ok, result.stop_reason or ("" if ok else "No users-stage request completed.")

    started = time.perf_counter()
    message = ""
    measures = {"latency": latency, "context": context, "users": users}
    for key in staged.RUN_STAGES:
        measure = measures[key]
        if cancelled():
            break
        report["running_stage"] = key
        save()
        telemetry = _begin_telemetry(metrics_scope, run_id)
        try:
            data, ok, error = measure(telemetry)
            report["stages"][key] = data
        except Exception as exc:
            # A crash in one stage says nothing about the endpoint; the next stage still runs.
            logger.exception("%s stage failed for run %d", staged.STAGE_LABELS[key], run_id)
            ok, error = None, f"{exc}"
        finally:
            if telemetry:
                telemetry.session.close()
        if error:
            error = f"{staged.STAGE_LABELS[key]} stage: {error}"
            report["stage_errors"][key] = error
            message = message or error
        if ok is False and not cancelled():
            # Nothing succeeded: the endpoint is down, so later stages would only fail too.
            report["notes"].append(f"Stopped after the {staged.STAGE_LABELS[key].lower()} stage: "
                                   "no request succeeded.")
            break
    report["cancelled"] = cancelled()
    report["finished"] = not report["cancelled"] and all(key in report["stages"] for key in staged.RUN_STAGES)
    report["running_stage"] = report["running_stage"] if report["cancelled"] else None
    _sanitize_sweep_errors(report)
    return json.dumps(report, separators=(",", ":")), message, (time.perf_counter() - started) * 1000


def _summarise_and_finish(
    run_id, results, total, workers, duration_ms, perf_json, error_message, client_config=None,
    suite=None,
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
            report = make_quality_report(done, client_config, max_concurrency=workers, suite=suite)
            avg = report["summary"]["category_balanced"] or 0.0
            db = _connect()
            try:
                db.execute(
                    "UPDATE test_runs SET quality_json=?,quality_summary_json=? WHERE id=? AND status IN ('pending','running')",
                    (pack_text(json.dumps(report, separators=(",", ":"))),
                     projection_json(report, [{"test_id": r.question.id, "response": r.response} for r in done]), run_id),
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
