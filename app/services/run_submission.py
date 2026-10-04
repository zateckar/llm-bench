"""One submission protocol for immediate, scheduled, edited and repeated runs."""

import json
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from app.database import fetch_all, get_db
from app.benchmarking.perf import DEFAULT_MAX_CONCURRENCY, MAX_CONCURRENCY, PerfConfig
from app.benchmarking.perf_sweep import SweepConfig
from app.benchmarking.quality_suite import QUALITY_WORKERS, load_questions, provenance


def make_run_options(*, mode="both", max_concurrency=DEFAULT_MAX_CONCURRENCY,
                     context_max=1_048_576, sweep_rounds=1, sweep_output_tokens=8192):
    if not isinstance(mode, str) or mode not in {"both", "quality", "performance", "sweep"}:
        raise HTTPException(status_code=422, detail="Choose quality, performance, both, or a context sweep")

    def integer(raw):
        if isinstance(raw, bool):
            raise ValueError("Sweep settings must be integers")
        value = int(raw)
        if str(value) != str(raw).strip():
            raise ValueError("Benchmark settings must be integers")
        return value

    try:
        value = integer(max_concurrency)
        if mode == "sweep":
            config = SweepConfig(integer(context_max), value, integer(sweep_rounds), integer(sweep_output_tokens))
            return {"mode": mode, "max_concurrency": value, "context_max": config.context_max,
                    "sweep_rounds": config.sweep_rounds, "sweep_output_tokens": config.sweep_output_tokens}
        PerfConfig(value)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"mode": mode, "max_concurrency": value}


def spec_defaults():
    return {"model_id": None, **make_run_options()}


def spec_from_run(run):
    """Reuse supported settings; historical runs always use the current suite."""
    try:
        previous = json.loads(run.get("run_options_json") or "{}")
        options = make_run_options(
            mode=previous.get("mode", "both"),
            max_concurrency=previous.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
            context_max=previous.get("context_max", 1_048_576),
            sweep_rounds=previous.get("sweep_rounds", 1),
            sweep_output_tokens=previous.get("sweep_output_tokens", 8192),
        )
    except (ValueError, TypeError, AttributeError, HTTPException):
        options = make_run_options()
    return {"model_id": run["model_id"], **options}


async def validated_specs(raw: str) -> list[tuple[int, dict, dict]]:
    try:
        specs = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail="Invalid benchmark payload") from error
    if not isinstance(specs, list) or not 1 <= len(specs) <= 50:
        raise HTTPException(status_code=422, detail="Submit between 1 and 50 runs")
    models = {model["id"] for model in await fetch_all("SELECT id FROM models")}
    prepared = []
    allowed = set(spec_defaults()) | {"context_max", "sweep_rounds", "sweep_output_tokens"}
    for index, spec in enumerate(specs, 1):
        if not isinstance(spec, dict):
            raise HTTPException(status_code=422, detail=f"Run {index} is malformed")
        model_id = spec.get("model_id")
        if isinstance(model_id, str):
            try:
                model_id = int(model_id)
            except ValueError:
                model_id = None
        if isinstance(model_id, bool) or not isinstance(model_id, int) or model_id not in models:
            raise HTTPException(status_code=422, detail=f"Run {index} needs a configured model")
        unknown = set(spec) - allowed
        if unknown:
            raise HTTPException(status_code=422, detail=f"Run {index} has unsupported settings")
        if spec.get("mode", "both") != "sweep" and set(spec) & {"context_max", "sweep_rounds", "sweep_output_tokens"}:
            raise HTTPException(status_code=422, detail=f"Run {index}: context sweep settings require sweep mode")
        options = make_run_options(
            mode=spec.get("mode", "both"),
            max_concurrency=spec.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
            context_max=spec.get("context_max", 1_048_576),
            sweep_rounds=spec.get("sweep_rounds", 1),
            sweep_output_tokens=spec.get("sweep_output_tokens", 8192),
        )
        prepared.append((model_id, options))
    # All rows share a single snapshot of the current fixed benchmark protocol.
    quality_config = provenance()
    return [(model_id, quality_config, options) for model_id, options in prepared]


def parse_browser_local(raw: str, tz_offset_min: str) -> str | None:
    """Convert the selected local time using its browser timezone offset."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        naive = datetime.strptime(raw, "%Y-%m-%dT%H:%M")
        offset = int(tz_offset_min)
        if not -840 <= offset <= 840:
            raise ValueError("Timezone offset is out of range")
        return (naive + timedelta(minutes=offset)).replace(tzinfo=timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError) as error:
        raise HTTPException(status_code=422, detail="Invalid schedule time or timezone") from error


def form_context(
    models,
    *,
    specs=None,
    name="",
    scheduled=None,
    action="/admin/run",
    heading="Run benchmark",
    submit_label="Run benchmark",
):
    try:
        question_count, suite_error = len(load_questions()), None
    except Exception as error:
        question_count, suite_error = 0, str(error)
    return {
        "models": models,
        "question_count": question_count,
        "suite_error": suite_error,
        "spec_defaults": spec_defaults(),
        "prefill": specs,
        "max_concurrency": MAX_CONCURRENCY,
        "submission_name": name,
        "scheduled_utc": scheduled,
        "action": action,
        "heading": heading,
        "submit_label": submit_label,
    }


async def _insert_runs(db, prepared, user_id, plan_id):
    from app.services.model_settings import decoding_settings
    from app.benchmarking.vllm_telemetry import metrics_scope

    run_ids = []
    created_at = datetime.now(timezone.utc).isoformat()
    for model_id, quality_config, options in prepared:
        model_cursor = await db.execute(
            "SELECT temperature, reasoning_effort, b300_metrics_model FROM models WHERE id = ?", (model_id,)
        )
        model_settings = await model_cursor.fetchone()
        if model_settings is None:
            raise HTTPException(status_code=422, detail="Model no longer exists")
        decoding = decoding_settings(*model_settings[:2])
        metrics = metrics_scope({"b300_metrics_model": model_settings[2]})
        cursor = await db.execute(
            """INSERT INTO test_runs
               (model_id, status, created_by, workers, quality_config_json,
                run_options_json, plan_id, created_at, decoding_config_json, metrics_config_json)
               VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_id,
                user_id,
                min(QUALITY_WORKERS, options["max_concurrency"]),
                json.dumps(quality_config),
                json.dumps(options),
                plan_id,
                created_at,
                json.dumps(decoding),
                json.dumps(metrics),
            ),
        )
        run_ids.append(cursor.lastrowid)
    return run_ids


async def submit_runs(name, user_id, scheduled, prepared):
    """Persist an ordered submission atomically, then wake the common dispatcher."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO run_plans (name, created_by, scheduled_at) VALUES (?, ?, ?)",
            ((name or "").strip() or None, user_id, scheduled),
        )
        plan_id = cursor.lastrowid
        run_ids = await _insert_runs(db, prepared, user_id, plan_id)
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()
    from app.services import run_queue

    run_queue.dispatch_next()
    return plan_id, run_ids


async def replace_runs(plan_id, name, user_id, scheduled, prepared):
    """Replace queued runs atomically and retain results of runs already started."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE run_plans SET name = ?, scheduled_at = ?, status = 'active' WHERE id = ?",
            ((name or "").strip() or None, scheduled, plan_id),
        )
        await db.execute(
            "UPDATE test_runs SET plan_id = NULL WHERE plan_id = ? AND status != 'pending'",
            (plan_id,),
        )
        await db.execute(
            "DELETE FROM test_runs WHERE plan_id = ? AND status = 'pending'", (plan_id,)
        )
        run_ids = await _insert_runs(db, prepared, user_id, plan_id)
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()
    from app.services import run_queue

    run_queue.dispatch_next()
    return run_ids
