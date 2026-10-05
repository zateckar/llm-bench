"""One submission protocol for immediate, scheduled, edited and repeated runs."""

import json
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from app.database import fetch_all, get_db
from app.services import run_modes
from app.benchmarking.perf import DEFAULT_MAX_CONCURRENCY, MAX_CONCURRENCY, PerfConfig
from app.benchmarking.load_workload import (
    DEFAULT_IN_FLIGHT, MAX_IN_FLIGHT, PRESET_LABELS, PRESETS, default_settings, error_text, parse_settings,
)
from app.benchmarking import staged_performance
from app.benchmarking.quality_suite import QUALITY_WORKERS
from app.benchmarking.suites import DEFAULT_SUITE, RETIRED_SUITES, SUITE_NAMES, get_suite, suite_choices
from app.benchmarking.usecase_suites import is_key as is_usecase_key

MAX_REPEATS = 10
MAX_PLAN_RUNS = 50


def _integer(raw):
    if isinstance(raw, bool):
        raise ValueError("Benchmark settings must be integers")
    value = int(raw)
    if str(value) != str(raw).strip():
        raise ValueError("Benchmark settings must be integers")
    return value


def make_run_options(*, mode="both", performance=None, max_concurrency=DEFAULT_MAX_CONCURRENCY,
                     context_max=None, sweep_rounds=None, sweep_output_tokens=None,
                     suite=DEFAULT_SUITE, load=None, in_flight_cap=None):
    """Canonical options of a new run (docs/design-consolidation.md).

    ``mode`` is quality, performance or both. Quality runs store their suite;
    performance runs store the standard staged test with its stage settings:
    ``max_concurrency`` (latency stage, also capping quality workers),
    ``context_max`` (optional context-stage limit), ``load`` and
    ``in_flight_cap`` (capacity stage). Legacy performance kinds and modes are
    mapped to the standard test, keeping compatible settings; the legacy sweep
    settings ``sweep_rounds`` and ``sweep_output_tokens`` are accepted and ignored."""
    requested = performance
    if requested is None and mode in ("performance", "both"):
        requested = "standard"
    try:
        quality, kind = run_modes.normalise(mode, requested)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    mode = "both" if quality and kind else "quality" if quality else "performance"
    if not isinstance(suite, str) or not (suite in SUITE_NAMES or is_usecase_key(suite)):
        detail = ("This suite is now part of the standard quality suite" if suite in RETIRED_SUITES
                  else "Choose a known quality suite")
        raise HTTPException(status_code=422, detail=detail)
    if suite != DEFAULT_SUITE and not quality:
        raise HTTPException(status_code=422, detail="A quality suite applies only to quality runs")
    options = {"mode": mode}
    try:
        if kind == "load":
            # The legacy load test's concurrency was its in-flight cap.
            in_flight_cap = max_concurrency if in_flight_cap is None else in_flight_cap
            max_concurrency = DEFAULT_MAX_CONCURRENCY
        elif kind == "sweep" and _integer(max_concurrency) > MAX_CONCURRENCY:
            # A sweep's concurrency does not fit the latency stage.
            max_concurrency = DEFAULT_MAX_CONCURRENCY
        options["max_concurrency"] = _integer(max_concurrency)
        PerfConfig(options["max_concurrency"])
        if quality:
            options["suite"] = suite
        if kind:
            options["performance"] = "standard"
            cap = DEFAULT_IN_FLIGHT if in_flight_cap is None else _integer(in_flight_cap)
            if not 1 <= cap <= MAX_IN_FLIGHT:
                raise ValueError(f"The in-flight cap must be between 1 and {MAX_IN_FLIGHT}")
            options["in_flight_cap"] = cap
            if context_max is not None:
                options["context_max"] = staged_performance.context_limit(_integer(context_max))
            options["load"] = parse_settings(load if load is not None else default_settings()).model_dump()
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=error_text(error)) from error
    return options


def spec_defaults():
    return {"model_id": None, **make_run_options()}


def spec_from_run(run):
    """Reuse supported settings with the current suite and performance test.

    Runs of a retired built-in suite rerun the standard suite. A member of a
    repeat group yields the whole group's specification."""
    try:
        previous = json.loads(run.get("run_options_json") or "{}")
        suite = previous.get("suite") or DEFAULT_SUITE
        if suite in RETIRED_SUITES:
            suite = DEFAULT_SUITE
        options = make_run_options(
            mode=previous.get("mode", "both"),
            performance=previous.get("performance"),
            max_concurrency=previous.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
            context_max=previous.get("context_max"),
            suite=suite,
            load=previous.get("load"),
            in_flight_cap=previous.get("in_flight_cap"),
        )
    except (ValueError, TypeError, AttributeError, HTTPException):
        options = make_run_options()
    spec = {"model_id": run["model_id"], **options}
    repeats = run.get("repeat_count") if hasattr(run, "get") else None
    if isinstance(repeats, int) and repeats > 1:
        spec["repeats"] = repeats
    return spec


def specs_from_runs(runs):
    """Collapse each repeat group back into one specification, in run order."""
    specs, seen = [], set()
    for run in runs:
        group = run.get("repeat_group_id")
        if group is not None:
            if group in seen:
                continue
            seen.add(group)
        specs.append(spec_from_run(run))
    return specs


def _repeats(raw, index):
    if raw is None:
        return 1
    if isinstance(raw, bool) or not isinstance(raw, (int, str)) or not str(raw).strip().isdecimal():
        raise HTTPException(status_code=422, detail=f"Run {index}: repeats must be an integer")
    value = int(raw)
    if str(value) != str(raw).strip() or not 1 <= value <= MAX_REPEATS:
        raise HTTPException(status_code=422, detail=f"Run {index}: repeats must be between 1 and {MAX_REPEATS}")
    return value


def quality_config_for(options):
    suite = get_suite(options.get("suite") or DEFAULT_SUITE)
    return {**suite.provenance(), "name": suite.name}


PERFORMANCE_SETTINGS = {"context_max", "load", "in_flight_cap"}
LEGACY_SETTINGS = {"sweep_rounds", "sweep_output_tokens"}


async def validated_specs(raw: str) -> list[tuple[int, dict, dict, int]]:
    try:
        specs = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail="Invalid benchmark payload") from error
    if not isinstance(specs, list) or not 1 <= len(specs) <= MAX_PLAN_RUNS:
        raise HTTPException(status_code=422, detail=f"Submit between 1 and {MAX_PLAN_RUNS} runs")
    models = {model["id"] for model in await fetch_all("SELECT id FROM models")}
    prepared = []
    allowed = {"model_id", "mode", "performance", "max_concurrency", "suite", "repeats",
               *PERFORMANCE_SETTINGS, *LEGACY_SETTINGS}
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
        mode = spec.get("mode", "both")
        if mode == "quality" and set(spec) & (PERFORMANCE_SETTINGS | LEGACY_SETTINGS):
            raise HTTPException(status_code=422,
                                detail=f"Run {index}: performance settings require a performance run")
        options = make_run_options(
            mode=mode,
            performance=spec.get("performance"),
            max_concurrency=spec.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
            context_max=spec.get("context_max"),
            suite=spec.get("suite", DEFAULT_SUITE),
            load=spec.get("load"),
            in_flight_cap=spec.get("in_flight_cap"),
        )
        prepared.append((model_id, options, _repeats(spec.get("repeats"), index)))
    if sum(repeats for _, _, repeats in prepared) > MAX_PLAN_RUNS:
        raise HTTPException(status_code=422,
                            detail=f"Repeats expand to more than {MAX_PLAN_RUNS} runs")
    # Rows share one snapshot of each suite's current fixed protocol.
    validated = []
    for index, (model_id, options, repeats) in enumerate(prepared, 1):
        try:
            config = quality_config_for(options)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=f"Run {index}: {error}") from error
        validated.append((model_id, config, options, repeats))
    return validated


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
    from app.benchmarking import usecase_suites

    try:
        standard, suite_error = get_suite(DEFAULT_SUITE).provenance(), None
    except Exception as error:
        standard, suite_error = {"questions": 0, "areas": []}, str(error)
    question_count = standard["questions"]
    suites = [{**choice, "questions": question_count if choice["name"] == DEFAULT_SUITE else None}
              for choice in suite_choices()]
    suites.extend(usecase_suites.choice(row) for row in usecase_suites.latest_versions())
    # Edited plans keep a pinned older (or archived) version selectable.
    for spec in specs or []:
        key = spec.get("suite")
        if is_usecase_key(key) and key not in {s["name"] for s in suites}:
            row = usecase_suites.load_version(*usecase_suites.split_key(key))
            if row:
                suites.append(usecase_suites.choice(row))
    return {
        "models": models,
        "question_count": question_count,
        "standard_areas": standard["areas"],
        "stages": staged_performance.form_info(),
        "suite_error": suite_error,
        "suites": suites,
        "max_repeats": MAX_REPEATS,
        "spec_defaults": spec_defaults(),
        "prefill": specs,
        "max_concurrency": MAX_CONCURRENCY,
        "load_form": {
            "defaults": default_settings(),
            "presets": {key: {"label": PRESET_LABELS[key], "workload": workload} for key, workload in PRESETS.items()},
            "max_in_flight": MAX_IN_FLIGHT,
            "default_in_flight": DEFAULT_IN_FLIGHT,
        },
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
    for item in prepared:
        model_id, quality_config, options = item[:3]
        repeats = item[3] if len(item) > 3 else 1
        model_cursor = await db.execute(
            "SELECT temperature, reasoning_effort, b300_metrics_model FROM models WHERE id = ?", (model_id,)
        )
        model_settings = await model_cursor.fetchone()
        if model_settings is None:
            raise HTTPException(status_code=422, detail="Model no longer exists")
        metrics = metrics_scope({"b300_metrics_model": model_settings[2]})
        group_id = None
        for repeat_index in range(repeats):
            # Repeat i samples with seed i, so repeats measure the sampling
            # variability users see; repeat 0 equals an ordinary run.
            decoding = decoding_settings(*model_settings[:2], seed=repeat_index)
            cursor = await db.execute(
                """INSERT INTO test_runs
                   (model_id, status, created_by, workers, quality_config_json,
                    run_options_json, plan_id, created_at, decoding_config_json, metrics_config_json,
                    repeat_group_id, repeat_index, repeat_count)
                   VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    group_id,
                    repeat_index if repeats > 1 else None,
                    repeats if repeats > 1 else None,
                ),
            )
            run_id = cursor.lastrowid
            if repeats > 1 and group_id is None:
                group_id = run_id
                await db.execute("UPDATE test_runs SET repeat_group_id = ? WHERE id = ?",
                                 (group_id, run_id))
            run_ids.append(run_id)
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
