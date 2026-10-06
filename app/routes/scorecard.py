"""Scorecards, decision profiles and decision records (docs/design-decision-dashboard.md)."""

from urllib.parse import quote

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.auth import get_current_user
from app.benchmarking import decision_gates
from app.services import scorecard
from app.services.scorecard import ScorecardError
from app.templates_config import templates

router = APIRouter()


async def _user(request):
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user


async def _admin(request):
    user = await _user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Administrators only")
    return user


def _back(url, error=None):
    return RedirectResponse(url=f"{url}?error={quote(str(error))}" if error else url, status_code=303)


async def _profile(profile_id):
    profile = await run_in_threadpool(scorecard.get_profile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


def _editor_options(collected):
    """Choices for the gate editor: scored suites, workloads, user models and categories with evidence."""
    from app.benchmarking import usecase_suites
    from app.benchmarking.load_workload import PRESET_LABELS, PRESETS
    from app.benchmarking.session_workload import PRESET_LABELS as USER_PRESET_LABELS, PRESETS as USER_PRESETS
    from app.benchmarking.suites import SUITE_NAMES

    suites = {key: scorecard.suite_label(key) for key in SUITE_NAMES}
    for row in usecase_suites.latest_versions():
        choice = usecase_suites.choice(row)
        suites[choice["name"]] = choice["label"]
    suites.update({s["key"]: s["label"] for s in collected["suites"]})
    workloads = {key: PRESET_LABELS[key] for key in PRESETS}
    workloads.update({w["key"]: w["label"] for w in collected["workloads"]})
    user_models = {key: USER_PRESET_LABELS[key] for key in USER_PRESETS}
    user_models.update({u["key"]: u["label"] for u in collected.get("user_models", [])})
    categories = {}
    for model in collected["models"]:
        for key, item in model["quality"].items():
            categories.setdefault(key, set()).update(item.get("categories") or {})
    return {
        "suites": [{"key": k, "label": v} for k, v in suites.items()],
        "workloads": [{"key": k, "label": v} for k, v in workloads.items()],
        "user_models": [{"key": k, "label": v} for k, v in user_models.items()],
        "categories": {k: sorted(v) for k, v in categories.items()},
        "latency_metrics": [{"key": k, "label": v[0], "unit": v[1], "lower": v[2]}
                            for k, v in decision_gates.LATENCY_METRICS.items()],
    }


@router.get("/scorecard")
async def scorecard_page(request: Request):
    user = await _user(request)
    collected = await run_in_threadpool(scorecard.collect)
    profiles = await run_in_threadpool(scorecard.profile_overview, collected)
    return templates.TemplateResponse(request, "scorecard.html", {
        "data": collected, "profiles": profiles, "is_admin": user["role"] == "admin",
        "old_days": scorecard.OLD_DAYS,
    })


@router.get("/scorecard/ties")
async def scorecard_ties(request: Request, suite: str):
    await _user(request)
    collected = await run_in_threadpool(scorecard.collect)
    marks = await run_in_threadpool(scorecard.ties, collected, suite)
    return JSONResponse({str(k): v for k, v in marks.items()})


@router.get("/scorecard.json")
async def scorecard_export(request: Request):
    await _user(request)

    def build():
        collected = scorecard.collect()
        profiles = []
        for profile in scorecard.list_profiles():
            profiles.append({"id": profile["id"], "name": profile["name"], "gates": profile["gates"],
                             "requirements": scorecard.gate_labels(profile, collected),
                             "models": [{"model": r["model"], "evaluation": r["evaluation"]}
                                        for r in scorecard.evaluate_profile(profile, collected)]})
        ties = {s["key"]: {str(k): v for k, v in scorecard.ties(collected, s["key"]).items()}
                for s in collected["suites"]}
        return scorecard.public({**collected, "ties": ties, "profiles": profiles,
                                 "gate_revision": decision_gates.REVISION})

    return JSONResponse(await run_in_threadpool(build))


@router.get("/scorecard/models/{model_id}")
async def model_scorecard(request: Request, model_id: int):
    user = await _user(request)
    collected = await run_in_threadpool(scorecard.collect)
    evidence = scorecard.model_evidence(collected, model_id)
    if evidence is None:
        raise HTTPException(status_code=404, detail="Model not found")
    profiles = []
    for profile in await run_in_threadpool(scorecard.list_profiles):
        profiles.append({**profile, "requirements": scorecard.gate_labels(profile, collected),
                         "evaluation": decision_gates.evaluate(profile["gates"], evidence)})
    return templates.TemplateResponse(request, "scorecard_model.html", {
        "data": collected, "e": evidence, "profiles": profiles,
        "studies": await run_in_threadpool(scorecard.studies_for, model_id),
        "decisions": await run_in_threadpool(lambda: scorecard.decisions(model_id=model_id)),
        "is_admin": user["role"] == "admin", "old_days": scorecard.OLD_DAYS,
    })


@router.get("/scorecard/profiles")
async def profiles_page(request: Request, error: str | None = None):
    user = await _user(request)
    collected = await run_in_threadpool(scorecard.collect)
    is_admin = user["role"] == "admin"
    return templates.TemplateResponse(request, "scorecard_profiles.html", {
        "profiles": await run_in_threadpool(scorecard.profile_overview, collected),
        "options": await run_in_threadpool(_editor_options, collected) if is_admin else None,
        "is_admin": is_admin, "error": error,
    })


@router.post("/scorecard/profiles")
async def create_profile(request: Request, name: str = Form(""), description: str = Form(""),
                         gates: str = Form("[]")):
    user = await _admin(request)
    try:
        profile_id = await run_in_threadpool(lambda: scorecard.create_profile(
            user["id"], name=name, description=description, gates=gates))
    except ScorecardError as error:
        return _back("/scorecard/profiles", error)
    return RedirectResponse(url=f"/scorecard/profiles/{profile_id}", status_code=303)


@router.get("/scorecard/profiles/{profile_id}")
async def profile_page(request: Request, profile_id: int, error: str | None = None):
    user = await _user(request)
    profile = await _profile(profile_id)
    collected = await run_in_threadpool(scorecard.collect)
    is_admin = user["role"] == "admin"
    return templates.TemplateResponse(request, "scorecard_profile.html", {
        "profile": profile, "requirements": scorecard.gate_labels(profile, collected),
        "rows": scorecard.evaluate_profile(profile, collected),
        "decisions": await run_in_threadpool(lambda: scorecard.decisions(profile_id=profile_id)),
        "options": await run_in_threadpool(_editor_options, collected) if is_admin else None,
        "decision_choices": scorecard.DECISIONS, "is_admin": is_admin, "error": error,
        "old_days": scorecard.OLD_DAYS,
    })


@router.get("/scorecard/profiles/{profile_id}/export.json")
async def profile_export(request: Request, profile_id: int):
    await _user(request)
    profile = await _profile(profile_id)

    def build():
        collected = scorecard.collect()
        return scorecard.public({
            "kind": "decision_profile", "revision": decision_gates.REVISION,
            "generated_at": collected["generated_at"],
            "profile": {k: profile[k] for k in ("id", "name", "description", "gates", "created_at", "updated_at")},
            "requirements": scorecard.gate_labels(profile, collected),
            "models": [{"model": r["model"], "fingerprint": r["fingerprint"], "evaluation": r["evaluation"]}
                       for r in scorecard.evaluate_profile(profile, collected)],
            "decisions": [{k: d[k] for k in ("id", "model_id", "model_name", "decision", "note", "revision",
                                             "fingerprint", "drifted", "superseded", "created_by_name",
                                             "created_at", "evaluation")}
                          for d in scorecard.decisions(profile_id=profile_id)],
        })

    return JSONResponse(await run_in_threadpool(build))


@router.post("/scorecard/profiles/{profile_id}/edit")
async def edit_profile(request: Request, profile_id: int, name: str = Form(""), description: str = Form(""),
                       gates: str = Form("[]")):
    await _admin(request)
    await _profile(profile_id)
    try:
        await run_in_threadpool(lambda: scorecard.update_profile(
            profile_id, name=name, description=description, gates=gates))
    except ScorecardError as error:
        return _back(f"/scorecard/profiles/{profile_id}", error)
    return _back(f"/scorecard/profiles/{profile_id}")


@router.post("/scorecard/profiles/{profile_id}/delete")
async def delete_profile(request: Request, profile_id: int):
    await _admin(request)
    await _profile(profile_id)
    await run_in_threadpool(scorecard.delete_profile, profile_id)
    return _back("/scorecard/profiles")


@router.post("/scorecard/profiles/{profile_id}/decisions")
async def record_decision(request: Request, profile_id: int, model_id: int = Form(...),
                          decision: str = Form(""), note: str = Form("")):
    user = await _admin(request)
    await _profile(profile_id)
    try:
        await run_in_threadpool(lambda: scorecard.record_decision(
            profile_id, model_id, decision=decision, note=note, user_id=user["id"]))
    except ScorecardError as error:
        return _back(f"/scorecard/profiles/{profile_id}", error)
    return RedirectResponse(url=f"/scorecard/profiles/{profile_id}#decisions", status_code=303)
