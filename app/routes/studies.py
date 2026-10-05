"""Blind A/B studies: results for everyone, blind voting, admin management."""

import json
from urllib.parse import quote

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from app.auth import get_current_user
from app.benchmarking import pairwise_judge
from app.database import fetch_all, fetch_one
from app.services import ab_studies
from app.services.ab_studies import StudyError
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


async def _study(study_id):
    study = await run_in_threadpool(ab_studies.get_study, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found")
    return study


async def _list_page(request, user, status_code=200, **context):
    is_admin = user["role"] == "admin"
    runs = await run_in_threadpool(ab_studies.candidate_runs) if is_admin else []
    return templates.TemplateResponse(request, "studies.html", {
        "studies": await run_in_threadpool(ab_studies.list_studies), "runs": runs, "is_admin": is_admin,
        "scopes": ab_studies.SCOPES, "max_pairs": ab_studies.MAX_PAIRS, **context,
    }, status_code=status_code)


@router.get("/studies")
async def studies_page(request: Request, run_a: int | None = None, run_b: int | None = None):
    user = await _user(request)
    return await _list_page(request, user, form={"run_a": run_a, "run_b": run_b, "scope": "open_ended", "name": ""})


@router.post("/studies")
async def create_study(request: Request, name: str = Form(""), run_a: int = Form(...), run_b: int = Form(...),
                       scope: str = Form("open_ended")):
    user = await _admin(request)
    try:
        study_id = await run_in_threadpool(ab_studies.create_study, name, run_a, run_b, scope, user["id"])
    except StudyError as error:
        return await _list_page(request, user, 422, error=str(error),
                                form={"run_a": run_a, "run_b": run_b, "scope": scope, "name": name})
    return RedirectResponse(url=f"/studies/{study_id}", status_code=303)


@router.get("/studies/{study_id}")
async def study_page(request: Request, study_id: int, error: str | None = None):
    user = await _user(request)
    study = await _study(study_id)
    pairs = await run_in_threadpool(ab_studies.study_pairs, study_id)
    summary = await run_in_threadpool(ab_studies.results, study_id, pairs)
    voted, total = await run_in_threadpool(ab_studies.vote_progress, study_id, user["id"])
    is_admin = user["role"] == "admin"
    models, self_judge = [], []
    if is_admin:
        models = await fetch_all("SELECT id, name, model_id FROM models ORDER BY name")
        row_ids, identifiers = await run_in_threadpool(ab_studies.contestant_models, study)
        self_judge = [m["id"] for m in models if m["id"] in row_ids or m["model_id"] in identifiers]
    return templates.TemplateResponse(request, "study_detail.html", {
        "study": study, "pairs": pairs, "summary": summary, "is_admin": is_admin, "models": models,
        "self_judge": self_judge, "voted": voted, "total": total, "error": error,
        "running": any(j["status"] == "running" for j in summary["judges"]),
        "scopes": ab_studies.SCOPES, "judge_revision": pairwise_judge.REVISION,
    })


@router.get("/studies/{study_id}/export.json")
async def study_export(request: Request, study_id: int):
    await _user(request)
    await _study(study_id)
    data = await run_in_threadpool(ab_studies.export, study_id)
    return Response(json.dumps(data, ensure_ascii=False, indent=1), media_type="application/json", headers={
        "Content-Disposition": f'attachment; filename="ab-study-{study_id}.json"', "Cache-Control": "no-store"})


@router.get("/studies/{study_id}/pairs/{pair_id}")
async def pair_page(request: Request, study_id: int, pair_id: int):
    await _user(request)
    study = await _study(study_id)
    pair = await run_in_threadpool(ab_studies.get_pair, study_id, pair_id)
    if pair is None:
        raise HTTPException(status_code=404, detail="Pair not found")
    summary = await run_in_threadpool(ab_studies.results, study_id, [pair])
    pair_votes = [v for v in await run_in_threadpool(ab_studies.votes, study_id) if v["pair_id"] == pair_id]
    return templates.TemplateResponse(request, "study_pair.html", {
        "study": study, "pair": pair, "summary": summary, "votes": pair_votes})


@router.post("/studies/{study_id}/status")
async def study_status(request: Request, study_id: int, status: str = Form(...)):
    await _admin(request)
    await _study(study_id)
    try:
        await run_in_threadpool(ab_studies.set_status, study_id, status)
    except StudyError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return RedirectResponse(url=f"/studies/{study_id}", status_code=303)


@router.post("/studies/{study_id}/delete")
async def study_delete(request: Request, study_id: int):
    await _admin(request)
    await _study(study_id)
    try:
        await run_in_threadpool(ab_studies.delete_study, study_id)
    except StudyError as error:
        return RedirectResponse(url=f"/studies/{study_id}?error={quote(str(error))}", status_code=303)
    return RedirectResponse(url="/studies", status_code=303)


@router.post("/studies/{study_id}/judges")
async def start_judge(request: Request, study_id: int, model_id: int = Form(...)):
    await _admin(request)
    await _study(study_id)
    model = await fetch_one("SELECT * FROM models WHERE id = ?", (model_id,))
    if model is None:
        raise HTTPException(status_code=422, detail="Unknown judge model")
    try:
        await run_in_threadpool(ab_studies.start_judge, study_id, model)
    except (StudyError, ValueError) as error:
        return RedirectResponse(url=f"/studies/{study_id}?error={quote(str(error))}", status_code=303)
    return RedirectResponse(url=f"/studies/{study_id}", status_code=303)


@router.post("/studies/{study_id}/judges/{judge_id}/cancel")
async def cancel_judge(request: Request, study_id: int, judge_id: int):
    await _admin(request)
    await run_in_threadpool(ab_studies.cancel_judge, study_id, judge_id)
    return RedirectResponse(url=f"/studies/{study_id}", status_code=303)


@router.get("/studies/{study_id}/vote")
async def vote_page(request: Request, study_id: int, skip: int | None = None, error: str | None = None):
    user = await _user(request)
    study = await _study(study_id)
    pair, shown_left = (None, None)
    if study["status"] == "open":
        pair, shown_left = await run_in_threadpool(ab_studies.next_pair, study_id, user["id"], skip)
    voted, total = await run_in_threadpool(ab_studies.vote_progress, study_id, user["id"])
    left = right = None
    if pair:
        left, right = ((pair["answer_a"], pair["answer_b"]) if shown_left == "a"
                       else (pair["answer_b"], pair["answer_a"]))
    return templates.TemplateResponse(request, "study_vote.html", {
        "study": study, "pair": pair, "shown_left": shown_left, "left": left, "right": right,
        "voted": voted, "total": total, "error": error, "comment_limit": ab_studies.COMMENT_LIMIT,
    }, headers={"Cache-Control": "no-store"})


@router.post("/studies/{study_id}/vote")
async def submit_vote(request: Request, study_id: int, pair_id: int = Form(...), choice: str = Form(...),
                      shown_left: str = Form(...), comment: str = Form("")):
    user = await _user(request)
    await _study(study_id)
    try:
        await run_in_threadpool(ab_studies.record_vote, study_id, pair_id, user["id"], choice, shown_left, comment)
    except StudyError as error:
        return RedirectResponse(url=f"/studies/{study_id}/vote?error={quote(str(error))}", status_code=303)
    return RedirectResponse(url=f"/studies/{study_id}/vote", status_code=303)
