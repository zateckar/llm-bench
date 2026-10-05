"""Admin pages for uploaded use-case suites."""

from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response

from app.benchmarking import usecase_suites
from app.benchmarking.usecase_suites import UsecaseSuiteError, make_key, parse_suite
from app.database import execute, fetch_all, fetch_one, get_db
from app.routes.admin import _admin_required
from app.storage import pack_text
from app.templates_config import templates

router = APIRouter()
EXAMPLE = Path(usecase_suites.__file__).with_name("usecase_example.yaml")


async def _suite_rows():
    rows = await fetch_all(
        """SELECT u.slug, u.name, u.owner, u.description, u.archived, u.suite_hash,
                  u.question_count, u.version, u.created_at,
                  (SELECT COUNT(*) FROM usecase_suites v WHERE v.slug = u.slug) AS versions,
                  (SELECT COUNT(*) FROM test_runs tr
                    WHERE json_valid(tr.run_options_json)
                      AND json_extract(tr.run_options_json, '$.suite') LIKE 'usecase:' || u.slug || '@%') AS runs
             FROM usecase_suites u
            WHERE u.version = (SELECT MAX(version) FROM usecase_suites v WHERE v.slug = u.slug)
            ORDER BY u.archived, u.name, u.slug"""
    )
    return rows


async def _render(request, status_code=200, **context):
    return templates.TemplateResponse(
        request, "admin/suites.html",
        {"suites": await _suite_rows(), "limits": {"bytes": usecase_suites.MAX_BYTES,
                                                    "questions": usecase_suites.MAX_QUESTIONS},
         "allowed_evaluators": sorted(usecase_suites.ALLOWED_EVALUATORS), **context},
        status_code=status_code,
    )


@router.get("/admin/suites")
async def suites_page(request: Request):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user
    return await _render(request)


@router.post("/admin/suites")
async def upload_suite(
    request: Request,
    action: str = Form("check"),
    yaml_text: str = Form(""),
    file: UploadFile | None = File(None),
):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user
    text = yaml_text
    if file is not None and file.filename:
        raw = await file.read(usecase_suites.MAX_BYTES + 1)
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return await _render(request, 422, problems=["The file is not UTF-8 text"], yaml_text="")
    try:
        parsed = parse_suite(text)
    except UsecaseSuiteError as error:
        return await _render(request, 422, problems=error.problems, warnings=error.warnings,
                             yaml_text=text if len(text) <= 200_000 else "")
    latest = await fetch_one(
        "SELECT version, suite_hash, name, owner FROM usecase_suites WHERE slug = ? ORDER BY version DESC LIMIT 1",
        (parsed.slug,),
    )
    if latest and latest["suite_hash"] == parsed.suite_hash:
        return await _render(request, 422, yaml_text=text, warnings=parsed.warnings, problems=[
            f"Version {latest['version']} of {parsed.slug} already has exactly these questions."])
    version = (latest["version"] + 1) if latest else 1
    summary = {
        "slug": parsed.slug, "name": parsed.name, "version": version, "questions": len(parsed.questions),
        "suite_hash": parsed.suite_hash, "categories": sorted({q.category for q in parsed.questions}),
        "structured": sum(q.request is not None for q in parsed.questions),
        "replaces": latest,
    }
    if action != "save":
        return await _render(request, yaml_text=text, warnings=parsed.warnings, checked=summary)
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO usecase_suites
                   (slug, version, name, description, owner, source_yaml, suite_hash, question_count,
                    warnings_json, created_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (parsed.slug, version, parsed.name, parsed.description, parsed.owner, pack_text(text),
             parsed.suite_hash, len(parsed.questions), json.dumps(parsed.warnings), user["id"],
             datetime.now(timezone.utc).isoformat()),
        )
        # A new upload makes the suite selectable again.
        await db.execute("UPDATE usecase_suites SET archived = 0 WHERE slug = ?", (parsed.slug,))
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()
    return RedirectResponse(url=f"/admin/suites/{parsed.slug}?saved={version}", status_code=302)


@router.get("/admin/suites/example.yaml")
async def example_suite(request: Request):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user
    return Response(EXAMPLE.read_text(encoding="utf-8"), media_type="application/x-yaml; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="usecase-suite-example.yaml"'})


@router.get("/admin/suites/{slug}")
async def suite_page(request: Request, slug: str, version: int | None = None):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user
    versions = await fetch_all(
        """SELECT u.id, u.version, u.name, u.owner, u.description, u.suite_hash, u.question_count,
                  u.warnings_json, u.created_at, u.archived, us.username AS created_by_name,
                  (SELECT COUNT(*) FROM test_runs tr
                    WHERE json_valid(tr.run_options_json)
                      AND json_extract(tr.run_options_json, '$.suite') = 'usecase:' || u.slug || '@' || u.version) AS runs
             FROM usecase_suites u LEFT JOIN users us ON us.id = u.created_by
            WHERE u.slug = ? ORDER BY u.version DESC""",
        (slug,),
    )
    if not versions:
        raise HTTPException(status_code=404, detail="Suite not found")
    for row in versions:
        try:
            row["warnings"] = json.loads(row.pop("warnings_json") or "[]")
        except (TypeError, ValueError):
            row["warnings"] = []
    shown = next((v for v in versions if v["version"] == version), versions[0])
    row = await fetch_one("SELECT source_yaml FROM usecase_suites WHERE slug = ? AND version = ?",
                          (slug, shown["version"]))
    try:
        questions = parse_suite(row["source_yaml"]).questions
        parse_error = None
    except UsecaseSuiteError as error:
        questions, parse_error = [], str(error)
    categories = {}
    for q in questions:
        categories.setdefault(q.category, []).append(q)
    return templates.TemplateResponse(request, "admin/suite_detail.html", {
        "slug": slug, "versions": versions, "shown": shown, "categories": categories,
        "parse_error": parse_error, "saved": request.query_params.get("saved"),
        "run_key": make_key(slug, shown["version"]),
    })


@router.get("/admin/suites/{slug}/{version}.yaml")
async def download_suite(request: Request, slug: str, version: int):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user
    row = await fetch_one("SELECT source_yaml FROM usecase_suites WHERE slug = ? AND version = ?",
                          (slug, version))
    if not row:
        raise HTTPException(status_code=404, detail="Suite version not found")
    return Response(row["source_yaml"], media_type="application/x-yaml; charset=utf-8", headers={
        "Content-Disposition": f'attachment; filename="{slug}-v{version}.yaml"',
        "Cache-Control": "no-store",
    })


@router.post("/admin/suites/{slug}/archive")
async def archive_suite(request: Request, slug: str, archived: int = Form(1)):
    user = _admin_required(request)
    if isinstance(user, RedirectResponse):
        return user
    await execute("UPDATE usecase_suites SET archived = ? WHERE slug = ?", (1 if archived else 0, slug))
    return RedirectResponse(url=f"/admin/suites/{slug}", status_code=302)
