"""Dashboard route."""

import json

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from app.auth import get_current_user
from app.database import fetch_all, fetch_one
from app.templates_config import templates

router = APIRouter()

CATEGORY_DESCRIPTIONS = {
    "Factual Knowledge": "Multi-hop and precise-value recall; naming half the answer earns nothing.",
    "Mathematical Reasoning": "Closed-form problems graded on the final asserted value only.",
    "Logical Reasoning": "Puzzles with a single verified answer; no guessable yes/no items.",
    "Code Generation": "Code is executed against fixtures including empty, duplicate and non-ASCII edge cases.",
    "Instruction Following": "Interacting formatting constraints, all of which must hold.",
    "Truthfulness": "Fabrication resistance, false-premise correction, and over-refusal controls.",
    "Reading Comprehension": "Multi-hop inference over a passage, never a verbatim lookup.",
    "Tool Using": "Tests ability to plan tool usage and API calls correctly.",
    "Long Context Coherence": "Tests ability to maintain coherence over long documents.",
    "Agentic Use Cases": "Tests multi-step planning and agentic workflow reasoning.",
    "Advanced Coding": "Tests complex coding tasks: data structures, algorithms, debugging.",
    "Security": "Tests knowledge of security vulnerabilities and secure coding practices.",
    "Needle Retrieval": "Tests ability to find specific facts buried in long contexts.",
    "Terminal Algorithms": "Tests understanding of algorithm implementation in terminal/shell contexts.",
    "Terminal Science": "Tests scientific knowledge applicable to terminal/computing contexts.",
    "Terminal System Admin": "Tests system administration knowledge and command-line skills.",
    "Terminal Debugging": "Tests debugging methodology and problem-solving skills.",
    "Terminal File Operations": "Tests knowledge of file system operations and manipulation.",
}


@router.get("/dashboard")
async def dashboard(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    # Active (running/pending) benchmarks
    active_runs = await fetch_all(
        """SELECT tr.*, m.name as model_name, m.model_id AS provider_model_id
           FROM test_runs tr
           JOIN models m ON tr.model_id = m.id
           WHERE tr.status IN ('running', 'pending')
           ORDER BY tr.id DESC"""
    )

    recent_runs = await fetch_all(
        """SELECT tr.*, m.name as model_name, m.model_id AS provider_model_id
           FROM test_runs tr
           JOIN models m ON tr.model_id = m.id
           ORDER BY tr.id DESC LIMIT 10"""
    )

    total_runs = (await fetch_one("SELECT COUNT(*) as cnt FROM test_runs"))["cnt"]
    completed_runs = (
        await fetch_one("SELECT COUNT(*) as cnt FROM test_runs WHERE status='completed'")
    )["cnt"]
    total_models = (await fetch_one("SELECT COUNT(*) as cnt FROM models"))["cnt"]

    # Select independently of the recent-run page: newer active runs or
    # completed performance-only runs must not hide existing quality scores.
    # Only rigorous-suite runs qualify; the category descriptions describe it.
    last_run = await fetch_one(
        """SELECT tr.id, tr.quality_json, m.name AS model_name
           FROM test_runs tr
           JOIN models m ON tr.model_id = m.id
           WHERE tr.status = 'completed'
             AND COALESCE(json_extract(CASE WHEN json_valid(tr.run_options_json)
                                            THEN tr.run_options_json END, '$.suite'),
                          'rigorous') = 'rigorous'
             AND EXISTS (SELECT 1 FROM test_results r
                         WHERE r.run_id = tr.id
                           AND COALESCE(r.quality_scored, r.request_ok) = 1)
           ORDER BY tr.id DESC LIMIT 1"""
    )
    last_run_categories = []
    if last_run:
        last_run_categories = await fetch_all(
            """SELECT category,
                      COUNT(*) as total,
                      SUM(COALESCE(quality_scored, request_ok)) as scored,
                      SUM(passed) as passed,
                      AVG(CASE WHEN COALESCE(quality_scored, request_ok) = 1 THEN score END) as avg_score
               FROM test_results WHERE run_id = ?
               GROUP BY category ORDER BY category""",
            (last_run["id"],),
        )

        try:
            quality = json.loads(last_run.get("quality_json") or "{}")
        except (TypeError, ValueError):
            quality = {}
        if quality and quality.get("schema_version") == 3:
            for category in last_run_categories:
                summary = quality["summary"]["categories"].get(category["category"])
                if summary:
                    category["avg_score"] = summary["score"]

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "active_runs": active_runs,
            "recent_runs": recent_runs,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "total_models": total_models,
            "last_run": last_run,
            "last_run_categories": last_run_categories,
            "category_descriptions": CATEGORY_DESCRIPTIONS,
        },
    )
