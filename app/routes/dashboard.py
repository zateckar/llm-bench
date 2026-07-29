"""Dashboard route."""

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
        """SELECT tr.*, m.name as model_name, m.model_id
           FROM test_runs tr
           JOIN models m ON tr.model_id = m.id
           WHERE tr.status IN ('running', 'pending')
           ORDER BY tr.id DESC"""
    )

    recent_runs = await fetch_all(
        """SELECT tr.*, m.name as model_name, m.model_id
           FROM test_runs tr
           JOIN models m ON tr.model_id = m.id
           ORDER BY tr.id DESC LIMIT 10"""
    )

    total_runs = (await fetch_one("SELECT COUNT(*) as cnt FROM test_runs"))["cnt"]
    completed_runs = (
        await fetch_one("SELECT COUNT(*) as cnt FROM test_runs WHERE status='completed'")
    )["cnt"]
    total_models = (await fetch_one("SELECT COUNT(*) as cnt FROM models"))["cnt"]

    last_run_categories = []
    if recent_runs and recent_runs[0]["status"] == "completed":
        last_run_id = recent_runs[0]["id"]
        last_run_categories = await fetch_all(
            """SELECT category,
                      COUNT(*) as total,
                      SUM(request_ok) as scored,
                      SUM(passed) as passed,
                      AVG(CASE WHEN request_ok = 1 THEN score END) as avg_score
               FROM test_results WHERE run_id = ?
               GROUP BY category ORDER BY category""",
            (last_run_id,),
        )

    return templates.TemplateResponse(
        request, "dashboard.html",
        {
            "active_runs": active_runs,
            "recent_runs": recent_runs,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "total_models": total_models,
            "last_run_categories": last_run_categories,
            "category_descriptions": CATEGORY_DESCRIPTIONS,
        },
    )
