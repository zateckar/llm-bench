"""Test browser route."""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
import logging
import yaml

from app.auth import get_current_user
from app.config import TESTS_DIR
from app.templates_config import templates

router = APIRouter()
logger = logging.getLogger(__name__)

CATEGORY_DESCRIPTIONS = {
    "Factual Knowledge": "Tests compositional scientific and factual reasoning with explicit typed answers.",
    "Mathematical Reasoning": "Tests ability to solve arithmetic, word problems, algebra, and geometry. Evaluates numerical accuracy.",
    "Logical Reasoning": "Tests logical deduction, pattern recognition, syllogisms, and sequential reasoning.",
    "Code Generation": "Tests ability to write correct, working Python code from a specification. Responses are executed against test cases.",
    "Instruction Following": "Tests ability to follow precise formatting, word count, structure, and content constraints.",
    "Truthfulness": "Closed-world evidence questions distinguish supported, contradicted and unresolved claims without inventing facts.",
    "Reading Comprehension": "Multi-hop questions over a passage: arithmetic, unit conversion, entity chaining and scope judgement. Answers are never stated verbatim in the text.",
    "Tool Using": "Tests ability to plan correct tool/API usage sequences for multi-step tasks.",
    "Long Context Coherence": "Tests ability to maintain accuracy and coherence when processing long documents.",
    "Agentic Use Cases": "Tests multi-step planning, workflow reasoning, and agentic decision-making.",
    "Advanced Coding": "Tests complex coding: data structures, algorithms, debugging, optimization.",
    "Security": "Tests knowledge of security vulnerabilities (SQL injection, XSS, etc.) and secure coding practices.",
    "Needle Retrieval": "Tests ability to find specific facts buried in long contexts.",
    "Terminal Algorithms": "Tests understanding of algorithm implementation in terminal/shell contexts.",
    "Terminal Science": "Tests scientific knowledge applicable to terminal/computing contexts.",
    "Terminal System Admin": "Tests system administration knowledge and command-line skills.",
    "Terminal Debugging": "Tests debugging methodology and problem-solving skills.",
    "Terminal File Operations": "Tests knowledge of file system operations and manipulation.",
}

EVALUATOR_DESCRIPTIONS = {
    "interactive_state": "JSON actions are executed in a local simulation. Every final-state and authorization requirement must pass; excess calls are reported separately.",
    "code_exec": "Code is extracted and executed in a sandboxed subprocess against boundary and combination fixtures, including protected input-state checks. The submitted program is not rewritten. Values are compared structurally, so int/float and tuple/list differences are not counted as wrong.",
    "format_check": "Formatting and instruction constraints (JSON structure, word/line/sentence/paragraph counts, regexes, table shape, allowed vocabulary). Every check must pass.",
    "json_match": "The JSON in the response is deep-compared against an expected document, with exact or subset key matching.",
}


def load_tests_from_yaml() -> tuple[dict, str | None]:
    """Load the suite through the real loader, grouped by category.

    Going through ``test_loader`` rather than raw YAML means the browser shows the
    *effective* evaluator and expected value - items written with the ``criteria:``
    shorthand previously displayed a blank evaluator - and shows the same
    difficulty and pass threshold the runner will apply.
    """
    categories: dict[str, list[dict]] = {}
    if not TESTS_DIR.exists():
        return categories, f"Tests directory not found: {TESTS_DIR}"

    try:
        from app.benchmarking.quality_suite import load_questions

        questions = load_questions()
    except Exception as e:  # noqa: BLE001 - reported in the page instead of a 500
        logger.error("Could not load the test suite: %s", e)
        return categories, str(e)

    for q in questions:
        try:
            expected = yaml.safe_dump(
                {
                    "success": "All final-state and authorization gates must pass.",
                    "environment": q.interaction,
                }
                if q.interaction
                else q.expected,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            ).rstrip()
        except Exception:  # noqa: BLE001
            expected = repr(q.expected)
        categories.setdefault(q.category, []).append(
            {
                "id": q.id,
                "prompt": q.prompt,
                "system_prompt": q.system_prompt,
                "evaluator": "interactive_state" if q.interaction else q.evaluator,
                "expected": expected,
                "difficulty": q.difficulty,
                "weight": q.effective_weight,
                "pass_threshold": q.pass_threshold,
                "description": q.description,
                "source": q.source,
            }
        )
    return categories, None


@router.get("/tests")
async def tests_browser(request: Request, category: str = None):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    all_categories, suite_error = load_tests_from_yaml()
    selected_tests = None
    selected_category = None

    if category and category in all_categories:
        selected_tests = all_categories[category]
        selected_category = category

    return templates.TemplateResponse(
        request,
        "tests_browser.html",
        {
            "all_categories": all_categories,
            "selected_tests": selected_tests,
            "selected_category": selected_category,
            "category_descriptions": CATEGORY_DESCRIPTIONS,
            "evaluator_descriptions": EVALUATOR_DESCRIPTIONS,
            "suite_error": suite_error,
        },
    )
