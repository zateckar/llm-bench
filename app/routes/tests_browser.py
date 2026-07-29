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
    "Factual Knowledge": "Tests basic world knowledge the model should confidently know. Covers geography, history, science, and general facts.",
    "Mathematical Reasoning": "Tests ability to solve arithmetic, word problems, algebra, and geometry. Evaluates numerical accuracy.",
    "Logical Reasoning": "Tests logical deduction, pattern recognition, syllogisms, and sequential reasoning.",
    "Code Generation": "Tests ability to write correct, working Python code from a specification. Responses are executed against test cases.",
    "Instruction Following": "Tests ability to follow precise formatting, word count, structure, and content constraints.",
    "Truthfulness": "Three item types: subjects that do not exist (the model must decline without fabricating specifics), questions with a false premise (it must reject the premise and supply the correct fact), and well-established facts (it must answer exactly, so always-hedging fails).",
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
    "exact_match": "The whole normalized response must equal the expected answer; extra prose fails.",
    "mcq": "A single option letter is extracted. Naming more than one option scores zero - hedging is not an answer.",
    "contains_keywords": "Every requirement must be met. A bare list means ALL keywords; `any`/`groups` express alternative spellings; `none` lists terms that must be absent.",
    "numeric_match": "Only the model's final asserted number counts, compared within the precision the prompt asks for. The right value appearing as an intermediate step earns nothing.",
    "numeric_set": "Every expected number must appear somewhere in the response, in any order.",
    "code_exec": "Code is extracted and executed in a sandboxed subprocess against fixtures. Values are compared structurally, so int/float and tuple/list differences are not counted as wrong.",
    "format_check": "Formatting and instruction constraints (JSON structure, word/line/sentence/paragraph counts, regexes, table shape, allowed vocabulary). Every check must pass.",
    "json_match": "The JSON in the response is deep-compared against an expected document, with exact or subset key matching.",
    "ordered_labels": "An enumerated answer is graded item by item: the label is read from the segment under each number, and wrong labels are explicitly rejected.",
    "set_match": "The response must name exactly the required items and none of the listed decoys.",
    "regex_all": "Every pattern must match, and any `must_not` pattern is an immediate failure.",
    "refusal_calibration": "The model must decline AND must not fabricate specifics; `forbidden` patterns catch hedging followed by an invented answer.",
    "admits_uncertainty": "Alias for refusal_calibration, kept so older runs still render.",
    "security_analysis": "Every required finding must be present (or `min_criteria` of them), and any disqualifying claim - declaring vulnerable code secure, or recommending a known-bad fix - scores zero.",
    "file_content_match": "Both the filename and the required content must be present.",
    "command_correctness": "The specific commands and flags must appear. Absent optional commands earn nothing, and `forbidden` entries fail the item.",
    "multi_step_solution": "All required steps must be present in the specified order. A step found out of order scores zero for that step.",
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
        from test_loader import load_all_tests

        questions = load_all_tests(TESTS_DIR)
    except Exception as e:  # noqa: BLE001 - reported in the page instead of a 500
        logger.error("Could not load the test suite: %s", e)
        return categories, str(e)

    for q in questions:
        try:
            expected = yaml.safe_dump(
                q.expected, sort_keys=False, allow_unicode=True, default_flow_style=False
            ).rstrip()
        except Exception:  # noqa: BLE001
            expected = repr(q.expected)
        categories.setdefault(q.category, []).append({
            "id": q.id,
            "prompt": q.prompt,
            "system_prompt": q.system_prompt,
            "evaluator": q.evaluator,
            "expected": expected,
            "difficulty": q.difficulty,
            "weight": q.effective_weight,
            "pass_threshold": q.pass_threshold,
            "description": q.description,
            "source": q.source,
        })
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
        request, "tests_browser.html",
        {
            "all_categories": all_categories,
            "selected_tests": selected_tests,
            "selected_category": selected_category,
            "category_descriptions": CATEGORY_DESCRIPTIONS,
            "evaluator_descriptions": EVALUATOR_DESCRIPTIONS,
            "suite_error": suite_error,
        },
    )
