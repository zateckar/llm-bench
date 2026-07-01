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
    "Truthfulness": "Tests whether the model admits uncertainty about fictional or impossible things vs. fabricating plausible-sounding answers.",
    "Reading Comprehension": "Tests ability to extract specific information from provided text passages.",
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
    "exact_match": "Response must exactly match the expected answer (normalized).",
    "contains_keywords": "Response must contain all specified keywords.",
    "numeric_match": "Response must contain the expected numeric value.",
    "code_exec": "Code is extracted from the response and executed against test cases.",
    "format_check": "Response is checked against formatting rules (JSON, word count, line count, etc.).",
    "admits_uncertainty": "Response must show uncertainty about fictional/impossible topics.",
    "security_analysis": "Response is checked for security-related patterns (criteria that should and shouldn't match).",
    "file_content_match": "Response must describe correct file content.",
    "command_correctness": "Response must contain correct commands matching specified patterns.",
    "multi_step_solution": "Response must contain all required steps in correct order.",
}


def load_tests_from_yaml() -> dict:
    """Load all tests from YAML files, grouped by category."""
    categories = {}
    if not TESTS_DIR.exists():
        return categories

    for yaml_file in sorted(TESTS_DIR.glob("*.yaml")):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, list):
                continue
            for item in data:
                cat = item.get("category", "Unknown")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(item)
        except Exception as e:
            logger.error("Error loading YAML test file %s: %s", yaml_file, e)
            continue

    return categories


@router.get("/tests")
async def tests_browser(request: Request, category: str = None):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    all_categories = load_tests_from_yaml()
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
        },
    )
