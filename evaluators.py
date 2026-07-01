"""Evaluator functions for LLM benchmark responses.

Each evaluator takes a response string and expected value, returning (score, detail).
Score is 0.0 to 1.0. Detail is a human-readable explanation.
"""

import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


def safe_re_search(pattern: str, string: str, flags: int = 0) -> re.Match | None:
    """re.search with error handling for invalid patterns."""
    try:
        return re.search(pattern, string, flags)
    except re.error as e:
        logger.warning("Invalid regex pattern %r: %s", pattern, e)
        return None

# Code execution is delegated to an out-of-process sandbox (see code_runner.py).
# The builtins/import restrictions live there so that untrusted, model-generated
# code never runs in this process.

TEST_HELPERS = {
    "test_lru": """
def test_lru():
    try:
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        if cache.get(1) != 1: return False
        cache.put(3, 3)
        if cache.get(2) != -1: return False
        cache.put(4, 4)
        if cache.get(1) != -1: return False
        if cache.get(3) != 3: return False
        if cache.get(4) != 4: return False
        return True
    except Exception:
        return False
""",
    "test_serialize": """
def test_serialize():
    try:
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.right.left = TreeNode(4)
        root.right.right = TreeNode(5)
        
        serialized = serialize_tree(root)
        deserialized = deserialize_tree(serialized)
        
        def is_same(p, q):
            if not p and not q: return True
            if not p or not q: return False
            return p.val == q.val and is_same(p.left, q.left) and is_same(p.right, q.right)
            
        return is_same(root, deserialized)
    except Exception:
        return False
""",
    "test_bst": """
def test_bst():
    try:
        # Valid BST
        root1 = TreeNode(2)
        root1.left = TreeNode(1)
        root1.right = TreeNode(3)
        
        # Invalid BST
        root2 = TreeNode(5)
        root2.left = TreeNode(1)
        root2.right = TreeNode(4)
        root2.right.left = TreeNode(3)
        root2.right.right = TreeNode(6)
        
        return is_valid_bst(root1) == True and is_valid_bst(root2) == False
    except Exception:
        return False
""",
    "test_ars": """
def test_ars():
    try:
        import inspect
        if 'ars_sample' in globals():
            func = globals()['ars_sample']
        elif 'ars' in globals():
            func = globals()['ars']
        else:
            return False
            
        sig = inspect.signature(func)
        if len(sig.parameters) == 0:
            res = func()
            return res is not None
        else:
            f = lambda x: -0.5 * x**2
            f_prime = lambda x: -x
            res = func(f, f_prime, 5, (-2, 2))
            return res is not None
    except Exception:
        return False
""",
    "test_escape": """
def test_escape():
    try:
        import inspect
        if 'find_escape_path' in globals():
            func = globals()['find_escape_path']
        else:
            return False
            
        sig = inspect.signature(func)
        if len(sig.parameters) == 0:
            res = func()
            return res is not None
        else:
            res = func(5, [(1,1)], [(2,2)])
            return res is not None or isinstance(res, list)
    except Exception:
        return False
""",
    "test_scheduling": """
def test_scheduling():
    try:
        import inspect
        if 'schedule_jobs' in globals():
            func = globals()['schedule_jobs']
        else:
            return False
            
        sig = inspect.signature(func)
        if len(sig.parameters) == 0:
            res = func()
            return res is not None
        else:
            jobs = [
                {"id": 1, "duration": 2, "deadline": 5, "dependencies": []},
                {"id": 2, "duration": 3, "deadline": 4, "dependencies": [1]},
            ]
            res = func(jobs)
            return res is not None or isinstance(res, list)
    except Exception:
        return False
""",
    "test_trie": """
def test_trie():
    try:
        try:
            root = TrieNode()
        except Exception:
            root = {}
        trie_insert(root, "apple")
        trie_insert(root, "app")
        if trie_search(root, "apple") is not True:
            return False
        if trie_search(root, "app") is not True:
            return False
        if trie_search(root, "appl") is not False:
            return False
        if trie_search(root, "banana") is not False:
            return False
        return True
    except Exception:
        return False
""",
}


def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip whitespace/punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def eval_exact_match(response: str, expected: str, **_) -> tuple[float, str]:
    """Check if normalized response matches expected."""
    if normalize(response) == normalize(expected):
        return 1.0, "Exact match"
    return 0.0, f"Expected: {expected}"


def eval_contains_keywords(response: str, expected: list[str], **_) -> tuple[float, str]:
    """Check if response contains required keywords (word-boundary matching)."""
    resp_lower = response.lower()
    found = []
    for kw in expected:
        if not kw:
            continue
        start_boundary = r'\b' if (kw[0].isalnum() or kw[0] == '_') else r'(?:^|(?<=\W))'
        end_boundary = r'\b' if (kw[-1].isalnum() or kw[-1] == '_') else r'(?:$|(?<=\W))'
        pattern = start_boundary + re.escape(kw.lower()) + end_boundary
        if re.search(pattern, resp_lower):
            found.append(kw)
    if len(found) == len(expected):
        return 1.0, f"All keywords found: {', '.join(expected)}"
    if found:
        ratio = len(found) / len(expected)
        return ratio, f"Found {len(found)}/{len(expected)}: {', '.join(found)}"
    return 0.0, f"Keywords not found: {', '.join(expected)}"


def eval_numeric_match(response: str, expected: float, **_) -> tuple[float, str]:
    """Extract a number from the response and compare to expected."""
    cleaned = response.replace(",", "").replace("$", "")

    sci_matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][+-]?\d+)?', cleaned)
    multiplier_matches = re.findall(r'(\d+\.?\d*)\s*[×x*]\s*10\s*\^?\s*([-+]?\d+)', cleaned)

    all_numbers = list(sci_matches)
    for base, exp in multiplier_matches:
        try:
            all_numbers.append(str(float(base) * (10 ** int(exp))))
        except ValueError:
            continue

    if not all_numbers:
        return 0.0, f"No number found, expected {expected}"

    def matches(num_str: str) -> bool:
        try:
            num = float(num_str)
        except ValueError:
            return False
        return abs(num - expected) < 0.01 or (expected != 0 and abs(num - expected) / abs(expected) < 0.01)

    # The final answer almost always appears last. Full credit if the LAST number
    # matches; partial credit if only an intermediate number matches (the model may
    # have computed it but failed to state a clean final answer). This stops a
    # response from passing merely because the expected value appears as an
    # intermediate step or is echoed from the prompt/context.
    if matches(all_numbers[-1]):
        return 1.0, f"Correct (final answer): {all_numbers[-1]}"

    if any(matches(n) for n in all_numbers):
        return 0.5, (
            f"Expected {expected} appears but not as the final answer "
            f"(last number was {all_numbers[-1]})"
        )

    numbers_str = ", ".join(all_numbers[:5])
    if len(all_numbers) > 5:
        numbers_str += "..."
    return 0.0, f"Got {numbers_str}, expected {expected}"


def _extract_code(response: str) -> str | None:
    """Pull a Python code block (or a best-effort def/class span) from a response."""
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if code_match:
        return code_match.group(1)
    lines = response.strip().split("\n")
    code_lines = []
    in_code = False
    for l in lines:
        stripped = l.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            in_code = True
        if in_code and stripped and not stripped.startswith("#"):
            code_lines.append(l)
    if len(code_lines) >= 2:
        return "\n".join(code_lines)
    return None


def _strip_disallowed_imports(code: str) -> str:
    """Remove import/from-import lines for modules not in the sandbox allowlist.

    LLMs sometimes add unnecessary imports (e.g. ``import numpy as np`` for a
    fibonacci function).  Those would fail inside the sandbox's restricted
    import gate.  Stripping them lets the core logic run without being blocked
    by a missing third-party package.
    """
    from code_runner import ALLOWED_IMPORTS

    allowed = set(ALLOWED_IMPORTS)
    out = []
    for line in code.splitlines():
        stripped = line.strip()
        # "import foo" or "import foo.bar"
        m = re.match(r"^import\s+(\S+)", stripped)
        if m:
            root = m.group(1).split(".")[0]
            if root not in allowed:
                continue
            out.append(line)
            continue
        # "from foo import ..." or "from foo.bar import ..."
        m = re.match(r"^from\s+(\S+)\s+import", stripped)
        if m:
            root = m.group(1).split(".")[0]
            if root not in allowed:
                continue
            out.append(line)
            continue
        out.append(line)
    return "\n".join(out)


def eval_code_exec(response: str, expected: list[dict], **_) -> tuple[float, str]:
    """Extract code from response and run test cases in an isolated subprocess.

    Execution happens out-of-process (see code_runner.run_code_tests) with a hard
    timeout and import allowlist, so a malicious or non-terminating solution can
    neither escape nor hang the benchmark.
    """
    from code_runner import run_code_tests

    code = _extract_code(response)
    if code is None:
        return 0.0, "No code block found"

    code = _strip_disallowed_imports(code)

    # All helper-based tests in a question share the same helper name.
    helper_name = next((t["function"] for t in expected if t["function"] in TEST_HELPERS), None)
    helper = TEST_HELPERS.get(helper_name) if helper_name else None

    tests = [{"function": t["function"], "args": t["args"]} for t in expected]
    outcome = run_code_tests(code, tests, helper=helper)

    if "error" in outcome:
        return 0.0, f"Execution failed: {outcome['error']}"

    results = outcome.get("results", [])
    passed = 0
    total = len(expected)
    details = []
    for i, (test, res) in enumerate(zip(expected, results)):
        want = test["expected"]
        if not res.get("ok"):
            details.append(f"Test {i+1}: ERROR ({res.get('error')})")
            continue
        if res.get("result") == repr(want):
            passed += 1
            details.append(f"Test {i+1}: PASS")
        else:
            details.append(f"Test {i+1}: FAIL (got {res.get('result')}, want {repr(want)})")

    score = passed / total if total > 0 else 0.0
    return score, f"{passed}/{total} tests passed. " + "; ".join(details)


def eval_format_check(response: str, expected: dict, **_) -> tuple[float, str]:
    """Check if response follows formatting instructions."""
    checks = expected.get("checks", [])
    passed = 0
    details = []

    for check in checks:
        ctype = check["type"]
        if ctype == "json":
            json_str = response.strip()
            # Strip markdown code blocks if present
            if json_str.startswith("```"):
                json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
                if json_match:
                    json_str = json_match.group(1).strip()
            else:
                # Find first { or [ to last } or ]
                first_brace = response.find("{")
                first_bracket = response.find("[")
                start_idx = -1
                if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                    start_idx = first_brace
                    end_idx = response.rfind("}")
                elif first_bracket != -1:
                    start_idx = first_bracket
                    end_idx = response.rfind("]")
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx+1].strip()

            try:
                json.loads(json_str)
                passed += 1
                details.append("Valid JSON: PASS")
            except json.JSONDecodeError:
                details.append("Valid JSON: FAIL")
        elif ctype == "contains":
            target = check["value"]
            if target.lower() in response.lower():
                passed += 1
                details.append(f"Contains '{target}': PASS")
            else:
                details.append(f"Contains '{target}': FAIL")
        elif ctype == "not_contains":
            target = check["value"]
            if target.lower() not in response.lower():
                passed += 1
                details.append(f"Does not contain '{target}': PASS")
            else:
                details.append(f"Does not contain '{target}': FAIL")
        elif ctype == "max_words":
            limit = check["value"]
            word_count = len(response.split())
            if word_count <= limit:
                passed += 1
                details.append(f"Word count {word_count} <= {limit}: PASS")
            else:
                details.append(f"Word count {word_count} > {limit}: FAIL")
        elif ctype == "min_words":
            limit = check["value"]
            word_count = len(response.split())
            if word_count >= limit:
                passed += 1
                details.append(f"Word count {word_count} >= {limit}: PASS")
            else:
                details.append(f"Word count {word_count} < {limit}: FAIL")
        elif ctype == "word_count_exact":
            target = check["value"]
            word_count = len(response.split())
            if word_count == target:
                passed += 1
                details.append(f"Word count {word_count} == {target}: PASS")
            else:
                details.append(f"Word count {word_count} != {target}: FAIL")
        elif ctype == "starts_with":
            prefix = check["value"]
            if response.strip().startswith(prefix):
                passed += 1
                details.append(f"Starts with '{prefix}': PASS")
            else:
                details.append(f"Starts with '{prefix}': FAIL")
        elif ctype == "regex":
            pattern = check["value"]
            if safe_re_search(pattern, response):
                passed += 1
                details.append(f"Regex match: PASS")
            else:
                details.append(f"Regex match: FAIL")
        elif ctype == "not_regex":
            pattern = check["value"]
            if not safe_re_search(pattern, response):
                passed += 1
                details.append(f"Regex absent: PASS")
            else:
                details.append(f"Regex absent: FAIL")
        elif ctype == "line_count":
            expected_count = check["value"]
            actual = len([l for l in response.strip().split("\n") if l.strip()])
            if actual == expected_count:
                passed += 1
                details.append(f"Line count {actual} == {expected_count}: PASS")
            else:
                details.append(f"Line count {actual} != {expected_count}: FAIL")
        elif ctype == "numbered_list":
            lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
            numbered_lines = [l for l in lines if re.match(r"^\d+[\.\)]\s", l)]
            if len(numbered_lines) >= 2:
                passed += 1
                details.append("Numbered list: PASS")
            else:
                details.append("Numbered list: FAIL")
        elif ctype == "bullet_list":
            lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
            bullet_lines = [l for l in lines if re.match(r"^[-\*\+•]\s", l)]
            if len(bullet_lines) >= 2:
                passed += 1
                details.append("Bullet list: PASS")
            else:
                details.append("Bullet list: FAIL")
        elif ctype == "min_sentences":
            limit = check["value"]
            sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
            actual = len(sentences)
            if actual >= limit:
                passed += 1
                details.append(f"Sentence count {actual} >= {limit}: PASS")
            else:
                details.append(f"Sentence count {actual} < {limit}: FAIL")
        elif ctype == "max_sentences":
            limit = check["value"]
            sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
            actual = len(sentences)
            if actual <= limit:
                passed += 1
                details.append(f"Sentence count {actual} <= {limit}: PASS")
            else:
                details.append(f"Sentence count {actual} > {limit}: FAIL")

    score = passed / len(checks) if checks else 0.0
    return score, f"{passed}/{len(checks)} format checks passed. " + "; ".join(details)


def eval_admits_uncertainty(response: str, expected: None = None, **_) -> tuple[float, str]:
    """Check if the model admits uncertainty about fictional/impossible things."""
    refusal_patterns = [
        r"i (?:do not|don't) know",
        r"i (?:cannot|can't) (?:find|locate|verify|confirm)",
        r"there is no (?:such|record of)",
        r"(?:does not|doesn't) exist",
        r"(?:did not|didn't) happen",
        r"(?:is not|isn't) (?:a real|a known|a valid)",
        r"(?:no|not a) (?:real|known|valid|documented)",
        r"i (?:am not|'m not) aware of",
        r"unable to (?:find|locate|verify|confirm)",
        r"(?:cannot|can't) (?:verify|confirm|validate)",
        r"(?:unverified|unconfirmed|fabricated)",
        r"no (?:evidence|record|documentation)",
        r"(?:fictional|fiction|made.?up|invented|not real|doesn't exist|never existed)",
        r"i (?:have no|don't have) (?:information|data|knowledge)",
        r"no one (?:has|have|is)",
        r"no human (?:has|have|is)",
        r"(?:never been|hasn't been) (?:done|achieved|accomplished)",
        r"(?:humans?|people) (?:do not|don't|cannot|can't) (?:perform|do|produce)",
        r"(?:is not|isn't) (?:possible|capable|able)",
        r"humans? (?:do not|don't) (?:have|possess)",
    ]
    resp_lower = response.lower()
    matches = [p for p in refusal_patterns if re.search(p, resp_lower)]
    if matches:
        return 1.0, "Properly admitted uncertainty"
    return 0.0, "Did not admit uncertainty (may have fabricated)"


def eval_security_analysis(response: str, expected: dict, **_) -> tuple[float, str]:
    """Evaluate security analysis responses using criteria and must_not patterns.

    Expected format:
        {
            "criteria": ["regex1", "regex2", ...],   # patterns that SHOULD match
            "must_not": ["regex1", ...]                # patterns that MUST NOT match
        }
    """
    criteria = expected.get("criteria", [])
    must_not = expected.get("must_not", [])

    # Match case-insensitively against the original response (DOTALL so patterns
    # like a JSON-array check can span newlines). We do NOT lowercase the response,
    # so uppercase literals in patterns (e.g. CERT_NONE, A01) still match.
    flags = re.IGNORECASE | re.DOTALL

    # Check criteria (should match)
    criteria_met = []
    criteria_missed = []
    for pattern in criteria:
        if safe_re_search(pattern, response, flags):
            criteria_met.append(pattern)
        else:
            criteria_missed.append(pattern)

    # Check must_not (must NOT match)
    must_not_violations = []
    for pattern in must_not:
        if safe_re_search(pattern, response, flags):
            must_not_violations.append(pattern)

    # Calculate score
    total_criteria = len(criteria)
    if total_criteria == 0:
        criteria_score = 1.0
    else:
        criteria_score = len(criteria_met) / total_criteria

    # Penalty for must_not violations: each violation reduces score by 0.2
    penalty = len(must_not_violations) * 0.2
    score = max(0.0, criteria_score - penalty)

    # Build detail
    details = []
    if criteria_met:
        details.append(f"Criteria met: {len(criteria_met)}/{total_criteria}")
    if criteria_missed:
        details.append(f"Criteria missed: {', '.join(criteria_missed[:3])}")
    if must_not_violations:
        details.append(f"Must-not violations: {', '.join(must_not_violations)}")

    return score, ". ".join(details) if details else "No criteria evaluated"


def eval_file_content_match(response: str, expected: dict, **_) -> tuple[float, str]:
    """Check if response describes correct file content for a task.

    Expected format:
        {
            "content": "expected content string",
            "content_patterns": ["regex1", "regex2"],  # alternative: regex patterns
            "file_name": "expected_filename.txt"        # optional filename check
        }
    """
    expected_content = expected.get("content", "")
    content_patterns = expected.get("content_patterns", [])
    expected_filename = expected.get("file_name", "")

    filename_score = 0.0
    details = []

    # Check filename if provided
    if expected_filename:
        if expected_filename.lower() in response.lower():
            filename_score = 0.2
            details.append(f"Filename '{expected_filename}' mentioned: PASS")
        else:
            details.append(f"Filename '{expected_filename}' not found: FAIL")

    # Check content or patterns
    if expected_content:
        content_max = 0.8 if expected_filename else 1.0
        if expected_content.lower() in response.lower():
            score = filename_score + content_max
            details.append("Content match: PASS")
        else:
            score = filename_score
            details.append(f"Content not found: FAIL")
    elif content_patterns:
        matched = sum(1 for p in content_patterns if safe_re_search(p, response, re.IGNORECASE))
        ratio = matched / len(content_patterns) if content_patterns else 0
        content_max = 0.8 if expected_filename else 1.0
        score = filename_score + (content_max * ratio)
        details.append(f"Patterns matched: {matched}/{len(content_patterns)}")
    else:
        score = filename_score
        details.append("No content or patterns to check")

    return min(1.0, score), ". ".join(details)


def eval_command_correctness(response: str, expected: list[dict], **_) -> tuple[float, str]:
    """Check if response contains correct commands for a task.

    Expected format: list of dicts, each with:
        {
            "pattern": "regex pattern the command should match",
            "description": "human-readable description",
            "required": true/false  # whether this command is mandatory
        }
    """
    if not expected:
        return 1.0, "No commands to check"

    passed = 0
    total = len(expected)
    details = []

    for i, cmd in enumerate(expected):
        pattern = cmd.get("pattern", "")
        desc = cmd.get("description", f"Command {i+1}")
        required = cmd.get("required", True)

        if safe_re_search(pattern, response, re.IGNORECASE):
            passed += 1
            details.append(f"{desc}: PASS")
        elif not required:
            passed += 1
            details.append(f"{desc}: PASS (optional)")
        else:
            details.append(f"{desc}: FAIL")

    score = passed / total if total > 0 else 0.0
    return score, f"{passed}/{total} commands correct. " + "; ".join(details)


def eval_multi_step_solution(response: str, expected: list[dict], **_) -> tuple[float, str]:
    """Check if response contains all required steps in correct order.

    Expected format: list of dicts, each with:
        {
            "step": "description of the step",
            "pattern": "regex pattern that should appear in this step",
            "order": 1  # expected order (1-indexed)
        }
    """
    if not expected:
        return 1.0, "No steps to check"

    # Sort by order
    sorted_steps = sorted(expected, key=lambda x: x.get("order", 0))

    passed = 0
    total = len(sorted_steps)
    details = []
    last_pos = -1

    for i, step in enumerate(sorted_steps):
        pattern = step.get("pattern", "")
        desc = step.get("step") or step.get("description") or f"Step {i+1}"

        match = safe_re_search(pattern, response, re.IGNORECASE)
        if match:
            current_pos = match.start()
            if current_pos > last_pos:
                passed += 1
                details.append(f"{desc}: PASS (correct order)")
                last_pos = current_pos
            else:
                passed += 0.5  # Partial credit for correct step but wrong order
                details.append(f"{desc}: PARTIAL (wrong order)")
        else:
            details.append(f"{desc}: FAIL (not found)")

    score = passed / total if total > 0 else 0.0
    return score, f"{passed}/{total} steps correct. " + "; ".join(details)


EVALUATORS: dict[str, Callable] = {
    "exact_match": eval_exact_match,
    "contains_keywords": eval_contains_keywords,
    "numeric_match": eval_numeric_match,
    "code_exec": eval_code_exec,
    "format_check": eval_format_check,
    "admits_uncertainty": eval_admits_uncertainty,
    "security_analysis": eval_security_analysis,
    "file_content_match": eval_file_content_match,
    "command_correctness": eval_command_correctness,
    "multi_step_solution": eval_multi_step_solution,
}
