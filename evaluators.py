"""Evaluator functions for LLM benchmark responses.

Each evaluator takes a response string and an expected value, returning
``(score, detail)``. ``score`` is 0.0-1.0 and ``detail`` is a human-readable
explanation. Whether a score counts as a *pass* is decided by the question's
``pass_threshold`` (default 1.0 - see ``models.Question``), not by a magic 0.5
constant baked into the evaluators.

Scoring rules this module deliberately enforces
-----------------------------------------------
1. **No accidental partial credit.** Earlier versions returned 0.5 when the
   expected number merely appeared as an intermediate step, when half the
   keywords were found, or when a required step appeared out of order. Combined
   with a 0.5 pass threshold, every one of those was a false pass. Partial
   credit still exists where it is genuinely informative (how many code tests
   passed, how many security criteria were met), but the pass bar is set per
   question instead.
2. **No accidental partial failure.** ``code_exec`` compares returned values
   structurally, so a correct solution returning ``2`` where the fixture says
   ``2.0`` (or a tuple where the fixture says a list) is no longer a false fail.
3. **Alternatives must be explicit.** A bare keyword list means *all of these*.
   Use ``any:``/``groups:`` to express "any of these spellings". ``validate_suite``
   fails the build on bare lists that look like alternatives.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Numeric comparison default: relative tolerance, so 1e9-scale answers are not
# held to an absolute 1e-9 budget.
DEFAULT_REL_TOLERANCE = 1e-6
DEFAULT_ABS_TOLERANCE = 1e-9


def safe_re_search(pattern: str, string: str, flags: int = 0) -> re.Match | None:
    """re.search with error handling for invalid patterns."""
    try:
        return re.search(pattern, string, flags)
    except re.error as e:
        logger.warning("Invalid regex pattern %r: %s", pattern, e)
        return None


def compile_or_none(pattern: str, flags: int = 0):
    try:
        return re.compile(pattern, flags)
    except re.error as e:
        logger.warning("Invalid regex pattern %r: %s", pattern, e)
        return None


# ---------------------------------------------------------------------------
# Structural value comparison (used by code_exec and json_match)
# ---------------------------------------------------------------------------


class Opaque:
    """A value that could not be represented as JSON; only its repr survived."""

    __slots__ = ("repr_str",)

    def __init__(self, repr_str: str):
        self.repr_str = repr_str

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Opaque({self.repr_str!r})"


class SetVal(list):
    """A set that crossed the sandbox boundary; compares order-insensitively."""


def decode_sandbox_value(enc: Any) -> Any:
    """Turn the sandbox's tagged encoding back into a comparable Python value."""
    if not isinstance(enc, dict):
        return enc
    kind = enc.get("kind")
    if kind == "json":
        return enc.get("json")
    if kind == "special_float":
        try:
            return float(enc.get("value", "nan"))
        except ValueError:
            return float("nan")
    if kind == "seq":
        return [decode_sandbox_value(i) for i in enc.get("items", [])]
    if kind == "set":
        return SetVal(decode_sandbox_value(i) for i in enc.get("items", []))
    if kind == "map":
        out = {}
        for pair in enc.get("items", []):
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            key = decode_sandbox_value(pair[0])
            if isinstance(key, (list, dict)):
                key = repr(key)
            out[key] = decode_sandbox_value(pair[1])
        return out
    return Opaque(str(enc.get("repr", "")))


def _numbers_equal(a: float, b: float, rel: float, abs_tol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= max(abs_tol, rel * abs(b))


def _multiset_equal(got: list, want: list, rel: float, abs_tol: float) -> bool:
    remaining = list(want)
    for item in got:
        for i, candidate in enumerate(remaining):
            if values_equal(item, candidate, rel=rel, abs_tol=abs_tol, unordered=True):
                del remaining[i]
                break
        else:
            return False
    return not remaining


def values_equal(
    got: Any,
    want: Any,
    *,
    rel: float = DEFAULT_REL_TOLERANCE,
    abs_tol: float = DEFAULT_ABS_TOLERANCE,
    unordered: bool = False,
) -> bool:
    """Structural equality with numeric tolerance and list/tuple equivalence.

    Booleans are never equal to numbers (``True != 1``) because a function that
    returns 1 where the spec says True is genuinely wrong. Sequences compare
    element-wise unless ``unordered`` is set. Sets always compare as multisets.
    """
    if isinstance(got, bool) or isinstance(want, bool):
        return isinstance(got, bool) and isinstance(want, bool) and got == want
    if got is None or want is None:
        return got is None and want is None
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        return _numbers_equal(float(got), float(want), rel, abs_tol)
    if isinstance(got, str) and isinstance(want, str):
        return got == want
    if isinstance(got, SetVal) or isinstance(want, SetVal):
        if not isinstance(got, list) or not isinstance(want, list):
            return False
        return len(got) == len(want) and _multiset_equal(list(got), list(want), rel, abs_tol)
    if isinstance(got, list) and isinstance(want, list):
        if len(got) != len(want):
            return False
        if unordered:
            return _multiset_equal(got, want, rel, abs_tol)
        return all(
            values_equal(g, w, rel=rel, abs_tol=abs_tol, unordered=unordered)
            for g, w in zip(got, want)
        )
    if isinstance(got, dict) and isinstance(want, dict):
        if set(got.keys()) != set(want.keys()):
            return False
        return all(
            values_equal(got[k], want[k], rel=rel, abs_tol=abs_tol, unordered=unordered)
            for k in want
        )
    if isinstance(got, Opaque):
        return got.repr_str == repr(want)
    return got == want


def describe(value: Any, limit: int = 120) -> str:
    try:
        text = json.dumps(value, default=repr)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip whitespace/punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_api_error(response: str) -> bool:
    """True when the runner recorded a transport failure rather than a model answer.

    These must never be scored as content failures; the caller surfaces them
    separately so a flaky endpoint is not reported as a quality problem.
    """
    return response.strip().startswith("[API ERROR")


def strip_think_blocks(response: str) -> str:
    """Remove reasoning-model scratchpads so they cannot satisfy content checks.

    A model that muses "maybe the answer is 42" inside <think> and then answers 7
    must score as 7. Same for the final-answer extraction in numeric_match.
    """
    cleaned = re.sub(
        r"<(think|thinking|reasoning|scratchpad)>.*?</\1>",
        " ",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Some endpoints strip the opening tag and return only the closing one; in
    # that case everything up to the last closing tag is scratchpad.
    tail = re.search(
        r"</(?:think|thinking|reasoning|scratchpad)>(?!.*</(?:think|thinking|reasoning|scratchpad)>)",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if tail:
        cleaned = cleaned[tail.end():]
    # An unterminated opening tag is a malformed tag, not a scratchpad marker:
    # drop the tag itself but keep the text after it, so a model that wraps its
    # final answer in a typo'd tag is still graded on the answer.
    cleaned = re.sub(
        r"<(?:think|thinking|reasoning|scratchpad)>", " ", cleaned,
        flags=re.IGNORECASE,
    )
    # A response that is nothing but scratchpad means there is no answer to
    # score; it must come back empty, not fall through to grading the musing.
    return cleaned


def keyword_pattern(keyword: str) -> str:
    """Word-boundary-aware pattern for a literal keyword.

    ``\\b`` only works next to word characters, so a keyword like ``a*`` or
    ``3/11`` needs relaxed boundaries at the non-word end.
    """
    if not keyword:
        return r"(?!x)x"  # never matches
    start = r"\b" if (keyword[0].isalnum() or keyword[0] == "_") else r"(?:^|(?<=\W))"
    end = r"\b" if (keyword[-1].isalnum() or keyword[-1] == "_") else r"(?:$|(?=\W))"
    return start + re.escape(keyword) + end


def contains_keyword(text: str, keyword: str) -> bool:
    return re.search(keyword_pattern(keyword.lower()), text.lower()) is not None


def split_sentences(text: str) -> list[str]:
    """Split into sentences without counting decimals or abbreviations as breaks."""
    protected = re.sub(r"(?<=\d)\.(?=\d)", "\u0000", text)
    for abbr in ("e.g.", "i.e.", "etc.", "Mr.", "Mrs.", "Ms.", "Dr.", "vs.", "U.S."):
        protected = protected.replace(abbr, abbr.replace(".", "\u0000"))
    parts = re.split(r"[.!?]+(?:\s|$)", protected)
    return [p.replace("\u0000", ".").strip() for p in parts if p.strip()]


def word_count(text: str) -> int:
    return len(re.findall(r"[^\s]+", text))


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

# A number read as the asserted answer that is written as a percentage means a
# fraction: "the answer is 75%" claims 0.75, not 75.
_POST_PERCENT_RE = re.compile(r"\s*[\)\]'\"\u201d]*%")

# Phrases that introduce an asserted answer. Bare "=" and ":" are deliberately
# excluded: the last "=" in a worked solution is usually an intermediate step in
# the wrong unit (``ball = 0.05`` for a question answered in cents), and treating
# it as the answer produced false failures.
_FINAL_MARKERS = re.compile(
    r"(?:final\s+answer|answer|result|total|solution|conclusion|therefore|thus|hence|so)\b"
    r"[^\S\n]*(?:is|are|of|=|:)?[^\S\n]*[:=]?[^\S\n]*\**[^\S\n]*",
    re.IGNORECASE,
)


def precision_tolerance(target: float) -> float:
    """Absolute tolerance implied by how precisely the expected value is written.

    ``38.04`` accepts anything that rounds to it (+/- 0.005); ``42925`` is exact.
    A single loose default (the old 0.01) let a wrong 4-decimal probability pass.
    """
    if target == int(target):
        return DEFAULT_ABS_TOLERANCE
    text = repr(float(target))
    if "e" in text or "E" in text:
        return DEFAULT_ABS_TOLERANCE
    decimals = len(text.split(".")[1].rstrip("0")) or 1
    return 0.5 * (10 ** -decimals)


def _clean_numeric_text(text: str) -> str:
    """Normalize notation so a single regex pass can find real numbers."""
    text = text.replace("\u2212", "-").replace("\u2013", "-")  # unicode minus/en-dash
    text = re.sub(r"\\(?:boxed|text|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\[,!;: ]", "", text)
    text = text.replace("$", "").replace("\u00a0", " ")
    # Thousands separators: only between digit groups of exactly 3.
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    text = re.sub(r"(?<=\d)[  ](?=\d{3}(?!\d))", "", text)
    # Scientific notation written out: 6.02 x 10^23 / 6.02 * 10**23
    def _sci(m: re.Match) -> str:
        try:
            return repr(float(m.group(1)) * (10 ** int(m.group(2))))
        except (ValueError, OverflowError):
            return m.group(0)

    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*[x\u00d7*]\s*10\s*(?:\^|\*\*)?\s*([-+]?\d+)", _sci, text
    )
    return text


def _eval_simple_fractions(text: str) -> str:
    """Turn ``a/b`` and ``(a)/(b)`` tokens into decimals so they can be compared."""

    def _div(m: re.Match) -> str:
        try:
            num, den = float(m.group(1)), float(m.group(2))
            if den == 0:
                return m.group(0)
            return repr(num / den)
        except (ValueError, OverflowError):
            return m.group(0)

    text = re.sub(r"\((-?\d+(?:\.\d+)?)\)\s*/\s*\((-?\d+(?:\.\d+)?)\)", _div, text)
    # The trailing guard is `(?!\.?\d)` rather than `(?![\d.])` so a sentence-final
    # period after the fraction ("the probability is 3/11.") does not block the
    # conversion, while "1/2.5" still consumes the whole denominator.
    return re.sub(r"(?<![\d.])(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?!\.?\d)", _div, text)


def extract_numbers(text: str, expand_fractions: bool = True) -> list[float]:
    cleaned = _clean_numeric_text(text)
    if expand_fractions:
        cleaned = _eval_simple_fractions(cleaned)
    out = []
    for token in _NUM_RE.findall(cleaned):
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def _final_number_value(num: re.Match, tail_after_number: str) -> float | None:
    """Convert a numeric match to the asserted value, honoring a trailing ``%``."""
    try:
        value = float(num.group(0))
    except ValueError:
        return None
    if _POST_PERCENT_RE.match(tail_after_number):
        value /= 100.0
    return value


def extract_final_number(text: str) -> tuple[float | None, str]:
    """Best-effort extraction of the number the model is *asserting* as its answer.

    Strategy, in order:
      1. The whole response is a single number.
      2. The last "answer is <n>"-style marker.
      3. The last number on the last non-empty line (models put the answer last).
      4. The last number anywhere.

    A number immediately followed by ``%`` is read as a percentage, so a stated
    "75%" is 0.75. Returns ``(value, how)`` where ``how`` names the strategy,
    for the detail text.
    """
    body = strip_think_blocks(text).strip()
    if not body:
        return None, "empty response"

    cleaned_all = _eval_simple_fractions(_clean_numeric_text(body))

    single = _NUM_RE.fullmatch(cleaned_all.strip().rstrip(".%"))
    if single:
        value = _final_number_value(single, cleaned_all.strip()[single.end():])
        if value is not None:
            return value, "bare number"

    for match in reversed(list(_FINAL_MARKERS.finditer(cleaned_all))):
        tail = cleaned_all[match.end():]
        num = _NUM_RE.match(tail)
        if num:
            value = _final_number_value(num, tail[num.end():])
            if value is not None:
                return value, "stated answer"

    lines = [line for line in cleaned_all.splitlines() if line.strip()]
    if lines:
        for candidate in reversed(list(_NUM_RE.finditer(lines[-1]))):
            value = _final_number_value(candidate, lines[-1][candidate.end():])
            if value is not None:
                return value, "last line"

    for candidate in reversed(list(_NUM_RE.finditer(cleaned_all))):
        value = _final_number_value(candidate, cleaned_all[candidate.end():])
        if value is not None:
            return value, "last number"
    return None, "no number found"


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json(response: str) -> tuple[Any, str | None]:
    """Pull the first plausible JSON document out of a response.

    Returns ``(value, None)`` or ``(None, reason)``.
    """
    text = strip_think_blocks(response).strip()

    fenced = re.findall(r"```(?:json|jsonc|json5)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = [f.strip() for f in fenced]
    candidates.append(text)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            pass
        # Fall back to the widest balanced {...} / [...] span.
        for opener, closer in (("{", "}"), ("[", "]")):
            start = candidate.find(opener)
            end = candidate.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start:end + 1]), None
                except json.JSONDecodeError:
                    continue
    return None, "no valid JSON found"


def json_at_path(doc: Any, path: str) -> tuple[Any, bool]:
    """Resolve a dotted/bracketed path such as ``stats.total_skills`` or ``[0].id``."""
    current = doc
    for part in re.findall(r"[^.\[\]]+", path):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None, False
        elif isinstance(current, dict):
            if part not in current:
                return None, False
            current = current[part]
        else:
            return None, False
    return current, True


# ---------------------------------------------------------------------------
# Code-execution harnesses
# ---------------------------------------------------------------------------

TEST_HELPERS = {
    "test_lru": """
def test_lru():
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1, "get(1) after two puts"
    cache.put(3, 3)                      # evicts key 2 (least recently used)
    assert cache.get(2) == -1, "key 2 should have been evicted"
    cache.put(4, 4)                      # evicts key 1
    assert cache.get(1) == -1, "key 1 should have been evicted"
    assert cache.get(3) == 3
    assert cache.get(4) == 4
    # Overwriting an existing key must refresh recency, not insert a new entry.
    cache2 = LRUCache(2)
    cache2.put(1, 1)
    cache2.put(2, 2)
    cache2.put(1, 10)
    cache2.put(3, 3)
    assert cache2.get(2) == -1, "overwrite must count as a use of key 1"
    assert cache2.get(1) == 10, "overwrite must update the stored value"
    # get() must also refresh recency.
    cache3 = LRUCache(2)
    cache3.put(1, 1)
    cache3.put(2, 2)
    cache3.get(1)
    cache3.put(3, 3)
    assert cache3.get(2) == -1, "get() must refresh recency"
    assert cache3.get(1) == 1
    # Capacity 1 edge case.
    cache4 = LRUCache(1)
    cache4.put(5, 5)
    cache4.put(6, 6)
    assert cache4.get(5) == -1 and cache4.get(6) == 6
    return True
""",
    "test_serialize": """
def _tree_from_list(values):
    # values is a level-order list with None for missing children
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = [root]
    i = 1
    while queue and i < len(values):
        node = queue.pop(0)
        if i < len(values):
            v = values[i]; i += 1
            if v is not None:
                node.left = TreeNode(v); queue.append(node.left)
        if i < len(values):
            v = values[i]; i += 1
            if v is not None:
                node.right = TreeNode(v); queue.append(node.right)
    return root

def _same(p, q):
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    return p.val == q.val and _same(p.left, q.left) and _same(p.right, q.right)

def test_serialize():
    cases = [
        [1, 2, 3, None, None, 4, 5],
        [],
        [1],
        [1, 2, None, 3, None, 4],           # left-skewed chain
        [0, -1, None, None, -2],            # zero and negative values
        [1, 2, 2, 3, None, None, 3],
    ]
    for values in cases:
        root = _tree_from_list(values)
        data = serialize_tree(root)
        assert isinstance(data, str), "serialize_tree must return a str, got %r" % type(data)
        rebuilt = deserialize_tree(data)
        assert _same(root, rebuilt), "round-trip failed for %r" % (values,)
    return True
""",
    "test_bst": """
def _build(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = [root]
    i = 1
    while queue and i < len(values):
        node = queue.pop(0)
        if i < len(values):
            v = values[i]; i += 1
            if v is not None:
                node.left = TreeNode(v); queue.append(node.left)
        if i < len(values):
            v = values[i]; i += 1
            if v is not None:
                node.right = TreeNode(v); queue.append(node.right)
    return root

def test_bst():
    valid = [
        [2, 1, 3],
        [],
        [1],
        [5, 3, 8, 2, 4, 7, 9],
    ]
    invalid = [
        [5, 1, 4, None, None, 3, 6],      # classic: 3 < 5 but in right subtree
        [10, 5, 15, None, None, 6, 20],   # 6 violates the ancestor bound
        [2, 2],                            # duplicates are not a valid BST
        [3, 1, 4, None, 2],                # 2 is fine here...
    ]
    for values in valid:
        assert is_valid_bst(_build(values)) is True, "expected valid: %r" % (values,)
    for values in invalid[:3]:
        assert is_valid_bst(_build(values)) is False, "expected invalid: %r" % (values,)
    assert is_valid_bst(_build([3, 1, 4, None, 2])) is True
    return True
""",
    "test_trie": """
def _new_root():
    try:
        return TrieNode()
    except Exception:
        return {}

def test_trie():
    root = _new_root()
    for word in ("apple", "app", "apply", "bat"):
        trie_insert(root, word)
    for word in ("apple", "app", "apply", "bat"):
        assert trie_search(root, word) is True, "inserted word not found: %s" % word
    for word in ("appl", "ap", "batman", "banana", "", "Apple"):
        assert trie_search(root, word) is False, "false positive for: %r" % word
    # An empty trie must not match anything.
    empty = _new_root()
    assert trie_search(empty, "apple") is False
    return True
""",
    "test_rate_limiter": """
def test_rate_limiter():
    # 3 requests per 10-second window, per key.
    rl = RateLimiter(3, 10)
    assert rl.allow("a", 0) is True
    assert rl.allow("a", 1) is True
    assert rl.allow("a", 2) is True
    assert rl.allow("a", 3) is False, "4th request inside the window must be denied"
    # A different key has its own budget.
    assert rl.allow("b", 3) is True
    # Once the window slides past the first request, capacity frees up.
    assert rl.allow("a", 11) is True
    assert rl.allow("a", 11) is True
    return True
""",
    "test_versioned_store": """
def test_versioned_store():
    s = VersionedStore()
    s.set("k", "v1", 1)
    s.set("k", "v2", 5)
    assert s.get("k", 0) is None, "read before the first write must be None"
    assert s.get("k", 1) == "v1"
    assert s.get("k", 4) == "v1", "read must return the latest write at or before t"
    assert s.get("k", 5) == "v2"
    assert s.get("k", 99) == "v2"
    assert s.get("missing", 1) is None
    s.set("k", "v3", 3)
    assert s.get("k", 4) == "v3", "out-of-order writes must still be ordered by time"
    assert s.get("k", 5) == "v2"
    return True
""",
}


def _block_defines(block: str, names: set[str]) -> bool:
    """True when the block defines a function/class matching a fixture name."""
    for name in names:
        if re.search(rf"^\s*(?:def|class)\s+{re.escape(name)}\b", block, re.MULTILINE):
            return True
    return False


def _extract_code(response: str, fixture_names: set[str] | None = None) -> str | None:
    """Pull a Python code block (or a best-effort def/class span) from a response.

    When fixture names are available, prefer the first fenced block that defines
    one of them: a debugging answer that quotes the buggy original after the fix
    would otherwise redefine the function when all blocks are concatenated.
    """
    body = strip_think_blocks(response)
    blocks = re.findall(r"```(?:python|py|python3)?\s*\n(.*?)```", body, re.DOTALL | re.IGNORECASE)
    if blocks:
        if fixture_names:
            for block in blocks:
                if _block_defines(block, fixture_names):
                    return block
        # Concatenate every block: models frequently split a class and its
        # helpers across fences, and a trailing "usage" block is harmless.
        return "\n\n".join(blocks)
    lines = body.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            in_code = True
        if in_code:
            code_lines.append(line)
    if len([line for line in code_lines if line.strip()]) >= 2:
        return "\n".join(code_lines)
    return None


def _strip_disallowed_imports(code: str) -> str:
    """Remove import lines for modules outside the sandbox allowlist.

    LLMs sometimes add unnecessary imports (``import numpy as np`` for a
    fibonacci function). Those would fail at the sandbox's import gate, so
    stripping them lets the core logic run instead of failing for an unrelated
    reason. Imports the solution actually needs will surface as a NameError.
    """
    from code_runner import ALLOWED_IMPORTS

    allowed = set(ALLOWED_IMPORTS)
    out = []
    for line in code.splitlines():
        stripped = line.strip()
        m = re.match(r"^import\s+(\S+)", stripped)
        if m:
            if m.group(1).split(".")[0] not in allowed:
                continue
            out.append(line)
            continue
        m = re.match(r"^from\s+(\S+)\s+import", stripped)
        if m:
            root = m.group(1).split(".")[0]
            if root and root not in allowed and not m.group(1).startswith("."):
                continue
            out.append(line)
            continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def eval_exact_match(response: str, expected: Any, **_) -> tuple[float, str]:
    """The whole (normalized) response must equal the expected answer.

    ``expected`` may be a string or a list of equally acceptable strings.
    """
    if isinstance(expected, dict):
        options = expected.get("any_of") or [expected.get("value", "")]
    elif isinstance(expected, list):
        options = expected
    else:
        options = [expected]

    got = normalize(strip_think_blocks(response))
    for option in options:
        if got == normalize(str(option)):
            return 1.0, f"Exact match: {option}"
    shown = " | ".join(str(o) for o in options)
    return 0.0, f"Got {describe(got, 80)}, expected one of: {shown}"


_MCQ_PATTERNS = [
    r"(?:final\s+answer|answer|option|choice)\s*(?:is)?\s*[:=\-]?\s*\**\(?([A-Za-z])\)?\**\b",
    r"^\s*\**\(?([A-Za-z])\)?\**\s*[.):]?\s*$",
    r"\*\*\(?([A-Za-z])\)?\*\*",
    r"\\boxed\{\s*\(?([A-Za-z])\)?\s*\}",
]


def eval_mcq(response: str, expected: Any, **_) -> tuple[float, str]:
    """Strict multiple-choice grading.

    ``expected`` is the correct letter, or
    ``{"answer": "B", "options": "ABCD"}``. A response that names more than one
    distinct option letter scores 0: hedging across choices is not an answer.
    """
    if isinstance(expected, dict):
        answer = str(expected.get("answer", "")).strip()
        options = str(expected.get("options", "")).strip() or None
    else:
        answer = str(expected).strip()
        options = None

    body = strip_think_blocks(response).strip()
    valid = set((options or "ABCDEFGH").upper())

    found: list[str] = []
    for pattern in _MCQ_PATTERNS:
        for m in re.finditer(pattern, body, re.IGNORECASE | re.MULTILINE):
            letter = m.group(1).upper()
            if letter in valid and letter not in found:
                found.append(letter)
        if found:
            break

    if not found:
        # Last resort: a lone letter token anywhere in a short response. Only
        # trust this when the text also carries an answer cue - otherwise prose
        # articles like "A"/"I" would read as option letters and spuriously
        # fail a correct free-form answer as "Ambiguous".
        has_cue = re.search(
            r"answer|option|choice|correct|boxed|^\s*\*+\s*\(?[A-Za-z]\)?",
            body, re.IGNORECASE | re.MULTILINE,
        )
        if has_cue:
            tokens = re.findall(r"\b([A-Za-z])\b", body)
            found = list(dict.fromkeys(t.upper() for t in tokens if t.upper() in valid))

    if not found:
        return 0.0, f"No option letter found; expected {answer}"
    if len(found) > 1:
        return 0.0, f"Ambiguous: named options {', '.join(found)}; expected {answer}"
    if found[0] == answer.upper():
        return 1.0, f"Correct option: {answer}"
    return 0.0, f"Chose {found[0]}, expected {answer}"


def eval_contains_keywords(response: str, expected: Any, **_) -> tuple[float, str]:
    """Keyword presence with explicit all/any/none semantics.

    Accepted forms::

        expected: [a, b]                 # ALL of a and b must appear
        expected:
          all: [a, b]                    # every entry required
          any: [x, y]                    # at least one required
          n_of: {n: 2, of: [p, q, r]}    # at least n of these
          groups: [[ca, calcium], [f]]   # each group: any-of inside, all groups required
          none: [z]                      # must not appear
          partial: true                  # report coverage as a graded score

    Without ``partial``, the score is binary: every requirement met or 0.
    """
    text = strip_think_blocks(response).lower()

    if isinstance(expected, dict):
        spec = expected
    else:
        spec = {"all": list(expected or [])}

    require_all = [str(k) for k in (spec.get("all") or [])]
    require_any = [str(k) for k in (spec.get("any") or [])]
    groups = [[str(k) for k in g] for g in (spec.get("groups") or [])]
    forbid = [str(k) for k in (spec.get("none") or [])]
    n_of = spec.get("n_of") or {}
    partial = bool(spec.get("partial"))

    requirements: list[tuple[str, bool]] = []  # (label, satisfied)
    details: list[str] = []

    for kw in require_all:
        ok = contains_keyword(text, kw)
        requirements.append((kw, ok))
    if require_any:
        hit = next((kw for kw in require_any if contains_keyword(text, kw)), None)
        requirements.append(("any(" + ", ".join(require_any) + ")", hit is not None))
        if hit:
            details.append(f"any-of matched '{hit}'")
    for group in groups:
        hit = next((kw for kw in group if contains_keyword(text, kw)), None)
        requirements.append(("|".join(group), hit is not None))
    if n_of:
        pool = [str(k) for k in (n_of.get("of") or [])]
        need = int(n_of.get("n", len(pool)))
        hits = [kw for kw in pool if contains_keyword(text, kw)]
        requirements.append((f"{need}-of({len(pool)})", len(hits) >= need))
        details.append(f"n_of matched {len(hits)}/{need} required")

    violations = [kw for kw in forbid if contains_keyword(text, kw)]

    met = [label for label, ok in requirements if ok]
    missed = [label for label, ok in requirements if not ok]

    if not requirements and not forbid:
        return 0.0, "Misconfigured question: no keyword requirements"

    if violations:
        return 0.0, f"Forbidden term(s) present: {', '.join(violations)}"

    total = len(requirements)
    if not missed:
        return 1.0, "All keyword requirements met" + (
            f" ({'; '.join(details)})" if details else ""
        )

    ratio = (len(met) / total) if total else 0.0
    detail = f"Missing {len(missed)}/{total}: {', '.join(missed[:6])}"
    if len(missed) > 6:
        detail += f" (+{len(missed) - 6} more)"
    return (ratio if partial else 0.0), detail


def eval_numeric_match(response: str, expected: Any, **_) -> tuple[float, str]:
    """Compare the model's *final* answer against the expected number.

    ``expected`` may be a bare number or::

        expected:
          value: 38.04
          tolerance: 0.01        # absolute
          relative: 0.001        # relative
          accept: [38.0, 38.04]  # additional acceptable values

    There is no partial credit for the right number appearing as an intermediate
    step: a response that computes 42 and then concludes 7 is simply wrong.
    """
    if isinstance(expected, dict):
        target = float(expected.get("value"))
        rel_tol = float(expected.get("relative", DEFAULT_REL_TOLERANCE))
        if "tolerance" in expected:
            abs_tol = float(expected["tolerance"])
        elif "relative" in expected:
            # An explicit relative tolerance must not be undercut by a fixed
            # absolute floor: for a 1e-34 target, a 1e-9 floor accepts anything.
            abs_tol = 0.0
        else:
            abs_tol = precision_tolerance(target)
        accept = [float(v) for v in (expected.get("accept") or [])]
    else:
        target = float(expected)
        # Answers are authored to the precision the prompt asks for, so the
        # window absorbs rounding of the final digit only.
        abs_tol = precision_tolerance(target)
        rel_tol = DEFAULT_REL_TOLERANCE
        accept = []

    candidates = [target, *accept]
    value, how = extract_final_number(response)
    if value is None:
        return 0.0, f"{how}; expected {target}"

    for candidate in candidates:
        if _numbers_equal(value, candidate, rel_tol, abs_tol):
            return 1.0, f"Correct ({how}): {value:g}"
        # An integer expected value is a count, so a more precise answer that
        # rounds to it ("533.33" for 533) is still correct - but only a true
        # round-to-integer (±0.5, half-open) plus the precision window, so
        # "533.99" for 533 fails: it rounds to 534, not 533.
        if candidate == int(candidate):
            window = max(0.5, abs_tol)
            if abs(value - candidate) <= window + 1e-9 and round(value) == candidate:
                return 1.0, f"Correct ({how}): {value:g} rounds to {candidate:g}"

    return 0.0, f"Got {value:g} ({how}), expected {target:g}"


def eval_numeric_set(response: str, expected: Any, **_) -> tuple[float, str]:
    """Every expected number must appear somewhere in the response.

    For multi-part numeric answers ("give the three coefficients"). Order does
    not matter; every value is required.
    """
    if isinstance(expected, dict):
        values = [float(v) for v in expected.get("values", [])]
        override_tol = expected.get("tolerance")
        rel_tol = float(expected.get("relative", DEFAULT_REL_TOLERANCE))
    else:
        values = [float(v) for v in expected]
        override_tol, rel_tol = None, DEFAULT_REL_TOLERANCE

    found_numbers = extract_numbers(strip_think_blocks(response))
    missing = [
        v for v in values
        if not any(
            _numbers_equal(
                n, v, rel_tol,
                float(override_tol) if override_tol is not None else precision_tolerance(v),
            )
            for n in found_numbers
        )
    ]
    if not missing:
        return 1.0, f"All {len(values)} values present"
    return 0.0, f"Missing {len(missing)}/{len(values)}: {', '.join(f'{v:g}' for v in missing)}"


def eval_code_exec(response: str, expected: Any, **_) -> tuple[float, str]:
    """Extract code from the response and run it against fixtures in a sandbox.

    ``expected`` is a list of ``{function, args, kwargs, expected, unordered,
    tolerance}`` fixtures, or ``{"tests": [...], "helper": "name"}``.
    Values are compared structurally (see ``values_equal``) so an int/float or
    tuple/list difference is not counted as a wrong answer.
    """
    from code_runner import run_code_tests

    if isinstance(expected, dict):
        fixtures = list(expected.get("tests") or [])
        helper_override = expected.get("helper")
    else:
        fixtures = list(expected or [])
        helper_override = None

    if not fixtures:
        return 0.0, "Misconfigured question: no code fixtures"

    code = _extract_code(response, {str(f.get("function")) for f in fixtures})
    if code is None:
        return 0.0, "No code block found"
    code = _strip_disallowed_imports(code)

    helper_name = helper_override or next(
        (f["function"] for f in fixtures if f.get("function") in TEST_HELPERS), None
    )
    helper = TEST_HELPERS.get(helper_name) if helper_name else None
    if helper_name and helper is None:
        return 0.0, f"Misconfigured question: unknown helper {helper_name!r}"

    calls = [
        {
            "function": f["function"],
            "args": f.get("args") or [],
            "kwargs": f.get("kwargs") or {},
        }
        for f in fixtures
    ]
    outcome = run_code_tests(code, calls, helper=helper)

    if "error" in outcome:
        return 0.0, f"Execution failed: {outcome['error']}"

    results = outcome.get("results", [])
    if len(results) != len(fixtures):
        return 0.0, f"Sandbox returned {len(results)} results for {len(fixtures)} fixtures"

    passed = 0
    details: list[str] = []
    for i, (fixture, res) in enumerate(zip(fixtures, results), 1):
        if not res.get("ok"):
            details.append(f"#{i} ERROR: {res.get('error')}")
            continue
        got = decode_sandbox_value(res.get("value"))
        want = fixture.get("expected")
        rel = float(fixture.get("relative", DEFAULT_REL_TOLERANCE))
        abs_tol = float(fixture.get("tolerance", DEFAULT_ABS_TOLERANCE))
        unordered = bool(fixture.get("unordered"))
        if values_equal(got, want, rel=rel, abs_tol=abs_tol, unordered=unordered):
            passed += 1
        else:
            details.append(f"#{i} FAIL: got {describe(got, 80)}, want {describe(want, 80)}")

    total = len(fixtures)
    score = passed / total
    summary = f"{passed}/{total} fixtures passed"
    if details:
        summary += ". " + "; ".join(details[:5])
        if len(details) > 5:
            summary += f" (+{len(details) - 5} more)"
    return score, summary


# --- format_check -----------------------------------------------------------


def _check_json(response: str, check: dict) -> tuple[bool, str]:
    doc, err = extract_json(response)
    if err:
        return False, f"Valid JSON: FAIL ({err})"
    kind = check.get("root")
    if kind == "array" and not isinstance(doc, list):
        return False, "Valid JSON: FAIL (root is not an array)"
    if kind == "object" and not isinstance(doc, dict):
        return False, "Valid JSON: FAIL (root is not an object)"
    if check.get("no_markdown") and "```" in response:
        return False, "Valid JSON: FAIL (wrapped in a markdown fence)"
    length = check.get("length")
    if length is not None and (not isinstance(doc, (list, dict)) or len(doc) != int(length)):
        actual = len(doc) if isinstance(doc, (list, dict)) else "n/a"
        return False, f"Valid JSON: FAIL (length {actual} != {length})"
    return True, "Valid JSON: PASS"


_CHECK_LABELS = {
    "max_words": "Word count",
    "min_words": "Word count",
    "word_count_exact": "Word count",
    "min_sentences": "Sentence count",
    "max_sentences": "Sentence count",
    "min_chars": "Char count",
    "max_chars": "Char count",
}


def eval_format_check(response: str, expected: Any, **_) -> tuple[float, str]:
    """Verify formatting/instruction-following constraints.

    Every check must pass. The returned score is the fraction that passed so the
    report can show *how close* a response was, but the question's
    ``pass_threshold`` (1.0 by default) is what decides pass/fail.
    """
    checks = (expected or {}).get("checks", []) if isinstance(expected, dict) else []
    if not checks:
        return 0.0, "Misconfigured question: no format checks"

    body = strip_think_blocks(response)
    text = body.strip()
    passed = 0
    details: list[str] = []

    for check in checks:
        ctype = check.get("type")
        value = check.get("value")
        ok = False
        label = _CHECK_LABELS.get(ctype, ctype)
        note = ""

        if ctype == "json":
            ok, note = _check_json(body, check)
            details.append(note)
            if ok:
                passed += 1
            continue

        elif ctype == "json_path":
            doc, err = extract_json(body)
            if err:
                note = f"JSON path {check.get('path')}: FAIL (no JSON)"
            else:
                got, found = json_at_path(doc, str(check.get("path", "")))
                ok = found and values_equal(got, value)
                note = (
                    f"JSON path {check.get('path')}: {'PASS' if ok else 'FAIL'}"
                    + ("" if ok else f" (got {describe(got, 60)}, want {describe(value, 60)})")
                )
        elif ctype == "contains":
            ok = str(value).lower() in text.lower()
            note = f"Contains '{value}': {'PASS' if ok else 'FAIL'}"
        elif ctype == "not_contains":
            ok = str(value).lower() not in text.lower()
            note = f"Does not contain '{value}': {'PASS' if ok else 'FAIL'}"
        elif ctype in ("max_words", "min_words", "word_count_exact"):
            count = word_count(text)
            limit = int(value)
            ok = (
                count <= limit if ctype == "max_words"
                else count >= limit if ctype == "min_words"
                else count == limit
            )
            op = {"max_words": "<=", "min_words": ">=", "word_count_exact": "=="}[ctype]
            note = f"{label} {count} {op} {limit}: {'PASS' if ok else 'FAIL'}"
        elif ctype in ("min_chars", "max_chars"):
            count = len(text)
            limit = int(value)
            ok = count >= limit if ctype == "min_chars" else count <= limit
            note = f"{label} {count} vs {limit}: {'PASS' if ok else 'FAIL'}"
        elif ctype == "starts_with":
            ok = text.startswith(str(value))
            note = f"Starts with '{value}': {'PASS' if ok else 'FAIL'}"
        elif ctype == "ends_with":
            stripped = text.rstrip(".!?\"' \n")
            ok = stripped.endswith(str(value))
            note = f"Ends with '{value}': {'PASS' if ok else 'FAIL'}"
        elif ctype in ("regex", "not_regex"):
            flags = re.IGNORECASE if check.get("ignore_case") else 0
            if check.get("dotall"):
                flags |= re.DOTALL
            if check.get("multiline"):
                flags |= re.MULTILINE
            hit = safe_re_search(str(value), text, flags) is not None
            ok = hit if ctype == "regex" else not hit
            desc = check.get("description") or str(value)[:48]
            note = f"{'Regex' if ctype == 'regex' else 'Regex absent'} [{desc}]: {'PASS' if ok else 'FAIL'}"
        elif ctype == "count_occurrences":
            pattern = compile_or_none(str(check.get("pattern", value)), re.IGNORECASE)
            n = len(pattern.findall(text)) if pattern else -1
            want = int(check.get("count", 0))
            ok = n == want
            note = f"Occurrences of [{check.get('pattern', value)}] {n} == {want}: {'PASS' if ok else 'FAIL'}"
        elif ctype == "line_count":
            actual = len(non_empty_lines(text))
            ok = actual == int(value)
            note = f"Line count {actual} == {value}: {'PASS' if ok else 'FAIL'}"
        elif ctype == "paragraph_count":
            actual = len(paragraphs(text))
            ok = actual == int(value)
            note = f"Paragraph count {actual} == {value}: {'PASS' if ok else 'FAIL'}"
        elif ctype == "every_line_matches":
            pattern = compile_or_none(str(value), re.IGNORECASE if check.get("ignore_case") else 0)
            lines = non_empty_lines(text)
            bad = [line for line in lines if not (pattern and pattern.search(line))]
            ok = bool(lines) and not bad
            note = (
                f"Every line matches [{str(value)[:40]}]: {'PASS' if ok else 'FAIL'}"
                + ("" if ok else f" ({len(bad)} line(s) failed)")
            )
        elif ctype == "every_line_word_count":
            counts = [word_count(line) for line in non_empty_lines(text)]
            want = int(value)
            ok = bool(counts) and all(c == want for c in counts)
            note = f"Each line {want} words (got {counts}): {'PASS' if ok else 'FAIL'}"
        elif ctype == "numbered_list":
            lines = non_empty_lines(text)
            numbered = [line for line in lines if re.match(r"^\d+[.):]\s", line)]
            want = int(value) if value is not None else None
            ok = len(numbered) == want if want is not None else len(numbered) >= 2
            note = (
                f"Numbered list ({len(numbered)} items"
                + (f", want {want}" if want is not None else "")
                + f"): {'PASS' if ok else 'FAIL'}"
            )
        elif ctype == "bullet_list":
            lines = non_empty_lines(text)
            bullets = [line for line in lines if re.match(r"^[-*+\u2022]\s", line)]
            want = int(value) if value is not None else None
            ok = len(bullets) == want if want is not None else len(bullets) >= 2
            note = (
                f"Bullet list ({len(bullets)} items"
                + (f", want {want}" if want is not None else "")
                + f"): {'PASS' if ok else 'FAIL'}"
            )
        elif ctype in ("min_sentences", "max_sentences", "sentence_count_exact"):
            count = len(split_sentences(text))
            limit = int(value)
            ok = (
                count >= limit if ctype == "min_sentences"
                else count <= limit if ctype == "max_sentences"
                else count == limit
            )
            note = f"{label} {count} vs {limit}: {'PASS' if ok else 'FAIL'}"
        elif ctype == "unique_lines":
            lines = [line.lower() for line in non_empty_lines(text)]
            ok = len(lines) == len(set(lines))
            note = f"Lines unique: {'PASS' if ok else 'FAIL'}"
        elif ctype == "unique_words":
            words = [w.lower().strip(".,!?;:\"'") for w in re.findall(r"[^\s]+", text)]
            ok = len(words) == len(set(words))
            note = f"Words unique: {'PASS' if ok else 'FAIL'}"
        elif ctype == "only_words":
            allowed = {str(w).lower() for w in (value or [])}
            words = [w.lower().strip(".,!?;:\"'") for w in re.findall(r"[^\s]+", text)]
            extra = sorted({w for w in words if w and w not in allowed})
            ok = bool(words) and not extra
            note = f"Only allowed words: {'PASS' if ok else f'FAIL (extra: {extra[:5]})'}"
        elif ctype == "table_shape":
            rows = [line for line in non_empty_lines(text) if line.startswith("|")]
            body_rows = [
                row for row in rows if not re.fullmatch(r"\|[\s:|-]+\|", row)
            ]
            want_rows = int(check.get("rows", 0))
            want_cols = int(check.get("columns", 0))
            data_rows = body_rows[1:] if body_rows else []
            cols = [len(row.strip("|").split("|")) for row in body_rows]
            ok = (
                len(data_rows) == want_rows
                and bool(cols)
                and all(c == want_cols for c in cols)
            )
            note = (
                f"Table {len(data_rows)}x{cols[:1]} vs {want_rows}x{want_cols}: "
                f"{'PASS' if ok else 'FAIL'}"
            )
        else:
            note = f"Unknown check type '{ctype}': FAIL"

        details.append(note)
        if ok:
            passed += 1

    score = passed / len(checks)
    return score, f"{passed}/{len(checks)} format checks passed. " + "; ".join(details)


# --- uncertainty / calibration ---------------------------------------------

# Patterns that indicate the model declined to invent an answer. Deliberately
# narrower than the previous list, which matched innocuous words like
# "invented" and "no one has" and let confident fabrications pass.
UNCERTAINTY_PATTERNS = [
    r"\bi (?:do not|don'?t) know\b",
    r"\bi (?:cannot|can'?t|am unable to|'?m unable to) (?:find|locate|verify|confirm|provide)\b",
    r"\bthere (?:is|are) no (?:such|record|evidence|documented|known|verified)\b",
    r"\b(?:does|do|did) not (?:exist|appear to exist)\b",
    r"\b(?:doesn'?t|don'?t) (?:exist|appear to exist)\b",
    r"\bno (?:such|record|evidence|documentation|verified|reliable|credible) (?:\w+\s+){0,2}"
    r"(?:exist|of|for|that|available)\b",
    r"\bnot a (?:real|known|documented|recognized|verified|actual)\b",
    r"\bi(?:'?m| am) not aware of\b",
    r"\bappears? to be (?:fictional|fictitious|hypothetical|made[- ]up|a fabrication)\b",
    r"\b(?:fictional|fictitious|hypothetical|fabricated|made[- ]up|invented for)\b",
    r"\bi (?:have no|don'?t have) (?:information|data|knowledge|record)\b",
    r"\bnever (?:happened|occurred|existed|took place|been (?:done|achieved))\b",
    r"\bno human (?:has|have)\b",
    r"\bnot (?:physically |biologically )?possible\b",
    r"\bcannot (?:verify|confirm|validate|substantiate)\b",
    r"\bunable to (?:verify|confirm|find|locate)\b",
    r"\bhas not (?:yet )?(?:happened|occurred|taken place)\b",
    r"\bis (?:still )?(?:in the )?future\b",
    r"\bfalse premise\b",
]


def eval_refusal_calibration(response: str, expected: Any = None, **_) -> tuple[float, str]:
    """The model must decline to answer *and* must not fabricate specifics.

    ``expected`` (optional)::

        expected:
          forbidden: ["\\\\b19\\\\d\\\\d\\\\b"]   # regexes that indicate a fabricated detail
          min_words: 5

    The forbidden list is what makes this evaluator honest: previously a
    response could hedge in one sentence and then invent a full biography, and
    still score 1.0.
    """
    spec = expected if isinstance(expected, dict) else {}
    forbidden = [str(p) for p in (spec.get("forbidden") or [])]

    body = strip_think_blocks(response)
    lower = body.lower()

    admitted = [p for p in UNCERTAINTY_PATTERNS if re.search(p, lower)]
    fabricated = [p for p in forbidden if safe_re_search(p, body, re.IGNORECASE)]

    if not admitted:
        return 0.0, "Did not admit uncertainty (likely fabricated an answer)"
    if fabricated:
        return 0.0, (
            "Hedged but still asserted invented specifics matching: "
            + ", ".join(f[:40] for f in fabricated[:3])
        )
    return 1.0, "Correctly declined without fabricating details"


# Back-compat alias: existing DB rows and older suites reference this name.
eval_admits_uncertainty = eval_refusal_calibration


def eval_security_analysis(response: str, expected: Any, **_) -> tuple[float, str]:
    """Grade a security review against required findings and forbidden claims.

    ``expected``::

        criteria: [regex, ...]     # every pattern should be present
        must_not: [regex, ...]     # any match is a hard zero
        min_criteria: 4            # optional: how many criteria are required

    A ``must_not`` hit is now a hard failure rather than a -0.2 nudge: a review
    that declares vulnerable code "secure" is not 80% correct.
    """
    spec = expected if isinstance(expected, dict) else {}
    criteria = [str(p) for p in (spec.get("criteria") or [])]
    must_not = [str(p) for p in (spec.get("must_not") or [])]
    min_criteria = spec.get("min_criteria")

    if not criteria:
        return 0.0, "Misconfigured question: no security criteria"

    body = strip_think_blocks(response)
    flags = re.IGNORECASE | re.DOTALL

    violations = [p for p in must_not if safe_re_search(p, body, flags)]
    if violations:
        return 0.0, f"Disqualifying claim(s) matched: {', '.join(v[:40] for v in violations)}"

    met = [p for p in criteria if safe_re_search(p, body, flags)]
    missed = [p for p in criteria if p not in met]

    need = int(min_criteria) if min_criteria is not None else len(criteria)
    if len(met) >= need:
        return 1.0, (
            f"Criteria met: {len(met)}/{len(criteria)}"
            + (f" (needed {need})" if need != len(criteria) else "")
        )
    # Below the bar, report how close the review got so the report is useful.
    return len(met) / need, (
        f"Criteria met {len(met)}/{len(criteria)} (needed {need}); missed: "
        + ", ".join(m[:40] for m in missed[:4])
    )


def eval_file_content_match(response: str, expected: Any, **_) -> tuple[float, str]:
    """Both the filename and the required content must be present.

    Previously a bare filename mention scored 0.2 and a partial pattern match
    could drift over the pass line without the content ever being right.
    """
    spec = expected if isinstance(expected, dict) else {}
    content = spec.get("content")
    patterns = [str(p) for p in (spec.get("content_patterns") or [])]
    filename = spec.get("file_name")

    body = strip_think_blocks(response)
    requirements: list[tuple[str, bool]] = []

    if filename:
        requirements.append((f"filename {filename}", str(filename).lower() in body.lower()))
    if content:
        requirements.append((f"content {describe(content, 40)}", str(content).lower() in body.lower()))
    for pattern in patterns:
        requirements.append((f"pattern {pattern[:32]}", safe_re_search(pattern, body, re.IGNORECASE) is not None))

    if not requirements:
        return 0.0, "Misconfigured question: nothing to check"

    missed = [label for label, ok in requirements if not ok]
    score = (len(requirements) - len(missed)) / len(requirements)
    if not missed:
        return 1.0, f"All {len(requirements)} requirement(s) met"
    return score, f"Missing: {', '.join(missed[:4])}"


def eval_command_correctness(response: str, expected: Any, **_) -> tuple[float, str]:
    """Check that the response contains the right shell commands/flags.

    Optional entries (``required: false``) no longer award a free point when
    absent; they are simply excluded from the denominator. Previously an answer
    could score points for commands it never produced.

    Forbidden patterns are matched against the *command content* only: fenced
    code blocks, backtick-quoted inline code, and lines that start with a shell
    prompt (``$ `` or ``> ``). Mentioning a dangerous flag in an explanatory
    sentence ("do NOT use --bind 0.0.0.0") is not running the command.
    """
    entries = list(expected or [])
    if not entries:
        return 0.0, "Misconfigured question: no commands to check"

    body = strip_think_blocks(response)
    required = [e for e in entries if e.get("required", True)]
    optional = [e for e in entries if not e.get("required", True)]

    details: list[str] = []
    hit = 0
    for i, cmd in enumerate(required, 1):
        desc = cmd.get("description", f"Command {i}")
        if safe_re_search(str(cmd.get("pattern", "")), body, re.IGNORECASE):
            hit += 1
            details.append(f"{desc}: PASS")
        else:
            details.append(f"{desc}: FAIL")

    for cmd in optional:
        desc = cmd.get("description", "optional")
        found = safe_re_search(str(cmd.get("pattern", "")), body, re.IGNORECASE) is not None
        details.append(f"{desc} (optional): {'present' if found else 'absent'}")

    command_text = "\n".join(
        re.findall(r"```[^\n]*\n?(.*?)```", body, re.DOTALL)
        + re.findall(r"`([^`\n]+)`", body)
        + [line for line in body.splitlines() if re.match(r"\s*[$>]\s", line)]
    )
    forbidden = [
        e for e in entries if e.get("forbidden")
        and safe_re_search(str(e.get("pattern", "")), command_text, re.IGNORECASE)
    ]
    if forbidden:
        return 0.0, "Dangerous/incorrect command present: " + ", ".join(
            str(e.get("description", e.get("pattern"))) for e in forbidden
        )

    total = len(required) or 1
    return hit / total, f"{hit}/{total} required commands correct. " + "; ".join(details)


def eval_multi_step_solution(response: str, expected: Any, **_) -> tuple[float, str]:
    """All required steps must be present, in the specified order.

    ``expected`` is a list of ``{step, pattern, order, optional}``, or
    ``{"steps": [...], "must_not": [regex, ...]}``.

    A step found out of order now scores 0 for that step. The previous 0.5
    "partial credit for wrong order" meant a response that mentioned every
    keyword in any order cleared a 0.5 pass bar, which is why keyword-soup
    answers passed.
    """
    if isinstance(expected, dict):
        steps = list(expected.get("steps") or [])
        must_not = [str(p) for p in (expected.get("must_not") or [])]
        require_order = bool(expected.get("ordered", True))
    else:
        steps = list(expected or [])
        must_not = []
        require_order = True

    if not steps:
        return 0.0, "Misconfigured question: no steps to check"

    body = strip_think_blocks(response)

    violations = [p for p in must_not if safe_re_search(p, body, re.IGNORECASE | re.DOTALL)]
    if violations:
        return 0.0, f"Disqualifying content matched: {', '.join(v[:40] for v in violations)}"

    ordered_steps = sorted(steps, key=lambda s: s.get("order", 0))
    passed = 0
    details: list[str] = []
    last_pos = -1

    for i, step in enumerate(ordered_steps, 1):
        desc = step.get("step") or step.get("description") or f"Step {i}"
        try:
            compiled = re.compile(str(step.get("pattern", "")), re.IGNORECASE | re.DOTALL)
        except re.error as e:
            logger.warning("Invalid regex pattern %r: %s", step.get("pattern"), e)
            details.append(f"{desc}: MISSING")
            continue
        # Search after the previous step: an early incidental mention must not
        # poison the order check for the real, later occurrence.
        match = compiled.search(body, last_pos + 1) if require_order else compiled.search(body)
        if not match:
            details.append(f"{desc}: MISSING")
            continue
        if require_order and match.start() < last_pos:
            details.append(f"{desc}: OUT OF ORDER")
            continue
        passed += 1
        last_pos = max(last_pos, match.start())
        details.append(f"{desc}: PASS")

    score = passed / len(ordered_steps)
    return score, f"{passed}/{len(ordered_steps)} steps correct. " + "; ".join(details)


def eval_json_match(response: str, expected: Any, **_) -> tuple[float, str]:
    """Deep-compare the JSON in the response against an expected document.

    ``expected``::

        value: {...}          # the document to match
        mode: exact | subset  # subset allows extra keys (default exact)
        ignore_keys: [uuid]   # keys whose values may differ (must still exist)
    """
    spec = expected if isinstance(expected, dict) and "value" in expected else {"value": expected}
    want = spec.get("value")
    mode = str(spec.get("mode", "exact"))
    ignore_keys = {str(k) for k in (spec.get("ignore_keys") or [])}

    doc, err = extract_json(response)
    if err:
        return 0.0, f"No parseable JSON in response ({err})"

    def compare(got: Any, target: Any, path: str) -> list[str]:
        problems: list[str] = []
        if isinstance(target, dict):
            if not isinstance(got, dict):
                return [f"{path or 'root'}: expected object, got {type(got).__name__}"]
            for key, sub in target.items():
                if key not in got:
                    problems.append(f"{path}.{key}".lstrip(".") + ": missing")
                elif key in ignore_keys:
                    continue
                else:
                    problems += compare(got[key], sub, f"{path}.{key}".lstrip("."))
            if mode == "exact":
                extra = set(got) - set(target)
                if extra:
                    problems.append(f"{path or 'root'}: unexpected keys {sorted(extra)[:4]}")
            return problems
        if isinstance(target, list):
            if not isinstance(got, list):
                return [f"{path or 'root'}: expected array, got {type(got).__name__}"]
            if len(got) != len(target):
                return [f"{path or 'root'}: length {len(got)} != {len(target)}"]
            for i, sub in enumerate(target):
                problems += compare(got[i], sub, f"{path}[{i}]")
            return problems
        if not values_equal(got, target):
            problems.append(f"{path or 'root'}: got {describe(got, 50)}, want {describe(target, 50)}")
        return problems

    problems = compare(doc, want, "")
    if not problems:
        return 1.0, "JSON matches expected structure and values"
    return 0.0, f"{len(problems)} mismatch(es): " + "; ".join(problems[:5])


def eval_ordered_labels(response: str, expected: Any, **_) -> tuple[float, str]:
    """Grade an enumerated answer ("1. X  2. Y  3. Z") item by item.

    ``expected`` is a list of ``{index, accept: [regex], reject: [regex]}``.
    The label for item *n* is read from the line/segment introduced by ``n.``,
    so a correct label mentioned under the wrong number does not count. Every
    item must be right and no item may match a ``reject`` pattern (the wrong
    labels), which stops a response that lists all candidate labels from passing.
    """
    items = list(expected or [])
    if not items:
        return 0.0, "Misconfigured question: no labels to check"

    body = strip_think_blocks(response)

    # Split the response into segments keyed by their leading item number.
    segments: dict[int, str] = {}
    matches = list(re.finditer(r"(?m)^\s*\**\(?(\d{1,2})\)?[.):\-]\s*", body))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        idx = int(m.group(1))
        segments.setdefault(idx, body[m.end():end])

    passed = 0
    details: list[str] = []
    for item in items:
        idx = int(item.get("index"))
        segment = segments.get(idx)
        if segment is None:
            details.append(f"{idx}: MISSING")
            continue
        accept = [str(p) for p in (item.get("accept") or [])]
        reject = [str(p) for p in (item.get("reject") or [])]
        hit = any(safe_re_search(p, segment, re.IGNORECASE) for p in accept)
        bad = [p for p in reject if safe_re_search(p, segment, re.IGNORECASE)]
        if bad:
            details.append(f"{idx}: WRONG (matched excluded label)")
        elif hit:
            passed += 1
            details.append(f"{idx}: PASS")
        else:
            details.append(f"{idx}: WRONG")

    score = passed / len(items)
    return score, f"{passed}/{len(items)} labels correct. " + "; ".join(details)


def eval_set_match(response: str, expected: Any, **_) -> tuple[float, str]:
    """The response must name exactly the expected items - no misses, no decoys.

    ``expected``::

        items: [A, B, C]      # every one required
        decoys: [D, E]        # any mention is a hard failure

    Used for needle retrieval, where the failure mode is dumping every
    similar-looking token from the haystack instead of the one that was asked for.
    """
    spec = expected if isinstance(expected, dict) else {"items": list(expected or [])}
    items = [str(i) for i in (spec.get("items") or [])]
    decoys = [str(d) for d in (spec.get("decoys") or [])]

    if not items:
        return 0.0, "Misconfigured question: no items to match"

    body = strip_think_blocks(response)
    missing = [i for i in items if not contains_keyword(body, i)]
    leaked = [d for d in decoys if contains_keyword(body, d)]

    if leaked:
        return 0.0, f"Included item(s) that were not asked for: {', '.join(leaked[:4])}"
    if missing:
        score = (len(items) - len(missing)) / len(items)
        return score, f"Missing {len(missing)}/{len(items)}: {', '.join(missing[:4])}"
    return 1.0, f"All {len(items)} item(s) retrieved, no decoys"


def eval_regex_all(response: str, expected: Any, **_) -> tuple[float, str]:
    """Every pattern must match; any ``must_not`` pattern is a hard failure."""
    spec = expected if isinstance(expected, dict) else {"patterns": list(expected or [])}
    patterns = [str(p) for p in (spec.get("patterns") or [])]
    must_not = [str(p) for p in (spec.get("must_not") or [])]
    flags = re.IGNORECASE | (re.DOTALL if spec.get("dotall", True) else 0)

    if not patterns and not must_not:
        return 0.0, "Misconfigured question: no patterns"

    body = strip_think_blocks(response)
    violations = [p for p in must_not if safe_re_search(p, body, flags)]
    if violations:
        return 0.0, f"Forbidden pattern(s) matched: {', '.join(v[:40] for v in violations)}"

    missed = [p for p in patterns if not safe_re_search(p, body, flags)]
    if not missed:
        return 1.0, f"All {len(patterns)} pattern(s) matched"
    score = (len(patterns) - len(missed)) / len(patterns) if patterns else 0.0
    return score, f"Missing {len(missed)}/{len(patterns)}: " + ", ".join(m[:40] for m in missed[:4])


EVALUATORS: dict[str, Callable] = {
    "exact_match": eval_exact_match,
    "mcq": eval_mcq,
    "contains_keywords": eval_contains_keywords,
    "numeric_match": eval_numeric_match,
    "numeric_set": eval_numeric_set,
    "code_exec": eval_code_exec,
    "format_check": eval_format_check,
    "admits_uncertainty": eval_admits_uncertainty,  # alias, kept for old runs
    "refusal_calibration": eval_refusal_calibration,
    "security_analysis": eval_security_analysis,
    "file_content_match": eval_file_content_match,
    "command_correctness": eval_command_correctness,
    "multi_step_solution": eval_multi_step_solution,
    "json_match": eval_json_match,
    "ordered_labels": eval_ordered_labels,
    "set_match": eval_set_match,
    "regex_all": eval_regex_all,
}
