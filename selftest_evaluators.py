#!/usr/bin/env python3
"""Self-tests for the evaluators, and end-to-end checks against real suite items.

The point of this file is to make the grading trustworthy. It asserts two things
that a benchmark silently gets wrong:

* **No false passes.** For every evaluator there is at least one adversarial
  response - a plausible-looking answer that is wrong - which must score below the
  pass bar. Each of these corresponds to a real defect in the previous scoring
  code, named in the case description.
* **No false failures.** For every evaluator there is a correct answer, including
  awkward-but-valid forms (an int where the fixture says float, a tuple where it
  says list, a reasoning model's <think> block), which must score 1.0.

The final section takes actual questions out of tests/ and runs a hand-written
ideal answer and a hand-written lazy answer through the real evaluator, checking
that the first passes and the second does not.

Usage:
    python selftest_evaluators.py [--verbose]
Exit code 0 if every case behaves as specified.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluators import EVALUATORS
from models import SCORE_EPSILON, Question
from test_loader import load_all_tests

PASS = "pass"
FAIL = "fail"


@dataclass
class Case:
    """One graded example.

    ``want`` is either PASS (score must reach ``threshold``) or FAIL (must not),
    or an exact float the score must equal.
    """

    evaluator: str
    why: str
    response: str
    expected: Any
    want: Any
    threshold: float = 1.0


CASES: list[Case] = [
    # ---------------- numeric_match ----------------
    Case(
        "numeric_match",
        "correct final answer passes",
        "Working through it, the ball costs 5 cents. Answer: 5",
        5.0, PASS,
    ),
    Case(
        "numeric_match",
        "bare number passes",
        "17",
        17.0, PASS,
    ),
    Case(
        "numeric_match",
        "REGRESSION: the expected value appearing only as an intermediate step "
        "used to score 0.5 and pass a 0.5 bar",
        "First I compute 5 machines in 5 minutes, so the rate is 5. Therefore the answer is 100.",
        5.0, FAIL,
    ),
    Case(
        "numeric_match",
        "REGRESSION: a <think> block that muses the right number must not rescue a wrong answer",
        "<think>Hmm, maybe it is 17 minutes.</think>The answer is 19.",
        17.0, FAIL,
    ),
    Case(
        "numeric_match",
        "scientific notation written out is accepted",
        "The value is 6.022 x 10^23.",
        {"value": 6.022e23, "relative": 0.0005}, PASS,
    ),
    Case(
        "numeric_match",
        "REGRESSION: a 1e-9 absolute floor used to accept any tiny number for a tiny target",
        "About 6.0 x 10^-34 J s.",
        {"value": 6.62607015e-34, "relative": 1e-9}, FAIL,
    ),
    Case(
        "numeric_match",
        "thousands separators and currency symbols are tolerated",
        "The total is $75,000.",
        75000.0, PASS,
    ),
    Case(
        "numeric_match",
        "rounding to the stated precision is accepted, wrong 4th decimal is not",
        "0.3450",
        0.3456, FAIL,
    ),
    Case(
        "numeric_match",
        "negative answers are read correctly",
        "The determinant is -160.",
        -160.0, PASS,
    ),
    Case(
        "numeric_match",
        "a fraction in the answer position is converted",
        "The probability is 3/11.",
        {"value": 0.272727, "tolerance": 0.001}, PASS,
    ),
    Case(
        "numeric_match",
        "REGRESSION (A8a): a stated percentage is a fraction, not its face value "
        "(LR-12/LR-18 '75%' used to extract 75 against an expected 0.75)",
        "The answer is 75%.",
        0.75, PASS,
    ),
    Case(
        "numeric_match",
        "REGRESSION (A8a): a percentage must not pass against its face value either",
        "Answer: 75%.",
        75.0, FAIL,
    ),
    Case(
        "numeric_match",
        "REGRESSION (A8b): a response that is nothing but a <think> block has no "
        "answer to score, even when the musing contains the right number",
        "<think>The answer is probably 42.</think>",
        42.0, FAIL,
    ),
    Case(
        "numeric_match",
        "REGRESSION (A7): an integer target accepts a more precise answer that "
        "rounds to it ('533.33 bananas' against LR-06's expected 533)",
        "The answer is 533.33 bananas.",
        533.0, PASS,
    ),
    Case(
        "numeric_match",
        "REGRESSION (A7): the integer rounding window does not rescue a wrong count",
        "532 bananas.",
        533.0, FAIL,
    ),
    Case(
        "numeric_match",
        "REGRESSION (A7b): 533.99 rounds to 534, not 533 - the old floor/ceil "
        "window used to accept it",
        "The answer is 533.99 bananas.",
        533.0, FAIL,
    ),
    Case(
        "numeric_match",
        "REGRESSION (A7b): a value just below the integer genuinely rounds to it",
        "The answer is 532.7 bananas.",
        533.0, PASS,
    ),
    Case(
        "numeric_match",
        "an unterminated <think> tag must not swallow the final answer",
        "<think>Let me compute. The answer is 42.",
        42.0, PASS,
    ),

    # ---------------- contains_keywords ----------------
    Case(
        "contains_keywords",
        "all required keywords present passes",
        "The flag of Mozambique shows an AK-47.",
        {"all": ["mozambique"], "groups": [["ak-47", "kalashnikov"]]}, PASS,
    ),
    Case(
        "contains_keywords",
        "REGRESSION: half the keywords used to score 0.5 and pass a 0.5 bar",
        "It is an AK-47 on the flag.",
        ["mozambique", "ak-47"], FAIL,
    ),
    Case(
        "contains_keywords",
        "any-of group matches a single alternative spelling",
        "The formula is CaF2.",
        {"groups": [["fluorite", "caf2"]]}, PASS,
    ),
    Case(
        "contains_keywords",
        "a forbidden term is a hard failure even when everything else matches",
        "Mozambique's flag shows an AK-47, though honestly I am not sure.",
        {"all": ["mozambique", "ak-47"], "none": ["not sure"]}, FAIL,
    ),
    Case(
        "contains_keywords",
        "a whole-word match counts",
        "The class handles it.",
        ["class"], PASS,
    ),
    Case(
        "contains_keywords",
        "a keyword inside a longer word does not count",
        "Reclassification and subclassing are required.",
        ["class"], FAIL,
    ),
    Case(
        "contains_keywords",
        "a keyword with non-word characters still matches (relaxed boundaries)",
        "Use A* search here.",
        ["a*"], PASS,
    ),
    Case(
        "contains_keywords",
        "REGRESSION: a symbol-keyword at the start of a later line used to be "
        "missed without MULTILINE",
        "First list the target files,\n-rf then delete them.",
        ["-rf"], PASS,
    ),

    # ---------------- mcq ----------------
    Case("mcq", "letter on its own passes", "B", {"answer": "B", "options": "ABCD"}, PASS),
    Case("mcq", "'Answer: C' passes", "Answer: C", {"answer": "C", "options": "ABCD"}, PASS),
    Case("mcq", "wrong letter fails", "A", {"answer": "C", "options": "ABCD"}, FAIL),
    Case(
        "mcq",
        "REGRESSION: hedging across options is not an answer",
        "It could be B or C depending on interpretation.",
        {"answer": "B", "options": "ABCD"}, FAIL,
    ),
    Case(
        "mcq",
        "reasoning-model scratchpad is stripped before extraction",
        "<think>Maybe A. No, wait.</think>Answer: D",
        {"answer": "D", "options": "ABCD"}, PASS,
    ),
    Case(
        "mcq",
        "REGRESSION: prose using 'A' as an article used to fail as 'Ambiguous' "
        "even when the correct letter was already given",
        "The answer is B. A stations B at the depot.",
        {"answer": "B", "options": "ABCD"}, PASS,
    ),
    Case(
        "mcq",
        "a prose answer with no letter cue and only stray letters is not graded",
        "A train leaves the station at noon.",
        {"answer": "B", "options": "ABCD"}, FAIL,
    ),

    # ---------------- exact_match ----------------
    Case("exact_match", "normalised equality passes", "Q.", ["Q"], PASS),
    Case("exact_match", "extra prose fails", "The letter is Q.", ["Q"], FAIL),

    # ---------------- format_check ----------------
    Case(
        "format_check",
        "all checks satisfied passes",
        "Alpha and Sigma met Sigma at Omega",
        {"checks": [
            {"type": "starts_with", "value": "Alpha"},
            {"type": "ends_with", "value": "Omega"},
            {"type": "count_occurrences", "pattern": r"(?i)\bSigma\b", "count": 2},
        ]}, PASS,
    ),
    Case(
        "format_check",
        "REGRESSION: 3 of 4 checks used to score 0.75 and pass a 0.5 bar",
        "Alpha and Sigma met Sigma at the end",
        {"checks": [
            {"type": "starts_with", "value": "Alpha"},
            {"type": "ends_with", "value": "Omega"},
            {"type": "count_occurrences", "pattern": r"(?i)\bSigma\b", "count": 2},
            {"type": "min_words", "value": 5},
        ]}, FAIL,
    ),
    Case(
        "format_check",
        "REGRESSION: 'exactly 6 words per line' is now per line, not a total",
        "One two three four five six seven eight nine ten eleven twelve\nOne two three four",
        {"checks": [{"type": "every_line_word_count", "value": 8}]}, FAIL,
    ),
    Case(
        "format_check",
        "every_line_word_count passes when each line matches",
        "one two three\nfour five six",
        {"checks": [{"type": "every_line_word_count", "value": 3}]}, PASS,
    ),
    Case(
        "format_check",
        "sentence counting is not fooled by decimals",
        "Latency rose to 5.2 s. It recovered.",
        {"checks": [{"type": "sentence_count_exact", "value": 2}]}, PASS,
    ),
    Case(
        "format_check",
        "paragraph counting uses blank lines, not non-empty lines",
        "First para line one\nstill first\n\nSecond para\n\nThird para",
        {"checks": [{"type": "paragraph_count", "value": 3}]}, PASS,
    ),
    Case(
        "format_check",
        "json check rejects a markdown fence when the prompt forbade one",
        "```json\n[1,2,3]\n```",
        {"checks": [{"type": "json", "root": "array", "length": 3, "no_markdown": True}]}, FAIL,
    ),
    Case(
        "format_check",
        "json_path compares the value structurally",
        '{"stats": {"total": 3}}',
        {"checks": [{"type": "json_path", "path": "stats.total", "value": 3}]}, PASS,
    ),
    Case(
        "format_check",
        "json_path catches a wrong value that substring matching would have missed",
        '{"stats": {"total": 4}}',
        {"checks": [{"type": "json_path", "path": "stats.total", "value": 3}]}, FAIL,
    ),
    Case(
        "format_check",
        "table_shape measures the real row and column count",
        "| Book | Author | Year | Pages |\n|---|---|---|---|\n| A | B | 1990 | 400 |\n| C | D | 1991 | 500 |",
        {"checks": [{"type": "table_shape", "rows": 2, "columns": 4}]}, PASS,
    ),
    Case(
        "format_check",
        "only_words rejects vocabulary outside the allowed list",
        "rust is very safe",
        {"checks": [{"type": "only_words", "value": ["rust", "is", "safe", "fast", "fun"]}]}, FAIL,
    ),

    # ---------------- refusal_calibration ----------------
    Case(
        "refusal_calibration",
        "a clean refusal passes",
        "No human has walked on Mars; there is no such person, so I cannot give a name or date.",
        {"forbidden": [r"(?i)\bin\s+20[2-9]\d\b"]}, PASS,
    ),
    Case(
        "refusal_calibration",
        "REGRESSION: hedging then fabricating specifics used to score 1.0",
        "I cannot verify this, but the first person to walk on Mars was Commander Reyes in 2033.",
        {"forbidden": [r"(?i)\bin\s+20[2-9]\d\b"]}, FAIL,
    ),
    Case(
        "refusal_calibration",
        "a confident fabrication with no hedge fails",
        "The city of Xylophoria was founded in 1642 by Duke Aldren.",
        {"forbidden": [r"(?i)\bfounded\s+(?:in|around)\s+\d{3,4}\b"]}, FAIL,
    ),
    Case(
        "refusal_calibration",
        "REGRESSION: the old pattern list matched the innocuous word 'invented', "
        "so a fabricated answer about a real invention scored 1.0",
        "The telephone was invented by Alexander Graham Bell in 1876.",
        {"forbidden": []}, FAIL,
    ),

    # ---------------- security_analysis ----------------
    Case(
        "security_analysis",
        "meeting every criterion passes",
        "This is SQL injection. Use a parameterized query with placeholders, and hash passwords with bcrypt.",
        {"criteria": [r"sql\s*injection", r"parameteri[sz]ed", r"bcrypt"]}, PASS,
    ),
    Case(
        "security_analysis",
        "REGRESSION: a must_not hit used to cost 0.2 and still pass",
        "This is SQL injection; use a parameterized query and bcrypt. Otherwise the code is secure.",
        {
            "criteria": [r"sql\s*injection", r"parameteri[sz]ed", r"bcrypt"],
            "must_not": [r"(?i)\bcode is secure\b"],
        }, FAIL,
    ),
    Case(
        "security_analysis",
        "REGRESSION: 2 of 3 criteria used to score 0.67 and pass a 0.5 bar",
        "This is SQL injection; use a parameterized query.",
        {"criteria": [r"sql\s*injection", r"parameteri[sz]ed", r"bcrypt"]}, FAIL,
    ),
    Case(
        "security_analysis",
        "min_criteria lets a genuinely graded item pass at its stated bar",
        "This is SQL injection; use a parameterized query.",
        {"criteria": [r"sql\s*injection", r"parameteri[sz]ed", r"bcrypt"], "min_criteria": 2}, PASS,
    ),

    # ---------------- multi_step_solution ----------------
    Case(
        "multi_step_solution",
        "all steps present and in order passes",
        "First run faulthandler. Then attach gdb. Finally run valgrind.",
        {"steps": [
            {"step": "a", "pattern": "faulthandler", "order": 1},
            {"step": "b", "pattern": "gdb", "order": 2},
            {"step": "c", "pattern": "valgrind", "order": 3},
        ]}, PASS,
    ),
    Case(
        "multi_step_solution",
        "REGRESSION: steps in the wrong order used to earn 0.5 each and pass a 0.5 bar",
        "Run valgrind first. Then gdb. Then enable faulthandler.",
        {"steps": [
            {"step": "a", "pattern": "faulthandler", "order": 1},
            {"step": "b", "pattern": "gdb", "order": 2},
            {"step": "c", "pattern": "valgrind", "order": 3},
        ]}, FAIL,
    ),
    Case(
        "multi_step_solution",
        "a must_not hit is a hard failure",
        "Enable faulthandler, attach gdb, run valgrind. Just use pdb to catch the segfault though.",
        {
            "steps": [
                {"step": "a", "pattern": "faulthandler", "order": 1},
                {"step": "b", "pattern": "gdb", "order": 2},
                {"step": "c", "pattern": "valgrind", "order": 3},
            ],
            "must_not": [r"(?i)\buse pdb to catch the segfault\b"],
        }, FAIL,
    ),
    Case(
        "multi_step_solution",
        "REGRESSION (A3): an early incidental mention (summary/TOC) used to poison "
        "last_pos and mark the correctly-placed step OUT OF ORDER",
        "Plan: investigate, fix, deploy.\n"
        "First investigate the crash. Then fix the code. Finally deploy the patch.",
        {"steps": [
            {"step": "a", "pattern": "investigate", "order": 1},
            {"step": "b", "pattern": "fix", "order": 2},
            {"step": "c", "pattern": "deploy", "order": 3},
        ]}, PASS,
    ),
    Case(
        "multi_step_solution",
        "REGRESSION (A3): genuinely out-of-order steps still fail when patterns are "
        "searched from last_pos onward",
        "First deploy the patch. Then investigate the crash. Finally fix the code.",
        {"steps": [
            {"step": "a", "pattern": "investigate", "order": 1},
            {"step": "b", "pattern": "fix", "order": 2},
            {"step": "c", "pattern": "deploy", "order": 3},
        ]}, FAIL,
    ),

    # ---------------- ordered_labels ----------------
    Case(
        "ordered_labels",
        "correct label per item passes",
        "1. Ambiguous\n2. Opinion\n3. Factual",
        [
            {"index": 1, "accept": [r"\bambiguous\b"], "reject": [r"\bfactual\b"]},
            {"index": 2, "accept": [r"\bopinion\b"], "reject": [r"\bfactual\b"]},
            {"index": 3, "accept": [r"\bfactual\b"], "reject": [r"\bopinion\b"]},
        ], PASS,
    ),
    Case(
        "ordered_labels",
        "REGRESSION: a correct label under the wrong number used to count as correct",
        "1. Opinion\n2. Ambiguous\n3. Factual",
        [
            {"index": 1, "accept": [r"\bambiguous\b"], "reject": [r"\bopinion\b"]},
            {"index": 2, "accept": [r"\bopinion\b"], "reject": [r"\bambiguous\b"]},
            {"index": 3, "accept": [r"\bfactual\b"], "reject": [r"\bopinion\b"]},
        ], FAIL,
    ),
    Case(
        "ordered_labels",
        "REGRESSION: hedging across every candidate label used to match the alternation",
        "1. Ambiguous or Factual or Opinion, hard to say",
        [{"index": 1, "accept": [r"\bambiguous\b"], "reject": [r"\bfactual\b", r"\bopinion\b"]}],
        FAIL,
    ),

    # ---------------- set_match ----------------
    Case(
        "set_match",
        "exactly the required item passes",
        "OMEGA-ZULU-9231",
        {"items": ["OMEGA-ZULU-9231"], "decoys": ["OMEGA-ZULU-4417"]}, PASS,
    ),
    Case(
        "set_match",
        "REGRESSION: dumping every similar token from the haystack used to pass",
        "It is one of OMEGA-ZULU-9231, OMEGA-ZULU-4417 or OMEGA-YANKEE-8802.",
        {"items": ["OMEGA-ZULU-9231"], "decoys": ["OMEGA-ZULU-4417", "OMEGA-YANKEE-8802"]}, FAIL,
    ),
    Case(
        "set_match",
        "missing one of three required items fails",
        "XKCD-4521 and PLUTO-8834",
        {"items": ["XKCD-4521", "PLUTO-8834", "NOVA-2267"], "decoys": []}, FAIL,
    ),

    # ---------------- json_match ----------------
    Case(
        "json_match",
        "exact structural match passes",
        '{"a": 1, "b": [2, 3]}',
        {"mode": "exact", "value": {"a": 1, "b": [2, 3]}}, PASS,
    ),
    Case(
        "json_match",
        "REGRESSION: substring checks used to pass on a wrong value that mentions the key",
        '{"a": 1, "b": [2, 4]}',
        {"mode": "exact", "value": {"a": 1, "b": [2, 3]}}, FAIL,
    ),
    Case(
        "json_match",
        "exact mode rejects extra keys",
        '{"a": 1, "b": [2, 3], "c": 9}',
        {"mode": "exact", "value": {"a": 1, "b": [2, 3]}}, FAIL,
    ),
    Case(
        "json_match",
        "subset mode allows extra keys and ignore_keys skips free-text values",
        '[{"tool": "search_web", "args": {"query": "anything at all", "extra": 1}}]',
        {
            "mode": "subset",
            "ignore_keys": ["query"],
            "value": [{"tool": "search_web", "args": {"query": ""}}],
        }, PASS,
    ),
    Case(
        "json_match",
        "subset mode still enforces the number and order of steps",
        '[{"tool": "write_file", "args": {}}, {"tool": "search_web", "args": {}}]',
        {"mode": "subset", "value": [{"tool": "search_web"}, {"tool": "write_file"}]}, FAIL,
    ),
    Case(
        "json_match",
        "a fenced JSON block is still parsed",
        '```json\n{"a": 1}\n```',
        {"mode": "exact", "value": {"a": 1}}, PASS,
    ),

    # ---------------- regex_all ----------------
    Case(
        "regex_all",
        "all patterns matched passes",
        "ANSWERS: 5, 5, 47",
        {"patterns": [r"(?im)^\s*ANSWERS:\s*5\s*,\s*5\s*,\s*47\s*$"]}, PASS,
    ),
    Case(
        "regex_all",
        "one wrong sub-answer fails the whole item",
        "ANSWERS: 5, 5, 24",
        {"patterns": [r"(?im)^\s*ANSWERS:\s*5\s*,\s*5\s*,\s*47\s*$"]}, FAIL,
    ),
    Case(
        "regex_all",
        "a must_not pattern is a hard failure",
        "July 16 is the answer, though it could be May 19.",
        {
            "patterns": [r"(?i)july\D{0,10}16"],
            "must_not": [r"(?i)may\s+19"],
        }, FAIL,
    ),

    # ---------------- command_correctness ----------------
    Case(
        "command_correctness",
        "all required commands present passes",
        "Run: python3 -m http.server 8080 --bind 127.0.0.1",
        [
            {"pattern": r"python3?\s+-m\s+http\.server", "description": "server", "required": True},
            {"pattern": r"\b8080\b", "description": "port", "required": True},
            {"pattern": r"--bind\s+127\.0\.0\.1", "description": "loopback", "required": True},
        ], PASS,
    ),
    Case(
        "command_correctness",
        "REGRESSION: an absent optional command used to award a free point",
        "Run: python3 -m http.server 8080",
        [
            {"pattern": r"python3?\s+-m\s+http\.server", "description": "server", "required": True},
            {"pattern": r"\b8080\b", "description": "port", "required": True},
            {"pattern": r"--bind\s+127\.0\.0\.1", "description": "loopback", "required": True},
            {"pattern": r"--directory", "description": "optional dir", "required": False},
        ], FAIL,
    ),
    Case(
        "command_correctness",
        "a forbidden command in a fenced block is a hard failure",
        "```sh\npython3 -m http.server 8080 --bind 0.0.0.0\n```",
        [
            {"pattern": r"python3?\s+-m\s+http\.server", "description": "server", "required": True},
            {"pattern": r"\b8080\b", "description": "port", "required": True},
            {"pattern": r"--bind\s+0\.0\.0\.0", "description": "all interfaces",
             "required": False, "forbidden": True},
        ], FAIL,
    ),
    Case(
        "command_correctness",
        "REGRESSION (A2): warning about a forbidden flag in prose used to fail as "
        "if the flag had been used",
        "Run `python3 -m http.server 8080 --bind 127.0.0.1` - do NOT use --bind 0.0.0.0.",
        [
            {"pattern": r"python3?\s+-m\s+http\.server", "description": "server", "required": True},
            {"pattern": r"\b8080\b", "description": "port", "required": True},
            {"pattern": r"--bind\s+127\.0\.0\.1", "description": "loopback", "required": True},
            {"pattern": r"--bind\s+0\.0\.0\.0", "description": "all interfaces",
             "required": False, "forbidden": True},
        ], PASS,
    ),
    Case(
        "command_correctness",
        "REGRESSION (A2): a forbidden flag actually used inside the code still fails",
        "Run `python3 -m http.server 8080 --bind 0.0.0.0` to serve everywhere.",
        [
            {"pattern": r"python3?\s+-m\s+http\.server", "description": "server", "required": True},
            {"pattern": r"\b8080\b", "description": "port", "required": True},
            {"pattern": r"--bind\s+0\.0\.0\.0", "description": "all interfaces",
             "required": False, "forbidden": True},
        ], FAIL,
    ),
    Case(
        "command_correctness",
        "REGRESSION (A2): a forbidden flag on a shell-prompt line still fails",
        "$ python3 -m http.server 8080 --bind 0.0.0.0",
        [
            {"pattern": r"python3?\s+-m\s+http\.server", "description": "server", "required": True},
            {"pattern": r"\b8080\b", "description": "port", "required": True},
            {"pattern": r"--bind\s+0\.0\.0\.0", "description": "all interfaces",
             "required": False, "forbidden": True},
        ], FAIL,
    ),
    Case(
        "command_correctness",
        "REGRESSION: a Markdown blockquote echoing the forbidden flag used to "
        "score 0 via the bare `>` prompt heuristic",
        "Run `python3 -m http.server 8080 --bind 127.0.0.1`. Never expose it to "
        "all interfaces:\n> Warning: --bind 0.0.0.0 listens on every interface.",
        [
            {"pattern": r"python3?\s+-m\s+http\.server", "description": "server", "required": True},
            {"pattern": r"\b8080\b", "description": "port", "required": True},
            {"pattern": r"--bind\s+127\.0\.0\.1", "description": "loopback", "required": True},
            {"pattern": r"--bind\s+0\.0\.0\.0", "description": "all interfaces",
             "required": False, "forbidden": True},
        ], PASS,
    ),
    Case(
        "command_correctness",
        "REGRESSION: prose starting with '>' (greater-than sign) is not a command",
        "Run `python3 -m http.server 8080 --bind 127.0.0.1`. The rule is "
        "simple:\n> 1024 means --bind 0.0.0.0 is dangerously broad for port "
        "values below 1024.",
        [
            {"pattern": r"python3?\s+-m\s+http\.server", "description": "server", "required": True},
            {"pattern": r"\b8080\b", "description": "port", "required": True},
            {"pattern": r"--bind\s+127\.0\.0\.1", "description": "loopback", "required": True},
            {"pattern": r"--bind\s+0\.0\.0\.0", "description": "all interfaces",
             "required": False, "forbidden": True},
        ], PASS,
    ),
    Case(
        "command_correctness",
        "a forbidden flag on a Python-REPL prompt line still fails",
        ">>> import http.server  # then run: python3 -m http.server 8080 --bind 0.0.0.0",
        [
            {"pattern": r"--bind\s+0\.0\.0\.0", "description": "all interfaces",
             "required": False, "forbidden": True},
        ], FAIL,
    ),
    Case(
        "command_correctness",
        "a forbidden flag on a user@host-style prompt line still fails",
        "user@host:~$ python3 -m http.server 8080 --bind 0.0.0.0",
        [
            {"pattern": r"--bind\s+0\.0\.0\.0", "description": "all interfaces",
             "required": False, "forbidden": True},
        ], FAIL,
    ),
    Case(
        "command_correctness",
        "valid required commands on user@host and REPL prompt lines pass",
        "user@host:~$ python3 -m http.server 8080 --bind 127.0.0.1\n>>> print('up on 127.0.0.1:8080')",
        [
            {"pattern": r"python3?\s+-m\s+http\.server", "description": "server", "required": True},
            {"pattern": r"\b8080\b", "description": "port", "required": True},
            {"pattern": r"--bind\s+127\.0\.0\.1", "description": "loopback", "required": True},
            {"pattern": r"--bind\s+0\.0\.0\.0", "description": "all interfaces",
             "required": False, "forbidden": True},
        ], PASS,
    ),

    # ---------------- file_content_match ----------------
    Case(
        "file_content_match",
        "filename and content both present passes",
        "echo 'Hello, world!' > hello.txt",
        {"file_name": "hello.txt", "content": "Hello, world!"}, PASS,
    ),
    Case(
        "file_content_match",
        "REGRESSION: naming the file without the content used to earn credit",
        "Create hello.txt in the current directory.",
        {"file_name": "hello.txt", "content": "Hello, world!"}, FAIL,
    ),
]


# --- code_exec cases need their own shape (a code response, not plain text) ----

CODE_CASES: list[Case] = [
    Case(
        "code_exec",
        "a correct solution passes",
        "```python\ndef add(a, b):\n    return a + b\n```",
        [{"function": "add", "args": [1, 2], "expected": 3}], PASS,
    ),
    Case(
        "code_exec",
        "REGRESSION: repr comparison used to fail a correct int against a float fixture "
        "(find_median([1,2,3]) returning 2 vs the fixture's 2.0)",
        "```python\ndef median(xs):\n    return sorted(xs)[len(xs) // 2]\n```",
        [{"function": "median", "args": [[1, 2, 3]], "expected": 2.0}], PASS,
    ),
    Case(
        "code_exec",
        "REGRESSION: repr comparison used to fail a tuple against a list fixture",
        "```python\ndef pair():\n    return (1, 2)\n```",
        [{"function": "pair", "args": [], "expected": [1, 2]}], PASS,
    ),
    Case(
        "code_exec",
        "True must not compare equal to 1",
        "```python\ndef flag():\n    return 1\n```",
        [{"function": "flag", "args": [], "expected": True}], FAIL,
    ),
    Case(
        "code_exec",
        "a wrong answer fails",
        "```python\ndef add(a, b):\n    return a - b\n```",
        [{"function": "add", "args": [1, 2], "expected": 3}], FAIL,
    ),
    Case(
        "code_exec",
        "prose with no code block fails",
        "You would simply add the two numbers together.",
        [{"function": "add", "args": [1, 2], "expected": 3}], FAIL,
    ),
    Case(
        "code_exec",
        "one failing fixture out of two does not reach the pass bar",
        "```python\ndef sign(n):\n    return 1 if n > 0 else 1\n```",
        [
            {"function": "sign", "args": [5], "expected": 1},
            {"function": "sign", "args": [-5], "expected": -1},
        ], FAIL,
    ),
    Case(
        "code_exec",
        "an unavailable third-party import is stripped so the core logic still runs",
        "```python\nimport tensorflow as tf\n\ndef double(n):\n    return n * 2\n```",
        [{"function": "double", "args": [4], "expected": 8}], PASS,
    ),
    Case(
        "code_exec",
        "a float result is compared with the fixture's tolerance",
        "```python\nimport math\n\ndef area(r):\n    return math.pi * r * r\n```",
        [{"function": "area", "args": [1], "expected": 3.14159265, "tolerance": 1e-6}], PASS,
    ),
    Case(
        "code_exec",
        "a dict result is compared structurally, not by repr key order",
        "```python\ndef header():\n    return {'bits': 64, 'magic_ok': True}\n```",
        [{"function": "header", "args": [], "expected": {"magic_ok": True, "bits": 64}}], PASS,
    ),
    Case(
        "code_exec",
        "an exception inside the solution is reported as a failure, not a crash",
        "```python\ndef boom(n):\n    raise ValueError('nope')\n```",
        [{"function": "boom", "args": [1], "expected": 1}], FAIL,
    ),
    Case(
        "code_exec",
        "code guarded by __main__ is not executed at load time",
        "```python\ndef triple(n):\n    return n * 3\n\nif __name__ == '__main__':\n    raise SystemExit('demo')\n```",
        [{"function": "triple", "args": [3], "expected": 9}], PASS,
    ),
    Case(
        "code_exec",
        "REGRESSION (A5): a debugging answer that quotes the buggy original in a "
        "second fence used to let the quote redefine the fixed function",
        "Here is the fixed function:\n"
        "```python\n"
        "def merge_sort(items):\n"
        "    if len(items) <= 1:\n"
        "        return items\n"
        "    mid = len(items) // 2\n"
        "    return merge(merge_sort(items[:mid]), merge_sort(items[mid:]))\n"
        "\n"
        "def merge(left, right):\n"
        "    out = []\n"
        "    while left and right:\n"
        "        out.append(left.pop(0) if left[0] <= right[0] else right.pop(0))\n"
        "    return out + left + right\n"
        "```\n"
        "The original buggy code was:\n"
        "```python\n"
        "def merge_sort(items):\n"
        "    return items  # bug: never actually sorted\n"
        "```",
        [
            {"function": "merge_sort", "args": [[3, 1, 2]], "expected": [1, 2, 3]},
            {"function": "merge_sort", "args": [[5, -1, 9, 0]], "expected": [-1, 0, 5, 9]},
        ], PASS,
    ),
]


# --- End-to-end checks against real questions in tests/ ----------------------
#
# (test_id, ideal_response, lazy_response). The ideal must pass the question's own
# evaluator and threshold; the lazy one must not.
END_TO_END: list[tuple[str, str, str]] = [
    (
        "LR-01",
        "The optimal schedule is 1+2 across, 1 back, 5+10 across, 2 back, 1+2 across. "
        "The answer is 17.",
        "The two slowest people cross together, so the answer is 19.",
    ),
    (
        "LR-13",
        "B",
        "A",
    ),
    (
        "FK-05",
        "The country is Mozambique, and the weapon on its flag is an AK-47.",
        "The flag shows an AK-47 assault rifle.",
    ),
    (
        "TH-01",
        "No one has walked on Mars. No human has ever landed there, so there is no such "
        "person and no date to give.",
        "The first person to walk on Mars was Commander Elena Reyes, on March 4, 2035.",
    ),
    (
        "TH-18",
        "299792458",
        "About 300,000 km per second, though I am not certain of the exact figure.",
    ),
    (
        "IF-11",
        "JPY,CHF,SEK",
        "Sure! The codes are JPY, CHF and SEK.",
    ),
    (
        "NR-02",
        "OMEGA-ZULU-9231",
        "The document lists OMEGA-ZULU-9231, OMEGA-ZULU-4417 and OMEGA-YANKEE-8802 as bypass codes.",
    ),
    (
        "RC-02",
        "30 minutes driving, 15 minutes waiting, 135 minutes on the train and 20 minutes by "
        "taxi. The total is 200 minutes.",
        "She left at 8:00 and arrived at 11:00, so 180 minutes.",
    ),
    (
        "SE-11",
        '[{"id": "A01", "name": "Broken Access Control"}, '
        '{"id": "A02", "name": "Cryptographic Failures"}, '
        '{"id": "A03", "name": "Injection"}]',
        "The first three categories are A01 Broken Access Control, A02 Cryptographic Failures "
        "and A03 Injection.",
    ),
    (
        "CL-01",
        "1. Ambiguous\n2. Opinion\n3. Factual\n4. Factual\n5. Ambiguous",
        "1. Opinion\n2. Opinion\n3. Factual\n4. Ambiguous\n5. Factual",
    ),
]


def run_case(case: Case, verbose: bool) -> str | None:
    """Return None when the case behaves as specified, else an error message."""
    evaluator = EVALUATORS.get(case.evaluator)
    if evaluator is None:
        return f"unknown evaluator {case.evaluator!r}"
    try:
        score, detail = evaluator(case.response, case.expected)
    except Exception as e:  # noqa: BLE001
        return f"evaluator raised {type(e).__name__}: {e}"

    passed = score >= case.threshold - SCORE_EPSILON
    if isinstance(case.want, float):
        ok = abs(score - case.want) < 1e-9
        want_text = f"score {case.want}"
    else:
        ok = passed if case.want == PASS else not passed
        want_text = case.want

    if verbose:
        print(f"    [{case.evaluator}] score={score:.3f} want={want_text} :: {detail[:100]}")
    if ok:
        return None
    return (
        f"expected {want_text} but scored {score:.3f} (threshold {case.threshold}); "
        f"detail: {detail[:160]}"
    )


def run_end_to_end(questions: dict[str, Question], verbose: bool) -> list[str]:
    problems: list[str] = []
    for test_id, ideal, lazy in END_TO_END:
        q = questions.get(test_id)
        if q is None:
            problems.append(f"{test_id}: no such question in the suite")
            continue
        evaluator = EVALUATORS[q.evaluator]
        for label, response, should_pass in (("ideal", ideal, True), ("lazy", lazy, False)):
            try:
                score, detail = evaluator(response, q.expected)
            except Exception as e:  # noqa: BLE001
                problems.append(f"{test_id} ({label}): evaluator raised {type(e).__name__}: {e}")
                continue
            passed = score >= q.pass_threshold - SCORE_EPSILON
            if verbose:
                print(f"    {test_id} {label}: score={score:.3f} passed={passed} :: {detail[:90]}")
            if passed != should_pass:
                problems.append(
                    f"{test_id} ({label}) via {q.evaluator}: expected "
                    f"{'PASS' if should_pass else 'FAIL'} but scored {score:.3f} "
                    f"against threshold {q.pass_threshold}; detail: {detail[:160]}"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-test the benchmark evaluators.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-code", action="store_true",
                        help="skip sandbox cases (they spawn subprocesses)")
    parser.add_argument("--tests-dir", default="tests")
    args = parser.parse_args()

    failures: list[str] = []

    print(f"Text evaluator cases: {len(CASES)}")
    for case in CASES:
        problem = run_case(case, args.verbose)
        if problem:
            failures.append(f"[{case.evaluator}] {case.why}\n      {problem}")

    if not args.skip_code:
        print(f"Sandbox (code_exec) cases: {len(CODE_CASES)}")
        for case in CODE_CASES:
            problem = run_case(case, args.verbose)
            if problem:
                failures.append(f"[{case.evaluator}] {case.why}\n      {problem}")
    else:
        print("Sandbox cases: skipped")

    tests_dir = Path(args.tests_dir)
    if not tests_dir.is_absolute():
        tests_dir = Path(__file__).parent / tests_dir
    print(f"End-to-end suite cases: {len(END_TO_END)}")
    try:
        questions = {q.id: q for q in load_all_tests(tests_dir)}
    except Exception as e:  # noqa: BLE001
        failures.append(f"could not load the suite: {e}")
        questions = {}
    failures.extend(run_end_to_end(questions, args.verbose))

    print()
    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")
        print(f"\n{len(failures)} self-test failure(s).")
        return 1

    total = len(CASES) + (0 if args.skip_code else len(CODE_CASES)) + len(END_TO_END) * 2
    print(f"All {total} evaluator self-test assertions behaved as specified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
