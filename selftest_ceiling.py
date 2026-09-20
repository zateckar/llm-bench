"""Database regression cases and independent checks of ceiling-v5 challenge keys."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import itertools
import json
from pathlib import Path
import random
import re
import unittest

from evaluators import EVALUATORS, eval_json_match, extract_json
from hard_cases import build_cases, ledger, revision_data, resolve_revisions, worlds_for
from hard_code_cases import CODE
from interactive_tasks import make_tasks, run_interaction
from models import RequestMetrics, Result, TokenUsage
from quality_report import result_record, summarize
from quality_suite import QualityConfig, assemble_questions
from selftest_specialists import corruptions
from test_loader import _parse_question, load_all_tests
from validate_suite import Report, check_json_match


class CeilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = build_cases()
        cls.questions = {q.id: q for q in load_all_tests(Path(__file__).parent / "tests")}

    def test_aliases_fix_observed_contract_defects_without_type_coercion(self):
        examples = {
            "TU2-01": [
                {"tool": "list_objects", "args": {"cursor": "p2", "snapshot": "s7"}},
                {"tool": "fetch_object", "args": {"id": "b", "version": "v4"}},
                {"tool": "fetch_object", "args": {"id": "c", "version": "v2"}},
            ],
            "SE-11": [
                {"id": "A01:2021", "name": "Broken Access Control"},
                {"id": "A02:2021", "name": "Cryptographic Failures"},
                {"id": "A03:2021", "name": "Injection"},
            ],
            "RV4-CL-02": {
                "labels": ["SUPPORTED", None, "CONTRADICTED", "CONTRADICTED", None, "SUPPORTED"]
            },
        }
        for id, answer in examples.items():
            self.assertEqual(eval_json_match(json.dumps(answer), self.questions[id].expected)[0], 1)
        for bad in ["4", "v3", "V4", True, 4.1]:
            value = deepcopy(examples["TU2-01"])
            value[1]["args"]["version"] = bad
            self.assertEqual(
                eval_json_match(json.dumps(value), self.questions["TU2-01"].expected)[0], 0
            )
        wrong = deepcopy(examples["RV4-CL-02"])
        wrong["labels"][0] = None
        self.assertEqual(
            eval_json_match(json.dumps(wrong), self.questions["RV4-CL-02"].expected)[0], 0
        )
        wrong = deepcopy(examples["SE-11"])
        wrong[0]["id"] = "A01:2025"
        self.assertEqual(eval_json_match(json.dumps(wrong), self.questions["SE-11"].expected)[0], 0)
        # Actual observed wrong numeric answer remains wrong after JSON parsing.
        self.assertEqual(
            eval_json_match(
                '{"count":96,"sum":240456,"smallest":29,"largest":4997}',
                self.questions["MR2-05"].expected,
            )[0],
            0,
        )

    def test_json_markup_is_data_and_alias_schema_is_validated(self):
        value = {"text": "<think>literal data</think>", "n": 2**60 + 1}
        for wrapper in (lambda x: x, lambda x: "```json\n" + x + "\n```"):
            self.assertEqual(extract_json(wrapper(json.dumps(value)), strict=True), (value, None))
        for bad in ['{"x":1,"x":1}', '{"x":NaN}', '{"x":1} {"x":1}', 'text {"x":1}']:
            self.assertIsNotNone(extract_json(bad, strict=True)[1])
        for aliases in [
            {"bad": [None]},
            {"x": []},
            {"x": [{}]},
            {"x": "1"},
            [],
            {"x": [float("nan")]},
        ]:
            report = Report()
            check_json_match(report, "test", {"value": {"x": 1}, "value_aliases": aliases})
            self.assertTrue(report.errors, aliases)

    def test_interactive_error_identifies_failing_turn(self):
        q = make_tasks(QualityConfig(), 1729, 0)[0]

        class Client:
            def __init__(self, final):
                self.i, self.final = 0, final

            def complete_messages(self, messages, **kwargs):
                self.i += 1
                response = (
                    json.dumps({"tool": "get_document", "args": {"id": q.interaction["doc"]}})
                    if self.i == 1
                    else self.final
                )
                return response, TokenUsage(10, 10), RequestMetrics()

        for final, outcome in [
            ("<think>Read again.</think>", "missing_answer"),
            ("I will read again.", "formatting"),
            ('I will do this: {"done":true}', "formatting"),
        ]:
            result = run_interaction(q, Client(final))
            self.assertEqual(result.outcome, outcome)
            self.assertIn("Turn 2", result.detail)
            self.assertEqual(result.diagnostics["calls"], 1)
            self.assertFalse(result.passed)

    def test_shipped_cases_and_mutated_answers(self):
        self.assertEqual(len(self.raw), 52)
        self.assertEqual(set(Counter(r["category"] for r in self.raw).values()), {4})
        rejected = 0
        for raw in self.raw:
            q = _parse_question(raw, raw["id"])
            self.assertEqual(q, self.questions[q.id])
            if q.evaluator != "json_match":
                continue
            correct = json.dumps(q.expected["value"])
            self.assertEqual(eval_json_match(correct, q.expected)[0], 1)
            for wrong in corruptions(q.expected["value"]):
                self.assertEqual(
                    eval_json_match(json.dumps(wrong), q.expected)[0], 0, (q.id, wrong)
                )
                rejected += 1
        print(f"Rejected {rejected} corrupted ceiling answers")

    def test_world_oracle_against_bitmask_and_revisions_against_sorting(self):
        rng = random.Random(993)
        for _ in range(25):
            clauses = [
                [rng.choice((-1, 1)) * rng.randrange(1, 11) for _ in range(3)] for _ in range(20)
            ]
            valid = []
            for mask in range(1024):
                bits = tuple(bool(mask & (1 << i)) for i in range(10))
                # A clause fails iff every literal is false.
                if not any(
                    all((bits[abs(x) - 1] if x < 0 else not bits[abs(x) - 1]) for x in row)
                    for row in clauses
                ):
                    valid.append(bits)
            self.assertEqual(sorted(valid), worlds_for(clauses))
        for variant in range(1, 17):
            rows, aliases, routes, queries = revision_data(variant, variant > 8)
            result = resolve_revisions(rows, aliases, routes, queries)
            totals = Counter()
            for route in queries:
                eligible = sorted(
                    [
                        r
                        for r in rows
                        if r["entity"] == aliases[routes[route]]
                        and r["approved"]
                        and r["effective"] <= 23
                    ],
                    key=lambda r: (r["effective"], r["revision"]),
                    reverse=True,
                )
                r = eligible[0] if eligible else None
                self.assertEqual(result["routes"][route]["record"], r["record"] if r else None)
                if r and not r["deleted"]:
                    totals[r["owner"]] += r["quota"]
            self.assertEqual(result["grand_total"], sum(totals.values()))
            self.assertEqual(
                resolve_revisions(list(reversed(rows)), aliases, routes, queries), result
            )

    def test_code_references_in_real_sandbox_and_independent_examples(self):
        functions = {}
        for id, code in CODE.items():
            q = self.questions[id]
            score, detail = EVALUATORS["code_exec"](code, q.expected)
            self.assertEqual(score, 1, (id, detail))
            scope = {}
            exec(code, scope)
            functions[id] = scope[q.expected[0]["function"]]
        parse = functions["H5-CG-jsonl-01"]
        text = '\n{"id":"a","version":1,"at":"2026-01-01T00:00:00Z","deleted":false,"amount":"-1.005"}\nnot json\n'
        self.assertEqual(
            parse(text, "2026-01-02T00:00:00Z"),
            {"live": [{"id": "a", "cents": -101}], "invalid_lines": [2], "total_cents": -101},
        )
        alloc = functions["H5-CG-waterfill-01"]
        self.assertEqual(
            alloc(5, [["b", 1, 9], ["a", 1, 9]]),
            {"allocations": [["a", 3], ["b", 2]], "unallocated": 0},
        )
        self.assertEqual(
            alloc(100, [["a", 9, 3], ["b", 1, 7]]),
            {"allocations": [["a", 3], ["b", 7]], "unallocated": 90},
        )
        transfer = functions["H5-CG-ledger-01"]
        events = [
            ["k", "a", "b", 8, 0],
            ["k", "a", "b", 8, 99],
            ["k", "a", "b", 9, 0],
            ["x", "b", "a", 3, 0],
            ["x", "b", "a", 3, 1],
        ]
        expected = {
            "decisions": ["applied", "replay", "key_conflict", "stale", "applied"],
            "balances": {"a": 5, "b": 5},
            "versions": {"a": 2, "b": 2},
            "committed_keys": ["k", "x"],
        }
        self.assertEqual(transfer({"a": 10, "b": 0}, events), expected)
        self.assertEqual(ledger(events, {"a": 10, "b": 0}), expected)
        # Reject common plausible wrong programs, rather than testing only happy paths.
        mutants = {
            "H5-CG-jsonl-01": CODE["H5-CG-jsonl-01"].replace(
                "rounding=ROUND_HALF_UP", "rounding='ROUND_HALF_EVEN'"
            ),
            "H5-CG-routes-01": CODE["H5-CG-routes-01"].replace("leave >= closes", "leave > closes"),
            "H5-CG-waterfill-01": CODE["H5-CG-waterfill-01"].replace(
                "remaining*weights[k]/denom", "remaining/len(active)"
            ),
            "H5-CG-ledger-01": CODE["H5-CG-ledger-01"].replace(
                "versions[dst] += 1", "versions[dst] += 0"
            ),
        }
        for id, code in mutants.items():
            self.assertLess(EVALUATORS["code_exec"](code, self.questions[id].expected)[0], 1, id)

    def test_routes_against_permutations_and_waterfill_against_breakpoints(self):
        for id in ("H5-CG-routes-01", "H5-CG-waterfill-01"):
            scope = {}
            exec(CODE[id], scope)
            for fixture in self.questions[id].expected:
                if "routes" in id:
                    edges, start, end, depart, budget = fixture["args"]
                    candidates = []
                    if start == end:
                        candidates = [(depart, 0, [start])]
                    else:
                        intermediate = sorted({x for e in edges for x in e[:2]} - {start, end})
                        for n in range(len(intermediate) + 1):
                            for middle in itertools.permutations(intermediate, n):
                                path = [start, *middle, end]
                                edge_sets = [
                                    [e for e in edges if e[:2] == [a, b]]
                                    for a, b in zip(path, path[1:])
                                ]
                                for choices in itertools.product(*edge_sets):
                                    now, cost, ok = depart, 0, True
                                    for _, _, duration, toll, opens, closes in choices:
                                        now = max(now, opens)
                                        if now >= closes:
                                            ok = False
                                            break
                                        now += duration
                                        cost += toll
                                    if ok and cost <= budget:
                                        candidates.append((now, cost, path))
                    best = min(candidates) if candidates else None
                    expected = (
                        None if best is None else dict(arrival=best[0], cost=best[1], path=best[2])
                    )
                else:
                    total, rows = fixture["args"]
                    usable = [r for r in rows if r[1] > 0 and r[2] > 0]
                    target = min(total, sum(r[2] for r in usable))
                    quotas = {r[0]: Fraction(0) for r in rows}
                    # Solve sum(min(cap, lambda*weight))=target by testing intervals
                    # between all cap/weight breakpoints, independently of the loop.
                    points = sorted({Fraction(c, w) for _, w, c in usable})
                    lower = Fraction(0)
                    for upper in points:
                        capped = [r for r in usable if Fraction(r[2], r[1]) <= lower]
                        active = [r for r in usable if r not in capped]
                        if not active:
                            break
                        lam = Fraction(
                            target - sum(r[2] for r in capped), sum(r[1] for r in active)
                        )
                        if lower <= lam <= upper:
                            quotas.update({k: min(Fraction(c), lam * w) for k, w, c in usable})
                            break
                        lower = upper
                    whole = {k: v.numerator // v.denominator for k, v in quotas.items()}
                    for key in sorted(whole, key=lambda k: (-(quotas[k] - whole[k]), k))[
                        : target - sum(whole.values())
                    ]:
                        whole[key] += 1
                    expected = {
                        "allocations": [[k, v] for k, v in sorted(whole.items())],
                        "unallocated": total - target,
                    }
                self.assertEqual(fixture["expected"], expected, (id, fixture["args"]))

    def test_family_balancing_and_challenge_cohort(self):
        selected = [
            self.questions["H5-CL-worlds-01"],
            self.questions["H5-CL-worlds-02"],
            self.questions["RV4-CL-01"],
        ]
        qs = assemble_questions(
            selected, QualityConfig(generated=False, interactive=False, strengthen_code=False)
        )
        rows = [
            result_record(Result(q, "{}", s, outcome="pass" if s else "task_failure"))
            for q, s in zip(qs, [0, 1, 1])
        ]
        report = summarize(rows)
        self.assertEqual(report["category_balanced"], 0.75)  # two variants share one family
        self.assertEqual(report["challenge"]["category_balanced"], 0.5)
        self.assertEqual(report["challenge"]["families"], 1)
        self.assertEqual(report["challenge"]["count"], 2)
        self.assertIsNone(summarize([rows[-1]])["challenge"]["category_balanced"])
        self.assertNotEqual(
            result_record(Result(replace(qs[0], expected={"value": 0}), "{}", 0))["fingerprint"],
            rows[0]["fingerprint"],
        )

    def test_published_science_and_allocation_keys_independently(self):
        for raw in self.raw:
            prompt = raw["prompt"]
            if raw["id"].startswith("H5-FK-"):
                n = int(re.search(r"Initial \(A,B,C\)=\((\d+),0,0\)", prompt)[1])
                # Closed form, rather than the authoring recurrence.
                a = Fraction(n, 32)
                b = 2 * n * (Fraction(3, 4) ** 5 - Fraction(1, 2) ** 5)
                answer = raw["expected"]["value"]
                for key, value in (("A", a), ("B", b), ("C", n - a - b)):
                    self.assertEqual(
                        answer[key],
                        {"numerator": value.numerator, "denominator": value.denominator},
                    )
            elif raw["id"].startswith("H5-TN-"):
                match = re.search(
                    r"differences are (\[.*?\]); PRE-SPECIFIED\s+integer weights are (\[.*?\])",
                    prompt,
                )
                terms = [a * b for a, b in zip(json.loads(match[1]), json.loads(match[2]))]
                observed = abs(sum(terms))
                counts = Counter()
                for n in range(9):
                    for subset in itertools.combinations(range(8), n):
                        counts[abs(2 * sum(terms[i] for i in subset) - sum(terms))] += 1
                extreme = sum(count for value, count in counts.items() if value >= observed)
                self.assertEqual(extreme, raw["expected"]["value"]["extreme_vectors"])
                self.assertEqual(sum(counts.values()), 256)
            elif raw["id"].startswith("H5-ER-"):
                people = json.loads(prompt.split("All facts:\n", 1)[1].split("\nReturn", 1)[0])
                cap = int(re.search(r"total cost <= (\d+)", prompt)[1])
                ranked = []
                for n in range(10):
                    for subset in itertools.combinations(people, n):
                        cost = sum(p["cost"] for p in subset)
                        if (
                            all(p["consent"] for p in subset)
                            and cost <= cap
                            and sum(p["group"] == "rural" for p in subset) >= 2
                        ):
                            ranked.append(
                                (
                                    -sum(p["benefit"] for p in subset),
                                    cost,
                                    sorted(p["id"] for p in subset),
                                )
                            )
                ranked.sort()
                expected = raw["expected"]["value"]
                self.assertEqual(
                    [expected["benefit"], expected["cost"]], [-ranked[0][0], ranked[0][1]]
                )
                self.assertEqual(
                    expected["allocations"],
                    [ids for benefit, cost, ids in ranked if (benefit, cost) == ranked[0][:2]],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
