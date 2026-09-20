"""Offline specialist answer derivations, counterexamples and suite integration."""

from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
import itertools
import json
from pathlib import Path
import re
import unittest

from evaluators import eval_json_match
from quality_suite import QualityConfig, assemble_questions, question_scope
from specialist_cases import build_cases
from test_loader import _parse_question, load_all_tests


def corruptions(value):
    """Change/delete each leaf, not just the first matching keyword."""
    if isinstance(value, dict):
        for key in value:
            missing = deepcopy(value)
            del missing[key]
            yield missing
            for bad in corruptions(value[key]):
                yield {**value, key: bad}
        yield {**value, "invented_fact": True}
    elif isinstance(value, list):
        yield value + ["invented"]
        for i, item in enumerate(value):
            yield value[:i] + value[i + 1 :]
            for bad in corruptions(item):
                yield value[:i] + [bad] + value[i + 1 :]
    elif isinstance(value, bool):
        yield not value
        yield int(value)
    elif isinstance(value, (int, float)):
        yield value + 1
    elif value is None:
        yield "unknown"
    else:
        yield value + " wrong"


class SpecialistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = build_cases()
        cls.questions = {q.id: q for q in load_all_tests(Path(__file__).parent / "tests")}

    def answer(self, suffix):
        return self.questions["SP1-" + suffix].expected["value"]

    def test_shipped_cases_and_category_coverage(self):
        counts = Counter(r["category"] for r in self.raw)
        self.assertEqual(
            counts,
            {
                "Legal": 4,
                "Finances": 4,
                "R&D": 4,
                "Language Translations": 6,
                "Code Review": 4,
                "Security": 8,
            },
        )
        self.assertEqual(len({r["id"] for r in self.raw}), 30)
        for raw in self.raw:
            q = _parse_question(raw, raw["id"])
            self.assertEqual(q, self.questions[q.id])
            self.assertEqual(question_scope(q), "capability")
        assembled = assemble_questions(list(self.questions.values()), QualityConfig())
        # All additions survive the same assembly used by UI/CLI; no category allowlist.
        actual = {q.id: q for q in assembled}
        for raw in self.raw:
            self.assertEqual(actual[raw["id"]].metadata["scope"], "capability")

    def test_every_answer_leaf_and_format_is_checked(self):
        checked = 0
        for raw in self.raw:
            expected = raw["expected"]
            text = json.dumps(expected["value"], ensure_ascii=False)
            with self.subTest(id=raw["id"]):
                self.assertEqual(eval_json_match(text, expected)[0], 1)
                self.assertEqual(eval_json_match("```json\n" + text + "\n```", expected)[0], 1)
                for mutant in corruptions(expected["value"]):
                    self.assertEqual(eval_json_match(json.dumps(mutant), expected)[0], 0, mutant)
                    checked += 1
                for bad in [text + "\nActually, the above is wrong.", text + text, "{}", "null"]:
                    self.assertEqual(eval_json_match(bad, expected)[0], 0)
        self.assertGreater(checked, 0)
        print(f"Rejected {checked} corrupted specialist answers")

    def test_review_negative_controls_and_all_files_present(self):
        for raw in self.raw:
            if raw["category"] not in {"Code Review", "Security"}:
                continue
            statuses = raw["expected"]["value"]["claims"]
            self.assertEqual(set(statuses), {"confirmed", "refuted", "unknown"})
            self.assertEqual(len(re.findall(r"^C\d+: ", raw["prompt"], re.M)), len(statuses))
            file_count = len(re.findall(r"^FILE ", raw["prompt"], re.M))
            self.assertGreaterEqual(file_count, 7 if raw["category"] == "Code Review" else 4)
            for blanket in ("confirmed", "refuted", "unknown"):
                wrong = deepcopy(raw["expected"]["value"])
                wrong["claims"] = [blanket] * len(statuses)
                self.assertEqual(eval_json_match(json.dumps(wrong), raw["expected"])[0], 0)

    def test_translation_directions_and_false_positive_penalties(self):
        directions = {
            r["id"].removeprefix("SP1-TR-")
            for r in self.raw
            if r["category"] == "Language Translations"
        }
        self.assertEqual(
            directions, {f"{a}-{b}" for a, b in itertools.permutations(("CS", "EN", "DE"), 2)}
        )
        for raw in self.raw:
            if raw["category"] != "Language Translations":
                continue
            values = raw["expected"]["value"]
            self.assertEqual(set(values), {"1", "2", "3"})
            self.assertEqual(raw["prompt"].count("Passage "), 3)
            self.assertTrue(any(len(v) > 1 for v in values.values()))
            for passage, accepted in values.items():
                self.assertEqual(accepted, sorted(set(accepted)))
                for choice in "ABCD":
                    changed = deepcopy(values)
                    changed[passage] = sorted(set(accepted) ^ {choice})
                    self.assertEqual(eval_json_match(json.dumps(changed), raw["expected"])[0], 0)

    def test_legal_set_reasoning_and_business_days(self):
        # Independent literal element/territory chart, not the serialized answer.
        products = {
            "X": {"motor", "sensor", "controller"},
            "Y": {"motor", "controller"},
            "Z": {"motor", "sensor", "optical"},
        }
        patents = [
            ("P1", "DE", {"motor", "sensor"}, {"X"}),
            ("P2", "CZ", {"motor", "controller"}, set()),
            ("P5", "DE", {"motor", "sensor", "controller"}, {"X"}),
        ]
        answer = self.answer("LE-02")
        for name, elements in products.items():
            for territory in ("DE", "CZ"):
                blockers = [
                    p
                    for p, t, claim, licensed in patents
                    if t == territory and claim <= elements and name not in licensed
                ]
                self.assertEqual(answer[name][territory], sorted(blockers))
        current, days = date(2026, 1, 9), 0
        while days < 5:
            current += timedelta(days=1)
            days += current.weekday() < 5
        self.assertEqual(self.answer("LE-04")["notice_deadline"], current.isoformat())
        # A+B from two publications cannot anticipate A+B under the stated rule.
        docs = {"D1": {"A"}, "D2": {"B", "C"}, "D3": {"A", "B"}}
        self.assertEqual(
            self.answer("LE-01")["C1"]["novelty_destroyers"],
            [id for id, elements in docs.items() if {"A", "B"} <= elements],
        )

    def test_finance_independent_arithmetic_and_conservation(self):
        a = self.answer("FI-01")
        # Conservation across all days; restricted 30 never pays obligations.
        expected_debts = [45, 0, 55, 10]
        self.assertEqual([d["debt"] for d in a["days"]], expected_debts)
        cumulative = 70
        debt = 0
        for flow, row in zip([-95, 50, -60, 45], a["days"]):
            cumulative += flow
            debt += row["draw"] - row["repay"]
            self.assertEqual(row["cash"], cumulative + debt)
            self.assertGreaterEqual(row["cash"], 20)
        self.assertEqual(a["D4_total_bank_cash"], a["days"][-1]["cash"] + 30)
        fx = self.answer("FI-02")
        for spot in (24, 26):
            self.assertEqual(
                fx[f"net_CZK_at_{spot}"], Decimal("24.80") * 300000 + spot * 200000 - 5000000
            )
        self.assertEqual(fx["scenario_spread_CZK"], (26 - 24) * fx["unhedged_EUR"])
        cov = self.answer("FI-03")
        interest = sum(Fraction(p * days * 6, 36000) for p, days in [(1200000, 30), (800000, 15)])
        self.assertEqual(cov["interest"], interest)
        ratio = Fraction(800000 + interest + 250000 - (210000 - 60000), 260000)
        self.assertEqual(
            cov["ratio"], {"numerator": ratio.numerator, "denominator": ratio.denominator}
        )
        self.assertEqual(cov["compliant"], ratio <= Fraction(7, 2))
        stress = self.answer("FI-04")
        total = 35 + Fraction(80 * 95, 100) + min(70, 25)
        self.assertEqual(stress["usable_by_D2"], total)
        self.assertEqual(stress["policy_shortfall"], 150 + 15 - total)

    def test_engineering_enumeration_and_exact_physics(self):
        stack = self.answer("RD-02")
        cold = [
            h - a - b
            for h, a, b in itertools.product((49920, 50080), (19970, 20030), (29450, 29550))
        ]
        hot = [g + 60 - 40 - 90 for g in cold]
        self.assertEqual(stack["cold_unshimmed"], [min(cold), max(cold)])
        self.assertEqual(stack["hot_unshimmed"], [min(hot), max(hot)])
        self.assertEqual(
            stack["feasible_shims"],
            [s for s in (0, 100, 200, 300) if all(100 <= g - s <= 450 for g in cold + hot)],
        )
        energy = self.answer("RD-01")
        kinetic = Fraction(2000, 2) * (20**2 - 10**2)
        regen = min(kinetic * Fraction(3, 5), 20000 * 5)
        friction = kinetic - regen / Fraction(3, 5)
        self.assertEqual(energy["recovered_J"], regen)
        self.assertEqual(energy["friction_brake_J"], round(friction))
        self.assertEqual(energy["net_battery_depletion_J"], 30000 * 40 + 2000 * 45 - regen)
        robotics = self.answer("RD-03")
        # Rotation +90 maps (x,y) to (-y,x), then translate.
        self.assertEqual(robotics["world_point"], [2 - 0.5, 3 + 0.2 + 1])
        stop = Fraction(2) * Fraction(3, 10) + Fraction(2**2, 2 * 4) + Fraction(1, 4)
        self.assertEqual(robotics["required_clearance_m"], float(stop))
        self.assertEqual(robotics["clearance_deficit_m"], float(stop - Fraction(13, 10)))
        p = Fraction(
            sum(abs(sum(signs)) >= 2 for signs in itertools.product((-1, 1), repeat=4)), 16
        )
        self.assertEqual(
            self.answer("RD-04")["sign_test_p"],
            {"numerator": p.numerator, "denominator": p.denominator},
        )

    def test_review_traces_against_small_state_models(self):
        # Different implementations of the critical interleavings, with safe controls.
        stock = 5
        approved = [stock >= qty for qty in (4, 4)]
        for ok, qty in zip(approved, (4, 4)):
            if ok:
                stock -= qty
        self.assertEqual(self.answer("CR-01")["trace"]["final_stock"], stock)
        self.assertEqual(self.answer("CR-01")["trace"]["A_total_charged"], sum([4 * 10] * 2))
        cache = {}
        results = []
        for tenant, body in [("TA", "<b>A secret</b>"), ("TB", "B data")]:
            results.append(cache.setdefault(7, body))
        self.assertEqual(self.answer("CR-02")["trace"]["B_body"], results[-1])
        self.assertEqual(
            self.answer("CR-03")["trace"]["exported_cents"], int(round(Decimal("1.99"))) * 100
        )
        objects = {}
        for tenant, body in [("TA", "A-data"), ("TB", "B-data")]:
            objects["result.csv"] = body
        self.assertEqual(self.answer("CR-04")["trace"]["A_download"], objects["result.csv"])
        seen, balance = set(), 0
        admitted = ["e" not in seen, "e" not in seen]
        for ok in admitted:
            if ok:
                balance += 10
                seen.add("e")  # second insert fails but cannot undo earlier credit
        self.assertEqual(self.answer("SE-02")["trace"]["balance"], balance)
        self.assertEqual(self.answer("SE-02")["trace"]["seen_rows"], len(seen))
        # Metadata version, not authenticated payload version, drives the guard.
        stored = 7
        payload_version, manifest_version = 3, 8
        if manifest_version > stored:
            flashed, stored = payload_version, manifest_version
        self.assertEqual(self.answer("SE-08")["trace"]["flashed_internal_version"], flashed)
        self.assertEqual(self.answer("SE-08")["trace"]["stored_version_after"], stored)
        self.assertFalse(7 > stored)


if __name__ == "__main__":
    unittest.main(verbosity=2)
