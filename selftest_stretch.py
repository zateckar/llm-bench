"""Independent derivations and adversarial regressions for stretch-v6 (no model calls)."""

from collections import defaultdict
from contextlib import closing
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import itertools
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from evaluators import eval_format_check, eval_json_match
from audit_results import audit_database
from models import Result
from quality_report import make_report, markdown, result_record, summarize
from llm_client import ClientConfig
from quality_suite import QualityConfig, assemble_questions
from selftest_specialists import corruptions
import stretch_cases as author
from test_loader import _parse_question, load_all_tests
from validate_suite import Report, check_format_check


class StretchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = author.build_cases()
        cls.questions = {q.id: q for q in load_all_tests(Path(__file__).parent / "tests")}

    def test_shipped_questions_and_mutations(self):
        self.assertEqual(len(self.raw), 30)
        self.assertEqual(len({q["category"] for q in self.raw}), 10)
        rejected = 0
        for raw in self.raw:
            q = _parse_question(raw, raw["id"])
            self.assertEqual(q, self.questions[q.id])
            self.assertEqual(eval_json_match(json.dumps(q.expected["value"]), q.expected)[0], 1)
            for bad in corruptions(q.expected["value"]):
                self.assertEqual(eval_json_match(json.dumps(bad), q.expected)[0], 0, (q.id, bad))
                rejected += 1
        print(f"Rejected {rejected} corrupted stretch answers")

    def test_minimax_via_budget_feasibility(self):
        for v in range(3):
            states, remedies, tests = author.planning_data(v)

            @lru_cache(None)
            def feasible(mask, budget):
                options = [i for i in range(len(states)) if mask & (1 << i)]
                if len({remedies[i] for i in options}) <= 1:
                    return True
                if budget <= 0:
                    return False
                for t in tests:
                    yes = sum(1 << i for i in options if t["outcomes"][i])
                    if yes in (0, mask) or t["cost"] > budget:
                        continue
                    if feasible(yes, budget - t["cost"]) and feasible(
                        mask ^ yes, budget - t["cost"]
                    ):
                        return True
                return False

            first = {}
            for t in tests:
                masks = [
                    sum(1 << i for i, bit in enumerate(t["outcomes"]) if bit == b) for b in (0, 1)
                ]
                first[t["id"]] = (
                    next(
                        b
                        for b in range(t["cost"], 50)
                        if all(feasible(m, b - t["cost"]) for m in masks)
                    )
                    if all(masks)
                    else None
                )
            best = min(c for c in first.values() if c is not None)
            self.assertEqual(
                author.planning_answer(remedies, tests),
                {
                    "worst_cost": best,
                    "cost_by_first_test": first,
                    "optimal_first_tests": sorted(k for k, c in first.items() if c == best),
                },
            )

    def test_tool_planning_via_subset_closure(self):
        for v in range(3):
            calls = author.tool_data(v)
            candidates = []
            for flags in itertools.product((False, True), repeat=len(calls)):
                pending = [c for c, included in zip(calls, flags) if included]
                cost = sum(c["cost"] for c in pending)
                available, seq = {"auth"}, []
                while pending:
                    ready = sorted(
                        (c for c in pending if set(c["requires"]) <= available),
                        key=lambda c: c["id"],
                    )
                    if not ready:
                        break
                    c = ready[0]
                    seq.append(c["id"])
                    available.update(c["produces"])
                    pending.remove(c)
                if not pending and set("abcdef") | {"receipt"} <= available:
                    candidates.append((cost, len(seq), seq))
            cost, n, seq = min(candidates)
            self.assertEqual(
                author.tool_answer(calls), {"cost": cost, "call_count": n, "calls": seq}
            )

    def test_bitemporal_by_sorted_candidates(self):
        for v in range(3):
            rows, queries = author.temporal_data(v)
            answers = []
            for t, cutoff in queries:
                sources, totals, voids = {}, dict(east=0, west=0), []
                for account in [f"A{i}" for i in range(7)]:
                    eligible = sorted(
                        (
                            r
                            for r in rows
                            if r["account"] == account
                            and r["recorded"] <= cutoff
                            and t in range(r["from"], r["to"])
                        ),
                        key=lambda r: (-r["recorded"], -r["priority"]),
                    )
                    sources[account] = eligible[0]["id"] if eligible else None
                    if eligible:
                        r = eligible[0]
                        if r["void"]:
                            voids.append(account)
                        else:
                            totals[r["owner"]] += r["amount"]
                answers.append(dict(sources=sources, totals=totals, void_accounts=voids))
            self.assertEqual(author.temporal_answer(rows, queries), {"snapshots": answers})
            # Order of archive presentation never changes knowledge-time semantics.
            self.assertEqual(author.temporal_answer(rows[::-1], queries), {"snapshots": answers})

    def test_summary_by_invoice_conservation(self):
        for v in range(3):
            rows = author.rollup_data(v)
            unique, ignored_positions = {}, set()
            for i, e in enumerate(rows):
                if e["id"] in unique:
                    ignored_positions.add(i)
                else:
                    unique[e["id"]] = (i, e)
            totals = {f"K{i}": dict(open=0, receivable=0, cash=0) for i in range(3)}
            voids = []
            for invoice in [f"I{i}" for i in range(9)]:
                events = sorted((i, e) for i, e in unique.values() if e["invoice"] == invoice)
                closure = next((i for i, e in events if e["op"] == "void"), len(rows))
                ignored_positions.update(i for i, _ in events if i > closure)
                active = [e for i, e in events if i < closure]
                paid = sum(
                    e.get("amount", 0)
                    * (1 if e["op"] == "pay" else -1 if e["op"] == "refund" else 0)
                    for e in active
                )
                due = sum(
                    e.get("amount", 0)
                    * (1 if e["op"] == "issue" else -1 if e["op"] == "credit" else 0)
                    for e in active
                )
                total = totals[active[0]["customer"]]
                total["cash"] += paid
                if closure < len(rows):
                    voids.append(invoice)
                else:
                    total["open"] += 1
                    total["receivable"] += due - paid
            self.assertEqual(
                author.rollup_answer(rows),
                dict(
                    customers=totals,
                    void_invoices=voids,
                    ignored=[rows[i]["id"] for i in sorted(ignored_positions)],
                ),
            )

    def test_routes_by_layer_products(self):
        for v in range(3):
            rows = author.graph_data(v)
            edges = {}
            for ident in {r["id"] for r in rows}:
                latest = max(
                    (r for r in rows if r["id"] == ident and r["approved"]), key=lambda r: r["rev"]
                )
                if not latest["disabled"]:
                    edges[latest["from"], latest["to"]] = latest["cost"]
            paths = []
            for middle in itertools.product(*[[f"N{i}{j}" for j in range(4)] for i in range(1, 5)]):
                path = ["start", *middle, "end"]
                pairs = list(zip(path, path[1:]))
                if all(pair in edges for pair in pairs):
                    paths.append((sum(edges[p] for p in pairs), path))
            minimum = min(c for c, _ in paths)
            best = sorted(p for c, p in paths if c == minimum)
            self.assertEqual(
                author.graph_answer(rows),
                dict(
                    cost=minimum,
                    optimal_count=len(best),
                    first_path=best[0],
                    mandatory_internal=sorted(
                        n for n in best[0][1:-1] if all(n in p for p in best)
                    ),
                ),
            )

    def test_scopes_using_full_snapshot_transactions(self):
        for v in range(3):
            # Full snapshots and undo logs instead of the author's sparse overlays.
            state = dict(color=11, quota=22, region=33)
            undo, reads = [], []
            for e in author.scopes_data(v):
                match e:
                    case ["push"]:
                        undo.append(state.copy())
                    case ["commit"]:
                        undo.pop()
                    case ["rollback"]:
                        state = undo.pop()
                    case ["set", key, val]:
                        state[key] = val
                    case ["unset", key]:
                        state[key] = None
                    case ["read"]:
                        reads.append([state[k] for k in ["color", "quota", "region"]])
            self.assertEqual(
                author.scopes_answer(author.scopes_data(v)), dict(reads=reads, root=state)
            )

    def test_fault_diagnosis_with_recursive_circuit_and_pair_cover(self):
        def output(bits, fault):
            inputs = dict(zip("abcd", map(int, bits)))
            gates = {g: (op, a, b) for g, op, a, b in author.GATES}

            def node(name):
                if fault.startswith(name + "="):
                    return int(fault[-1])
                if name in inputs:
                    return inputs[name]
                op, a, b = gates[name]
                a, b = bool(node(a)), bool(node(b))
                return int(a != b if op == "xor" else a and b if op == "and" else a or b)

            return node("g5")

        for v in range(3):
            faults, observed, costs = author.diagnosis_data(v)
            candidates = sorted(
                f for f in faults if all(output(p, f) == value for p, value in observed.items())
            )
            pairs = {
                pair
                for pair in itertools.combinations(candidates, 2)
                if any(output(p, pair[0]) != output(p, pair[1]) for p in costs)
            }
            classes = []
            for f in candidates:
                if not any(f in group for group in classes):
                    classes.append(
                        [g for g in candidates if all(output(p, f) == output(p, g) for p in costs)]
                    )
            solutions = []
            for mask in range(1 << len(costs)):
                probes = sorted(p for i, p in enumerate(costs) if mask >> i & 1)
                if all(any(output(p, f) != output(p, g) for p in probes) for f, g in pairs):
                    solutions.append((sum(costs[p] for p in probes), len(probes), probes))
            optimum = min((c, n) for c, n, _ in solutions)
            self.assertEqual(
                author.diagnosis_answer(faults, observed, costs),
                dict(
                    candidates=candidates,
                    indistinguishable=sorted(classes),
                    cost=optimum[0],
                    probe_sets=sorted(ps for c, n, ps in solutions if (c, n) == optimum),
                ),
            )

    def test_genetics_by_integer_frequency_table(self):
        for v in range(3):
            r, s, penalty = author.genetics_data(v)
            # Out of 200 gametes per parent: independent 40,000 equally weighted pairings.
            counts1 = dict(AB=100 - r, ab=100 - r, Ab=r, aB=r)
            counts2 = dict(Ab=100 - s, aB=100 - s, AB=s, ab=s)
            weights = defaultdict(int)
            hetero = 0
            for x, y in itertools.product(counts1, counts2):
                a_count = (x[0] == "A") + (y[0] == "A")
                b_count = (x[1] == "B") + (y[1] == "B")
                survivors = counts1[x] * counts2[y] * (1 if a_count == 2 else penalty)
                color = "white" if a_count == 0 else "light" if b_count == 0 else "dark"
                weights[color] += survivors
                if a_count == b_count == 1:
                    hetero += survivors
            total = sum(weights.values())

            def pair(n, d):
                return [Fraction(n, d).numerator, Fraction(n, d).denominator]

            expected = dict(
                survival=pair(total, 40000 * penalty),
                phenotypes={k: pair(weights[k], total) for k in weights},
                double_heterozygote_given_dark=pair(hetero, weights["dark"]),
            )
            self.assertEqual(author.genetics_answer(r, s, penalty), expected)
            self.assertEqual(sum(Fraction(*p) for p in expected["phenotypes"].values()), 1)

    def test_reshape_by_coordinate_mapping(self):
        for v in range(3):
            matrix = author.reshape_data(v)
            values = []
            for position in range(30):
                row, col = divmod(position, 5)
                original_row = 4 - col if row % 2 == 0 else col
                value = matrix[original_row][row]
                if position % 4 != 1 and value % 3:
                    values.append(value)
            emitted = [
                sum((-1) ** j * x for j, x in enumerate(values[: i + 1])) % 7
                for i in range(len(values))
            ]
            runs = []
            for x in emitted:
                if runs and runs[-1][0] == x:
                    runs[-1][1] += 1
                else:
                    runs.append([x, 1])
            self.assertEqual(
                author.reshape_answer(matrix),
                dict(
                    runs=runs,
                    kept=len(values),
                    checksum=sum((i + 1) * x for i, x in enumerate(emitted)),
                ),
            )

    def test_translation_reviewed_semantic_keys(self):
        # Manual truth tables: necessary all / sufficient any / sufficient not-all.
        expected = [
            [
                "unknown",
                "unknown",
                "forbidden",
                "forbidden",
                "unknown",
                "forbidden",
                "unknown",
                "forbidden",
            ],
            [
                "forbidden",
                "allowed",
                "forbidden",
                "allowed",
                "allowed",
                "forbidden",
                "forbidden",
                "forbidden",
            ],
            [
                "forbidden",
                "forbidden",
                "allowed",
                "allowed",
                "forbidden",
                "forbidden",
                "forbidden",
                "allowed",
            ],
        ]
        for v, statuses in enumerate(expected):
            self.assertEqual(author.translation_case(v)["expected"]["value"]["statuses"], statuses)

    def test_uuid_contract_checks_every_field(self):
        expected = self.questions["IF-07"].expected
        # Uniqueness was never requested, so repeated syntactically valid UUIDs pass.
        rows = [
            dict(
                index=i,
                uuid="12345678-1234-4123-8123-123456789abc",
                status="pending" if i % 2 == 0 else "completed",
            )
            for i in range(6)
        ]
        self.assertEqual(eval_format_check(json.dumps(rows), expected)[0], 1)
        for i in range(6):
            for key, val in [
                ("index", 99),
                ("index", True),
                ("index", float(i)),
                ("status", "wrong"),
                ("uuid", rows[i]["uuid"] + " extra"),
                ("extra", rows[i]["uuid"]),
            ]:
                bad = deepcopy(rows)
                bad[i][key] = val
                self.assertLess(eval_format_check(json.dumps(bad), expected)[0], 1, (i, key, val))
        for wrapped in ["prefix " + json.dumps(rows), "```json\n" + json.dumps(rows) + "\n```"]:
            self.assertLess(eval_format_check(wrapped, expected)[0], 1)
        for fields in [
            [],
            {},
            {"x": {"type": "unknown"}},
            {"x": {"type": "integer", "pattern": ".*"}},
        ]:
            report = Report()
            check_format_check(
                report,
                "test",
                {"checks": [{"type": "json", "root": "array", "item_fields": fields}]},
            )
            self.assertTrue(report.errors)

    def test_family_reporting_and_default_assembly(self):
        questions = assemble_questions(list(self.questions.values()), QualityConfig())
        new = [q for q in questions if q.id.startswith("S6-")]
        self.assertEqual(len(new), 30)
        self.assertEqual(len({q.metadata["family"] for q in new}), 10)
        self.assertTrue(
            all(q.metadata["cohort"] == "stretch-v6" and q.max_tokens == 65536 for q in new)
        )
        rows = [result_record(Result(q, "", 1 if i % 3 else 0, "test")) for i, q in enumerate(new)]
        summary = summarize(rows)
        self.assertEqual(summary["stretch"]["count"], 30)
        self.assertEqual(summary["stretch"]["families"], 10)
        self.assertAlmostEqual(summary["stretch"]["category_balanced"], 2 / 3)
        self.assertEqual(summary["challenge"]["count"], 0)
        self.assertIsNone(summarize([])["stretch"]["category_balanced"])
        report = make_report(
            [Result(q, "{}", 1, "test") for q in new],
            QualityConfig(), ClientConfig("https://fake.invalid", "unused", "fake"),
        )
        self.assertIn("Stretch-v6 subset: **100.0%**", markdown(report))
        self.assertIn("30/30 full passes across 10 families", markdown(report))

    def test_audit_separates_budgets_and_excludes_partial_cohorts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.db"
            with closing(sqlite3.connect(path)) as db:
                db.executescript((Path(__file__).parent / "app/schema.sql").read_text())
                db.execute(
                    "INSERT INTO models(id,name,base_url,api_key,model_id) VALUES(1,'fake','fake','fake','fake')"
                )
                for run, budget, status in [
                    (1, 16384, "completed"),
                    (2, 16384, "completed"),
                    (3, 65536, "completed"),
                    (4, 16384, "running"),
                ]:
                    report = json.dumps({"protocol": {"quality_max_output_tokens": budget}})
                    db.execute(
                        "INSERT INTO test_runs(id,model_id,status,test_suite_hash,quality_json) VALUES(?,1,?,'same',?)",
                        (run, status, report),
                    )
                    for id in ["Q", "missing"] if run == 1 else ["Q"]:
                        db.execute(
                            "INSERT INTO test_results(run_id,test_id,category,score,passed,quality_scored,quality_metadata_json) VALUES(?,?,'Test',1,1,1,?)",
                            (run, id, json.dumps({"scope": "capability", "scored": True})),
                        )
                db.commit()
            before = path.read_bytes()
            report = audit_database(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(report["runs"]), 4)
            self.assertEqual(len(report["cohorts"]), 2)
            key = next(k for k, c in report["cohorts"].items() if c["run_ids"] == [1, 2])
            self.assertTrue(report["item_analysis_by_cohort"][key]["Q"]["all_passed"])
            self.assertFalse(report["item_analysis_by_cohort"][key]["missing"]["all_passed"])
            self.assertFalse(report["item_analysis_by_cohort"][key]["missing"]["complete_coverage"])


if __name__ == "__main__":
    unittest.main()
