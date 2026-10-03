"""Prompt-derived v13 oracles, semantic near misses and grading regressions."""

from copy import deepcopy
from dataclasses import replace
import itertools
import inspect
import json
import sqlite3
import unittest

from evidence_cases import load_evidence_questions
from evaluators import values_equal, json_at_path
from models import Question, RequestMetrics, TokenUsage
from quality_execution import score_response
from selftest_specialists import corruptions


def score(q, answer):
    return score_response(q, json.dumps(answer), TokenUsage(), RequestMetrics(finish_reason="stop"))


def prompt_data(q):
    return json.loads(q.prompt.split("\nINPUT=", 1)[1].split("\n\nOutput contract", 1)[0])


def evidence_oracle(data):
    """Set algebra over truth-table columns, exhaustive subset model sets."""
    atoms = data["atoms"]
    universe = set(itertools.product((False, True), repeat=len(atoms)))

    def truth(formula):
        if isinstance(formula, str):
            return {world for world in universe if world[atoms.index(formula)]}
        if formula[0] == "not":
            return universe - truth(formula[1])
        a, b = truth(formula[1]), truth(formula[2])
        return {"and": a & b, "or": a | b, "xor": a ^ b,
                "iff": universe - (a ^ b), "implies": (universe - a) | b}[formula[0]]

    allowed = {s["id"]: truth(s["formula"]) for s in data["sources"]}
    ids = sorted(allowed)
    models = {frozenset(subset): universe.intersection(*(allowed[s] for s in subset))
              for size in range(len(ids) + 1) for subset in itertools.combinations(ids, size)}

    def minimal(predicate):
        accepted = []
        for subset, worlds in models.items():
            if predicate(worlds) and not any(other < subset for other in accepted):
                accepted.append(subset)
        return sorted(sorted(s) for s in accepted)

    maximum = max(len(subset) for subset, worlds in models.items() if worlds)
    best = [subset for subset, worlds in models.items() if worlds and len(subset) == maximum]
    repaired = set().union(*(models[subset] for subset in best))
    full = models[frozenset(ids)]

    def status(worlds, true):
        if not worlds:
            return "inconsistent"
        if worlds <= true:
            return "supported"
        return "refuted" if worlds.isdisjoint(true) else "unknown"

    def witness(worlds):
        return list(min(worlds)) if worlds else None

    return {"full_model_count": len(full),
            "minimal_inconsistent_cores": minimal(lambda w: not w),
            "minimum_deletions": len(ids) - maximum,
            "repairs": sorted(sorted(set(ids) - subset) for subset in best),
            "repaired_model_count": len(repaired),
            "claims": [{"id": claim["id"], "full_status": status(full, true),
                        "supports": minimal(lambda w: bool(w) and w <= true),
                        "refutations": minimal(lambda w: bool(w) and w.isdisjoint(true)),
                        "repaired_status": status(repaired, true),
                        "true_witness": witness(repaired & true),
                        "false_witness": witness(repaired - true)}
                       for claim in data["claims"] for true in [truth(claim["formula"])]]}


def trace_oracle(data):
    """A relational interpreter: SQL balances, version checks and receipt storage."""
    db = sqlite3.connect(":memory:")
    try:
        db.executescript("CREATE TABLE accounts(id TEXT PRIMARY KEY, balance INTEGER, version INTEGER);"
                         "CREATE TABLE receipts(key TEXT PRIMARY KEY, arguments TEXT, result TEXT);")
        db.executemany("INSERT INTO accounts VALUES (:id,:balance,:version)", data["accounts"])
        output = []
        for index, call in enumerate(data["calls"]):
            resolved, bad = dict(call["args"]), False
            for k, value in resolved.items():
                if type(value) is dict:
                    valid = (set(value) == {"ref", "field"} and type(value["ref"]) is int
                             and 0 <= value["ref"] < index and type(value["field"]) is str)
                    if not valid or value["field"] not in output[value["ref"]]:
                        bad = True
                        break
                    resolved[k] = output[value["ref"]][value["field"]]
            tool = call["tool"]
            required = {"account"} if tool == "read" else {"src", "dst", "amount", "src_version", "dst_version", "key"}
            texts = ["account"] if tool == "read" else ["src", "dst", "key"]
            numeric = [] if tool == "read" else ["amount", "src_version", "dst_version"]
            result = None
            if bad:
                status = "bad_ref"
            elif tool not in ("read", "transfer"):
                status = "unknown_tool"
            elif (set(resolved) != required or any(type(resolved[k]) is not str for k in texts)
                  or any(type(resolved[k]) is not int for k in numeric)):
                status = "schema"
            elif tool == "read":
                row = db.execute("SELECT balance,version FROM accounts WHERE id=?", (resolved["account"],)).fetchone()
                status = "read" if row else "missing"
                if row:
                    result = {"status": status, "balance": row[0], "version": row[1]}
            elif resolved["amount"] <= 0 or min(resolved["src_version"], resolved["dst_version"]) < 0 or not resolved["key"]:
                status = "schema"
            else:
                encoded = json.dumps(resolved, sort_keys=True)
                previous = db.execute("SELECT arguments,result FROM receipts WHERE key=?", (resolved["key"],)).fetchone()
                src = db.execute("SELECT balance,version FROM accounts WHERE id=?", (resolved["src"],)).fetchone()
                dst = db.execute("SELECT balance,version FROM accounts WHERE id=?", (resolved["dst"],)).fetchone()
                if previous:
                    status = "key_conflict"
                    if encoded == previous[0]:
                        result = json.loads(previous[1])
                elif not src or not dst:
                    status = "missing"
                elif resolved["src"] == resolved["dst"]:
                    status = "same_account"
                elif (src[1], dst[1]) != (resolved["src_version"], resolved["dst_version"]):
                    status = "stale"
                elif src[0] < resolved["amount"]:
                    status = "funds"
                else:
                    status = "applied"
                    db.execute("UPDATE accounts SET balance=balance-?,version=version+1 WHERE id=?", (resolved["amount"], resolved["src"]))
                    db.execute("UPDATE accounts SET balance=balance+?,version=version+1 WHERE id=?", (resolved["amount"], resolved["dst"]))
                    result = {"status": status, "src_version": src[1] + 1, "dst_version": dst[1] + 1}
                    db.execute("INSERT INTO receipts VALUES (?,?,?)", (resolved["key"], encoded, json.dumps(result)))
            output.append(result if result is not None else {"status": status})
        return {"results": output, "accounts": [{"id": i, "balance": b, "version": v}
                                               for i, b, v in db.execute("SELECT * FROM accounts ORDER BY id")],
                "cached_keys": [key for (key,) in db.execute("SELECT key FROM receipts ORDER BY key")],
                "applied_count": db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]}
    finally:
        db.close()


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = [q for seed in (19, 23) for variant in (0, 1)
                         for q in load_evidence_questions(seed, variant)]

    def test_prompt_derived_independent_answers(self):
        for q in self.questions:
            oracle = evidence_oracle if "evidence-audit" in q.id else trace_oracle
            self.assertEqual(q.expected["value"], oracle(prompt_data(q)), q.id)

    def test_every_corruption_is_scored_and_rejected(self):
        rejected = 0
        for q in self.questions:
            good = score(q, q.expected["value"])
            self.assertTrue(good.passed, (q.id, good.detail))
            self.assertAlmostEqual(good.evaluation.criterion_achievement, 1)
            for bad in corruptions(q.expected["value"]):
                result = score(q, bad)
                self.assertTrue(result.is_scored, (q.id, result.detail))
                self.assertFalse(result.passed, (q.id, bad))
                rejected += 1
            for text in ("```json\n" + json.dumps(q.expected["value"]) + "\n```", "{}", "null"):
                result = score_response(q, text, TokenUsage(), RequestMetrics(finish_reason="stop"))
                self.assertTrue(result.is_scored)
                self.assertFalse(result.passed)
        print(f"Rejected {rejected} corrupted v13 answers")

    def test_pairs_and_output_contracts(self):
        contracts = {}
        for q in self.questions:
            contract = q.prompt.split("Output contract (types, not answer values): ")[1].splitlines()[0]
            family = q.metadata["family"]
            self.assertEqual(contract, contracts.setdefault(family, contract))
        for seed in (19, 23):
            for family, array, field in (("evidence-audit", "sources", "formula"),
                                         ("typed-tool-trace", "calls", "args")):
                pair = [q for q in self.questions if q.metadata["seed"] == seed and family in q.id]
                first, second = map(prompt_data, pair)
                differences = [k for k in first if first[k] != second[k]]
                self.assertEqual(differences, [array])
                changed = [(a, b) for a, b in zip(first[array], second[array]) if a != b]
                self.assertEqual(len(changed), 1)
                self.assertEqual(set(k for k in changed[0][0] if changed[0][0][k] != changed[0][1][k]), {field})
                if family == "evidence-audit":
                    a, b = (q.expected["value"] for q in pair)
                    self.assertGreater(a["full_model_count"], 0)
                    self.assertEqual(b["full_model_count"], 0)
                    self.assertGreater(len(b["repairs"]), 1)
                    self.assertTrue(any(row["repaired_status"] == "unknown" for row in b["claims"]))
                else:
                    self.assertEqual(pair[0].expected["value"]["results"][4]["status"], "applied")
                    self.assertEqual(pair[1].expected["value"]["results"][4]["status"], "key_conflict")
                    self.assertEqual(pair[0].expected["value"]["accounts"], pair[1].expected["value"]["accounts"])

    def test_nonvacuous_evidence_and_exhaustive_repairs(self):
        for q in self.questions:
            if "evidence-audit" not in q.id:
                continue
            answer = q.expected["value"]
            data = prompt_data(q)
            tautology = next(i for i, c in enumerate(data["claims"]) if c["formula"][0] == "or" and
                             isinstance(c["formula"][2], list) and c["formula"][2][0] == "not")
            self.assertEqual(answer["claims"][tautology]["supports"], [[]])
            self.assertEqual(answer["claims"][tautology]["refutations"], [])
            if answer["minimum_deletions"]:
                # Contradiction never licenses an arbitrary claim as supported.
                self.assertEqual({r["full_status"] for r in answer["claims"]}, {"inconsistent"})
                wrong = deepcopy(answer)
                wrong["repairs"] = wrong["repairs"][:1]
                self.assertFalse(score(q, wrong).passed)
            self.assertTrue(any(len(e) >= 3 for row in answer["claims"] for e in row["supports"] + row["refutations"]))

    def test_trace_caches_only_success_and_preserves_original_receipts(self):
        for q in self.questions:
            if "typed-tool-trace" not in q.id:
                continue
            answer = q.expected["value"]
            results = answer["results"]
            self.assertEqual(results[3], results[2])
            self.assertEqual([r["status"] for r in results[5:10]], ["stale", "funds", "applied", "bad_ref", "schema"])
            self.assertEqual(results[7], results[16])
            self.assertEqual(answer["applied_count"], 3)
            self.assertEqual(sum(row["balance"] for row in answer["accounts"]),
                             sum(row["balance"] for row in prompt_data(q)["accounts"]))
            # Give each call the same diagnostic share, even for longer receipts.
            for i in (2, 3, 5, 6):
                wrong = deepcopy(answer)
                wrong["results"][i] = {"status": "invented"}
                result = score(q, wrong)
                self.assertAlmostEqual(result.evaluation.criterion_achievement, 1 - 1 / 4 / len(results))
            wrong = deepcopy(answer)
            wrong["results"][2]["status"] = "funds"
            result = score(q, wrong)
            self.assertAlmostEqual(result.evaluation.criterion_achievement, 1 - 1 / 4 / len(results))
            for criterion in result.evaluation.criteria:
                if criterion.criterion_id in {"json:results[2].src_version", "json:results[2].dst_version"}:
                    self.assertEqual(criterion.reason_code, "blocked_by_dependency")
                    self.assertEqual(criterion.earned, 0)

    def test_format_diagnostics_do_not_parse_untrusted_labels(self):
        q = Question("label", "Creative Writing", "Check format", "format_check",
                     {"checks": [{"type": "contains", "value": "fake: PASS)", "id": "fake"}]})
        result = score_response(q, "ordinary words", TokenUsage(), RequestMetrics(finish_reason="stop"))
        self.assertFalse(result.passed)
        self.assertEqual(result.evaluation.criteria[0].status, "fail")
        self.assertEqual(result.evaluation.criteria[0].earned, 0)
        self.assertEqual(result.evaluation.contract_score, 0)
        # Regex descriptions are source text too; changing one cannot change scoring.
        q = replace(q, expected={"checks": [{"type": "regex", "value": "^yes$", "description": "test: PASS)"}]})
        result = score_response(q, "no", TokenUsage(), RequestMetrics(finish_reason="stop"))
        self.assertEqual(result.evaluation.criteria[0].status, "fail")

    def test_large_numeric_outputs_are_scored(self):
        q = Question("large", "Code Generation", "Return one", "code_exec",
                     [{"function": "f", "expected": 1.0}])
        for expression in ("10**400", "-(10**400)"):
            result = score_response(q, f"def f(): return {expression}", TokenUsage(), RequestMetrics(finish_reason="stop"))
            self.assertTrue(result.is_scored, result.detail)
            self.assertFalse(result.passed)
            self.assertEqual(result.evaluation.criterion_achievement, 0)
        for sign in (-1, 1):
            self.assertTrue(values_equal(sign * 10**400, sign * 10**400))
            self.assertFalse(values_equal(sign * 10**400, sign * 10**400 + 1, rel=1e-6))
            self.assertFalse(values_equal(sign * 10**400, -sign * 10**400, rel=1e-6))
        # A mixed float/integer comparison at a huge rational boundary must
        # distinguish a difference of one without binary64 rounding.
        target = 2**1100
        self.assertTrue(values_equal(0.0, target, rel=1.0))
        self.assertFalse(values_equal(-1.0, target, rel=1.0))

    def test_integer_output_contracts_are_explicit_and_validated(self):
        from validate_suite import Report, check_json_match
        def float_substitutions(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    for bad in float_substitutions(child):
                        yield {**value, key: bad}
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    for bad in float_substitutions(child):
                        yield value[:i] + [bad] + value[i + 1:]
            elif type(value) is int:
                yield float(value)

        rejected = 0
        for q in self.questions:
            for bad in float_substitutions(q.expected["value"]):
                result = score(q, bad)
                self.assertTrue(result.is_scored)
                self.assertFalse(result.passed, q.id)
                self.assertEqual(result.evaluation.contract_score, 0)
                rejected += 1
            for path in q.expected["integer_paths"]:
                value, found = json_at_path(q.expected["value"], path)
                self.assertTrue(found)
                # One-field contracts isolate numeric syntax from other requirements.
                probe = Question("integer", "Reasoning", "Return integer", "json_match",
                                 {"value": {"count": value}, "integer_paths": ["count"], "strict_json": True})
                good = score(probe, {"count": value})
                self.assertTrue(good.passed)
                wrong = score(probe, {"count": float(value)})
                self.assertFalse(wrong.passed)
                self.assertEqual(wrong.evaluation.contract_score, 0)
                self.assertEqual(wrong.evaluation.criteria[1].reason_code, "type_mismatch")
                # Preserve numeric equivalence for tasks that request numbers.
                probe.expected.pop("integer_paths")
                self.assertTrue(score(probe, {"count": float(value)}).passed)
        for paths in (["missing"], ["count", "count"], ["flag"], ["text"], [True], "count", [{}]):
            report = Report()
            check_json_match(report, "bad", {"value": {"count": 1, "flag": True, "text": "1"}, "integer_paths": paths})
            self.assertTrue(report.errors, paths)
        report = Report()
        check_json_match(report, "bad-alias", {"value": {"count": 1}, "integer_paths": ["count"],
                                               "value_aliases": {"count": [1.0]}})
        self.assertTrue(report.errors)
        for path, ignore in (("count", ["count"]), ("stats.count", ["stats"])):
            report = Report()
            check_json_match(report, "bad-ignore", {"value": {"count": 1, "stats": {"count": 2}},
                "integer_paths": [path], "ignore_keys": ignore})
            self.assertTrue(report.errors)
        print(f"Rejected {rejected} numerically equal float substitutions")

    def test_transformed_code_fixtures_reject_small_range_shortcuts(self):
        from challenge_oracles import CODE
        from quality_suite import load_questions
        from test_loader import load_all_tests
        from pathlib import Path
        static = {q.id: q for q in load_all_tests(Path(__file__).parent / "tests")}
        active = {q.id: q for q in load_questions()}
        wrappers = {
            "AC2-02": "def interval_overlay(intervals):\n    return _reference([[int(float(a)), int(float(b))] for a,b in intervals])\n",
            "AC2-04": "def versioned_read(events, queries):\n    return _reference([[k,int(float(t)),v] for k,t,v in events], [[k,int(float(t))] for k,t in queries])\n",
        }
        for ident, wrapper in wrappers.items():
            original = inspect.getsource(CODE[ident])
            mutant = original.replace(f"def {CODE[ident].__name__}(", "def _reference(", 1) + "\n" + wrapper
            # These plausible shortcuts really pass the old public examples.
            for q, passed in ((static[ident], True), (active[ident], False)):
                result = score_response(q, mutant, TokenUsage(), RequestMetrics(finish_reason="stop"))
                self.assertTrue(result.is_scored, (ident, result.detail))
                self.assertEqual(result.passed, passed, (ident, result.detail))


if __name__ == "__main__":
    unittest.main()
