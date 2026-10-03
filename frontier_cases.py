"""Original v10 tasks: interacting constraints with executable answer derivations.

Public authoring fixtures inspired by third-party evaluation methods; these are
not copies of benchmark datasets or an empirically calibrated leaderboard.
"""

from fractions import Fraction
import itertools
import json
import unicodedata

from models import Question
from rigorous_cases import structured


def task(family, category, variant, seed, prompt, answer, data):
    q = structured(family, category, variant, "evaluation", prompt, answer)
    q.id = f"Q10-{family}-s{seed}-v{variant + 1:02d}"
    q.metadata.update(family=f"Q10-{family}", oracle_input=data, seed=seed, variant=variant)
    q.source = "Original rigorous-v10; design sources and oracle checks in BENCHMARK_DESIGN.md"
    return q


def robust_portfolio(rng, variant, seed):
    projects = [
        {
            "id": f"p{i}",
            "cost": rng.randint(2, 9),
            "returns": [rng.randint(-4, 16) for _ in range(3)],
        }
        for i in range(10)
    ]
    budget = rng.randint(17, 24)
    dependencies = [["p4", "p1"], ["p7", "p4"], ["p9", "p2"]]
    exclusions = [["p1", "p6"], ["p3", "p8"]]
    candidates = []
    for bits in itertools.product((False, True), repeat=len(projects)):
        selected = {p["id"] for p, bit in zip(projects, bits) if bit}
        cost = sum(p["cost"] for p in projects if p["id"] in selected)
        if not 3 <= len(selected) <= 6 or cost > budget:
            continue
        if any(a in selected and b not in selected for a, b in dependencies):
            continue
        if any(a in selected and b in selected for a, b in exclusions):
            continue
        returns = [sum(p["returns"][i] for p in projects if p["id"] in selected) for i in range(3)]
        candidates.append(((-min(returns), -sum(returns), cost, sorted(selected)), returns))
    candidates.sort()
    if len(candidates) < 2:
        raise ValueError("Portfolio fixture needs at least two feasible solutions")
    key, returns = candidates[0]
    answer = {
        "selected": key[3],
        "scenario_returns": returns,
        "worst_return": -key[0],
        "total_return": -key[1],
        "cost": key[2],
        "runner_up": candidates[1][0][3],
        "worst_return_gap": candidates[1][0][0] - key[0],
    }
    data = {
        "projects": projects,
        "budget": budget,
        "dependencies": dependencies,
        "exclusions": exclusions,
    }
    prompt = (
        "Choose 3..6 indivisible projects under a total cost budget. For [a,b] dependencies, "
        "selecting a requires b. Never select both members of an exclusion pair. Returns are "
        "additive within each of THREE scenarios, which may contain losses. Rank all feasible "
        "portfolios by greatest WORST scenario return, then greatest SUM of scenario returns, "
        "then least cost, then lexicographically smallest sorted ID list (Python list order). "
        "Return the best portfolio, its three scenario returns in given order, worst_return, "
        "total_return, cost, the distinct second-ranked portfolio as runner_up, and the best "
        "minus runner-up worst_return_gap. All arithmetic is exact; the gap can be zero.\n"
        + json.dumps(data)
    )
    return task("robust-portfolio", "Finances", variant, seed, prompt, answer, data)


def causal_intervention(rng, variant, seed):
    priors = [Fraction(rng.randint(1, 4), 5) for _ in range(4)]
    # U,V,W,Z are independent exogenous bits; intervention changes B alone.
    observational = []
    for u, v, w, z in itertools.product((0, 1), repeat=4):
        weight = Fraction(1)
        for bit, probability in zip((u, v, w, z), priors):
            weight *= probability if bit else 1 - probability
        a = u ^ v
        b = int(bool(a or w))

        def outcome(b_value):
            c = b_value ^ u
            return int(bool((c and not z) or (a and z)))

        observational.append((weight, b, outcome(b), outcome(1), outcome(0)))
    denom = sum(p for p, b, _, _, _ in observational if b)
    observed = sum(p for p, b, y, _, _ in observational if b and y) / denom
    treated = sum(p for p, _, _, y1, _ in observational if y1)
    untreated = sum(p for p, _, _, _, y0 in observational if y0)
    benefit = sum(p for p, _, _, y1, y0 in observational if y1 and not y0)
    harm = sum(p for p, _, _, y1, y0 in observational if y0 and not y1)

    def pair(value):
        return [value.numerator, value.denominator]

    answer = {
        "observational": pair(observed),
        "interventional": pair(treated),
        "untreated": pair(untreated),
        "average_effect": pair(treated - untreated),
        "benefit_probability": pair(benefit),
        "harm_probability": pair(harm),
    }
    data = {"priors": [pair(p) for p in priors]}
    prompt = (
        "In this fully specified structural causal model, U,V,W,Z are independent Boolean "
        "exogenous variables, with probabilities of being 1 given below in that order. "
        "A=U XOR V; B=A OR W; C=B XOR U; Y=(C AND NOT Z) OR (A AND Z). "
        "An intervention do(B=b) replaces ONLY the equation for B, leaving the same U,V,W,Z "
        "distribution and all other equations. Counterfactual Y1 and Y0 use the SAME exogenous "
        "bits in both interventions. Return observational=P(Y=1|B=1), interventional=P(Y1=1), "
        "untreated=P(Y0=1), average_effect=E[Y1-Y0], benefit_probability=P(Y1=1,Y0=0), "
        "and harm_probability=P(Y1=0,Y0=1). Represent every probability/effect as a reduced "
        "[numerator,positive_denominator], including zero as [0,1].\n" + json.dumps(data)
    )
    return task("causal-counterfactual", "Terminal Science", variant, seed, prompt, answer, data)


def sql_semantics(rng, variant, seed):
    customers = [["a", "red"], ["b", "red"], ["c", "blue"], ["d", "red"], ["e", "red"]]
    orders = [
        ["o1", "a", "paid", rng.randint(10, 30)],
        ["o2", "a", "paid", None],
        ["o3", "b", "paid", rng.randint(10, 30)],
        ["o4", "b", "void", 90],
        ["o5", "c", "paid", 70],
        ["o6", "d", "paid", None],
    ]
    tags = [
        ["o1", "urgent"],
        ["o1", "urgent"],
        ["o1", "gift"],
        ["o2", None],
        ["o3", "gift"],
        ["o3", "rush"],
        ["o6", "gift"],
    ]
    tags.extend([["o1", "urgent"]] * rng.randint(0, 2))
    tags.extend([["o3", rng.choice([None, "gift", "rush"])]] * rng.randint(0, 2))
    orders.append(
        ["o7", "b", rng.choice(["paid", "void"]), rng.choice([None, rng.randint(-5, 20)])]
    )
    if rng.randrange(2):
        orders.append(["o8", "a", "paid", rng.randint(-8, 20)])
    rows = []
    for customer, tenant in customers:
        if tenant != "red":
            continue
        joined = []
        for oid, cid, status, amount in orders:
            if cid == customer and status == "paid":
                matches = [tag for ident, tag in tags if ident == oid] or [None]
                joined.extend((oid, amount, tag) for tag in matches)
        joined = joined or [(None, None, None)]
        rows.append(
            {
                "customer": customer,
                "joined_rows": len(joined),
                "orders": len({oid for oid, _, _ in joined if oid is not None}),
                "nonnull_amounts": sum(amount is not None for _, amount, _ in joined),
                "sum_amount": sum(amount for _, amount, _ in joined if amount is not None)
                if any(amount is not None for _, amount, _ in joined)
                else None,
                "tags": sorted({tag for _, _, tag in joined if tag is not None}),
            }
        )
    # SQL UNKNOWN: even a value absent from the list cannot pass NOT IN (...,NULL).
    answer = {
        "rows": rows,
        "not_in_rows": [],
        "distinct_order_total": sum(
            o[3]
            for o in orders
            if o[2] == "paid" and o[1] in {"a", "b", "d", "e"} and o[3] is not None
        ),
    }
    data = {"customers": customers, "orders": orders, "tags": tags}
    prompt = (
        "Evaluate this relational job using SQL bag semantics (duplicates remain), SQL NULL, "
        "and case-sensitive identifiers. Tables: customers(id,tenant); "
        "orders(id,customer,status,amount); tags(order_id,tag). Start with tenant='red' "
        "customers, LEFT JOIN orders ON customer=id AND status='paid', then LEFT JOIN tags "
        "ON order_id=orders.id. Do not move the paid predicate into WHERE. Group by customer. "
        "For each customer in ID order report joined_rows=COUNT(*), orders=COUNT(DISTINCT order ID), "
        "nonnull_amounts=COUNT(amount), sum_amount=SUM(amount), and tags=sorted distinct nonnull "
        "tag strings. SUM over all-null groups is null, not zero. Separately return not_in_rows "
        "as IDs selected by customer ID NOT IN ('z',NULL). Finally compute distinct_order_total "
        "by summing every paid order of red customers ONCE before the tag join; ignore null amounts.\n"
        + json.dumps(data)
    )
    return task(
        "sql-null-cardinality", "Reading Comprehension", variant, seed, prompt, answer, data
    )


def clause_true(clause, bits):
    return any(bits[index] == truth for index, truth in clause)


def weighted_repair(rng, variant, seed):
    required = [[[0, 1]], [[1, 0]], [[2, 0], [3, 1]]]
    optional = [
        {"id": "c00", "weight": rng.randint(2, 8), "clause": [[0, 0]]},
        {"id": "c01", "weight": rng.randint(2, 8), "clause": [[1, 1]]},
    ]
    for i in range(2, 13):
        variables = rng.sample(range(7), rng.randint(1, 3))
        optional.append(
            {
                "id": f"c{i:02d}",
                "weight": rng.randint(1, 9),
                "clause": [[j, rng.randint(0, 1)] for j in variables],
            }
        )
    candidates = []
    for bits in itertools.product((0, 1), repeat=7):
        if not all(clause_true(c, bits) for c in required):
            continue
        removed = [c["id"] for c in optional if not clause_true(c["clause"], bits)]
        cost = sum(c["weight"] for c in optional if c["id"] in removed)
        candidates.append((cost, len(removed), removed, list(bits)))
    best = min(candidates)
    answer = {
        "removed": best[2],
        "cost": best[0],
        "assignment": best[3],
        "remaining_solutions": sum(
            all(clause_true(c["clause"], bits) for c in optional if c["id"] not in best[2])
            for bits in itertools.product((0, 1), repeat=7)
            if all(clause_true(c, bits) for c in required)
        ),
    }
    data = {"required": required, "optional": optional}
    prompt = (
        "Repair an inconsistent Boolean policy over X0..X6 by removing optional clauses only. "
        "A literal [i,b] means Xi=b; each clause is OR of its literals; the retained clauses "
        "and all REQUIRED clauses must hold simultaneously. Minimize total removed weight, "
        "then number removed, then the lexicographically smallest sorted removed-ID list. "
        "For this chosen repair return the lexicographically smallest satisfying 0/1 assignment "
        "in X0..X6 order and remaining_solutions, the number of ALL satisfying assignments. "
        "Do not count only optimal assignments, and never remove a required clause.\n"
        + json.dumps(data)
    )
    return task("weighted-policy-repair", "Logical Reasoning", variant, seed, prompt, answer, data)


def vector_frontier(rng, variant, seed):
    records = [
        {"id": "v0", "clock": [1, 0, 0], "value": "old", "approved": True},
        {"id": "v1", "clock": [2, 1, 0], "value": "blue", "approved": True},
        {"id": "v2", "clock": [1, 2, 0], "value": None, "approved": True},
        {"id": "v3", "clock": [1, 1, 2], "value": "green", "approved": True},
        {"id": "v4", "clock": [9, 9, 9], "value": "draft", "approved": False},
    ]
    for i in range(5, 12):
        records.append(
            {
                "id": f"v{i:02d}",
                "clock": [rng.randint(0, 3) for _ in range(3)],
                "value": rng.choice([None, "blue", "green", "amber"]),
                "approved": True,
            }
        )
    approved = [r for r in records if r["approved"]]

    def dominates(a, b):
        return all(x >= y for x, y in zip(a, b)) and a != b

    heads = sorted(
        [r for r in approved if not any(dominates(s["clock"], r["clock"]) for s in approved)],
        key=lambda r: r["id"],
    )
    values = sorted({r["value"] for r in heads if r["value"] is not None})
    tombstone = any(r["value"] is None for r in heads)
    answer = {
        "heads": [r["id"] for r in heads],
        "live_values": values,
        "has_tombstone": tombstone,
        "conflict": len(values) + int(tombstone) > 1,
        "dominated": sorted(r["id"] for r in approved if r not in heads),
    }
    data = {"records": records}
    prompt = (
        "Reconcile a multiwriter register without total-ordering vector clocks. Ignore unapproved "
        "records entirely. Clock A causally dominates B iff every component of A>=B AND at least "
        "one is greater. Equal clocks do NOT dominate; keep both record IDs. The frontier contains "
        "ALL approved records not dominated by any approved record. Tombstone value null removes "
        "only causally dominated versions; a concurrent live value remains a possible value. "
        "Return heads (sorted frontier IDs), live_values (sorted distinct nonnull values), "
        "has_tombstone, conflict (more than one distinct frontier value, counting null as one), "
        "and dominated (sorted approved IDs outside the frontier). No last-arrival tie breaker.\n"
        + json.dumps(data)
    )
    return task("causal-register", "Agentic Use Cases", variant, seed, prompt, answer, data)


UNICODE_REFERENCE = """import unicodedata
def canonical_labels(records):
    chosen = {}
    for i, row in enumerate(records):
        raw, version, deleted, payload = row
        key = unicodedata.normalize('NFC', unicodedata.normalize('NFKC', raw).strip().casefold())
        if not key: continue
        order = (version, i)
        if key not in chosen or order > chosen[key][0]: chosen[key] = (order, deleted, payload)
    return [[key, row[2]] for key, row in sorted(chosen.items()) if not row[1]]
"""


def unicode_case(rng, variant, seed):
    samples = [
        [],
        [["Straße", 1, False, "a"], ["STRASSE", 2, True, "b"]],
        [[" e\u0301 ", 3, False, "a"], ["É", 3, False, "b"]],
        [["①", 2, False, "circle"], ["1", 1, True, "old"]],
        [["\u00a0\u2003", 7, False, "blank"], ["İ", 2, False, "dot"], ["I", 9, False, "plain"]],
    ]
    forms = [
        "Straße",
        "STRASSE",
        "K",
        "K",
        "Ｋ",
        "①",
        "1",
        "É",
        "e\u0301",
        "Σ",
        "ς",
        "σ",
        "\ufeff",
        " ",
    ]
    for _ in range(20):
        samples.append(
            [
                [
                    rng.choice(forms),
                    rng.randrange(5),
                    bool(rng.randrange(3) == 0),
                    rng.choice([None, 0, [1, 2], {"tag": "value"}]),
                ]
                for _ in range(rng.randint(8, 22))
            ]
        )
    # Host oracle is separate from the reference code evaluated by the child.
    expected = []
    for records in samples:
        groups = {}
        for index, (raw, version, deleted, payload) in enumerate(records):
            key = unicodedata.normalize(
                "NFC", unicodedata.normalize("NFKC", raw).strip().casefold()
            )
            if key:
                groups.setdefault(key, []).append((version, index, deleted, payload))
        value = [
            [key, max(rows, key=lambda r: r[:2])[3]]
            for key, rows in sorted(groups.items())
            if not max(rows, key=lambda r: r[:2])[2]
        ]
        expected.append(
            {
                "id": f"unicode-{len(expected):03d}",
                "function": "canonical_labels",
                "args": [records],
                "expected": value,
                "relative": 0,
                "tolerance": 0,
                "preserve_inputs": True,
            }
        )
    boundary = 5
    return Question(
        f"Q10-unicode-revisions-s{seed}-v{variant + 1:02d}",
        "Code Generation",
        "Implement canonical_labels(records) in Python 3. Each row is [label,version,deleted,payload], "
        "where label is a Unicode string, version is a nonnegative integer, deleted is Boolean, "
        "and payload is any JSON value. Canonicalize the label in exactly this order: NFKC, "
        "str.strip(), str.casefold(), then NFC. Ignore rows whose resulting key is empty. For each "
        "canonical key select the greatest version, ties by LAST physical input index. A selected "
        "tombstone deletes the key; do not resurrect an older row. Return [[key,payload],...] "
        "sorted by Unicode codepoint order of key. Null payload is live data when deleted=false. "
        "Do not replace casefold with lower, strip BOMs specially, or mutate inputs. "
        "Return one fenced Python code block, standard library only. All fixtures must pass.",
        "code_exec",
        expected,
        difficulty="expert",
        rubric=[
            {
                "id": f["id"],
                "mandatory": True,
                "dimension": "content",
                "group": "boundary" if i < boundary else "generated-combination",
                "weight": 0.5 / (boundary if i < boundary else len(expected) - boundary),
            }
            for i, f in enumerate(expected)
        ],
        metadata={"family": "Q10-unicode-revisions", "seed": seed, "variant": variant},
        source="Original rigorous-v10; EvalPlus-inspired boundary and combination fixtures",
    )


BUILDERS = (
    robust_portfolio,
    causal_intervention,
    sql_semantics,
    weighted_repair,
    vector_frontier,
    unicode_case,
)


def load_frontier_questions(seed, variant):
    from quality_suite import rng_for

    return [
        builder(rng_for(seed, variant, builder.__name__), variant, seed) for builder in BUILDERS
    ]
