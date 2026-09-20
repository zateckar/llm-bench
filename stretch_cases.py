"""Original stretch-v6 tasks. Regenerate tests/stretch.yaml without model calls.

These are design adaptations, not imported benchmark items or equivalent scores.
Answers are computed offline and never included in the model's prompt.
"""

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import itertools
import json
from pathlib import Path
import random

import yaml


SOURCES = {
    "reasoning": "https://github.com/google-deepmind/bbeh",
    "data": "https://github.com/LiveBench/LiveBench",
    "instructions": "https://github.com/allenai/IFBench",
}


def case(family, variant, category, prompt, answer, source="reasoning"):
    return {
        "id": f"S6-{family}-{variant + 1:02d}",
        "category": category,
        "difficulty": "expert",
        "source": f"Original stretch-v6; design inspiration: {SOURCES[source]}",
        "description": "Compositional task with an executable oracle; difficulty needs fresh calibration.",
        "prompt": prompt.strip() + "\nReturn only one JSON object with exactly the requested keys. "
        "Arrays have the order specified above; do not add prose.",
        "evaluator": "json_match",
        "expected": {"value": answer, "mode": "exact", "strict_json": True},
        "max_tokens": 65536,
    }


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def planning_data(v):
    rng = random.Random(6100 + v)
    states = [f"s{i}" for i in range(8)]
    remedies = [
        "repair-A",
        "repair-B",
        "repair-C",
        "repair-A",
        "repair-C",
        "repair-B",
        "repair-D",
        "repair-D",
    ]
    rng.shuffle(remedies)
    tests = []
    for i in range(7):
        bits = (
            [int((j >> i) & 1) for j in range(8)] if i < 3 else [rng.randrange(2) for _ in states]
        )
        tests.append({"id": f"T{i}", "cost": rng.randint(1, 5), "outcomes": bits})
    return states, remedies, tests


def planning_answer(remedies, tests):
    @lru_cache(None)
    def cost(belief):
        if len({remedies[i] for i in belief}) == 1:
            return 0
        choices = []
        for t in tests:
            parts = [tuple(i for i in belief if t["outcomes"][i] == bit) for bit in (0, 1)]
            if all(parts):
                choices.append(t["cost"] + max(cost(p) for p in parts))
        return min(choices)

    belief = tuple(range(len(remedies)))
    first = {}
    for t in tests:
        parts = [tuple(i for i in belief if t["outcomes"][i] == bit) for bit in (0, 1)]
        first[t["id"]] = t["cost"] + max(cost(p) for p in parts) if all(parts) else None
    best = cost(belief)
    return {
        "worst_cost": best,
        "optimal_first_tests": sorted(k for k, c in first.items() if c == best),
        "cost_by_first_test": first,
    }


def planning_case(v):
    states, remedies, tests = planning_data(v)
    return case(
        "policy",
        v,
        "Agentic Use Cases",
        f"""
An incident has one hidden state, fixed throughout the run. Every state below is possible.
Only the named remedy for the true state is safe. Tests are deterministic, reveal a binary
outcome, have the stated positive cost and do not change the state. You may adapt the next
test to previous outcomes. You can stop as soon as all remaining states have the SAME
remedy; you need not identify the exact state. Minimize the WORST total test cost, not the
mean or the number of tests. Remedy application costs zero. Repeating a test adds no evidence.
States in column order: {dump(states)}
Corresponding safe remedies: {dump(remedies)}
Tests: {dump(tests)}
Return worst_cost, optimal_first_tests (ALL minimizing first test IDs, sorted), and
cost_by_first_test (test ID -> minimum worst total cost when forced to start with that test,
including its cost). A test that never splits the initial states has value null in that map
and is not a candidate first test. Branches may stop at different depths.
""",
        planning_answer(remedies, tests),
    )


def tool_data(v):
    rng = random.Random(6200 + v)
    calls = [
        ("C0", ["auth"], ["snapshot"], 2),
        ("C1", ["snapshot"], ["a", "b"], 4),
        ("C2", ["snapshot"], ["c", "d"], 4),
        ("C3", ["a"], ["e"], 2),
        ("C4", ["b", "c"], ["e", "f"], 3),
        ("C5", ["auth"], ["b", "d"], 6),
        ("C6", ["d"], ["c", "f"], 2),
        ("C7", ["e", "f"], ["receipt"], 1),
        ("C8", ["a", "d"], ["receipt"], 5),
        ("C9", ["auth"], ["a", "c", "e"], 9),
    ]
    return [
        {"id": i, "requires": req, "produces": out, "cost": c + rng.randrange(3)}
        for i, req, out, c in calls
    ]


def tool_answer(calls):
    # Exhaustive reachable subsets, carrying the cheapest lexicographic sequence.
    best = {0: (0, ())}
    for mask in range(1 << len(calls)):
        if mask not in best:
            continue
        available = {"auth"} | {
            x for i, c in enumerate(calls) if mask >> i & 1 for x in c["produces"]
        }
        for i, c in enumerate(calls):
            if mask >> i & 1 or not set(c["requires"]) <= available:
                continue
            target = mask | 1 << i
            candidate = (best[mask][0] + c["cost"], best[mask][1] + (c["id"],))
            if target not in best or candidate < best[target]:
                best[target] = candidate
    goals = {"a", "b", "c", "d", "e", "f", "receipt"}
    winners = []
    for mask, (cost, seq) in best.items():
        available = {x for i, c in enumerate(calls) if mask >> i & 1 for x in c["produces"]}
        if goals <= available:
            winners.append((cost, len(seq), seq))
    cost, count, seq = min(winners)
    return {"cost": cost, "call_count": count, "calls": list(seq)}


def tool_case(v):
    calls = tool_data(v)
    return case(
        "toolplan",
        v,
        "Tool Using",
        f"""
Plan calls for a read-only evidence collection job. Initially only token auth is available.
Each call may run at most once and only when ALL its required tokens already exist. It adds
all its produced tokens; tokens are never consumed. No parallel calls. The final state must
contain a,b,c,d,e,f,receipt. A produced receipt does not remove the other requirements.
Minimize total cost; break ties by fewer calls, then lexicographically smallest sequence of
call IDs. Ignore tools not listed. Calls: {dump(calls)}
Return {{"cost":integer,"call_count":integer,"calls":[IDs in execution order]}}.
""",
        tool_answer(calls),
        "data",
    )


def temporal_data(v):
    rng = random.Random(6300 + v)
    rows = []
    for account in range(7):
        for revision in range(5):
            start = rng.randrange(0, 9)
            rows.append(
                {
                    "id": f"r{account}{revision}",
                    "account": f"A{account}",
                    "from": start,
                    "to": start + rng.randrange(2, 8),
                    "recorded": rng.randrange(1, 12),
                    "priority": revision,
                    "amount": rng.randrange(-30, 80),
                    "owner": rng.choice(["east", "west"]),
                    "void": revision == 4 and account % 2 == 0,
                }
            )
    rng.shuffle(rows)
    return rows, [[5, 6], [7, 9], [9, 12]]


def temporal_answer(rows, queries):
    answers = []
    for valid, known in queries:
        selected = {}
        for r in rows:
            if r["from"] <= valid < r["to"] and r["recorded"] <= known:
                old = selected.get(r["account"])
                if old is None or (r["recorded"], r["priority"]) > (
                    old["recorded"],
                    old["priority"],
                ):
                    selected[r["account"]] = r
        sums = {"east": 0, "west": 0}
        for r in selected.values():
            if not r["void"]:
                sums[r["owner"]] += r["amount"]
        answers.append(
            {
                "sources": {
                    f"A{i}": selected[f"A{i}"]["id"] if f"A{i}" in selected else None
                    for i in range(7)
                },
                "totals": sums,
                "void_accounts": sorted(a for a, r in selected.items() if r["void"]),
            }
        )
    return {"snapshots": answers}


def temporal_case(v):
    rows, queries = temporal_data(v)
    return case(
        "bitemporal",
        v,
        "Reading Comprehension",
        f"""
Reconstruct three accounting snapshots. Each query is [business_time,knowledge_cutoff].
For each account A0..A6 first retain rows with from <= business_time < to AND recorded <=
knowledge_cutoff. Among these select greatest (recorded,priority), lexicographically as an
integer pair. Priority is a tie-breaker, not a global revision number. A selected void row
is still the source but contributes no money; never resurrect an older row after a void.
Missing accounts have null sources. Negative amounts reduce totals. Owners come from the
selected rows, not from any newer or older row.
Rows: {dump(rows)}
Queries: {dump(queries)}
Return snapshots in query order. Each has sources (all seven account -> source ID or null),
totals (east and west, include zero), and void_accounts (sorted account IDs).
""",
        temporal_answer(rows, queries),
        "data",
    )


def rollup_data(v):
    rng = random.Random(6400 + v)
    rows = []
    for i in range(9):
        invoice = f"I{i}"
        rows.extend(
            [
                {
                    "id": f"e{i}a",
                    "invoice": invoice,
                    "op": "issue",
                    "customer": f"K{i % 3}",
                    "amount": rng.randrange(30, 100),
                },
                {"id": f"e{i}b", "invoice": invoice, "op": "pay", "amount": rng.randrange(10, 70)},
                {
                    "id": f"e{i}c",
                    "invoice": invoice,
                    "op": "credit",
                    "amount": rng.randrange(5, 25),
                },
            ]
        )
    # Replay IDs have contradictory payloads: first delivered payload wins.
    rows.insert(8, {**rows[1], "amount": 999})
    rows.extend(
        [
            {"id": "void1", "invoice": "I1", "op": "void"},
            {"id": "late1", "invoice": "I1", "op": "pay", "amount": 40},
            {"id": "refund2", "invoice": "I2", "op": "refund", "amount": 12},
            {"id": "void4", "invoice": "I4", "op": "void"},
            {"id": "late4", "invoice": "I4", "op": "credit", "amount": 20},
        ]
    )
    return rows


def rollup_answer(rows):
    seen, invoices, ignored = set(), {}, []
    for e in rows:
        if e["id"] in seen:
            ignored.append(e["id"])
            continue
        seen.add(e["id"])
        if e["op"] == "issue":
            invoices[e["invoice"]] = {
                "customer": e["customer"],
                "due": e["amount"],
                "cash": 0,
                "void": False,
            }
            continue
        inv = invoices[e["invoice"]]
        if inv["void"]:
            ignored.append(e["id"])
        elif e["op"] == "void":
            inv["void"] = True
            inv["due"] = 0
        elif e["op"] == "credit":
            inv["due"] -= e["amount"]
        else:
            inv["cash"] += e["amount"] * (-1 if e["op"] == "refund" else 1)
    totals = {f"K{i}": {"open": 0, "receivable": 0, "cash": 0} for i in range(3)}
    for inv in invoices.values():
        t = totals[inv["customer"]]
        t["cash"] += inv["cash"]
        if not inv["void"]:
            t["open"] += 1
            t["receivable"] += inv["due"] - inv["cash"]
    return {
        "customers": totals,
        "ignored": ignored,
        "void_invoices": sorted(k for k, i in invoices.items() if i["void"]),
    }


def rollup_case(v):
    rows = rollup_data(v)
    return case(
        "eventsummary",
        v,
        "Summarization",
        f"""
Produce a compact structured reconciliation of the delivered event log. Process in listed
order. Globally deduplicate by event id before doing anything else: first delivery wins even
if a later payload differs. issue creates an invoice with its customer and gross amount.
pay adds cash; refund subtracts cash; credit reduces gross amount. void sets gross to zero
and closes the invoice, WITHOUT returning any cash. All later events for a void invoice are
ignored. Every non-issue references an already issued invoice; no invoice is issued twice.
For each customer, open counts nonvoid invoices, receivable sums gross-minus-cash on NONVOID
invoices (allow negatives), and cash sums net cash on ALL invoices including void ones.
Log: {dump(rows)}
Return customers (K0,K1,K2 each with open,receivable,cash), ignored (event IDs in delivery
order, one entry per ignored delivery, whether duplicate or post-void), and void_invoices
(sorted invoice IDs). Do not summarize by mentioning keywords; reconcile the actual totals.
""",
        rollup_answer(rows),
        "data",
    )


def graph_data(v):
    rng = random.Random(6500 + v)
    rows = []
    layers = [["start"]] + [[f"N{i}{j}" for j in range(4)] for i in range(1, 5)] + [["end"]]
    for left, right in zip(layers, layers[1:]):
        for a in left:
            for b in right:
                ident = f"{a}-{b}"
                rows.append(
                    {
                        "id": ident,
                        "from": a,
                        "to": b,
                        "cost": rng.randint(1, 6),
                        "rev": 1,
                        "approved": True,
                        "disabled": False,
                    }
                )
                if rng.random() < 0.25:
                    rows.append(
                        {
                            **rows[-1],
                            "rev": 2,
                            "disabled": rng.random() < 0.5,
                            "cost": rng.randint(1, 6),
                        }
                    )
                if rng.random() < 0.3:
                    rows.append({**rows[-1], "rev": 3, "approved": False, "cost": 0})
    rng.shuffle(rows)
    return rows


def active_edges(rows):
    latest = {}
    for r in rows:
        if r["approved"] and (r["id"] not in latest or r["rev"] > latest[r["id"]]["rev"]):
            latest[r["id"]] = r
    return [r for r in latest.values() if not r["disabled"]]


def graph_answer(rows):
    edges = defaultdict(list)
    for r in active_edges(rows):
        edges[r["from"]].append(r)
    paths = []

    def walk(node, cost, path):
        if node == "end":
            paths.append((cost, path))
        for r in edges[node]:
            walk(r["to"], cost + r["cost"], path + [r["to"]])

    walk("start", 0, ["start"])
    minimum = min(c for c, _ in paths)
    optimal = sorted(p for c, p in paths if c == minimum)
    mandatory = set.intersection(*(set(p[1:-1]) for p in optimal))
    return {
        "cost": minimum,
        "optimal_count": len(optimal),
        "first_path": optimal[0],
        "mandatory_internal": sorted(mandatory),
    }


def graph_case(v):
    rows = graph_data(v)
    return case(
        "archivepaths",
        v,
        "Needle Retrieval",
        f"""
The archive encodes directed handoff links rather than a directly retrievable answer. For
each link ID, choose its highest approved revision. Unapproved revisions have no effect.
If that chosen row is disabled the link does not exist; do not fall back to an older row.
Find the minimum total cost path from start to end. Count ALL paths achieving that cost,
select the lexicographically first node sequence, and find the INTERNAL nodes present on
EVERY optimal path. Paths use only the selected links. Link costs add; there are no cycles.
Archive: {dump(rows)}
Return cost, optimal_count, first_path (nodes in traversal order), mandatory_internal
(sorted nodes, excluding start and end). Relevant evidence includes rejected revisions.
""",
        graph_answer(rows),
        "data",
    )


def scopes_data(v):
    rng = random.Random(6600 + v)
    events = []
    for _ in range(6):
        key = rng.choice(["color", "quota", "region"])
        events.extend(
            [
                ["push"],
                ["set", key, rng.randrange(10, 99)],
                ["push"],
                ["unset", rng.choice(["color", "quota", "region"])],
                ["read"],
                [rng.choice(["commit", "rollback"])],
                ["read"],
                ["commit"],
                ["read"],
            ]
        )
    return events


def scopes_answer(events):
    stack = [{"color": 11, "quota": 22, "region": 33}]
    reads = []
    for e in events:
        op = e[0]
        if op == "push":
            stack.append({})
        elif op in ("set", "unset"):
            stack[-1][e[1]] = e[2] if op == "set" else None
        elif op in ("commit", "rollback"):
            child = stack.pop()
            if op == "commit":
                stack[-1].update(child)
        else:
            values = {}
            for frame in stack:
                values.update(frame)
            reads.append([values[k] for k in ["color", "quota", "region"]])
    return {"reads": reads, "root": stack[0]}


def scopes_case(v):
    events = scopes_data(v)
    return case(
        "scopedstate",
        v,
        "Long Context Coherence",
        f"""
Follow a nested configuration session. The root initially has color=11, quota=22, region=33.
push creates an EMPTY child overlay; reads inherit through parents. set writes only the
current overlay. unset writes an explicit null mask: reads return null, not a parent's
value. commit pops the child and copies only its explicit entries (including null masks)
into its parent. rollback pops and discards the child. Neither operation pops the root.
read records visible [color,quota,region]. All operations below are valid and ordered.
The final stack has only the root. Integer values are opaque configuration data.
Operations: {dump(events)}
Return reads (one three-element array for every read in chronological order) and root
(final explicit root entries for color,quota,region). Do not conflate null with inheritance.
""",
        scopes_answer(events),
    )


GATES = [
    ("g0", "xor", "a", "b"),
    ("g1", "and", "b", "c"),
    ("g2", "or", "g0", "d"),
    ("g3", "xor", "g1", "g2"),
    ("g4", "and", "a", "g3"),
    ("g5", "xor", "g4", "c"),
]


def circuit(bits, fault):
    values = dict(zip("abcd", map(int, bits)))
    for name, op, a, b in GATES:
        x, y = values[a], values[b]
        result = {"xor": x ^ y, "and": x & y, "or": x | y}[op]
        values[name] = int(fault[-1]) if fault.startswith(name + "=") else result
    return values["g5"]


def diagnosis_data(v):
    rng = random.Random(6700 + v)
    probes = sorted(rng.sample(["".join(b) for b in itertools.product("01", repeat=4)], 10))
    faults = ["healthy"] + [f"g{i}={b}" for i in range(6) for b in range(2)]
    hidden = ["g3=0", "g1=1", "g4=0"][v % 3]
    observations = {p: circuit(p, hidden) for p in probes[:2]}
    costs = {p: rng.randrange(1, 6) for p in probes[2:]}
    return faults, observations, costs


def diagnosis_answer(faults, observations, costs):
    candidates = sorted(
        f for f in faults if all(circuit(p, f) == b for p, b in observations.items())
    )
    groups = defaultdict(list)
    probes = sorted(costs)
    for f in candidates:
        groups[tuple(circuit(p, f) for p in probes)].append(f)
    classes = sorted(sorted(g) for g in groups.values())
    representatives = [g[0] for g in classes]
    feasible = []
    for n in range(len(probes) + 1):
        for subset in itertools.combinations(probes, n):
            if len({tuple(circuit(p, f) for p in subset) for f in representatives}) == len(classes):
                feasible.append((sum(costs[p] for p in subset), n, list(subset)))
    best = min(feasible)
    return {
        "candidates": candidates,
        "indistinguishable": classes,
        "cost": best[0],
        "probe_sets": sorted(s for c, n, s in feasible if (c, n) == best[:2]),
    }


def diagnosis_case(v):
    faults, observations, costs = diagnosis_data(v)
    return case(
        "diagnosis",
        v,
        "R&D",
        f"""
Diagnose this combinational controller. Inputs a,b,c,d and all gates are bits. Gate order
is topological. xor=exclusive OR, and=AND, or=inclusive OR. Only output g5 is measurable.
There is either no fault (healthy) or exactly one gate output is permanently forced to 0
or 1, replacing that gate's result before downstream gates execute. Faults stay fixed.
Netlist [gate,operation,left,right]: {dump(GATES)}
Fault labels: {dump(faults)}
Observed input abcd -> output: {dump(observations)}
Available additional input abcd -> measurement cost: {dump(costs)}
Return candidates (all consistent fault labels sorted), indistinguishable (partition
candidates by identical outcomes on ALL available additional probes; sort each class and
then sort classes lexicographically), cost, and probe_sets. Select a NONADAPTIVE subset
of additional probes that distinguishes those classes, not faults inside an inseparable
class. Minimize cost, then number of probes. probe_sets contains ALL subsets tied on both
objectives, each sorted lexicographically and the outer list also sorted. Include singleton
classes. With only one class, the minimum probe set is empty and costs zero.
""",
        diagnosis_answer(faults, observations, costs),
    )


def genetics_data(v):
    # A stipulated finite inheritance model: no assumptions about real populations.
    return [12, 18, 24][v % 3], [20, 30, 10][v % 3], [2, 3, 4][v % 3]


def genetics_answer(r1, r2, penalty):
    p1 = {
        "AB": Fraction(100 - r1, 200),
        "ab": Fraction(100 - r1, 200),
        "Ab": Fraction(r1, 200),
        "aB": Fraction(r1, 200),
    }
    p2 = {
        "Ab": Fraction(100 - r2, 200),
        "aB": Fraction(100 - r2, 200),
        "AB": Fraction(r2, 200),
        "ab": Fraction(r2, 200),
    }
    masses = defaultdict(Fraction)
    total = Fraction()
    hidden = Fraction()
    for a, b in itertools.product(p1, p2):
        dominant_a = "A" in (a[0], b[0])
        dominant_b = "B" in (a[1], b[1])
        phenotype = "white" if not dominant_a else "dark" if dominant_b else "light"
        viability = Fraction(1, penalty) if a[0] == b[0] == "A" else Fraction(1)
        mass = p1[a] * p2[b] * viability
        total += mass
        masses[phenotype] += mass
        if phenotype == "dark" and a[0] != b[0] and a[1] != b[1]:
            hidden += mass

    def ratio(f):
        return [f.numerator, f.denominator]

    return {
        "survival": ratio(total),
        "phenotypes": {k: ratio(masses[k] / total) for k in ["dark", "light", "white"]},
        "double_heterozygote_given_dark": ratio(hidden / masses["dark"]),
    }


def genetics_case(v):
    r1, r2, penalty = genetics_data(v)
    return case(
        "inheritance",
        v,
        "Factual Knowledge",
        f"""
Apply the stipulated linked-locus inheritance model, with no outside population assumptions.
Parent 1 haplotypes are AB/ab and recombination probability is {r1}/100. Parent 2 haplotypes
are Ab/aB and recombination probability is {r2}/100. Each parent's two parental gametes
each have probability (1-r)/2; the other two gametes each have probability r/2. Gametes
from different parents are independent. At locus A, AA offspring survive with probability
1/{penalty}; Aa and aa survive with probability 1. No other viability selection occurs.
Among survivors phenotype is white if aa (regardless of B), dark if at least one A AND at
least one B, light otherwise. A double heterozygote has Aa AND Bb, irrespective of phase.
Return survival (probability an offspring survives), phenotypes (dark/light/white conditional
on survival), and double_heterozygote_given_dark (conditional on surviving AND being dark).
Every probability must be an exact reduced [numerator,denominator] pair with positive
denominator. Distinguish preselection probabilities from conditional survivor frequencies.
""",
        genetics_answer(r1, r2, penalty),
    )


def reshape_data(v):
    rng = random.Random(6800 + v)
    return [[rng.randrange(-9, 10) for _ in range(6)] for _ in range(5)]


def reshape_answer(matrix):
    rotated = list(zip(*matrix[::-1]))
    flat = [x for i, row in enumerate(rotated) for x in (row if i % 2 == 0 else row[::-1])]
    kept = [x for i, x in enumerate(flat) if i % 4 != 1 and x % 3 != 0]
    acc = 0
    tokens = []
    for i, x in enumerate(kept):
        acc += x if i % 2 == 0 else -x
        tokens.append(acc % 7)
    runs = []
    for token, group in itertools.groupby(tokens):
        runs.append([token, len(list(group))])
    return {
        "runs": runs,
        "kept": len(kept),
        "checksum": sum((i + 1) * x for i, x in enumerate(tokens)),
    }


def reshape_case(v):
    matrix = reshape_data(v)
    return case(
        "reshape",
        v,
        "Instruction Following",
        f"""
Apply EVERY stage in order to this five-row, six-column integer matrix: {dump(matrix)}
1. Rotate the entire matrix 90 degrees clockwise, producing six rows and five columns.
2. Traverse rotated rows from top to bottom; rows with even zero-based index left-to-right,
   odd rows right-to-left. This produces one flat sequence of 30 values.
3. Using indices in that ORIGINAL flat sequence, drop positions i where i modulo 4 is 1.
   Also drop values divisible by 3. Do not renumber until both filters are complete.
4. Renumber survivors j from zero. Starting accumulator at zero, add survivor when j is
   even, subtract it when j is odd. After EACH operation emit accumulator modulo 7 as a
   number 0..6 (Euclidean remainder even for a negative accumulator).
5. Run-length encode consecutive equal emitted numbers as [number,count]. Do not merge
   equal numbers separated by other numbers. checksum is sum((j+1)*emitted[j]).
Return runs, kept (number of survivors), and checksum. Output numbers, not strings.
""",
        reshape_answer(matrix),
        "instructions",
    )


def translation_case(v):
    # Semantic translation with a finite answer space, not a fluency rubric.
    clauses = [
        (
            "Translate the German policy into the requested English-keyed semantic representation.",
            "Nur wenn alle freigegebenen Berichte unterschrieben sind und kein freigegebener "
            "Bericht widerrufen wurde, darf die Lieferung erfolgen. Entwuerfe zaehlen "
            "nicht. Bei einer leeren Menge freigegebener Berichte gilt die Bedingung als erfuellt.",
            "all",
            False,
        ),
        (
            "Translate the Czech policy into the requested English-keyed semantic representation.",
            "Dodavka smi probehnout prave tehdy, kdyz je alespon jedna schvalena zprava podepsana "
            "a zadna schvalena zprava nebyla odvolana. Navrhy se nepocitaji. Prazdna mnozina "
            "schvalenych zprav podminku nesplnuje.",
            "any",
            True,
        ),
        (
            "Translate the German policy into the requested English-keyed semantic representation.",
            "Die Lieferung darf genau dann erfolgen, wenn nicht jeder freigegebene Bericht "
            "unterschrieben ist und kein freigegebener Bericht widerrufen wurde. Entwuerfe "
            "zaehlen nicht. Bei einer leeren Menge ist 'jeder' wahr.",
            "not_all",
            True,
        ),
    ]
    intro, policy, quantifier, sufficient = clauses[v]
    scenarios = [
        [],
        [[True, True, False]],
        [[True, False, False]],
        [[True, True, False], [True, False, False]],
        [[False, False, True], [True, True, False]],
        [[True, True, True], [True, False, False]],
        [[False, True, False]],
        [[True, False, False], [True, False, False]],
    ]
    statuses = []
    for rows in scenarios:
        selected = [r for r in rows if r[0]]
        signed = [r[1] for r in selected]
        condition = {"all": all(signed), "any": any(signed), "not_all": not all(signed)}[quantifier]
        condition = condition and not any(r[2] for r in selected)
        statuses.append(
            "allowed" if condition and sufficient else "unknown" if condition else "forbidden"
        )
    answer = {
        "signature_quantifier": quantifier,
        "condition_sufficient": sufficient,
        "drafts_count": False,
        "statuses": statuses,
    }
    return case(
        "translation",
        v,
        "Translation",
        f"""
{intro}
Source policy (ASCII transliteration preserves the original language): {policy}
English domain definitions: freigegeben/schvalena means approved; unterschrieben/podepsana
means signed; widerrufen/odvolana means revoked; Entwurf/navrh means draft.
Treat a necessary condition (only if) as insufficient to establish permission; a biconditional
(if and only if) establishes both permission when true and prohibition when false. A revoked
approved report always prohibits delivery under these policies. No other permissions exist
in the evidence, but do NOT assume missing permissions are false: label them unknown.
Scenarios in order, each report is [approved,signed,revoked]: {dump(scenarios)}
Return signature_quantifier (all, any, or not_all), condition_sufficient (Boolean),
drafts_count (Boolean), statuses (allowed/forbidden/unknown per scenario). These keys encode
the translated meaning; no free-form translation or outside assumptions are requested.
""",
        answer,
        "reasoning",
    )


BUILDERS = [
    planning_case,
    tool_case,
    temporal_case,
    rollup_case,
    graph_case,
    scopes_case,
    diagnosis_case,
    genetics_case,
    reshape_case,
    translation_case,
]


def build_cases():
    return [builder(v) for builder in BUILDERS for v in range(3)]


if __name__ == "__main__":
    cases = build_cases()
    path = Path(__file__).parent / "tests" / "stretch.yaml"
    path.write_text(
        yaml.safe_dump(cases, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
    print(f"Wrote {len(cases)} stretch-v6 questions to {path}")
