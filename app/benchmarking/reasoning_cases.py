"""Original v14 tasks: shortcut-resistant retrieval, inference and execution.

Method inspiration and scope are documented in BENCHMARK_DESIGN.md. No third-
party questions are copied. Reference algorithms are checked independently in
selftest_reasoning.py, using only the problem data emitted in each prompt.
"""

from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import random


def task(family, category, variant, split, prompt, data, answer, *, seed=None, atomic_paths=()):
    from app.benchmarking.evaluators import _json_leaf_paths, json_at_path
    from app.benchmarking.rigorous_cases import balanced_rubric, structured

    q = structured(family, category, variant, split,
                   prompt + "\nINPUT=" + json.dumps(data, sort_keys=True), answer)
    q.source = "Original rigorous-v14; see BENCHMARK_DESIGN.md"
    q.expected["integer_paths"] = [p for p in _json_leaf_paths(answer)
                                   if type(json_at_path(answer, p)[0]) is int]
    q.expected["atomic_paths"] = list(atomic_paths)
    q.rubric = balanced_rubric(answer, atomic_paths=atomic_paths)
    for criterion in q.rubric:
        if "evidence_ids" in criterion["id"]:
            criterion["dimension"] = "evidence"
    q.prompt += "\nUse JSON integer literals for all integer outputs, not floats or strings."
    if seed is not None:
        q.id = f"Q14-{family}-s{seed}-v{variant + 1:02d}"
        q.metadata.update(family=f"Q14-{family}", seed=seed, variant=variant)
    return q


def probability_case(rng, variant, split):
    weights = [[rng.randrange(1, 6) for _ in range(5)] for _ in range(2)]
    prior = [rng.randrange(1, 5), rng.randrange(1, 5)]
    threshold = rng.randrange(7, 11)
    data = {"weights": weights, "prior_weights": prior, "threshold": threshold,
            "audit_modulus": 2 + variant % 2, "future_threshold": rng.randrange(6, 9)}
    report = [Fraction(0), Fraction(0)]
    event = expected_sum = Fraction(0)
    for regime, ws in enumerate(weights):
        for draws in itertools.product(range(1, 6), repeat=3):
            if (sum(draws) < threshold or draws[0] == draws[2]
                    or (draws[0] + 2 * draws[1]) % data["audit_modulus"]):
                continue
            mass = Fraction(prior[regime], sum(prior))
            for value in draws:
                mass *= Fraction(ws[value - 1], sum(ws))
            report[regime] += mass
            event += mass * (len(set(draws)) == 3)
            expected_sum += mass * sum(draws)
    total = sum(report)
    posterior = [p / total for p in report]
    future = Fraction(0)
    for probability, ws in zip(posterior, weights):
        future += probability * sum(
            Fraction(ws[a - 1] * ws[b - 1], sum(ws) ** 2)
            for a in range(1, 6) for b in range(1, 6)
            if a + b >= data["future_threshold"])
    means = [sum(Fraction((i + 1) * w, sum(ws)) for i, w in enumerate(ws)) for ws in weights]
    covariance = sum(p * m * m for p, m in zip(posterior, means)) - sum(
        p * m for p, m in zip(posterior, means)) ** 2

    def pair(value):
        return [value.numerator, value.denominator]

    answer = {"report_probability": pair(total), "posterior_regime_1": pair(posterior[1]),
              "all_distinct_given_report": pair(event / total),
              "expected_sum_given_report": pair(expected_sum / total),
              "future_sum_probability": pair(future), "future_covariance": pair(covariance)}
    return task("selective-report", "Mathematical Reasoning", variant, split,
        "A hidden regime Z in {0,1} is drawn ONCE with prior_weights in that order. "
        "Given Z, all past and future draws are independent with replacement from values "
        "1..5 using weights[Z]. Divide weights by their sum to obtain probabilities. "
        "A reporter reveals a past triple (X1,X2,X3) occurred, but not its values, exactly "
        "when X1+X2+X3 >= threshold, X1 != X3 AND (X1+2*X2) modulo audit_modulus is zero. "
        "Condition only on this revelation. Find its unconditional probability, P(Z=1|report), "
        "P(all three past values distinct|report), E[X1+X2+X3|report], and for two NEW draws "
        "Y1,Y2 under the SAME Z find P(Y1+Y2 >= future_threshold|report) and "
        "Cov(Y1,Y2|report)=E[Y1*Y2|report]-E[Y1|report]*E[Y2|report]. "
        "Do not redraw the regime for each draw. Output every number as a reduced "
        "[numerator,denominator] pair with positive denominator.", data, answer,
        atomic_paths=list(answer))


def resolve_records(data):
    selected = {}
    for row in data["records"]:
        if row["key"] not in selected or (row["revision"], row["id"]) > (
                selected[row["key"]]["revision"], selected[row["key"]]["id"]):
            selected[row["key"]] = row
    results = []
    for start in data["starts"]:
        path, evidence = [], []
        cursor = start
        while True:
            if cursor in path:
                status, value = "cycle", None
                break
            path.append(cursor)
            row = selected.get(cursor)
            if row is None:
                status, value = "missing", None
                break
            evidence.append(row["id"])
            if not row["active"]:
                status, value = "deleted", None
                break
            if row["kind"] == "result":
                status, value = "resolved", row["value"]
                break
            cursor = row["value"]
        results.append({"path": path, "evidence_ids": evidence, "status": status, "result": value})
    return {"queries": results}


def retrieval_case(rng, variant, split):
    names = [f"K{n}" for n in rng.sample(range(100000, 999999), 184)]
    # Every key participates in the same grammar, and distractors have real links,
    # results, revisions and tombstones. IDs reveal neither age nor relevance.
    rows = []
    ids = iter(rng.sample([f"R{n:04d}" for n in range(1000, 9999)], 800))

    def add(key, revision, active, kind, value):
        rows.append(dict(id=next(ids), key=key, revision=revision,
                         active=active, kind=kind, value=value))

    for key in names[:180]:
        for revision in rng.sample(range(1, 10), 2):
            kind = rng.choice(("link", "result"))
            add(key, revision, rng.random() > 0.15, kind,
                rng.choice(names[:180]) if kind == "link" else rng.choice(("red", "amber", "green")))
    def current(key):
        return max((r for r in rows if r["key"] == key), key=lambda r: (r["revision"], r["id"]))

    # Distractors also contain equal-revision conflicts. Revision values, like
    # identifiers and record kinds, cannot identify the answer-bearing rows.
    for key in rng.sample(names[166:180], 8):
        add(key, current(key)["revision"], True, "link", rng.choice(names[:180]))
    chain = rng.sample(names[:160], 7)
    for a, b in zip(chain, chain[1:]):
        current(a).update(active=True, kind="link", value=b)
    current(chain[-1]).update(active=True, kind="result", value=rng.choice(("red", "amber", "green")))
    # Equal-revision tie participates in the actual selected chain.
    add(chain[2], current(chain[2])["revision"], True, "link", chain[-1])
    cycle = names[160:163]
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        current(a).update(active=True, kind="link", value=b)
    current(names[163]).update(active=True, kind="link", value=names[180])  # absent key
    current(names[164]).update(active=True, kind="link", value=names[165])
    for row in rows:
        if row["key"] in (names[165], chain[-1]):
            row.update(active=True, kind="result", value="green")
    current(names[165]).update(active=False)
    # Alternate variants include a resolvable chain or a tombstone stop.
    current(chain[-1]).update(active=variant % 2 == 0, value=rng.choice(("red", "amber", "green")))
    starts = [chain[0], cycle[0], names[163], names[164], names[181]]
    rng.shuffle(starts)
    rng.shuffle(rows)
    data = {"records": rows, "starts": starts}
    return task("revision-multihop", "Needle Retrieval", variant, split,
        "Resolve every starting key in starts order. First select greatest revision per EXACT "
        "key; ties select the lexicographically greatest record id, regardless of packet order. "
        "Then honor active=false as a tombstone; never resurrect older rows. All records use "
        "the same grammar and may be relevant. An active link's value is the next exact key; "
        "an active result terminates successfully. For each query output path, evidence_ids, "
        "status and result. Append each visited key to path, including a missing or deleted "
        "terminal key. Append the selected record id to evidence_ids whenever a record exists, "
        "including a tombstone. Stop BEFORE visiting a repeated key, so path contains no repeats. "
        "status is resolved/deleted/missing/cycle. result is the result record's string only "
        "when resolved, otherwise null. Evidence follows traversal order. Never infer a missing "
        "link from a similar identifier or another query.", data, resolve_records(data),
        atomic_paths=[f"queries[{i}].{key}" for i in range(len(starts))
                      for key in ("path", "evidence_ids")])


def grid_holds(clue, positions):
    a, b = positions[clue["a"]], positions[clue["b"]]
    return {"before": a < b, "next": b == a + 1, "same": a == b,
            "adjacent": abs(a - b) == 1, "apart": abs(a - b) > 1}[clue["op"]]


def grid_worlds(groups):
    permutations = list(itertools.permutations(range(1, 5)))
    labels = sum(groups, [])
    return [dict(zip(labels, sum((list(p) for p in ps), [])))
            for ps in itertools.product(permutations, repeat=len(groups))]


def grid_answer(data):
    worlds = grid_worlds(data["groups"])
    labels = sorted(worlds[0])
    output = []
    for omitted in data["omit"]:
        feasible = [w for w in worlds if all(grid_holds(c, w) for c in data["clues"]
                                           if c["id"] not in omitted)]
        witnesses = sorted([w[k] for k in labels] for w in feasible)
        output.append({"count": len(feasible),
                       "domains": {k: sorted({w[k] for w in feasible}) for k in labels},
                       "first_two": witnesses[:2]})
    return {"scenarios": output}


def grid_case(rng, seed, variant):
    labels = rng.sample([f"e{i:02d}" for i in range(40)], 12)
    groups = [sorted(labels[i:i + 4]) for i in range(0, 12, 4)]
    worlds = grid_worlds(groups)
    solution = rng.choice(worlds)
    candidates = [dict(a=a, b=b, op=op) for a, b in itertools.combinations(labels, 2)
                  for op in ("before", "next", "same", "adjacent", "apart")
                  if grid_holds(dict(a=a, b=b, op=op), solution)]
    rng.shuffle(candidates)
    remaining, clues = worlds, []
    for clue in candidates:
        narrowed = [w for w in remaining if grid_holds(clue, w)]
        if len(narrowed) < len(remaining):
            clues.append(clue)
            remaining = narrowed
        if len(remaining) == 1:
            break
    assert len(remaining) == 1
    # Keep an irredundant puzzle: every clue deletion must restore ambiguity.
    for clue in clues[:]:
        trial = [c for c in clues if c is not clue]
        if sum(all(grid_holds(c, w) for c in trial) for w in worlds) == 1:
            clues.remove(clue)
    for i, clue in enumerate(clues):
        clue["id"] = f"c{i:02d}"
    omitted = rng.sample(clues, 2)
    data = {"groups": groups, "clues": clues,
            "omit": [[], [omitted[0]["id"]], [c["id"] for c in omitted]]}
    return task("grid-ambiguity", "Logical Reasoning", variant, "evaluation",
        "Twelve labeled entities form three groups. Each group independently occupies positions "
        "1..4 with each position used exactly once per group; entities from different groups may "
        "share a position. A clue compares positions of a and b: before means a<b; next means "
        "b=a+1; same means a=b; adjacent means |a-b|=1; apart means |a-b|>1. All clues are "
        "conjoined. For EACH omit list in order, remove exactly those clue IDs and solve afresh. "
        "Report count of ALL satisfying complete assignments, domains mapping EVERY entity to "
        "all its possible positions in ascending order, and first_two containing the first two "
        "lexicographically smallest distinct assignments (or all if fewer than two). Each witness "
        "is a position array in lexicographically sorted ENTITY label order. Domains are marginals; "
        "their Cartesian product need not be feasible. Do not treat one witness as proof of uniqueness.",
        data, grid_answer(data), seed=seed,
        atomic_paths=[f"scenarios[{i}].domains.{key}" for i in range(3) for key in labels]
        + [f"scenarios[{i}].first_two" for i in range(3)])


PROGRAM = '''def run(xs):
    a = [xs[0], xs[1]]
    rows = [a, a, a.copy()]
    for i, value in enumerate(xs[2:]):
        rows[(value + i) % 3].append(value)
        a[0] += rows[(i + 1) % 3][-1]
        if a[0] % 2:
            rows[2].reverse()
    return [sum(a) % 7, sum(rows[2]) % 5, len(a)]
'''


def execute_trace(xs, copied=False):
    # Explicit object IDs and a heap instead of relying on Python list aliases.
    heap = {0: list(xs[:2]), 1: list(xs[:2]), 2: list(xs[:2])}
    refs = [0, 1 if copied else 0, 2]
    for i, value in enumerate(xs[2:]):
        heap[refs[(value + i) % 3]].append(value)
        heap[0][0] += heap[refs[(i + 1) % 3]][-1]
        if heap[0][0] % 2:
            heap[2] = list(reversed(heap[2]))
    return [sum(heap[0]) % 7, sum(heap[2]) % 5, len(heap[0])]


def code_answer(data):
    domain = list(itertools.product(data["alphabet"], repeat=4))
    outputs = {x: execute_trace(x) for x in domain}
    differing = [x for x in domain if outputs[x] != execute_trace(x, True)]
    inverse = []
    for target in data["targets"]:
        inputs = [list(x) for x in domain if outputs[x] == target]
        inverse.append({"count": len(inputs), "first_two": inputs[:2], "last_two": inputs[-2:]})
    first = differing[0]
    return {"forward": [outputs[tuple(x)] for x in data["probes"]], "inverse": inverse,
            "distinguishing_count": len(differing), "first_distinguishing_input": list(first),
            "first_distinguishing_outputs": [outputs[first], execute_trace(first, True)]}


def code_case(rng, seed, variant):
    alphabet = list(range(-2 - variant, 3 - variant))
    domain = list(itertools.product(alphabet, repeat=4))
    counts = Counter(tuple(execute_trace(x)) for x in domain)
    ambiguous = sorted(out for out, n in counts.items() if n > 2)
    targets = [list(out) for out in rng.sample(ambiguous, 2)] + [[7, 0, 2]]
    rng.shuffle(targets)
    data = {"alphabet": alphabet, "probes": [list(x) for x in rng.sample(domain, 8)],
            "targets": targets}
    return task("alias-preimages", "Advanced Coding", variant, "evaluation",
        "Analyze this Python 3 function, using unbounded integers and Python modulo semantics "
        "(nonnegative remainder for positive divisor). No execution tools are supplied.\n"
        + PROGRAM + "\nThe input domain is EVERY length-4 array with entries from alphabet, "
        "with repetition allowed. forward gives run(input) for each probe in order. For each "
        "target in order, inverse gives the exact number of domain inputs producing it and the "
        "lexicographically smallest first_two and largest last_two distinct input arrays, each "
        "list ascending (use all inputs when fewer than two, [] when none). Also consider a "
        "mutant changing ONLY rows = [a, a, a.copy()] to rows = [a, a.copy(), a.copy()]. "
        "Give distinguishing_count over the complete domain, first_distinguishing_input "
        "(lexicographically smallest), and first_distinguishing_outputs in [original,mutant] "
        "order. Each call starts with fresh local lists. Track shared objects, not just values.",
        data, code_answer(data), seed=seed,
        atomic_paths=[f"forward[{i}]" for i in range(len(data["probes"]))]
        + [f"inverse[{i}].{key}" for i in range(len(targets)) for key in ("first_two", "last_two")]
        + ["first_distinguishing_input", "first_distinguishing_outputs"])


def load_reasoning_questions(seed, variant):
    questions = []
    for family, builder in (("grid-ambiguity", grid_case), ("alias-preimages", code_case)):
        digest = hashlib.sha256(f"rigorous-v14:{family}:{seed}:{variant}".encode()).digest()
        questions.append(builder(random.Random(int.from_bytes(digest, "big")), seed, variant))
    return questions
