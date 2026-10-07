"""Original capability ladder, with exact objectives and complete-answer grading.

These public, deterministic tasks target the saturated families in the October
2026 panel. Source correctness is distinct from measured model discrimination.
Independent prompt/fixture oracles live in selftest_discrimination.py.
"""

from functools import lru_cache
import hashlib
import itertools
import random

from app.benchmarking.models import Question
from app.benchmarking.reasoning_cases import task

REVISION = "capability-ladder-v1"


def schedule_answer(data):
    jobs, horizon = data["jobs"], data["horizon"]
    best, count, canonical = None, 0, None

    def visit(slots, loads):
        nonlocal best, count, canonical
        i = len(slots)
        if i == len(jobs):
            cost = sum(j["weight"] * max(0, s + 1 - j["due"]) ** 2
                       for j, s in zip(jobs, slots))
            if best is None or cost < best:
                best, count, canonical = cost, 1, list(slots)
            elif cost == best:
                count += 1
                canonical = min(canonical, list(slots))
            return
        job = jobs[i]
        for slot in range(job["release"], horizon):
            if loads[slot] + job["units"] > data["capacity"]:
                continue
            if any(slots[p] >= slot for p in job["after"]):
                continue
            loads[slot] += job["units"]
            visit([*slots, slot], loads)
            loads[slot] -= job["units"]

    visit([], [0] * horizon)
    return {"minimum_cost": best, "optimal_count": count, "slots": canonical}


def diagnosis_answer(data):
    full = (1 << len(data["states"])) - 1
    masks = [sum(1 << s for s, bit in enumerate(t["positive"]) if bit) for t in data["tests"]]

    @lru_cache(None)
    def cost(states):
        if states.bit_count() <= 1:
            return 0
        options = []
        for t, mask in zip(data["tests"], masks):
            yes, no = states & mask, states & ~mask
            if not yes or not no:
                continue
            a, b = cost(yes), cost(no)
            if a is not None and b is not None:
                options.append(t["cost"] + max(a, b))
        return min(options) if options else None

    optimum = cost(full)
    roots = []
    for t, mask in zip(data["tests"], masks):
        yes, no = full & mask, full & ~mask
        if yes and no and cost(yes) is not None and cost(no) is not None:
            if t["cost"] + max(cost(yes), cost(no)) == optimum:
                roots.append(t["id"])
    return {"worst_cost": optimum, "optimal_first_tests": sorted(roots),
            "within_budget": optimum is not None and optimum <= data["budget"]}


def transducer_answer(data):
    # DP counts languages while retaining only lexical extrema, rather than
    # enumerating the strings used by the independent oracle.
    states = {(data["initial"], 0): (1, "", "")}
    for position in range(data["length"]):
        nxt = {}
        for (state, checksum), (count, first, last) in states.items():
            for symbol in data["alphabet"]:
                target, delta = data["transitions"][state][symbol]
                key = (target, (checksum + (position + 1) * delta) % data["modulus"])
                old = nxt.get(key)
                new = (count, first + symbol, last + symbol)
                nxt[key] = new if old is None else (old[0] + count, min(old[1], new[1]), max(old[2], new[2]))
        states = nxt
    accepted = [v for (s, c), v in states.items() if s in data["accepting"] and c == data["checksum"]]
    return {"count": sum(v[0] for v in accepted),
            "first": min((v[1] for v in accepted), default=None),
            "last": max((v[2] for v in accepted), default=None)}


def portfolio_answer(data):
    valid = []
    for bits in itertools.product((0, 1), repeat=len(data["items"])):
        chosen = [i for i, b in enumerate(bits) if b]
        spend = sum(data["items"][i]["cost"] for i in chosen)
        if spend > data["budget"] or any(bits[a] and bits[b] for a, b in data["conflicts"]):
            continue
        if any(bits[b] and not bits[a] for a, b in data["requires"]):
            continue
        profits = [sum(data["items"][i]["profit"][s] for i in chosen)
                   for s in range(data["scenarios"])]
        valid.append((chosen, spend, profits))
    ideals = [max(p[s] for _, _, p in valid) for s in range(data["scenarios"])]
    def key(row):
        chosen, spend, profits = row
        regrets = [a - b for a, b in zip(ideals, profits)]
        return max(regrets), sum(regrets), spend, chosen
    chosen, spend, profits = min(valid, key=key)
    return {"ideal_profits": ideals, "chosen": chosen, "spend": spend, "profits": profits,
            "regrets": [a - b for a, b in zip(ideals, profits)]}


def route_answer(data):
    # Layered relaxation in product state: node, used coupon, visited mask.
    current = {(data["start"], False, 1 << data["start"]): (0, [data["start"]], -1)}
    candidates = []
    for hops in range(data["max_hops"] + 1):
        for (node, _, mask), (cost, path, coupon) in current.items():
            if node == data["end"] and mask & data["required_mask"] == data["required_mask"]:
                candidates.append((cost, hops, path, coupon))
        nxt = {}
        for (node, used, mask), (cost, path, coupon) in current.items():
            for edge in data["edges"]:
                a, b, toll = edge
                if a != node:
                    continue
                for use in (False, True) if not used else (False,):
                    value = (cost + (toll // 2 if use else toll), [*path, b], hops if use else coupon)
                    state = (b, used or use, mask | (1 << b))
                    if state not in nxt or value < nxt[state]:
                        nxt[state] = value
        current = nxt
    if not candidates:
        return None
    cost, hops, path, coupon = min(candidates)
    return {"cost": cost, "hops": hops, "path": path, "coupon": coupon}


def assignment_answer(data):
    permutations = list(itertools.permutations(range(len(data["matrices"][0]))))
    totals = [[sum(m[i][p[i]] for i in range(len(p))) for m in data["matrices"]] for p in permutations]
    ideals = [min(row[s] for row in totals) for s in range(len(data["matrices"]))]
    candidates = []
    for p, row in zip(permutations, totals):
        regrets = [a - b for a, b in zip(row, ideals)]
        candidates.append((max(regrets), sum(regrets), p, row, regrets))
    _, _, p, row, regrets = min(candidates)
    return {"assignment": list(p), "costs": row, "regrets": regrets, "ideals": ideals}


def _rng(seed, variant, family):
    return random.Random(int.from_bytes(hashlib.sha256(f"{REVISION}:{seed}:{variant}:{family}".encode()).digest(), "big"))


def _data(family, rng, size):
    if family == "resource-schedule":
        return {"horizon": 6, "capacity": 3, "jobs": [
            {"release": rng.randrange(2), "due": rng.randrange(2, 6), "weight": rng.randrange(1, 7),
             "units": rng.randrange(1, 3), "after": [p for p in range(i) if rng.random() < .16]}
            for i in range(size)]}
    if family == "adaptive-diagnosis":
        return {"states": list(range(size)), "budget": rng.randrange(4, 15), "tests": [
            {"id": f"t{i}", "cost": rng.randrange(1, 6), "positive": [rng.randrange(2) for _ in range(size)]}
            for i in range(7)]}
    if family == "checksum-language":
        return {"alphabet": ["a", "b", "c"], "length": size, "initial": "s0", "modulus": 7,
                "checksum": rng.randrange(7), "accepting": ["s1", "s3"], "transitions": {
                    f"s{i}": {s: [f"s{rng.randrange(4)}", rng.randrange(-4, 5)] for s in "abc"}
                    for i in range(4)}}
    if family == "robust-portfolio":
        return {"items": [{"cost": rng.randrange(1, 8), "profit": [rng.randrange(-5, 16) for _ in range(3)]}
                          for _ in range(size)], "scenarios": 3, "budget": size * 2,
                "conflicts": [[a, b] for a in range(size) for b in range(a + 1, size) if rng.random() < .09],
                "requires": [[a, b] for a in range(size) for b in range(a + 1, size) if rng.random() < .07]}
    if family == "coupon-route":
        return {"start": 0, "end": size - 1, "max_hops": 6, "required_mask": rng.randrange(1 << size),
                "edges": [[a, b, rng.randrange(0, 12)] for a in range(size) for b in range(size)
                          if a != b and rng.random() < .4]}
    return {"matrices": [[[rng.randrange(0, 20) for _ in range(size)] for _ in range(size)] for _ in range(3)]}


def _diagnosis_data(rng, size, variant):
    # Source admission rule, independent of any model results: every state
    # must be distinguishable, and the minimax objective must be nontrivial.
    for _ in range(64):
        data = _data("adaptive-diagnosis", rng, size)
        optimum = diagnosis_answer(data)["worst_cost"]
        if optimum is not None and optimum >= 6 + 2 * variant:
            return data
    raise ValueError("Could not construct an informative diagnosis instance")


FAMILIES = {
    "resource-schedule": ("Agentic Use Cases", schedule_answer,
        "Schedule every indexed job once in an integer slot 0..horizon-1. A job lasts one slot, "
        "uses units of capacity, and starts no earlier than release. Total units in each slot <= capacity. "
        "Every job listed in after must finish BEFORE the job starts (strictly smaller slot). "
        "Minimize sum(weight * max(0,slot+1-due)^2). Return minimum_cost, optimal_count of ALL distinct "
        "slot vectors attaining it, and the lexicographically smallest slots vector in job-index order. "
        "If infeasible return null, 0, null respectively."),
    "adaptive-diagnosis": ("Logical Reasoning", diagnosis_answer,
        "Exactly one state is true. A test deterministically reports its positive bit for that state. "
        "Choose tests adaptively after seeing previous results until the state is uniquely identified. "
        "Each performed test costs cost. Minimize the WORST total cost over states, not expected cost "
        "or the number of tests. Return worst_cost, ALL optimal_first_tests sorted by ID, and within_budget "
        "(whether worst_cost <= budget). If identification is impossible return null, [], false. "
        "Repeating a test gives the same bit; tests cannot change the state."),
    "checksum-language": ("Mathematical Reasoning", transducer_answer,
        "Consider ALL strings of length length over alphabet. Start at initial with checksum 0. "
        "At 1-based position i read symbol, look up transitions[current_state][symbol]=[next_state,delta], "
        "move to next_state and add i*delta to the checksum modulo modulus (nonnegative residue). "
        "Accept iff the final state is in accepting AND the final checksum equals checksum. Return "
        "count of accepted strings and lexicographically first and last; use null for first/last if none."),
    "robust-portfolio": ("Finances", portfolio_answer,
        "Choose a subset of indexed items with total cost <= budget. For each [a,b] in conflicts, "
        "never choose both. For each [a,b] in requires, choosing b requires choosing a. "
        "For each scenario s, ideal_profits[s] is the maximum achievable summed profit by ANY feasible "
        "subset in that scenario. Regret is ideal profit minus your chosen subset's profit. Choose the "
        "subset minimizing (maximum regret, sum of regrets, spend, sorted chosen index list), "
        "in that exact lexicographic priority. Return ideal_profits, chosen, spend, profits, regrets. "
        "Empty subsets are allowed, negative profits are possible; the choice is made before the scenario."),
}

CODE_PROMPTS = {
    "coupon-route": "Implement solve(data). Find a directed walk from start to end with at most max_hops "
        "edges visiting every vertex whose bit is set in required_mask. Repeated vertices are allowed. "
        "An optional coupon discounts exactly one traversed toll to toll//2. Minimize "
        "(cost,hops,vertex path,coupon position) lexicographically. Coupon positions are 0-based edge "
        "positions; -1 means unused and wins ties. Return null (Python None) if infeasible, otherwise "
        "{cost,hops,path,coupon}. Edges are unique [from,to,toll], with nonnegative integer tolls. "
        "Data contains start,end,max_hops,required_mask,edges. Do not mutate data.",
    "robust-assignment": "Implement solve(data). data['matrices'] contains three n by n nonnegative "
        "integer cost matrices (n<=7). Assignment p is a permutation: row i uses column p[i] in "
        "every scenario. First find each scenario's independent minimum assignment cost (ideals). "
        "Regret = cost - ideal. Choose p minimizing (maximum regret,sum of regrets,p) "
        "lexicographically; optimizing worst raw cost instead is incorrect. Return "
        "{assignment:list(p),costs:three totals,regrets:three regrets,ideals:three minima}. "
        "Do not mutate data. Every scenario has equal importance.",
}


def make_discrimination_tasks(seed, variant):
    questions = []
    for family, (category, oracle, instructions) in FAMILIES.items():
        size = (7 if family == "resource-schedule" else 10 if family == "robust-portfolio" else 8) + variant
        rng = _rng(seed, variant, family)
        data = _diagnosis_data(rng, size, variant) if family == "adaptive-diagnosis" else _data(family, rng, size)
        answer = oracle(data)
        q = task(family, category, variant, "evaluation", instructions, data, answer,
                 seed=seed, atomic_paths=[k for k, value in answer.items() if isinstance(value, (list, dict))])
        q.id = f"Q16-{family}-s{seed}-v{variant + 1:02d}"
        q.source = "Original capability ladder; see BENCHMARK_DESIGN.md"
        q.metadata.update(family=f"Q16-{family}", challenge=REVISION, tier="extended" if variant else "core")
        questions.append(q)
    for family, oracle in (("coupon-route", route_answer), ("robust-assignment", assignment_answer)):
        rng = _rng(seed, variant, family)
        fixtures = []
        for i in range(28):
            size = 2 + i % ((4 if family == "coupon-route" else 5) + variant)
            data = _data(family, rng, size)
            if family == "coupon-route":
                # Keep random alternative routes but guarantee a feasible
                # baseline walk; no-solution controls are explicit below.
                pairs = {(a, b) for a, b, _ in data["edges"]}
                data["edges"].extend([a, a + 1, rng.randrange(1, 12)]
                                     for a in range(size - 1) if (a, a + 1) not in pairs)
            if family == "coupon-route" and i < 5:
                boundaries = [
                    {"start": 0, "end": 0, "max_hops": 0, "required_mask": 1, "edges": []},
                    {"start": 0, "end": 1, "max_hops": 1, "required_mask": 3, "edges": [[0, 1, 0]]},
                    {"start": 0, "end": 3, "max_hops": 2, "required_mask": 0,
                     "edges": [[0, 2, 4], [2, 3, 4], [0, 1, 4], [1, 3, 4]]},
                    {"start": 0, "end": 0, "max_hops": 4, "required_mask": 7,
                     "edges": [[0, 1, 1], [1, 0, 1], [0, 2, 1], [2, 0, 1]]},
                    {"start": 0, "end": 1, "max_hops": 6, "required_mask": 2, "edges": []},
                ]
                data = boundaries[i]
            if family == "robust-assignment" and i < 4:
                n = (1, 2, 3, 6 + variant)[i]
                data = {"matrices": [[[0 if a == b or i < 2 else 1 for b in range(n)]
                                      for a in range(n)] for _ in range(3)]}
                if i == 1:
                    data = {"matrices": [[[0, 30], [30, 0]], [[0, 0], [0, 0]], [[100, 95], [95, 100]]]}
            fixtures.append({"id": f"case-{i:02d}", "function": "solve", "args": [data],
                             "expected": oracle(data), "preserve_inputs": True,
                             "relative": 0, "tolerance": 0})
        questions.append(Question(
            f"Q16-{family}-s{seed}-v{variant + 1:02d}", "Advanced Coding", CODE_PROMPTS[family],
            "code_exec", fixtures, difficulty="expert", source="Original capability ladder",
            metadata={"family": f"Q16-{family}", "challenge": REVISION, "seed": seed,
                      "variant": variant, "tier": "extended" if variant else "core"},
            rubric=[{"id": f["id"], "weight": 1 / len(fixtures), "mandatory": True,
                     "dimension": "content"} for f in fixtures]))
    return questions
