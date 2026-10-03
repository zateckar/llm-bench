"""Original v12 tasks: concurrency witnesses and nested public knowledge.

Inspired by BBEH's composed reasoning and IFEval's per-requirement checks.
Public, deterministic instances; empirical difficulty still requires model runs.
"""

from collections import deque
import hashlib
import itertools
import json
import random

from rigorous_cases import structured


class _Uninformative(ValueError):
    pass


def _task(family, category, seed, variant, prompt, data, answer):
    q = structured(family, category, variant, "evaluation",
                   prompt + "\nINPUT=" + json.dumps(data, sort_keys=True), answer)
    q.id = f"Q12-{family}-s{seed}-v{variant + 1:02d}"
    q.metadata.update(family=f"Q12-{family}", seed=seed, variant=variant, oracle_input=data)
    q.source = "Original rigorous-v12; see BENCHMARK_DESIGN.md"
    return q


def queue_orders(ops, capacity):
    """Enumerate feasible topological orders, pruning invalid queue transitions."""
    before = {op["id"]: {a["id"] for a in ops if a["end"] < op["start"]} for op in ops}
    found = []

    def visit(order, queue, remaining):
        if not remaining:
            found.append((order, queue))
        for op in remaining:
            if not before[op["id"]].issubset(order):
                continue
            after = list(queue)
            if op["kind"] == "put":
                result = len(queue) < capacity
                if result:
                    after.append(op["value"])
            else:
                result = after.pop(0) if after else None
            if type(result) is type(op["result"]) and result == op["result"]:
                visit(order + [op["id"]], after, [a for a in remaining if a is not op])
    visit([], [], ops)
    return sorted(found)


def concurrency_case(rng, seed, variant):
    # Start from a feasible history, then change one observation for odd variants.
    # IDs are shuffled so the canonical order is not the prompt's physical order.
    names = rng.sample([f"o{i}" for i in range(8)], 8)
    x, y = rng.sample(range(11, 80), 2)
    ops = [
        {"kind": "put", "value": x, "result": True, "start": 0, "end": 4},
        {"kind": "put", "value": y, "result": True, "start": 1, "end": 5},
        {"kind": "take", "result": x, "start": 3, "end": 7},
        {"kind": "put", "value": x, "result": True, "start": 6, "end": 10},
        {"kind": "put", "value": y, "result": False, "start": 8, "end": 12},
        {"kind": "take", "result": y, "start": 11, "end": 15},
        {"kind": "take", "result": x, "start": 13, "end": 17},
        {"kind": "take", "result": None, "start": 16, "end": 20},
    ]
    for name, op in zip(names, ops):
        op["id"] = name
    if variant:
        # A value that was actually enqueued, but consuming it here conflicts
        # with later results. Removing one read need not repair the whole log.
        ops[2]["result"] = y
    rng.shuffle(ops)
    orders = queue_orders(ops, 2)
    repairs = []
    ids = sorted(op["id"] for op in ops)
    for size in range(len(ops) + 1):
        for removed in itertools.combinations(ids, size):
            solutions = queue_orders([op for op in ops if op["id"] not in removed], 2)
            if solutions:
                repairs.append({"removed": list(removed), "order": solutions[0][0],
                                "final_queue": solutions[0][1], "count": len(solutions)})
        if repairs:
            break
    answer = {"linearizable": bool(orders), "linearization_count": len(orders),
              "canonical_order": orders[0][0] if orders else [],
              "possible_final_queues": sorted({tuple(state) for _, state in orders}),
              "minimum_removals": size, "repairs": repairs}
    # Use JSON-native arrays in the answer key.
    answer["possible_final_queues"] = [list(state) for state in answer["possible_final_queues"]]
    return _task("queue-linearizability", "Advanced Coding", seed, variant,
        "Audit a concurrent log of a capacity-2 FIFO queue, initially empty. Each completed "
        "operation must act atomically once between its start and end. Real-time precedence "
        "requires a before b exactly when a.end < b.start; equal endpoints impose no order. "
        "put(value) returns true and appends iff space exists, otherwise returns false and "
        "does nothing. take returns and removes the oldest item, or null if empty. Equal "
        "values are indistinguishable, but operation IDs distinguish orders. All recorded "
        "results must match, including failed puts and empty takes. Determine linearizable, "
        "linearization_count (number of valid ID permutations), canonical_order (lexically "
        "smallest valid ID list; [] if none), and possible_final_queues (distinct numeric "
        "lists sorted lexically; [] if none). Then delete the fewest completed operations "
        "to obtain a linearizable log, preserving every retained timestamp and result and "
        "restarting from the empty queue. Report minimum_removals and ALL minimum repairs, "
        "ordered by their sorted removed-ID lists. Each repair has removed, its canonical "
        "order, final_queue for that order, and count of valid retained permutations. "
        "If already valid, the only repair removes []. Do not treat deletion as undoing an "
        "operation after execution: deleted operations never execute.",
        {"capacity": 2, "operations": ops}, answer)


def truth_set(formula, worlds):
    kind, *args = formula
    ids = {w["id"] for w in worlds}
    if kind == "atom":
        return {w["id"] for w in worlds if w[args[0]]}
    if kind == "not":
        return ids - truth_set(args[0], worlds)
    if kind == "and":
        return truth_set(args[0], worlds) & truth_set(args[1], worlds)
    if kind == "K":
        agent, inner = args
        valid = truth_set(inner, worlds)
        bad_observations = {w[agent] for w in worlds if w["id"] not in valid}
        return {w["id"] for w in worlds if w[agent] not in bad_observations}
    raise ValueError(kind)


def knowledge_case(rng, seed, variant):
    worlds = [{"id": f"w{i:02d}", "A": rng.randrange(5), "B": rng.randrange(5),
               "C": rng.randrange(5), "p": bool(rng.randrange(4)), "q": bool(rng.randrange(2))}
              for i in range(14)]
    p, q = ["atom", "p"], ["atom", "q"]
    # Pick informative announcements by semantics, never by expected answer text.
    candidates = [["not", ["K", a, q]] for a in "ABC"]
    candidates += [["K", a, ["not", ["K", b, f]]]
                   for a in "ABC" for b in "ABC" if a != b for f in (q,)]
    candidates += [["not", f] for f in candidates[3:]]
    rng.shuffle(candidates)
    current = worlds
    announcements, rounds = [], []
    for _ in range(2):
        options = [(f, truth_set(f, current)) for f in candidates]
        options = [(f, valid) for f, valid in options if 3 <= len(valid) < len(current)]
        if not options:
            raise _Uninformative("No informative announcement")
        formula, valid = max(options, key=lambda option: len(option[1]))
        announcements.append(formula)
        current = [w for w in current if w["id"] in valid]
        rounds.append(sorted(valid))
    probes = [["K", "A", p], ["K", "B", ["K", "A", p]],
              ["not", ["K", "C", q]], ["K", "A", ["not", ["K", "B", q]]]]
    answers = [sorted(truth_set(f, current)) for f in probes]
    everyone = sorted(set.intersection(*(truth_set(["K", a, p], current) for a in "AB")))
    common, counterexamples = [], []
    by_id = {w["id"]: w for w in current}
    adjacency = {w["id"]: sorted(v["id"] for v in current
                                if any(v[a] == w[a] for a in "AB")) for w in current}
    for start in sorted(by_id):
        paths, seen, counterexample = deque([[start]]), {start}, []
        while paths:
            path = paths.popleft()
            if not by_id[path[-1]]["p"]:
                counterexample = path
                break
            for other in adjacency[path[-1]]:
                if other not in seen:
                    seen.add(other)
                    paths.append(path + [other])
        if not counterexample:
            common.append(start)
        counterexamples.append(counterexample)
    if (everyone == common or max(map(len, counterexamples), default=0) < 3
        or (variant == 1 and not common)
        or not any(f[0] == "K" or f[1][0] == "K" and f[1][2][0] == "not"
                   for f in announcements)):
        raise _Uninformative("Need nested knowledge and a mutual/common distinction")
    answer = {"survivors_after_each": rounds, "probe_worlds": answers,
              "everyone_AB_p": everyone, "common_AB_p": common,
              "counterexample_paths": counterexamples}
    return _task("public-knowledge", "Logical Reasoning", seed, variant,
        "Use only the finite epistemic model given below. Each world has truth values p,q "
        "and observation labels A,B,C. An agent considers exactly the CURRENT worlds with "
        "the same observation label possible; labels from different agents are unrelated. "
        "Formula ASTs use ['atom',name], ['not',f], ['and',f,g], and ['K',agent,f]. K means "
        "f holds in every world that agent considers possible. Apply the announcements in "
        "order: evaluate an announcement at ALL current worlds using the same pre-announcement "
        "model, then simultaneously remove the false worlds. Recompute knowledge on the "
        "restricted model before the next announcement. Return survivors_after_each as one "
        "sorted ID list per announcement. On the FINAL model return probe_worlds in probe "
        "order, each listing all worlds where that formula holds. everyone_AB_p lists worlds "
        "where both K_A(p) and K_B(p) hold. common_AB_p lists worlds where p holds at every "
        "world reachable by any finite chain (including length zero) of A-or-B observation "
        "links. These are different notions. counterexample_paths has one world-ID path "
        "per FINAL world in sorted ID order: the shortest path starting there to a world "
        "where p is false; among equally short paths choose the lexically smallest full "
        "world-ID list. Adjacent worlds must share A or B; do not use C. Use [] when no "
        "such world is reachable. A false starting world has the one-element path [itself]. "
        "All world lists are lexically sorted unless path or probe order specifies otherwise.",
        {"worlds": worlds, "announcements": announcements, "probes": probes}, answer)


def load_compositional_questions(seed, variant):
    questions = []
    for family in (concurrency_case, knowledge_case):
        instance = 0 if family is concurrency_case else variant
        digest = hashlib.sha256(f"v12:{family.__name__}:{seed}:{instance}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        for _ in range(1000):
            try:
                questions.append(family(rng, seed, variant))
                break
            except _Uninformative:
                continue
        else:
            raise ValueError("Cannot construct contrastive knowledge instance")
    return questions
