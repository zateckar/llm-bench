"""Original v11 composition tasks; answer keys are checked by separate algorithms."""

from functools import lru_cache
import itertools
import json

from app.benchmarking.rigorous_cases import schema, structured


def task(family, category, seed, variant, prompt, answer, data, *, contract=None):
    q = structured(family, category, variant, "evaluation", prompt + "\n" + json.dumps(data), answer,
                   output_contract=contract)
    q.id = f"Q11-{family}-s{seed}-v{variant + 1:02d}"
    q.metadata.update(family=f"Q11-{family}", oracle_input=data, seed=seed, variant=variant)
    q.source = "Original rigorous-v11; research and independent checks in BENCHMARK_DESIGN.md"
    return q


def adaptive_decisions(rng, seed, variant):
    vectors = rng.sample(list(itertools.product(range(3), repeat=5)), 12)
    worlds = [{"id": f"w{i:02d}", "label": f"L{rng.randrange(3)}", "results": list(v)}
              for i, v in enumerate(vectors)]
    sensors = [{"id": f"s{i}", "cost": rng.randint(1, 7)} for i in range(5)]

    @lru_cache(None)
    def solve(indices, available):
        labels = {worlds[i]["label"] for i in indices}
        if len(labels) == 1:
            return 0, next(iter(labels)), None
        options = []
        for sensor in available:
            parts = [(outcome, tuple(i for i in indices if worlds[i]["results"][sensor] == outcome))
                     for outcome in range(3)]
            parts = [(value, part) for value, part in parts if part]
            if len(parts) < 2:
                continue
            children = [(value, solve(part, tuple(s for s in available if s != sensor)))
                        for value, part in parts]
            cost = sensors[sensor]["cost"] + max(child[0] for _, child in children)
            policy = sensors[sensor]["id"] + "{" + ",".join(
                f"{value}:{child[1]}" for value, child in children) + "}"
            options.append((cost, sensors[sensor]["id"], policy))
        cost, first, policy = min(options)
        return cost, policy, first

    cost, policy, first = solve(tuple(range(len(worlds))), tuple(range(5)))
    fixed = []
    for mask in itertools.product((0, 1), repeat=5):
        selected = [i for i, bit in enumerate(mask) if bit]
        if all(a["label"] == b["label"] or any(a["results"][i] != b["results"][i] for i in selected)
               for a, b in itertools.combinations(worlds, 2)):
            fixed.append((sum(sensors[i]["cost"] for i in selected), [f"s{i}" for i in selected]))
    fixed_cost, fixed_sensors = min(fixed)
    answer = {"adaptive_cost": cost, "first_query": first, "policy": policy,
              "fixed_cost": fixed_cost, "fixed_sensors": fixed_sensors,
              "adaptivity_gain": fixed_cost - cost}
    return task("adaptive-minimax", "Agentic Use Cases", seed, variant,
        "One adversary chooses one of the listed worlds once and never changes it. Querying a "
        "sensor reveals that world's corresponding result, at the listed cost. Identify the LABEL, "
        "not necessarily the world. An adaptive policy may choose the next sensor from previous "
        "results, querying each sensor at most once per path. Minimize the WORST total query cost "
        "over all worlds. Stop immediately once all remaining worlds share a label. At every "
        "nonterminal node, among queries attaining the optimal remaining worst cost choose the "
        "lexicographically smallest sensor ID; apply that tie rule recursively. Encode its policy "
        "as a string sN{0:child,1:child,2:child}, with only feasible outcomes in numeric order, "
        "no spaces, and label strings as leaves. Give adaptive_cost, first_query and policy. "
        "Also find a fixed sensor subset whose combined results always identify the label, "
        "minimizing summed costs, then the sorted ID list lexicographically. Return fixed_cost, "
        "fixed_sensors and fixed_cost minus adaptive_cost as adaptivity_gain. No probabilities "
        "or external facts are available.", answer, {"worlds": worlds, "sensors": sensors})


def routes(edges, removed=()):
    paths = []
    def visit(node, path, cost):
        if node == "t":
            paths.append((cost, path))
            return
        for edge in edges:
            if edge["id"] not in removed and edge["from"] == node and edge["to"] not in path:
                visit(edge["to"], path + [edge["to"]], cost + edge["cost"])
    visit("s", ["s"], 0)
    return min(paths) if paths else (None, [])


def network_failures(rng, seed, variant):
    pairs = [("s", "a"), ("a", "b"), ("b", "t"), ("s", "c"), ("c", "d"), ("d", "t"),
             ("a", "c"), ("c", "a"), ("b", "d"), ("d", "b"), ("a", "d"), ("c", "b")]
    if variant % 2:
        # Three edge-disjoint corridors: a two-edge attack cannot disconnect.
        pairs += [("s", "b"), ("a", "t")]
    rng.shuffle(pairs)
    edges = [{"id": f"e{i:02d}", "from": a, "to": b, "cost": rng.randint(1, 9)}
             for i, (a, b) in enumerate(pairs)]
    baseline, path = routes(edges)
    scenarios = []
    cuts = []
    for size in range(3):
        for removed in itertools.combinations([e["id"] for e in edges], size):
            cost, reroute = routes(edges, removed)
            scenarios.append(((0 if cost is None else 1, -(cost or 0), size, removed), cost, reroute))
            if cost is None and not any(set(cut) <= set(removed) for cut in cuts):
                cuts.append(list(removed))
    key, worst_cost, reroute = min(scenarios)
    worst = list(key[3])
    repairs = [(routes(edges, set(worst) - {edge}), edge) for edge in worst]
    viable = [(cost, edge, repair_path) for (cost, repair_path), edge in repairs if cost is not None]
    restore_cost, restore_edge, restore_path = min(viable) if viable else (None, None, [])
    answer = {"baseline_cost": baseline, "baseline_path": path, "worst_failure": worst,
              "worst_cost": worst_cost, "worst_path": reroute, "minimal_cuts": sorted(cuts),
              "restore_edge": restore_edge, "restored_cost": restore_cost, "restored_path": restore_path}
    contract = schema(answer)
    contract.update(worst_cost="number or null", restored_cost="number or null", restore_edge="string or null")
    return task("network-interdiction", "Mathematical Reasoning", seed, variant,
        "This is a directed graph with positive additive edge costs. Routes run from s to t. "
        "Find the minimum-cost baseline route; ties choose the lexicographically smallest "
        "vertex list. An adversary may remove AT MOST two edges. Rank failure sets by unreachable "
        "first, then greatest shortest-route cost, then fewest removed edges, then lexicographically "
        "smallest sorted edge-ID list. Return the worst_failure, its shortest worst_path and "
        "worst_cost (null and [] when unreachable). List ALL inclusion-minimal disconnecting "
        "sets of size at most two as minimal_cuts, each sorted internally and the outer list "
        "sorted lexicographically. Restore at most ONE removed edge from worst_failure; among "
        "restorations reaching t minimize route cost then restored edge ID. Return restore_edge, "
        "restored_cost and restored_path; use null,null,[] if none can reconnect. Route ties "
        "always use vertex-list order. All other failed edges stay removed.", answer, {"edges": edges},
        contract=contract)


def history_facts(history, names):
    writers, reads = {}, []
    commits = {op["tx"]: i for i, op in enumerate(history) if op["op"] == "c"}
    recoverable = cascadeless = strict = True
    for i, op in enumerate(history):
        tx, kind = op["tx"], op["op"]
        if kind == "c":
            continue
        previous = writers.get(op["key"])
        if previous and previous != tx and commits[previous] > i:
            strict = False
        if kind == "r":
            reads.append([op["id"], previous])
            if previous and previous != tx:
                recoverable &= commits[previous] < commits[tx]
                cascadeless &= commits[previous] < i
        else:
            writers[op["key"]] = tx
    edges = set()
    for earlier, later in itertools.combinations(history, 2):
        if (earlier["op"] != "c" and later["op"] != "c" and earlier["tx"] != later["tx"]
            and earlier["key"] == later["key"] and "w" in (earlier["op"], later["op"])):
            edges.add((earlier["tx"], later["tx"]))
    conflict, view = [], []
    for order in itertools.permutations(names):
        position = {tx: i for i, tx in enumerate(order)}
        if all(position[a] < position[b] for a, b in edges):
            conflict.append(list(order))
        serial_writers, serial_reads = {}, {}
        for tx in order:
            for op in history:
                if op["tx"] == tx and op["op"] == "r":
                    serial_reads[op["id"]] = serial_writers.get(op["key"])
                elif op["tx"] == tx and op["op"] == "w":
                    serial_writers[op["key"]] = tx
        if serial_reads == dict(reads) and serial_writers == writers:
            view.append(list(order))
    return {"reads_from": reads, "final_writers": writers, "conflict_edges": [list(e) for e in sorted(edges)],
            "conflict_orders": conflict, "view_orders": view,
            "recoverable": bool(recoverable), "cascadeless": bool(cascadeless), "strict": bool(strict)}


def transaction_history(rng, seed, variant):
    programs = {"T1": [("r", "x"), ("w", "y")], "T2": [("w", "x"), ("r", "y")],
                "T3": [("w", "y"), ("w", "x")], "T4": [("r", "y"), ("w", "z"), ("r", "x")]}
    mode = (variant + (2 if seed % 8 >= 4 else 0)) % 4
    if mode == 1:
        programs = {"T1": [("w", "x"), ("w", "y")], "T2": [("w", "x"), ("w", "y")],
                    "T3": [("w", "x"), ("w", "y")], "T4": [("r", "x"), ("w", "z")]}
    def pending_events():
        return {tx: [{"id": f"{tx}-{i}", "tx": tx, "op": kind, "key": key}
                    for i, (kind, key) in enumerate(ops)] + [{"id": f"{tx}-c", "tx": tx, "op": "c"}]
               for tx, ops in programs.items()}
    pending = pending_events()
    history = []
    if mode == 0:
        order = list(programs)
        rng.shuffle(order)
        history = [op for tx in order for op in pending[tx]]
    elif mode in (1, 3):
        order = (["T1", "T2", "T2", "T1", "T2", "T1", "T3", "T3", "T3", "T4", "T4", "T4"]
                 if mode == 1 else
                 ["T2", "T2", "T2", "T4", "T4", "T3", "T3", "T3", "T4", "T4", "T1", "T1", "T1"])
        history = [pending[tx].pop(0) for tx in order]
    else:
        # Force a dirty-read/nonrecoverable case, rather than four random
        # histories that accidentally all share the same classification.
        for _ in range(1000):
            pending, history = pending_events(), []
            while pending:
                tx = rng.choice(sorted(pending))
                history.append(pending[tx].pop(0))
                if not pending[tx]:
                    del pending[tx]
            if not history_facts(history, sorted(programs))["recoverable"]:
                break
        else:
            raise ValueError("Could not generate a nonrecoverable history")
    aliases = dict(zip(sorted(programs), rng.sample(sorted(programs), len(programs))))
    for op in history:
        original = op["tx"]
        op["tx"] = aliases[original]
        op["id"] = op["id"].replace(original, aliases[original])
    answer = history_facts(history, sorted(programs))
    return task("transaction-consistency", "Advanced Coding", seed, variant,
        "Analyze this complete transaction history in displayed order. Each tx preserves its "
        "internal operation order and commits at c; there are no aborts. A read sees the most "
        "recent preceding write to that key even if uncommitted, or the initial state (writer null). "
        "All writes to a given key write the SAME value; writer identity still matters. "
        "Return reads_from as [read-event-ID,writer-tx-or-null] pairs in history order, and "
        "final_writers by key. A conflict edge Ti->Tj exists when an earlier Ti operation and "
        "later Tj operation access the same key with at least one write. Ignore commits for "
        "conflicts. Give ALL conflict-compatible serial tx orders and ALL view-equivalent serial "
        "orders (same source writer for EACH read and same final writer for EACH key); arrays "
        "are sorted lexicographically. Equal written values do not establish view equivalence. "
        "Recoverable: every writer read by another tx commits before that reader commits. "
        "Cascadeless: every external source writer commits before the read. Strict: after a "
        "tx writes a key, no OTHER tx reads OR writes that key before the writer commits. "
        "Return those three booleans separately. Sort conflict_edges lexicographically.",
        answer, {"history": history})


def closure(facts, rules):
    known = set(facts)
    while True:
        added = {r["then"] for r in rules if set(r["if"]) <= known}
        if added <= known:
            return known
        known |= added


def grounded_proofs(rng, seed, variant):
    letters = rng.sample(list("ABCDEFGHIJKLMNPQRST"), 10)
    rename = dict(zip("ABCDEXYUVG", letters))
    def atom(value):
        return value[0] + rename[value[1]]
    literals = ["+A", "+B", "-B", "+C", "+D", "-A", "+A", "+E"]
    rng.shuffle(literals)
    facts = [{"id": f"F{i:02d}", "atom": atom(value)} for i, value in enumerate(literals)]
    templates = [(["+A", "+B"], "+X"), (["+C", "-B"], "+X"), (["+X"], "+G"),
                 (["+D", "+E"], "+G"), (["-A", "+D"], "-G"), (["-B", "+E"], "-G"),
                 (["+U"], "+V"), (["+V"], "+U"), (["+U"], "+G"),
                 (["+X"], "+Y"), (["+Y"], "+X")]
    rng.shuffle(templates)
    rules = [{"id": f"R{i:02d}", "if": [atom(a) for a in premises], "then": atom(out)}
             for i, (premises, out) in enumerate(templates)]
    target = rename["G"]
    supports = {"+": [], "-": []}
    false_only = []
    ids = [f["id"] for f in facts]
    for size in range(len(facts) + 1):
        for subset in itertools.combinations(ids, size):
            known = closure([f["atom"] for f in facts if f["id"] in subset], rules)
            for sign in supports:
                if sign + target in known and not any(set(s) <= set(subset) for s in supports[sign]):
                    supports[sign].append(list(subset))
            removed = sorted(set(ids) - set(subset))
            if "-" + target in known and "+" + target not in known:
                false_only.append(removed)
    minimal_size = min(map(len, false_only)) if false_only else None
    answer = {"status": "both", "positive_supports": sorted(supports["+"]),
              "negative_supports": sorted(supports["-"]),
              "unfounded_cycle_derived": False,
              "minimum_retractions": sorted(s for s in false_only if len(s) == minimal_size)}
    return task("grounded-proof-repair", "Truthfulness", seed, variant,
        "Use ONLY these signed facts and conjunction rules. +A and -A are distinct atoms; "
        "contradiction does not imply unrelated atoms. Rules add their conclusion when ALL "
        "premises have been derived; apply to the least fixed point, without contraposition "
        "or negation-as-failure. Cycles cannot justify themselves. status for the target is "
        "true_only (+ only), false_only (- only), both, or neither. For each sign, list ALL "
        "inclusion-minimal sets of base FACT IDs that derive it with all rules available. "
        "Duplicate facts with different IDs are independent evidence alternatives. Rules "
        "are free and are never included in support lists. Sort every support internally and "
        "each outer list lexicographically. unfounded_cycle_derived asks whether either of "
        "the two displayed unseeded cycle atoms is derivable from the full facts. Finally list "
        "ALL minimum-CARDINALITY fact-ID retraction sets that leave the target false_only "
        "(negative derivable and positive NOT derivable); sort as above. Only facts may be "
        "retracted, not rules. Use [] if impossible.", answer,
        {"facts": facts, "rules": rules, "target": target, "unseeded_atoms": [atom("+U"), atom("+V")]})


BUILDERS = (adaptive_decisions, network_failures, transaction_history, grounded_proofs)


def load_adversarial_questions(seed, variant):
    from app.benchmarking.quality_suite import rng_for
    return [builder(rng_for(seed, variant, builder.__name__), seed, variant) for builder in BUILDERS]
