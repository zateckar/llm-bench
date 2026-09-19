"""Executable answer-key derivations for the challenge-v2 questions.

These are offline authoring/verification tools, never sent to the model. Small
exhaustive searches deliberately favor auditability over clever closed forms.
"""

from collections import Counter, deque
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations, product
from math import comb, gcd


def fraction(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def lr01():
    valid = []
    for p in permutations("ABCDEFG"):
        pos = {v: p.index(v) for v in p}
        if (
            pos["A"] < pos["D"] < pos["G"]
            and pos["B"] < pos["E"]
            and abs(pos["C"] - pos["F"]) == 2
            and pos["E"] < pos["G"]
            and pos["B"] != 0
            and abs(pos["A"] - pos["B"]) != 1
        ):
            valid.append("".join(p))
    return {"count": len(valid), "first": min(valid), "last": max(valid)}


def lr02():
    models = []
    for a, b, c, d, e, f in product([False, True], repeat=6):
        claims = [b != c, a and not d, sum([a, b, d]) == 2, c == e, not f or b, d != a]
        if list((a, b, c, d, e, f)) == claims:
            models.append("".join("T" if x else "F" for x in claims))
    return {"models": sorted(models), "count": len(models)}


def lr03():
    edges = [("A", "D"), ("B", "D"), ("B", "E"), ("C", "F"), ("D", "G"), ("E", "G")]
    durations = dict(zip("ABCDEFG", [3, 5, 2, 4, 3, 5, 2]))

    # Enumerate all integral starts within a known feasible upper bound using
    # event-based scheduling; include intentional idling as a real choice.
    @lru_cache(None)
    def solve(done, running):
        if len(done) == 7:
            return 0
        busy = {x for x, _ in running}
        ready = [
            x
            for x in durations
            if x not in done and x not in busy and all(a in done for a, b in edges if b == x)
        ]
        answers = []
        for k in range(min(2 - len(running), len(ready)) + 1):
            for chosen in combinations(ready, k):
                active = list(running) + [(x, durations[x]) for x in chosen]
                if not active:
                    continue
                elapsed = min(t for _, t in active)
                finished = "".join(sorted(done + "".join(x for x, t in active if t == elapsed)))
                pending = tuple(sorted((x, t - elapsed) for x, t in active if t > elapsed))
                answers.append(elapsed + solve(finished, pending))
        return min(answers)

    return {"two_workers": solve("", ()), "total_work": sum(durations.values())}


def lr04():
    # Shortest path in the complete state graph, tracking number of shortest paths.
    start, target = (0, 0), (6, 0)
    dist, ways = {start: 0}, {start: 1}
    q = deque([start])
    while q:
        a, b = q.popleft()
        moves = {
            (9, b),
            (a, 4),
            (0, b),
            (a, 0),
            (a - min(a, 4 - b), b + min(a, 4 - b)),
            (a + min(b, 9 - a), b - min(b, 9 - a)),
        } - {(a, b)}
        for state in moves:
            if state not in dist:
                dist[state] = dist[(a, b)] + 1
                ways[state] = 0
                q.append(state)
            if dist[state] == dist[(a, b)] + 1:
                ways[state] += ways[(a, b)]
    return {"operations": dist[target], "shortest_state_paths": ways[target]}


def lr05():
    @lru_cache(None)
    def win(n, previous):
        return any(k != previous and k < n and not win(n - k, k) for k in [1, 3, 4])

    return {
        "winning_first_moves": [k for k in [1, 3, 4] if not win(23 - k, k)],
        "losing_sizes": [n for n in range(1, 25) if not win(n, 0)],
    }


def lr06():
    models = []
    for bits in product([False, True], repeat=6):
        a, b, c, d, e, f = bits
        if (
            (not a or b)
            and (not b or c)
            and (not (c and d))
            and (d or e)
            and (e == (a != f))
            and sum(bits) == 3
        ):
            models.append(bits)
    return {
        "count": len(models),
        "always_true": [x for i, x in enumerate("ABCDEF") if all(m[i] for m in models)],
        "always_false": [x for i, x in enumerate("ABCDEF") if all(not m[i] for m in models)],
    }


def lr07():
    dates = [
        ("Apr", 11),
        ("Apr", 14),
        ("Apr", 17),
        ("Jun", 12),
        ("Jun", 16),
        ("Sep", 11),
        ("Sep", 13),
        ("Dec", 13),
        ("Dec", 14),
        ("Dec", 12),
    ]
    counts = Counter(d for _, d in dates)
    first = [(m, d) for m, d in dates if all(counts[x] > 1 for month, x in dates if month == m)]
    second_counts = Counter(d for _, d in first)
    second = [(m, d) for m, d in first if second_counts[d] == 1]
    third_counts = Counter(m for m, _ in second)
    third = [(m, d) for m, d in second if third_counts[m] == 1]
    return {
        "remaining_counts": [len(first), len(second), len(third)],
        "dates": [f"{m}-{d}" for m, d in sorted(third)],
    }


def lr08():
    # Hitting sets: every alarm path must contain a faulty component, and quiet
    # paths contain none. Return ALL smallest explanations, not just one.
    alarms = [set("ABC"), set("BDE"), set("CEF"), set("ADG"), set("EFG")]
    good = set("D")
    for n in range(8):
        hits = [
            "".join(c)
            for c in combinations("ABCDEFG", n)
            if not set(c) & good and all(set(c) & a for a in alarms)
        ]
        if hits:
            return {"faults": n, "explanations": hits}


def lr09():
    edges = [
        ("S", "A", 4, 2),
        ("S", "B", 2, 4),
        ("A", "B", 1, 1),
        ("A", "C", 5, 1),
        ("B", "C", 2, 3),
        ("B", "D", 6, 1),
        ("C", "D", 1, 1),
        ("C", "T", 5, 2),
        ("D", "T", 2, 3),
    ]
    options = []

    def visit(node, cost, fuel, path):
        if fuel > 9:
            return
        if node == "T":
            options.append((cost, fuel, path))
        for u, v, c, f in edges:
            if u == node:
                visit(v, cost + c, fuel + f, path + v)

    visit("S", 0, 0, "S")
    cost, fuel, path = min(options)
    return {"cost": cost, "fuel": fuel, "path": path, "feasible_paths": len(options)}


def lr10():
    # Conflict graph from the trace, then enumerate every topological order.
    trace = [
        ("1", "r", "x"),
        ("2", "r", "y"),
        ("3", "r", "z"),
        ("1", "w", "y"),
        ("4", "r", "x"),
        ("2", "w", "z"),
        ("3", "w", "x"),
        ("4", "w", "w"),
    ]
    edges = sorted(
        {
            a[0] + b[0]
            for i, a in enumerate(trace)
            for b in trace[i + 1 :]
            if a[0] != b[0] and a[2] == b[2] and "w" in (a[1], b[1])
        }
    )
    orders = [
        "".join(p) for p in permutations("1234") if all(p.index(a) < p.index(b) for a, b in edges)
    ]
    return {"edges": edges, "serial_orders": orders}


def lr11():
    ballots = [(6, "ABCD"), (5, "BCDA"), (4, "CDAB"), (3, "DCBA"), (2, "DBAC")]
    live, rounds, eliminated = set("ABCD"), [], []
    while len(live) > 1:
        counts = {
            c: sum(n for n, ranking in ballots if next(x for x in ranking if x in live) == c)
            for c in sorted(live)
        }
        rounds.append(counts)
        loser = min(live, key=lambda c: (counts[c], c))
        eliminated.append(loser)
        live.remove(loser)
    return {"rounds": rounds, "eliminated": eliminated, "winner": next(iter(live))}


def lr12():
    costs = [[9, 2, 7, 8, 6], [6, 4, 3, 7, 5], [5, 8, 1, 8, 3], [7, 6, 9, 4, 2], [8, 5, 2, 9, 7]]
    candidates = [
        (sum(costs[i][p[i]] for i in range(5)), "".join(str(j + 1) for j in p))
        for p in permutations(range(5))
        if p[0] != 1 and p[2] != 2 and p[1] < p[4]
    ]
    best = min(candidates)[0]
    return {"cost": best, "assignments": sorted(p for c, p in candidates if c == best)}


def mr01():
    counts = [0, 0]
    for digits in product(range(4), repeat=8):
        if digits[0] == 0 or any(a == b for a, b in zip(digits, digits[1:])):
            continue
        n = sum(d * 4 ** (7 - i) for i, d in enumerate(digits))
        if sum(digits) == 12:
            counts[0] += 1
            counts[1] += n % 7 == 0
    return {"sum_12": counts[0], "also_divisible_by_7": counts[1]}


def mr02():
    possibilities = [
        p
        for p in product(range(7), repeat=5)
        if sum(p) == 17 and p[0] >= 2 and p[1] % 2 == 0 and p[2] < p[3] and p[4] >= 1
    ]
    return {
        "count": len(possibilities),
        "with_last_at_least_4": sum(p[4] >= 4 for p in possibilities),
    }


def mr03():
    # Prior urn probabilities are unequal; likelihood includes a selective report.
    weights = []
    for red, blue, prior in [(4, 2, Fraction(1, 4)), (2, 5, Fraction(3, 4))]:
        weights.append(
            [
                prior
                * Fraction(comb(red, r) * comb(blue, 3 - r), comb(red + blue, 3))
                * (Fraction(1) if r == 2 else Fraction(1, 3) if r == 1 else 0)
                for r in range(4)
            ]
        )
    evidence = sum(map(sum, weights))
    return {
        "report_probability": fraction(evidence),
        "urn_A_given_report": fraction(sum(weights[0]) / evidence),
        "two_red_given_report": fraction(sum(w[2] for w in weights) / evidence),
    }


def mr04():
    # Fraction Gaussian elimination for absorbing Markov chain expectations.
    a = [
        [Fraction(x) for x in row]
        for row in [
            [1, Fraction(-1, 2), Fraction(-1, 2), 1],
            [Fraction(-1, 3), 1, Fraction(-1, 3), 1],
            [Fraction(-1, 4), 0, Fraction(1, 2), 1],
        ]
    ]
    for i in range(3):
        pivot = a[i][i]
        a[i] = [v / pivot for v in a[i]]
        for j in range(3):
            if j != i:
                factor = a[j][i]
                a[j] = [x - factor * y for x, y in zip(a[j], a[i])]
    return {"A": fraction(a[0][-1]), "B": fraction(a[1][-1]), "C": fraction(a[2][-1])}


def mr05():
    valid = [n for n in range(1, 5001) if n % 12 == 5 and n % 18 == 11 and gcd(n, 35) == 1]
    return {"count": len(valid), "sum": sum(valid), "smallest": min(valid), "largest": max(valid)}


def mr06():
    # Count labeled permutations under simultaneous local and global restrictions.
    valid = [
        p
        for p in permutations(range(1, 8))
        if all(x != i for i, x in enumerate(p, 1))
        and abs(p.index(1) - p.index(2)) > 1
        and p.index(3) < p.index(4)
    ]
    even = sum(sum(p[i] > p[j] for i in range(7) for j in range(i + 1, 7)) % 2 == 0 for p in valid)
    return {"count": len(valid), "even_permutations": even}


def mr07():
    counts = Counter()
    for rolls in product(range(1, 7), repeat=4):
        if sum(rolls) >= 17 and len(set(rolls)) == 3:
            counts[0] += 1
            counts[1] += max(rolls) == 6
    return {
        "event_probability": fraction(Fraction(counts[0], 6**4)),
        "six_given_event": fraction(Fraction(counts[1], counts[0])),
    }


def mr08():
    # DP over lattice positions with obstacles and a required waypoint.
    blocked = {(2, 2), (3, 4), (5, 3)}

    @lru_cache(None)
    def ways(x, y, seen):
        if x > 7 or y > 6 or (x, y) in blocked or y > x + 1:
            return 0
        seen = seen or (x, y) == (4, 2)
        if (x, y) == (7, 6):
            return int(seen)
        return ways(x + 1, y, seen) + ways(x, y + 1, seen)

    return {"paths": ways(0, 0, False)}


def mr09():
    # Exact quadratic minimization on a bounded integer feasible region.
    options = [
        (3 * x * x + 2 * y * y + z * z - 2 * x * y - 4 * y * z, [x, y, z])
        for x in range(13)
        for y in range(13)
        for z in range(13)
        if 2 * x + 3 * y + z == 24 and x + y + z >= 10
    ]
    best = min(v for v, _ in options)
    return {"minimum": best, "minimizers": [xyz for v, xyz in options if v == best]}


def mr10():
    # Explicit orbit canonicalization avoids accidentally quotienting reflections.
    valid = [
        "".join(p)
        for p in product("ABC", repeat=9)
        if all(p.count(c) == 3 for c in "ABC") and all(p[i] != p[(i + 1) % 9] for i in range(9))
    ]

    def rotation(s):
        return min(s[i:] + s[:i] for i in range(len(s)))

    return {
        "labeled": len(valid),
        "up_to_rotation": len({rotation(s) for s in valid}),
        "up_to_rotation_and_reflection": len({min(rotation(s), rotation(s[::-1])) for s in valid}),
    }


def mr11():
    # Exact expectation DP over the subsets of types already collected.
    @lru_cache(None)
    def expectation(mask):
        if mask == 15:
            return Fraction(0)
        weights = [1, 2, 3, 4]
        unseen = [i for i in range(4) if not mask & (1 << i)]
        return (10 + sum(weights[i] * expectation(mask | (1 << i)) for i in unseen)) / sum(
            weights[i] for i in unseen
        )

    return {"from_empty": fraction(expectation(0)), "after_types_1_and_4": fraction(expectation(9))}


def mr12():
    def v(n, prime):
        out = 0
        while n:
            n //= prime
            out += n
        return out

    n = next(n for n in range(1, 1000) if min(v(n, 2) // 7, v(n, 3) // 4, v(n, 5) // 2) >= 8)
    return {
        "n": n,
        "valuations": [v(n, p) for p in [2, 3, 5]],
        "previous_max_power": min(v(n - 1, 2) // 7, v(n - 1, 3) // 4, v(n - 1, 5) // 2),
    }


REASONING = {f"LR2-{i:02}": globals()[f"lr{i:02}"] for i in range(1, 13)}
REASONING.update({f"MR2-{i:02}": globals()[f"mr{i:02}"] for i in range(1, 13)})


def temporal_route(n, edges, start, end, departure):
    """Reference: repeated relaxation; departures must be inside [open, close)."""
    times = [float("inf")] * n
    times[start] = departure
    for _ in range(n):
        for u, v, duration, opening, closing in edges:
            leave = max(times[u], opening)
            if leave < closing:
                times[v] = min(times[v], leave + duration)
    return None if times[end] == float("inf") else times[end]


def interval_overlay(intervals):
    """Reference: partition at every boundary, count, then coalesce."""
    boundaries = sorted({x for a, b in intervals if a < b for x in [a, b]})
    out = []
    for a, b in zip(boundaries, boundaries[1:]):
        count = sum(lo <= a < hi for lo, hi in intervals)
        if count:
            if out and out[-1][1] == a and out[-1][2] == count:
                out[-1][1] = b
            else:
                out.append([a, b, count])
    return out


def dependency_batches(nodes, edges):
    remaining, batches = set(nodes), []
    while remaining:
        ready = sorted(v for v in remaining if not any(b == v and a in remaining for a, b in edges))
        if not ready:
            return {"batches": batches, "blocked": sorted(remaining)}
        batches.append(ready)
        remaining.difference_update(ready)
    return {"batches": batches, "blocked": []}


def versioned_read(events, queries):
    out = []
    for key, time in queries:
        matching = [
            (t, i, value) for i, (k, t, value) in enumerate(events) if k == key and t <= time
        ]
        out.append(max(matching, key=lambda v: (v[0], v[1]))[2] if matching else None)
    return out


def shortest_subarray(nums, target):
    best = None
    for start in range(len(nums)):
        total = 0
        for end in range(start + 1, len(nums) + 1):
            total += nums[end - 1]
            if total >= target:
                candidate = (end - start, start, end)
                if best is None or candidate < best:
                    best = candidate
    return list(best[1:]) if best else [-1, -1]


def optimal_jobs(jobs):
    # Exhaustive subset oracle independent of the expected DP implementation.
    best = (0, ())
    for mask in range(1 << len(jobs)):
        chosen = [job for i, job in enumerate(jobs) if mask & (1 << i)]
        ordered = sorted(chosen, key=lambda j: (j[1], j[2], j[0]))
        if any(a[2] > b[1] for a, b in zip(ordered, ordered[1:])):
            continue
        profit = sum(j[3] for j in chosen)
        ids = tuple(sorted(j[0] for j in chosen))
        if profit > best[0] or (profit == best[0] and ids < best[1]):
            best = (profit, ids)
    return {"profit": best[0], "ids": list(best[1])}


def reconcile(events):
    seen, balances, rejected = set(), {}, []
    for event_id, source, dest, amount in events:
        if event_id in seen:
            continue
        seen.add(event_id)  # failed attempts consume their id too
        if amount <= 0 or source == dest or (source != "BANK" and balances.get(source, 0) < amount):
            rejected.append(event_id)
            continue
        if source != "BANK":
            balances[source] = balances.get(source, 0) - amount
        if dest != "BANK":
            balances[dest] = balances.get(dest, 0) + amount
    return {"balances": dict(sorted(balances.items())), "rejected": rejected}


def glob_match(pattern, text):
    # Tokenize escaped literals before applying full-string DP.
    tokens, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            i += 1
            if i == len(pattern):
                return False
            tokens.append(("literal", pattern[i]))
        else:
            tokens.append(("wild" if c in "*?" else "literal", c))
        i += 1
    states = {0}
    for kind, c in tokens:
        nxt = set()
        for pos in states:
            if kind == "wild" and c == "*":
                nxt.update(range(pos, len(text) + 1))
            elif pos < len(text) and ((kind == "wild" and c == "?") or c == text[pos]):
                nxt.add(pos + 1)
        states = nxt
    return len(text) in states


CODE = dict(
    zip(
        [f"AC2-{i:02}" for i in range(1, 9)],
        [
            temporal_route,
            interval_overlay,
            dependency_batches,
            versioned_read,
            shortest_subarray,
            optimal_jobs,
            reconcile,
            glob_match,
        ],
    )
)


def au01():
    # Exhaustive deployment batches; old instances consume 2 units, new ones 3.
    # Initially 4 old, 0 new; keep >=3 ready and <=12 resource units.
    start = (4, 0)
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        (old, new), path = queue.popleft()
        if (old, new) == (0, 4):
            return {"steps": len(path), "plan": path}
        # Lexicographic tie rule: add before remove, then smaller batch first.
        for action in ["add", "remove"]:
            for count in [1, 2]:
                state = (old, new + count) if action == "add" else (old - count, new)
                o, n = state
                if o < 0 or n > 4 or o + n < 3 or 2 * o + 3 * n > 12 or state in seen:
                    continue
                seen.add(state)
                queue.append((state, path + [[action, count]]))


def au02():
    # Circuit breaker: request order includes successes which reset failure count.
    records = [
        (0, False),
        (1, False),
        (2, True),
        (3, False),
        (4, False),
        (5, False),
        (7, True),
        (10, False),
        (11, True),
        (15, True),
        (16, False),
        (17, False),
        (18, True),
    ]
    failures, until, admitted, skipped = 0, None, [], []
    for t, succeeds in records:
        if until is not None and t < until:
            skipped.append(t)
            continue
        probe = until is not None
        admitted.append(t)
        if succeeds:
            failures, until = 0, None
        elif probe:
            until = t + 5
        else:
            failures += 1
            if failures == 3:
                until = t + 5
    return {
        "admitted": admitted,
        "skipped": skipped,
        "state": "closed",
        "consecutive_failures": failures,
    }


def au03():
    state = {"a": 10, "b": 20}
    # Ignore <= snapshot LSN, buffer uncommitted transactions, apply on commit.
    events = [
        (99, "x", "set", "a", 9),
        (101, "p", "set", "a", 11),
        (102, "q", "delete", "b", None),
        (103, "p", "set", "c", 30),
        (104, "q", "abort", None, None),
        (105, "p", "commit", None, None),
        (105, "p", "commit", None, None),
        (106, "r", "set", "a", 12),
        (107, "s", "set", "b", 25),
        (108, "s", "commit", None, None),
        (109, "r", "set", "c", 31),
    ]
    pending, seen, committed = {}, set(), []
    for lsn, txn, op, key, value in events:
        if lsn <= 100 or lsn in seen:
            continue
        seen.add(lsn)
        if op == "abort":
            pending.pop(txn, None)
        elif op == "commit":
            for action, k, v in pending.pop(txn, []):
                if action == "delete":
                    state.pop(k, None)
                else:
                    state[k] = v
            committed.append(txn)
        else:
            pending.setdefault(txn, []).append((op, key, value))
    return {
        "rows": state,
        "committed": committed,
        "pending": sorted(pending),
        "safe_to_cut_over": not pending,
    }


def au04():
    # Bounded capacity planning; prices are integer daily units.
    choices = [
        (8 * a + 13 * b + 21 * c, a + b + c, [a, b, c])
        for a in range(9)
        for b in range(7)
        for c in range(5)
        if 3 * a + 5 * b + 9 * c >= 32
        and 2 * a + 4 * b + 5 * c >= 21
        and a + b + c <= 8
        and b + c >= 2
    ]
    cost, count, allocation = min(choices)
    return {
        "cost": cost,
        "instances": count,
        "allocation": allocation,
        "cpu": 3 * allocation[0] + 5 * allocation[1] + 9 * allocation[2],
        "ram": 2 * allocation[0] + 4 * allocation[1] + 5 * allocation[2],
    }


def au05():
    # A reviewed recovery plan: retries reuse the committed payment's key;
    # compensation is derived from actual effects, not requested effects.
    return {
        "calls": [
            {"tool": "cancel_reservation", "args": {"reservation_id": "r-81"}},
            {
                "tool": "refund",
                "args": {"charge_id": "ch-19", "amount": 7200, "key": "refund:o-31"},
            },
            {"tool": "mark_order", "args": {"order_id": "o-31", "state": "cancelled"}},
        ],
        "net_charged": 0,
    }


def au06():
    # Single worker with delayed retries, selecting by (ready_at, job ID).
    queue = [(0, "A", 1), (0, "B", 1), (2, "C", 1)]
    failures = {"A": 2, "B": 3, "C": 0}
    time, trace, dead = 0, [], []
    while queue:
        ready, job, attempt = min(queue)
        queue.remove((ready, job, attempt))
        time = max(time, ready)
        trace.append([time, job, attempt])
        time += 1
        if attempt <= failures[job]:
            if attempt == 3:
                dead.append(job)
            else:
                queue.append((time + 2**attempt, job, attempt + 1))
    return {"starts": trace, "dead_letters": dead, "finished_at": time}


def au07():
    snapshots = {
        "T1": {"a": 1, "b": 1, "c": 1},
        "T2": {"a": 1, "b": 1, "c": 1},
        "T3": {"a": 1, "b": 1, "c": 1},
    }
    writes = {txn: key for txn, key in zip(snapshots, "abc") if sum(snapshots[txn].values()) >= 2}
    state = {"a": 1, "b": 1, "c": 1}
    for key in writes.values():
        state[key] = 0
    return {
        "snapshot_commits": list(writes),
        "snapshot_final": state,
        "serial_final": {"a": 0, "b": 0, "c": 1},
        "violates_at_least_one": sum(state.values()) == 0,
    }


def au08():
    # Intersect replicated version maps; no read repair in this exercise.
    nodes = {
        "A": {"x": (4, "red"), "y": (7, "up")},
        "B": {"x": (5, "blue"), "y": (6, "down")},
        "C": {"x": (5, "green"), "y": (7, "up")},
        "D": {"x": (3, "red"), "y": (8, None)},
    }
    out = []
    for key, responders in [("x", "AB"), ("x", "BC"), ("y", "AC"), ("y", "BD")]:
        latest = max(nodes[n][key][0] for n in responders)
        values = {nodes[n][key][1] for n in responders if nodes[n][key][0] == latest}
        out.append(
            {"status": "conflict", "version": latest}
            if len(values) > 1
            else {
                "status": "deleted" if None in values else "ok",
                "version": latest,
                "value": next(iter(values)),
            }
        )
    return {"reads": out}


def tu01():
    return [
        {"tool": "list_objects", "args": {"cursor": "p2", "snapshot": "s7"}},
        {"tool": "fetch_object", "args": {"id": "b", "version": 4}},
        {"tool": "fetch_object", "args": {"id": "c", "version": 2}},
    ]


def tu02():
    return [
        {"tool": "get_document", "args": {"id": "d4"}},
        {
            "tool": "put_document",
            "args": {
                "id": "d4",
                "if_match": "e9",
                "body": {"owner": "Mira", "labels": ["blue", "red"], "limit": 12},
            },
        },
    ]


def tu03():
    return [
        {"tool": "lookup_payment", "args": {"key": "pay:o7"}},
        {"tool": "record_receipt", "args": {"order": "o7", "payment": "p31", "amount": 4800}},
    ]


def tu04():
    # Reviewed parameterized SQL call: exact values matter, SQL spelling does not
    # enter this task's output since the query is a named server-side operation.
    records = [("u3", 80, 2), ("u1", 100, 1), ("u2", 100, 2), ("u4", 60, 3)]
    selected = sorted((r for r in records if r[2] >= 2), key=lambda r: (-r[1], r[0]))[:2]
    return [
        {
            "tool": "export_rows",
            "args": {
                "ids": [r[0] for r in selected],
                "columns": ["id", "net"],
                "destination": "audit.csv",
                "overwrite": False,
            },
        }
    ]


def tu05():
    return [
        {"tool": "wait_until", "args": {"time": 106}},
        {"tool": "poll_job", "args": {"job": "j8"}},
        {"tool": "fetch_result", "args": {"job": "j8", "token": "rt4"}},
    ]


def tu06():
    return [
        {"tool": "read_file", "args": {"path": "src/a.py", "revision": "h9"}},
        {
            "tool": "apply_patch",
            "args": {"path": "src/a.py", "base": "h9", "old": "limit = 8", "new": "limit = 12"},
        },
        {"tool": "run_tests", "args": {"suite": "limits"}},
    ]


def tu07():
    return [
        {"tool": "delete_object", "args": {"id": "k2", "if_version": 8, "dry_run": True}},
        {"tool": "delete_object", "args": {"id": "k5", "if_version": 3, "dry_run": True}},
    ]


def tu08():
    return [
        {"tool": "resolve_identity", "args": {"external": "e5", "tenant": "t2"}},
        {"tool": "get_entitlements", "args": {"user": "u9", "tenant": "t2"}},
        {
            "tool": "grant_role",
            "args": {"user": "u9", "tenant": "t2", "role": "reader", "if_version": 6},
        },
    ]


def rc01():
    records = [
        ("a", 12, False, False),
        ("b", 9, True, False),
        ("c", 7, True, True),
        ("d", 16, False, True),
        ("e", 11, False, False),
    ]
    included = [(k, n) for k, n, withdrawn, reinstated in records if not withdrawn or reinstated]
    return {
        "included": [k for k, _ in included],
        "total": sum(n for _, n in included),
        "over_ten": sum(n > 10 for _, n in included),
    }


def rc02():
    # Corrections affect earlier service dates, but only after being recorded.
    rows = [(1, 1, 100), (5, 8, 120), (5, 12, 110), (10, 10, 150), (10, 15, 140)]
    values = []
    for day, known in [(6, 7), (6, 9), (6, 13), (11, 11), (11, 14), (11, 16)]:
        candidates = [
            (effective, recorded, value)
            for effective, recorded, value in rows
            if effective <= day and recorded <= known
        ]
        values.append(max(candidates)[2])
    return {"rates": values}


def rc03():
    records = [
        ("a", 40, False, False),
        ("b", 40, True, False),
        ("c", 95, True, False),
        ("d", 120, False, True),
        ("e", 30, False, False),
        ("f", 91, False, False),
    ]
    delete, retain = [], []
    for ident, age, premium, hold in records:
        (delete if not hold and age > (90 if premium else 30) else retain).append(ident)
    return {"delete": delete, "retain": retain}


def rc04():
    stock, reserved, lost = 37, 0, 0
    stock += 18
    reserved += 20
    stock -= 12
    reserved -= 12
    reserved -= 3
    stock += 4  # only inspected, resellable returns
    lost += 6
    stock -= 6
    reserved += 9
    return {
        "physical_sellable": stock,
        "reserved": reserved,
        "available": stock - reserved,
        "damaged_removed": lost,
    }


def rc05():
    a, b = 240, 160
    switch_a, switch_b = a // 4, b // 8
    a, b = a - switch_a + switch_b, b - switch_b + switch_a
    a -= 15
    b -= 25
    return {
        "final_A": a,
        "final_B": b,
        "retained_original_A_in_A": 240 - 60 - 15,
        "A_fraction": fraction(Fraction(a, a + b)),
    }


def rc06():
    # Half-open downtime union; maintenance and outside-window portions excluded.
    incident = [(4, 15), (12, 23), (26, 35), (39, 48)]
    excluded = [(10, 14), (28, 32)]
    bad = {m for a, b in incident for m in range(max(0, a), min(45, b))}
    bad -= {m for a, b in excluded for m in range(a, b)}
    return {
        "counted_downtime": len(bad),
        "meets_budget": len(bad) <= 25,
        "excess_minutes": max(0, len(bad) - 25),
    }


def rc07():
    # Integer cents with explicit half-up rule. Coupon only on merchandise.
    from decimal import Decimal, ROUND_HALF_UP

    def cents(v):
        return int(v.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    subtotal = 3 * 1999 + 2 * 1250
    discount = cents(Decimal(subtotal) * Decimal("0.15"))
    net = subtotal - discount
    shipping = 0 if net >= 7500 else 650
    tax = cents(Decimal(net + shipping) * Decimal("0.0825"))
    return {
        "subtotal": subtotal,
        "discount": discount,
        "shipping": shipping,
        "tax": tax,
        "total": net + shipping + tax,
    }


def rc08():
    # Latest approved revision effective by day 20, ties by revision.
    entries = [
        ("north", 1, 5, 40, True),
        ("south", 1, 5, 30, True),
        ("north", 2, 12, 45, False),
        ("south", 2, 18, 35, True),
        ("north", 3, 21, 50, True),
        ("east", 1, 15, 20, True),
        ("east", 2, 15, 22, True),
        ("south", 3, 19, 99, False),
    ]
    chosen = {}
    for region, rev, day, value, approved in entries:
        if approved and day <= 20 and (region not in chosen or (day, rev) > chosen[region][:2]):
            chosen[region] = (day, rev, value)
    return {
        "quotas": {k: v[2] for k, v in sorted(chosen.items())},
        "revisions": {k: v[1] for k, v in sorted(chosen.items())},
        "total": sum(v[2] for v in chosen.values()),
    }


for prefix, name in [("AU2", "au"), ("TU2", "tu"), ("RC2", "rc")]:
    REASONING.update({f"{prefix}-{i:02}": globals()[f"{name}{i:02}"] for i in range(1, 9)})
