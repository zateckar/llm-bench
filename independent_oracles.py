"""Alternative algorithms for differential checking of the code answer keys.

Keep these independent of challenge_oracles: changes to one must not silently
change the other. Inputs are small for exponential oracles; larger cases use
the polynomial algorithms below.
"""

import bisect
import heapq
from functools import lru_cache
from collections import defaultdict, deque


def temporal_route(n, edges, start, end, departure):
    graph = defaultdict(list)
    for a, b, duration, opening, closing in edges:
        graph[a].append((b, duration, opening, closing))
    best = {start: departure}
    queue = [(departure, start)]
    while queue:
        t, node = heapq.heappop(queue)
        if t != best[node]:
            continue
        if node == end:
            return t
        for dest, duration, opening, closing in graph[node]:
            if t >= closing:
                continue
            arrival = max(t, opening) + duration
            if arrival < best.get(dest, float("inf")):
                best[dest] = arrival
                heapq.heappush(queue, (arrival, dest))
    return None


def interval_overlay(intervals):
    delta = defaultdict(int)
    for start, end in intervals:
        delta[start] += 1
        delta[end] -= 1
    points = sorted(k for k, v in delta.items() if v)
    count, result = 0, []
    for i, start in enumerate(points[:-1]):
        count += delta[start]
        if count:
            result.append([start, points[i + 1], count])
    return result


def dependency_batches(nodes, edges):
    incoming = dict.fromkeys(nodes, 0)
    outgoing = defaultdict(set)
    for a, b in edges:
        if b not in outgoing[a]:
            incoming[b] += 1
            outgoing[a].add(b)
    ready = sorted(x for x, degree in incoming.items() if degree == 0)
    result, removed = [], set()
    while ready:
        result.append(ready)
        nxt = []
        for node in ready:
            removed.add(node)
            for dest in outgoing[node]:
                incoming[dest] -= 1
                if incoming[dest] == 0:
                    nxt.append(dest)
        ready = sorted(nxt)
    return {"batches": result, "blocked": sorted(set(nodes) - removed)}


def versioned_read(events, queries):
    values = {}
    for key, time, value in events:
        values.setdefault(key, {})[time] = value
    times = {k: sorted(v) for k, v in values.items()}
    result = []
    for key, time in queries:
        index = bisect.bisect_right(times.get(key, []), time) - 1
        result.append(None if index < 0 else values[key][times[key][index]])
    return result


def shortest_subarray(nums, target):
    prefix, queue, best = 0, deque([(0, 0)]), None
    for end, number in enumerate(nums, 1):
        prefix += number
        while queue and prefix - queue[0][1] >= target:
            start, _ = queue.popleft()
            candidate = (end - start, start, end)
            if best is None or candidate < best:
                best = candidate
        while queue and queue[-1][1] >= prefix:
            queue.pop()
        queue.append((end, prefix))
    return list(best[1:]) if best else [-1, -1]


def optimal_jobs(jobs):
    best_profit, best_ids = 0, []

    def visit(index, chosen, profit):
        nonlocal best_profit, best_ids
        ids = sorted(j[0] for j in chosen)
        if profit > best_profit or (profit == best_profit and ids < best_ids):
            best_profit, best_ids = profit, ids
        for i in range(index, len(jobs)):
            j = jobs[i]
            if all(j[2] <= k[1] or k[2] <= j[1] for k in chosen):
                visit(i + 1, chosen + [j], profit + j[3])

    visit(0, [], 0)
    return {"profit": best_profit, "ids": best_ids}


def reconcile(events):
    first = {}
    for event in events:
        first.setdefault(event[0], event)
    ledger, rejected = defaultdict(list), []
    for ident, source, dest, amount in first.values():
        valid = (
            amount > 0
            and source != dest
            and (source == "BANK" or sum(ledger.get(source, [])) >= amount)
        )
        if not valid:
            rejected.append(ident)
            continue
        if source != "BANK":
            ledger[source].append(-amount)
        if dest != "BANK":
            ledger[dest].append(amount)
    return {"balances": {k: sum(v) for k, v in sorted(ledger.items())}, "rejected": rejected}


def glob_match(pattern, text):
    @lru_cache(None)
    def match(i, j):
        if i == len(pattern):
            return j == len(text)
        char = pattern[i]
        if char == "\\":
            return (
                i + 1 < len(pattern)
                and j < len(text)
                and pattern[i + 1] == text[j]
                and match(i + 2, j + 1)
            )
        if char == "*":
            return match(i + 1, j) or (j < len(text) and match(i, j + 1))
        return j < len(text) and (char == "?" or char == text[j]) and match(i + 1, j + 1)

    return match(0, 0)


CODE = {
    f"AC2-{i:02}": fn
    for i, fn in enumerate(
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
        1,
    )
}


def code_cases(rng, count=24):
    cases = {ident: [] for ident in CODE}
    for _ in range(count):
        n = rng.randint(2, 9)
        edges = []
        for _ in range(rng.randint(0, n * 3)):
            opening = rng.randint(0, 12)
            edges.append(
                [
                    rng.randrange(n),
                    rng.randrange(n),
                    rng.randint(1, 5),
                    opening,
                    opening + rng.randint(1, 8),
                ]
            )
        cases["AC2-01"].append([n, edges, rng.randrange(n), rng.randrange(n), rng.randint(0, 15)])
        intervals = []
        for _ in range(rng.randint(0, 15)):
            start = rng.randint(-8, 8)
            intervals.append([start, start + rng.randint(0, 10)])
        cases["AC2-02"].append([intervals])
        nodes = list("abcdefgh"[:n])
        links = [[rng.choice(nodes), rng.choice(nodes)] for _ in range(rng.randint(0, n * 2))]
        cases["AC2-03"].append([nodes, links])
        events = [
            [rng.choice("abc"), rng.randint(-5, 8), rng.choice([None, 0, False, "", "v"])]
            for _ in range(15)
        ]
        queries = [[rng.choice("abcd"), rng.randint(-6, 10)] for _ in range(12)]
        cases["AC2-04"].append([events, queries])
        cases["AC2-05"].append(
            [[rng.randint(-9, 9) for _ in range(rng.randint(0, 30))], rng.randint(-5, 30)]
        )
        jobs = []
        for i in range(rng.randint(0, 10)):
            start = rng.randint(-3, 8)
            jobs.append([str(i), start, start + rng.randint(1, 5), rng.randint(-4, 12)])
        cases["AC2-06"].append([jobs])
        events = [
            [
                str(rng.randrange(15)),
                rng.choice(["BANK", "a", "b", "c"]),
                rng.choice(["BANK", "a", "b", "c"]),
                rng.randint(-2, 10),
            ]
            for _ in range(30)
        ]
        cases["AC2-07"].append([events])
        cases["AC2-08"].append(
            [
                "".join(rng.choices("ab*?\\", k=rng.randrange(10))),
                "".join(rng.choices("ab*?\n", k=rng.randrange(10))),
            ]
        )
    # Larger inputs expose algorithms that only work on the tiny examples.
    cases["AC2-01"].append([100, [[i, i + 1, 1, i, i + 2] for i in range(99)][::-1], 0, 99, 0])
    cases["AC2-02"].append([[[i, i + 10] for i in range(2000)]])
    cases["AC2-05"].append([[1, -1] * 999 + [2, 3], 5])
    cases["AC2-06"].append([[[str(i), i, i + 1, i % 3] for i in range(16)]])
    cases["AC2-08"].append(["*a" * 30 + "b", "a" * 70 + "c"])
    return cases
