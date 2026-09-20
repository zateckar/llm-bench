"""Executable composition challenges and trusted reference implementations."""

import json
import random

from hard_cases import SOURCES, ledger


CODE = {
    "H5-CG-jsonl-01": """import json
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
def reconcile_jsonl(text, cutoff):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError('duplicate')
            out[key] = value
        return out
    selected, invalid = {}, []
    limit = datetime.fromisoformat(cutoff)
    for i, line in enumerate(text.splitlines()):
        if not line.strip(): continue
        try:
            r = json.loads(line, object_pairs_hook=pairs)
            if not isinstance(r, dict) or set(r) != {'id','version','at','deleted','amount'}: raise ValueError()
            if type(r['id']) is not str or not r['id']: raise ValueError()
            if type(r['version']) is not int or r['version'] < 0 or type(r['deleted']) is not bool: raise ValueError()
            if type(r['amount']) is not str or type(r['at']) is not str: raise ValueError()
            amount = Decimal(r['amount'])
            if not amount.is_finite(): raise ValueError()
            at = datetime.fromisoformat(r['at'])
            if at.utcoffset() is None: raise ValueError()
            cents = int((amount * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        except (ValueError, TypeError, InvalidOperation, OverflowError):
            invalid.append(i)
            continue
        if at > limit: continue
        order = (at, r['version'], i)
        if r['id'] not in selected or order > selected[r['id']][0]:
            selected[r['id']] = (order, r['deleted'], cents)
    live = [{'id':k,'cents':v[2]} for k,v in sorted(selected.items()) if not v[1]]
    return {'live':live,'invalid_lines':invalid,'total_cents':sum(r['cents'] for r in live)}
""",
    "H5-CG-routes-01": """def route_plan(edges, start, end, depart, budget):
    best = None
    def visit(node, now, cost, path):
        nonlocal best
        if node == end:
            candidate = (now, cost, path)
            if best is None or candidate < best: best = candidate
            return
        for src,dst,duration,toll,opens,closes in edges:
            if src != node or dst in path or cost+toll > budget: continue
            leave = max(now, opens)
            if leave >= closes: continue
            visit(dst, leave+duration, cost+toll, path+[dst])
    visit(start, depart, 0, [start])
    return None if best is None else {'arrival':best[0],'cost':best[1],'path':best[2]}
""",
    "H5-CG-waterfill-01": """from fractions import Fraction
def allocate_cents(total, rows):
    weights = {r[0]:r[1] for r in rows}
    caps = {r[0]:r[2] for r in rows}
    quotas = {k:Fraction(0) for k in weights}
    active = {k for k in weights if weights[k] > 0 and caps[k] > 0}
    remaining = Fraction(total)
    while active and remaining:
        denom = sum(weights[k] for k in active)
        proposals = {k:remaining*weights[k]/denom for k in active}
        saturated = {k for k in active if proposals[k] >= caps[k]}
        if not saturated:
            quotas.update(proposals)
            remaining = Fraction(0)
            break
        for k in saturated:
            quotas[k] = Fraction(caps[k])
            remaining -= caps[k]
        active -= saturated
    allocated = {k:int(v) for k,v in quotas.items()}
    extras = int(sum(quotas.values())) - sum(allocated.values())
    order = sorted(quotas, key=lambda k:(-(quotas[k]-allocated[k]),k))
    for k in order[:extras]: allocated[k] += 1
    return {'allocations':[[k,allocated[k]] for k in sorted(allocated)],
            'unallocated':total-sum(allocated.values())}
""",
    "H5-CG-ledger-01": """def replay_transfers(initial, events):
    balances = dict(initial)
    versions = {k:0 for k in initial}
    keys, decisions = {}, []
    for key,src,dst,amount,expected in events:
        identity = (src,dst,amount)
        if key in keys:
            status = 'replay' if keys[key] == identity else 'key_conflict'
        elif versions[src] != expected: status = 'stale'
        elif balances[src] < amount: status = 'insufficient'
        else:
            balances[src] -= amount
            balances[dst] += amount
            versions[src] += 1
            versions[dst] += 1
            keys[key] = identity
            status = 'applied'
        decisions.append(status)
    return {'decisions':decisions,'balances':balances,'versions':versions,'committed_keys':sorted(keys)}
""",
}

PROMPTS = {
    "H5-CG-jsonl-01": """Implement reconcile_jsonl(text, cutoff) in Python. Parse NDJSON in memory.
cutoff is a valid offset-aware datetime.fromisoformat string. Use zero-based physical
line numbers (str.splitlines); ignore blank lines. Invalid nonblank lines go into
invalid_lines. A valid row has EXACT keys id,version,at,deleted,amount: nonempty string
id, nonnegative int version (bool forbidden), bool deleted, offset-aware fromisoformat
string at, and finite Decimal-parseable STRING amount. Duplicate JSON keys are invalid.
Reject missing/extra keys, wrong types, nonfinite amounts or invalid datetimes even
on future rows. Inputs never contain finite amounts with absolute value above 10**14
or more than three decimal places. Ignore valid rows after cutoff. Select one row per id by greatest instant,
then version, then physical line number. Compare instants across offsets. A selected
tombstone suppresses the id; never resurrect older rows. For live rows round amount
*100 to integer cents using ROUND_HALF_UP (away from zero on ties). Return
{'live':[{'id':id,'cents':integer} sorted by id], 'invalid_lines':[ascending indices],
'total_cents':sum of live cents}. Do not use floats or mutate inputs. No file/network I/O.
""",
    "H5-CG-routes-01": """Implement route_plan(edges,start,end,depart,budget) in Python.
Each directed edge is [src,dst,duration,toll,opens,closes], integer duration>0, toll>=0.
A departure is permitted at opens <= time < closes. Waiting at vertices is allowed;
arrival may be after closes (it constrains DEPARTURE only). Start at integer depart;
total toll <= nonnegative budget. Find a path with no repeated vertex, minimizing
arrival time, then toll, then lexicographic vertex-name sequence. Return None if no
feasible path, otherwise {'arrival':integer,'cost':integer,'path':[names]}. start=end
returns zero-cost empty travel with path=[start]. Parallel edges, cycles and vertices
with no outgoing edges are possible; at most eight vertices and 22 edges. Do not
collapse all states at a vertex to the earliest arrival: money and visited set matter.
No external I/O or input mutation. Return the function, not example output.
""",
    "H5-CG-waterfill-01": """Implement allocate_cents(total, rows) in Python. total>=0 integer;
rows are [unique_string_id,nonnegative_integer_weight,nonnegative_integer_cap].
Allocate a pool by capped proportional water filling, then largest remainders:
ignore zero-weight/zero-cap rows (allocation 0). Repeatedly distribute remaining pool
proportionally to weights of active rows. If any tentative share >= its cap, fix ALL
such rows to their cap, remove them, subtract their caps and recompute for the rest.
Otherwise the tentative shares are final fractional quotas. If all rows cap out,
the excess pool remains unallocated. Floor final quotas, distribute remaining integral
cents by descending fractional remainder, ties by id ascending. Caps apply to final
allocation; fractional computations must be exact, including totals above 2**53.
Return {'allocations':[[id,integer_cents] sorted by id], 'unallocated':integer} with
every input id, including zeros. Empty rows return the full pool unallocated. Do not
mutate rows. Python standard-library Fraction is available. No external I/O.
""",
    "H5-CG-ledger-01": """Implement replay_transfers(initial, events) in Python. initial maps
account names to nonnegative integer balances, versions start at 0. Each event is
[key,source,destination,positive_integer_amount,expected_source_version]; accounts
exist and source!=destination. Apply in order: committed key with same
(source,destination,amount) -> replay, with changed payload -> key_conflict. The
expected version is not part of business identity. Otherwise version mismatch ->
stale; otherwise insufficient funds -> insufficient; otherwise atomically transfer,
increment BOTH versions, commit the key -> applied. Rejections never commit keys.
Return {'decisions':[statuses in event order], 'balances':mapping,'versions':mapping,
'committed_keys':[sorted keys]}. Handle empty input and arbitrarily large integers;
do not mutate either input. Keys may be retried after failure or successful commit.
""",
}


def code_cases():
    rng = random.Random(89123)
    fixtures = {id: [] for id in CODE}
    cutoff = "2026-01-02T00:00:00+00:00"

    def row(id="a", version=1, at="2026-01-01T12:00:00+00:00", deleted=False, amount="1.005"):
        return dict(id=id, version=version, at=at, deleted=deleted, amount=amount)

    samples = [
        "",
        "\n",
        "not-json",
        '{"id":"a","id":"b"}',
        "[]",
        json.dumps(row()),
        json.dumps(row(amount="-1.005")),
        json.dumps(row(version=True)),
        json.dumps(row(amount="NaN")),
        json.dumps(row(at="2026-01-01")),
        json.dumps({**row(), "extra": 1}),
        json.dumps(row()) + "\n" + json.dumps(row(version=2, deleted=True)),
        json.dumps(row(at="2026-01-02T01:00:00+02:00", amount="3.33"))
        + "\n"
        + json.dumps(row(at="2026-01-01T23:30:00+00:00", amount="4.44")),
    ]
    for _ in range(35):
        lines = [
            json.dumps(
                row(
                    id=rng.choice("abcd"),
                    version=rng.randrange(5),
                    at=f"2026-01-0{rng.randrange(1, 4)}T12:00:00+00:00",
                    deleted=rng.random() < 0.25,
                    amount=f"{rng.randrange(-50, 51)}.005",
                )
            )
            for _ in range(18)
        ]
        lines.insert(rng.randrange(len(lines)), "")
        lines.insert(rng.randrange(len(lines)), "{broken")
        samples.append("\n".join(lines))
    fixtures["H5-CG-jsonl-01"] = [[text, cutoff] for text in samples]
    fixtures["H5-CG-routes-01"] = [
        [[], "a", "a", 3, 0],
        [[], "a", "b", 0, 3],
        [[["a", "b", 1, 5, 0, 9], ["a", "b", 2, 1, 0, 9], ["b", "c", 1, 2, 0, 9]], "a", "c", 0, 3],
        [[["a", "b", 10, 0, 0, 1]], "a", "b", 0, 0],
        [[["a", "b", 1, 0, 0, 1]], "a", "b", 1, 0],
    ]
    for _ in range(50):
        edges = []
        for _ in range(rng.randrange(8, 23)):
            a, b = rng.sample(list("abcdefg"), 2)
            opens = rng.randrange(0, 12)
            edges.append(
                [
                    a,
                    b,
                    rng.randrange(1, 7),
                    rng.randrange(0, 5),
                    opens,
                    opens + rng.randrange(1, 12),
                ]
            )
        fixtures["H5-CG-routes-01"].append(
            [edges, "a", "g", rng.randrange(0, 5), rng.randrange(0, 14)]
        )
    fixtures["H5-CG-waterfill-01"] = [
        [10, []],
        [5, [["a", 0, 100]]],
        [10, [["a", 1, 2], ["b", 1, 100]]],
        [5, [["b", 1, 10], ["a", 1, 10]]],
        [2**54 + 7, [["x", 3, 2**55], ["y", 2, 2**55], ["z", 1, 2**55]]],
    ]
    for _ in range(55):
        fixtures["H5-CG-waterfill-01"].append(
            [
                rng.randrange(0, 151),
                [
                    [chr(97 + i), rng.randrange(0, 7), rng.randrange(0, 50)]
                    for i in range(rng.randrange(1, 10))
                ],
            ]
        )
    fixtures["H5-CG-ledger-01"] = [
        [{}, []],
        [{"a": 2**55, "b": 0}, [["x", "a", "b", 2**54, 0], ["x", "a", "b", 2**54, 999]]],
    ]
    for _ in range(50):
        initial = {k: rng.randrange(0, 100) for k in "abcd"}
        events = []
        for i in range(25):
            if events and rng.random() < 0.3:
                event = list(rng.choice(events))
                if rng.random() < 0.5:
                    event[3] += 1
            else:
                a, b = rng.sample(list(initial), 2)
                state = ledger(events, initial)
                event = [
                    str(i),
                    a,
                    b,
                    rng.randrange(1, 120),
                    max(0, state["versions"][a] - int(rng.random() < 0.3)),
                ]
            events.append(event)
        fixtures["H5-CG-ledger-01"].append([initial, events])
    out = []
    for id, code in CODE.items():
        scope = {}
        exec(code, scope)
        name = {
            "H5-CG-jsonl-01": "reconcile_jsonl",
            "H5-CG-routes-01": "route_plan",
            "H5-CG-waterfill-01": "allocate_cents",
            "H5-CG-ledger-01": "replay_transfers",
        }[id]
        expected = [
            dict(function=name, args=args, expected=scope[name](*args), relative=0, tolerance=0)
            for args in fixtures[id]
        ]
        out.append(
            dict(
                id=id,
                category="Code Generation",
                difficulty="expert",
                max_tokens=16384,
                source="Original ceiling-v5; " + SOURCES["code"],
                prompt=PROMPTS[id],
                evaluator="code_exec",
                expected=expected,
            )
        )
    return out
