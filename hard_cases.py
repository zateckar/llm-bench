"""Ceiling-v5 original tasks inspired by benchmark designs, not copied items.

Regenerate tests/ceiling.yaml. Each family uses a fixed local seed; variants belong
to ONE reporting cluster, not four independent measures of capability.
"""

from collections import Counter, defaultdict
from fractions import Fraction
import itertools
import json
from pathlib import Path
import random

import yaml

SOURCES = {
    "reasoning": "BBEH: https://github.com/google-deepmind/bbeh (compositional reasoning design only)",
    "evidence": "LiveBench: https://github.com/LiveBench/LiveBench ; NoLiMa: https://github.com/adobe-research/NoLiMa (objective synthesis and indirect retrieval design only)",
    "instructions": "IFEval: https://huggingface.co/datasets/google/IFEval (jointly verifiable constraints design only)",
    "code": "BigCodeBench: https://github.com/bigcode-project/bigcodebench (composed library use and executable tests design only)",
}


def case(id, category, prompt, answer, inspiration="reasoning"):
    return dict(
        id=id,
        category=category,
        difficulty="expert",
        max_tokens=16384,
        source="Original ceiling-v5; " + SOURCES[inspiration],
        description="Compositional challenge; deterministic oracle in hard_cases.py. Difficulty awaits fresh model calibration.",
        prompt=prompt.strip()
        + "\nReturn only the requested JSON, with no extra keys or prose. Null is allowed only where explicitly requested. All ID lists must be sorted lexicographically.",
        evaluator="json_match",
        expected={"value": answer, "mode": "exact", "strict_json": True},
    )


def ledger(events, initial):
    balances = dict(initial)
    versions = {k: 0 for k in initial}
    applied, decisions = {}, []
    for e in events:
        key, src, dst, amount, expected_version = e
        payload = (src, dst, amount)  # optimistic version is not business identity
        if key in applied:
            decisions.append("replay" if applied[key] == payload else "key_conflict")
        elif versions[src] != expected_version:
            decisions.append("stale")
        elif balances[src] < amount:
            decisions.append("insufficient")
        else:
            balances[src] -= amount
            balances[dst] += amount
            versions[src] += 1
            versions[dst] += 1
            applied[key] = payload
            decisions.append("applied")
    return {
        "decisions": decisions,
        "balances": balances,
        "versions": versions,
        "committed_keys": sorted(applied),
    }


def ledger_case(variant, category):
    rng = random.Random(2100 + variant)
    initial = dict(zip("ABCD", [rng.randrange(35, 70) for _ in range(4)]))
    events = []
    for i in range(24):
        src, dst = rng.sample(list(initial), 2)
        state = ledger(events, initial)
        if i in (3, 6, 10, 15, 21):
            event = list(events[rng.randrange(len(events))])
            if i in (6, 15):
                event[3] += 1  # key reuse with a different business payload
            if i == 21:
                event[4] += 99  # replay ignores optimistic version
        else:
            event = [
                f"k{i:02}",
                src,
                dst,
                rng.randrange(8, 61),
                max(0, state["versions"][src] - int(i % 5 == 0)),
            ]
        events.append(event)
    prefix = "AU" if category == "Agentic Use Cases" else "TU"
    prompt = f"""Audit this transactional tool-service event log. Each call commits atomically.
Initial balances: {json.dumps(initial)}; every account version starts at zero.
Each row is [idempotency_key,source,destination,positive_integer_amount,expected_source_version].
Apply these checks IN ORDER: (1) if key is already committed, return replay for the
same (source,destination,amount), otherwise key_conflict; expected version is NOT
part of this identity. (2) reject stale source version as stale. (3) reject insufficient
source funds as insufficient. (4) atomically transfer, increment BOTH account versions,
record the key, and return applied. Rejections do not record keys or change state.
Rows are the complete serialized execution order, not concurrent starts. An earlier
failure therefore does not prevent later reuse of its key. No fee or overdraft.
Events: {json.dumps(events)}
Return {{"decisions":[one status per input row],"balances":{{account:integer}},
"versions":{{account:integer}},"committed_keys":[IDs]}}.
"""
    return case(f"H5-{prefix}-ledger-{variant:02}", category, prompt, ledger(events, initial))


def worlds_for(clauses):
    # Clause is a disjunction; positive i means Xi true, negative i means Xi false.
    return [
        bits
        for bits in itertools.product((False, True), repeat=10)
        if all(any(bits[abs(lit) - 1] == (lit > 0) for lit in clause) for clause in clauses)
    ]


def evidence_case(variant, category):
    rng = random.Random(3100 + variant)
    planted = [rng.choice((False, True)) for _ in range(10)]
    clauses = []
    for _ in range(23):
        clause = [i * rng.choice((-1, 1)) for i in rng.sample(range(1, 11), rng.choice((2, 3)))]
        if not any(planted[abs(i) - 1] == (i > 0) for i in clause):
            clause[0] *= -1
        clauses.append(clause)
    worlds = worlds_for(clauses)
    claims = [[i] for i in range(1, 11)] + [[1, -4], [-2, 8], [3, 9], [-6, -10]]
    supporting = [
        sum(all(w[abs(i) - 1] == (i > 0) for i in claim) for w in worlds) for claim in claims
    ]
    answer = {
        "world_count": len(worlds),
        "support_counts": supporting,
        "labels": [
            "SUPPORTED" if n == len(worlds) else "CONTRADICTED" if n == 0 else "UNKNOWN"
            for n in supporting
        ],
    }
    prefix = "CL" if category == "Classification" else "TH"
    prompt = f"""Evaluate claims against a closed fictional evidence dossier. There are ten
Boolean propositions X1..X10. Every satisfying assignment is a possible world;
absence of a proposition does not make it false. The dossier supplies clauses in
signed-integer notation: [2,-5,7] means X2 OR NOT X5 OR X7, inclusive OR.
ALL clauses must hold. Do not read a disjunction as exclusive or an implication
as a biconditional. The dossier is consistent.
Clauses: {json.dumps(clauses)}
Claims are CONJUNCTIONS of their signed literals, unlike the dossier clauses:
{json.dumps(claims)}
Return {{"world_count":integer,"support_counts":[number of possible worlds satisfying
each claim],"labels":[SUPPORTED if all worlds satisfy it, CONTRADICTED if none,
UNKNOWN otherwise]}}. Labels are JSON strings, including "UNKNOWN"; never null.
"""
    return case(f"H5-{prefix}-worlds-{variant:02}", category, prompt, answer)


def revision_data(variant, long=False):
    rng = random.Random(4100 + variant)
    count = 80 if long else 14
    rows = []
    aliases, routes = {}, {}
    for i in range(count):
        entity, alias, route = f"e{i:03}", f"alias{(i * 17 + 9) % 997:03}", f"route{i:03}"
        aliases[alias] = entity
        routes[route] = alias
        for rev in range(1, 5):
            rows.append(
                dict(
                    record=f"r{i:03}-{rev}",
                    entity=entity,
                    revision=rev,
                    effective=rng.randrange(1, 36),
                    approved=(rev == 1 or rng.random() > 0.25),
                    deleted=(rev > 1 and rng.random() < 0.2),
                    quota=rng.randrange(10, 151),
                    owner=rng.choice(["north", "south", "east"]),
                )
            )
    rng.shuffle(rows)
    queries = sorted(rng.sample(list(routes), 7))
    return rows, aliases, routes, queries


def resolve_revisions(rows, aliases, routes, queries, cutoff=23):
    chosen = {}
    for r in rows:
        if r["approved"] and r["effective"] <= cutoff:
            old = chosen.get(r["entity"])
            if old is None or (r["effective"], r["revision"]) > (old["effective"], old["revision"]):
                chosen[r["entity"]] = r
    result, totals = {}, defaultdict(int)
    for route in queries:
        r = chosen.get(aliases[routes[route]])
        result[route] = {
            "record": r["record"] if r else None,
            "state": "unknown" if r is None else "deleted" if r["deleted"] else "active",
            "quota": r["quota"] if r and not r["deleted"] else None,
        }
        if r and not r["deleted"]:
            totals[r["owner"]] += r["quota"]
    return {
        "routes": result,
        "owner_totals": {k: totals[k] for k in ("east", "north", "south")},
        "grand_total": sum(totals.values()),
    }


def revision_case(variant, category):
    long = category in {"Long Context Coherence", "Needle Retrieval"}
    rows, aliases, routes, queries = revision_data(variant, long)
    prefix = {
        "Long Context Coherence": "LC",
        "Needle Retrieval": "NR",
        "Reading Comprehension": "RC",
        "Summarization": "SU",
    }[category]
    prompt = f"""Prepare a source-grounded operational summary from this fictional dossier.
Requested route handles are not entity IDs: join route->alias->entity before resolving
its record. At cutoff day 23, take the APPROVED revision with greatest (effective day,
revision number), among effective days <=23. File order is irrelevant. Unapproved and
future revisions do not supersede. If the chosen revision is deleted, do NOT resurrect
an earlier live one. No eligible record means unknown, not deleted or zero quota.
Sum quotas only for ACTIVE requested routes, grouped by that chosen record's owner.
Return {{"routes":{{requested_route:{{"record":record ID or null,"state":"active"|
"deleted"|"unknown","quota":integer for active else null}}}},
"owner_totals":{{"east":integer,"north":integer,"south":integer}},"grand_total":integer}}.
Include every requested route and no others. Zero totals must still appear.
Route directory: {json.dumps(routes, separators=(",", ":"))}
Alias directory: {json.dumps(aliases, separators=(",", ":"))}
Records, one JSON object per line:
{chr(10).join(json.dumps(r, separators=(",", ":")) for r in rows)}
Requested routes: {json.dumps(queries)}
"""
    return case(
        f"H5-{prefix}-revision-{variant:02}",
        category,
        prompt,
        resolve_revisions(rows, aliases, routes, queries),
        "evidence",
    )


def transform_records(records):
    selected, rejected = {}, []
    for index, r in enumerate(records):
        if r["state"] != "approved" or r["deleted"]:
            rejected.append(index)
            continue
        key = r["key"].strip().casefold()
        if key not in selected or (r["version"], index) > selected[key][:2]:
            selected[key] = (r["version"], index, r["amount"])
    ordered = sorted(selected.items(), key=lambda pair: (-pair[1][2], pair[0]))
    return {
        "rows": [
            {"key": k, "source": v[1], "cents": v[2], "rank": i + 1}
            for i, (k, v) in enumerate(ordered[:5])
        ],
        "rejected_indices": rejected,
        "eligible_distinct": len(selected),
        "omitted_total": sum(v[2] for _, v in ordered[5:]),
    }


def instruction_case(variant):
    rng = random.Random(5100 + variant)
    records = [
        dict(
            key=rng.choice([" A ", "a", "B", " b ", "C", "d", "E", "f", "G", "h"]),
            version=rng.randrange(1, 5),
            state=rng.choice(["approved"] * 3 + ["draft"]),
            deleted=rng.random() < 0.15,
            amount=rng.randrange(-5, 8) * 125,
        )
        for _ in range(30)
    ]
    prompt = f"""Transform this ledger with ALL rules, in this order.
1. Reject rows not approved OR marked deleted; record only those raw row indices as rejected.
2. Normalize key by stripping outside whitespace and case-folding.
3. Among eligible rows per normalized key keep greatest version; tied versions use
the later input row. A rejected row never supersedes an eligible row. Do not sum duplicates.
4. Sort survivors by amount DESCENDING then normalized key ascending. Keep the first
five (or all if fewer). Negative amounts and zero remain eligible.
5. Return {{"rows":[{{"key":normalized key,"source":zero-based raw row index,
"cents":unchanged integer amount,"rank":one-based final rank}}],
"rejected_indices":[ascending raw indices],"eligible_distinct":count BEFORE top-five,
"omitted_total":sum amounts of surviving keys omitted by top-five}}.
Never list superseded eligible rows in rejected_indices. Preserve final row order
from rule 4; only rejected_indices is numerically sorted.
Input: {json.dumps(records)}
"""
    return case(
        f"H5-IF-pipeline-{variant:02}",
        "Instruction Following",
        prompt,
        transform_records(records),
        "instructions",
    )


def permutation_case(variant):
    rng = random.Random(6100 + variant)
    differences = [rng.choice([-8, -5, -3, -1, 1, 2, 4, 7, 9]) for _ in range(8)]
    weights = [rng.randrange(1, 4) for _ in differences]
    observed = abs(sum(d * w for d, w in zip(differences, weights)))
    extreme = sum(
        abs(sum(s * d * w for s, d, w in zip(signs, differences, weights))) >= observed
        for signs in itertools.product((-1, 1), repeat=8)
    )
    p = Fraction(extreme, 256)
    prompt = f"""A paired randomized experiment has eight independently randomized blocks.
The observed treatment-minus-control differences are {differences}; PRE-SPECIFIED
integer weights are {weights}. Test statistic is the absolute weighted SUM, not a
mean of signs. Under the sharp null, independently swap treatment/control labels in
each block: all 256 sign-flip vectors are equally likely, even when vectors yield
the same statistic. Exact two-sided p is proportion with statistic >= observed.
Count ties as extreme; do not add an extra pseudo-observation. Each block contains
100 repeated measurements, which do not increase the number of randomized blocks.
Return {{"independent_units":integer,"observed_statistic":integer,"extreme_vectors":integer,
"p":{{"numerator":integer,"denominator":integer}},"reject_at_0_05":bool,
"causal_effect_in_unstudied_population_established":bool}}. Reduce p. No transportability
assumptions or data about any other population are supplied.
"""
    return case(
        f"H5-TN-randomization-{variant:02}",
        "Terminal Science",
        prompt,
        {
            "independent_units": 8,
            "observed_statistic": observed,
            "extreme_vectors": extreme,
            "p": {"numerator": p.numerator, "denominator": p.denominator},
            "reject_at_0_05": p <= Fraction(1, 20),
            "causal_effect_in_unstudied_population_established": False,
        },
    )


def allocation_case(variant):
    rng = random.Random(7100 + variant)
    people = [
        dict(
            id=chr(65 + i),
            group="rural" if i % 3 == 0 else "urban",
            cost=rng.randrange(2, 6),
            benefit=rng.randrange(3, 13),
            consent=(i != 6),
        )
        for i in range(9)
    ]
    cap = 14 + variant
    eligible = [p for p in people if p["consent"]]
    options = []
    for bits in itertools.product((0, 1), repeat=len(eligible)):
        chosen = [p for p, b in zip(eligible, bits) if b]
        cost = sum(p["cost"] for p in chosen)
        if cost > cap or sum(p["group"] == "rural" for p in chosen) < 2:
            continue
        options.append((sum(p["benefit"] for p in chosen), -cost, [p["id"] for p in chosen]))
    best = max((benefit, cost) for benefit, cost, _ in options)
    ties = sorted(ids for benefit, cost, ids in options if (benefit, cost) == best)
    prompt = f"""Apply this fictional agreed allocation policy; do not substitute an unstated
ethical theory. Each person is indivisible. Require consent; total cost <= {cap};
at least TWO rural recipients. Among feasible allocations maximize total benefit,
then minimize total cost; return ALL tied allocations. Do not maximize number of
people ahead of benefit. Group membership has no other role. All facts:
{json.dumps(people)}
Return {{"benefit":integer,"cost":integer,"allocations":[sorted recipient-ID lists,
outer list lexicographically sorted],"nonconsenting_eligible":bool,
"universally_ethically_optimal_proved":bool}}. The last question asks whether optimizing
this specified policy proves optimality under every possible ethical framework.
"""
    return case(
        f"H5-ER-allocation-{variant:02}",
        "Ethical Reasoning",
        prompt,
        {
            "benefit": best[0],
            "cost": -best[1],
            "allocations": ties,
            "nonconsenting_eligible": False,
            "universally_ethically_optimal_proved": False,
        },
    )


def science_case(variant):
    # Grounded mechanism reasoning with explicit model assumptions and exact fractions.
    n = 128 * (variant + 1)
    capture = Fraction(variant, variant + 2)
    # Sequential decay: each discrete step A->B at 1/2, B->C at 1/4 of PRE-step B.
    a, b, c = Fraction(n), Fraction(0), Fraction(0)
    for _ in range(5):
        da, db = a / 2, b / 4
        a, b, c = a - da, b + da - db, c + db

    def fraction(x):
        return {"numerator": x.numerator, "denominator": x.denominator}

    prompt = f"""Reference-grounded scientific knowledge: a simplified discrete decay chain
A->B->C conserves nuclei; C is stable. In each step HALF the A present at START
becomes B, and ONE QUARTER of the B present at START becomes C. Newly created B
cannot decay again in the same step. Initial (A,B,C)=({n},0,0). These are expected
populations, so fractional counts are valid. Run five steps. A detector then records
each C nucleus with efficiency {capture.numerator}/{capture.denominator}; it records
neither A nor B. Counts lost to inefficiency do not imply physical nuclei vanished.
An analyst substitutes sequential in-place updates allowing newborn B to decay in
the same step. Decide whether that matches the stipulated model. Do not assume
that an observed detector count identifies all three populations without the model.
Return {{"A":fraction,"B":fraction,"C":fraction,"detected_C":fraction,
"conserved_total":integer,"in_place_update_valid":bool,
"one_count_alone_identifies_three_populations":bool}}. Each fraction has exactly
"numerator" and "denominator", reduced with positive denominator, including integers.
"""
    return case(
        f"H5-FK-decay-{variant:02}",
        "Factual Knowledge",
        prompt,
        {
            "A": fraction(a),
            "B": fraction(b),
            "C": fraction(c),
            "detected_C": fraction(c * capture),
            "conserved_total": n,
            "in_place_update_valid": False,
            "one_count_alone_identifies_three_populations": False,
        },
    )


def build_cases():
    result = []
    for variant in range(1, 5):
        for category in ("Agentic Use Cases", "Tool Using"):
            result.append(ledger_case(variant + (4 if category == "Tool Using" else 0), category))
        for category in ("Classification", "Truthfulness"):
            result.append(
                evidence_case(variant + (4 if category == "Truthfulness" else 0), category)
            )
        for offset, category in enumerate(
            ("Reading Comprehension", "Summarization", "Long Context Coherence", "Needle Retrieval")
        ):
            result.append(revision_case(variant + 4 * offset, category))
        result.append(instruction_case(variant))
        result.append(permutation_case(variant))
        result.append(allocation_case(variant))
        result.append(science_case(variant))
    from hard_code_cases import code_cases

    result.extend(code_cases())
    return result


if __name__ == "__main__":
    cases = build_cases()
    target = Path(__file__).parent / "tests" / "ceiling.yaml"
    target.write_text(
        yaml.safe_dump(cases, allow_unicode=True, sort_keys=False, width=100), encoding="utf-8"
    )
    print(f"Wrote {len(cases)} questions: {dict(Counter(c['category'] for c in cases))}")
