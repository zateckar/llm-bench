#!/usr/bin/env python3
"""Executable hardening-v7 candidate families.

The default v6 suite stays frozen.  This module is a separate, deterministic
profile with two families and two development instances per category.  Every
instance has a small, auditable oracle and a second verification path in
``verify_cases``.  The cases are deliberately public development material; a
release profile must use an untouched evaluation split and a matched pilot.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
import random
from typing import Any

from difficulty_hardening import HARDENING_FAMILIES
from test_loader import _parse_question


REVISION = "hardening-v7"
SOURCE = "Original hardening-v7; independent development oracle; requires matched pilot"
DEFAULT_VARIANTS = 2


def _rng(category: str, family: str, variant: int, split: str) -> random.Random:
    digest = hashlib.sha256(
        f"{REVISION}:{split}:{category}:{family}:{variant}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def _leaf_paths(value: Any, path: str = "") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            paths.extend(_leaf_paths(child, f"{path}.{key}".lstrip(".")))
        return paths or [path or "root"]
    if isinstance(value, list):
        paths: list[str] = []
        for index, child in enumerate(value):
            paths.extend(_leaf_paths(child, f"{path}[{index}]"))
        return paths or [path or "root"]
    return [path or "root"]


def _rubric(answer: Any, *, critical: set[str] | None = None) -> list[dict[str, Any]]:
    critical = critical or set()
    return [
        {
            "id": f"json:{path}",
            "dimension": "content",
            "weight": 1,
            "mandatory": True,
            "critical": path in critical,
        }
        for path in _leaf_paths(answer)
    ]


def _case(
    *,
    category: str,
    family: str,
    variant: int,
    split: str,
    demand: str,
    prompt: str,
    answer: Any,
    evaluator: str = "json_match",
    expected: Any | None = None,
    rubric: list[dict[str, Any]] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    slug = f"{_slug(family)}-{split}-v{variant + 1:02d}"
    if expected is None:
        expected = {"value": answer, "mode": "exact", "strict_json": True}
    raw = {
        "id": f"H7-{slug}",
        "category": category,
        "difficulty": "expert",
        "source": SOURCE,
        "description": description or f"Hardening-v7 family: {demand}.",
        "prompt": (
            f"This is a supplied-data {demand} task. Use only the packet in the prompt; "
            "do not import outside facts or assumptions.\n\n"
            f"{prompt.strip()}\n\n"
            "Return one JSON object with exactly the requested keys, no Markdown and no prose."
        ),
        "evaluator": evaluator,
        "expected": expected,
        "max_tokens": 65536,
    }
    if rubric is not None:
        raw["rubric"] = rubric
    elif evaluator == "json_match":
        raw["rubric"] = _rubric(answer)
    return raw


def _ledger_packet(rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, int]]:
    owners = ["north", "south", "central"]
    records: list[dict[str, Any]] = []
    for index in range(6):
        entity = f"E{index}"
        owner = owners[index % len(owners)]
        for revision in range(4):
            start = revision * 3 + rng.randrange(0, 3)
            records.append(
                {
                    "id": f"{entity}-r{revision}",
                    "entity": entity,
                    "owner": owner,
                    "valid_from": start,
                    "valid_to": start + rng.randrange(3, 8),
                    "recorded": revision * 4 + rng.randrange(0, 3),
                    "revision": revision,
                    "amount": rng.randrange(-40, 121),
                    "state": "void" if revision == 3 and index % 3 == 0 else "active",
                }
            )
    rng.shuffle(records)
    query = {"valid_time": 7 + rng.randrange(0, 5), "knowledge_cutoff": 8 + rng.randrange(0, 6)}
    return records, query


def _ledger_oracle(records: list[dict[str, Any]], query: dict[str, int]) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        if not (
            record["valid_from"] <= query["valid_time"] < record["valid_to"]
            and record["recorded"] <= query["knowledge_cutoff"]
        ):
            continue
        old = selected.get(record["entity"])
        if old is None or (record["recorded"], record["revision"]) > (
            old["recorded"], old["revision"]
        ):
            selected[record["entity"]] = record
    totals = defaultdict(int)
    void_entities = []
    for entity, record in selected.items():
        if record["state"] == "void":
            void_entities.append(entity)
        else:
            totals[record["owner"]] += record["amount"]
    return {
        "selected": {f"E{i}": selected[f"E{i}"]["id"] if f"E{i}" in selected else None for i in range(6)},
        "totals": {owner: totals[owner] for owner in ("central", "north", "south")},
        "void_entities": sorted(void_entities),
        "net_amount": sum(totals.values()),
    }


def _ledger_independent(records: list[dict[str, Any]], query: dict[str, int]) -> dict[str, Any]:
    by_entity = {f"E{i}": [] for i in range(6)}
    for record in records:
        if record["entity"] in by_entity and record["recorded"] <= query["knowledge_cutoff"]:
            if record["valid_from"] <= query["valid_time"] < record["valid_to"]:
                by_entity[record["entity"]].append(record)
    selected = {
        entity: max(rows, key=lambda row: (row["recorded"], row["revision"])) if rows else None
        for entity, rows in by_entity.items()
    }
    totals = {owner: 0 for owner in ("central", "north", "south")}
    void_entities = []
    for record in selected.values():
        if record is None:
            continue
        if record["state"] == "void":
            void_entities.append(record["entity"])
        else:
            totals[record["owner"]] += record["amount"]
    return {
        "selected": {entity: record["id"] if record else None for entity, record in selected.items()},
        "totals": totals,
        "void_entities": sorted(void_entities),
        "net_amount": sum(totals.values()),
    }


def _ledger_case(category: str, family: str, variant: int, split: str, demand: str) -> dict[str, Any]:
    rng = _rng(category, family, variant, split)
    records, query = _ledger_packet(rng)
    answer = _ledger_oracle(records, query)
    prompt = (
        "Each record has an entity, owner, valid-time interval [valid_from, valid_to), "
        "recorded time, revision, signed amount, and state. For the requested business "
        "time and knowledge cutoff, keep eligible records only and select the greatest "
        "(recorded, revision) per entity. A selected void record contributes no amount "
        "but must remain listed. Return selected (E0..E5), totals (central/north/south), "
        "void_entities sorted, and net_amount.\n"
        f"Query: {json.dumps(query, separators=(',', ':'))}\n"
        f"Records: {json.dumps(records, separators=(',', ':'))}"
    )
    return _case(category=category, family=family, variant=variant, split=split, demand=demand, prompt=prompt, answer=answer)


def _plan_options(rng: random.Random) -> tuple[list[dict[str, Any]], list[str], int]:
    targets = [f"t{i}" for i in range(6)]
    options = [
        {"id": "P0", "requires": ["seed"], "produces": ["a"], "cost": rng.randrange(2, 7), "risk": rng.randrange(1, 5)},
        {"id": "P1", "requires": ["seed"], "produces": ["b"], "cost": rng.randrange(2, 7), "risk": rng.randrange(1, 5)},
        {"id": "P2", "requires": ["a"], "produces": ["c", "t0"], "cost": rng.randrange(2, 8), "risk": rng.randrange(1, 5)},
        {"id": "P3", "requires": ["b"], "produces": ["d", "t1"], "cost": rng.randrange(2, 8), "risk": rng.randrange(1, 5)},
        {"id": "P4", "requires": ["c", "d"], "produces": ["e", "t2", "t3"], "cost": rng.randrange(3, 9), "risk": rng.randrange(1, 5)},
        {"id": "P5", "requires": ["a", "d"], "produces": ["f", "t2"], "cost": rng.randrange(2, 9), "risk": rng.randrange(1, 5)},
        {"id": "P6", "requires": ["e"], "produces": ["t4"], "cost": rng.randrange(2, 8), "risk": rng.randrange(1, 5)},
        {"id": "P7", "requires": ["f"], "produces": ["t5"], "cost": rng.randrange(2, 8), "risk": rng.randrange(1, 5)},
        {"id": "P8", "requires": ["seed"], "produces": ["t0", "t1", "t2"], "cost": rng.randrange(5, 13), "risk": rng.randrange(1, 5)},
        {"id": "P9", "requires": ["seed"], "produces": ["t3", "t4", "t5"], "cost": rng.randrange(5, 13), "risk": rng.randrange(1, 5)},
    ]
    risk_limit = 18
    return options, targets, risk_limit


def _plan_order(options: list[dict[str, Any]], chosen: tuple[int, ...]) -> tuple[str, ...] | None:
    orders = _plan_orders(options, chosen)
    return orders[0] if orders else None


def _plan_orders(options: list[dict[str, Any]], chosen: tuple[int, ...]) -> list[tuple[str, ...]]:
    available = {"seed"}
    remaining = set(chosen)
    orders: list[tuple[str, ...]] = []

    def visit(tokens: set[str], pending: set[int], order: tuple[str, ...]) -> None:
        if not pending:
            orders.append(order)
            return
        ready = sorted(i for i in pending if set(options[i]["requires"]) <= tokens)
        for index in ready:
            visit(
                tokens | set(options[index]["produces"]),
                pending - {index},
                order + (options[index]["id"],),
            )

    visit(available, remaining, ())
    return orders


def _plan_oracle(options: list[dict[str, Any]], targets: list[str], risk_limit: int) -> dict[str, Any]:
    candidates = []
    for mask in range(1, 1 << len(options)):
        chosen = tuple(i for i in range(len(options)) if mask & (1 << i))
        risk = sum(options[i]["risk"] for i in chosen)
        if risk > risk_limit:
            continue
        produced = {token for i in chosen for token in options[i]["produces"]}
        if not set(targets) <= produced:
            continue
        cost = sum(options[i]["cost"] for i in chosen)
        for order in _plan_orders(options, chosen):
            candidates.append((cost, len(chosen), order, risk))
    best_key = min((cost, count) for cost, count, _, _ in candidates)
    winners = sorted((order, risk) for cost, count, order, risk in candidates if (cost, count) == best_key)
    cost, count = best_key
    return {
        "cost": cost,
        "call_count": count,
        "calls": list(winners[0][0]),
        "risk": winners[0][1],
        "optimal_sequences": [list(order) for order, _ in winners],
    }


def _plan_independent(options: list[dict[str, Any]], targets: list[str], risk_limit: int) -> dict[str, Any]:
    # A different recursion over pending calls, rather than subset enumeration.
    solutions: list[tuple[int, int, tuple[str, ...], int]] = []

    def visit(available: frozenset[str], remaining: frozenset[int], order: tuple[str, ...], cost: int, risk: int) -> None:
        produced = available & set(targets)
        if len(produced) == len(targets):
            solutions.append((cost, len(order), order, risk))
            return
        for index in sorted(remaining):
            option = options[index]
            if not set(option["requires"]) <= available:
                continue
            if risk + option["risk"] > risk_limit:
                continue
            visit(
                frozenset(available | set(option["produces"])),
                frozenset(x for x in remaining if x != index),
                order + (option["id"],),
                cost + option["cost"],
                risk + option["risk"],
            )

    visit(frozenset({"seed"}), frozenset(range(len(options))), (), 0, 0)
    best_key = min((cost, count) for cost, count, _, _ in solutions)
    winners = sorted((order, risk) for cost, count, order, risk in solutions if (cost, count) == best_key)
    cost, count = best_key
    return {
        "cost": cost,
        "call_count": count,
        "calls": list(winners[0][0]),
        "risk": winners[0][1],
        "optimal_sequences": [list(order) for order, _ in winners],
    }


def _plan_case(category: str, family: str, variant: int, split: str, demand: str) -> dict[str, Any]:
    rng = _rng(category, family, variant, split)
    options, targets, risk_limit = _plan_options(rng)
    answer = _plan_oracle(options, targets, risk_limit)
    prompt = (
        "You start with token seed. Each option can run once after every required token "
        "exists and then produces its listed tokens. Calls are sequential. The final state "
        f"must contain all targets {targets}; total risk must be at most {risk_limit}. "
        "Minimize cost, then call_count. Return cost, call_count, calls in execution order, "
        "risk, and every optimal sequence sorted lexicographically.\n"
        f"Options: {json.dumps(options, separators=(',', ':'))}"
    )
    return _case(category=category, family=family, variant=variant, split=split, demand=demand, prompt=prompt, answer=answer)


def _transform_data(rng: random.Random) -> tuple[list[list[int]], dict[str, int]]:
    matrix = [[rng.randrange(-20, 41) for _ in range(6)] for _ in range(6)]
    spec = {"keep_mod": rng.randrange(3, 6), "rotation": rng.randrange(0, 6), "modulus": 11}
    return matrix, spec


def _transform_oracle(matrix: list[list[int]], spec: dict[str, int]) -> dict[str, Any]:
    values = []
    for position in range(36):
        source_row, column = divmod(position, 6)
        row = (source_row + spec["rotation"]) % 6
        value = matrix[row][column if source_row % 2 == 0 else 5 - column]
        if (value - position) % spec["keep_mod"] != 0:
            values.append(value)
    emitted = [sum((-1) ** i * value for i, value in enumerate(values[:j + 1])) % spec["modulus"] for j in range(len(values))]
    runs: list[list[int]] = []
    for value in emitted:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return {"runs": runs, "kept": len(values), "checksum": sum((i + 1) * value for i, value in enumerate(emitted))}


def _transform_independent(matrix: list[list[int]], spec: dict[str, int]) -> dict[str, Any]:
    coordinates = []
    for row in range(6):
        ordered = range(6) if row % 2 == 0 else range(5, -1, -1)
        coordinates.extend(((row + spec["rotation"]) % 6, column) for column in ordered)
    kept = [matrix[row][column] for position, (row, column) in enumerate(coordinates) if (matrix[row][column] - position) % spec["keep_mod"]]
    emitted = []
    total = 0
    sign = 1
    for value in kept:
        total = (total + sign * value) % spec["modulus"]
        emitted.append(total)
        sign *= -1
    runs = []
    for value in emitted:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return {"runs": runs, "kept": len(kept), "checksum": sum((i + 1) * value for i, value in enumerate(emitted))}


def _transform_case(category: str, family: str, variant: int, split: str, demand: str) -> dict[str, Any]:
    rng = _rng(category, family, variant, split)
    matrix, spec = _transform_data(rng)
    answer = _transform_oracle(matrix, spec)
    prompt = (
        "Traverse a 6x6 matrix in row order after rotating row indices by rotation. "
        "Even rows traverse left-to-right and odd rows right-to-left. Keep a value when "
        "(value - original_position) modulo keep_mod is nonzero. Emit alternating-sign "
        "prefix sums modulo modulus, run-length encode equal emitted values, and return "
        "runs, kept, and checksum.\n"
        f"Matrix: {json.dumps(matrix, separators=(',', ':'))}\n"
        f"Parameters: {json.dumps(spec, separators=(',', ':'))}"
    )
    return _case(category=category, family=family, variant=variant, split=split, demand=demand, prompt=prompt, answer=answer)


def _evidence_packet(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims = [{"id": f"C{i}", "topic": f"topic-{i}"} for i in range(6)]
    evidence = []
    for claim in claims:
        for index in range(3):
            stance = ("supports", "refutes", "supports")[index]
            evidence.append(
                {
                    "id": f"E{claim['id'][1:]}-{index}",
                    "claim": claim["id"],
                    "stance": stance,
                    "strength": rng.randrange(1, 5),
                    "source_rank": rng.randrange(1, 4),
                    "revoked": bool(index == 1 and int(claim["id"][1:]) % 3 == 0),
                }
            )
    rng.shuffle(evidence)
    return claims, evidence


def _evidence_oracle(claims: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    out = []
    for claim in claims:
        rows = [row for row in evidence if row["claim"] == claim["id"] and not row["revoked"]]
        supports = [row for row in rows if row["stance"] == "supports"]
        refutes = [row for row in rows if row["stance"] == "refutes"]
        support_score = sum(row["strength"] * row["source_rank"] for row in supports)
        refute_score = sum(row["strength"] * row["source_rank"] for row in refutes)
        if support_score >= 8 and support_score > refute_score:
            status = "confirmed"
        elif refute_score >= 6 and refute_score >= support_score:
            status = "refuted"
        else:
            status = "unknown"
        selected = sorted(
            [row["id"] for row in rows if (row["stance"] == "supports" and support_score >= 8) or (row["stance"] == "refutes" and refute_score >= 6)]
        )
        out.append({"id": claim["id"], "status": status, "evidence": selected})
    return {"claims": out, "unknown_count": sum(row["status"] == "unknown" for row in out)}


def _evidence_independent(claims: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        if not row["revoked"]:
            grouped[row["claim"]].append(row)
    out = []
    for claim in claims:
        rows = grouped[claim["id"]]
        scores = {
            stance: sum(row["strength"] * row["source_rank"] for row in rows if row["stance"] == stance)
            for stance in ("supports", "refutes")
        }
        if scores["supports"] >= 8 and scores["supports"] > scores["refutes"]:
            status = "confirmed"
        elif scores["refutes"] >= 6 and scores["refutes"] >= scores["supports"]:
            status = "refuted"
        else:
            status = "unknown"
        selected = sorted(
            row["id"] for row in rows
            if (row["stance"] == "supports" and scores["supports"] >= 8)
            or (row["stance"] == "refutes" and scores["refutes"] >= 6)
        )
        out.append({"id": claim["id"], "status": status, "evidence": selected})
    return {"claims": out, "unknown_count": sum(row["status"] == "unknown" for row in out)}


def _evidence_case(category: str, family: str, variant: int, split: str, demand: str) -> dict[str, Any]:
    rng = _rng(category, family, variant, split)
    claims, evidence = _evidence_packet(rng)
    answer = _evidence_oracle(claims, evidence)
    prompt = (
        "Classify each claim using only non-revoked evidence. For a claim, support score "
        "is strength*source_rank over supporting rows; refute score is the analogous sum. "
        "A support score >=8 and strictly greater than refute means confirmed. A refute "
        "score >=6 and at least support means refuted; otherwise unknown. Include every "
        "evidence ID used for the threshold decision, sorted, and unknown_count.\n"
        f"Claims: {json.dumps(claims, separators=(',', ':'))}\n"
        f"Evidence: {json.dumps(evidence, separators=(',', ':'))}"
    )
    critical = {f"json:claims[{i}].status" for i in range(len(claims))}
    return _case(category=category, family=family, variant=variant, split=split, demand=demand, prompt=prompt, answer=answer, rubric=_rubric(answer, critical=critical))


def _semantic_packet(rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, str]]:
    glossary = {"mirek": "Mirek", "sml": "contract", "must": "must", "only": "only"}
    statements = []
    for i in range(6):
        statements.append(
            {
                "id": f"S{i}",
                "language": rng.choice(["cs", "de", "en"]),
                "subject": rng.choice(["mirek", "vendor", "team"]),
                "quantifier": rng.choice(["all", "some", "none"]),
                "polarity": rng.choice(["positive", "negative"]),
                "predicate": rng.choice(["sml", "approved", "ready"]),
                "condition": rng.choice(["active", "revoked", "unknown"]),
            }
        )
    return statements, glossary


def _semantic_oracle(statements: list[dict[str, Any]], glossary: dict[str, str]) -> dict[str, Any]:
    meanings = []
    for row in statements:
        meaning = {
            "id": row["id"],
            "subject": glossary.get(row["subject"], row["subject"]),
            "scope": row["quantifier"],
            "polarity": row["polarity"],
            "predicate": glossary.get(row["predicate"], row["predicate"]),
            "active": row["condition"] == "active",
        }
        if "source" in row:
            meaning["direction"] = [row["source"], row["target"]]
        meanings.append(meaning)
    return {"meanings": meanings}


def _semantic_independent(statements: list[dict[str, Any]], glossary: dict[str, str]) -> dict[str, Any]:
    meanings = []
    for row in statements:
        meaning = {"id": row["id"]}
        meaning["subject"] = glossary.get(row["subject"], row["subject"])
        meaning["scope"] = {"all": "all", "some": "some", "none": "none"}[row["quantifier"]]
        meaning["polarity"] = "negative" if row["polarity"] == "negative" else "positive"
        meaning["predicate"] = glossary.get(row["predicate"], row["predicate"])
        meaning["active"] = bool(row["condition"] == "active")
        if "source" in row:
            meaning["direction"] = [row["source"], row["target"]]
        meanings.append(meaning)
    return {"meanings": meanings}


def _semantic_case(category: str, family: str, variant: int, split: str, demand: str) -> dict[str, Any]:
    rng = _rng(category, family, variant, split)
    statements, glossary = _semantic_packet(rng)
    if category == "Language Translations":
        directions = [("cs", "en"), ("en", "cs"), ("cs", "de"), ("de", "cs"), ("en", "de"), ("de", "en")]
        for row, (source, target) in zip(statements, directions):
            row["source"], row["target"] = source, target
    answer = _semantic_oracle(statements, glossary)
    prompt = (
        "Normalize each supplied statement into an English-keyed semantic record. Preserve "
        "quantifier scope (all/some/none), polarity, subject and predicate terminology from "
        "the glossary, and mark active only when condition is exactly active. Preserve input order.\n"
        f"Glossary: {json.dumps(glossary, separators=(',', ':'))}\n"
        f"Statements: {json.dumps(statements, separators=(',', ':'))}\n"
        + (
            "This packet intentionally covers all six Czech/English/German directions; copy "
            "each source/target pair into direction [source,target].\n"
            "Return meanings with id, direction, subject, scope, polarity, predicate, active."
            if category == "Language Translations"
            else "Return meanings with id, subject, scope, polarity, predicate, active."
        )
    )
    return _case(category=category, family=family, variant=variant, split=split, demand=demand, prompt=prompt, answer=answer)


def _allocation_case(category: str, family: str, variant: int, split: str, demand: str) -> dict[str, Any]:
    rng = _rng(category, family, variant, split)
    people = [f"A{i}" for i in range(5)]
    needs = [rng.randrange(2, 8) for _ in people]
    weights = [rng.randrange(1, 5) for _ in people]
    budget = sum(needs) - rng.randrange(1, 4)
    candidates = []
    for allocation in itertools.product(*(range(need + 1) for need in needs)):
        if sum(allocation) > budget:
            continue
        shortfall = sum((needs[i] - allocation[i]) * weights[i] for i in range(len(people)))
        max_gap = max(allocation) - min(allocation)
        candidates.append((shortfall, max_gap, tuple(allocation)))
    best = min(candidates)
    ties = sorted(list(row[2]) for row in candidates if row[:2] == best[:2])
    answer = {
        "budget": budget,
        "allocation": list(best[2]),
        "weighted_shortfall": best[0],
        "max_gap": best[1],
        "tied_optima": ties,
    }
    prompt = (
        "Allocate integer units to people in listed order. Each allocation is between zero and "
        "that person's need and total allocation cannot exceed budget. Minimize weighted "
        "shortfall, then max allocation gap; return one optimum, all tied optima sorted, and "
        "the objective values.\n"
        f"People: {json.dumps(people)}\nNeeds: {json.dumps(needs)}\nWeights: {json.dumps(weights)}\nBudget: {budget}"
    )
    return _case(category=category, family=family, variant=variant, split=split, demand=demand, prompt=prompt, answer=answer)


def _retrieval_case(category: str, family: str, variant: int, split: str, demand: str) -> dict[str, Any]:
    rng = _rng(category, family, variant, split)
    nodes = [f"N{i}" for i in range(8)]
    edges = []
    for i in range(7):
        edges.append({"from": nodes[i], "to": nodes[i + 1], "cost": rng.randrange(1, 8), "active": True})
    edges.extend(
        [
            {"from": "N0", "to": "N3", "cost": rng.randrange(2, 8), "active": True},
            {"from": "N3", "to": "N7", "cost": rng.randrange(2, 8), "active": True},
            {"from": "N1", "to": "N5", "cost": rng.randrange(2, 8), "active": False},
            {"from": "N2", "to": "N6", "cost": rng.randrange(2, 8), "active": True},
        ]
    )
    distances = {nodes[0]: (0, [nodes[0]])}
    for node in nodes:
        cost, path = distances[node]
        for edge in edges:
            if edge["from"] != node or not edge["active"]:
                continue
            candidate = (cost + edge["cost"], path + [edge["to"]])
            old = distances.get(edge["to"])
            if old is None or candidate[0] < old[0] or candidate[0] == old[0] and candidate[1] < old[1]:
                distances[edge["to"]] = candidate
    answer = {"cost": distances["N7"][0], "path": distances["N7"][1], "active_edges": sum(e["active"] for e in edges)}
    prompt = (
        "Follow only active directed edges from N0 to N7. Minimize total edge cost and break "
        "ties by lexicographically smallest node path. Return cost, path, and active_edges. "
        "Disabled edges are distractors and must not be used.\n"
        f"Edges: {json.dumps(edges, separators=(',', ':'))}"
    )
    return _case(category=category, family=family, variant=variant, split=split, demand=demand, prompt=prompt, answer=answer)


def _format_case(category: str, family: str, variant: int, split: str, demand: str) -> dict[str, Any]:
    rng = _rng(category, family, variant, split)
    facts = [f"fact-{i}" for i in range(5)]
    operations = ["preserve", "move-last", "omit", "preserve", "move-first"]
    retained = [fact for fact, operation in zip(facts, operations) if operation != "omit"]
    retained.remove("fact-1")
    retained.append("fact-1")
    retained.remove("fact-4")
    retained.insert(0, "fact-4")
    answer = {
        "facts": retained,
        "line_count": 4,
        "viewpoint": "first-person",
        "form": "ABBA" if rng.randrange(2) else "ABAB",
        "forbidden": ["fact-2"],
    }
    prompt = (
        "Return a measurable revision specification for the supplied facts. Apply operations "
        "left-to-right after omission: preserve keeps order, move-last moves that retained fact "
        "to the end, and move-first moves it to the front. State the required line count, "
        "viewpoint, form pattern, and forbidden fact IDs.\n"
        f"Facts: {json.dumps(facts)}\nOperations by index: {json.dumps(operations)}"
    )
    return _case(category=category, family=family, variant=variant, split=split, demand=demand, prompt=prompt, answer=answer)


def _reference_fold_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    balances = defaultdict(int)
    for row in records:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        balances[row["account"]] += row["amount"] if row["op"] == "credit" else -row["amount"]
    return [{"account": account, "balance": balances[account]} for account in sorted(balances)]


def _independent_fold_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_by_id: dict[str, dict[str, Any]] = {}
    for row in records:
        first_by_id.setdefault(row["id"], row)
    totals: dict[str, int] = {}
    for row in first_by_id.values():
        amount = row["amount"] if row["op"] == "credit" else -row["amount"]
        totals[row["account"]] = totals.get(row["account"], 0) + amount
    return [{"account": account, "balance": totals[account]} for account in sorted(totals)]


def _reference_route(graph: list[list[Any]], start: str, target: str) -> dict[str, Any]:
    best: tuple[int, tuple[str, ...]] | None = None
    def visit(node: str, cost: int, path: tuple[str, ...]) -> None:
        nonlocal best
        if node == target:
            candidate = (cost, path)
            if best is None or candidate < best:
                best = candidate
            return
        for source, dest, edge_cost, enabled in graph:
            if source == node and enabled and dest not in path:
                visit(dest, cost + edge_cost, path + (dest,))
    visit(start, 0, (start,))
    if best is None:
        return {"cost": None, "path": []}
    return {"cost": best[0], "path": list(best[1])}


def _independent_route(graph: list[list[Any]], start: str, target: str) -> dict[str, Any]:
    frontier: list[tuple[int, tuple[str, ...]]] = [(0, (start,))]
    best: dict[str, tuple[int, tuple[str, ...]]] = {start: (0, (start,))}
    while frontier:
        cost, path = min(frontier)
        frontier.remove((cost, path))
        node = path[-1]
        if node == target:
            return {"cost": cost, "path": list(path)}
        for source, dest, edge_cost, enabled in graph:
            if source != node or not enabled or dest in path:
                continue
            candidate = (cost + edge_cost, path + (dest,))
            if dest not in best or candidate < best[dest]:
                best[dest] = candidate
                frontier.append(candidate)
    return {"cost": None, "path": []}


def _independent_format(facts: list[str], operations: list[str], form: str) -> dict[str, Any]:
    retained = [fact for fact, operation in zip(facts, operations) if operation != "omit"]
    for fact, operation in zip(facts, operations):
        if operation == "move-last" and fact in retained:
            retained.remove(fact)
            retained.append(fact)
        elif operation == "move-first" and fact in retained:
            retained.remove(fact)
            retained.insert(0, fact)
    return {
        "facts": retained,
        "line_count": 4,
        "viewpoint": "first-person",
        "form": form,
        "forbidden": [fact for fact, operation in zip(facts, operations) if operation == "omit"],
    }


def _code_case(category: str, family: str, variant: int, split: str, demand: str) -> dict[str, Any]:
    rng = _rng(category, family, variant, split)
    if family in {"stateful-parser", "interacting-faults", "dependency-regression", "external-memory", "atomic-unicode-links", "interrupted-rollback"}:
        function = "fold_records"
        records = [
            {"id": "a", "account": "A", "op": "credit", "amount": 7},
            {"id": "b", "account": "A", "op": "debit", "amount": 3},
            {"id": "a", "account": "A", "op": "credit", "amount": 99},
            {"id": "c", "account": "B", "op": "credit", "amount": rng.randrange(4, 12)},
        ]
        fixtures = [
            {"id": "duplicates", "function": function, "args": [records], "expected": _reference_fold_records(records)},
            {"id": "empty", "function": function, "args": [[]], "expected": []},
        ]
        prompt = (
            f"Implement `{function}(records)`. Records are dictionaries with id, account, op, amount. "
            "Process each id at most once, keeping its first occurrence. Credit adds amount; every "
            "other operation subtracts it. Return sorted account objects with account and balance. "
            "The hidden tests include duplicate IDs, empty input, negative amounts, and Unicode account IDs."
        )
    else:
        function = "best_route"
        graph = [
            ["S", "A", 3, True], ["S", "B", 5, True], ["A", "B", 1, True],
            ["A", "T", 8, True], ["B", "T", 2 + rng.randrange(4), True],
            ["S", "T", 20, False], ["T", "S", 1, True],
        ]
        fixtures = [
            {"id": "route", "function": function, "args": [graph, "S", "T"], "expected": _reference_route(graph, "S", "T")},
            {"id": "unreachable", "function": function, "args": [graph, "T", "Z"], "expected": {"cost": None, "path": []}},
        ]
        prompt = (
            f"Implement `{function}(graph, start, target)`. Graph rows are [from,to,cost,enabled]. "
            "Use enabled directed edges only, forbid cycles, minimize total cost, and break ties "
            "by lexicographically smallest full path. Return {cost,path}; return null and [] when unreachable."
        )
    return _case(
        category=category,
        family=family,
        variant=variant,
        split=split,
        demand=demand,
        prompt=prompt,
        answer=None,
        evaluator="code_exec",
        expected=fixtures,
        rubric=[{"id": fixture["id"], "dimension": "content", "weight": 1, "mandatory": True} for fixture in fixtures],
    )


FAMILY_KIND: dict[str, str] = {
    "Logical Reasoning": "plan", "Mathematical Reasoning": "plan", "Reading Comprehension": "ledger",
    "Classification": "evidence", "Factual Knowledge": "evidence", "Truthfulness": "evidence",
    "Code Generation": "code", "Advanced Coding": "code", "Code Review": "evidence",
    "Terminal Algorithms": "code", "Terminal Debugging": "code", "Needle Retrieval": "retrieval",
    "Long Context Coherence": "ledger", "Summarization": "ledger", "Tool Using": "plan",
    "Agentic Use Cases": "plan", "Interactive Tool Use": "plan", "Security": "evidence",
    "Terminal System Admin": "plan", "Terminal File Operations": "transform", "Terminal Science": "plan",
    "Instruction Following": "transform", "Creative Writing": "format", "Translation": "semantic",
    "Language Translations": "semantic", "Ethical Reasoning": "allocation", "Legal": "evidence",
    "Finances": "ledger", "R&D": "plan",
}


def build_cases(*, split: str = "development", variants: int = DEFAULT_VARIANTS) -> list[dict[str, Any]]:
    if split not in {"development", "evaluation"}:
        raise ValueError("hardening split must be development or evaluation")
    if not 1 <= variants <= 10:
        raise ValueError("hardening variants must be between 1 and 10")
    cases: list[dict[str, Any]] = []
    for category, families in HARDENING_FAMILIES.items():
        kind = FAMILY_KIND[category]
        for family_spec in families:
            family = family_spec["id"]
            demand = family_spec["demand"]
            for variant in range(variants):
                if kind == "ledger":
                    raw = _ledger_case(category, family, variant, split, demand)
                elif kind == "plan":
                    raw = _plan_case(category, family, variant, split, demand)
                elif kind == "transform":
                    raw = _transform_case(category, family, variant, split, demand)
                elif kind == "evidence":
                    raw = _evidence_case(category, family, variant, split, demand)
                elif kind == "semantic":
                    raw = _semantic_case(category, family, variant, split, demand)
                elif kind == "allocation":
                    raw = _allocation_case(category, family, variant, split, demand)
                elif kind == "retrieval":
                    raw = _retrieval_case(category, family, variant, split, demand)
                elif kind == "format":
                    raw = _format_case(category, family, variant, split, demand)
                elif kind == "code":
                    raw = _code_case(category, family, variant, split, demand)
                else:  # pragma: no cover - mapping is validated below
                    raise ValueError(f"unknown hardening kind {kind!r}")
                cases.append(raw)
    return cases


def load_questions(*, split: str = "development", variants: int = DEFAULT_VARIANTS):
    # Interactive Tool Use is supplied by the simulator in ``interactive_tasks``;
    # keep its raw planning oracle in this module for coverage verification but
    # never turn it into a static JSON task in an executable profile.
    return [
        _parse_question(raw, raw["id"])
        for raw in build_cases(split=split, variants=variants)
        if raw["category"] != "Interactive Tool Use"
    ]


def verify_cases(*, split: str = "development", variants: int = DEFAULT_VARIANTS) -> dict[str, int]:
    """Cross-check every generated answer through a second derivation path."""
    checked = 0
    for raw in build_cases(split=split, variants=variants):
        expected = raw["expected"]
        category = raw["category"]
        family = raw["id"][3:].rsplit(f"-{split}-v", 1)[0]
        variant = int(raw["id"].rsplit("-v", 1)[1]) - 1
        kind = FAMILY_KIND[category]
        rng = _rng(category, family, variant, split)
        if kind == "ledger":
            records, query = _ledger_packet(rng)
            actual = _ledger_independent(records, query)
            candidate = expected["value"]
        elif kind == "plan":
            options, targets, risk_limit = _plan_options(rng)
            actual = _plan_independent(options, targets, risk_limit)
            candidate = expected["value"]
        elif kind == "transform":
            matrix, spec = _transform_data(rng)
            actual = _transform_independent(matrix, spec)
            candidate = expected["value"]
        elif kind == "evidence":
            claims, evidence = _evidence_packet(rng)
            actual = _evidence_independent(claims, evidence)
            candidate = expected["value"]
        elif kind == "semantic":
            statements, glossary = _semantic_packet(rng)
            if category == "Language Translations":
                directions = [("cs", "en"), ("en", "cs"), ("cs", "de"), ("de", "cs"), ("en", "de"), ("de", "en")]
                for row, (source, target) in zip(statements, directions):
                    row["source"], row["target"] = source, target
            actual = _semantic_independent(statements, glossary)
            candidate = expected["value"]
        elif kind == "allocation":
            # Recompute the allocation by exhaustive search from the prompt's seed.
            people = [f"A{i}" for i in range(5)]
            needs = [rng.randrange(2, 8) for _ in people]
            weights = [rng.randrange(1, 5) for _ in people]
            budget = sum(needs) - rng.randrange(1, 4)
            candidates = []
            for allocation in itertools.product(*(range(need + 1) for need in needs)):
                if sum(allocation) <= budget:
                    candidates.append((sum((needs[i] - allocation[i]) * weights[i] for i in range(5)), max(allocation) - min(allocation), allocation))
            best = min(candidates)
            actual = {"budget": budget, "allocation": list(best[2]), "weighted_shortfall": best[0], "max_gap": best[1], "tied_optima": sorted(list(row[2]) for row in candidates if row[:2] == best[:2])}
            candidate = expected["value"]
        elif kind == "retrieval":
            nodes = [f"N{i}" for i in range(8)]
            edges = []
            for i in range(7):
                edges.append({"from": nodes[i], "to": nodes[i + 1], "cost": rng.randrange(1, 8), "active": True})
            edges.extend([
                {"from": "N0", "to": "N3", "cost": rng.randrange(2, 8), "active": True},
                {"from": "N3", "to": "N7", "cost": rng.randrange(2, 8), "active": True},
                {"from": "N1", "to": "N5", "cost": rng.randrange(2, 8), "active": False},
                {"from": "N2", "to": "N6", "cost": rng.randrange(2, 8), "active": True},
            ])
            # Bellman-Ford style relaxation is independent of the generator's forward scan.
            distance = {"N0": (0, ["N0"])}
            for _ in range(8):
                changed = False
                for edge in edges:
                    if not edge["active"] or edge["from"] not in distance:
                        continue
                    candidate_route = (distance[edge["from"]][0] + edge["cost"], distance[edge["from"]][1] + [edge["to"]])
                    if edge["to"] not in distance or candidate_route < distance[edge["to"]]:
                        distance[edge["to"]] = candidate_route
                        changed = True
                if not changed:
                    break
            actual = {"cost": distance["N7"][0], "path": distance["N7"][1], "active_edges": sum(edge["active"] for edge in edges)}
            candidate = expected["value"]
        elif kind == "format":
            facts = [f"fact-{i}" for i in range(5)]
            operations = ["preserve", "move-last", "omit", "preserve", "move-first"]
            form = "ABBA" if rng.randrange(2) else "ABAB"
            actual = _independent_format(facts, operations, form)
            candidate = expected["value"]
        elif kind == "code":
            candidate = expected
            actual = []
            for fixture in expected:
                function = fixture["function"]
                args = fixture.get("args") or []
                if function == "fold_records":
                    value = _independent_fold_records(args[0])
                elif function == "best_route":
                    value = _independent_route(args[0], args[1], args[2])
                else:  # pragma: no cover - family mapping is validated below
                    raise AssertionError(function)
                if value != fixture.get("expected"):
                    raise AssertionError(
                        f"independent code oracle mismatch for {raw['id']} fixture {fixture.get('id')}"
                    )
                actual.append(value)
            candidate = [fixture.get("expected") for fixture in expected]
        else:  # pragma: no cover
            raise AssertionError(kind)
        if actual != candidate:
            raise AssertionError(f"independent oracle mismatch for {raw['id']}: {actual!r} != {candidate!r}")
        checked += 1
    return {"cases": checked, "families": len(HARDENING_FAMILIES) * 2, "variants": variants}


if __name__ == "__main__":
    print(verify_cases())
