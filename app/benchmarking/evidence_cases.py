"""Original v13 finite-evidence and typed tool-trace tasks.

Method inspiration: FEVER/HoVer evidence sufficiency, BFCL typed/executed
calls and ComplexBench criterion dependencies. No dataset questions copied.
"""

import hashlib
import itertools
import json
import random

from app.benchmarking.evaluators import _json_leaf_paths, json_at_path
from app.benchmarking.rigorous_cases import structured


def _task(family, category, seed, variant, prompt, data, answer):
    q = structured(family, category, variant, "evaluation",
                   prompt + "\nINPUT=" + json.dumps(data, sort_keys=True), answer)
    q.id = f"Q13-{family}-s{seed}-v{variant + 1:02d}"
    q.source = "Original rigorous-v13; see BENCHMARK_DESIGN.md"
    q.metadata.update(family=f"Q13-{family}", seed=seed, variant=variant)
    q.expected["integer_paths"] = [path for path in _json_leaf_paths(answer)
                                   if type(json_at_path(answer, path)[0]) is int]
    q.prompt += "\nOutput all counts, balances and versions as JSON integer literals, not floats or strings."
    return q


def holds(formula, world):
    if isinstance(formula, str):
        return world[formula]
    op, *args = formula
    if op == "not":
        return not holds(args[0], world)
    a, b = (holds(arg, world) for arg in args)
    return {"and": a and b, "or": a or b, "xor": a != b,
            "iff": a == b, "implies": not a or b}[op]


def evidence_answer(data):
    """Bitset model intersection and minimal-set antichains."""
    atoms = data["atoms"]
    worlds = [dict(zip(atoms, bits)) for bits in itertools.product((False, True), repeat=len(atoms))]
    universe = (1 << len(worlds)) - 1
    sources = sorted(data["sources"], key=lambda s: s["id"])
    masks = [sum(1 << i for i, w in enumerate(worlds) if holds(s["formula"], w)) for s in sources]
    subsets = [universe]
    for mask in range(1, 1 << len(sources)):
        lowest = mask & -mask
        subsets.append(subsets[mask ^ lowest] & masks[lowest.bit_length() - 1])

    def names(mask):
        return [s["id"] for i, s in enumerate(sources) if mask & (1 << i)]

    def minimal(predicate):
        # Entailment is monotone within satisfiable subsets. For a consistent
        # candidate, checking every one-source deletion proves minimality.
        return sorted(names(mask) for mask, valid in enumerate(subsets)
                      if predicate(valid) and all(not predicate(subsets[mask ^ (1 << i)])
                                                 for i in range(len(sources)) if mask & (1 << i)))

    cores = minimal(lambda valid: valid == 0)
    retained = max(mask.bit_count() for mask, valid in enumerate(subsets) if valid)
    best = [mask for mask, valid in enumerate(subsets) if valid and mask.bit_count() == retained]
    repaired = 0
    for mask in best:
        repaired |= subsets[mask]
    full = subsets[-1]

    def status(valid, true):
        if not valid:
            return "inconsistent"
        return "supported" if not valid & ~true else "refuted" if not valid & true else "unknown"

    def witness(valid):
        return list(worlds[(valid & -valid).bit_length() - 1].values()) if valid else None

    claims = []
    for claim in data["claims"]:
        true = sum(1 << i for i, w in enumerate(worlds) if holds(claim["formula"], w))
        claims.append({"id": claim["id"], "full_status": status(full, true),
                       "supports": minimal(lambda valid: bool(valid) and not valid & ~true),
                       "refutations": minimal(lambda valid: bool(valid) and not valid & true),
                       "repaired_status": status(repaired, true),
                       "true_witness": witness(repaired & true),
                       "false_witness": witness(repaired & ~true)})
    return {"full_model_count": full.bit_count(), "minimal_inconsistent_cores": cores,
            "minimum_deletions": len(sources) - retained,
            "repairs": sorted(names(((1 << len(sources)) - 1) ^ mask) for mask in best),
            "repaired_model_count": repaired.bit_count(), "claims": claims}


def evidence_case(rng, seed, variant):
    # Relabel and reorder sources/atoms; paired variants change just one assertion.
    atoms = sorted(rng.sample([f"p{i}" for i in range(20)], 8))
    a, b, c, d, e, f, g, h = rng.sample(atoms, len(atoms))
    formulas = [a, ["implies", a, b], ["implies", b, c], ["iff", d, ["not", c]],
                ["or", d, e], ["iff", f, ["xor", b, e]], ["implies", g, ["not", e]],
                ["implies", a, c], ["implies", ["and", a, e], ["not", g]],
                ["not", h] if not variant else ["not", c]]
    ids = rng.sample([f"s{i:02d}" for i in range(30)], len(formulas))
    sources = [{"id": ident, "formula": formula} for ident, formula in zip(ids, formulas)]
    rng.shuffle(sources)
    probes = [c, ["not", c], d, h, ["or", a, h], ["implies", h, e],
              ["xor", b, e], ["or", h, ["not", h]]]
    rng.shuffle(probes)
    data = {"atoms": atoms, "sources": sources,
            "claims": [{"id": f"c{i}", "formula": formula} for i, formula in enumerate(probes)]}
    q = _task("evidence-audit", "Truthfulness", seed, variant,
        "Audit the assertions of eight Boolean indicators in a closed evidence dossier. "
        "All 2^8 assignments are initially possible; unmentioned facts are not false. "
        "Sources are assertions, not inference-rule licenses. A formula is either an atom "
        "ID or an array [operator, operands...]. not is unary; and, or, xor (exactly one), "
        "iff (same truth value), and implies (false antecedent or true consequent) are binary. "
        "A model of a source subset satisfies every formula in it. Sources have equal authority. "
        "Report full_model_count and ALL inclusion-minimal inconsistent source subsets as "
        "minimal_inconsistent_cores. Delete the fewest sources to restore consistency; report "
        "minimum_deletions and ALL sorted deletion-ID lists in repairs. If already consistent "
        "repairs is [[]]. repaired_model_count counts the UNION of models of every minimum "
        "repair, counting a world once even if multiple repairs admit it. Counts are not probabilities. "
        "For each claim in INPUT order, output id, full_status, supports, refutations, "
        "repaired_status, true_witness and false_witness. Status is supported if all models "
        "make it true, refuted if all make it false, unknown if both occur, and inconsistent "
        "if there are no models. Never use vacuous entailment as support. supports/refutations "
        "list ALL inclusion-minimal CONSISTENT source subsets that respectively entail the "
        "claim/its negation, even when the full dossier is inconsistent. A tautology can have "
        "the empty support [] as one member of supports. Sort IDs inside each set, then sort "
        "the set lists lexically. repaired_status uses the union of worlds across all minimum "
        "repairs, not a vote over repair labels. Each witness is the lexically smallest Boolean "
        "vector in that union satisfying/falsifying the claim, in INPUT.atoms order with false "
        "before true, or null if none. Do not substitute a source ID for a witness.",
        data, evidence_answer(data))
    # Give each claim's judgment, evidence and witnesses equal diagnostic weight.
    # A long list of supports must not overwhelm a different claim's judgment.
    q.rubric = []
    for key, value in q.expected["value"].items():
        blocks = [(key, value)] if key != "claims" else [
            (f"claims[{i}].{field}", child)
            for i, row in enumerate(value) for field, child in row.items()]
        for path, child in blocks:
            leaves = _json_leaf_paths(child, path)
            for leaf in leaves:
                q.rubric.append({"id": f"json:{leaf}", "weight": 1 / len(leaves),
                                 "group": path, "mandatory": True,
                                 "dimension": "contract" if path.endswith(".id") else
                                 "evidence" if any(s in path for s in ("supports", "refutations", "witness")) else "content"})
    return q


def trace_answer(data):
    """Typed reference resolution, validation precedence and atomic execution."""
    state = {row["id"]: dict(row) for row in data["accounts"]}
    cache, results = {}, []
    signatures = {"read": {"account": str}, "transfer": {
        "src": str, "dst": str, "amount": int, "src_version": int, "dst_version": int, "key": str}}
    for call in data["calls"]:
        args, error = {}, None
        for name, value in call["args"].items():
            if isinstance(value, dict):
                ref, field = value.get("ref"), value.get("field")
                if (set(value) != {"ref", "field"} or type(ref) is not int
                        or not isinstance(field, str) or not 0 <= ref < len(results)
                        or field not in results[ref]):
                    error = "bad_ref"
                    break
                value = results[ref][field]
            args[name] = value
        sig = signatures.get(call["tool"])
        if error:
            result = {"status": error}
        elif sig is None:
            result = {"status": "unknown_tool"}
        elif set(args) != set(sig) or any(type(args[k]) is not t for k, t in sig.items()):
            result = {"status": "schema"}
        elif any(args[k] < 0 for k in ("src_version", "dst_version") if k in args) or (
                "amount" in args and args["amount"] <= 0) or ("key" in args and not args["key"]):
            result = {"status": "schema"}
        elif call["tool"] == "read":
            result = {"status": "missing"} if args["account"] not in state else {
                "status": "read", "balance": state[args["account"]]["balance"],
                "version": state[args["account"]]["version"]}
        elif args["key"] in cache:
            old, receipt = cache[args["key"]]
            result = dict(receipt) if args == old else {"status": "key_conflict"}
        elif args["src"] not in state or args["dst"] not in state:
            result = {"status": "missing"}
        elif args["src"] == args["dst"]:
            result = {"status": "same_account"}
        else:
            src, dst = state[args["src"]], state[args["dst"]]
            if src["version"] != args["src_version"] or dst["version"] != args["dst_version"]:
                result = {"status": "stale"}
            elif src["balance"] < args["amount"]:
                result = {"status": "funds"}
            else:
                src["balance"] -= args["amount"]
                dst["balance"] += args["amount"]
                src["version"] += 1
                dst["version"] += 1
                result = {"status": "applied", "src_version": src["version"], "dst_version": dst["version"]}
                cache[args["key"]] = (dict(args), dict(result))
        results.append(result)
    return {"results": results, "accounts": [state[k] for k in sorted(state)],
            "cached_keys": sorted(cache), "applied_count": len(cache)}


def tool_case(rng, seed, variant):
    a, b, c = rng.sample([f"a{i}" for i in range(8)], 3)
    money = rng.randrange(30, 60)
    v = rng.randrange(2, 7)
    def ref(index, field):
        return {"ref": index, "field": field}
    transfer = {"src": a, "dst": b, "amount": money, "src_version": v,
                "dst_version": v, "key": "commit"}
    calls = [{"tool": "read", "args": {"account": a}},
             {"tool": "read", "args": {"account": b}},
             {"tool": "transfer", "args": {**transfer, "src_version": ref(0, "version"), "dst_version": ref(1, "version")}},
             {"tool": "transfer", "args": dict(transfer)},
             {"tool": "transfer", "args": {**transfer, "amount": money + variant}},
             {"tool": "transfer", "args": {**transfer, "key": "retry", "amount": money + 1000}},
             {"tool": "transfer", "args": {**transfer, "key": "retry", "src_version": v + 1, "dst_version": v + 1, "amount": money + 1000}},
             {"tool": "transfer", "args": {**transfer, "key": "retry", "src_version": v + 1, "dst_version": v + 1, "amount": money // 2}},
             {"tool": "transfer", "args": {**transfer, "key": "retry", "src_version": ref(5, "src_version")}},
             {"tool": "transfer", "args": {**transfer, "src_version": True}},
             {"tool": "read", "args": {"account": c, "unused": 1}},
             {"tool": "refund", "args": {"account": ref(1, "version")}},
             {"tool": "refund", "args": {"account": ref(11, "version")}},
             {"tool": "transfer", "args": {**transfer, "key": "self", "dst": a, "src_version": 0}},
             {"tool": "transfer", "args": {**transfer, "key": "lost", "dst": "absent", "src_version": 0}},
             {"tool": "transfer", "args": {**transfer, "key": "return", "src": b, "dst": a,
                 "amount": money // 3, "src_version": ref(7, "dst_version"), "dst_version": ref(7, "src_version")}},
             {"tool": "transfer", "args": {**transfer, "key": "retry", "src_version": v + 1, "dst_version": v + 1, "amount": money // 2}},
             {"tool": "read", "args": {"account": a}},
             {"tool": "read", "args": {"account": ref(-1, "status")}},
             {"tool": "read", "args": {"account": ref(True, "status")}},
             {"tool": "read", "args": {"account": ref(21, "status")}},
             {"tool": "transfer", "args": {**transfer, "amount": float(money)}}]
    data = {"accounts": [{"id": ident, "balance": balance, "version": v}
                         for ident, balance in ((a, money * 2), (b, money), (c, 0))], "calls": calls}
    q = _task("typed-tool-trace", "Tool Using", seed, variant,
        "Execute this trace of a local account API in order. No external tools are needed. "
        "Each argument is a scalar or {ref: earlier zero-based call index, field: top-level "
        "result field}. Reference objects must have exactly ref and field, ref must be a "
        "nonnegative integer (not a Boolean/float), and field a string. Resolve ALL references "
        "before selecting the tool: any malformed reference, missing field or non-earlier "
        "reference returns {status: bad_ref}. Only read and transfer exist; "
        "after reference resolution an unknown tool returns unknown_tool. read requires "
        "exactly account:string. transfer requires exactly src:string, dst:string, amount:integer>0, "
        "src_version:integer>=0, dst_version:integer>=0, key:nonempty string. "
        "Booleans and floating numbers are not integers. No coercion or extra keys; violations "
        "return schema BEFORE idempotency lookup. read of an absent ID returns missing, otherwise "
        "{status: read, balance: integer, version: integer}. For a valid transfer, check the "
        "successful-call cache FIRST. Same key and identical fully resolved arguments returns "
        "the ORIGINAL applied receipt without any state change, even when its versions are "
        "now stale. Same cached key with different arguments returns key_conflict. Otherwise "
        "check, in this order: absent account (missing), equal src/dst (same_account), either "
        "version mismatch (stale), insufficient source balance (funds). Errors do not change "
        "state or populate the cache. Success atomically subtracts/adds amount, increments "
        "BOTH versions by one, caches the resolved args plus {status: applied, src_version: "
        "new integer, dst_version: new integer}, and returns that receipt. Error results "
        "contain ONLY status. Report results in call order, accounts as {id,balance,version} "
        "sorted by ID, cached_keys sorted lexically, and applied_count counting actual "
        "state-changing successful calls, excluding cached replays. A failed attempt does "
        "not reserve its key. References see the returned receipt, not the current state.",
        data, trace_answer(data))
    # Balance trace adjudication, persistent effects and replay accounting.
    for item in q.rubric:
        if item["group"] == "results":
            item["dimension"] = "execution"
            # Equal credit per call regardless of whether its result is a
            # one-field error or a receipt with version fields.
            index = int(item["id"].split("[")[1].split("]")[0])
            item["weight"] = 1 / len(data["calls"]) / len(_json_leaf_paths(q.expected["value"]["results"][index]))
            if not item["id"].endswith(".status"):
                item["depends_on"] = [f"json:results[{index}].status"]
    return q


def load_evidence_questions(seed, variant):
    # Paired cases share their generated inputs, except the selected observation.
    digest = hashlib.sha256(f"v13:{seed}:evidence".encode()).digest()
    evidence_rng = random.Random(int.from_bytes(digest, "big"))
    digest = hashlib.sha256(f"v13:{seed}:tools".encode()).digest()
    tool_rng = random.Random(int.from_bytes(digest, "big"))
    return [evidence_case(evidence_rng, seed, variant), tool_case(tool_rng, seed, variant)]
