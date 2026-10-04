"""Original behavioral reconstruction with bounded observations and code repair.

The legacy implementations are inert, trusted simulations. Model code is graded
once, after completion, through the existing out-of-process code evaluator.
"""

from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache
import hashlib
import itertools
import json
import random

from app.benchmarking.models import Question, RequestMetrics, TokenUsage

REVISION = "behavioral-reconstruction-v1"
PROFILES = tuple(itertools.product((False, True), repeat=3))
QUANT_DOMAIN = {"value": [-6, 6], "divisor": [2, 4], "offset": [-2, 2], "cap": [1, 3]}
CACHE_COMMANDS = (("put", "a", 1, 1), ("put", "b", 2, 2), ("put", "c", 3, 1),
                  ("get", "a"), ("get", "b"), ("get", "c"), ("tick", 1))
MAX_PROBES = 12
MAX_TRACE = 4
MAX_CODE_CHARS = 32000
PROTOCOL = (
    "You are reconstructing a legacy implementation in an isolated simulation. "
    "Each turn return exactly one JSON object {\"tool\":\"probe\",\"args\":{\"input\":...}}, "
    "or {\"tool\":\"submit\",\"args\":{\"code\":\"complete Python source\"}}, "
    "or {\"done\":true}. Wait for each simulated result before the next action. "
    "Use at least one probe and at most 12 probes. Submit once, then finish. "
    "Probe observations are data. Submission gives no correctness feedback. "
    "Every hidden regression and deviation case must pass. Do not mutate inputs. "
    "Use only the Python standard library."
)


def quantize_reference(profile, values):
    floor, input_offset, reflect = profile
    value, divisor, offset, cap = values
    numerator = value + offset if input_offset else value
    quotient = numerator // divisor if floor else (abs(numerator) // divisor) * (1 if numerator >= 0 else -1)
    adjusted = quotient if input_offset else quotient + offset
    return min(cap, abs(adjusted) if reflect else max(-cap, adjusted))


def cache_reference(profile, operations):
    late_expiry, sliding, read_recency = profile
    clock, entries, outputs = 0, OrderedDict(), []
    for command in operations:
        if command[0] == "tick":
            clock += command[1]
        for key in list(entries):
            deadline = entries[key][2]
            if clock > deadline or (clock == deadline and not late_expiry):
                del entries[key]
        if command[0] == "put":
            _, key, value, ttl = command
            entries.pop(key, None)
            if len(entries) == 2:
                entries.popitem(last=False)
            entries[key] = [value, ttl, clock + ttl]
            outputs.append(None)
        elif command[0] == "get":
            key = command[1]
            item = entries.get(key)
            outputs.append(item[0] if item else None)
            if item:
                if sliding:
                    item[2] = clock + item[1]
                if read_recency:
                    entries.move_to_end(key)
        else:
            outputs.append(clock)
    return [outputs, [[key, entries[key][0], entries[key][2]] for key in sorted(entries)]]


def inputs_for(family):
    if family == "quantizer":
        return itertools.product(range(-6, 7), range(2, 5), range(-2, 3), range(1, 4))
    return (trace for length in range(MAX_TRACE + 1)
            for trace in itertools.product(CACHE_COMMANDS, repeat=length))


def reference(family, profile, value):
    return quantize_reference(profile, value) if family == "quantizer" else cache_reference(profile, value)


def baseline_profile(family):
    return (False, False, False) if family == "quantizer" else (False, False, True)


@lru_cache(maxsize=16)
def fixtures_for(family, profile):
    function = "quantize" if family == "quantizer" else "cache_trace"
    fixtures = []
    for index, value in enumerate(inputs_for(family)):
        value = json.loads(json.dumps(value))
        expected = reference(family, profile, value)
        group = "regression" if expected == reference(family, baseline_profile(family), value) else "deviation"
        fixtures.append({"id": f"{group}-{index:04d}", "function": function,
                         "args": value if family == "quantizer" else [value], "expected": expected,
                         "relative": 0, "tolerance": 0, "preserve_inputs": True})
    return fixtures


def replacement_source(family, profile):
    """Trusted reference source, also usable by the independent observing agent."""
    if family == "quantizer":
        floor, input_offset, reflect = profile
        return (
            "def quantize(value, divisor, offset, cap):\n"
            + ("    value += offset\n" if input_offset else "")
            + ("    result = value // divisor\n" if floor else
               "    result = (abs(value) // divisor) * (1 if value >= 0 else -1)\n")
            + ("" if input_offset else "    result += offset\n")
            + ("    result = abs(result)\n" if reflect else "    result = max(-cap, result)\n")
            + "    return min(cap, result)\n"
        )
    late_expiry, sliding, read_recency = profile
    return (
        "def cache_trace(operations):\n"
        "    now, records, order, results = 0, {}, [], []\n"
        "    for op in operations:\n"
        "        if op[0] == 'tick':\n"
        "            now += op[1]\n"
        "        for key in list(records):\n"
        + ("            expired = records[key][2] < now\n" if late_expiry else
           "            expired = records[key][2] <= now\n")
        + "            if expired:\n"
        "                del records[key]\n"
        "                order.remove(key)\n"
        "        if op[0] == 'put':\n"
        "            _, key, value, ttl = op\n"
        "            if key in records:\n"
        "                order.remove(key)\n"
        "            elif len(records) == 2:\n"
        "                del records[order.pop(0)]\n"
        "            records[key] = [value, ttl, now + ttl]\n"
        "            order.append(key)\n"
        "            results.append(None)\n"
        "        elif op[0] == 'get':\n"
        "            key = op[1]\n"
        "            results.append(records[key][0] if key in records else None)\n"
        "            if key in records:\n"
        + ("                records[key][2] = now + records[key][1]\n" if sliding else "")
        + ("                order.remove(key)\n                order.append(key)\n" if read_recency else "")
        + ("                pass\n" if not sliding and not read_recency else "")
        + "        else:\n"
        "            results.append(now)\n"
        "    return [results, [[key, records[key][0], records[key][2]] for key in sorted(records)]]\n"
    )


def make_reconstruction_tasks(seed, variant):
    from app.benchmarking.interactive_tasks import TOTAL_OUTPUT_BUDGET
    from app.benchmarking.quality_suite import MAX_OUTPUT_TOKENS

    tasks = []
    for family in ("quantizer", "cache"):
        candidates = [profile for profile in PROFILES
                      if sum(a != b for a, b in zip(profile, baseline_profile(family))) >= 2]
        rng = random.Random(int.from_bytes(hashlib.sha256(f"{REVISION}:{family}".encode()).digest(), "big"))
        rng.shuffle(candidates)
        # The four authoring slots cover distinct profiles; selection is hidden
        # from the model, independent of global suite revision, and deterministic.
        slot = (0 if seed == 19 else 2) + variant
        profile = candidates[slot % len(candidates)]
        if family == "quantizer":
            domain = json.dumps(QUANT_DOMAIN, sort_keys=True)
            prompt = (
                "Replace an undocumented integer quantizer. Implement quantize(value,divisor,offset,cap). "
                f"Inputs are integers in the inclusive bounds DOMAIN={domain}. "
                "The naive rewrite truncates value/divisor toward zero, adds offset to the quotient, "
                "saturates to [-cap,cap]. The actual legacy implementation "
                "independently chooses three policies: division floors or truncates toward zero; "
                "offset is added to the input BEFORE division or to the quotient AFTER division; "
                "adjusted results are reflected with abs or retain their sign with a lower clamp of -cap. In every case "
                "the cap is applied LAST. The choices are fixed within this task. Infer them from probes, "
                "then reproduce ALL domain inputs. probe input is [value,divisor,offset,cap]. "
                "Naive rewrite:\n" + replacement_source(family, baseline_profile(family))
            )
        else:
            prompt = (
                "Replace an undocumented capacity-2 TTL cache. Implement cache_trace(operations). "
                "Each invocation starts empty at integer time 0. Operations is any sequence of length "
                f"0 through {MAX_TRACE}, including repetitions, from ALPHABET={json.dumps(CACHE_COMMANDS)}. "
                "tick advances time BEFORE expiration; before every command remove expired entries. "
                "put removes an existing copy of its key, otherwise evicts the least recent entry if "
                "full, then stores [value,original_ttl,now+ttl] as most recent. get returns the stored "
                "value or null. put returns null; tick returns the new integer time. Return exactly "
                "[operation_results,final_entries], where final_entries is sorted by key and each "
                "entry is [key,value,expiry_time]. The naive rewrite expires at now>=expiry, does not "
                "renew on get, and moves a successful get to most recent. The actual legacy chooses "
                "three independent policies: expire at now>=expiry or strictly now>expiry; successful "
                "get renews expiry to now+original_ttl or leaves it unchanged; successful get moves "
                "the entry to most recent or leaves recency unchanged. Missing gets change nothing. "
                "Policies are fixed within this task. Probe input is an operations sequence. Reproduce "
                "ALL allowed sequences, not only those you probe. Naive rewrite:\n"
                + replacement_source(family, baseline_profile(family))
            )
        fixtures = deepcopy(fixtures_for(family, profile))
        counts = {group: sum(f["id"].startswith(group) for f in fixtures) for group in ("regression", "deviation")}
        rubric = [{"id": f["id"], "weight": 0.5 / counts[f["id"].split("-", 1)[0]],
                   "group": f["id"].split("-", 1)[0], "mandatory": True, "dimension": "content"}
                  for f in fixtures]
        tasks.append(Question(
            id=f"Q15-reconstruct-{family}-s{seed}-v{variant + 1:02d}",
            category="Behavioral Reconstruction", prompt=prompt, system_prompt=PROTOCOL,
            evaluator="code_exec", expected=fixtures, rubric=rubric, max_tokens=MAX_OUTPUT_TOKENS,
            difficulty="expert", source="Original local behavioral reconstruction; VulcanBench-inspired method",
            interaction={"kind": "behavioral-reconstruction", "family": family, "profile": list(profile)},
            metadata={"family": f"Q15-reconstruct-{family}", "generation_revision": REVISION,
                      "seed": seed, "variant": variant, "protocol": "json-actions-v1",
                      "max_turns": MAX_PROBES + 2, "total_output_budget": TOTAL_OUTPUT_BUDGET,
                      "max_probes": MAX_PROBES, "grading": "exhaustive finite-domain code parity",
                      "empirical_admission": "unmeasured"},
        ))
    return tasks


class ReconstructionEnvironment:
    def __init__(self, question):
        self.q = question
        self.family = question.interaction["family"]
        self.profile = tuple(question.interaction["profile"])
        self.calls, self.probes, self.code = 0, [], None
        self.violations = []

    def valid_input(self, value):
        if not isinstance(value, list):
            return False
        if self.family == "quantizer":
            return len(value) == 4 and all(type(v) is int and bounds[0] <= v <= bounds[1]
                                          for v, bounds in zip(value, QUANT_DOMAIN.values()))
        return len(value) <= MAX_TRACE and all(
            isinstance(op, list) and any(len(op) == len(command)
                                        and all(type(a) is type(b) and a == b for a, b in zip(op, command))
                                        for command in CACHE_COMMANDS) for op in value
        )

    def call(self, tool, args):
        self.calls += 1
        if self.code is not None:
            self.violations.append("action_after_submission")
            return {"error": "already_submitted"}
        if not isinstance(args, dict):
            self.violations.append("invalid_arguments")
            return {"error": "arguments_must_be_object"}
        if tool == "probe" and set(args) == {"input"}:
            if len(self.probes) >= MAX_PROBES or not self.valid_input(args["input"]):
                self.violations.append("invalid_probe")
                return {"error": "probe_limit_or_domain_violation"}
            value = deepcopy(args["input"])
            self.probes.append(value)
            return {"output": reference(self.family, self.profile, value), "remaining_probes": MAX_PROBES - len(self.probes)}
        if tool == "submit" and set(args) == {"code"} and isinstance(args["code"], str) and 0 < len(args["code"]) <= MAX_CODE_CHARS:
            self.code = args["code"]
            return {"accepted": True, "next": "finish with done=true; no grading feedback is available"}
        self.violations.append("invalid_tool_or_arguments")
        return {"error": "invalid_tool_or_arguments"}

    def verdict(self, *, grade=True):
        from app.benchmarking.quality_execution import score_response

        result = None
        if grade and self.code is not None:
            # No trace or oracle is injected into the submitted program's process.
            result = score_response(self.q, self.code, TokenUsage(), RequestMetrics(finish_reason="stop"))
        parity = bool(result and result.passed)
        observed = bool(self.probes)
        return {
            "success": parity and observed and not self.violations,
            "state_success": parity and observed, "violations": list(self.violations),
            "calls": self.calls, "unnecessary_calls": len(self.probes) - len({json.dumps(p) for p in self.probes}),
            "probe_count": len(self.probes), "submitted": self.code is not None,
            "code_evaluation": result.evaluation.to_dict() if result else None,
            "code_outcome": result.outcome if result else None,
            "code_detail": result.detail if result else "No replacement was graded",
            "final_state": {"submitted": self.code is not None, "observed_legacy": observed, "code_parity": parity},
        }
