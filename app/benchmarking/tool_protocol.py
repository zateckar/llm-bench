"""Native tool-calling and structured-output execution (``native-tools-v1``).

Unlike the text JSON-action protocol in ``interactive_tasks``, these tasks send
OpenAI ``tools`` / ``tool_choice`` / ``parallel_tool_calls`` / ``response_format``
fields and read native ``tool_calls``. They therefore evaluate the deployment
(chat template, tool-call parser, guided decoding) as well as the model.

Grading separates two kinds of criteria:

* wire conformance (dimension ``contract``): arguments are a JSON object, the
  tool exists, arguments satisfy the declared schema, call ids are present and
  unique, stream framing is sound and no raw call markup leaks into content;
* behaviour (dimension ``content``): the right calls with the right arguments,
  abstention, ``tool_choice`` and ``parallel_tool_calls`` honoured, and final
  answers grounded in tool results.

Criterion achievement therefore reports tool-use correctness, the contract
score reports wire conformance, and a full pass requires both.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
import re
import unicodedata

from app.benchmarking.evaluators import eval_json_match, strip_think_blocks, values_equal
from app.benchmarking.llm_client import REPETITION_FINISH_REASON, TOOL_CLIENT_REVISION
from app.benchmarking.models import RequestMetrics, Result, TokenUsage
from app.benchmarking.schema_subset import validate

PROTOCOL = "native-tools-v1"
PER_TURN_OUTPUT_TOKENS = 32768
TOTAL_OUTPUT_BUDGET = 131072
NATIVE_EVALUATOR_VERSIONS = {
    "native_tool_use": "1",
    "native_structured_output": "1",
    "native_simulation": "1",
}

# Raw tool-call markup that a working parser would have converted into
# ``tool_calls``. Its presence in content means the call was lost.
LEAK_PATTERNS = [
    ("hermes", re.compile(r"</?tool_call>", re.I)),
    ("mistral", re.compile(r"\[TOOL_CALLS\]")),
    ("llama", re.compile(r"<\|python_tag\|>|<\|eom_id\|>")),
    ("function-tag", re.compile(r"<function=[\w.-]+>|</function>")),
    ("kimi", re.compile(r"<\|tool_calls?_(?:section_)?begin\|>")),
    ("deepseek", re.compile(r"<｜tool▁calls?▁begin｜>|<｜tool▁sep｜>")),
    ("harmony", re.compile(r"<\|channel\|>|to=functions\.")),
    ("qwen-xml", re.compile(r"<tool_call>|<function_call>|<invoke name=", re.I)),
    ("bare-json", re.compile(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:')),
]
REJECTION_MARKERS = re.compile(
    r"tool|function|response_format|json_schema|json_object|guided|structured|parallel_tool_calls",
    re.I,
)


def protocol(suite_revision):
    return {
        "revision": PROTOCOL,
        "suite_revision": suite_revision,
        "per_turn_output_tokens": PER_TURN_OUTPUT_TOKENS,
        "total_output_budget": TOTAL_OUTPUT_BUDGET,
        "followup_recovery": False,
        "history": "assistant tool_calls echoed verbatim (raw arguments); reasoning omitted; "
                   "tool results as role=tool with tool_call_id",
        "missing_call_ids": "synthetic id substituted and recorded as a wire defect",
        "transport_retries": False,
        "request_fields_negotiated": False,
        "tool_client": TOOL_CLIENT_REVISION,
        "evaluator_versions": dict(NATIVE_EVALUATOR_VERSIONS),
    }


def is_feature_rejection(metrics: RequestMetrics) -> bool:
    return (metrics.http_status in (400, 404, 422, 501)
            and bool(REJECTION_MARKERS.search(metrics.error or "")))


def parse_arguments(raw):
    """Arguments must be one JSON object; anything else is unusable."""
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _normalize(text):
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).casefold().strip()


def text_contains(text, alternative):
    """Case- and whitespace-insensitive containment; numbers need boundaries."""
    haystack, needle = _normalize(text), _normalize(alternative)
    if re.fullmatch(r"[-+]?[\d.,\s]+", needle):
        return re.search(rf"(?<![\d.,]){re.escape(needle)}(?![\d])", haystack) is not None
    return needle in haystack


def leaf_paths(value, atomic=frozenset(), prefix=""):
    """Flatten expected arguments into ``(path, value)`` leaves."""
    if prefix in atomic:
        return [(prefix, value)]
    if isinstance(value, dict) and value:
        out = []
        for key, item in value.items():
            out.extend(leaf_paths(item, atomic, f"{prefix}.{key}" if prefix else key))
        return out
    if isinstance(value, list) and value:
        out = []
        for index, item in enumerate(value):
            out.extend(leaf_paths(item, atomic, f"{prefix}[{index}]"))
        return out
    return [(prefix, value)]


_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def value_at(doc, path):
    current = doc
    for name, index in _PATH_TOKEN.findall(path):
        if name:
            if not isinstance(current, dict) or name not in current:
                return False, None
            current = current[name]
        else:
            position = int(index)
            if not isinstance(current, list) or position >= len(current):
                return False, None
            current = current[position]
    return True, current


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def leaf_matches(got, want, path, spec):
    alternatives = [want, *((spec.get("aliases") or {}).get(path, []))]
    unordered = path in set(spec.get("unordered") or [])
    for option in alternatives:
        if unordered and isinstance(got, list) and isinstance(option, list):
            if sorted(map(_canonical, got)) == sorted(map(_canonical, option)):
                return True
        elif isinstance(option, str) and isinstance(got, str):
            if got.strip() == option:
                return True
        elif values_equal(got, option, rel=0, abs_tol=0):
            return True
    return False


def arguments_match(parsed, spec):
    """True when every expected leaf matches (extra optional arguments allowed)."""
    if parsed is None:
        return False
    atomic = frozenset(spec.get("unordered") or [])
    for path, want in leaf_paths(spec.get("arguments") or {}, atomic):
        found, got = value_at(parsed, path)
        if not found or not leaf_matches(got, want, path, spec):
            return False
    return True


def _match_count(parsed, spec):
    if parsed is None:
        return -1
    atomic = frozenset(spec.get("unordered") or [])
    count = 0
    for path, want in leaf_paths(spec.get("arguments") or {}, atomic):
        found, got = value_at(parsed, path)
        count += bool(found and leaf_matches(got, want, path, spec))
    return count


class ScriptedWorld:
    """Deterministic tool replies keyed by tool name and expected arguments."""

    def __init__(self, q):
        self.replies = q.interaction.get("replies") or []
        self.default = q.interaction.get("default") or {"error": "no_matching_record"}
        self.tools = {tool["function"]["name"] for tool in (q.request or {}).get("tools") or []}
        self.calls = 0

    def call(self, name, args):
        self.calls += 1
        if name not in self.tools:
            return {"error": "unknown_tool"}
        if args is None:
            return {"error": "invalid_arguments", "detail": "arguments must be one JSON object"}
        for entry in self.replies:
            if entry["name"] == name and (
                entry.get("arguments") is None or arguments_match(args, entry)
            ):
                return copy.deepcopy(entry["reply"])
        return copy.deepcopy(self.default)


class SimulationWorld:
    """The text-protocol environments, reached through native tool calls."""

    def __init__(self, q):
        from app.benchmarking.interactive_tasks import Environment

        self.env = Environment(q.interaction["environment"])

    def call(self, name, args):
        # Unparseable arguments still count as a call, as in the text protocol.
        return self.env.call(name, args)


def _crit(ident, ok, *, dimension="content", mandatory=True, critical=False,
          depends_on=None, reason=None, evidence=None, status=None):
    return {
        "id": ident,
        "status": status or ("pass" if ok else "fail"),
        "earned": 1.0 if ok and status is None else 0.0,
        "possible": 1.0,
        "dimension": dimension,
        "mandatory": mandatory,
        "critical": critical,
        "depends_on": depends_on or [],
        "reason_code": reason or ("met" if ok else "not_met"),
        "evidence": evidence or {},
    }


def observed_calls(transcript):
    calls = []
    for turn in transcript:
        for call in turn.get("tool_calls") or []:
            calls.append({**call, "turn": turn["turn"], "parsed": parse_arguments(call.get("arguments"))})
    return calls


def wire_criteria(q, transcript, calls):
    tools = {t["function"]["name"]: t["function"].get("parameters") or {}
             for t in (q.request or {}).get("tools") or []}
    forced = (q.request or {}).get("tool_choice")
    criteria = []
    if calls:
        invalid = [c.get("name") for c in calls if c["parsed"] is None]
        criteria.append(_crit("arguments-json", not invalid, dimension="contract",
                              reason="arguments_object" if not invalid else "arguments_not_json_object",
                              evidence={"invalid_calls": invalid[:8]}))
        unknown = [c.get("name") for c in calls if c.get("name") not in tools]
        criteria.append(_crit("tool-known", not unknown, dimension="contract",
                              reason="offered_tool" if not unknown else "unknown_tool",
                              evidence={"unknown": unknown[:8]}))
        errors = []
        for c in calls:
            if c["parsed"] is not None and c.get("name") in tools:
                for error in validate(c["parsed"], tools[c["name"]]):
                    errors.append({"tool": c["name"], **error})
        checked = [c for c in calls if c["parsed"] is not None and c.get("name") in tools]
        criteria.append(_crit("arguments-schema", bool(checked) and not errors, dimension="contract",
                              status=None if checked else "not_evaluated",
                              reason="schema_valid" if checked and not errors else
                              ("schema_violation" if checked else "blocked_by_structure"),
                              evidence={"errors": errors[:8]}))
    defects = [d for turn in transcript for d in turn.get("wire_defects") or []]
    id_defects = [d for d in defects if d["code"].startswith("tool_id")]
    if calls:
        criteria.append(_crit("call-ids", not id_defects, dimension="contract",
                              reason="ids_valid" if not id_defects else "id_defect",
                              evidence={"defects": id_defects[:8]}))
    framing = [d for d in defects if not d["code"].startswith("tool_id")]
    criteria.append(_crit("framing", not framing, dimension="contract",
                          reason="framing_valid" if not framing else "framing_defect",
                          evidence={"defects": framing[:8]}))
    leaks = sorted({name for turn in transcript for name, pattern in LEAK_PATTERNS
                    if pattern.search(turn.get("content") or "")})
    criteria.append(_crit("no-markup-leak", not leaks, dimension="contract",
                          reason="no_leak" if not leaks else "markup_in_content",
                          evidence={"patterns": leaks}))
    finish_problems = []
    for turn in transcript:
        finish = turn.get("finish_reason")
        if turn.get("tool_calls"):
            allowed = {"tool_calls"} | ({"stop"} if forced not in (None, "auto", "none") else set())
        else:
            allowed = {"stop"}
        if finish not in allowed:
            finish_problems.append({"turn": turn["turn"], "finish_reason": finish})
    criteria.append(_crit("finish-reason", not finish_problems, dimension="contract", mandatory=False,
                          reason="consistent" if not finish_problems else "inconsistent_finish_reason",
                          evidence={"turns": finish_problems[:8]}))
    return criteria


def final_text(transcript):
    if not transcript or transcript[-1].get("tool_calls"):
        return ""
    return strip_think_blocks(transcript[-1].get("content") or "").strip()


def behaviour_criteria(q, transcript, calls):
    spec = q.expected or {}
    steps = spec.get("steps") or []
    criteria, used, pointer, number = [], set(), 0, 0
    for step_index, step in enumerate(steps, 1):
        positions = []
        for expected in step["calls"]:
            number += 1
            ident = f"call-{number}"
            candidates = [i for i in range(pointer, len(calls))
                          if i not in used and calls[i].get("name") == expected["name"]]
            best = max(candidates, key=lambda i: (_match_count(calls[i]["parsed"], expected), -i),
                       default=None)
            if best is not None:
                used.add(best)
                positions.append(best)
            criteria.append(_crit(ident, best is not None,
                                  reason="call_made" if best is not None else "call_missing",
                                  evidence={"name": expected["name"],
                                            "observed_turn": calls[best]["turn"] if best is not None else None}))
            atomic = frozenset(expected.get("unordered") or [])
            for path, want in leaf_paths(expected.get("arguments") or {}, atomic):
                found, got = value_at(calls[best]["parsed"], path) if best is not None and calls[best]["parsed"] is not None else (False, None)
                ok = found and leaf_matches(got, want, path, expected)
                criteria.append(_crit(f"{ident}:{path}", ok, depends_on=[ident],
                                      reason="argument_match" if ok else
                                      ("argument_missing" if not found else "argument_mismatch"),
                                      evidence={"path": path}))
        if step.get("same_turn") and len(step["calls"]) > 1:
            turns = {calls[i]["turn"] for i in positions}
            ok = len(positions) == len(step["calls"]) and len(turns) == 1
            criteria.append(_crit(f"parallel-{step_index}", ok,
                                  reason="same_turn" if ok else "split_or_missing",
                                  evidence={"turns": sorted(turns)}))
        if positions:
            pointer = max(positions) + 1
    choice = spec.get("choice")
    if steps and not spec.get("allow_extra_calls"):
        extra = [calls[i].get("name") for i in range(len(calls)) if i not in used]
        criteria.append(_crit("no-extra-calls", not extra,
                              reason="no_extra_calls" if not extra else "extra_calls",
                              evidence={"extra": extra[:8]}))
    if not steps and choice != "none":
        criteria.append(_crit("no-call", not calls, reason="abstained" if not calls else "unneeded_call",
                              evidence={"calls": [c.get("name") for c in calls][:8]}))
    limit = spec.get("max_calls_per_turn")
    if limit:
        per_turn = [len(t.get("tool_calls") or []) for t in transcript]
        ok = all(n <= limit for n in per_turn)
        criteria.append(_crit("parallel-control", ok,
                              reason="limit_respected" if ok else "parallel_calls_despite_disabled",
                              evidence={"calls_per_turn": per_turn}))
    if choice is not None:
        first = transcript[0].get("tool_calls") or [] if transcript else []
        if choice == "none":
            ok = not calls
        elif choice == "required":
            ok = bool(first)
        else:
            ok = bool(first) and all(c.get("name") == choice["name"] for c in first)
        criteria.append(_crit("tool-choice", ok, reason="honored" if ok else "ignored",
                              evidence={"tool_choice": choice}))
    final = spec.get("final")
    if final:
        text = final_text(transcript)
        criteria.append(_crit("final-answer", bool(text),
                              reason="answered" if text else "no_final_answer"))
        for index, group in enumerate(final.get("contains") or [], 1):
            ok = bool(text) and any(text_contains(text, alternative) for alternative in group)
            criteria.append(_crit(f"answer-{index}", ok, depends_on=["final-answer"],
                                  reason="grounded" if ok else "missing_fact",
                                  evidence={"expected_any": group[:4]}))
    return criteria


def structured_criteria(q, transcript):
    content = (transcript[0].get("content") or "") if transcript else ""
    sink = {}
    score, detail = eval_json_match(content, q.expected["json"], _diagnostics=sink)
    criteria = []
    for item in sink.get("criteria") or []:
        # Every requested value is part of the structured-output contract.
        criteria.append({"dimension": "content", "mandatory": True, "critical": False,
                         "depends_on": [], "evidence": {}, **item})
    response_format = (q.request or {}).get("response_format") or {}
    if response_format.get("type") == "json_schema":
        schema = response_format["json_schema"]["schema"]
        doc = parse_arguments(content.strip())
        errors = validate(doc, schema) if doc is not None else [{"path": "$", "error": "not a JSON object"}]
        criteria.append(_crit("schema-valid", not errors, dimension="contract",
                              reason="schema_valid" if not errors else "schema_violation",
                              evidence={"errors": errors[:8]}))
    leaks = sorted({name for name, pattern in LEAK_PATTERNS if pattern.search(content)})
    if leaks:
        criteria.append(_crit("no-markup-leak", False, dimension="contract",
                              reason="markup_in_content", evidence={"patterns": leaks}))
    return criteria, score >= 1, detail


def simulation_criteria(world, completed, outcome):
    verdict = world.env.verdict()
    violations = list(verdict["violations"])
    unnecessary = verdict.get("unnecessary_calls", 0)
    criteria = [
        _crit("final-state", verdict["state_success"],
              reason="state_match" if verdict["state_success"] else "state_mismatch"),
        _crit("authorization", not violations, dimension="safety", critical=True,
              reason="no_violations" if not violations else "unauthorized_action",
              evidence={"violations": violations[:8]}),
        _crit("protocol", completed, dimension="contract",
              reason="completed_protocol" if completed else outcome),
        _crit("call-efficiency", not unnecessary, dimension="efficiency", mandatory=False,
              reason="no_excess_calls" if not unnecessary else "excess_calls",
              evidence={"unnecessary_calls": unnecessary}),
    ]
    return criteria, verdict


def run_native(q, client, cancelled=lambda: False):
    from app.benchmarking.quality_execution import _availability_evaluation, _evaluation_for

    kind = q.interaction["kind"]
    world = ScriptedWorld(q) if kind == "scripted" else SimulationWorld(q) if kind == "simulation" else None
    messages = []
    if q.system_prompt:
        messages.append({"role": "system", "content": q.system_prompt})
    messages.append({"role": "user", "content": q.prompt})
    request = dict(q.request or {})
    stream = q.metadata.get("transport", "stream") == "stream"
    per_turn = min(q.max_tokens or PER_TURN_OUTPUT_TOKENS, PER_TURN_OUTPUT_TOKENS)
    remaining = TOTAL_OUTPUT_BUDGET
    transcript, tokens = [], TokenUsage()
    aggregate = RequestMetrics(attempts=0, streamed=stream)
    cache_reporting = []
    outcome, detail, completed = None, "", False
    for turn in range(1, q.metadata["max_turns"] + 1):
        if cancelled():
            outcome, detail = "cancelled", "Run stopped"
            break
        message, usage, metrics = client.complete_chat(
            messages, **request, max_tokens=min(per_turn, remaining), stream=stream)
        tokens.prompt_tokens += usage.prompt_tokens
        tokens.completion_tokens += usage.completion_tokens
        tokens.cached_tokens += usage.cached_tokens
        tokens.prompt_tokens_estimated |= usage.prompt_tokens_estimated
        tokens.completion_tokens_estimated |= usage.completion_tokens_estimated
        cache_reporting.append(usage.cached_tokens_reported)
        aggregate.latency_ms += metrics.latency_ms
        aggregate.attempts += metrics.attempts
        aggregate.streamed = aggregate.streamed and metrics.streamed
        aggregate.finish_reason = metrics.finish_reason
        aggregate.attempt_diagnostics.extend(metrics.attempt_diagnostics)
        if aggregate.ttft_ms is None and metrics.ttft_ms is not None:
            aggregate.ttft_ms = aggregate.latency_ms - metrics.latency_ms + metrics.ttft_ms
        if not metrics.ok:
            aggregate.ok, aggregate.error, aggregate.http_status = False, metrics.error, metrics.http_status
            outcome = "feature_rejected" if is_feature_rejection(metrics) else "endpoint_error"
            detail = f"Request failed: {metrics.error}"
            transcript.append({"turn": turn, "error": (metrics.error or "")[:2000],
                               "http_status": metrics.http_status})
            break
        remaining -= usage.completion_tokens
        record = {
            "turn": turn,
            "content": message.content,
            "reasoning_chars": len(message.reasoning),
            "finish_reason": message.finish_reason,
            "streamed": metrics.streamed,
            "tool_calls": [asdict(call) for call in message.tool_calls],
            "wire_defects": message.wire_defects,
            "tool_results": [],
        }
        transcript.append(record)
        if metrics.finish_reason == REPETITION_FINISH_REASON:
            outcome, detail = "repetition", f"Turn {turn}: degenerate repetition"
            break
        if metrics.finish_reason in {"length", "max_tokens"}:
            outcome, detail = "truncation", f"Turn {turn}: output limit reached"
            break
        if not message.tool_calls:
            if not strip_think_blocks(message.text).strip():
                outcome, detail = "missing_answer", f"Turn {turn}: no content and no tool call"
                break
            completed = True
            break
        if world is None:
            break
        echoed = []
        for call in message.tool_calls:
            echoed.append({"id": call.id or f"call_synthetic_{turn}_{call.index}", "type": "function",
                           "function": {"name": call.name or "", "arguments": call.arguments}})
        messages.append({"role": "assistant", "content": message.content or None, "tool_calls": echoed})
        for call, sent in zip(message.tool_calls, echoed):
            reply = world.call(call.name, parse_arguments(call.arguments))
            record["tool_results"].append({"tool_call_id": sent["id"], "name": call.name, "reply": reply})
            messages.append({"role": "tool", "tool_call_id": sent["id"],
                             "content": json.dumps(reply, ensure_ascii=False)})
        if remaining <= 0:
            outcome, detail = "truncation", "Total output budget exhausted"
            break
    aggregate.prompt_tokens, aggregate.completion_tokens = tokens.prompt_tokens, tokens.completion_tokens
    aggregate.prompt_tokens_estimated = tokens.prompt_tokens_estimated
    aggregate.completion_tokens_estimated = tokens.completion_tokens_estimated
    aggregate.cached_tokens = tokens.cached_tokens
    tokens.cached_tokens_reported = bool(cache_reporting) and all(cache_reporting)
    aggregate.cached_tokens_reported = tokens.cached_tokens_reported
    response = json.dumps(transcript, ensure_ascii=False)
    diagnostics = {"transcript": transcript, "protocol": PROTOCOL,
                   "wire_defects": [d for t in transcript for d in t.get("wire_defects") or []]}
    result = Result(q, response, 0.0, detail, tokens, aggregate, diagnostics=diagnostics)
    evaluator_version = NATIVE_EVALUATOR_VERSIONS.get(q.evaluator, "1")

    if outcome == "feature_rejected":
        result.outcome = outcome
        result.evaluation = _evaluation_for(q, score=0.0, full_pass=False, outcome=outcome, sink={
            "criteria": [_crit("feature-accepted", False, dimension="contract", reason="feature_rejected",
                               evidence={"http_status": aggregate.http_status,
                                         "detail": (aggregate.error or "")[:400]})],
            "contract_score": 0.0,
        })
        result.evaluation.evaluator_version = evaluator_version
        return result
    if outcome in {"cancelled", "endpoint_error", "truncation", "repetition", "missing_answer"} and kind != "simulation":
        result.outcome = outcome
        result.evaluation = _availability_evaluation(q, outcome, detail)
        result.evaluation.evaluator_version = evaluator_version
        return result
    if outcome in {"cancelled", "endpoint_error"}:
        result.outcome = outcome
        result.evaluation = _availability_evaluation(q, outcome, detail)
        result.evaluation.evaluator_version = evaluator_version
        return result

    calls = observed_calls(transcript)
    verdict = None
    if kind == "structured":
        content_criteria, content_pass, result.detail = structured_criteria(q, transcript)
        criteria = content_criteria
        full = content_pass
    elif kind == "simulation":
        sim, verdict = simulation_criteria(world, completed, outcome or "turn_budget_exhausted")
        criteria = sim + wire_criteria(q, transcript, calls)
        full = completed and verdict["success"]
        diagnostics.update({k: v for k, v in verdict.items()})
    else:
        criteria = behaviour_criteria(q, transcript, calls) + wire_criteria(q, transcript, calls)
        full = True
    contract = [c for c in criteria if c["dimension"] == "contract" and c["status"] != "not_evaluated"]
    content = [c for c in criteria if c["dimension"] not in {"contract", "availability", "efficiency"}]
    score = sum(c["earned"] for c in content) / len(content) if content else 0.0
    mandatory_failed = [c for c in criteria
                        if (c["mandatory"] or c["critical"]) and c["status"] != "pass"
                        and not (c["status"] == "not_evaluated" and c["dimension"] == "contract")]
    full = full and not mandatory_failed
    if full:
        provisional = "pass"
    elif any(c["dimension"] == "contract" and c["status"] == "fail" and c["mandatory"] for c in criteria):
        provisional = "formatting"
    else:
        provisional = "task_failure"
    if outcome in {"truncation", "repetition", "missing_answer"}:
        provisional = outcome
        full = False
    sink = {"criteria": criteria,
            "contract_score": sum(c["earned"] for c in contract) / len(contract) if contract else None}
    try:
        evaluation = _evaluation_for(q, score=score, full_pass=full, outcome=provisional, sink=sink)
    except ValueError as error:
        result.outcome, result.detail = "evaluator_error", f"Invalid evaluation diagnostics: {error}"
        result.evaluation = _availability_evaluation(q, result.outcome, result.detail)
        return result
    evaluation.evaluator_version = evaluator_version
    result.evaluation = evaluation
    result.score = score
    result.outcome = evaluation.outcome if not evaluation.full_pass else "pass"
    if not result.detail:
        failed = [c.criterion_id for c in evaluation.criteria
                  if c.status != "pass" and (c.mandatory or c.critical)]
        result.detail = ("All criteria met" if evaluation.full_pass
                         else "Unmet: " + ", ".join(failed[:12]) if failed else detail or "Not passed")
    return result
