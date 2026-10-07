#!/usr/bin/env python3
"""Offline tests for the native tool-calling / structured-output suite.

An oracle client answers every task correctly through native ``tool_calls``
and ``response_format``; targeted corruptions must each lose the criterion
that describes them. Native simulations are driven by the same action policy
as their text-protocol twins and must reach the same environment verdict.
"""

import copy
from dataclasses import replace
import json
import unittest

from app.benchmarking import tool_suite
from app.benchmarking.interactive_tasks import make_tasks, run_interaction
from app.benchmarking.models import ChatMessage, RequestMetrics, TokenUsage, ToolCall
from app.benchmarking.quality_execution import execute_question
from app.benchmarking.quality_report import make_report
from app.benchmarking.quality_suite import load_questions, suite_hash
from app.benchmarking.llm_client import ClientConfig
from app.benchmarking.suites import get_suite

QUESTIONS = tool_suite.load_questions()
BY_ID = {q.id: q for q in QUESTIONS}


def ok_metrics(finish, stream=True):
    return RequestMetrics(latency_ms=10, ttft_ms=2, prompt_tokens=50, completion_tokens=10,
                          finish_reason=finish, streamed=stream)


def reply(content="", calls=(), defects=()):
    tool_calls = [ToolCall(index=i, id=f"call_{i}_{abs(hash(name)) % 997}", name=name,
                           arguments=args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
                           type="function")
                  for i, (name, args) in enumerate(calls)]
    finish = "tool_calls" if tool_calls else "stop"
    return ChatMessage(text=content, content=content, tool_calls=tool_calls,
                       finish_reason=finish, wire_defects=list(defects))


def tool_turns(messages):
    """Number of assistant turns already answered."""
    return sum(m["role"] == "assistant" for m in messages)


def full_arguments(q, name, arguments):
    """Expected arguments plus placeholders for required free-text fields."""
    schema = next(t["function"]["parameters"] for t in q.request["tools"] if t["function"]["name"] == name)
    out = copy.deepcopy(arguments)
    for key in schema.get("required") or []:
        out.setdefault(key, "Placeholder summary of the request")
    return out


def oracle_plan(q):
    """Turn-by-turn ideal assistant messages for a scripted or structured task."""
    if q.interaction["kind"] == "structured":
        return [reply(json.dumps(q.expected["json"]["value"], ensure_ascii=False))]
    spec = q.expected
    turns = []
    one_per_turn = spec.get("max_calls_per_turn") == 1
    for step in spec.get("steps") or []:
        calls = [(c["name"], full_arguments(q, c["name"], c["arguments"])) for c in step["calls"]]
        if one_per_turn:
            turns.extend(reply(calls=[call]) for call in calls)
        else:
            turns.append(reply(calls=calls))
    final = spec.get("final")
    if final is not None or not spec.get("steps"):
        facts = [group[0] for group in (final or {}).get("contains") or []]
        turns.append(reply("Answer: " + "; ".join(facts) if facts else "I cannot check that here."))
    return turns


class PlanClient:
    """Plays a fixed list of assistant messages, one per request."""

    def __init__(self, plan, *, rejected=None):
        self.plan = list(plan)
        self.requests = []
        self.rejected = rejected

    def complete_chat(self, messages, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), **kwargs})
        if self.rejected:
            metrics = RequestMetrics(ok=False, error=self.rejected, http_status=400, attempts=1)
            return ChatMessage(), TokenUsage(), metrics
        message = self.plan[min(tool_turns(messages), len(self.plan) - 1)]
        return message, TokenUsage(50, 10), ok_metrics(message.finish_reason, kwargs.get("stream", True))


def run(q, plan, **kwargs):
    return execute_question(q, PlanClient(plan, **kwargs))


def criteria(result):
    return {c.criterion_id: c for c in result.evaluation.criteria}


def failed(result):
    return {c.criterion_id for c in result.evaluation.criteria if c.status == "fail"}


# --- Native simulation policies, shared with the text protocol ---------------------

def document_policy(params):
    resources = params.get("resources") or [{"id": params["doc"], "limit": params["target"]}]
    for spec in resources:
        while True:
            got = yield "get_document", {"id": spec["id"]}
            body = copy.deepcopy(got["body"])
            body.update(limit=spec["limit"], labels=sorted(set(body["labels"]) | {"red"}))
            result = yield "put_document", {"id": spec["id"], "if_match": got["etag"], "body": body}
            if result.get("ok"):
                break


def careless_document_policy(params):
    """Writes the stale body after a 412: loses the concurrent update."""
    resources = params.get("resources") or [{"id": params["doc"], "limit": params["target"]}]
    for spec in resources:
        got = yield "get_document", {"id": spec["id"]}
        body = copy.deepcopy(got["body"])
        body.update(limit=spec["limit"], labels=sorted(set(body["labels"]) | {"red"}))
        result = yield "put_document", {"id": spec["id"], "if_match": got["etag"], "body": body}
        if not result.get("ok"):
            fresh = yield "get_document", {"id": spec["id"]}
            yield "put_document", {"id": spec["id"], "if_match": fresh["etag"], "body": body}


def payment_policy(params):
    args = {"order": params["order"], "amount": params["amount"], "key": params["key"]}
    result = yield "charge", args
    if result.get("error") == "timeout":
        result = yield "lookup_payment", {"key": params["key"]}
        if result.get("status") == "absent":
            result = yield "charge", args
    yield "record_receipt", {"order": result["order"], "payment": result["payment"], "amount": result["amount"]}


def preview_policy(params):
    rows, cursor = [], None
    while True:
        page = yield "list_objects", {"cursor": cursor}
        rows.extend(page["rows"])
        cursor = page["next"]
        if cursor is None:
            break
    latest = {}
    for row in rows:
        if row["id"] not in latest or row["version"] > latest[row["id"]]["version"]:
            latest[row["id"]] = row
    for ident, row in sorted(latest.items()):
        if row["tenant"] == params["tenant"] and not row["pinned"] and row["expires_at"] <= params["now"]:
            yield "delete_object", {"id": ident, "if_version": row["version"], "dry_run": True}


def eager_preview_policy(params):
    """Acts on page one before reconciling page two."""
    page = yield "list_objects", {"cursor": None}
    for row in page["rows"]:
        if row["tenant"] == params["tenant"] and not row["pinned"] and row["expires_at"] <= params["now"]:
            yield "delete_object", {"id": row["id"], "if_version": row["version"], "dry_run": True}


def permission_policy(params):
    rows, cursor = [], None
    while True:
        page = yield "list_permissions", {"cursor": cursor}
        rows.extend(page["rows"])
        cursor = page["next"]
        if cursor is None:
            break
    target = params["target_resource"]
    version = max(r["version"] for r in rows if r["resource"] == target)
    yield "set_permission", {"resource": target, "scopes": ["read", "write"], "if_version": version}


POLICIES = {"document": document_policy, "payment": payment_policy,
            "preview": preview_policy, "permission": permission_policy}


class PolicyDriver:
    def __init__(self, policy, params):
        self.generator = policy(params)
        self.started = False

    def next_action(self, observation):
        try:
            action = next(self.generator) if not self.started else self.generator.send(observation)
        except StopIteration:
            return None
        self.started = True
        return action


class TextPolicyClient:
    def __init__(self, policy, params):
        self.driver = PolicyDriver(policy, params)

    def complete_messages(self, messages, **kwargs):
        last = messages[-1]["content"]
        observation = (json.loads(last.removeprefix("Simulated tool result: "))
                       if last.startswith("Simulated tool result: ") else None)
        action = self.driver.next_action(observation)
        text = json.dumps({"done": True} if action is None else {"tool": action[0], "args": action[1]})
        return text, TokenUsage(50, 10), ok_metrics("stop")


class NativePolicyClient:
    def __init__(self, policy, params):
        self.driver = PolicyDriver(policy, params)

    def complete_chat(self, messages, **kwargs):
        last = messages[-1]
        observation = json.loads(last["content"]) if last["role"] == "tool" else None
        action = self.driver.next_action(observation)
        message = reply("Done: the task is complete.") if action is None else reply(calls=[action])
        return message, TokenUsage(50, 10), ok_metrics(message.finish_reason)


class SuiteShapeTests(unittest.TestCase):
    def test_inventory_is_deterministic_and_valid(self):
        self.assertEqual(len(QUESTIONS), 96)
        self.assertEqual(tool_suite.validate_suite(QUESTIONS), [])
        self.assertEqual(suite_hash(QUESTIONS), suite_hash(tool_suite.load_questions()))
        categories = {q.category for q in QUESTIONS}
        self.assertEqual(len(categories), 6)
        for q in QUESTIONS:
            self.assertEqual(q.metadata["protocol"], "native-tools-v1")
            self.assertEqual(q.metadata["scope"], "capability")
            self.assertTrue(q.request)
        languages = {q.metadata["language"] for q in QUESTIONS}
        self.assertEqual(languages, {"en", "cs", "de"})

    def test_rigorous_suite_identity_unchanged(self):
        rigorous = load_questions()
        self.assertEqual(len(rigorous), 347)
        self.assertEqual(suite_hash(rigorous), "fa1e19790c3d84f0")
        self.assertTrue(all(q.request is None for q in rigorous))

    def test_native_prompts_do_not_describe_the_text_protocol(self):
        for q in QUESTIONS:
            if q.interaction["kind"] == "simulation":
                self.assertNotIn("done=true", q.prompt)
                self.assertNotIn("Tools:", q.prompt)
                self.assertNotIn('{"tool"', q.system_prompt)


class OracleTests(unittest.TestCase):
    def test_oracle_passes_every_scripted_and_structured_task(self):
        for q in QUESTIONS:
            if q.interaction["kind"] == "simulation":
                continue
            with self.subTest(q.id):
                result = run(q, oracle_plan(q))
                self.assertEqual(result.outcome, "pass", (result.detail, failed(result)))
                self.assertEqual(result.achievement_score, 1.0)
                self.assertEqual(result.evaluation.contract_score, 1.0)

    def test_oracle_passes_every_native_simulation(self):
        for q in QUESTIONS:
            if q.interaction["kind"] != "simulation":
                continue
            params = q.interaction["environment"]
            with self.subTest(q.id):
                result = execute_question(q, NativePolicyClient(POLICIES[params["kind"]], params))
                self.assertEqual(result.outcome, "pass", (result.detail, failed(result)))

    def test_request_fields_and_transport_reach_the_client(self):
        q = BY_ID["T2-tool-choice-named-s19-v01"]
        client = PlanClient(oracle_plan(q))
        execute_question(q, client)
        self.assertEqual(client.requests[0]["tool_choice"],
                         {"type": "function", "function": {"name": "create_ticket"}})
        self.assertEqual(len(client.requests[0]["tools"]), len(tool_suite.BASE_TOOLS))
        blocking = next(q for q in QUESTIONS if q.metadata["transport"] == "blocking")
        client = PlanClient(oracle_plan(blocking))
        execute_question(blocking, client)
        self.assertFalse(client.requests[0]["stream"])
        structured = next(q for q in QUESTIONS if q.interaction["kind"] == "structured")
        client = PlanClient(oracle_plan(structured))
        execute_question(structured, client)
        self.assertIn("response_format", client.requests[0])
        self.assertNotIn("tools", client.requests[0])

    def test_tool_results_are_sent_back_with_call_ids(self):
        q = next(q for q in QUESTIONS if q.metadata["family"] == "T4-dependent-chain")
        client = PlanClient(oracle_plan(q))
        result = execute_question(q, client)
        self.assertEqual(result.outcome, "pass")
        second = client.requests[1]["messages"]
        assistant, tool_message = second[-2], second[-1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], assistant["tool_calls"][0]["id"])
        self.assertIn("manager_id", json.loads(tool_message["content"])["matches"][0])

    def test_report_for_the_suite_records_its_identity(self):
        sample = [q for q in QUESTIONS if q.interaction["kind"] != "simulation"][:12]
        results = [run(q, oracle_plan(q)) for q in sample]
        config = ClientConfig("http://x", "k", "m")
        report = make_report(results, config, suite=get_suite("tool-conformance"))
        self.assertEqual(report["suite"]["name"], "tool-conformance")
        self.assertEqual(report["protocol"]["revision"], tool_suite.REVISION)
        self.assertEqual(report["protocol"]["execution"]["revision"], "native-tools-v1")
        self.assertIn("native_tools", report["protocol"]["client"])
        self.assertEqual(report["summary"]["passes"], len(sample))


class CorruptionTests(unittest.TestCase):
    def selection(self, transport="stream"):
        return next(q for q in QUESTIONS if q.metadata["family"] == "T1-typed-arguments"
                    and q.metadata["transport"] == transport)

    def calendar_call(self, q, **changes):
        expected = q.expected["steps"][0]["calls"][0]
        args = full_arguments(q, "create_calendar_event", expected["arguments"])
        args.update(changes)
        return args

    def test_unparseable_arguments(self):
        q = self.selection()
        result = run(q, [reply(calls=[("create_calendar_event", '{"title": "x", ')])])
        self.assertIn("arguments-json", failed(result))
        self.assertEqual(result.outcome, "formatting")
        self.assertEqual(criteria(result)["arguments-schema"].status, "not_evaluated")

    def test_unknown_tool(self):
        q = self.selection()
        result = run(q, [reply(calls=[("book_meeting", self.calendar_call(q))])])
        self.assertIn("tool-known", failed(result))
        self.assertIn("call-1", failed(result))

    def test_schema_violation_without_content_error(self):
        q = self.selection()
        result = run(q, [reply(calls=[("create_calendar_event", self.calendar_call(q, notes="extra"))])])
        self.assertEqual(failed(result), {"arguments-schema"})
        self.assertEqual(result.outcome, "formatting")
        self.assertEqual(result.achievement_score, 1.0)

    def test_wrong_argument_value(self):
        q = self.selection()
        result = run(q, [reply(calls=[("create_calendar_event", self.calendar_call(q, duration_minutes=61))])])
        self.assertEqual(failed(result), {"call-1:duration_minutes"})
        self.assertEqual(result.outcome, "task_failure")
        self.assertLess(result.achievement_score, 1.0)

    def test_unordered_attendees_accepted(self):
        q = self.selection()
        args = self.calendar_call(q)
        args["attendees"] = list(reversed(args["attendees"]))
        self.assertEqual(run(q, [reply(calls=[("create_calendar_event", args)])]).outcome, "pass")

    def test_markup_leak_and_lost_call(self):
        q = self.selection()
        leaked = '<tool_call>{"name": "create_calendar_event", "arguments": {}}</tool_call>'
        result = run(q, [reply(leaked)])
        self.assertIn("no-markup-leak", failed(result))
        self.assertIn("call-1", failed(result))

    def test_wire_defects(self):
        q = self.selection()
        call = ("create_calendar_event", self.calendar_call(q))
        framing = run(q, [reply(calls=[call], defects=[{"code": "tool_delta_invalid", "detail": ""}])])
        self.assertEqual(failed(framing), {"framing"})
        ids = run(q, [reply(calls=[call], defects=[{"code": "tool_id_missing", "detail": ""}])])
        self.assertEqual(failed(ids), {"call-ids"})

    def test_inconsistent_finish_reason_is_not_mandatory(self):
        q = self.selection()
        message = reply(calls=[("create_calendar_event", self.calendar_call(q))])
        message.finish_reason = "stop"
        result = run(q, [message])
        self.assertEqual(failed(result), {"finish-reason"})
        self.assertEqual(result.outcome, "pass")

    def test_unneeded_tool_call(self):
        q = next(q for q in QUESTIONS if q.metadata["family"] == "T2-no-tool-needed")
        plan = [reply(calls=[("get_weather", {"city": "Brno", "unit": "celsius"})]), *oracle_plan(q)]
        result = run(q, plan)
        self.assertIn("no-call", failed(result))

    def test_tool_choice_none_ignored(self):
        q = next(q for q in QUESTIONS if q.metadata["family"] == "T2-tool-choice-none")
        result = run(q, [reply(calls=[("get_weather", {"city": "Brno", "unit": "celsius"})]),
                         reply("It is sunny.")])
        self.assertIn("tool-choice", failed(result))

    def test_parallel_disabled_but_parallel_calls(self):
        q = next(q for q in QUESTIONS if q.metadata["family"] == "T3-parallel-disabled")
        calls = [(c["name"], c["arguments"]) for c in q.expected["steps"][0]["calls"]]
        facts = [g[0] for g in q.expected["final"]["contains"]]
        result = run(q, [reply(calls=calls), reply("; ".join(facts))])
        self.assertEqual(failed(result), {"parallel-control"})

    def test_parallel_calls_split_across_turns(self):
        q = next(q for q in QUESTIONS if q.metadata["family"] == "T3-parallel-mixed")
        calls = [(c["name"], c["arguments"]) for c in q.expected["steps"][0]["calls"]]
        facts = [g[0] for g in q.expected["final"]["contains"]]
        q = replace(q, metadata={**q.metadata, "max_turns": 3})
        result = run(q, [reply(calls=[calls[0]]), reply(calls=[calls[1]]), reply("; ".join(facts))])
        self.assertEqual(failed(result), {"parallel-1"})

    def test_ungrounded_final_answer(self):
        q = next(q for q in QUESTIONS if q.metadata["family"] == "T4-result-grounding")
        plan = oracle_plan(q)
        plan[-1] = reply("It is in stock at Mladá Boleslav, 12 pieces.")
        result = run(q, plan)
        self.assertTrue({"answer-2"} <= failed(result))
        self.assertEqual(result.outcome, "task_failure")

    def test_structured_output_fence_and_schema(self):
        q = next(q for q in QUESTIONS if q.metadata["family"] == "T6-schema-extraction")
        value = q.expected["json"]["value"]
        fenced = run(q, [reply("```json\n" + json.dumps(value) + "\n```")])
        self.assertIn("json-document", failed(fenced))
        self.assertIn("schema-valid", failed(fenced))
        bad = dict(value, mileage_km=float(value["mileage_km"]))
        typed = run(q, [reply(json.dumps(bad))])
        self.assertIn("schema-valid", failed(typed))

    def test_feature_rejection_is_scored_and_separate(self):
        q = self.selection()
        result = run(q, [], rejected='HTTP 400: "auto" tool choice requires --enable-auto-tool-choice')
        self.assertEqual(result.outcome, "feature_rejected")
        self.assertTrue(result.is_scored)
        self.assertFalse(result.passed)
        self.assertEqual(failed(result), {"feature-accepted"})
        outage = run(q, [], rejected="HTTP 400: context length exceeded")
        self.assertEqual(outage.outcome, "endpoint_error")
        self.assertFalse(outage.is_scored)


class SimulationParityTests(unittest.TestCase):
    """The native re-host must grade a trajectory exactly as the text protocol does."""

    def twin(self, q):
        text = {t.id: t for t in make_tasks(q.metadata["seed"], q.metadata["variant"])}
        return text[q.metadata["text_twin"]]

    def assert_parity(self, q, policy):
        params = q.interaction["environment"]
        native = execute_question(q, NativePolicyClient(policy, params))
        twin = self.twin(q)
        twin = replace(twin, metadata={**twin.metadata, "total_output_budget": 10**6})
        text = run_interaction(twin, TextPolicyClient(policy, copy.deepcopy(twin.interaction)))
        self.assertEqual(native.passed, text.passed, (native.detail, text.detail))
        self.assertEqual(native.diagnostics["final_state"], text.diagnostics["final_state"])
        self.assertEqual(native.diagnostics["violations"], text.diagnostics["violations"])
        self.assertEqual(native.diagnostics["calls"], text.diagnostics["calls"])
        return native, text

    def test_correct_policies_pass_in_both_protocols(self):
        for q in QUESTIONS:
            if q.interaction["kind"] == "simulation":
                with self.subTest(q.id):
                    native, _ = self.assert_parity(q, POLICIES[q.interaction["environment"]["kind"]])
                    self.assertTrue(native.passed)

    def test_flawed_policies_fail_in_both_protocols(self):
        flawed = {"document": careless_document_policy, "preview": eager_preview_policy}
        checked = 0
        for q in QUESTIONS:
            kind = q.interaction.get("environment", {}).get("kind")
            if kind in flawed:
                with self.subTest(q.id):
                    native, _ = self.assert_parity(q, flawed[kind])
                    self.assertFalse(native.passed)
                    checked += 1
        self.assertGreater(checked, 0)

    def test_simulation_wire_criteria_apply(self):
        q = next(q for q in QUESTIONS if q.interaction.get("environment", {}).get("kind") == "payment")
        params = q.interaction["environment"]

        class Leaky(NativePolicyClient):
            def complete_chat(self, messages, **kwargs):
                message, usage, metrics = super().complete_chat(messages, **kwargs)
                if message.tool_calls:
                    message.content = "[TOOL_CALLS]"
                return message, usage, metrics

        result = execute_question(q, Leaky(payment_policy, params))
        self.assertEqual(failed(result), {"no-markup-leak"})
        self.assertEqual(result.outcome, "formatting")


if __name__ == "__main__":
    unittest.main()
