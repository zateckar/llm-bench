#!/usr/bin/env python3
"""Offline integration, differential, property, and regression tests for quality v3.

No LLM endpoints or live user databases are used. The reference tokenizer may
download its verified vocabulary once unless TIKTOKEN_CACHE_DIR is prepopulated.
"""

import copy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import random
import re
import unittest
from unittest.mock import patch

from challenge_oracles import CODE as ORIGINAL_CODE
from independent_oracles import CODE, code_cases
from interactive_tasks import Environment, make_tasks, run_interaction
from llm_client import ChatClient, ClientConfig
from models import Question, RequestMetrics, Result, TokenUsage
from quality_execution import score_response
from quality_report import result_record, summarize
from quality_suite import load_questions
from test_loader import load_all_tests

ROOT = Path(__file__).parent


class FakeAgent:
    """Adaptive client using only messages/tool observations, not hidden state."""

    def __init__(self, question):
        self.q = question
        self.history = []
        self.actions = []

    def complete_messages(self, messages, **kwargs):
        self.history.append(copy.deepcopy(messages))
        prompt = messages[1]["content"]
        responses = [
            json.loads(m["content"].removeprefix("Simulated tool result: "))
            for m in messages[2:]
            if m["role"] == "user"
        ]
        last = responses[-1] if responses else None
        action = {"done": True}
        if "get_document" in prompt:
            ident = re.search(r"on document (d\d+)", prompt)[1]
            target = int(re.search(r"limit to (\d+)", prompt)[1])
            if last is None or last.get("status") == 412:
                action = {"tool": "get_document", "args": {"id": ident}}
            elif "body" in last:
                body = copy.deepcopy(last["body"])
                body.update(limit=target, labels=sorted(set(body["labels"]) | {"red"}))
                action = {
                    "tool": "put_document",
                    "args": {"id": ident, "if_match": last["etag"], "body": body},
                }
        elif "lookup_payment" in prompt:
            order, amount = re.search(r"order (o\d+) exactly (\d+)", prompt).groups()
            key = "pay:" + order
            if last is None or last.get("status") == "absent":
                action = {
                    "tool": "charge",
                    "args": {"order": order, "amount": int(amount), "key": key},
                }
            elif last.get("error") == "timeout":
                action = {"tool": "lookup_payment", "args": {"key": key}}
            elif last.get("status") == "succeeded":
                action = {
                    "tool": "record_receipt",
                    "args": {k: last[k] for k in ["order", "payment", "amount"]},
                }
        else:
            if last is None:
                action = {"tool": "list_objects", "args": {"cursor": None}}
            elif last.get("next"):
                action = {"tool": "list_objects", "args": {"cursor": last["next"]}}
            else:
                now = int(re.search(r"Current time is (\d+)", prompt)[1])
                tenant = re.search(r"tenant (t\d+)", prompt)[1]
                objects = {}
                for response in responses:
                    for obj in response.get("rows", []):
                        if (
                            obj["id"] not in objects
                            or obj["version"] > objects[obj["id"]]["version"]
                        ):
                            objects[obj["id"]] = obj
                previewed = {r.get("would_delete") for r in responses}
                for ident, obj in sorted(objects.items()):
                    if (
                        ident not in previewed
                        and obj["tenant"] == tenant
                        and not obj["pinned"]
                        and obj["expires_at"] <= now
                    ):
                        action = {
                            "tool": "delete_object",
                            "args": {"id": ident, "if_version": obj["version"], "dry_run": True},
                        }
                        break
        self.actions.append(action)
        return (
            json.dumps(action),
            TokenUsage(100, 20),
            RequestMetrics(
                latency_ms=10,
                ttft_ms=2,
                prompt_tokens=100,
                completion_tokens=20,
                finish_reason="stop",
            ),
        )


class SuiteTests(unittest.TestCase):
    def test_differential_code_oracles_and_properties(self):
        cases = code_cases(random.Random(900), count=40)
        base = {q.id: q for q in load_all_tests(ROOT / "tests")}
        for ident, fn in CODE.items():
            for fixture in base[ident].expected:
                self.assertEqual(fn(*fixture["args"]), fixture["expected"], ident)
            for args in cases[ident]:
                self.assertEqual(
                    fn(*copy.deepcopy(args)),
                    ORIGINAL_CODE[ident](*copy.deepcopy(args)),
                    (ident, args),
                )
        # Metamorphic properties: translation, duplication, irrelevant events.
        for intervals in [[], [[0, 3], [1, 5], [5, 7]], [[-4, -1], [-2, 1]]]:
            result = CODE["AC2-02"](intervals)
            self.assertEqual(
                CODE["AC2-02"]([[a + 17, b + 17] for a, b in intervals]),
                [[a + 17, b + 17, c] for a, b, c in result],
            )
            self.assertEqual(CODE["AC2-02"](intervals * 2), [[a, b, c * 2] for a, b, c in result])
        events = [["1", "BANK", "a", 8], ["2", "a", "b", 3]]
        self.assertEqual(CODE["AC2-07"](events), CODE["AC2-07"](events + events))
        self.assertEqual(CODE["AC2-04"]([["a", 1, 2], ["b", 0, 9]], [["a", 2]]), [2])

    def test_expanded_fixtures_execute_in_real_sandbox(self):
        from evaluators import eval_code_exec

        expanded = [q for q in load_questions() if q.id in CODE]
        for q in expanded:
            self.assertGreater(len(q.expected), 30)
            score, detail = eval_code_exec(
                "```python\n" + inspect.getsource(ORIGINAL_CODE[q.id]) + "\n```", q.expected
            )
            self.assertEqual(score, 1, (q.id, detail))


    def test_multi_resource_and_permission_trajectories(self):
        class ScriptClient:
            def __init__(self, actions):
                self.actions = iter(actions)

            def complete_messages(self, *args, **kwargs):
                return (
                    json.dumps(next(self.actions)),
                    TokenUsage(10, 5),
                    RequestMetrics(latency_ms=2),
                )

        for seed in (19, 23):
            for variant in (0, 1):
                for q in make_tasks(seed, variant)[3:]:
                    env = Environment(q.interaction)
                    actions = []

                    def call(tool, args):
                        actions.append({"tool": tool, "args": copy.deepcopy(args)})
                        return env.call(tool, args)

                    if q.interaction["kind"] == "document":
                        for spec in q.interaction["resources"]:
                            initial = call("get_document", {"id": spec["id"]})
                            body = {
                                **initial["body"],
                                "labels": sorted(set(initial["body"]["labels"]) | {"red"}),
                                "limit": spec["limit"],
                            }
                            stale = call(
                                "put_document",
                                {"id": spec["id"], "if_match": initial["etag"], "body": body},
                            )
                            self.assertEqual(stale["status"], 412)
                            fresh = call("get_document", {"id": spec["id"]})
                            body = {
                                **fresh["body"],
                                "labels": sorted(set(fresh["body"]["labels"]) | {"red"}),
                                "limit": spec["limit"],
                            }
                            self.assertTrue(
                                call(
                                    "put_document",
                                    {"id": spec["id"], "if_match": fresh["etag"], "body": body},
                                )["ok"]
                            )
                    else:
                        call("list_permissions", {"cursor": None})
                        second = call("list_permissions", {"cursor": "page2"})
                        target = next(
                            r
                            for r in second["rows"]
                            if r["resource"] == q.interaction["target_resource"]
                        )
                        self.assertTrue(
                            call(
                                "set_permission",
                                {
                                    "resource": target["resource"],
                                    "scopes": ["read", "write"],
                                    "if_version": target["version"],
                                },
                            )["ok"]
                        )
                    self.assertTrue(env.verdict()["success"])
                    result = run_interaction(q, ScriptClient(actions + [{"done": True}]))
                    self.assertTrue(result.passed, (q.id, result.detail))
                    self.assertEqual(result.diagnostics["unnecessary_calls"], 0)
                    env = Environment(q.interaction)
                    if q.interaction["kind"] == "permission":
                        env.call(
                            "set_permission",
                            {"resource": "r-control", "scopes": ["read", "write"], "if_version": 5},
                        )
                        self.assertIn("unauthorized_permission_change", env.verdict()["violations"])

    def test_adaptive_agents_and_transcripts(self):
        branches = set()
        for seed in range(8):
            for q in make_tasks(seed, 0)[:3]:
                agent = FakeAgent(q)
                result = run_interaction(q, agent)
                self.assertEqual(result.score, 1, (q.id, result.detail, result.response))
                self.assertEqual(result.outcome, "pass")
                self.assertEqual(result.diagnostics["unnecessary_calls"], 0)
                self.assertEqual(result.tokens.prompt_tokens, 100 * len(agent.actions))
                self.assertEqual(result.metrics.attempts, len(agent.actions))
                self.assertGreater(len(agent.history[-1]), len(agent.history[0]))
                if q.interaction["kind"] == "payment":
                    branches.add(q.interaction["commit_on_timeout"])
        self.assertEqual(branches, {True, False})

    def test_violations_cannot_be_repaired_into_a_pass(self):
        q = make_tasks(3, 0)[2]
        env = Environment(q.interaction)
        env.call("delete_object", {"id": "k2", "if_version": 8, "dry_run": False})
        env.call("list_objects", {"cursor": None})
        env.call("list_objects", {"cursor": "page2"})
        for ident, version in [("k2", 8), ("k5", 3)]:
            env.call("delete_object", {"id": ident, "if_version": version, "dry_run": True})
        self.assertFalse(env.verdict()["success"])
        self.assertEqual(env.verdict()["violations"], ["unauthorized_delete"])

    def test_stale_write_loses_no_concurrent_data(self):
        q = make_tasks(3, 0)[0]
        env = Environment(q.interaction)
        p = q.interaction
        old = env.call("get_document", {"id": p["doc"]})
        self.assertEqual(
            env.call(
                "put_document", {"id": p["doc"], "if_match": old["etag"], "body": old["body"]}
            )["status"],
            412,
        )
        current = env.call("get_document", {"id": p["doc"]})
        body = {**old["body"], "limit": p["target"], "labels": ["blue", "red"]}
        env.call("put_document", {"id": p["doc"], "if_match": current["etag"], "body": body})
        self.assertIn("lost_concurrent_update", env.verdict()["violations"])

    def test_bounds_malformed_and_cancelled(self):
        q = make_tasks(3, 0)[0]

        class Client:
            def __init__(self, text):
                self.text = text

            def complete_messages(self, *args, **kwargs):
                return self.text, TokenUsage(1, 1), RequestMetrics(ok=True)

        for text, outcome in [
            ("oops", "formatting"),
            ('{"done":1}', "formatting"),
            ('{"done":true}', "task_failure"),
        ]:
            result = run_interaction(q, Client(text))
            self.assertEqual(result.outcome, outcome)
            self.assertEqual(result.score, 0)
        result = run_interaction(q, Client('{"tool":"get_document","args":{"id":"missing"}}'))
        self.assertEqual(
            len([x for x in result.diagnostics["transcript"] if x["role"] == "assistant"]), 16
        )
        cancelled = run_interaction(q, Client("oops"), cancelled=lambda: True)
        self.assertEqual(cancelled.outcome, "cancelled")
        self.assertFalse(cancelled.is_scored)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.q = Question(
            "q", "Reasoning", "reply", "json_match", {"value": {"x": 1}}, metadata={"family": "f"}
        )
        self.config = ClientConfig("http://fake.invalid", "unused", "fake")

    def result(self, score=1, **kwargs):
        return Result(
            self.q,
            "{}",
            score,
            tokens=TokenUsage(100, 20),
            metrics=RequestMetrics(latency_ms=50, prompt_tokens=100, completion_tokens=20),
            **kwargs,
        )

    def test_failure_attribution_and_denominators(self):
        from models import CategoryResult

        cases = [
            ('{"x":1}', RequestMetrics(), "pass", True),
            ("no json", RequestMetrics(), "formatting", True),
            ('{"x":2}', RequestMetrics(), "task_failure", True),
            ('{"x":1}', RequestMetrics(finish_reason="length"), "truncation", True),
            ("", RequestMetrics(ok=False, error="HTTP 500"), "endpoint_error", False),
        ]
        results = []
        for response, metrics, outcome, scored in cases:
            r = score_response(self.q, response, TokenUsage(), metrics)
            self.assertEqual((r.outcome, r.is_scored), (outcome, scored))
            results.append(r)
        q = replace(self.q, metadata={"context_tokens": 8192})
        r = score_response(
            q, "", TokenUsage(), RequestMetrics(ok=False, error="HTTP 400 context_length_exceeded")
        )
        self.assertEqual(r.outcome, "unsupported_context")
        self.assertFalse(r.is_scored)
        broken = score_response(
            replace(self.q, evaluator="absent"), "{}", TokenUsage(), RequestMetrics()
        )
        self.assertEqual(broken.outcome, "evaluator_error")
        self.assertFalse(broken.is_scored)
        summary = summarize([result_record(r) for r in results + [r, broken]])
        self.assertEqual(summary["scored"], 4)
        self.assertEqual(len(CategoryResult("test", results + [r, broken]).scored), 4)

    def test_client_finish_reason_blocking_and_streamed(self):
        class Response:
            status_code = 200

            def json(self):
                return {
                    "choices": [{"message": {"content": '{"x":1}'}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }

            def iter_lines(self, **kwargs):
                for value in [
                    {"choices": [{"delta": {"content": '{"x":1}'}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "length"}]},
                    {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                ]:
                    yield ("data: " + json.dumps(value)).encode()
                    yield b""
                yield b"data: [DONE]"
                yield b""

            def iter_content(self, **kwargs):
                for line in self.iter_lines():
                    yield line + b"\n"

            def close(self):
                pass

        for streaming in [False, True]:
            client = ChatClient(replace(self.config, stream=streaming))
            with patch.object(client.session, "post", return_value=Response()):
                text, tokens, metrics = client.complete("test")
            self.assertEqual(metrics.finish_reason, "length")
            self.assertEqual(tokens.prompt_tokens, 10)
            self.assertFalse(metrics.prompt_tokens_estimated)
            self.assertEqual(score_response(self.q, text, tokens, metrics).outcome, "truncation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
