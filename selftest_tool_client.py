"""Native tool-call assembly, wire defects and rejected request fields."""

import json
import unittest
from unittest.mock import patch

from app.benchmarking.llm_client import ChatClient, ClientConfig
from selftest_client import Response

CONFIG = ClientConfig("http://fake.invalid/v1", "unused", "fake", max_retries=1)
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def tool_event(calls, finish=None, content=None):
    delta = {"tool_calls": calls}
    if content is not None:
        delta["content"] = content
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def fragment(index, *, id=None, name=None, arguments=None, kind="function"):
    item = {"index": index}
    if id is not None:
        item["id"] = id
        item["type"] = kind
    function = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    if function:
        item["function"] = function
    return item


class ToolClientTests(unittest.TestCase):
    def chat(self, response, **kwargs):
        client = ChatClient(CONFIG)
        try:
            with patch.object(client.session, "post", return_value=response) as post:
                result = client.complete_chat(
                    [{"role": "user", "content": "weather?"}], tools=TOOLS, **kwargs)
                return (*result, post.call_args.kwargs["json"])
        finally:
            client.session.close()

    def test_incremental_arguments_and_request_fields(self):
        response = Response([
            tool_event([fragment(0, id="call_1", name="get_weather", arguments="")]),
            tool_event([fragment(0, arguments='{"ci')]),
            tool_event([fragment(0, arguments='ty": "Brno"}')]),
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 7}},
            b"[DONE]",
        ])
        message, usage, metrics, payload = self.chat(
            response, tool_choice="auto", parallel_tool_calls=False)
        self.assertTrue(metrics.ok)
        self.assertEqual(payload["tools"], TOOLS)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertIs(payload["parallel_tool_calls"], False)
        self.assertNotIn("response_format", payload)
        self.assertEqual(len(message.tool_calls), 1)
        call = message.tool_calls[0]
        self.assertEqual((call.id, call.name, call.type), ("call_1", "get_weather", "function"))
        self.assertEqual(json.loads(call.arguments), {"city": "Brno"})
        self.assertEqual(message.finish_reason, "tool_calls")
        self.assertEqual(message.wire_defects, [])
        self.assertEqual(message.text, "")
        self.assertIsNotNone(metrics.ttft_ms)
        self.assertEqual(metrics.stream_chunks, 3)
        self.assertEqual(usage.completion_tokens, 7)

    def test_interleaved_parallel_calls_and_multibyte_arguments(self):
        encoded = '{"city": "Plzeň"}'
        response = Response([
            tool_event([fragment(0, id="a", name="get_weather", arguments=encoded[:12])]),
            tool_event([fragment(1, id="b", name="get_weather", arguments='{"city":')]),
            tool_event([fragment(0, arguments=encoded[12:]), fragment(1, arguments=' "Ústí"}')],
                       finish="tool_calls"),
            b"[DONE]",
        ])
        message, _, metrics, _ = self.chat(response)
        self.assertTrue(metrics.ok)
        self.assertEqual([json.loads(c.arguments)["city"] for c in message.tool_calls],
                         ["Plzeň", "Ústí"])
        self.assertEqual([c.id for c in message.tool_calls], ["a", "b"])
        self.assertEqual(message.wire_defects, [])

    def test_framing_defects_are_recorded_not_raised(self):
        response = Response([
            tool_event([{"function": {"name": "get_weather", "arguments": '{"city":"A"}'}}]),
            tool_event([fragment(1, id="x", name="get_weather", arguments={"city": "B"})]),
            tool_event([fragment(1, name="other_name")], finish="tool_calls"),
            tool_event([fragment(1, arguments="")]),
            b"[DONE]",
        ])
        message, _, metrics, _ = self.chat(response)
        self.assertTrue(metrics.ok)
        codes = [d["code"] for d in message.wire_defects]
        for code in ("tool_index_invalid", "tool_arguments_not_string", "tool_name_changed",
                     "tool_delta_after_finish", "tool_id_missing"):
            self.assertIn(code, codes)
        self.assertEqual(json.loads(message.tool_calls[1].arguments), {"city": "B"})

    def test_duplicate_ids_and_blocking_calls(self):
        data = {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [
                {"id": "same", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city":"A"}'}},
                {"id": "same", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city":"B"}'}},
            ]}}], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
        message, _, metrics, payload = self.chat(Response(data=data), stream=False)
        self.assertTrue(metrics.ok)
        self.assertNotIn("stream", payload)
        self.assertEqual([c.index for c in message.tool_calls], [0, 1])
        self.assertEqual([d["code"] for d in message.wire_defects], ["tool_id_duplicate"])
        self.assertEqual(message.content, "")

    def test_markup_in_content_is_preserved_for_grading(self):
        leaked = '<tool_call>{"name": "get_weather", "arguments": {"city": "Brno"}}</tool_call>'
        message, _, metrics, _ = self.chat(Response([
            {"choices": [{"index": 0, "delta": {"content": leaked}, "finish_reason": "stop"}]},
            b"[DONE]"]))
        self.assertTrue(metrics.ok)
        self.assertEqual(message.tool_calls, [])
        self.assertEqual(message.content, leaked)

    def test_rejected_tool_choice_reports_status_without_dropping_fields(self):
        body = json.dumps({"object": "error", "message": '"auto" tool choice requires '
                           '--enable-auto-tool-choice and --tool-call-parser to be set',
                           "type": "BadRequestError", "code": 400})
        client = ChatClient(ClientConfig("http://fake.invalid/v1", "unused", "fake", max_retries=3))
        try:
            with patch.object(client.session, "post",
                              return_value=Response(status=400, text=body)) as post:
                message, _, metrics = client.complete_chat(
                    [{"role": "user", "content": "x"}], tools=TOOLS, tool_choice="auto")
        finally:
            client.session.close()
        self.assertFalse(metrics.ok)
        self.assertEqual(metrics.http_status, 400)
        self.assertIn("enable-auto-tool-choice", metrics.error)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(message.tool_calls, [])

    def test_tool_history_with_null_content_estimates_prompt(self):
        history = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city":"Brno"}'}}]},
            {"role": "tool", "tool_call_id": "a", "content": '{"temp": 12}'},
        ]
        client = ChatClient(CONFIG)
        try:
            with patch.object(client.session, "post", return_value=Response([
                    {"choices": [{"index": 0, "delta": {"content": "12 degrees"},
                                  "finish_reason": "stop"}]}, b"[DONE]"])):
                message, usage, metrics = client.complete_chat(history, tools=TOOLS)
        finally:
            client.session.close()
        self.assertTrue(metrics.ok)
        self.assertEqual(message.text, "12 degrees")
        self.assertTrue(usage.prompt_tokens_estimated)
        self.assertGreater(usage.prompt_tokens, 0)

    def test_text_api_is_unchanged_for_tool_only_answers(self):
        client = ChatClient(CONFIG)
        try:
            with patch.object(client.session, "post", return_value=Response([
                    tool_event([fragment(0, id="a", name="get_weather", arguments="{}")],
                               finish="tool_calls"), b"[DONE]"])):
                text, _, metrics = client.complete_messages([{"role": "user", "content": "x"}])
        finally:
            client.session.close()
        self.assertTrue(metrics.ok)
        self.assertEqual(text, "")
        self.assertEqual(metrics.finish_reason, "tool_calls")


if __name__ == "__main__":
    unittest.main()
