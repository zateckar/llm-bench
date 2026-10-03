"""Offline client protocol and measurement regressions, including real HTTP framing."""

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import Mock, patch

from llm_client import ChatClient, ClientConfig, _parse_usage, _sse_payloads
from models import RequestMetrics, TokenUsage
from quality_execution import run_quality
from models import Question


CONFIG = ClientConfig("http://fake.invalid/v1", "unused", "fake", max_retries=1)


class Response:
    def __init__(self, events=(), *, status=200, data=None, text="", raw=None):
        self.status_code, self.text, self.data = status, text, data
        self.raw = raw
        self.events = events
        self.close = Mock()
        self.read_options = None

    def json(self):
        return self.data

    def iter_lines(self, **kwargs):
        if self.raw is not None:
            yield from self.raw
            return
        for event in self.events:
            yield b"data: " + (event if isinstance(event, bytes) else json.dumps(event).encode())
            yield b""

    def iter_content(self, **kwargs):
        self.read_options = kwargs
        for line in self.iter_lines():
            yield line + b"\n"


def event(content=None, reasoning=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class ClientTests(unittest.TestCase):
    def complete(self, response, config=CONFIG):
        client = ChatClient(config)
        try:
            with patch.object(client.session, "post", return_value=response):
                return client.complete("input")
        finally:
            client.session.close()

    def test_reasoning_and_final_text_in_one_event_count_together(self):
        response = Response([event("answer", "reasoning " * 20, "stop"), b"[DONE]"])
        text, usage, metrics = self.complete(response)
        self.assertEqual(text, "answer")
        self.assertGreater(usage.completion_tokens, len(text) // 4)
        self.assertEqual(metrics.stream_chunks, 1)
        self.assertEqual(response.read_options, {"chunk_size": 1})
        response.close.assert_called_once()

    def test_partial_streams_and_error_events_are_not_successes(self):
        for response in (Response([event("correct-looking answer")]),
                         Response([b"{invalid", event("answer", finish="stop")]),
                         Response([{"error": {"message": "overloaded"}}, b"[DONE]"]),
                         Response([{"choices": [{"index": 1, "delta": {"content": "wrong choice"}}]}, b"[DONE]"]),
                         Response(raw=[b'data: {"choices":[]}', b'data: [DONE]']),
                         Response(raw=[b'data: \xff', b''])):
            _, _, metrics = self.complete(response)
            self.assertFalse(metrics.ok)
            response.close.assert_called_once()

    def test_terminal_finish_reason_is_sufficient_without_done_sentinel(self):
        text, _, metrics = self.complete(Response([event("answer", finish="stop")]))
        self.assertTrue(metrics.ok)
        self.assertEqual(text, "answer")

    def test_multiline_sse_and_comments(self):
        response = Response(raw=[b": heartbeat", b"", b'data: {"choices":',
                                b'data: [{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
                                b"", b"data: [DONE]", b""])
        text, _, metrics = self.complete(response)
        self.assertTrue(metrics.ok, metrics.error)
        self.assertEqual(text, "ok")

    def test_crlf_utf8_and_bom_across_arbitrary_chunks(self):
        data = ('\ufeffdata: {"choices":\r\ndata: [{"delta":{"content":"žluťoučký"},'
                '"finish_reason":"stop"}]}\r\n\r\ndata: [DONE]\r\n\r\n').encode()
        for width in (1, 2, 7, 65536):
            response = Response()
            response.iter_content = lambda **kwargs: (data[i:i + width] for i in range(0, len(data), width))
            text, _, metrics = self.complete(response)
            self.assertTrue(metrics.ok, (width, metrics.error))
            self.assertEqual(text, "žluťoučký")

    def test_chunked_connections_return_to_pool_after_done(self):
        ports = []
        body = b"data: " + json.dumps(event("ok", finish="stop")).encode() + b"\n\ndata: [DONE]\n\n"
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def do_POST(self):
                ports.append(self.client_address[1])
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n")
                self.wfile.flush()
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = ChatClient(replace(CONFIG, base_url=f"http://127.0.0.1:{server.server_port}", timeout=2))
        try:
            for _ in range(2):
                _, _, metrics = client.complete("input")
                self.assertTrue(metrics.ok, metrics.error)
            self.assertEqual(len(ports), 2)
            self.assertEqual(len(set(ports)), 1, "Warm connection was discarded")
        finally:
            client.session.close()
            server.shutdown()
            server.server_close()
            thread.join(5)

    def test_bad_usage_is_estimated_and_reported_zero_is_preserved(self):
        for malformed in (None, [], {"prompt_tokens": -1, "completion_tokens": "nan"},
                          {"prompt_tokens": True, "completion_tokens": 1.5}):
            usage = _parse_usage(malformed)
            self.assertTrue(usage.prompt_tokens_estimated)
            self.assertTrue(usage.completion_tokens_estimated)
        response = Response([event("answer", finish="stop"),
                             {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}])
        _, usage, metrics = self.complete(response)
        self.assertTrue(metrics.ok)
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens), (0, 0))
        self.assertFalse(usage.completion_tokens_estimated)
        self.assertFalse(usage.prompt_tokens_estimated)

    def test_partial_usage_updates_preserve_prior_reported_counts(self):
        response = Response([event("answer", finish="stop"),
                             {"choices": [], "usage": {"prompt_tokens": 123, "prompt_cache_hit_tokens": 80}},
                             {"choices": [], "usage": {"completion_tokens": 17}}, b"[DONE]"])
        _, usage, metrics = self.complete(response)
        self.assertTrue(metrics.ok)
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens), (123, 17, 80))
        self.assertFalse(usage.prompt_tokens_estimated or usage.completion_tokens_estimated)

    def test_http_error_body_read_failure_closes_response(self):
        class BrokenResponse:
            status_code = 503
            close = Mock()
            @property
            def text(self):
                raise OSError("interrupted error body")
        response = BrokenResponse()
        _, _, metrics = self.complete(response)
        self.assertFalse(metrics.ok)
        response.close.assert_called_once()

    def test_blocking_response_validation_and_cleanup(self):
        config = replace(CONFIG, stream=False)
        for data in (None, {}, {"choices": []}, {"choices": [{}]}, {"error": {"message": "failed"}}):
            response = Response(data=data)
            _, _, metrics = self.complete(response, config)
            self.assertFalse(metrics.ok)
            response.close.assert_called_once()
        response = Response(data={"choices": [{"message": {"content": [{"text": "ok"}]}, "finish_reason": "stop"}]})
        text, _, metrics = self.complete(response, config)
        self.assertTrue(metrics.ok, metrics.error)
        self.assertEqual(text, "ok")
        response.close.assert_called_once()

    def test_blocking_estimates_include_delivered_reasoning(self):
        response = Response(data={"choices": [{"message": {"content": "ok", "reasoning": "reason " * 50},
                                               "finish_reason": "stop"}]})
        text, usage, metrics = self.complete(response, replace(CONFIG, stream=False))
        self.assertTrue(metrics.ok)
        self.assertEqual(text, "ok")
        self.assertGreater(usage.completion_tokens, 50)
        self.assertTrue(usage.completion_tokens_estimated)

    def test_retry_success_includes_backoff_and_prior_attempt(self):
        clock = [0.0]
        client = ChatClient(replace(CONFIG, max_retries=2, retry_delay=2))
        failed = Response(status=503, text="busy")
        successful = Response([event("answer", finish="stop")])
        responses = iter([failed, successful])
        def post(*args, **kwargs):
            clock[0] += 1
            return next(responses)
        def sleep(seconds):
            clock[0] += seconds
        try:
            with (patch.object(client.session, "post", side_effect=post),
                  patch("llm_client.time.perf_counter", side_effect=lambda: clock[0]),
                  patch("llm_client.time.sleep", side_effect=sleep)):
                _, _, metrics = client.complete("input")
            self.assertTrue(metrics.ok)
            self.assertEqual(metrics.attempts, 2)
            self.assertEqual(metrics.latency_ms, 4000)
            self.assertEqual(metrics.ttft_ms, 4000)
            self.assertEqual(metrics.successful_attempt_latency_ms, 1000)
            failed.close.assert_called_once()
            successful.close.assert_called_once()
        finally:
            client.session.close()

    def test_negotiation_changes_only_implicated_fields(self):
        responses = [Response(status=400, text="Unsupported stream_options"),
                     Response([event("ok", finish="stop")])]
        client = ChatClient(replace(CONFIG, max_retries=2))
        try:
            with patch.object(client.session, "post", side_effect=responses) as post:
                _, _, metrics = client.complete("test")
            self.assertTrue(metrics.ok)
            payloads = [call.kwargs["json"] for call in post.call_args_list]
            self.assertIn("stream_options", payloads[0])
            self.assertNotIn("stream_options", payloads[1])
            self.assertTrue(all(p["stream"] and p["temperature"] == 0 for p in payloads))
        finally:
            client.session.close()

    def test_deadline_applies_to_sse_heartbeats(self):
        response = Response(raw=[b": heartbeat"])
        with patch("llm_client.time.perf_counter", return_value=5):
            with self.assertRaisesRegex(Exception, "deadline"):
                list(_sse_payloads(response, 0, 1))

    def test_canonical_quality_disables_repetition_intervention(self):
        clients = []
        class Client:
            def __init__(self, config):
                self.config, self.session = config, Mock()
                clients.append(self)
            def complete(self, *args, **kwargs):
                return "ok", TokenUsage(), RequestMetrics()
        q = Question("q", "Test", "return ok", "exact_match", "ok")
        with patch("llm_client.ChatClient", Client):
            results, _, _ = run_quality([q], replace(CONFIG, detect_repetition=True))
        self.assertTrue(results[0].passed)
        self.assertFalse(clients[0].config.detect_repetition)

    def test_nonchunked_http_delivers_short_event_before_body_finishes(self):
        # A live local socket verifies framing that mocks cannot: the 512-byte
        # default iterator would hold this event until the response body ended.
        released = threading.Event()
        dispatched = threading.Event()
        first = b"data: " + json.dumps(event("ok")).encode() + b"\n\n"
        last = b"data: " + json.dumps(event(finish="stop")).encode() + b"\n\ndata: [DONE]\n\n"
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(first) + len(last)))
                self.end_headers()
                self.wfile.write(first)
                self.wfile.flush()
                released.wait(5)
                self.wfile.write(last)
                self.wfile.flush()
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        client = ChatClient(replace(CONFIG, base_url=f"http://127.0.0.1:{server.server_port}", timeout=3))
        from llm_client import _sse_payloads as original
        def observe(*args):
            for payload in original(*args):
                dispatched.set()
                yield payload
        result = []
        try:
            with patch("llm_client._sse_payloads", side_effect=observe):
                worker = threading.Thread(target=lambda: result.append(client.complete("input")))
                worker.start()
                self.assertTrue(dispatched.wait(2), "First SSE event waited for body completion")
                self.assertTrue(worker.is_alive())
                released.set()
                worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertTrue(result[0][2].ok, result)
        finally:
            released.set()
            client.session.close()
            server.shutdown()
            server.server_close()
            server_thread.join(5)


if __name__ == "__main__":
    unittest.main()
