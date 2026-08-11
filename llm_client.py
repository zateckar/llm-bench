"""A single instrumented OpenAI-compatible chat client.

The CLI runner and the web runner previously had two near-identical copies of the
request code, which meant retry behaviour and (now) timing measurement could
drift between them. Everything goes through :class:`ChatClient` so latency is
measured the same way no matter who is calling.

Streaming is used when available because it is the only way to separate
*time to first token* (prefill + queueing) from *decode throughput*. When the
server does not support streaming - or the caller opts out - only end-to-end
latency is reported and ``ttft_ms`` stays ``None``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import requests

from models import RequestMetrics, TokenUsage

logger = logging.getLogger(__name__)

# Rough characters-per-token used only for prompt-size estimates in the perf
# suite when the server does not report prompt_tokens.
CHARS_PER_TOKEN = 4.0


@dataclass
class ClientConfig:
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.0
    seed: int | None = None
    timeout: float = 180.0
    stream: bool = True
    max_retries: int = 3
    retry_delay: float = 2.0
    # Ask for usage in the final streaming chunk. Some gateways reject the
    # field, so the client drops it and retries once if that happens.
    request_stream_usage: bool = True
    extra_headers: dict = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


class ChatClient:
    def __init__(self, config: ClientConfig):
        self.config = config
        self._stream_usage_supported = config.request_stream_usage
        self._streaming_supported = config.stream
        self.session = requests.Session()

    # -- public API ---------------------------------------------------------

    @property
    def streaming_supported(self) -> bool:
        """False once the endpoint has rejected a streaming request."""
        return self._streaming_supported

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        max_tokens: int | None = None,
        stream: bool | None = None,
        retries: int | None = None,
    ) -> tuple[str, TokenUsage, RequestMetrics]:
        """Send one chat completion and return (text, usage, metrics).

        Never raises for transport problems: a failed call comes back as
        ``("[API ERROR: ...]", TokenUsage(), RequestMetrics(ok=False, ...))`` so
        the caller can record it as an infrastructure error rather than a wrong
        answer.
        """
        cfg = self.config
        want_stream = cfg.stream if stream is None else stream
        want_stream = want_stream and self._streaming_supported
        attempts_allowed = cfg.max_retries if retries is None else retries
        attempts_allowed = max(1, attempts_allowed)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = "unknown error"
        for attempt in range(attempts_allowed):
            started = time.perf_counter()
            try:
                if want_stream:
                    text, usage, ttft = self._stream_once(messages, max_tokens)
                else:
                    text, usage, ttft = self._blocking_once(messages, max_tokens)
            except _RetryableStatus as e:
                # Keep the response body in the error string: downstream
                # detection (e.g. context-window overflows) pattern-matches
                # text that only exists there.
                last_error = str(e)
                if e.status == 400 and want_stream and self._stream_usage_supported:
                    # A gateway that rejects stream_options: drop it and retry
                    # immediately without consuming a backoff cycle.
                    logger.info("Endpoint rejected stream_options; disabling it")
                    self._stream_usage_supported = False
                    continue
                if (
                    e.status in (400, 404, 501)
                    and want_stream
                    and "stream" in e.body.lower()
                ):
                    # Only an endpoint actually complaining about streaming
                    # should turn streaming off for good; an unrelated 400 is
                    # just a request error.
                    logger.info("Endpoint rejected streaming; falling back to blocking")
                    self._streaming_supported = False
                    want_stream = False
                    continue
                if not e.retryable:
                    # A permanent 4xx (401, 403, 413, ...) will fail the same
                    # way on every attempt, so don't pay the backoff.
                    return self._error(last_error, attempt + 1)
                if attempt < attempts_allowed - 1:
                    self._sleep_backoff(attempt, last_error)
                    continue
                return self._error(last_error, attempt + 1)
            except Exception as e:  # noqa: BLE001 - transport errors of any kind
                last_error = f"{type(e).__name__}: {e}"
                if attempt < attempts_allowed - 1:
                    self._sleep_backoff(attempt, last_error)
                    continue
                return self._error(last_error, attempt + 1)

            latency_ms = (time.perf_counter() - started) * 1000.0
            metrics = RequestMetrics(
                latency_ms=latency_ms,
                ttft_ms=ttft,
                completion_tokens=usage.completion_tokens,
                prompt_tokens=usage.prompt_tokens,
                ok=True,
                attempts=attempt + 1,
                streamed=want_stream,
            )
            return text, usage, metrics

        return self._error(last_error, attempts_allowed)

    # -- internals ----------------------------------------------------------

    def _headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.config.extra_headers)
        return headers

    def _payload(self, messages: list[dict], max_tokens: int | None, stream: bool) -> dict:
        cfg = self.config
        payload: dict = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": int(max_tokens or cfg.max_tokens),
            "temperature": cfg.temperature,
        }
        if cfg.seed is not None:
            payload["seed"] = cfg.seed
        if stream:
            payload["stream"] = True
            if self._stream_usage_supported:
                payload["stream_options"] = {"include_usage": True}
        return payload

    def _blocking_once(
        self, messages: list[dict], max_tokens: int | None
    ) -> tuple[str, TokenUsage, float | None]:
        resp = self.session.post(
            self.config.endpoint,
            json=self._payload(messages, max_tokens, stream=False),
            headers=self._headers(),
            timeout=self.config.timeout,
        )
        if resp.status_code >= 400:
            raise _RetryableStatus(resp.status_code, resp.text[:200])
        data = resp.json()
        text = _extract_message_text(data.get("choices") or [{}])
        usage_raw = data.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
        )
        if not usage.completion_tokens and text:
            usage.completion_tokens = _estimate_tokens(text)
        return text, usage, None

    def _stream_once(
        self, messages: list[dict], max_tokens: int | None
    ) -> tuple[str, TokenUsage, float | None]:
        started = time.perf_counter()
        resp = self.session.post(
            self.config.endpoint,
            json=self._payload(messages, max_tokens, stream=True),
            headers=self._headers(),
            timeout=self.config.timeout,
            stream=True,
        )
        if resp.status_code >= 400:
            body = resp.text[:200]
            resp.close()
            raise _RetryableStatus(resp.status_code, body)

        chunks: list[str] = []
        reasoning_chunks: list[str] = []
        ttft: float | None = None
        delta_count = 0
        usage = TokenUsage()

        try:
            for line in resp.iter_lines(decode_unicode=True):
                if time.perf_counter() - started > self.config.timeout * 2:
                    # ``timeout`` bounds each socket op, not the whole stream;
                    # a server trickling one byte per minute otherwise holds
                    # the iterator open forever.
                    resp.close()
                    raise _StreamDeadlineExceeded(
                        f"stream deadline exceeded after {self.config.timeout * 2:.0f}s"
                    )
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                usage_raw = event.get("usage")
                if usage_raw:
                    usage = TokenUsage(
                        prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
                        completion_tokens=int(usage_raw.get("completion_tokens") or 0),
                    )

                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                    if piece:
                        if ttft is None:
                            ttft = (time.perf_counter() - started) * 1000.0
                        chunks.append(piece)
                        delta_count += 1
                    elif reasoning:
                        # Reasoning tokens still count as generation work, and
                        # the first one is the real time-to-first-token.
                        if ttft is None:
                            ttft = (time.perf_counter() - started) * 1000.0
                        reasoning_chunks.append(reasoning)
                        delta_count += 1
        finally:
            resp.close()

        text = "".join(chunks).strip()
        if not text:
            text = "".join(reasoning_chunks).strip()
        if not usage.completion_tokens:
            # Most servers emit one token per delta; fall back to that, then to
            # a character-based estimate. Marked as an estimate by the caller.
            usage.completion_tokens = delta_count or _estimate_tokens(text)
        if not usage.prompt_tokens:
            usage.prompt_tokens = _estimate_tokens(
                "".join(m.get("content", "") for m in messages)
            )
        return text, usage, ttft

    def _sleep_backoff(self, attempt: int, reason: str) -> None:
        wait = self.config.retry_delay * (2 ** attempt)
        logger.info("Request failed (%s); retrying in %.0fs", reason, wait)
        time.sleep(wait)

    @staticmethod
    def _error(message: str, attempts: int) -> tuple[str, TokenUsage, RequestMetrics]:
        return (
            f"[API ERROR: {message}]",
            TokenUsage(),
            RequestMetrics(ok=False, error=message, attempts=attempts),
        )


class _RetryableStatus(Exception):
    def __init__(self, status: int, body: str = ""):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body
        # Only statuses that may succeed on a later attempt are worth paying
        # exponential backoff for; other 4xx are permanent failures.
        self.retryable = status >= 500 or status in (408, 409, 425, 429)


class _StreamDeadlineExceeded(Exception):
    """The response stream stayed open past the total wall-clock deadline."""


def _extract_message_text(choices: list[dict]) -> str:
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        # Reasoning models sometimes return only a reasoning field.
        content = message.get("reasoning") or message.get("reasoning_content") or ""
    if isinstance(content, list):
        # Some gateways return content as a list of parts.
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return (content or "").strip()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))
