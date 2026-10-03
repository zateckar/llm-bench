"""A single instrumented OpenAI-compatible chat client.

The CLI runner and the web runner previously had two near-identical copies of the
request code, which meant retry behaviour and (now) timing measurement could
drift between them. Everything goes through :class:`ChatClient` so latency is
measured the same way no matter who is calling.

Streaming exposes time to first delivered content or reasoning. Provider
buffering affects these observations. Without streaming only end-to-end
latency is available and ``ttft_ms`` stays ``None``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

import requests
import urllib3

from models import RequestMetrics, TokenUsage

logger = logging.getLogger(__name__)
CLIENT_PROTOCOL_VERSION = "chat-client-v3"

# Rough characters-per-token used only for prompt-size estimates in the perf
# suite when the server does not report prompt_tokens.
CHARS_PER_TOKEN = 4.0

# --- degenerate repetition -------------------------------------------------
#
# This optional heuristic can stop repeated output early, but repeated code or
# structured data can be legitimate and recovery cannot be ruled out. Canonical
# quality and performance runs disable the intervention; the budget is fixed.
#
# The signal is the fraction of *distinct* word n-grams in the recent tail.
# Exact cycle detection is not enough: real loops drift ("...pattern X true.
# Good. ...pattern Y false. Good."), so a measure tolerant of small variation
# is needed. Calibrated against one 404-question run: of the 326 answers that
# finished, the lowest scored 0.853, while 25 loops scored below 0.06. The
# default threshold sits roughly 3x below the closest legitimate answer, and
# flagged nothing that a model actually completed.
REPETITION_WINDOW_CHARS = 4000
REPETITION_NGRAM = 8
# Avoid classifying short structured output as repetition.
REPETITION_MIN_CHARS = 8000
# Deltas between checks. Each check is O(window), so this keeps the detector
# off the hot path of the decode loop.
REPETITION_CHECK_EVERY = 200
# Reported like a finish_reason so it travels with the request, not the answer.
REPETITION_FINISH_REASON = "repetition"
SSE_READ_BYTES = 65536
SSE_LINE_LIMIT = 16 * 1024 * 1024
_SSE_LINE_END = re.compile(br"\r\n|\r|\n")


def distinct_ngram_ratio(
    text: str, window: int = REPETITION_WINDOW_CHARS, n: int = REPETITION_NGRAM
) -> float:
    """Fraction of word n-grams in the last `window` chars that are unique.

    1.0 means every n-gram is new; a looping model tends to 0. Returns 1.0 for
    text too short to judge, so a caller can treat "not enough evidence" and
    "healthy" the same way.
    """
    words = text[-window:].split()
    if len(words) < n * 4:
        return 1.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def _tail(parts: list[str], window: int) -> str:
    """Join just enough trailing pieces to cover `window` characters."""
    collected: list[str] = []
    total = 0
    for piece in reversed(parts):
        collected.append(piece)
        total += len(piece)
        if total >= window:
            break
    collected.reverse()
    return "".join(collected)[-window:]


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
    # field, so the client drops it when explicitly rejected and retry budget remains.
    request_stream_usage: bool = True
    # Off for both canonical protocols: heuristic early exits alter accuracy
    # and throughput. Retained for explicit noncanonical client use.
    detect_repetition: bool = False
    repetition_threshold: float = 0.30
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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.complete_messages(
            messages, max_tokens=max_tokens, stream=stream, retries=retries
        )

    def complete_messages(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        stream: bool | None = None,
        retries: int | None = None,
    ) -> tuple[str, TokenUsage, RequestMetrics]:
        """Continue a bounded conversation; callers own and retain its history."""
        cfg = self.config
        attempts_allowed = max(1, cfg.max_retries if retries is None else retries)

        last_error = "unknown error"
        request_started = time.perf_counter()

        def failed(attempts: int) -> tuple[str, TokenUsage, RequestMetrics]:
            return self._error(
                last_error,
                attempts,
                latency_ms=(time.perf_counter() - request_started) * 1000.0,
            )

        for attempt in range(attempts_allowed):
            started = time.perf_counter()
            self._last_finish_reason = None
            self._stream_chunks = 0
            self._stream_span_ms = None
            # Recomputed per attempt: a fallback branch below may clear the
            # capability flags, and the next retry must honour that instead
            # of re-sending the payload that was just rejected.
            want_stream = (cfg.stream if stream is None else stream) and self._streaming_supported
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
                if (
                    e.status in (400, 422)
                    and want_stream
                    and self._stream_usage_supported
                    and attempt < attempts_allowed - 1
                    and _rejects_feature(e.body, "stream_options")
                ):
                    # A gateway complaining about stream_options: drop it and
                    # retry immediately without consuming a backoff cycle. The
                    # body must implicate the field — an unrelated 400 must not
                    # disable usage reporting for the client's lifetime.
                    logger.info("Endpoint rejected stream_options; disabling it")
                    self._stream_usage_supported = False
                    continue
                if (e.status in (400, 404, 422, 501) and want_stream
                    and attempt < attempts_allowed - 1
                    and _rejects_feature(e.body, "stream")):
                    # Only an endpoint actually complaining about streaming
                    # should turn streaming off for good; an unrelated 400 is
                    # just a request error.
                    logger.info("Endpoint rejected streaming; falling back to blocking")
                    self._streaming_supported = False
                    continue
                if not e.retryable:
                    # A permanent 4xx (401, 403, 413, ...) will fail the same
                    # way on every attempt, so don't pay the backoff.
                    return failed(attempt + 1)
                if attempt < attempts_allowed - 1:
                    self._sleep_backoff(attempt, last_error)
                    continue
                return failed(attempt + 1)
            except Exception as e:  # noqa: BLE001 - transport errors of any kind
                last_error = f"{type(e).__name__}: {e}"
                if attempt < attempts_allowed - 1:
                    self._sleep_backoff(attempt, last_error)
                    continue
                return failed(attempt + 1)

            finished = time.perf_counter()
            latency_ms = (finished - request_started) * 1000.0
            if ttft is not None:
                ttft += (started - request_started) * 1000.0
            metrics = RequestMetrics(
                latency_ms=latency_ms,
                successful_attempt_latency_ms=(finished - started) * 1000.0,
                ttft_ms=ttft,
                completion_tokens=usage.completion_tokens,
                prompt_tokens=usage.prompt_tokens,
                ok=True,
                attempts=attempt + 1,
                streamed=want_stream,
                cached_tokens=usage.cached_tokens,
                finish_reason=self._last_finish_reason,
                prompt_tokens_estimated=usage.prompt_tokens_estimated,
                completion_tokens_estimated=usage.completion_tokens_estimated,
                stream_chunks=self._stream_chunks,
                stream_span_ms=self._stream_span_ms,
            )
            return text, usage, metrics

        return failed(attempts_allowed)

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
        try:
            if resp.status_code >= 400:
                raise _RetryableStatus(resp.status_code, resp.text[:2000])
            data = resp.json()
            choices = _validated_choices(data, blocking=True)
            self._last_finish_reason = choices[0].get("finish_reason")
            text = _extract_message_text(choices)
            message = choices[0]["message"]
            delivered = _content_text(message.get("content")) + _content_text(
                message.get("reasoning") or message.get("reasoning_content")
            )
            usage_raw = data.get("usage") or {}
        finally:
            resp.close()
        usage = _parse_usage(usage_raw)
        if usage.prompt_tokens_estimated:
            usage.prompt_tokens = _estimate_tokens("".join(m.get("content", "") for m in messages))
            usage.prompt_tokens_estimated = True
        if usage.completion_tokens_estimated:
            usage.completion_tokens = _estimate_tokens(delivered)
            usage.completion_tokens_estimated = True
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
            try:
                body = resp.text[:2000]
            finally:
                resp.close()
            raise _RetryableStatus(resp.status_code, body)

        chunks: list[str] = []
        reasoning_chunks: list[str] = []
        # Content and reasoning interleaved in arrival order: a loop can happen
        # in either stream, and on reasoning endpoints it usually happens in
        # the one that never reaches `chunks`.
        generated: list[str] = []
        generated_chars = 0
        since_check = 0
        looped = False
        ttft: float | None = None
        delta_count = 0
        first_arrival = last_arrival = None
        usage = _parse_usage({})
        terminated = False
        done_sent = False
        saw_choice = False

        try:
            # Decode complete SSE lines, preserving UTF-8 across TCP fragments.
            # Requests' default 512-byte buffer can delay a short first event.
            # read1 returns available bytes without waiting to fill its buffer.
            for payload in _sse_payloads(resp, started, self.config.timeout * 2):
                if done_sent:
                    raise _ProtocolError("Completion data after [DONE]")
                if payload.strip() == "[DONE]":
                    terminated = True
                    done_sent = True
                    # Finish consuming HTTP framing so the socket returns to
                    # the pool. Closing early can make every warm request cold.
                    continue
                try:
                    event = json.loads(payload)
                except (json.JSONDecodeError, TypeError) as error:
                    raise _ProtocolError("Malformed JSON in completion stream") from error

                choices = _validated_choices(event)
                if choices and terminated:
                    raise _ProtocolError("Completion choice after terminal finish reason")
                saw_choice |= bool(choices)

                usage_raw = event.get("usage")
                if usage_raw:
                    update = _parse_usage(usage_raw)
                    for field in ("prompt_tokens", "completion_tokens"):
                        estimated = field + "_estimated"
                        if not getattr(update, estimated):
                            setattr(usage, field, getattr(update, field))
                            setattr(usage, estimated, False)
                    if isinstance(usage_raw, dict):
                        details = usage_raw.get("prompt_tokens_details")
                        if "prompt_cache_hit_tokens" in usage_raw or (
                            isinstance(details, dict) and "cached_tokens" in details
                        ):
                            usage.cached_tokens = update.cached_tokens

                for choice in choices:
                    if choice.get("finish_reason"):
                        self._last_finish_reason = choice["finish_reason"]
                        terminated = True
                    delta = choice.get("delta", {})
                    if delta is None:
                        delta = {}
                    if not isinstance(delta, dict):
                        raise _ProtocolError("Completion delta must be an object")
                    piece = _content_text(delta.get("content"))
                    reasoning = _content_text(delta.get("reasoning") or delta.get("reasoning_content"))
                    if piece or reasoning:
                        if ttft is None:
                            ttft = (time.perf_counter() - started) * 1000.0
                        if piece:
                            chunks.append(piece)
                        if reasoning:
                            reasoning_chunks.append(reasoning)
                        delta_count += 1
                    else:
                        continue

                    arrival = time.perf_counter()
                    if first_arrival is None:
                        first_arrival = arrival
                    last_arrival = arrival
                    if not self.config.detect_repetition:
                        continue
                    generated.append(reasoning + piece)
                    generated_chars += len(reasoning) + len(piece)
                    since_check += 1
                    if (
                        generated_chars < REPETITION_MIN_CHARS
                        or since_check < REPETITION_CHECK_EVERY
                    ):
                        continue
                    since_check = 0
                    ratio = distinct_ngram_ratio(_tail(generated, REPETITION_WINDOW_CHARS))
                    if ratio < self.config.repetition_threshold:
                        logger.info(
                            "Stopping a looping generation after %d chars "
                            "(distinct %d-gram ratio %.3f < %.2f)",
                            generated_chars,
                            REPETITION_NGRAM,
                            ratio,
                            self.config.repetition_threshold,
                        )
                        # Overrides any finish_reason the server has sent:
                        # the generation is being abandoned, not completed.
                        self._last_finish_reason = REPETITION_FINISH_REASON
                        looped = True
                        break
                if looped:
                    break
        finally:
            resp.close()

        if not saw_choice:
            raise _ProtocolError("Completion stream contained no completion choice")
        if not terminated and not looped:
            raise _ProtocolError("Completion stream ended without a finish reason or [DONE]")
        text = "".join(chunks).strip()
        if not text:
            reasoning = "".join(reasoning_chunks).strip()
            text = f"<think>{reasoning}</think>" if reasoning else ""
        if usage.completion_tokens_estimated:
            # SSE events can contain many tokens. Estimate from all delivered
            # text instead of treating one event as one token.
            usage.completion_tokens = _estimate_tokens("".join(chunks + reasoning_chunks))
            usage.completion_tokens_estimated = True
        if usage.prompt_tokens_estimated:
            usage.prompt_tokens_estimated = True
            usage.prompt_tokens = _estimate_tokens("".join(m.get("content", "") for m in messages))
        self._stream_chunks = delta_count
        self._stream_span_ms = (
            (last_arrival - first_arrival) * 1000 if first_arrival is not None else None
        )
        return text, usage, ttft

    def _sleep_backoff(self, attempt: int, reason: str) -> None:
        wait = self.config.retry_delay * (2**attempt)
        logger.info("Request failed (%s); retrying in %.0fs", reason, wait)
        time.sleep(wait)

    @staticmethod
    def _error(
        message: str,
        attempts: int,
        *,
        latency_ms: float = 0.0,
    ) -> tuple[str, TokenUsage, RequestMetrics]:
        return (
            f"[API ERROR: {message}]",
            TokenUsage(),
            RequestMetrics(ok=False, error=message, attempts=attempts, latency_ms=latency_ms),
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


class _ProtocolError(Exception):
    """An endpoint returned an invalid or incomplete completion."""


def _response_lines(response, started, deadline):
    """Available-byte reads, UTF-8-safe line boundaries and bounded buffering."""
    reader = getattr(getattr(response, "raw", None), "read1", None)
    if callable(reader):
        def chunks():
            while True:
                chunk = reader(SSE_READ_BYTES, decode_content=True)
                if not chunk:
                    break
                yield chunk
        source = chunks()
    else:
        # Compatibility for older urllib3/custom transport adapters. One byte
        # avoids their read-to-fill behavior, at greater Python overhead.
        source = response.iter_content(chunk_size=1)
    pending, skip_lf = bytearray(), False
    for chunk in source:
        if time.perf_counter() - started > deadline:
            raise _StreamDeadlineExceeded(f"stream deadline exceeded after {deadline:.0f}s")
        if not chunk:
            continue
        if skip_lf:
            if chunk.startswith(b"\n"):
                chunk = chunk[1:]
            skip_lf = False
        scan_from = len(pending)
        pending.extend(chunk)
        consumed = 0
        for boundary in _SSE_LINE_END.finditer(pending, scan_from):
            if boundary.start() - consumed > SSE_LINE_LIMIT:
                raise _ProtocolError("Completion SSE line exceeds buffer limit")
            yield bytes(pending[consumed:boundary.start()])
            consumed = boundary.end()
            skip_lf = boundary.group() == b"\r" and consumed == len(pending)
        del pending[:consumed]
        if len(pending) > SSE_LINE_LIMIT:
            raise _ProtocolError("Completion SSE line exceeds buffer limit")
    if pending:
        yield bytes(pending)


def _sse_payloads(response, started, deadline):
    """Dispatch complete SSE events; join multiline data and ignore comments."""
    data = []
    data_bytes = 0
    first_line = True
    for raw_line in _response_lines(response, started, deadline):
        try:
            line = raw_line.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise _ProtocolError("Invalid UTF-8 in completion stream") from error
        if first_line:
            line = line.removeprefix("\ufeff")
            first_line = False
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
                data_bytes = 0
        elif line == "data" or line.startswith("data:"):
            value = line[5:] if line.startswith("data:") else ""
            data.append(value[1:] if value.startswith(" ") else value)
            data_bytes += len(raw_line)
            if data_bytes > SSE_LINE_LIMIT:
                raise _ProtocolError("Completion SSE event exceeds buffer limit")
    # A final unterminated event is not dispatched by the SSE protocol.
    if data:
        raise _ProtocolError("Completion stream ended inside an SSE event")


def _validated_choices(data, *, blocking=False):
    if not isinstance(data, dict):
        raise _ProtocolError("Completion response must be an object")
    if data.get("error"):
        raise _ProtocolError("Endpoint returned an error event")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) > 1 or (blocking and not choices):
        raise _ProtocolError("Expected one completion choice (or a streaming usage event)")
    for choice in choices:
        if (not isinstance(choice, dict) or type(choice.get("index", 0)) is not int
            or choice.get("index", 0) != 0):
            raise _ProtocolError("Unexpected completion choice index")
        finish = choice.get("finish_reason")
        if finish is not None and (not isinstance(finish, str) or not finish.strip()):
            raise _ProtocolError("Completion finish reason must be a nonempty string or null")
        if blocking and not isinstance(choice.get("message"), dict):
            raise _ProtocolError("Completion message must be an object")
    return choices


def _rejects_feature(body, feature):
    """Negotiate only an explicitly unsupported field, never incidental mentions.

    In particular, 'upstream' and a stream_options validation error do not mean
    that streaming is unavailable. One-attempt measured calls never negotiate.
    """
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict) and error.get("param") == feature:
        code = error.get("code")
        if isinstance(code, str) and code in {"unsupported_parameter", "unknown_parameter", "unsupported_value"}:
            return True
    name = r"stream(?:ing)?" if feature == "stream" else re.escape(feature)
    label = rf"\b{name}\b"
    separator = r"[\s\"'`:=_-]*"
    before = rf"\b(?:unsupported|unknown|unrecognized|unexpected)\s+(?:(?:field|parameter|argument)\b)?{separator}{label}"
    after = rf"{label}{separator}(?:(?:is|are)\s+)?(?:not\s+(?:supported|implemented|allowed|permitted)|unsupported|disabled)\b"
    negative = rf"\b(?:does\s+not|doesn't|cannot)\s+support\s+{separator}{label}"
    return bool(re.search(f"{before}|{after}|{negative}", body, re.IGNORECASE))


def _parse_usage(usage_raw: dict) -> TokenUsage:
    """Normalise the `usage` JSON object into TokenUsage.

    Prefix-cache hit counts arrive in two divergent shapes: vLLM-style emits a
    top-level ``prompt_cache_hit_tokens``, OpenAI-style nests it under
    ``prompt_tokens_details.cached_tokens``. The explicit top-level field wins;
    Missing or invalid prompt/completion counts are marked for estimation;
    explicit zero counts remain reported counts.
    """
    usage_raw = usage_raw if isinstance(usage_raw, dict) else {}
    def count(value):
        # Optional provider telemetry must not turn a completed answer into an
        # infrastructure error. Invalid counts become explicitly marked estimates.
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            try:
                return int(value)
            except ValueError:
                return None
        return None

    prompt = count(usage_raw.get("prompt_tokens"))
    completion = count(usage_raw.get("completion_tokens"))
    cached = usage_raw.get("prompt_cache_hit_tokens")
    if cached is None:
        details = usage_raw.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
    return TokenUsage(
        prompt_tokens=prompt or 0,
        completion_tokens=completion or 0,
        cached_tokens=count(cached) or 0,
        prompt_tokens_estimated=prompt is None,
        completion_tokens_estimated=completion is None,
    )


def _extract_message_text(choices: list[dict]) -> str:
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        # Preserve diagnostics without promoting private reasoning to an answer.
        reasoning = _content_text(message.get("reasoning") or message.get("reasoning_content"))
        content = f"<think>{reasoning}</think>" if reasoning else ""
    return _content_text(content).strip()


def _content_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict) and isinstance(part.get("text"), str) for part in content
    ):
        return "".join(part["text"] for part in content)
    raise _ProtocolError("Completion content must be text or text parts")


def client_protocol(config):
    """Result-affecting client settings, without endpoints, headers or secrets."""
    return {
        "revision": CLIENT_PROTOCOL_VERSION,
        "timeout_seconds": config.timeout,
        "stream_deadline_seconds": config.timeout * 2,
        "stream_requested": config.stream,
        "stream_usage_requested": config.request_stream_usage,
        "max_attempts": config.max_retries,
        "retry_delay_seconds": config.retry_delay,
        "detect_repetition": config.detect_repetition,
        "repetition_threshold": config.repetition_threshold if config.detect_repetition else None,
        "latency_scope": "all_attempts_and_backoff",
        "sse_reader": "read1" if hasattr(urllib3.response.HTTPResponse, "read1") else "single_byte_fallback",
        "sse_read_max_bytes": SSE_READ_BYTES,
        "requests_version": requests.__version__,
        "urllib3_version": urllib3.__version__,
    }


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))
