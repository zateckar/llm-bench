"""Read-only B300 vLLM timelines. Network work stays outside measured requests."""

from dataclasses import dataclass, field
import json
import math
import os
import time
from urllib.parse import urlsplit, urlunsplit

import requests

REVISION = "b300-vllm-telemetry-v2"
BASELINE_SECONDS = 120
MAX_POINTS = 1200


def metrics_model(value):
    if value is not None and not isinstance(value, str):
        raise ValueError("B300 metric model name must be text")
    value = (value or "").strip()
    if len(value) > 512 or any(ord(c) < 32 for c in value):
        raise ValueError("B300 metric model name must be at most 512 characters without control characters")
    return value or None


def metrics_scope(model):
    """Snapshot only selectors, never the API credentials."""
    name = metrics_model(model.get("b300_metrics_model"))
    if not name:
        return None
    scope = {"hardware": "B300", "host": os.getenv("PROMETHEUS_B300_HOST", "").strip(),
             "job": "vllm", "model_name": name}
    if os.getenv("PROMETHEUS_TIMESTAMP_SHIFT_SECONDS", "").strip():
        scope["timestamp_shift_seconds"] = os.environ["PROMETHEUS_TIMESTAMP_SHIFT_SECONDS"].strip()
    return scope


def alignment_shift(data):
    """Correction is saved with the run; source samples are never rewritten."""
    shift = (data.get("alignment") or {}).get("timestamp_shift_seconds", 0)
    offset = data.get("api_clock_offset_seconds")
    if type(shift) not in (int, float) or not math.isfinite(shift) or abs(shift) > 600:
        return 0, False
    return shift, type(offset) in (int, float) and math.isfinite(offset) and abs(offset+shift) <= 15


@dataclass(frozen=True)
class Settings:
    url: str = field(repr=False)
    subscription_key: str = field(repr=False)
    authorization: str = field(repr=False)

    @classmethod
    def from_env(cls):
        return cls(*(os.getenv(key, "").strip() for key in (
            "PROMETHEUS_API_URL", "PROMETHEUS_API_SUBSCRIPTION_KEY", "PROMETHEUS_API_AUTHORIZATION")))


def queries(scope):
    if not isinstance(scope, dict) or scope.get("hardware") != "B300" or scope.get("job") != "vllm":
        raise ValueError("Only B300 vLLM telemetry is supported")
    if not scope.get("host") or not scope.get("model_name"):
        raise ValueError("B300 hostname and metric model name are required")
    selector = ",".join(f"{label}={json.dumps(scope[key], ensure_ascii=False)}" for label, key in (
        ("Hostname", "host"), ("job", "job"), ("model_name", "model_name")))
    result = []
    for key, title, metric, unit, operation in (
        ("running", "Running requests", "num_requests_running", "requests", "sum"),
        ("waiting", "Waiting requests", "num_requests_waiting", "requests", "sum"),
        ("kv_cache", "KV cache occupancy", "kv_cache_usage_perc", "fraction", "max"),
        ("output", "Server generation rate", "generation_tokens_total", "tok/s", "rate"),
        ("input", "Server prompt rate", "prompt_tokens_total", "tok/s", "rate"),
        ("preemptions", "Preemption rate", "num_preemptions_total", "events/s", "rate"),
    ):
        source = f"vllm:{metric}{{{selector}}}"
        expression = f"sum by (instance, model_name) (rate({source}[1m]))" if operation == "rate" else f"{operation} by (instance, model_name) ({source})"
        result.append({"id": key, "title": title, "unit": unit, "query": expression})
    for key, title, metric in (
        ("ttft", "Server first token", "time_to_first_token"),
        ("queue", "Server queue time", "request_queue_time"),
        ("prefill", "Server prefill time", "request_prefill_time"),
        ("decode", "Server decode time", "request_decode_time"),
        ("latency", "Server response time", "e2e_request_latency"),
    ):
        expression = f"histogram_quantile(0.95, sum by (le, instance, model_name) (rate(vllm:{metric}_seconds_bucket{{{selector}}}[1m])))"
        result.append({"id": key, "title": title + " · rolling p95", "unit": "seconds", "query": expression})
    result.append({"id": "prefix_hit", "title": "Prefix cache hit fraction", "unit": "fraction",
                   "query": f"sum by (instance, model_name) (rate(vllm:prefix_cache_hits_total{{{selector}}}[1m])) / sum by (instance, model_name) (rate(vllm:prefix_cache_queries_total{{{selector}}}[1m]))"})
    return result, selector


class TelemetryError(Exception):
    pass


class VllmTelemetry:
    def __init__(self, scope, cancelled=lambda: False, settings=None, session=None):
        self.scope = scope
        self.cancelled = cancelled
        self.settings = settings or Settings.from_env()
        self.session = session or requests.Session()
        self.data = {"revision": REVISION, "status": "pending", "scope": scope,
                     "metrics": [], "warnings": [], "attribution": "All traffic for the selected vLLM deployment; not benchmark-only traffic"}
        self.ready = False
        self.shift = 0

    def _request(self, url, params):
        auth = self.settings.authorization
        if not auth.lower().startswith("basic "):
            auth = "Basic " + auth
        try:
            response = self.session.get(url, params={**params, "timeout": "5s"},
                headers={"Ocp-Apim-Subscription-Key": self.settings.subscription_key,
                         "Authorization": auth, "Accept": "application/json"},
                timeout=(3, 5), allow_redirects=False, stream=True)
            try:
                if response.status_code != 200:
                    raise TelemetryError(f"Prometheus returned HTTP {response.status_code}")
                chunks, size = [], 0
                deadline = time.monotonic() + 8
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > 8 * 1024 * 1024 or time.monotonic() >= deadline:
                        raise TelemetryError("Prometheus response exceeds the telemetry size or time limit")
                    chunks.append(chunk)
                payload = json.loads(b"".join(chunks))
                if not isinstance(payload, dict) or payload.get("status") != "success":
                    raise TelemetryError("Prometheus rejected the query")
                data = payload.get("data")
                if not isinstance(data, dict) or not isinstance(data.get("result"), list):
                    raise TelemetryError("Prometheus returned an invalid result")
                return data
            finally:
                response.close()
        except (requests.RequestException, ValueError, TypeError, OverflowError) as error:
            # Exceptions and gateway bodies can contain URLs or credentials.
            raise TelemetryError(f"Prometheus request failed ({type(error).__name__})") from None

    def begin(self):
        self.started_at = time.time()
        try:
            self.definitions, selector = queries(self.scope)
            self.shift = float(self.scope.get("timestamp_shift_seconds", 0))
            if not math.isfinite(self.shift) or abs(self.shift) > 600:
                raise TelemetryError("Telemetry timestamp correction must be within 600 seconds")
            self.data["alignment"] = {"method": "configured" if self.shift else "source",
                                      "timestamp_shift_seconds": self.shift,
                                      "source_timestamps_preserved": True}
            parsed = urlsplit(self.settings.url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise TelemetryError("Configure a Prometheus query URL without embedded credentials or query parameters")
            if not self.settings.subscription_key or not self.settings.authorization:
                raise TelemetryError("Prometheus credentials are not configured")
            self.range_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rsplit("/", 1)[0] + "/query_range", "", ""))
            before = time.time()
            # An explicit evaluation time bypasses the gateway's stale default time.
            result = self._request(self.settings.url, {"query": f"vllm:num_requests_running{{{selector}}}",
                                                       "time": before-self.shift})
            after = time.time()
            if result.get("resultType") != "vector" or not result["result"]:
                raise TelemetryError("No vLLM metrics match this B300 model mapping")
            stamp = float(result["result"][0]["value"][0])
            offset = stamp - (before + after) / 2
            self.data["api_clock_offset_seconds"] = offset if math.isfinite(offset) else None
            if not math.isfinite(offset) or abs(offset+self.shift) > 15:
                self.data["warnings"].append("API evaluation time disagrees with the configured correction; phase alignment remains uncertain.")
            if self.shift:
                self.data["warnings"].append(f"Server timelines use a configured {self.shift:+g}-second correction. This assumes source clock lag, not transport delay; original timestamps remain in JSON.")
            self.ready = True
            self.data["status"] = "collecting"
        except (TelemetryError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError):
            self.data["status"] = "unavailable"
            self.data["warnings"].append("B300 telemetry setup failed; check credentials, hostname and exact metric model name.")
        # Preparation is excluded from the measurement window.
        self.started_at = time.time()
        return self.data

    def finish(self):
        ended_at = time.time()
        try:
            if not self.ready or self.cancelled():
                if self.cancelled():
                    self.data["status"] = "cancelled"
                return self.data
            start = self.started_at - BASELINE_SECONDS
            source_start, source_end = start-self.shift, ended_at-self.shift
            step = max(5, math.ceil((ended_at - start) / MAX_POINTS))
            self.data["window"] = {"start": start, "end": ended_at, "measurement_start": self.started_at,
                                   "step_seconds": step, "baseline_seconds": BASELINE_SECONDS}
            if self.shift:
                self.data["window"].update(query_start=source_start, query_end=source_end)
            deadline = time.monotonic() + 20
            for definition in self.definitions:
                if self.cancelled() or time.monotonic() >= deadline:
                    self.data["warnings"].append("Telemetry collection stopped before all queries completed.")
                    break
                metric = {**definition, "series": [], "status": "missing"}
                self.data["metrics"].append(metric)
                try:
                    result = self._request(self.range_url, {"query": definition["query"], "start": source_start,
                                                           "end": source_end, "step": f"{step}s"})
                    if result.get("resultType") != "matrix":
                        raise TelemetryError("Prometheus did not return a historical matrix")
                    if len(result["result"]) > 32:
                        raise TelemetryError("Too many series match the B300 mapping")
                    for row in result["result"]:
                        values = []
                        for stamp, value in row.get("values", [])[:MAX_POINTS + 2]:
                            stamp, value = float(stamp), float(value)
                            if math.isfinite(stamp) and source_start <= stamp <= source_end:
                                values.append([stamp, value if math.isfinite(value) else None])
                        metric["series"].append({"labels": row.get("metric", {}), "values": values})
                    if any(any(v is not None for _, v in s["values"]) for s in metric["series"]):
                        metric["status"] = "collected"
                except (TelemetryError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError):
                    metric["status"] = "error"
                    self.data["warnings"].append(f"{definition['title']}: historical data unavailable.")
            count = sum(m["status"] == "collected" for m in self.data["metrics"])
            self.data["status"] = "collected" if count == len(self.definitions) else "partial" if count else "unavailable"
            latest = max((t for m in self.data["metrics"] for s in m["series"] for t, v in s["values"] if v is not None), default=None)
            self.data["latest_sample_timestamp"] = latest
            self.data["latest_aligned_sample_timestamp"] = latest+self.shift if latest is not None else None
            if latest is None or ended_at-(latest+self.shift) > max(15, step*3):
                self.data["warnings"].append("Recent telemetry is missing or delayed; the end of the measurement may not be covered.")
            self.data["warnings"].append("Rates and histogram percentiles use rolling 60-second windows; short phases overlap neighboring traffic.")
            return self.data
        finally:
            self.session.close()
