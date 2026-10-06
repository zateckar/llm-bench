"""Read-only B300 vLLM timelines. Network work stays outside measured requests."""

from dataclasses import dataclass, field
import json
import math
import os
import re
import time
from urllib.parse import urlsplit, urlunsplit

import requests

REVISION = "b300-vllm-telemetry-v3"
BASELINE_SECONDS = 120
MAX_POINTS = 1200
# All queries of one collection, after the measurement.
COLLECTION_SECONDS = 60
MAX_GPUS = 64
# DCGM exporter series carry the node in this label and the GPU index in "gpu".
GPU_HOST_LABEL = "Hostname"
# Labels kept from vllm:cache_config_info; everything else is dropped.
CACHE_CONFIG_KEYS = {"instance", "engine", "block_size", "cache_dtype", "num_gpu_blocks", "num_cpu_blocks",
                     "enable_prefix_caching", "gpu_memory_utilization", "swap_space", "cpu_offload_gb",
                     "kv_offloading_size", "kv_offloading_backend", "prefix_caching_hash_algo", "sliding_window",
                     "calculate_kv_scales", "is_attention_free"}


def metrics_model(value):
    if value is not None and not isinstance(value, str):
        raise ValueError("B300 metric model name must be text")
    value = (value or "").strip()
    if len(value) > 512 or any(ord(c) < 32 for c in value):
        raise ValueError("B300 metric model name must be at most 512 characters without control characters")
    return value or None


def gpu_indices(value):
    """GPU indices of a deployment from text such as ``0-3`` or ``4,5,6,7``; None for all GPUs on the host."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("GPU indices must be text such as 0-3 or 4,5,6,7")
    value = value.strip()
    if not value:
        return None
    indices = set()
    for part in value.replace(" ", "").split(","):
        first, dash, last = part.partition("-")
        if not first.isdecimal() or (dash and not last.isdecimal()):
            raise ValueError("GPU indices must be text such as 0-3 or 4,5,6,7")
        low, high = int(first), int(last or first)
        if not 0 <= low <= high < MAX_GPUS:
            raise ValueError(f"GPU indices must be between 0 and {MAX_GPUS - 1}")
        indices.update(range(low, high + 1))
    return sorted(indices)


def gpu_text(indices):
    return ",".join(str(i) for i in indices) if indices else None


def metrics_scope(model):
    """Snapshot only selectors, never the API credentials."""
    name = metrics_model(model.get("b300_metrics_model"))
    if not name:
        return None
    scope = {"hardware": "B300", "host": os.getenv("PROMETHEUS_B300_HOST", "").strip(),
             "job": "vllm", "model_name": name}
    gpus = gpu_indices(model.get("b300_gpus"))
    if gpus:
        scope["gpus"] = gpus
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


def gpu_selector(scope):
    """DCGM selector: the deployment's GPUs when configured, else every GPU on the host."""
    selector = f"{GPU_HOST_LABEL}={json.dumps(scope['host'], ensure_ascii=False)}"
    gpus = scope.get("gpus")
    if gpus:
        selector += f',gpu=~"{"|".join(str(int(g)) for g in gpus)}"'
    return selector


def queries(scope):
    """Range queries. ``kind`` says how a value was formed (gauge, 1-minute rate or rolling
    histogram p95), ``combine`` how series of several instances add up, and ``optional``
    metrics (GPU, newer vLLM) may be absent without making the collection partial."""
    if not isinstance(scope, dict) or scope.get("hardware") != "B300" or scope.get("job") != "vllm":
        raise ValueError("Only B300 vLLM telemetry is supported")
    if not scope.get("host") or not scope.get("model_name"):
        raise ValueError("B300 hostname and metric model name are required")
    selector = ",".join(f"{label}={json.dumps(scope[key], ensure_ascii=False)}" for label, key in (
        ("Hostname", "host"), ("job", "job"), ("model_name", "model_name")))
    result = []
    for key, title, metric, unit, operation, combine in (
        ("running", "Running requests", "num_requests_running", "requests", "sum", "sum"),
        ("waiting", "Waiting requests", "num_requests_waiting", "requests", "sum", "sum"),
        ("kv_cache", "KV cache occupancy", "kv_cache_usage_perc", "fraction", "max", "max"),
        ("output", "Server generation rate", "generation_tokens_total", "tok/s", "rate", "sum"),
        ("input", "Server prompt rate", "prompt_tokens_total", "tok/s", "rate", "sum"),
        ("preemptions", "Preemption rate", "num_preemptions_total", "events/s", "rate", "sum"),
    ):
        source = f"vllm:{metric}{{{selector}}}"
        expression = f"sum by (instance, model_name) (rate({source}[1m]))" if operation == "rate" else f"{operation} by (instance, model_name) ({source})"
        result.append({"id": key, "title": title, "unit": unit, "query": expression,
                       "kind": "rate" if operation == "rate" else "gauge", "combine": combine})

    def p95(metric):
        return f"histogram_quantile(0.95, sum by (le, instance, model_name) (rate(vllm:{metric}_seconds_bucket{{{selector}}}[1m])))"

    for key, title, metric in (
        ("ttft", "Server first token", "time_to_first_token"),
        ("queue", "Server queue time", "request_queue_time"),
        ("prefill", "Server prefill time", "request_prefill_time"),
        ("decode", "Server decode time", "request_decode_time"),
        ("latency", "Server response time", "e2e_request_latency"),
    ):
        result.append({"id": key, "title": title + " · rolling p95", "unit": "seconds", "query": p95(metric),
                       "kind": "histogram", "combine": "max"})
    result.append({"id": "prefix_hit", "title": "Prefix cache hit fraction", "unit": "fraction",
                   "query": f"sum by (instance, model_name) (rate(vllm:prefix_cache_hits_total{{{selector}}}[1m])) / sum by (instance, model_name) (rate(vllm:prefix_cache_queries_total{{{selector}}}[1m]))",
                   "kind": "rate", "combine": "mean"})
    # Newer vLLM renamed time per output token to inter-token latency.
    result.append({"id": "tpot", "title": "Server time per output token · rolling p95", "unit": "seconds",
                   "query": f"{p95('inter_token_latency')} or {p95('time_per_output_token')}",
                   "kind": "histogram", "combine": "max", "optional": True})
    gpu = gpu_selector(scope)
    for key, title, unit, expression in (
        ("gpu_util", "GPU busy (any kernel)", "fraction", f"avg(DCGM_FI_DEV_GPU_UTIL{{{gpu}}}) / 100"),
        ("sm_active", "GPU SM activity", "fraction", f"avg(DCGM_FI_PROF_SM_ACTIVE{{{gpu}}})"),
        ("tensor_active", "GPU tensor-core activity", "fraction", f"avg(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{{{gpu}}})"),
        ("dram_active", "GPU memory-bandwidth activity", "fraction", f"avg(DCGM_FI_PROF_DRAM_ACTIVE{{{gpu}}})"),
        ("gpu_memory", "GPU memory used", "fraction",
         f"sum(DCGM_FI_DEV_FB_USED{{{gpu}}}) / (sum(DCGM_FI_DEV_FB_USED{{{gpu}}}) + sum(DCGM_FI_DEV_FB_FREE{{{gpu}}}))"),
        ("pcie", "GPU PCIe traffic", "B/s",
         f"sum(DCGM_FI_PROF_PCIE_TX_BYTES{{{gpu}}}) + sum(DCGM_FI_PROF_PCIE_RX_BYTES{{{gpu}}})"),
        ("power", "GPU power", "W", f"sum(DCGM_FI_DEV_POWER_USAGE{{{gpu}}})"),
    ):
        result.append({"id": key, "title": title, "unit": unit, "query": expression, "kind": "gauge",
                       "combine": "mean", "optional": True, "group": "gpu"})
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
            self.instances = sorted({str(row.get("metric", {}).get("instance")) for row in result["result"]
                                     if isinstance(row, dict) and isinstance(row.get("metric"), dict)
                                     and row["metric"].get("instance")})[:8]
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

    def _cache_config(self, at):
        """vLLM's cache_config_info labels (KV blocks, dtype, prefix caching) for the measured instances.
        The info metric carries no model_name, so it is matched by the instances found at setup."""
        instances = getattr(self, "instances", [])
        if not instances:
            return None
        selector = ",".join(f"{label}={json.dumps(value, ensure_ascii=False)}" for label, value in (
            ("Hostname", self.scope["host"]), ("job", self.scope["job"])))
        # Regex-escape, then escape for the PromQL string literal.
        pattern = "|".join(re.escape(i).replace("\\", "\\\\").replace('"', '\\"') for i in instances)
        try:
            result = self._request(self.settings.url, {"query": f'vllm:cache_config_info{{{selector},instance=~"{pattern}"}}',
                                                       "time": at})
        except TelemetryError:
            return None
        configs = []
        for row in result.get("result", [])[:8]:
            labels = row.get("metric") if isinstance(row, dict) else None
            if isinstance(labels, dict):
                kept = {key: str(value)[:64] for key, value in labels.items() if key in CACHE_CONFIG_KEYS}
                if kept:
                    configs.append(kept)
        return configs or None

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
            deadline = time.monotonic() + COLLECTION_SECONDS
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
                    if not definition.get("optional"):
                        self.data["warnings"].append(f"{definition['title']}: historical data unavailable.")
            if not self.cancelled() and time.monotonic() < deadline:
                self.data["cache_config"] = self._cache_config(ended_at - self.shift)
            required = [d["id"] for d in self.definitions if not d.get("optional")]
            collected = {m["id"] for m in self.data["metrics"] if m["status"] == "collected"}
            count = len(collected.intersection(required))
            self.data["status"] = "collected" if count == len(required) else "partial" if count else "unavailable"
            absent = [d["title"] for d in self.definitions if d.get("optional") and d["id"] not in collected]
            if absent:
                self.data["warnings"].append(
                    "Optional metrics unavailable: " + ", ".join(absent)
                    + ". GPU metrics need a DCGM exporter labelled with the hostname; the constraint diagnosis uses what is present.")
            if not self.scope.get("gpus") and any(d.get("group") == "gpu" for d in self.definitions):
                self.data["warnings"].append("GPU metrics cover every GPU on the host; set the model's GPU indices when other deployments share it.")
            latest = max((t for m in self.data["metrics"] for s in m["series"] for t, v in s["values"] if v is not None), default=None)
            self.data["latest_sample_timestamp"] = latest
            self.data["latest_aligned_sample_timestamp"] = latest+self.shift if latest is not None else None
            if latest is None or ended_at-(latest+self.shift) > max(15, step*3):
                self.data["warnings"].append("Recent telemetry is missing or delayed; the end of the measurement may not be covered.")
            self.data["warnings"].append("Rates and histogram percentiles use rolling 60-second windows; short phases overlap neighboring traffic.")
            return self.data
        finally:
            self.session.close()
