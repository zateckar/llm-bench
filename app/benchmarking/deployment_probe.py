"""Deployment snapshots and fingerprints (``deployment-probe-v1``).

A snapshot records what an OpenAI-compatible endpoint says about itself and how
its chat template and tokenizer count a few fixed prompts. Prompt-token counts
are deterministic for a template and tokenizer, so the fingerprint changes
exactly when what is served changes in a way the API can observe. The short
behaviour probe is shown for context only: batching makes temperature-0 output
nondeterministic on most engines, so it never enters the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit, urlunsplit

REVISION = "deployment-probe-v1"
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 30
TEXT_LIMIT = 400

_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
PROBES = {
    "plain": {"messages": [{"role": "user", "content": "Reply with OK."}]},
    "system": {"messages": [{"role": "system", "content": "You are a terse assistant."},
                            {"role": "user", "content": "Reply with OK."}]},
    "multilingual": {"messages": [{"role": "user", "content":
                                   "Příliš žluťoučký kůň úpěl ďábelské ódy. Größenwahn – ½ € · 漢字 🙂 Reply with OK."}]},
    "multi_turn": {"messages": [{"role": "system", "content": "You are a terse assistant."},
                                {"role": "user", "content": "Remember the number 7."},
                                {"role": "assistant", "content": "Noted."},
                                {"role": "user", "content": "Reply with OK."}]},
    "tools": {"messages": [{"role": "user", "content": "Reply with OK."}], "tools": [_TOOL]},
}
BEHAVIOUR = {"messages": [{"role": "user", "content": "List the first eight prime numbers, separated by commas."}],
             "max_tokens": 48}
CONFIGURATION_FIELDS = ("endpoint", "model")


def sanitize_endpoint(url):
    """Scheme, host, port and path only: never credentials or query strings."""
    parts = urlsplit(url or "")
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path.rstrip("/"), "", ""))


def origin_of(base_url):
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


class Unreachable(Exception):
    pass


def _request(session, method, url, headers, payload=None):
    try:
        response = session.request(method, url, headers=headers, json=payload,
                                   timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except Exception as error:  # noqa: BLE001 - every transport failure means "unreachable"
        raise Unreachable(f"{type(error).__name__}: {error}"[:300]) from error
    try:
        body = response.json()
    except ValueError:
        body = None
    return response.status_code, body, response.headers


def _served_entry(body, model):
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return {"listed": None}
    for item in data:
        if isinstance(item, dict) and item.get("id") == model:
            return {"listed": True, **{k: item.get(k) for k in ("id", "root", "max_model_len", "owned_by", "parent")
                                       if item.get(k) is not None}}
    return {"listed": False}


def capture(base_url, api_key, model, *, session=None, extra_headers=None):
    """Capture a snapshot. Never raises: failures become ``ok: False``."""
    import requests

    own = session is None
    session = session or requests.Session()
    headers = {"Authorization": f"Bearer {api_key}", **(extra_headers or {})}
    base = base_url.rstrip("/")
    snapshot = {"revision": REVISION, "ok": False,
                "hard": {"endpoint": sanitize_endpoint(base_url), "model": model}}
    hard = snapshot["hard"]
    try:
        status, body, response_headers = _request(session, "GET", f"{base}/models", headers)
        hard["served"] = _served_entry(body, model) if status == 200 else {"status": status}
        hard["server_header"] = response_headers.get("server") if response_headers else None
        try:
            status, body, _ = _request(session, "GET", f"{origin_of(base_url)}/version", headers)
            hard["version"] = body.get("version") if status == 200 and isinstance(body, dict) else None
        except Unreachable:
            hard["version"] = None
        probes, models, fingerprints = {}, set(), set()
        for name, extra in PROBES.items():
            status, body, _ = _request(session, "POST", f"{base}/chat/completions", headers, {
                "model": model, "max_tokens": 1, "temperature": 0, "seed": 0, "stream": False, **extra})
            if status == 200 and isinstance(body, dict):
                usage = body.get("usage") or {}
                probes[name] = {"prompt_tokens": usage.get("prompt_tokens")}
                if body.get("model"):
                    models.add(str(body["model"]))
                if body.get("system_fingerprint"):
                    fingerprints.add(str(body["system_fingerprint"]))
            else:
                probes[name] = {"status": status}
        hard["probes"] = probes
        hard["response_model"] = sorted(models)
        hard["system_fingerprint"] = sorted(fingerprints)
        if all("status" in p for p in probes.values()):
            snapshot["error"] = f"Every chat probe was rejected (HTTP {probes['plain']['status']})"
            return snapshot
        status, body, _ = _request(session, "POST", f"{base}/chat/completions", headers, {
            "model": model, "temperature": 0, "seed": 0, "stream": False, **BEHAVIOUR})
        if status == 200 and isinstance(body, dict) and body.get("choices"):
            choice = body["choices"][0]
            text = ((choice.get("message") or {}).get("content") or "")
            snapshot["behaviour"] = {"text": text[:TEXT_LIMIT], "finish_reason": choice.get("finish_reason"),
                                     "sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
        else:
            snapshot["behaviour"] = {"status": status}
        snapshot["ok"] = True
    except Unreachable as error:
        snapshot["error"] = f"Endpoint unreachable: {error}"
    finally:
        if own:
            session.close()
    if snapshot["ok"]:
        snapshot["fingerprint"] = fingerprint(snapshot)
    return snapshot


def fingerprint(snapshot):
    canonical = json.dumps(snapshot["hard"], sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: value}


def diff(before, after):
    """Changed hard fields as ``[{"field", "before", "after"}]``, sorted by field."""
    a, b = _flatten(before["hard"]), _flatten(after["hard"])
    return [{"field": key, "before": a.get(key), "after": b.get(key)}
            for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)]


def _short(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return "—" if value in (None, [], "") else str(value)


def summary(changes):
    """Short readable description, most telling changes first."""
    labels = {"served.max_model_len": "context limit", "version": "server version", "served.root": "served weights",
              "served.id": "served model", "served.listed": "listed by /models", "response_model": "responding model",
              "system_fingerprint": "system fingerprint", "server_header": "server", "endpoint": "endpoint",
              "model": "configured model"}
    parts = [f"{labels[c['field']]} {_short(c['before'])} → {_short(c['after'])}"
             for key in labels for c in changes if c["field"] == key]
    probes = {c["field"].split(".")[1] for c in changes if c["field"].startswith("probes.")}
    if probes:
        parts.append(f"chat template or tokenizer ({len(probes)} of {len(PROBES)} probes count differently)")
    known = set(labels) | {c["field"] for c in changes if c["field"].startswith("probes.")}
    parts.extend(c["field"] for c in changes if c["field"] not in known)
    return "; ".join(parts)


def classify(changes):
    """``configuration`` when only the configured endpoint/model changed, else ``deployment``."""
    if not changes:
        return None
    if all(c["field"] in CONFIGURATION_FIELDS for c in changes):
        return "configuration"
    return "deployment"
