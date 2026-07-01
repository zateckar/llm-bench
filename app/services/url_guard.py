"""SSRF guard for operator-supplied model endpoints.

A model's ``base_url`` is fetched server-side with the model's API key in the
Authorization header. Without validation, an admin (or anyone who can reach the
model form, e.g. via a future bug) could point it at an internal address
(169.254.169.254 metadata, localhost services, RFC1918 hosts) and have the
server make authenticated requests there, leaking the bearer token or pivoting
into the internal network.

This module enforces https/http only and blocks hosts that resolve to private,
loopback, link-local, or reserved IP ranges. Set ALLOW_PRIVATE_ENDPOINTS=true to
opt out (e.g. for a self-hosted model on the LAN you trust).
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

ALLOW_PRIVATE = os.getenv("ALLOW_PRIVATE_ENDPOINTS", "false").lower() == "true"


class UnsafeURLError(ValueError):
    """Raised when a URL is not allowed by the SSRF policy."""


def _ip_is_blocked(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def validate_endpoint(url: str) -> str:
    """Validate a base URL for outbound LLM calls. Returns the URL or raises.

    Checks scheme and that every resolved IP for the host is publicly routable.
    """
    if not url or not isinstance(url, str):
        raise UnsafeURLError("URL is required")

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError("URL scheme must be http or https")
    if not parsed.hostname:
        raise UnsafeURLError("URL must include a host")

    if ALLOW_PRIVATE:
        return url

    host = parsed.hostname
    # Resolve all addresses; block if ANY resolves into a disallowed range
    # (defends against DNS records that point at internal IPs).
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as e:
        raise UnsafeURLError(f"could not resolve host: {e}") from e

    for info in infos:
        ip = info[4][0]
        try:
            if _ip_is_blocked(ip):
                raise UnsafeURLError(
                    f"host {host} resolves to a non-public address ({ip}); "
                    "set ALLOW_PRIVATE_ENDPOINTS=true to allow internal endpoints"
                )
        except ValueError as e:
            if isinstance(e, UnsafeURLError):
                raise
            raise UnsafeURLError(f"invalid resolved address: {ip}") from e

    return url
