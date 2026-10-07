"""Lossless, versioned compression of large SQLite payloads.

Declared TEXT converters keep callers and JSON downloads working with strings.
Only payload columns are compressed; searchable identifiers and error details
remain plain text. Legacy uncompressed rows are readable on the same connection.
"""

import sqlite3
import zlib
import lzma
import json

MAGIC = b"LLMBZ1\x00"
LZMA_MAGIC = b"LLMBX1\x00"
DETECT_TYPES = sqlite3.PARSE_DECLTYPES


def pack_text(value):
    if not isinstance(value, str):
        return value
    raw = value.encode("utf-8")
    # Declared-type converters receive bytes for TEXT and BLOB alike, and
    # sqlite3 bypasses converters for empty values (returning None). Escape
    # empty strings and literal marker prefixes so new writes stay lossless.
    escape_marker = not raw or raw.startswith((MAGIC, LZMA_MAGIC))
    if len(value) < 512 and not escape_marker:
        return value
    packed = MAGIC + zlib.compress(raw, 6)
    # A larger dictionary preserves repetition across large fixtures and audit
    # records that falls outside zlib's 32 KiB window. Small writes stay cheap.
    if len(raw) >= 65536:
        candidate = LZMA_MAGIC + lzma.compress(raw, preset=1)
        if len(candidate) < len(packed):
            packed = candidate
    return packed if escape_marker or len(packed) < len(raw) * 0.85 else value


def unpack_text(value):
    if isinstance(value, bytes):
        if value.startswith(MAGIC):
            value = zlib.decompress(value[len(MAGIC):])
        elif value.startswith(LZMA_MAGIC):
            value = lzma.decompress(value[len(LZMA_MAGIC):])
        return value.decode("utf-8")
    return value


def cancel_performance(value):
    """SQLite write helper; JSON SQL functions cannot inspect compressed blobs."""
    try:
        report = json.loads(unpack_text(value) or "null")
    except (TypeError, ValueError):
        return value
    if not isinstance(report, dict):
        return value
    if (report.get("schema_version"), report.get("kind")) not in {
            (4, "context_sweep"), (5, "open_loop"), (6, "staged")}:
        return value
    report["cancelled"] = True
    if report["schema_version"] in {4, 6}:
        report["finished"] = False
    return pack_text(json.dumps(report, separators=(",", ":")))


sqlite3.register_converter("TEXT", unpack_text)
# Preserve the application's ISO timestamps as strings. sqlite3's deprecated
# default TIMESTAMP converter cannot parse our timezone-aware ISO representation.
sqlite3.register_converter("TIMESTAMP", unpack_text)
