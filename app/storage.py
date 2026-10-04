"""Lossless, versioned compression of large SQLite payloads.

Declared TEXT converters keep callers and JSON downloads working with strings.
Only payload columns are compressed; searchable identifiers and error details
remain plain text. Legacy uncompressed rows are readable on the same connection.
"""

import sqlite3
import zlib

MAGIC = b"LLMBZ1\x00"
DETECT_TYPES = sqlite3.PARSE_DECLTYPES


def pack_text(value):
    if not isinstance(value, str) or len(value) < 512:
        return value
    raw = value.encode("utf-8")
    packed = MAGIC + zlib.compress(raw, 6)
    return packed if len(packed) < len(raw) * 0.85 else value


def unpack_text(value):
    if isinstance(value, bytes):
        if value.startswith(MAGIC):
            value = zlib.decompress(value[len(MAGIC):])
        return value.decode("utf-8")
    return value


sqlite3.register_converter("TEXT", unpack_text)
# Preserve the application's ISO timestamps as strings. sqlite3's deprecated
# default TIMESTAMP converter cannot parse our timezone-aware ISO representation.
sqlite3.register_converter("TIMESTAMP", unpack_text)
# Preserve the application's ISO timestamps as strings. sqlite3's deprecated
# default TIMESTAMP converter cannot parse our timezone-aware ISO representation.
sqlite3.register_converter("TIMESTAMP", unpack_text)
