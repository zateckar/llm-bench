"""Out-of-process sandbox for executing model-generated code.

Running untrusted, model-generated Python in the main process is an RCE risk and
can hang the benchmark (infinite loops, runaway memory). This module executes the
code in a separate, short-lived subprocess with:

- a hard wall-clock timeout (the parent kills the child if it overruns),
- a restricted import allowlist and a stripped-down builtins set,
- best-effort CPU/memory/file-size limits via the ``resource`` module on POSIX.

It is NOT a security boundary strong enough for adversarial attackers, but it
contains the common failure modes (loops, accidental file/network access, simple
escapes) and prevents a single test from taking down the runner. For untrusted
input in production, run this inside a container/VM as well.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Modules a solution is allowed to import inside the sandbox.
ALLOWED_IMPORTS = [
    # Standard library
    "math", "collections", "heapq", "json", "re", "itertools", "functools",
    "datetime", "time", "random", "bisect", "array", "copy", "hashlib",
    "string", "queue", "inspect", "io", "os", "pathlib", "typing",
    "abc", "dataclasses", "enum", "textwrap", "operator", "statistics",
    "decimal", "fractions", "uuid", "base64", "struct",
    # Data science
    "numpy", "pandas", "scipy", "sklearn", "pytz", "dateutil", "pyarrow",
    # Cloud / IoT
    "boto3", "botocore",
    # Common utilities
    "requests", "httpx",
]

DEFAULT_TIMEOUT = 10  # seconds, wall-clock, enforced by the parent process
MEMORY_LIMIT_MB = 512


# The child program. It reads a JSON job on stdin and writes a JSON result on
# stdout. Kept as a string so we can launch it with `python -c` without shipping
# an extra file that could be imported accidentally.
_CHILD_SOURCE = r'''
import builtins, json, sys

def _read_job():
    return json.loads(sys.stdin.read())

ALLOWED = set(json.loads(sys.argv[1]))

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in ALLOWED:
        return builtins.__import__(name, globals, locals, fromlist, level)
    raise ImportError("Import of module %r is not allowed in this sandbox." % name)

SAFE_BUILTINS = {
    n: getattr(builtins, n)
    for n in dir(builtins)
    if (not n.startswith("_") or n == "__build_class__") and n not in {
        "eval", "exec", "compile",
        "open", "input", "breakpoint", "exit", "quit",
        "help", "license", "copyright", "credits", "__import__",
    }
}
SAFE_BUILTINS["__import__"] = safe_import

def _apply_limits():
    try:
        import resource
    except ImportError:
        return
    mb = int(sys.argv[2]) * 1024 * 1024
    for res in ("RLIMIT_AS", "RLIMIT_DATA"):
        try:
            soft, hard = getattr(resource, res), None
            resource.setrlimit(getattr(resource, res), (mb, mb))
        except (ValueError, OSError):
            pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no file writes
    except (ValueError, OSError):
        pass

def main():
    _apply_limits()
    job = _read_job()
    code = job["code"]
    helper = job.get("helper")
    tests = job["tests"]

    namespace = {"__builtins__": SAFE_BUILTINS, "__name__": "__main__", "__doc__": None}
    results = []
    try:
        exec(code, namespace)
        if helper:
            exec(helper, namespace)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": "compile/exec failed: %s" % e}))
        return

    for t in tests:
        fn = t["function"]
        args = t["args"]
        try:
            func = namespace[fn]
            out = func(*args)
            results.append({"ok": True, "result": repr(out)})
        except Exception as e:  # noqa: BLE001
            results.append({"ok": False, "error": str(e)})
    print(json.dumps({"results": results}))

main()
'''


def run_code_tests(
    code: str,
    tests: list[dict],
    helper: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Execute ``code`` against ``tests`` in an isolated subprocess.

    Each test is ``{"function": str, "args": list}``. Returns
    ``{"results": [{"ok": bool, "result": "<repr>"|None, "error": str|None}, ...]}``
    or ``{"error": "..."}`` on a global failure (timeout, exec error, crash).

    ``result`` is the ``repr()`` of the returned value; the caller compares it
    against ``repr(expected)`` so values never cross the process boundary as live
    objects.
    """
    job = json.dumps({"code": code, "tests": tests, "helper": helper})
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _CHILD_SOURCE,
             json.dumps(ALLOWED_IMPORTS), str(MEMORY_LIMIT_MB)],
            input=job,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent),
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"subprocess failed: {e}"}

    if proc.returncode != 0 and not proc.stdout.strip():
        msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "non-zero exit"
        return {"error": f"sandbox crashed: {msg}"}

    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": "could not parse sandbox output"}
