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

Return values cross the process boundary as JSON so the caller can compare them
*structurally* rather than by ``repr()`` string. Comparing reprs produced false
failures for correct solutions (``2`` vs ``2.0``, a tuple vs a list, dict key
ordering); see ``evaluators.values_equal``.
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
    "decimal", "fractions", "uuid", "base64", "struct", "numbers",
    "unicodedata", "contextlib", "warnings",
    # Data science
    "numpy", "pandas", "scipy", "sklearn", "pytz", "dateutil", "pyarrow",
    # Cloud / IoT
    "boto3", "botocore",
    # Common utilities
    "requests", "httpx",
]

DEFAULT_TIMEOUT = 15  # seconds, wall-clock, enforced by the parent process
MEMORY_LIMIT_MB = 512

# Per-test CPU budget inside the child. A solution with accidental exponential
# blow-up (naive recursive fibonacci on a large n) is a genuine failure, but it
# must not eat the whole wall-clock budget and starve the remaining tests.
PER_TEST_TIMEOUT = 6


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
            resource.setrlimit(getattr(resource, res), (mb, mb))
        except (ValueError, OSError, AttributeError):
            pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no file writes
    except (ValueError, OSError, AttributeError):
        pass


# --- value encoding -------------------------------------------------------
# Encode a returned value so the parent can compare it structurally. Anything
# JSON-representable travels as JSON; everything else degrades to its repr.

def _encode(value, _depth=0):
    if _depth > 12:
        return {"kind": "repr", "repr": repr(value)}
    if value is None or isinstance(value, (bool, int, str)):
        return {"kind": "json", "json": value}
    if isinstance(value, float):
        # inf/nan are not valid JSON; carry them as tagged strings.
        if value != value or value in (float("inf"), float("-inf")):
            return {"kind": "special_float", "value": repr(value)}
        return {"kind": "json", "json": value}
    if isinstance(value, (list, tuple)):
        return {"kind": "seq", "items": [_encode(v, _depth + 1) for v in value]}
    if isinstance(value, (set, frozenset)):
        try:
            items = sorted(value, key=lambda v: (type(v).__name__, repr(v)))
        except Exception:
            items = list(value)
        return {"kind": "set", "items": [_encode(v, _depth + 1) for v in items]}
    if isinstance(value, dict):
        return {
            "kind": "map",
            "items": [[_encode(k, _depth + 1), _encode(v, _depth + 1)]
                      for k, v in value.items()],
        }
    # numpy scalars / anything with a plain numeric or list interface
    for attr in ("tolist", "item"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _encode(fn(), _depth + 1)
            except Exception:
                pass
    return {"kind": "repr", "repr": repr(value)}


class _Timeout(Exception):
    pass


def _run_with_limit(fn, args, kwargs, seconds):
    """Run fn under a best-effort per-call time limit.

    signal.setitimer only exists on POSIX; on Windows we rely on the parent's
    wall-clock timeout instead, which is still a hard bound on the whole job.
    """
    try:
        import signal
        has_alarm = hasattr(signal, "setitimer")
    except ImportError:
        has_alarm = False

    if not has_alarm:
        return fn(*args, **kwargs)

    def _handler(signum, frame):
        raise _Timeout("call exceeded %ss" % seconds)

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def main():
    _apply_limits()
    job = _read_job()
    code = job["code"]
    helper = job.get("helper")
    tests = job["tests"]
    per_test_timeout = job.get("per_test_timeout", 6)

    namespace = {"__builtins__": SAFE_BUILTINS, "__name__": "__benchmark__", "__doc__": None}
    results = []
    try:
        exec(compile(code, "<solution>", "exec"), namespace)
    except Exception as e:
        print(json.dumps({"error": "solution failed to load: %s: %s" % (type(e).__name__, e)}))
        return
    try:
        if helper:
            exec(compile(helper, "<harness>", "exec"), namespace)
    except Exception as e:
        print(json.dumps({"error": "harness failed to load: %s: %s" % (type(e).__name__, e)}))
        return

    for t in tests:
        fn_name = t["function"]
        args = t.get("args") or []
        kwargs = t.get("kwargs") or {}
        func = namespace.get(fn_name)
        if func is None:
            results.append({"ok": False, "error": "name %r is not defined" % fn_name})
            continue
        if not callable(func):
            results.append({"ok": False, "error": "%r is not callable" % fn_name})
            continue
        try:
            out = _run_with_limit(func, args, kwargs, per_test_timeout)
            results.append({"ok": True, "value": _encode(out), "repr": repr(out)[:400]})
        except _Timeout as e:
            results.append({"ok": False, "error": "timeout: %s" % e})
        except RecursionError:
            results.append({"ok": False, "error": "RecursionError (unbounded recursion)"})
        except Exception as e:
            results.append({"ok": False, "error": "%s: %s" % (type(e).__name__, e)})

    print(json.dumps({"results": results}))

main()
'''


def run_code_tests(
    code: str,
    tests: list[dict],
    helper: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    per_test_timeout: int = PER_TEST_TIMEOUT,
) -> dict:
    """Execute ``code`` against ``tests`` in an isolated subprocess.

    Each test is ``{"function": str, "args": list, "kwargs": dict}``. Returns
    ``{"results": [{"ok": bool, "value": <encoded>, "repr": str, "error": str}, ...]}``
    or ``{"error": "..."}`` on a global failure (timeout, load error, crash).

    ``value`` is a tagged encoding of the returned object (see ``_encode`` in the
    child source) so the caller can compare structurally instead of by repr.
    """
    # The parent's wall-clock timeout must not fire before every test has had
    # its per-test budget, so clamp per-test to a fair share of the total
    # (integer division leaves headroom for the subprocess load/import overhead).
    per_test_timeout = min(per_test_timeout, max(1, timeout // max(1, len(tests))))
    job = json.dumps({
        "code": code,
        "tests": tests,
        "helper": helper,
        "per_test_timeout": per_test_timeout,
    })
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
