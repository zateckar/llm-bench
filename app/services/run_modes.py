"""What a run measures: quality, performance or both (docs/design-run-modes.md).

``mode`` says what runs; ``performance`` says which performance test runs. A
fixed workload is implicit, so historical option snapshots stay identical.
The former performance-only modes ``sweep`` and ``load`` are still accepted:
queued runs, saved plans and API callers keep working.
"""

MODES = ("quality", "performance", "both")
PERFORMANCE_KINDS = ("fixed", "sweep", "load")
LEGACY_MODES = ("sweep", "load")
PERFORMANCE_LABELS = {"fixed": "Fixed workload", "sweep": "Context & reasoning sweep", "load": "Open-loop load"}


def normalise(mode="both", performance=None):
    """``(runs quality, performance kind or None)`` for current and legacy options.

    Raises ValueError for unknown or contradictory values."""
    if mode in LEGACY_MODES:
        if performance not in (None, mode):
            raise ValueError("The run mode and the performance test disagree")
        mode, performance = "performance", mode
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("Choose quality, performance or both")
    if mode == "quality":
        if performance is not None:
            raise ValueError("A performance test applies only to runs that measure performance")
        return True, None
    if performance is None:
        performance = "fixed"
    if not isinstance(performance, str) or performance not in PERFORMANCE_KINDS:
        raise ValueError("Choose a fixed workload, a context sweep or an open-loop load test")
    return mode == "both", performance


def parts(options):
    """``normalise`` for stored options; unreadable options count as the historical default."""
    options = options if isinstance(options, dict) else {}
    try:
        return normalise(options.get("mode", "both"), options.get("performance"))
    except ValueError:
        return True, "fixed"


def describe(run_options_json):
    """Label for a stored run, with the quality suite's name; None without stored options."""
    import json

    from app.benchmarking.suites import DEFAULT_SUITE
    from app.services.scorecard import suite_label

    try:
        options = json.loads(run_options_json or "null")
    except (TypeError, ValueError):
        return None
    if not isinstance(options, dict):
        return None
    return label(options, suite_label(options.get("suite") or DEFAULT_SUITE))


def label(options, suite_label=None):
    """Short description such as ``Quality · rigorous + Open-loop load``."""
    quality, kind = parts(options)
    items = []
    if quality:
        items.append(f"Quality · {suite_label}" if suite_label else "Quality")
    if kind:
        items.append(PERFORMANCE_LABELS[kind])
    return " + ".join(items)
