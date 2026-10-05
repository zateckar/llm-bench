"""What a run measures: quality, performance or both.

``mode`` says what runs; ``performance`` names the performance test. New runs
use the one staged ``standard`` test (docs/design-consolidation.md). The
former tests ``fixed``, ``sweep`` and ``load`` and the former modes ``sweep``
and ``load`` still describe historical and already queued runs; a stored
performance run without a ``performance`` key is a historical fixed workload.
"""

MODES = ("quality", "performance", "both")
LEGACY_KINDS = ("fixed", "sweep", "load")
PERFORMANCE_KINDS = ("standard", *LEGACY_KINDS)
LEGACY_MODES = ("sweep", "load")
PERFORMANCE_LABELS = {"standard": "Performance · latency, context, capacity", "fixed": "Fixed workload",
                      "sweep": "Context & reasoning sweep", "load": "Open-loop load"}


def normalise(mode="both", performance=None):
    """``(runs quality, performance kind or None)`` for current and stored options.

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
        raise ValueError("Unknown performance test")
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

    from app.benchmarking.suites import HISTORICAL_SUITE
    from app.services.scorecard import suite_label

    try:
        options = json.loads(run_options_json or "null")
    except (TypeError, ValueError):
        return None
    if not isinstance(options, dict):
        return None
    return label(options, suite_label(options.get("suite") or HISTORICAL_SUITE))


def label(options, suite_label=None):
    """Short description such as ``Quality · Standard quality suite + Performance · …``."""
    quality, kind = parts(options)
    items = []
    if quality:
        items.append(f"Quality · {suite_label}" if suite_label else "Quality")
    if kind:
        items.append(PERFORMANCE_LABELS[kind])
    return " + ".join(items)
