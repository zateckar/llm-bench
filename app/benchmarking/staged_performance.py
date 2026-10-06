"""The standard performance test: latency, context and users stages in one report.

See docs/design-consolidation.md. Each stage runs an existing engine unchanged
and stores that engine's own report under ``stages``, so every consumer that
reads a fixed-workload (schema 3), context-sweep (schema 4), open-loop
(schema 5) or sessions (schema 7) report also reads the matching stage.
Historical single-kind reports are treated as a staged report with one stage.
The capacity stage (open-loop) is no longer run; it stays registered so
earlier staged runs keep it.
"""

import copy

REVISION = "standard-performance-v4"
SCHEMA_VERSION = 6
KIND = "staged"

# Every stage a report may contain, in display order:
# (key, label, engine schema, engine kind or None, description)
STAGES = (
    ("latency", "Latency", 3, None,
     "First-token and end-to-end latency, throughput and prefix-cache effect at concurrency 1, 2, 4, 8."),
    ("context", "Context", 4, "context_sweep",
     "Limits by input context and concurrency, up to the deployment's declared context limit: each context "
     "climbs 1, 2, 4, 8, 16, 24, 32, 48, … concurrent requests until latency or output speed is far beyond "
     "the targets, and larger contexts skip the loads that already failed."),
    ("capacity", "Capacity", 5, "open_loop",
     "Sustainable arrival rate under per-class SLOs for an open-loop workload (earlier runs only)."),
    ("users", "Users", 7, "sessions",
     "Simulated users served within their SLO: multi-turn chat and agent sessions with growing history, "
     "searched by user count for each context cap."),
)
STAGE_KEYS = tuple(key for key, *_ in STAGES)
STAGE_LABELS = {key: label for key, label, *_ in STAGES}
# Stages shown only when a report has them (standard-performance-v1 to v3 ran capacity).
RETIRED_STAGES = ("capacity",)
# The stages a new run measures, in order.
RUN_STAGES = tuple(key for key in STAGE_KEYS if key not in RETIRED_STAGES)

CONTEXT_LADDER = (256, 8_192, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576)
MIN_CONTEXT = CONTEXT_LADDER[0]
MAX_CONTEXT = CONTEXT_LADDER[-1]
DEFAULT_CONTEXT = 131_072
CONTEXT_OUTPUT_TOKENS = 4_096
# Chat framing and the sweep's instructions around the measured input.
FRAMING_MARGIN = 512
# Limit search: highest concurrency and the per-request targets that define
# "within targets"; the limit is LIMIT_FACTOR beyond them (perf_sweep.judge).
DEFAULT_CONTEXT_CONCURRENCY = 256
DEFAULT_TARGET_TTFT_MS = 2000
DEFAULT_TARGET_OUTPUT_RATE = 40


def context_concurrency(value):
    from app.benchmarking.perf_sweep import MAX_CONCURRENCY

    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_CONCURRENCY:
        raise ValueError(f"The context-stage concurrency must be an integer between 1 and {MAX_CONCURRENCY}")
    return value


def targets(ttft_ms=None, output_rate=None):
    """Validated per-request targets of the context stage."""
    from app.benchmarking.perf_sweep import Targets

    return Targets(DEFAULT_TARGET_TTFT_MS if ttft_ms is None else ttft_ms,
                   DEFAULT_TARGET_OUTPUT_RATE if output_rate is None else output_rate)


def context_limit(value):
    """Validated context-stage limit in reference tokens."""
    if isinstance(value, bool) or not isinstance(value, int) or not MIN_CONTEXT <= value <= MAX_CONTEXT:
        raise ValueError(f"The context limit must be an integer between {MIN_CONTEXT} and {MAX_CONTEXT:,}")
    return value


def context_limit_for(declared=None, override=None):
    """``(limit, source)``: an explicit override, else the declared limit minus headroom, else 128k."""
    if override is not None:
        return context_limit(override), "override"
    if isinstance(declared, int) and not isinstance(declared, bool):
        usable = declared - CONTEXT_OUTPUT_TOKENS - FRAMING_MARGIN
        if usable >= MIN_CONTEXT:
            return min(usable, MAX_CONTEXT), "declared"
    return DEFAULT_CONTEXT, "default"


def contexts_for(limit):
    """Ladder lengths up to ``limit``, ending exactly at the limit."""
    return tuple(sorted({*(n for n in CONTEXT_LADDER if n <= limit), limit}))


def context_effort(reasoning_effort):
    """The sweep effort matching the model's configured reasoning effort."""
    from app.benchmarking.perf_sweep import EFFORTS

    return reasoning_effort if reasoning_effort in EFFORTS else "default"


def form_info():
    from app.benchmarking.perf_sweep import MAX_CONCURRENCY

    return {"revision": REVISION,
            "stages": [{"key": key, "label": label, "description": description}
                       for key, label, _, _, description in STAGES if key in RUN_STAGES],
            "default_context": DEFAULT_CONTEXT, "min_context": MIN_CONTEXT, "max_context": MAX_CONTEXT,
            "context_output_tokens": CONTEXT_OUTPUT_TOKENS,
            "default_context_concurrency": DEFAULT_CONTEXT_CONCURRENCY,
            "max_context_concurrency": MAX_CONCURRENCY,
            "default_target_ttft_ms": DEFAULT_TARGET_TTFT_MS, "default_target_output_rate": DEFAULT_TARGET_OUTPUT_RATE}


def new_report(model, protocol):
    return {"schema_version": SCHEMA_VERSION, "kind": KIND, "revision": REVISION, "model": model,
            "protocol": {"revision": REVISION, **protocol}, "stages": {}, "running_stage": None,
            "cancelled": False, "finished": False, "stage_errors": {}, "notes": []}


def is_staged(perf):
    return isinstance(perf, dict) and perf.get("schema_version") == SCHEMA_VERSION and perf.get("kind") == KIND


def stage_of(report):
    """Stage key of a single-engine report, or None."""
    if not isinstance(report, dict):
        return None
    for key, _, schema, kind, _ in STAGES:
        if report.get("schema_version") == schema and (kind is None or report.get("kind") == kind):
            return key
    return None


def is_performance_report(perf):
    return is_staged(perf) or stage_of(perf) is not None


def stages_of(perf):
    """``{stage key: engine report}`` in stage order, for staged and historical reports.

    A stop recorded on the staged report marks its running stage as cancelled.
    The engine reports are the stored objects, not copies."""
    if is_staged(perf):
        stages = perf.get("stages") if isinstance(perf.get("stages"), dict) else {}
        result = {key: stages[key] for key in STAGE_KEYS
                  if isinstance(stages.get(key), dict) and stage_of(stages[key]) == key}
        running = perf.get("running_stage")
        if perf.get("cancelled") and running in result:
            result[running]["cancelled"] = True
            if running == "context":
                result[running]["finished"] = False
        return result
    key = stage_of(perf)
    return {key: perf} if key else {}


def stage(perf, key):
    return stages_of(perf).get(key)


def split(perf):
    """Deep copies of the stage reports, safe to change."""
    return copy.deepcopy(stages_of(perf))
