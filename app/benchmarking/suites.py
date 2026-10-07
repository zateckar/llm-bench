"""Registry of quality suites.

New runs use the one built-in ``standard`` suite (docs/design-consolidation.md)
or an uploaded use-case suite. The four formerly separate built-in suites are
retired: no longer offered, but still loadable so historical runs, reports,
studies and comparisons keep working. Reports of different suites are never
paired.
"""

from dataclasses import dataclass
from typing import Callable

DEFAULT_SUITE = "standard"
# Runs and reports without a stored suite name predate named suites.
HISTORICAL_SUITE = "rigorous"


@dataclass(frozen=True)
class SuiteDef:
    name: str
    label: str
    description: str
    revision: Callable[[], str]
    load: Callable[[], list]
    provenance: Callable[[], dict]
    # Protocol execution block for reports; receives the client's output cap.
    execution: Callable[[int], dict]
    # Extra client protocol fields recorded only for this suite.
    client_extra: Callable[[], dict]


def _rigorous():
    from app.benchmarking import quality_suite
    from app.benchmarking.quality_protocol import protocol

    return SuiteDef(
        name="rigorous",
        label="Rigorous capability suite",
        description="Fixed capability suite with deterministic grading.",
        revision=lambda: quality_suite.REVISION,
        load=quality_suite.load_questions,
        provenance=quality_suite.provenance,
        execution=protocol,
        client_extra=dict,
    )


def _tool_conformance():
    from app.benchmarking import tool_suite
    from app.benchmarking.llm_client import TOOL_CLIENT_REVISION
    from app.benchmarking.tool_protocol import protocol

    return SuiteDef(
        name="tool-conformance",
        label="Tool-calling & structured-output conformance",
        description="Native tools, tool_choice, parallel calls and response_format "
                    "against this deployment's parsers and templates.",
        revision=lambda: tool_suite.REVISION,
        load=tool_suite.load_questions,
        provenance=tool_suite.provenance,
        execution=lambda _max_tokens: protocol(tool_suite.REVISION),
        client_extra=lambda: {"native_tools": TOOL_CLIENT_REVISION},
    )


def _assistant_open():
    from app.benchmarking import open_suite
    from app.benchmarking.quality_protocol import protocol

    return SuiteDef(
        name="assistant-open",
        label="Open-ended assistant requests",
        description="Everyday workplace requests (en/cs/de) without an answer key; "
                    "answers are compared in blind A/B studies.",
        revision=lambda: open_suite.REVISION,
        load=open_suite.load_questions,
        provenance=open_suite.provenance,
        execution=protocol,
        client_extra=dict,
    )


def _safety_language():
    from app.benchmarking import quality_protocol, safety_suite, tool_protocol
    from app.benchmarking.llm_client import TOOL_CLIENT_REVISION

    return SuiteDef(
        name="safety-language",
        label="Safety & language adherence",
        description="Answer language (en/cs/sk/de), prompt and tool-result injection, confidentiality, "
                    "scope policies and over-refusal, all graded deterministically.",
        revision=lambda: safety_suite.REVISION,
        load=safety_suite.load_questions,
        provenance=safety_suite.provenance,
        # Text items use the bounded protocol; tool-result items run natively.
        execution=lambda max_tokens: {**quality_protocol.protocol(max_tokens),
                                      "native": tool_protocol.protocol(safety_suite.REVISION)},
        client_extra=lambda: {"native_tools": TOOL_CLIENT_REVISION},
    )


def _standard():
    from app.benchmarking import quality_protocol, standard_suite, tool_protocol
    from app.benchmarking.llm_client import TOOL_CLIENT_REVISION

    return SuiteDef(
        name="standard",
        label="Standard quality suite",
        description="Compact bank of retained reasoning, tool and safety challenges, plus "
                    "open-ended requests answered for blind A/B studies; universally passed panel tasks are retired.",
        revision=lambda: standard_suite.REVISION,
        load=standard_suite.load_questions,
        provenance=standard_suite.provenance,
        # Text tasks use the bounded protocol; tool tasks run natively.
        execution=lambda max_tokens: {**quality_protocol.protocol(max_tokens),
                                      "native": tool_protocol.protocol(standard_suite.REVISION)},
        client_extra=lambda: {"native_tools": TOOL_CLIENT_REVISION},
    )


_FACTORIES = {"standard": _standard, "rigorous": _rigorous, "tool-conformance": _tool_conformance,
              "assistant-open": _assistant_open, "safety-language": _safety_language}
# Offered for new runs, canaries and scorecard gates.
SUITE_NAMES = ("standard",)
# Merged into the standard suite; kept for historical runs only.
RETIRED_SUITES = ("rigorous", "tool-conformance", "assistant-open", "safety-language")


def is_builtin(name) -> bool:
    return name in _FACTORIES


def get_suite(name=None) -> SuiteDef:
    # A missing name means a historical rigorous run, not a new default.
    key = name or HISTORICAL_SUITE
    if key.startswith("usecase:"):
        # Uploaded suites live in the database; the key pins one version.
        from app.benchmarking.usecase_suites import suite_def

        return suite_def(key)
    if key not in _FACTORIES:
        raise ValueError(f"Unknown quality suite {key!r}")
    return _FACTORIES[key]()


def suite_name(report) -> str:
    """Suite of a saved report; reports without a name are rigorous-suite reports."""
    return ((report or {}).get("suite") or {}).get("name") or HISTORICAL_SUITE


def suite_choices():
    return [{"name": name, "label": get_suite(name).label,
             "description": get_suite(name).description} for name in SUITE_NAMES]
