"""Registry of separately versioned quality suites.

Each suite owns its question inventory, provenance and execution protocol.
Suites never share fingerprints, so reports from different suites are never
paired, and adding a suite cannot change an existing suite's identity.
"""

from dataclasses import dataclass
from typing import Callable

DEFAULT_SUITE = "rigorous"


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


_FACTORIES = {"rigorous": _rigorous, "tool-conformance": _tool_conformance,
              "assistant-open": _assistant_open, "safety-language": _safety_language}
SUITE_NAMES = tuple(_FACTORIES)


def get_suite(name=None) -> SuiteDef:
    key = name or DEFAULT_SUITE
    if key.startswith("usecase:"):
        # Uploaded suites live in the database; the key pins one version.
        from app.benchmarking.usecase_suites import suite_def

        return suite_def(key)
    if key not in _FACTORIES:
        raise ValueError(f"Unknown quality suite {key!r}")
    return _FACTORIES[key]()


def suite_name(report) -> str:
    """Suite of a saved report; historical reports are rigorous-suite reports."""
    return ((report or {}).get("suite") or {}).get("name") or DEFAULT_SUITE


def suite_choices():
    return [{"name": name, "label": get_suite(name).label,
             "description": get_suite(name).description} for name in SUITE_NAMES]
