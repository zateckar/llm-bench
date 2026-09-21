#!/usr/bin/env python3
"""Machine-readable backlog for the next harder benchmark profile.

This file describes candidate families; it does not add unverified questions to
the default suite. A candidate becomes benchmark content only after an
independent oracle, adversarial mutations, and a calibrated pilot accept it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


HARDENING_FAMILIES: dict[str, list[dict[str, str]]] = {
    "Logical Reasoning": [
        {"id": "minimal-counterexamples", "demand": "constrained models with minimal counterexamples", "contract": "exact JSON models plus counterexample witnesses"},
        {"id": "nonmonotonic-updates", "demand": "nonmonotonic rule updates and all optimal explanations", "contract": "ordered labels and evidence IDs"},
    ],
    "Mathematical Reasoning": [
        {"id": "selection-probability", "demand": "exact dependent probabilities with selection bias", "contract": "reduced rational fields and conditioning event"},
        {"id": "discrete-certificates", "demand": "constrained discrete optimization with certificates and all ties", "contract": "objective, feasibility certificate, and sorted optima"},
    ],
    "Reading Comprehension": [
        {"id": "exception-precedence", "demand": "multi-document exceptions and precedence", "contract": "answer fields paired with source IDs"},
        {"id": "bitemporal-evidence", "demand": "bitemporal contradictions with exact supporting sources", "contract": "snapshots, voids, and signed totals"},
    ],
    "Classification": [
        {"id": "compositional-abstention", "demand": "compositional multi-label decisions with abstention", "contract": "labels, abstentions, and minimum evidence"},
        {"id": "decision-reversal", "demand": "minimal evidence changes that reverse a classification", "contract": "original/revised labels and minimal change set"},
    ],
    "Factual Knowledge": [
        {"id": "mechanism-counterfactual", "demand": "reference-grounded mechanisms and counterfactuals", "contract": "staged calculation with supplied references"},
        {"id": "cross-source-reconciliation", "demand": "cross-source factual reconciliation with missing evidence", "contract": "claim status and evidence provenance"},
    ],
    "Truthfulness": [
        {"id": "underdetermined-claims", "demand": "answerable versus underdetermined linked claims", "contract": "confirmed/refuted/unknown with local evidence"},
        {"id": "conflicting-sources", "demand": "conflicting sources and false-premise rejection", "contract": "claim status, source priority, and uncertainty"},
    ],
    "Code Generation": [
        {"id": "stateful-parser", "demand": "stateful parsers with malformed inputs and exact arithmetic", "contract": "executable fixtures with typed error cases"},
        {"id": "idempotent-transform", "demand": "transactional transformations with idempotency and invariants", "contract": "replay traces and invariant checks"},
    ],
    "Advanced Coding": [
        {"id": "persistent-structures", "demand": "persistent/versioned structures under mixed operations", "contract": "state snapshots and version isolation"},
        {"id": "multiobjective-graph", "demand": "multi-objective graph and interval algorithms under scale limits", "contract": "optimal value, tie set, and resource bound"},
    ],
    "Code Review": [
        {"id": "cross-file-dataflow", "demand": "cross-file dataflow defects and minimal fixes", "contract": "finding IDs, evidence lines, and fix scope"},
        {"id": "migration-false-alarms", "demand": "migration/concurrency traces containing convincing false alarms", "contract": "confirmed/refuted/unknown claims with trace"},
    ],
    "Terminal Algorithms": [
        {"id": "external-memory", "demand": "streaming and external-memory algorithms", "contract": "commands plus memory and ordering constraints"},
        {"id": "resource-schedule", "demand": "deterministic graph and scheduling under resource limits", "contract": "schedule, proof checks, and tie-breaking"},
    ],
    "Terminal Debugging": [
        {"id": "interacting-faults", "demand": "interacting faults requiring an executable patch", "contract": "patch plus regression fixture results"},
        {"id": "dependency-regression", "demand": "regression preservation across dependency and configuration boundaries", "contract": "diagnosis, changed files, and test output"},
    ],
    "Needle Retrieval": [
        {"id": "indirect-multihop", "demand": "indirect multi-hop retrieval across similar identifiers", "contract": "answer with complete evidence chain"},
        {"id": "revision-distractors", "demand": "retrieval after revisions, deletions, and adversarial distractors", "contract": "current record, tombstone, and source IDs"},
    ],
    "Long Context Coherence": [
        {"id": "nested-transactions", "demand": "nested transactions and delayed references", "contract": "ordered checkpoints and explicit null masks"},
        {"id": "cross-document-invariants", "demand": "cross-document invariants after conflicting updates", "contract": "final state plus violated invariant IDs"},
    ],
    "Summarization": [
        {"id": "event-reversals", "demand": "evidence-linked reconciliation of events and reversals", "contract": "auditable totals and event provenance"},
        {"id": "contradictory-compact", "demand": "compact structured summaries with omissions and contradictions checked", "contract": "required fields, omissions, and contradiction flags"},
    ],
    "Tool Using": [
        {"id": "cost-prerequisites", "demand": "cost-optimal plans under prerequisites and partial observability", "contract": "call sequence, cost, and tie-break proof"},
        {"id": "stale-recovery", "demand": "recovery planning after stale reads and ambiguous tool outcomes", "contract": "idempotency key, recovery calls, and final state"},
    ],
    "Agentic Use Cases": [
        {"id": "adaptive-risk", "demand": "adaptive policies under explicit risk and cost constraints", "contract": "policy tree and stopping condition"},
        {"id": "rollback-workflow", "demand": "multi-step workflows with rollback and evidence-dependent stopping", "contract": "actions, authorization gates, and rollback trace"},
    ],
    "Interactive Tool Use": [
        {"id": "resource-races", "demand": "multi-resource races and uncertain commit results", "contract": "simulated transcript and authorized final state"},
        {"id": "permission-replanning", "demand": "paginated scope and permission changes requiring safe replanning", "contract": "page coverage, dry-run evidence, and mutation guard"},
    ],
    "Security": [
        {"id": "precondition-chain", "demand": "multi-hop exploitability with necessary preconditions", "contract": "finding graph and disqualifying overclaims"},
        {"id": "least-privilege", "demand": "least-privilege remediation and verifiable negative findings", "contract": "remediation diff and checks for absent paths"},
    ],
    "Terminal System Admin": [
        {"id": "config-precedence", "demand": "configuration precedence and dependency diagnosis", "contract": "commands, effective values, and causal chain"},
        {"id": "service-recovery", "demand": "recovery sequences with service-state invariants in a simulator", "contract": "ordered actions and invariant checkpoints"},
    ],
    "Terminal File Operations": [
        {"id": "atomic-unicode-links", "demand": "atomic transforms over duplicates, Unicode, and links", "contract": "exact directory state and safe command sequence"},
        {"id": "interrupted-rollback", "demand": "interrupted operations with rollback and exact directory-state checks", "contract": "recovery commands and before/after manifest"},
    ],
    "Terminal Science": [
        {"id": "confounded-estimand", "demand": "confounded experiments with valid estimands", "contract": "estimand, adjustment, and limitation"},
        {"id": "stable-inference", "demand": "exact inference and numerical analysis with stability constraints", "contract": "exact result, tolerance, and conditioning check"},
    ],
    "Instruction Following": [
        {"id": "exceptional-transform", "demand": "composed transformations with interacting exceptions", "contract": "ordered structured output and source indices"},
        {"id": "stateful-constraints", "demand": "global and local output constraints with state-dependent requirements", "contract": "schema, counts, and cross-field invariants"},
    ],
    "Creative Writing": [
        {"id": "continuity-constraints", "demand": "narrative continuity under interacting measurable constraints", "contract": "constraint compliance only; no artistic-quality claim"},
        {"id": "fact-preserving-revision", "demand": "controlled revisions preserving facts, viewpoint, and form", "contract": "continuity and form checks"},
    ],
    "Translation": [
        {"id": "scope-negation", "demand": "logical scope, negation, and reference preservation", "contract": "meaning representation with scope markers"},
        {"id": "terminology-ambiguity", "demand": "ambiguity-aware meaning preservation under supplied terminology", "contract": "alternatives, disambiguation, and omission checks"},
    ],
    "Language Translations": [
        {"id": "technical-modality", "demand": "technical/legal terminology and modality in all six directions", "contract": "direction-specific terminology and modality fields"},
        {"id": "units-dates-reference", "demand": "units, dates, gender/reference, and exception-scope preservation", "contract": "normalized semantic fields plus rendered translation"},
    ],
    "Ethical Reasoning": [
        {"id": "policy-counterfactual", "demand": "explicit-policy conflicts with counterfactual consistency", "contract": "policy application, trade-off, and alternatives"},
        {"id": "infeasible-fairness", "demand": "consent/fairness allocations with infeasibility and all valid alternatives", "contract": "feasibility status and tied optima"},
    ],
    "Legal": [
        {"id": "jurisdiction-exceptions", "demand": "supplied jurisdiction/date-specific rules with exceptions", "contract": "rule, exception, date, and evidence references"},
        {"id": "claim-deadline", "demand": "evidence/claim mapping with deadlines and insufficient-fact cases", "contract": "claim status, deadline, and missing fact"},
    ],
    "Finances": [
        {"id": "stress-covenants", "demand": "liquidity, covenant, and currency constraints under stress", "contract": "timed cash flows, haircuts, and policy shortfall"},
        {"id": "rounding-sensitivity", "demand": "cash-flow reconciliation with timing, exact rounding, and sensitivities", "contract": "conservation rows and sensitivity bounds"},
    ],
    "R&D": [
        {"id": "identifiability-design", "demand": "fault identifiability and optimal experimental design", "contract": "candidate classes, probes, and all tied designs"},
        {"id": "coupled-uncertainty", "demand": "coupled mechanical/control constraints with uncertainty propagation", "contract": "feasibility, bounds, and sensitivity"},
    ],
}


def validate_backlog(backlog: dict[str, list[dict[str, str]]] | None = None) -> None:
    source = backlog or HARDENING_FAMILIES
    if len(source) != 29:
        raise ValueError(f"expected 29 category backlogs, got {len(source)}")
    for category, families in source.items():
        if len(families) < 2:
            raise ValueError(f"{category}: expected at least two candidate families")
        ids = [family.get("id") for family in families]
        if any(not family.get("demand") or not family.get("contract") for family in families):
            raise ValueError(f"{category}: every family needs demand and contract text")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{category}: duplicate family id")


def build_backlog(calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_backlog()
    latest = None
    if calibration and calibration.get("cohorts"):
        latest = sorted(calibration["cohorts"].values(), key=lambda c: c.get("reports", 0))[-1]
    categories = []
    for category, families in HARDENING_FAMILIES.items():
        recommendations = [
            item for item in (latest or {}).get("recommendations", [])
            if item.get("category") == category
        ]
        priority = "high" if recommendations else "baseline"
        categories.append({
            "category": category,
            "priority": priority,
            "recommendations": recommendations,
            "families": families,
        })
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_families_per_category": 2,
        "categories": categories,
        "note": "Candidate descriptions are a hardening backlog, not accepted benchmark items. Each family requires an independent oracle, adversarial mutations, contract validation, and a matched pilot before release.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
    backlog = build_backlog(calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(backlog, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"{len(backlog['categories'])} categories, {sum(len(c['families']) for c in backlog['categories'])} candidate families")
    print(f"High-priority categories: {sum(c['priority'] == 'high' for c in backlog['categories'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
