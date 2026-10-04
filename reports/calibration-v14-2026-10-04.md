# Existing-task calibration

Snapshot: 2026-10-04T10:07:35.596887+00:00

Descriptive evidence only. Historical scores are unchanged; no task is automatically pruned.

## Suite b11260da421820b5

Evidence: preliminary; 2 models.

| Run | Model | Strict passes | Balanced success | Completion/format failures | Other contract failures |
|---|---|---:|---:|---:|---:|
| 80 | Qwen3.8-Flash-Next-FP8 | 258/315 | 83.78% | 24 | 23 |
| 81 | Qwen3.8-27B.long | 254/315 | 83.11% | 26 | 21 |

Capability task patterns: {"all_observed_pass": 227, "mixed_observed": 52, "no_observed_pass": 33}

### Families with at least one observed failure

| Family | Passes/observations | Completion/format failures | Other contract failures |
|---|---:|---:|---:|
| Q11-transaction-consistency | 3/8 | 0 | 5 |
| Q12-queue-linearizability | 0/8 | 0 | 8 |
| Q14-alias-preimages | 1/8 | 0 | 7 |
| Q9-atomic-settlement | 1/8 | 7 | 0 |
| R8A-AC2-01 | 1/2 | 1 | 0 |
| R8A-AC2-02 | 1/2 | 1 | 0 |
| R8A-AC2-07 | 1/2 | 1 | 0 |
| Q11-adaptive-minimax | 5/8 | 0 | 1 |
| R8A-H5-AU-ledger | 7/8 | 0 | 0 |
| R8A-S6-policy | 2/6 | 1 | 2 |
| R8A-H5-CL-worlds | 7/8 | 0 | 0 |
| Q10-unicode-revisions | 2/8 | 6 | 0 |
| R8A-H5-CG-jsonl | 0/2 | 2 | 0 |
| R8A-H5-CG-routes | 1/2 | 1 | 0 |
| R8A-SP1-CR-01 | 1/2 | 0 | 1 |
| Q9-consent-constrained | 7/8 | 0 | 0 |
| Q10-robust-portfolio | 5/8 | 3 | 0 |
| Q9-liquidity-waterfall | 5/8 | 0 | 3 |
| R8A-S6-reshape | 3/6 | 2 | 1 |
| I9-permission-replanning | 6/8 | 0 | 2 |
| interactive-document | 4/8 | 4 | 0 |
| Q9-deadline-exception | 2/8 | 0 | 0 |
| R8A-SP1-LE-01 | 1/2 | 0 | 0 |
| Q12-public-knowledge | 7/8 | 0 | 1 |
| Q14-grid-ambiguity | 7/8 | 1 | 0 |
| Q9-unsat-cores | 5/8 | 0 | 3 |
| R8A-LR2-09 | 1/2 | 0 | 0 |
| Q10-context-relational-revisions | 7/8 | 1 | 0 |
| Q10-context-shipment-aggregation | 0/8 | 3 | 5 |
| Q9-selective-report | 6/8 | 0 | 0 |
| R8A-MR2-01 | 0/2 | 2 | 0 |
| R8A-MR2-06 | 1/2 | 1 | 0 |
| R8A-S6-diagnosis | 3/6 | 0 | 3 |
| Q10-sql-null-cardinality | 7/8 | 0 | 1 |
| R8A-RC2-07 | 1/2 | 0 | 0 |
| R8A-SP1-SE-08 | 0/2 | 0 | 0 |
| Q9-dependency-regression | 1/8 | 7 | 0 |
| Q9-escaped-parser | 4/8 | 4 | 0 |
| Q9-inode-rollback | 7/8 | 0 | 1 |
| R8A-H5-TN-randomization | 6/8 | 1 | 0 |
| Q13-typed-tool-trace | 7/8 | 0 | 0 |
| R8A-S6-toolplan | 5/6 | 0 | 0 |
| Q13-evidence-audit | 6/8 | 1 | 0 |
| R8A-H5-TH-worlds | 7/8 | 0 | 0 |

### Paired observations

- Runs 80 and 81: 227 both pass, 33 both fail, 28 left-only passes, 24 right-only passes (capability tasks).

## Limits

- Patterns describe only this observed model panel; all-pass does not establish saturation.
- Variants and seeds are task instances, not repeated model runs.
- Budget, missing-answer and contract failures cannot establish semantic difficulty.
- Source correctness gates and adjudication of ambiguous prompts remain necessary.
- Historical single-call protocols stay distinct from bounded finalization protocols.
- Model identity uses recorded requested IDs; gateway aliases and provider revisions may obscure actual identity.

## Excluded runs

- Run 1: unsupported_report_schema
- Run 2: unsupported_report_schema
- Run 4: unsupported_report_schema
- Run 5: unsupported_report_schema
- Run 6: unsupported_report_schema
- Run 8: unsupported_report_schema
- Run 9: unsupported_report_schema
- Run 10: unsupported_report_schema
- Run 11: unsupported_report_schema
- Run 12: unsupported_report_schema
- Run 13: unsupported_report_schema
- Run 14: unsupported_report_schema
- Run 15: unsupported_report_schema
- Run 16: unsupported_report_schema
- Run 17: unsupported_report_schema
- Run 19: unsupported_report_schema
- Run 20: unsupported_report_schema
- Run 21: unsupported_report_schema
- Run 25: unsupported_report_schema
- Run 26: unsupported_report_schema
- Run 27: unsupported_report_schema
- Run 28: unsupported_report_schema
- Run 29: unsupported_report_schema
- Run 31: unsupported_report_schema
- Run 33: unsupported_report_schema
- Run 34: unsupported_report_schema
- Run 44: unsupported_report_schema
- Run 45: unsupported_report_schema
- Run 46: unsupported_report_schema
- Run 47: unsupported_report_schema
- Run 48: unsupported_report_schema
- Run 49: unsupported_report_schema
- Run 50: unsupported_report_schema
- Run 51: unsupported_report_schema
- Run 52: unsupported_report_schema
- Run 53: unsupported_report_schema
- Run 54: unsupported_report_schema
- Run 55: unsupported_report_schema
- Run 56: unsupported_report_schema
- Run 57: unsupported_report_schema
- Run 58: unsupported_report_schema
- Run 59: unsupported_report_schema
- Run 62: unsupported_report_schema
- Run 65: unsupported_report_schema
- Run 67: different_requested_suite
- Run 82: run_not_completed
- Run 83: run_not_completed
