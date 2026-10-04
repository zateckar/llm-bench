# V14 calibration contract review

The accompanying [snapshot](calibration-v14-2026-10-04.md) and
[per-question data](calibration-v14-2026-10-04.json) describe completed runs 80/81
on v14 suite `b11260da421820b5`. Protocols and every current v14 question fingerprint
matched before any source changes. The database was opened read-only; historical
responses, scores, runs and queue entries were not edited.

Run 80 (Qwen3.8-Flash-Next-FP8) passed 258/315, balanced score 83.7785%.
Run 81 (Qwen3.8-27B.long) passed 254/315, balanced score 83.1072%.
Capability-only pairs: 227 both pass, 33 both fail, 28 run-80-only passes,
24 run-81-only passes. The 0.6713 percentage-point difference does not establish
a reliable ranking with one run per model and unavailable family-bootstrap
intervals. There are 24/26 completion or formatting failures, respectively;
23/21 additional failures have contract score zero. A zero contract score does
not automatically establish an ambiguous prompt or a correct semantic answer.

## Audited representation ambiguities

| Family | Problem found in emitted prompt | Historical matches before/after diagnostic representation mapping | Future contract correction |
|---|---|---:|---|
| Queue linearizability | Repair descriptions never name `order`; models reasonably reuse the outer `canonical_order` key. Some also reuse `linearization_count` for `count`. | 0/8 to 4/8 | Explicitly name every nested key and its type |
| Transaction consistency | Edge notation `Ti->Tj` is explained, but the required array-pair representation is omitted. | 3/8 to 7/8 | Require two-element `[source,target]` arrays |
| Shipment aggregation | Evidence is called a winning record/revision; both models return the full objects instead of the required record IDs. | 0/8 to 4/8 | Require an array of record-ID strings |
| Fuel-constrained route | `path: "S...T"` does not explicitly forbid separators. One otherwise matching answer uses `S-B-C-T`. | 1/2 to 2/2 | Require concatenated node IDs without separators |

The diagnostic mapping changes only representation: rename the two repair keys,
split arrow-form edges, project evidence objects to their existing `record` IDs,
and remove route hyphens. It never changes arithmetic, orders, membership,
timestamps, counts or conclusions. The full strict evaluator is then applied
against the unchanged answer keys. Thirteen additional responses match after
this audit mapping. This is counterfactual adjudication evidence, not a score
correction or a v15 model result; mapped representations do not prove that every
omitted field or full evidence object was itself correct.

## Decisions

- Clarify the 13 affected prompts in v15, keep their answer values and strict graders.
- Preserve all existing tasks: 227 all-observed-pass capability tasks are candidates
  for future saturation checks, not proven saturated tasks.
- Queue and shipment families' observed zero pass rates cannot be presented as
  evidence of extreme semantic difficulty while these ambiguities remain.
- Code-generation, settlement, Unicode revision and dependency failures dominated
  by truncation need matched bounded-finalization runs before difficulty conclusions.
- Retain difficult content cases, including alias-sensitive execution and policy
  reasoning, for a broader repeated panel. Avoid tuning tasks to just these two models.
- Require at least two models and three fully scored repeats per model before a
  repeated panel is available. Review contracts, independent oracles and near-miss
  rejection separately; no automatic task removal or difficulty relabeling occurs.
- Add original behavioral reconstruction as a new capability category. Its source
  correctness is verified, and its empirical admission remains unmeasured.

The older v14 runs used one-call generation. Current bounded finalization adds
a second call only on truncation or reasoning-only output. These protocols must
be calibrated separately even when the question bank matches.
