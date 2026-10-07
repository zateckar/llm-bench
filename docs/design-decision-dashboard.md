Update 2026-10-07: users gates require a measured result at the exact configured context cap and exclude inherited results. Cache and batching effects need not be monotonic.

# Design: model scorecards and decision profiles

Status: accepted (recommended decisions), 2026-10-05. Item 7 of the evaluation roadmap.

## Problem

The application now produces many kinds of evidence about one model:

- quality on the rigorous suite, tool-calling conformance and use-case suites (single runs and repeat groups);
- closed-loop latency and throughput, context sweeps and open-loop capacity under SLOs;
- blind A/B preferences from people and judges;
- deployment fingerprints, canary status and open monitoring events.

Each lives on its own page, and each page answers its own question. The questions people actually bring are different:

- "Which of our models should the HR assistant use?"
- "Is the new checkpoint good enough to replace the current one for coding agents?"
- "Is what we approved in spring still what is being served?"

Answering them means opening a dozen runs, checking that they are comparable and recent, and remembering whether the deployment changed since. Two capabilities are missing:

1. **One view per model across all evidence**, with its freshness.
2. **Explicit requirements per use case**, evaluated against that evidence, with a recorded decision.

## Decisions

1. **No composite score.** The application never multiplies evidence by weights into one number.
   - **Why:** weights are arbitrary, hide trade-offs (a model can be best at quality and unusable at the needed load) and invite optimizing the weights.
   - **Instead:** each dimension is shown in its own unit with its own uncertainty, and decisions use explicit requirements (decision 5).
2. **Evidence is the latest, not the best.** For every model and evidence key, the scorecard uses the most recent completed run.
   - **Evidence keys:**
     - quality: the suite key (`rigorous`, `tool-conformance`, `usecase:slug@v`, …);
     - capacity: the open-loop workload (preset name, or the workload hash for custom workloads);
     - latency: the closed-loop fixed workload;
     - context: the context sweep.
   - **Never best-of:** picking the best of several runs would turn noise into a ranking.
   - **Repeat groups:** when the latest quality run belongs to a repeat group, the group is the evidence: the mean of its usable runs with the interval of the run spread (with only two usable runs, the pooled task bootstrap interval).
   - **Canary runs** count. They are ordinary quality runs and usually the freshest.
   - **Exclusions:** failed or stopped runs, and quality runs without scored answers, are not evidence. The open-ended suite has no automatic score and is shown only as a link to its A/B studies.
3. **Freshness against the deployment fingerprint.** Every piece of evidence is labelled:
   - **current:** the run's deployment snapshot has the model's current fingerprint.
   - **stale:** the fingerprint differs, or (for runs from before snapshots existed) a deployment change was recorded after the run started. Stale evidence describes a deployment that is no longer served.
   - **unverified:** no snapshot exists for the run and no later change is known.

   The age of the evidence is shown everywhere. Evidence older than 90 days is marked as old, but age alone never makes it stale.
4. **Ties are shown, not ranked.**
   - **Leader:** in each quality column, the model with the highest score among current or unverified evidence is the leader.
   - **Ties:** every other model is compared with the leader using the existing paired test (`compare_groups`, family bootstrap and sign-flip test), Holm-adjusted across the column. Models without a detectable difference are marked "tied with the leader".
   - **Incompatible protocols** (different decoding settings or suite revision) are marked "not comparable" rather than ranked.
   - **Why:** the scorecard must not suggest that a 0.4-point lead means anything.
   - **On demand:** a paired comparison takes about half a second for single runs and several seconds for repeat groups. The page loads first and fills the marks column by column. Comparisons are cached by the run ids they compare, because completed runs never change.
5. **Decision profiles hold requirements for one use case.** Admins create profiles, for example "Coding agents" or "HR assistant", each with a closed list of gates:

   | Gate | Evidence | Passes when |
   |---|---|---|
   | `quality` | latest run or group of a suite; optionally one category; metric achievement or full pass | value ≥ threshold |
   | `capacity` | latest open-loop run of a workload | sustainable rate ≥ threshold req/s; a lower bound counts when it already reaches the threshold |
   | `latency` | latest closed-loop run, at a chosen concurrency | TTFT p95 or whole-answer p95 ≤ threshold ms, or per-request output p50 ≥ threshold tok/s |
   | `context` | measured maximum context of the latest sweep, otherwise the context limit from the deployment snapshot (labelled declared) | ≥ threshold tokens |
   | `operations` | monitoring | no unacknowledged alert for the model and, if the model has canaries, every latest canary run is ok |

   - **Gate result:** each gate gives *pass*, *fail*, *missing* (no evidence) or *stale* (only stale evidence).
   - **Uncertain flag:** a quality pass or fail whose 95% interval contains the threshold is flagged uncertain. The point estimate still decides.
   - **Profile verdict:**
     - *does not meet* if any gate fails on current or unverified evidence;
     - otherwise *incomplete* if any gate is missing or stale;
     - otherwise *meets*.
   - **Gates are validated on save:** known suites, known workload presets or hashes, thresholds in range, at most 20 gates.
6. **Decision records.** On a profile, admins record a decision for a model (*approved*, *approved with conditions* or *rejected*) with a required note.
   - **Snapshot:** the record stores the full gate evaluation at that moment: run ids, values, fingerprints and freshness. Later evidence never rewrites it.
   - **Drift warning:** an approval whose recorded fingerprint differs from the model's current fingerprint is shown with "deployment changed since this decision", linking to the monitoring timeline.
   - **Append-only:** records cannot be edited. A new decision supersedes the previous one for the same model and profile, and the history stays visible.
7. **A/B studies are context, not gates.** A pairwise preference is relative to one opponent and one study, so it cannot be a requirement on one model. The model scorecard lists the studies involving the model with their people and judge verdicts.
8. **Revision.** Gate evaluation is versioned (`decision-gates-v2`) and the revision is stored in every decision record.
9. **Access.**
   - **Everyone signed in:** sees scorecards, profiles, verdicts and decisions. Application teams should see why a model was chosen.
   - **Admins:** create, edit and delete profiles, and record decisions.
   - **Deletion:** deleting a profile deletes its decision records, after a confirmation that names the number of records.
10. **Export.** `/scorecard.json` and `/scorecard/profiles/{id}.json` return the same evidence and evaluations for external reporting.

## Data model

```
decision_profiles   id, name, description, gates_json, created_by→users set null, created_at, updated_at
decision_records    id, profile_id→decision_profiles cascade, model_id→models set null, model_name,
                    decision (approved|conditional|rejected), note, revision, evaluation_json,
                    fingerprint, created_by→users set null, created_at
```

Evidence is computed on request from runs, checks and events. It is not stored, so it can never disagree with the runs. Parsed quality summaries of completed runs are cached in memory by run id and completion time.

## UI

- **Navigation:** "Scorecard", visible to all.
- **`/scorecard`:** a matrix with one row per model and these columns:
  - one column per quality suite with evidence, showing the achievement with its interval, leader and tie marks;
  - one column per load workload, showing the sustainable rate;
  - latency at concurrency 1: TTFT p95 and per-request output speed;
  - context: measured or declared;
  - operations: canary status, open alerts and fingerprint.

  Each cell links to its run and carries a freshness mark. Profiles are listed above the matrix with a per-model verdict count.
- **`/scorecard/models/{id}`:**
  - all evidence for one model, with links;
  - the verdict of every profile for that model;
  - the A/B studies involving it;
  - its decision records.
- **`/scorecard/profiles`:** the list of profiles, and for admins a create form with a gate editor.
- **`/scorecard/profiles/{id}`:**
  - a models × gates matrix with values, thresholds, freshness and verdicts;
  - the decision history;
  - for admins: edit, delete and a "Record decision" form per model.
- **Dashboard:** a "Decision profiles" card shows each profile with the models that meet it.

## Verification

`selftest_scorecard.py`:

- **Evidence selection:** latest rather than best; repeat groups; exclusion of failed, stopped and unscored runs; canary runs included.
- **Freshness:** current, stale by fingerprint, stale by a later deployment change for runs without snapshots, unverified and old.
- **Ties:** leader and tie marks, Holm adjustment, and incompatible protocols.
- **Gates:** validation, each gate type's pass, fail, missing and stale paths, lower-bound capacity, declared versus measured context, the uncertain flag and the profile verdict rules.
- **Decisions:** snapshots that are not rewritten by later runs, the drift warning, append-only history and deletion with the profile.
- **Pages and access:** pages, JSON exports, admin-only changes and the dashboard card.

## Limits

- **Coverage:** gates encode only what the application measures. Licensing, data residency and cost are outside it and belong in the decision note.
- **Latest run as evidence:** one deliberately bad configuration run can make a model fail a gate until a new run exists. The run link and its settings are always one click away, and repeat groups make quality evidence more robust.
- **Not a statistical ranking:** ties are tested against the leader only, not as a full ranking. Two models tied with the leader may still differ from each other.
