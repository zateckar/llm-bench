# Design: deployment fingerprints and regression canaries

Status: accepted (recommended decisions), 2026-10-05. Item 6 of the evaluation roadmap.

## Problem

A benchmark result describes the deployment as it was when the run happened. Deployments in our datacenter change underneath the evaluations:

- an engine upgrade, a new chat template or tokenizer, a different quantization;
- a raised context limit, a gateway that routes the same model name to another backend;
- a model silently replaced by its newer checkpoint.

Some changes are deliberate but go unannounced; others are accidents. Users notice them as "the assistant got worse this week" long before anyone checks. Two capabilities are missing:

1. **Know what was measured.** Every run should record a fingerprint of the deployment it measured, and the application should say when the fingerprint changes.
2. **Notice regressions without anyone asking.** A small, scheduled canary evaluation per production model, compared against an accepted baseline, should raise an alert when quality, availability or latency regresses.

## Decisions

1. **Snapshot at the start of every run** (`deployment-probe-v1`).
   - **When:** before any measured request, including performance, sweep and load runs, and on demand from the monitoring page.
   - **Cost:** at most eight small requests, all bounded by short timeouts.
   - **Failure:** a failed capture never fails the run. It is stored as unsuccessful and is not compared.
2. **What a snapshot contains.**
   - **Configuration:** the endpoint (scheme, host, port, path; no credentials or query) and the configured model id.
   - **Served model:** the matching entry of `GET {base}/models`: `id`, `root`, `max_model_len`, `owned_by`, `parent`. If the model is not listed, that is recorded.
   - **Server:** the `version` from `GET {origin}/version` when it answers with JSON (vLLM does), and the `Server` response header.
   - **Template and tokenizer probes:** five fixed chat requests with `max_tokens: 1` at temperature 0:
     - a plain user message
     - with a system prompt
     - multilingual text (Czech, German, symbols, CJK, emoji)
     - a multi-turn conversation
     - a request offering one tool

     Each probe records `usage.prompt_tokens` (or the HTTP status when rejected), the response `model` and `system_fingerprint`. Prompt-token counts are deterministic for a given template and tokenizer, so they change exactly when those change.
   - **Behaviour probe (soft):** one short deterministic question (48 output tokens) whose answer is stored for display. It is not part of the fingerprint, because batching makes outputs at temperature 0 nondeterministic on most engines.
3. **Fingerprint and change detection.**
   - **Fingerprint:** the first 16 hex characters of the SHA-256 of the canonical JSON of the hard fields (everything except the behaviour probe and timestamps).
   - **Comparison:** each successful snapshot is compared with the previous successful snapshot of the same configured model.
   - **Change report:** a change lists every changed field (dotted path, before, after).
   - **Configuration changes:** if only the configuration changed (someone edited the model's endpoint or model id), it is reported as such, not as a deployment change.
4. **Canaries** (`canary-v1`) are scheduled quality runs.
   - **Settings:** an admin chooses a model, any quality suite (a use-case suite or `tool-conformance` is recommended; `rigorous` is allowed but long), the concurrency and an interval of 1–168 hours.
   - **Queueing:** canary runs go through the normal queue, so they never overlap other runs. A canary waits behind long runs.
   - **No backlog:** a canary is skipped while its previous run is still pending or running, and the next time is set from the current time, not from missed slots.
   - **Run now:** admins can trigger a canary immediately.
5. **Baseline.** The first completed canary run becomes the baseline automatically. Admins can promote any completed run of the canary (for example after accepting an upgrade). The baseline stays pinned until changed; if it is deleted, the next completed run becomes the baseline again.
6. **Evaluation after every canary run.** Each run gets a status: *ok*, *warning* or *alert*.

   | Event | Severity | Condition |
   |---|---|---|
   | `run_failed` | alert | The run failed or nothing could be scored. |
   | `quality_regression` | alert | The paired comparison with the baseline (the existing tested `compare_groups`: family bootstrap and sign-flip test, plus McNemar on full passes) finds the baseline significantly higher. |
   | `quality_improvement` | info | The run is significantly higher than the baseline. |
   | `baseline_incompatible` | warning | The comparison is impossible: a new suite version, decoding settings or protocol. A new baseline is needed. |
   | `latency_regression` | warning | Median TTFT is above 1.5× the baseline's and at least 250 ms slower, or median request latency is above 1.5× and at least 1 s slower. Both runs use the canary's fixed concurrency. |
   | `availability` | warning | Some requests failed at the endpoint. |

   `deployment_changed` (alert) and `configuration_changed` (info) come from snapshots, for every run and check, not only canaries.
7. **Events** are stored per model.
   - **Visibility:** every signed-in user sees them on the Monitoring page. The dashboard shows a banner while alerts or warnings are unacknowledged.
   - **Acknowledgement:** admins acknowledge events; the acknowledgement records who and when.
8. **Notifications.**
   - **Webhook:** a canary may have a webhook URL (Teams, Slack or Mattermost incoming-webhook compatible: `{"text": "..."}`). Alert and warning events from that canary, and deployment changes of its model, are posted once.
   - **Validation:** the URL is checked with the same SSRF guard as model endpoints.
   - **Failures:** a delivery failure is recorded on the event and never retried, so a broken hook cannot block evaluation.
9. **Retention.** Each canary keeps its newest 100 runs, its baseline and every run that raised an alert or warning. Older canary runs are deleted after each evaluation. Ordinary runs are never deleted automatically.
10. **Access.**
    - **Admins:** create, edit, pause, delete and trigger canaries, set baselines, run deployment checks and acknowledge events.
    - **Everyone signed in:** sees monitoring pages, run snapshots and events.

## Data model

```
deployment_checks  id, model_id→models cascade, run_id→test_runs set null, revision, created_at, ok, error,
                   fingerprint, snapshot_json, previous_id, changed
canaries           id, name, model_id→models cascade, suite, max_concurrency, interval_hours, enabled,
                   baseline_run_id→test_runs set null, next_run_at, webhook_url, created_by→users set null, created_at
test_runs          + canary_id, + canary_status (ok|warning|alert), + canary_json (evaluation detail)
monitor_events     id, model_id→models cascade, canary_id→canaries set null, run_id→test_runs set null,
                   check_id→deployment_checks set null, kind, severity, title, detail_json, notified,
                   created_at, acknowledged_by→users set null, acknowledged_at
```

## UI

- **Navigation:** "Monitoring", visible to all.
- **`/monitoring`:**
  - unacknowledged and recent events (admins can acknowledge);
  - a canary table with model, suite, interval, last status, baseline, next run and a short achievement trend;
  - for admins, a create form;
  - every model's latest fingerprint with a "Check now" button for admins.
- **`/monitoring/canaries/{id}`:**
  - run history with achievement, the difference from the baseline and its verdict, TTFT and latency medians, fingerprint and status;
  - for admins: run now, pause/resume, edit, set as baseline, delete.
- **`/monitoring/models/{id}`:** the snapshot timeline with field-level diffs and the latest full snapshot.
- **Run detail:** a "Deployment" section with the snapshot, its fingerprint and whether it changed; canary runs link to their canary.
- **Run list:** canary runs carry a canary badge.

## Verification

`selftest_monitoring.py`:

- **Probes:** probe construction against a fake HTTP session (served-model matching, version, rejected tool probe, unreachable endpoint), canonical fingerprints, diffs and configuration-only classification.
- **Runner:** snapshot capture writes checks and change events, and a failed capture never fails a run.
- **Scheduling:** due canaries create pending runs with the canary's settings, there is no backlog, paused canaries are skipped and the next time is computed correctly.
- **Evaluation:** with a fake runner, automatic baselines, regressions, improvements, incompatibility, latency, availability and failed runs.
- **Notifications and retention:** webhook payloads and recorded failures, and retention that keeps baselines and alert runs.
- **Pages:** the pages and access rules, acknowledgement, the dashboard banner and the run-detail section.

## Limits

- **Probe coverage:** a fingerprint detects changes visible through the API, its tokenizer and its template. It cannot detect weights replaced behind an identical tokenizer and template. Canary quality runs exist for that.
- **Mixed backends:** a load balancer over replicas with different versions makes fingerprints alternate. That is reported as repeated changes and is itself worth knowing.
- **Canary power:** a canary detects regressions large enough to be significant on its suite. Small suites only catch large regressions, so the page shows the interval width.
