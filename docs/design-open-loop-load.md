# Design: open-loop, trace-shaped load testing

Status: accepted (recommended decisions), 2026-10-05. Item 5 of the evaluation roadmap.

Update: open-loop load is now the capacity stage of the standard performance test ([design-consolidation.md](design-consolidation.md)). `performance: "load"` and `mode: "load"` are still accepted and map to the staged test.

## Problem

Both existing performance modes are **closed loop**. A fixed number of workers each send their next request only when the previous one finishes. When the server slows down, offered load drops with it, so queueing, tail latency under bursts and the arrival rate a deployment can sustain are never observed. `capacity.py` therefore reports *maximum capacity: not established*, and it models the chat/agent mix instead of measuring it.

Company traffic is open loop. People and applications send requests when they need to, whether or not the server is busy. Lengths vary widely, many requests share a system prompt or tool definitions, and chat and agent traffic hit the same deployment. The question to answer is:

> At which arrival rate does this deployment still meet our latency targets for this traffic mix, and does it keep meeting them over time?

## Decisions

1. **Separate mode.** A new run mode `load` ("Performance · open-loop load") with its own engine (`app/benchmarking/load_test.py`). The revision is `open-loop-v1`, and the report is `schema_version 5, kind "open_loop"`. Closed-loop modes, their reports and comparisons are unchanged.
2. **Workloads are request classes with length histograms.** A workload is 1–6 classes. Each class has:
   - a mix weight
   - an input-length histogram and an output-length histogram, given as buckets `[min, max, weight]`, sampled uniformly within the chosen bucket
   - a shared-prefix length (system prompt / tool definitions common to every request of the class)
   - its own SLO
   
   Presets: `chat`, `agent`, `rag` and `mixed` (80% chat, 20% agent, matching the capacity defaults). An advanced JSON field accepts a custom workload, for example histograms exported from gateway logs. The whole workload is snapshotted into the run options and fingerprinted.
3. **Output length is controlled.** The prompt asks the model to count upward until stopped, and `max_tokens` is the sampled output length. `finish_reason = length` therefore means "target delivered" and counts as complete; an early stop is complete with a recorded shortfall. Provider counts include reasoning tokens.
4. **Seeded, reproducible arrivals.**
   - Arrival processes are **Poisson** (exponential gaps) or **bursty** (gamma gaps with coefficient of variation `burstiness`, default 2; 1 is Poisson).
   - Schedules, classes and lengths derive from `(seed, step index, rate, duration)`. Two runs with the same settings see identical traffic, so they can be compared step by step.
5. **True open loop.**
   - A dispatcher sends each request at its scheduled time, whether or not earlier ones have finished.
   - In-flight requests are capped by the run's concurrency setting (default 256, maximum 1,024). An arrival that finds the cap full is **dropped** and counts as an SLO miss; it is never delayed, because delaying it would make the test closed loop again.
   - Dispatch lag (actual send time minus scheduled time) is recorded and **added to the user-visible TTFT and latency**, which avoids coordinated omission.
   - A step is flagged *client-limited* if any arrival was dropped or the p99 lag exceeds 100 ms. A client-limited step cannot establish capacity.
6. **Rate ladder.**
   - The run lists 1–12 offered rates (requests/s). Each step generates arrivals for `step_seconds` (10 s–4 h, default 120 s), then waits for in-flight requests to finish. A request ends within `request_timeout` (default 300 s), so every step drains.
   - Steps run in ascending order. The ladder stops after `stop_after_failed_steps` (default 2) consecutive SLO failures, or when a step produces no complete request.
   - Connections are warmed before the first step. Additional connections opened under load are flagged per request.
7. **Per-request SLOs and goodput.**
   - A request meets its class SLO when all of these hold:
     - it completed;
     - lag + TTFT ≤ `ttft_ms`;
     - its output token time ≤ `tpot_ms` (optional);
     - lag + latency ≤ `e2e_ms` (optional).
   - A required metric that cannot be measured (no streaming, burst delivery, estimated counts) is a miss and is reported as such.
   - **Attainment** = SLO-met requests ÷ all arrivals (drops and errors included).
   - **Goodput** = SLO-met requests ÷ step duration.
   - A step **passes** when attainment ≥ the target (default 95%, i.e. a p95 SLO), every class individually reaches the target, and it is not client-limited.
   - **Sustainable rate** = the highest passing offered rate. It is *established* when a higher tested rate failed, a *lower bound* when the highest tested rate passed, and *not met* when nothing passed.
8. **Soak and drift.** A soak is a ladder with one rate and a long step (preset: 60 min). Every step is summarised in time windows of max(10 s, step/30). Steps with at least six windows report drift:
   - p95 TTFT and attainment in the first vs last third
   - error bursts
   - output-rate trend
   
   Each compared third needs at least 20 arrivals. A step is flagged *degrading* in either of two cases:
   - The last-third p95 TTFT is more than 25% above the first third *and* uses at least half of the tightest first-token SLO. Rises far below the SLO are noise, not lost headroom.
   - Attainment drops by more than 5 points.
9. **Capacity integration.**
   - `estimate_capacity` gains an `arrival` block for open-loop runs: the sustainable rate and its status, goodput, mean input/output tokens at that step, and per-class attainment.
   - Daily traffic budgets apply the existing headroom, availability and busy-hour assumptions to the **measured** rate.
   - With a mixed workload the mix is measured rather than modeled.
   - Maximum capacity becomes `established` / `lower_bound` for that workload and SLO. It never covers other workloads.
   - The capacity revision becomes `serving-capacity-v3`. Closed-loop scenarios are unchanged.
10. **Comparison inside the open-loop view.** Runs are comparable when the workload fingerprint, arrival process, burstiness, step duration, seed, SLOs, attainment target, temperature and reasoning effort are identical. They align by offered rate. Incomparable runs are shown separately with the reason.
11. **Storage.** Per-step statistics are kept in full. Per-request samples are stored column-wise in the packed `perf_json`. Beyond 100,000 requests per run, a seeded uniform sample is retained and flagged; statistics are always computed from every request.
12. **vLLM telemetry.** The existing B300 collection wraps the whole run. Steps record wall-clock start/end, so server queueing and KV pressure can be read per step.

## Request construction

```
system:  "Workload {class} context {run_nonce}." + " pad" × (shared_prefix − header)   # identical for every request of the class
user:    "Request {run_nonce}-{index}. Background records follow:\n" + " pad" × body
         + "\nIgnore the background. Count upward from 1, writing every integer separated by commas. Continue until you are stopped."
```

Lengths are cl100k_base reference tokens for the whole input, shared prefix included, as in the sweep's `PromptBank`. Every input bucket must be at least `shared_prefix + 128`. The run nonce stops caches from earlier runs from helping. Within a run, the shared prefix behaves like a real deployment's system prompt.

## Limits (validated at submission)

| Setting | Range |
|---|---|
| classes | 1–6, unique names `[a-z0-9-]{1,24}` |
| buckets per histogram | 1–16, `1 ≤ min ≤ max`, input ≤ 1,048,576, output ≤ 65,536, weight > 0 |
| shared prefix | 0–524,288 |
| SLO | `ttft_ms` 1–600,000 (required); `tpot_ms` 1–10,000, `e2e_ms` 1–3,600,000 (optional) |
| rates | 1–12 ascending values in 0.01–500 req/s |
| step | 10–14,400 s; total arrival time ≤ 12 h; expected arrivals ≤ 500,000 |
| in-flight cap | 1–1,024 (run concurrency) |
| attainment target | 0.5–0.999 |
| seed | 0–2³¹−1 |

## Report shape (`schema_version 5`, `kind "open_loop"`)

```
protocol: revision, load_model "open_loop", workload, workload_hash, arrival, burstiness, seed,
          rates, step_seconds, request_timeout, in_flight_cap, attainment_target,
          stop_after_failed_steps, temperature, reasoning_effort, client, streaming, definitions…
steps[]:  offered_rate, step_seconds, started_at, ended_at, arrivals, dispatched, dropped, errors,
          completed, slo_met, attainment, goodput_rps, passed, client_limited, failure_reasons,
          ttft/latency/tpot/lag stats (user-visible = lag included), max/mean in flight,
          input/output tokens, output_tokens_per_sec, cache_metrics, classes{name: same subset},
          windows[], drift{}, error_examples[], samples{columnar}, samples_retained
summary:  sustainable_rate, sustainable_status, goodput_at_sustainable, first_failed_rate, notes
cancelled, stop_reason, notes
```

## UI

- **Run form:** mode "Performance · open-loop load" with:
  - a workload preset (chat / agent / RAG / mixed / custom JSON)
  - rates, step seconds, a ladder/soak toggle, arrival process, burstiness, seed and attainment target
  - the in-flight cap (concurrency)
  - expected arrivals and duration, shown live
- **Run page** (Performance tab): a sustainable-rate summary with metric cards, then:
  - an attainment and goodput vs offered rate chart
  - a step table, and per-class tables with SLO and failure-reason breakdowns
  - drift windows for long steps
  - a comparison table when several comparable runs are shown
- **Offline HTML:** the same tables, without scripts.
- **Capacity page:** an arrival-rate capacity block.

## Verification

`selftest_load_test.py` uses fake clients and short steps (module minimums patched):

- workload validation and limits
- reproducible schedules (same seed gives the same arrivals; Poisson/gamma mean rate and CV within tolerance)
- prompt lengths exact, with a shared prefix
- open-loop dispatch: a slow fake server does not slow arrivals
- cap drops counted, never delayed
- lag included in the SLO
- attainment, goodput and pass rules; sustainable-rate status (established / lower bound / not met)
- stop after consecutive failures, and cancellation
- windows and drift detection
- sample retention cap
- submission round trip, rerun/clone and 422s
- runner end to end with a stored report; run page, offline HTML and capacity arrival block; comparison compatibility

## Out of scope (v1)

- Replaying timestamped production traces.
- Multi-turn sessions with growing history. The shared prefix approximates their caching.
- Tool-call payloads under load. Item 1 covers conformance; load uses text completions.
- Server-side admission control or priority classes.
- Automatic rate search (bisection). The ladder is explicit, and a follow-up run can refine it.
