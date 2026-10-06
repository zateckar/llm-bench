# Users at SLO: session-based capacity

Status: accepted, 2026-10-06. Stage 3 of the standard performance test (`standard-performance-v4`, which dropped the open-loop capacity stage; v3 ran it as stage 4); engine revision `sessions-v1`, report `schema_version` 7, `kind` "sessions".

## Problem

Capacity questions arrive as user counts: *with a 40/60 chat/agent mix and sessions up to 256k tokens, how many people can one deployment serve, each getting the first token within 2 s and at least 40 tok/s?* The other stages cannot answer that:

- The latency and context stages run fixed requests at fixed concurrency. Concurrency is not users: a person spends most of a session reading and typing, an agent spends part of it running tools.
- The capacity stage sends independent single requests at an arrival rate. It has no history, so it misses the dominant cost of real sessions: every turn resends a growing context, and the KV cache holds many long histories at once.

## Decision

1. **Simulated users run sessions** ([session_workload.py](../app/benchmarking/session_workload.py), [session_load.py](../app/benchmarking/session_load.py)).
   - A user model is 1–4 weighted classes. Each class has a shared system prompt (tool definitions for agents), histograms (`[min, max, weight]` buckets) for the first message, later messages, answer length, pause between turns and turns per session, and its own SLO: first token and output speed.
   - A turn sends system prompt + history + new message, waits for the whole answer, appends both to the history, pauses, and continues. A session ends after its planned turns or when the next turn would exceed the **context cap**; the user then starts a new session.
   - Users are assigned to classes by largest remainder and interleaved, so every prefix of the user list keeps the mix.
   - Plans are a pure function of (seed, cap, user count, user index): every model receives the same sessions.
   - Prompts have exact cl100k_base reference lengths. The answer length is enforced by `max_tokens` on a "count upward until stopped" instruction, as in the capacity stage. The real answer becomes history.
   - Defaults are estimates (README table) until they can be fitted to session traces from gateway logs.

2. **Steady state, not a cold start.**
   - Each user starts in a session already in progress, drawn length-biased (longer sessions are proportionally more likely to be in progress at a random moment) with a uniform current turn. Its earlier turns are synthetic history.
   - Users start at random times within the first half of the warmup. Each user's first request (a cold prefill of its whole history) is excluded.
   - The measured window starts when the warmup has passed **and** every user has had a first answer (bounded by the request timeout). Requests *sent* in the window are judged; those still running at its end are awaited.

3. **Pass rule.**
   - A request meets its class SLO when it completes, dispatch lag + first token ≤ `ttft_ms`, and output token time ≤ 1000 / `output_tokens_per_sec`. Unmeasurable timings (no streaming, burst delivery) are misses.
   - A level passes when every class with users reaches the attainment target (default 0.95).
   - Dispatch lag p99 over 100 ms means the client could not keep up. Such a level is *client-limited*: it neither passes nor bounds capacity.
   - **Early stop:** once a third of the window (at least 30 s) has passed, a class with at least 20 judged requests and attainment below 50% stops the level as a failure.

4. **Search per context cap.**
   - Start at `start_users` (8). Double while levels pass, up to `max_users` (256). Then bisect between the highest passing and the lowest failing count until the gap is within `resolution` (10%) of the passing count.
   - Caps run in increasing order; default caps are 32k, 128k and the usable context limit. A larger cap lets sessions grow longer and cannot serve more users, so it starts at the previous cap's result and inherits its first failing count. Counts known to fail are never measured; if no count passed at a smaller cap, larger caps are *not met* without measuring.
   - Results: *established* (a higher count failed), *lower bound* (the maximum passed, or only client-limited failures above), *not met* (one user failed), *inconclusive* (only client-limited levels).
   - The stage stops when a level completes no request at all (endpoint down).

5. **Limiting factor.** Each failed level names the most common miss reason over the classes below target: first token, output speed, errors, incomplete answers, unmeasurable timings or the client. This stays the client-observed miss; the cause is the constraint (8).

6. **Report and display.** Levels are saved as they finish. The report stores the protocol (user model and its hash, caps, windows, target, seed, client settings), every level with per-class attainment, miss reasons, first-token and output-speed statistics, context and cache metrics, dispatch lag, at most 1,000 sampled requests, and server signals. The run page shows users at SLO by cap, an attainment chart, every level and the user model. Runs compare by cap when their `comparison_key` (user model hash, windows, target, seed) matches.

7. **Scorecard and gates.** The model scorecard lists users at SLO by cap per user model (key: preset name or `custom:<hash>`). A `users` gate requires at least *N* users on a user model with sessions up to *C* tokens; it is judged at the smallest measured cap ≥ *C*, which is conservative because larger caps serve no more users. A lower bound below the threshold fails with an uncertainty mark. The scorecard also shows where each cap was exhausted and the constraint.

### Saturation and constraints (added 2026-10-06)

Users at SLO says how many users one deployment serves well. It does not say how far beyond that the deployment still works, or *why* it stops: the answer that decides what to change (KV cache quantization or offloading, prefill/decode disaggregation, batching settings, more replicas).

8. **Saturation search** (setting `saturation`, default on; `saturation_max_users`, default 1,024).
   - After a cap's SLO search, user counts double from the largest measured count. Each level is judged against **hard limits**, the context stage's rules applied to sessions: more than 10% of requests failed or incomplete; a class's first token p95 over 10× its target or median output speed under a tenth of it; not every user had a first answer within the request timeout; or a **throughput plateau** — output grew less than 10% over a level with at least 1.5× fewer users while the median first token grew 1.5×.
   - Levels of the SLO search count: a cap whose first failing count is already beyond a hard limit needs no extra level. The search stops at the first level beyond a hard limit, at a client-limited level, or at `saturation_max_users`.
   - Saturation levels stop early on the failure and first-token rules (≥ 20 judged requests) instead of the SLO rule, and a saturation level with no completed request is the limit itself, not a down endpoint.
   - Larger caps inherit the exhausting count: a count at or above it is not measured, and the cap reports *≤ N* if its own search would reach it.
   - Result per cap (`entry.saturation`, also in `summary.results[].saturation`): status `reached` (users, the last count below, reason), `not_reached` (no limit up to the largest measured count), `inherited`, `client_limited` or `not_measured`, plus the peak output throughput and its user count.
   - Closed-loop users self-regulate, so a server past saturation shows long waits rather than an unbounded queue. The hard-limit rules detect that state from latency, failures and the throughput curve.

9. **Constraint diagnosis** ([constraints.py](../app/benchmarking/constraints.py), `constraints-v1`). Every failing level and every level beyond a hard limit gets `level.constraint`: one constraint with a label, summary, basis, evidence lines, remedies and other constraints also present. Cap results carry the constraint at the first failing count and at exhaustion.
   - **Inputs from the client**: the failed SLO part; the level's prompt and output token rates; the **expected prefix reuse** (per request, the system prompt and history before the new message, divided by the whole context; what a cache that kept every history would serve); the client-reported cached-token fraction; and the **working set** (the sum of all users' current histories, with shared system prompts counted once, sampled through the window).
   - **Inputs from telemetry** (`window_signals`), over the level's wall-clock measured window (`window_started_at`/`window_ended_at`) with rate and histogram series skipping their first 60 s in windows over 120 s: mean/max of running and waiting requests, KV occupancy, preemptions, prompt and generation rates, prefix-cache hit fraction, server first-token/queue/prefill/time-per-token p95, DCGM SM, tensor and DRAM activity and memory use, the share of the window with running requests at their ceiling while requests wait, and the KV capacity in tokens from `cache_config_info` (Σ `num_gpu_blocks` × `block_size`). Without clock alignment the diagnosis falls back to the client.
   - **Rules**, in order; the first match names the constraint, later ones are listed as "also":
     1. Client: dispatch lag p99 > 100 ms.
     2. **KV capacity**: KV occupancy max ≥ 90% or any preemption. (vLLM counts blocks held by running requests; cached idle prefixes are evictable free blocks, so a full cache means running sessions alone fill it.)
     3. **Eviction / cache miss**: expected reuse ≥ 30% and observed hits below 60% of it. With occupancy ≥ 80% or a working set larger than the KV capacity, idle histories are being **evicted** between turns; otherwise prompts are **not reused** (prefix caching off, history re-rendered differently, sessions spread across replicas).
     4. **Batch slots**: waiting mean ≥ 0.5 and running at its ceiling for ≥ 50% of the window, occupancy < 80% — `max-num-seqs`.
     5. Output-speed misses: **prefill interference** when the load is prompt-heavy (≥ 2 uncached prompt tokens per generated token), else **decode**.
     6. Queue with KV room: **scheduler** when SM activity < 40%, **prefill** when prompt-heavy, **compute** when SM activity ≥ 70%, else **queueing** (GPU activity unknown).
     7. First-token misses: **prefill**. Without telemetry: **prefill or queueing**.
     8. **Errors**: more than 10% failed or incomplete. Last, because errors under load are usually timeouts of one of the above.
   - **Remedies** per constraint are fixed lists, filtered by the cache config: FP8 KV cache is not suggested when `cache_dtype` is FP8, offloading not when CPU blocks or an offloading size are configured, enabling prefix caching only when it is off.
   - The rules are heuristics over deployment-wide signals. The evidence lines carry the numbers so a reader can disagree with the label.

10. **Telemetry v3** ([vllm_telemetry.py](../app/benchmarking/vllm_telemetry.py)). Definitions gain `kind` (gauge, rate, histogram), `combine` (how instances add up) and `optional`. New optional series: server time per output token (`inter_token_latency`, falling back to `time_per_output_token`), and DCGM GPU metrics selected by `Hostname` plus the model's `b300_gpus` indices (`gpu=~"0|1|…"`; blank selects every GPU on the host, with a warning). `cache_config_info` is read once after the stage by instant query, matched by the instances found at setup because the info metric carries no `model_name`, and only known labels are kept. Optional series never make a collection partial; their absence is one warning. The collection deadline is 60 s.

## Consequences and limits

- **Runtime.** A level takes warmup + window (4 minutes by default, less when it stops early). The SLO search over three caps takes roughly 45–90 minutes; the saturation search usually adds 1–3 levels per cap, making the whole performance test 1.5–3 hours. Fewer caps, shorter windows or a lower `saturation_max_users` shorten it.
- **Saturation load.** Doubling past the SLO limit deliberately overloads the deployment for a few minutes per cap. Do not run it against a deployment serving production traffic unless that is acceptable; turn the saturation search off instead.
- **GPU metric names.** DCGM metric and label names follow the default dcgm-exporter configuration; an exporter with other labels yields no GPU series (the diagnosis then reports "queueing" instead of compute or scheduler).

- **Closed loop.** Users wait for answers, so they slow down with the server, as people do. Overload from independent arrivals queueing faster than they are served is not simulated; the former capacity stage measured that and was dropped in v4 because user counts answer the question asked.
- **Client.** One thread and connection per user. Hundreds of users with short pauses can saturate one client machine; dispatch lag detects this and the result becomes a lower bound.
- **Model of use.** Results hold for this user model, these SLOs and this deployment state. Synthetic history is padding text, so prefix-cache behaviour within a session is realistic but content-dependent effects (speculative decoding acceptance, reasoning length) are not.
- **Token counts** are cl100k_base reference tokens with a framing allowance; provider counts are stored per request.
