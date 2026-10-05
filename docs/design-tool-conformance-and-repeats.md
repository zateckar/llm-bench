# Design: native tool-calling conformance and repeated runs

Status: draft for review · 2026-10-05

Update: the tool-conformance suite is now the tool-calling area of the standard suite and is no longer offered on its own ([design-consolidation.md](design-consolidation.md)).

This document designs two additions:

- **A. Tool-conformance suite.** Tests native OpenAI `tools` / `tool_choice` / `tool_calls` and `response_format` structured output against each deployment.
- **B. Repeated runs with paired statistics.** Measures run-to-run variability and tests whether a difference between two models is real.

Both fit the existing principles. Grading is deterministic. Suites and protocols are fingerprinted. Unscored outcomes stay explicit. Nothing is silently dropped or substituted.

---

## A. Tool-conformance suite

### A.1 Why a separate suite

On self-hosted vLLM, the same weights can fail as a deployment. Common causes:

- a missing `--enable-auto-tool-choice`;
- the wrong `--tool-call-parser`;
- a chat template that doesn't render tools;
- a reasoning parser that swallows calls;
- a streaming parser that splits arguments incorrectly.

Applications and agents see these failures. The rigorous suite doesn't, because its interactive tasks deliberately use a text JSON-action protocol (`json-actions-v1`). Its module docstring says it "does not claim to evaluate a provider's native function-call wire format".

The new tasks go into a **separately versioned suite**, `tool-conformance-v1`, not into Rigorous v15:

- Rigorous v15 fingerprints, the 323-task inventory, calibration cohorts and historical comparability stay untouched.
- Conformance is a property of the deployment. It should be cheap to rerun after every vLLM or config change, without the full 323-task run.
- It establishes a general `suite` run option, which item 2 of the roadmap (use-case suites) needs anyway.

### A.2 Client changes (`llm_client.py`)

The current client returns `(text, usage, metrics)` and silently ignores `tool_calls`. A streamed tool-only delta hits `continue`, so it gets no TTFT and its arguments are lost. A blocking `message.tool_calls` is never read.

**New API, additive.** Existing callers are unchanged.

```python
@dataclass
class ToolCall:
    index: int
    id: str | None
    name: str | None
    arguments: str            # raw string exactly as delivered (may be invalid JSON)

@dataclass
class ChatMessage:
    content: str              # final text ("" when none)
    reasoning: str            # reasoning / reasoning_content, kept separate
    tool_calls: list[ToolCall]
    finish_reason: str | None
    wire_defects: list[dict]  # [{code, detail}] observed while assembling, never raised

def complete_chat(self, messages, *, tools=None, tool_choice=None,
                  parallel_tool_calls=None, response_format=None,
                  max_tokens=None, stream=None) -> tuple[ChatMessage, TokenUsage, RequestMetrics]
```

Internally, `_stream_once` and `_blocking_once` build a `ChatMessage`. `complete_messages` keeps its current contract by deriving the existing `text` from it (content, else the `<think>` wrapper), so the text path keeps the same behaviour.

**Streaming assembly.** Key tool-call deltas by `index`:

- `id`, `type` and `function.name` normally arrive on the first delta for that index.
- `function.arguments` fragments are concatenated in order.
- Interleaved indexes (parallel calls) are supported.

**Being lenient is the point.** Malformed tool calls are model or parser output, not transport failures. Record them as `wire_defects` and let evaluators score them. Only SSE framing errors stay `_ProtocolError`. Defects to record:

| code | condition |
|---|---|
| `tool_index_invalid` | `index` missing or not an int |
| `tool_id_missing` / `tool_id_duplicate` | missing or repeated call id |
| `tool_name_missing` / `tool_name_changed` | name absent, or changed for the same index |
| `tool_delta_after_finish` | tool delta after a finish_reason |
| `tool_type_invalid` | `type` present and not `"function"` |

Tool deltas count as first delivery for TTFT, `stream_chunks` and the chunk gaps. The trailing-choice-after-finish check must also reject `tool_calls` deltas; today it checks only content and reasoning. Token estimation includes the argument text and tolerates `content: None` in history. Today `"".join(m.get("content", ""))` raises on `None`.

**Request extras.** `_payload(messages, max_tokens, stream, extra=None)` merges `tools`, `tool_choice`, `parallel_tool_calls` and `response_format` when they are given. These fields are **never negotiated away**. A rejection is a result, not a compatibility problem.

**Rejected features.** `RequestMetrics` gains `http_status`. A 400 or 422 whose body names `tools`, `tool_choice`, `parallel_tool_calls`, `response_format`, `guided_*`, `tool-call-parser` or `auto-tool-choice` is classified by the executor as the new outcome **`feature_rejected`**. This is exactly what vLLM returns for `"auto" tool choice requires --enable-auto-tool-choice …`, so the report tells operators which launch flag to fix.

**Protocol revision.** The text path keeps behaving the same, which the existing `selftest_client` must prove unchanged. So `CLIENT_PROTOCOL_VERSION` stays `chat-client-v6` and Rigorous runs stay comparable with history. Conformance reports add `protocol.client.native_tools = "tool-client-v1"`.

### A.3 Question model and suite registry

- `Question.request: dict | None = None` holds the extra request fields shown to the model (`tools`, `tool_choice`, `parallel_tool_calls`, `response_format`).
- `fingerprint()` must drop `request` when it is `None`, exactly as it already drops an empty `rubric`. Every existing fingerprint and the Rigorous suite hash then stay unchanged, and a selftest asserts this.
- `q.interaction` keeps the hidden local definition: scripted tool responders or simulation parameters.
- `metadata.protocol = "native-tools-v1"` routes execution and reporting.

New `app/benchmarking/suites.py`:

```python
SUITES = {
    "rigorous":         SuiteDef(load=quality_suite.load_questions, provenance=quality_suite.provenance),
    "tool-conformance": SuiteDef(load=tool_suite.load_questions,    provenance=tool_suite.provenance),
}
```

Run option `suite` (default `"rigorous"`):

- It is validated in `make_run_options` and passed through `run_options_json` to `start_benchmark`.
- `benchmark_runner` loads `SUITES[suite]`.
- The report's `suite` block records `name`.
- The dashboard's "Last run scores" filters to the rigorous suite.
- Compare reports "different suites" explicitly instead of "no shared tasks".

### A.4 Execution (`app/benchmarking/tool_protocol.py`)

`execute_question` routes `metadata.protocol == "native-tools-v1"` to `run_native(q, client, cancelled)`. One loop serves single-turn and multi-turn tasks:

```
messages = [system?, user]
for turn in 1..max_turns:
    msg, usage, metrics = client.complete_chat(messages, **q.request, max_tokens=per_turn)
    endpoint failure        -> endpoint_error (unscored) | feature_rejected (scored fail)
    length/max_tokens       -> truncation
    no content, no calls    -> missing_answer
    append assistant {content or None, tool_calls as delivered (raw arguments)}
    if no tool_calls: final answer -> stop
    for each call: reply = responder(name, parsed_args)  (or Environment.call)
                   append {"role":"tool","tool_call_id": id, "content": json.dumps(reply)}
```

- **Budgets.** 32,768 output tokens per turn and 131,072 per task, including reasoning. There is no finalize recovery: a follow-up prompt makes no sense after a truncated tool call. Truncation stays a scored failure, as in the rest of the app.
- **Missing ids.** If the model omits an id, the executor substitutes a synthetic one so the conversation can continue, and records `tool_id_missing`.
- **History.** Earlier reasoning is not sent back, following the Chat Completions convention. Assistant messages carry `content: null` when there is no text.
- **Storage.** `Result.response` stores the canonical transcript JSON: messages, calls, replies, defects and finish reasons. The Answers view and HTML reports render it like `interactive_turns`.

### A.5 Grading: new evaluator `native_tool_use`

Grading reuses the criterion machinery (`_evaluation_for`, rubric weights, mandatory, critical, depends_on). Two groups of criteria:

**Wire conformance** (dimension `contract`; mandatory unless noted):
- `feature-accepted`: the request was not rejected.
- `arguments-json`: arguments parse as a JSON **object**.
- `tool-known`: the name is among the offered tools.
- `arguments-schema`: the arguments validate against the declared parameter schema (required keys, types, enums, `additionalProperties: false`).
- `no-markup-leak`: content contains no raw call markup. Checked markers: `<tool_call>`, `[TOOL_CALLS]`, `<|python_tag|>`, `<function=`, `<|tool_calls_section_begin|>`, `<｜tool▁call▁begin｜>`, Harmony `to=functions.`, and a bare `{"name":…,"arguments":…}` object.
- `call-ids`: ids are present and unique.
- `finish-reason` (non-mandatory, informational): `tool_calls` when calls are present. `stop` is accepted for a forced named `tool_choice`, which is what OpenAI returns.

**Behaviour** (dimension `content`):
- `selected-tool`
- `argument:<path>` for each expected argument leaf, reusing `_json_mismatches` and its aliases and integer paths.
- `call-set`: parallel calls, unordered within a turn.
- `no-call`: abstention.
- `tool-choice-honored`
- `result-used`: the final answer contains values that only the tool results provided.
- `error-recovery`
- For simulations: the existing `final-state`, `authorization` (critical) and `call-efficiency` criteria.

Because `criterion_achievement` already excludes the `contract` dimension, the report headline gives **tool-use correctness**. The existing `contract_compliance_mean` gives **wire conformance**. Full pass requires both. No new aggregation code is needed.

**Schema validation.** There is no `jsonschema` dependency today. Add a small internal validator for a declared subset: `type` (including lists for nullable), `properties`, `required`, `additionalProperties`, `enum`, `items`, `minimum`/`maximum`, `minItems`/`maxItems`. `validate_suite.py` rejects suite schemas that use anything outside that subset, so the oracle never sees a keyword it can't check.

### A.6 Task families (≈100 tasks, generated from seeds (19, 23) × 2 variants)

| Category | Family | What it catches |
|---|---|---|
| Tool Selection & Arguments | 3–6 offered tools, one correct call. Typed args: enums, int vs string, nested objects, arrays, optional args omitted, ISO dates parsed from Czech/German phrasing ("3. října 2026"). Run in **both streaming and blocking** variants | Parser args corruption, streaming-only bugs, type coercion |
| Tool Abstention & Control | Answerable without tools; no tool fits; `tool_choice:"none"`; `"required"`; named function | Over-calling, ignored `tool_choice`, rejected `required` |
| Parallel Tool Calls | 2–4 independent calls in one turn; `parallel_tool_calls:false` must serialize them | Parsers that drop the second call, interleaved indexes |
| Multi-turn Tool Use | Dependent chain (call 2 needs a value from result 1); a tool error reply that requires corrected args; final answer must cite result-only values | `tool_call_id` handling, template rendering of `role:"tool"`, hallucinated results |
| Native Agentic Simulations | The 5 existing `Environment` families (document, payment, preview, resource races, permission replanning) re-hosted behind native tools, with the same seeds, params, verdicts and criteria | Agentic ability lost through the tool path vs. the text protocol |
| Structured Output | `response_format` `json_schema` (strict) and `json_object`; nested arrays, enums, nullable fields; value correctness through `json_match` | Missing guided decoding, schema ignored, values corrupted by constrained decoding |

**Re-hosting the simulations** (`interactive_tasks.py`):

- The `schemas` arg tables map one-to-one onto function tools with `additionalProperties:false`.
- The prompt drops the `PROTOCOL` text and the prose tool docs.
- "Done" is a final assistant message without calls.
- Each parallel call increments `calls` (and so `unnecessary_calls`) exactly as sequential calls do.
- Report and HTML code keyed on `json-actions-v1` (`quality_report.py:164`, `html_reports.py:33`) accepts both protocols.

Because seeds and variants match Rigorous, the conformance report can show a **text-protocol vs native success table per family** for the same model. A large drop points at the serving stack, not the model. This table is a follow-up and is not required for v1.

### A.7 Reporting

- The quality summary already shows correctness, contract and full pass. Add a **Wire diagnostics** panel:
  - defect counts by code;
  - streaming vs blocking split;
  - the verbatim `feature_rejected` error bodies (URLs sanitised as today).
- Outcome labels gain `feature_rejected`.
- The Answers view renders native transcripts, with tool calls shown as name + raw args + parsed args, and defects highlighted.

---

## B. Repeated runs and paired statistics

### B.1 What the app reports today, and what's missing

The bootstrap resamples task families and explicitly "does not measure run-to-run variability". `paired_comparison` gives a difference and a CI, but no p-value. It only pairs runs whose full `protocol` dicts are identical, including `model_seed`. Calibration needs ≥3 runs per model, and today nothing produces them in a structured way.

### B.2 Seeds: repeats vary the seed

Every run currently sends `seed=0`. With a fixed seed, repeats at temperature > 0 would be almost identical. That understates what users see, since real clients don't send seeds. Only batching nondeterminism would remain.

- **Decision (recommended).** Repeat *i* uses `model_seed = i`. Repeat 0 then equals a normal run, so singles remain comparable with repeat 0.
- **Comparability.** Group comparisons compare `protocol` with `model_seed` removed and record the seed list.
- **Proposed policy change.** `paired_comparison` should also ignore `model_seed` for compatibility, and report `same_seed: false` as a note rather than refusing. This also lets calibration cohorts pool repeats; the cohort key must drop `model_seed` too.

### B.3 Data model and submission

**Columns.** Add to `test_runs` in `schema.sql` and to `MIGRATIONS`:
- `repeat_group_id INTEGER`: id of the group's first run, NULL for singles.
- `repeat_index INTEGER`
- `repeat_count INTEGER`

**Spec field `repeats`.** An integer from 1 to 10, default 1:
- It is accepted by `validated_specs` but kept *out of* `run_options_json`, because those keys become `start_benchmark` kwargs.
- `_insert_runs` expands one spec into N pending rows in the same plan, in order, all counting toward the existing 50-run plan cap.
- The first row's id becomes the `repeat_group_id`.

**Seed plumbing.**
- `decoding_settings(temperature, reasoning_effort, seed=0)` adds `seed` to the frozen `decoding_config_json`.
- `run_queue` already merges those settings into `model`.
- `_build_client_config` reads `model.get("seed", 0)` instead of the hard-coded 0. Old rows default to 0.

**Edit, clone and rerun.**
- `spec_from_run` and the plan-edit prefill collapse a group back into one spec with `repeats=N`.
- "Run again" on a member reruns that member with its seed.
- A group page offers "Run group again".

**Form.** Add a "Repeats" number field per run card. Show the expanded run count and an estimated duration from the last comparable run.

Each repeat is an ordinary run: queue, storage, cancellation, run detail and downloads all keep working without change. Repeats apply to the whole spec, so a `both` run repeats performance too. The form text should recommend `quality` mode for repeats.

### B.4 Statistics (`app/benchmarking/repeat_stats.py`)

**Group membership.**
- Only *completed* members with a schema-3 report are used.
- Their suite hashes must be equal, and their protocols must be equal once `model_seed` is removed. A mid-plan deployment that changes the suite excludes the affected member, with a stated reason.
- The report shows "4 of 5 repeats usable", with a reason for each excluded member.

**Within one group (one model configuration):**
- **Run-to-run spread.** Each run's headline `category_balanced` score. Show mean, SD and min–max. Show a 95% t-interval of the mean when N ≥ 3, otherwise state that more runs are needed.
- **Per task:** `n` scored repeats and `c` full passes.
  - **pass^k = C(c,k)/C(n,k).** Reliability: every attempt succeeds.
  - **pass@k = 1 − C(n−c,k)/C(n,k).**
  - Both are aggregated category-balanced over families, for k = 1..n.
  - pass^n is the headline reliability number for agentic and application use.
- **Consistency.** The share of tasks whose outcome is identical across repeats.
- **Flaky tasks.** A table of tasks with mixed outcomes, showing the per-repeat outcome and score. This is directly actionable.
- **Pooled score.** The per-task mean achievement over scored repeats, aggregated with the existing `clusters` and `balanced` functions. The existing family bootstrap applies to it.

**Between two groups A and B (or two single runs, when N = 1):**
- **Pairing.**
  - Pair tasks by fingerprint.
  - `d_t` = B's mean achievement on the task minus A's.
  - A task needs ≥1 scored repeat on each side; exclusions are counted.
  - Family difference `d_f` = mean of `d_t` within the family.
  - Statistic `D = balanced(d_f)`, which uses the same weighting as the headline.
- **Interval.** A hierarchical bootstrap:
  - resample families within categories;
  - for each drawn task, resample repeats within each group.
  - The interval therefore covers both task sampling and run noise. With N = 1 it reduces to today's interval.
- **Significance.** A paired sign-flip randomization test over families:
  - flip each `d_f` with probability ½ and recompute D;
  - `p = (1 + #{|D*| ≥ |D|}) / (1 + 10 000)`, using a fixed RNG seed so reports are reproducible.
  - This works for single runs too, so every existing comparison gains a p-value.
- **Secondary binary view.** For full pass with N = 1, show the 2×2 discordant counts (already computed in calibration) and an exact McNemar p.
- **Multiple comparisons.** The compare view tests the baseline against each other selection. Adjust p-values with Holm. Per-category differences are shown with intervals only, with no p-values.
- **Verdict wording:**
  - "B higher by X pp (95% CI a–b, p = …)" only when p < 0.05 and the CI excludes 0;
  - otherwise "No detectable difference with this evidence", plus the CI width. An equivalence margin is optional later.

### B.5 UI

- **Runs list.** Add a "repeat i/N" badge with a link to the group.
- **Group page** (`/runs/groups/{id}`). Shows:
  - the run-to-run spread;
  - pass^k and pass@k curves;
  - consistency;
  - the flaky-task table;
  - per-category spread;
  - JSON and HTML downloads.
- **Compare.**
  - The selector lists groups as single entries, e.g. "Group #12 · model X · 5/5 repeats".
  - Comparisons operate on groups, with singles treated as groups of one.
  - The comparison line gains the p-value and verdict.
  - The Answers tab shows per-task pass counts ("3/5") instead of a single score.
- **Calibration.** Repeat groups satisfy the existing `MIN_REPEATS = 3` panel requirement once the cohort key drops `model_seed`.

---

## C. Files touched

| Area | Files |
|---|---|
| Client | `llm_client.py`, `models.py` (`ToolCall`, `ChatMessage`, `RequestMetrics.http_status`, `Question.request`) |
| Suite | new `suites.py`, `tool_suite.py`, `tool_protocol.py`, `schema_subset.py`; `quality_suite.fingerprint`; `quality_execution.execute_question` / `score_response` outcomes; `evaluators.py` (+ `EVALUATOR_VERSIONS`); `interactive_tasks.py` (tool schemas, protocol-agnostic report hooks); `validate_suite.py` |
| Repeats | `schema.sql`, `database.py` MIGRATIONS, `model_settings.decoding_settings`, `run_submission.py`, `benchmark_runner._build_client_config`, new `repeat_stats.py`, `quality_report.paired_comparison`, `calibration.py` cohort key |
| UI | `admin/run_test.html` (suite select, repeats), `runs_list.html`, new `run_group.html`, `compare.html` / `compare.py` / `html_reports.comparison_context`, `quality_summary.html` (wire diagnostics), `run_detail.html` (native transcripts), `dashboard.py` (rigorous-only) |
| Docs | `README.md`, `BENCHMARK_DESIGN.md` |

## D. Verification

- **`selftest_tool_client.py`.** SSE and blocking fixtures:
  - OpenAI-style incremental arguments;
  - a single-chunk call;
  - interleaved parallel indexes;
  - arguments split inside a multi-byte character;
  - a missing id;
  - a name change;
  - a delta after finish;
  - markup leaked into content;
  - `finish_reason: stop` with calls;
  - a vLLM 400 auto-tool-choice body classified as `feature_rejected`;
  - `content: None` history with missing usage.
  - Existing `selftest_client` passes unchanged.
- **`selftest_tool_conformance.py`.**
  - A perfect reference responder scores 100% on every task.
  - Each corruption loses exactly the intended criterion: wrong enum, string instead of int, extra key, dropped parallel call, leaked markup, ignored `tool_choice`, fabricated result value.
  - Native simulations reach the same verdicts as their text twins given the same action sequence.
- **Fingerprint stability.** The Rigorous v15 suite hash and all 323 fingerprints are unchanged.
- **`validate_suite.py --strict`** covers the conformance suite, including the schema-subset check.
- **`selftest_repeats.py`:**
  - pass^k and pass@k against brute-force enumeration;
  - identical groups give D = 0 and p ≈ 1;
  - a planted effect gives a small p and a CI that excludes 0;
  - N = 1 reproduces today's interval;
  - Holm adjustment;
  - excluded members, with reasons;
  - submission expansion, seed plumbing and plan-edit collapse;
  - `selftest_run_options` expectations updated for the `repeats` spec key and the `suite` option.

## E. Delivery order

1. **B1: `repeat_stats.py` plus a p-value on existing comparisons.** Small and self-contained, with immediate value for single runs.
2. **A1: client `complete_chat`, tool-call assembly, wire defects, `feature_rejected`.** No user-visible change; fully covered by fixtures.
3. **A2: `Question.request`, suite registry, `suite` run option, executor, evaluator, schema subset.**
4. **B2: repeats submission, DB columns, seed plumbing.**
5. **A3: task families** (selection, abstention/control, parallel, multi-turn, structured output).
6. **A4: native simulations.** **A5: wire-diagnostics panel.**
7. **B3: group page, compare integration, calibration cohort key.**

Each step lands with its selftests and keeps `ruff` and `validate_suite --strict` green.

## F. Decisions needed

1. **Separate `tool-conformance` suite vs. a new category inside Rigorous.** Recommended: separate suite.
2. **Repeat seeds.** Recommended: vary the seed per repeat (`seed = i`). Also stop `model_seed` alone from making runs incomparable.
3. **`feature_rejected`.** Recommended: a scored failure, shown separately, because a rejected `tool_choice` is a real deployment defect. The alternative is to treat it as unscored, like endpoint errors.
4. **Schema validation.** Recommended: an internal subset validator. The alternative is adding a `jsonschema` dependency.

## G. Out of scope here

- Real coding-agent harnesses over repositories, MCP servers, and vision or audio tool inputs.
- LLM judges.
- Use-case suites. Item 2 of the roadmap builds on the suite registry introduced here.
