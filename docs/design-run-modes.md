# Quality, performance or both in one run

## Problem

A run had five mutually exclusive modes: `both` (quality + the fixed performance workload), `quality`, `performance`, `sweep` and `load`. The fixed workload was the only performance test that could follow a quality suite. To measure a model's quality and its capacity under an open-loop load, or its context sweep, the user had to submit two runs. Those two runs were not linked, and could be separated by other queued runs.

## Decisions

1. **Two independent parts.** A run measures quality, performance or both. When it measures performance, it runs exactly one performance test: the fixed workload, the context and reasoning sweep, or open-loop load. Any combination is allowed.
2. **Stored options.** `mode` is one of `quality`, `performance` or `both`. `performance` is one of `fixed`, `sweep` or `load`, and is omitted when it is `fixed`. The default suite is omitted too, so historical option snapshots stay byte-identical, as before (`{"mode": "both", "max_concurrency": 8}`).
3. **Legacy modes stay valid.** `mode: "sweep"` and `mode: "load"` are accepted everywhere: queued runs, saved plans, "Run again" and API submissions. They are normalised to `mode: "performance"` with the matching `performance`. Contradictions, such as `mode: "sweep"` with `performance: "load"`, are rejected. One helper, `app/services/run_modes.py`, does this for submission, the runner, the run list and the scorecard.
4. **Order.** Quality runs first. The performance test starts after the quality phase finishes, so the two never compete for the endpoint. If no quality answer could be scored, the endpoint is broken and the performance test is skipped; this is the existing rule for `both`.
5. **One row.** Both results go into the same `test_runs` row: `quality_json` and `test_results` for quality, and `perf_json` (schema 3, 4 or 5) for performance. The run page already has Quality and Performance tabs, and the scorecard already reads quality and performance evidence from the same run, choosing by report schema.
6. **Concurrency.** There is one concurrency setting, and it belongs to the performance test: the maximum concurrency for the fixed workload and the sweep, or the in-flight cap for load. Quality always uses up to four workers, as before.
7. **Failures.**
   - A performance failure after a successful quality phase keeps the quality results and fails the run with "Performance measurement failed: …".
   - Steps that a sweep or load test checkpointed are kept: the final update no longer overwrites a stored `perf_json` with nothing.
   - Performance-only runs follow exactly the previous code path.
8. **Display.**
   - The run form has a **Measure** pair of checkboxes (Quality, Performance), with at least one checked, and a **Performance test** select.
   - The run list's Suite column becomes **Measures**, with a quality badge (suite) and a performance badge (test).
   - Run detail and progress pages show the same description. Rows from before run options were stored are described from their results.

## Not changed

- No report schema, evaluator, suite or comparison protocol changed.
- Canaries stay quality-only.
- Repeat groups summarise quality; a combined repeat group also stores performance in each run.
