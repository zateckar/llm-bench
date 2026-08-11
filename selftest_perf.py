#!/usr/bin/env python3
"""Self-tests for the context-length scalability grid in perf.py.

The context sweep turns sizes x concurrency levels into one ContextPoint per
cell; these checks pin the parts that are easy to silently get wrong:

* a grid actually emits one point per (size, level) and fires exactly
  ``ContextSweepConfig.total_requests()`` probes;
* a size whose level-1 probes all refuse on context length is skipped at the
  remaining levels *without* spending more probes;
* levels are clamped (MAX_CONCURRENCY globally, ``max_large_concurrency`` for
  large sizes) and the clamp is recorded on the point;
* the markdown report sorts by (size, level) and closes with one prefill
  degradation line per level;
* run_detail.html still renders against grid-shaped perf_json data.

No network: ``perf._probe`` is replaced with a canned-sample stub, which is
the seam directly below the thread-pool, aggregation and skip logic, so
everything above it runs exactly as in production.

Usage:
    python selftest_perf.py
Exit code 0 if every case behaves as specified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import perf
import llm_client
from llm_client import ClientConfig
from models import ConcurrencyPoint, ContextPoint, LatencyStats, PerfReport, RequestMetrics

ROOT = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
        return
    print(f"  FAIL {name}" + (f" - {detail}" if detail else ""))
    FAILURES.append(name)


def _fake_config() -> ClientConfig:
    return ClientConfig(
        base_url="http://stub.invalid/v1",
        api_key="stub",
        model="stub-model",
        timeout=1.0,
        stream=True,
    )


def _ok_sample() -> perf._Sample:
    return perf._Sample(
        metrics=RequestMetrics(
            ok=True,
            latency_ms=120.0,
            ttft_ms=50.0,
            prompt_tokens=1000,
            completion_tokens=8,
        ),
        text="OK",
    )


def _context_limit_sample() -> perf._Sample:
    return perf._Sample(
        metrics=RequestMetrics(
            ok=False,
            error="HTTP 400: maximum context length exceeded for this model",
        )
    )


class ProbeStub:
    """Context manager replacing perf._probe with a scripted fake.

    ``fail_at`` holds context sizes whose probes must return context-limit
    errors; everything else returns a canned ok sample. The fake recognises
    the probed size from the prompt length, which ``_context_prompt``
    calibrates to 4 chars per target token.
    """

    def __init__(self, fail_at: tuple[int, ...] = ()):
        self.fail_at = fail_at
        self.calls = 0
        self._real = perf._probe

    def __enter__(self) -> ProbeStub:
        def fake(_client, prompt: str, _max_tokens: int, _attempts: int) -> perf._Sample:
            self.calls += 1
            approx_size = len(prompt) // 4
            for size in self.fail_at:
                if abs(approx_size - size) <= size * 0.15 + 50:
                    return _context_limit_sample()
            return _ok_sample()

        perf._probe = fake
        return self

    def __exit__(self, *exc) -> None:
        perf._probe = self._real


def test_grid_emits_one_point_per_cell() -> None:
    print("test_grid_emits_one_point_per_cell")
    sweep = perf.ContextSweepConfig(
        context_sizes=(2000, 300000),
        concurrency_levels=(1, 2),
    )
    config = _fake_config()
    with ProbeStub() as stub:
        points = perf.run_context_sweep(config, sweep)

    check("one ContextPoint per (size, level) cell", len(points) == 4,
          f"got {len(points)} points")
    cells = sorted((p.context_tokens, p.concurrency) for p in points)
    check("cells are (2000,1),(2000,2),(300000,1),(300000,2)",
          cells == [(2000, 1), (2000, 2), (300000, 1), (300000, 2)],
          f"got {cells}")
    check("no cell skipped or errored",
          all(not p.skipped and p.errors == 0 for p in points))
    check("total_requests() matches probes actually fired",
          sweep.total_requests() == stub.calls,
          f"total_requests={sweep.total_requests()} calls={stub.calls}")
    # 2000 (<192k): 3 cold per level; 300000 (>=192k): max(level, 1) = 1, 2;
    # plus one warm probe per cell (4 cells, warm_probes defaults to 1).
    expected = 3 + 3 + 1 + 2 + 4
    check("probe counts follow the docs (cold + warm pair)",
          sweep.total_requests() == expected,
          f"expected {expected}")


def test_skip_short_circuits_remaining_levels() -> None:
    print("test_skip_short_circuits_remaining_levels")
    sweep = perf.ContextSweepConfig(
        context_sizes=(2000, 999000),
        concurrency_levels=(1, 2),
    )
    config = _fake_config()
    with ProbeStub(fail_at=(999000,)) as stub:
        points = perf.run_context_sweep(config, sweep)

    check("still one point per (size, level) cell", len(points) == 4,
          f"got {len(points)} points")
    big = [p for p in points if p.context_tokens == 999000]
    check("both levels of the refused size are marked skipped",
          all(p.skipped for p in big))
    unprobed = [p for p in big if p.concurrency == 2]
    check("the level>1 skipped cell was never probed (requests=0)",
          len(unprobed) == 1 and unprobed[0].requests == 0)
    check("the unprobed cell says why it was not probed",
          bool(unprobed and unprobed[0].notes and "without probing" in unprobed[0].notes[0]))
    # Level 1 of 999000 costs probes_per_size_large=1 probe; levels 2+ cost 0.
    # Warm pairs fire only on the two fully-served 2000-token cells; the
    # refused 999000 cell has nothing to warm against.
    expected = 3 + 3 + 1 + 2
    check("skip saved the level-2 probes of the refused size",
          stub.calls == expected, f"expected {expected} calls, got {stub.calls}")


def test_levels_for_size_clamping() -> None:
    print("test_levels_for_size_clamping")
    sweep = perf.ContextSweepConfig(
        context_sizes=(2000, 300000),
        concurrency_levels=(1, 8),
    )
    small = sweep.levels_for_size(2000)
    large = sweep.levels_for_size(300000)
    check("small sizes keep every requested level", small == (1, 8), f"got {small}")
    check("level 8 at a large size clamps to max_large_concurrency",
          large == (1, 4), f"got {large}")
    check("clamping past MAX_CONCURRENCY merges duplicates",
          perf.ContextSweepConfig(concurrency_levels=(perf.MAX_CONCURRENCY, 9999))
          .levels_for_size(2000) == (perf.MAX_CONCURRENCY,),)
    zero = perf.ContextSweepConfig(concurrency_levels=(0, -3))
    check("levels below 1 clamp up to 1",
          zero.levels_for_size(2000) == (1,), f"got {zero.levels_for_size(2000)}")
    legacy = perf.ContextSweepConfig(context_sizes=(2000, 300000), concurrency_per_size=4)
    check("no grid -> legacy single level per size",
          legacy.levels_for_size(2000) == (4,) and legacy.levels_for_size(300000) == (1,))

    # The clamp note must travel on the point itself, not just the config.
    config = _fake_config()
    with ProbeStub():
        points = perf.run_context_sweep(config, sweep)
    clamped = [p for p in points if p.context_tokens == 300000 and p.concurrency == 4]
    check("clamped point carries a note",
          len(clamped) == 1 and any("clamped" in n for n in clamped[0].notes),
          f"notes={clamped[0].notes if clamped else '<no point>'}")


def _point(size: int, level: int, ttft_ms: float, prompt_tokens: int,
           skipped: bool = False) -> ContextPoint:
    return ContextPoint(
        context_tokens=size,
        concurrency=level,
        requests=1,
        errors=0,
        wall_ms=1000.0,
        latency=LatencyStats.from_samples([ttft_ms * 2]),
        ttft=LatencyStats.from_samples([ttft_ms]),
        prompt_tokens=prompt_tokens,
        output_tokens=8,
        skipped=skipped,
        skip_reason="context length exceeded" if skipped else "",
    )


def test_markdown_summary_lines() -> None:
    print("test_markdown_summary_lines")
    points = [
        _point(64000, 4, ttft_ms=40.0, prompt_tokens=2000),
        _point(32000, 1, ttft_ms=50.0, prompt_tokens=2000),
        _point(64000, 1, ttft_ms=100.0, prompt_tokens=2000),
        _point(32000, 4, ttft_ms=20.0, prompt_tokens=2000),
        _point(128000, 1, ttft_ms=1.0, prompt_tokens=1, skipped=True),
    ]
    text = perf.format_context_sweep_markdown(points)
    rows = [line for line in text.splitlines() if line.startswith("| 3") or line.startswith("| 6")]
    cells = [tuple(int(c.strip().replace(",", "")) for c in row.split("|")[1:3] if c.strip().replace(",", "").isdigit())
             for row in rows]
    check("rows sorted by (context, concurrency)",
          cells == [(32000, 1), (32000, 4), (64000, 1), (64000, 4)], f"got {cells}")
    check("per-level prefill summary line for c=1",
          any(line.startswith("- c=1: prefill fell 2.0x from 32k to 64k")
              for line in text.splitlines()),
          detail="\n".join(line for line in text.splitlines() if line.startswith("- ")))
    check("per-level prefill summary line for c=4",
          any(line.startswith("- c=4: prefill fell 2.0x from 32k to 64k")
              for line in text.splitlines()))
    # Fewer than two measured sizes at a level: nothing to say, and no crash.
    single = perf.format_context_sweep_markdown([_point(32000, 1, 50.0, 2000)])
    check("a single measurable point emits no summary line",
          all(not line.startswith("- c=") for line in single.splitlines()))


def _sweep_point(concurrency: int, latency_ms: float, ttft_ms: float,
                 stream_tps: float, error_rate: float = 0.0) -> ConcurrencyPoint:
    """A ConcurrencyPoint with the SLO-relevant stats populated."""
    requests = 8
    errors = round(requests * error_rate)
    return ConcurrencyPoint(
        concurrency=concurrency, requests=requests, errors=errors,
        wall_ms=1000.0,
        latency=LatencyStats.from_samples([latency_ms] * requests),
        ttft=LatencyStats.from_samples([ttft_ms] * requests),
        output_tokens=96 * requests,
        prompt_tokens=1000 * requests,
        stream_tps=LatencyStats.from_samples([stream_tps] * (requests - errors)),
    )


def test_capacity_verdict_and_knee_note() -> None:
    print("test_capacity_verdict_and_knee_note")
    report = PerfReport(
        slo_ttft_p95_ms=2000.0,
        slo_stream_tps_p50=15.0,
        slo_error_rate=0.01,
        concurrency=[
            _sweep_point(1, 400.0, 100.0, 60.0),
            _sweep_point(2, 450.0, 120.0, 55.0),
            _sweep_point(4, 3000.0, 2500.0, 8.0),
        ],
    )
    check("meets SLO up to the last passing level", report.slo_capacity == 2,
          f"got {report.slo_capacity}")
    check("capacity_users follows from the Little's-law mapping",
          report.capacity_users is not None and abs(report.capacity_users - 2.03) < 0.1,
          f"got {report.capacity_users}")
    check("a recovered level after the knee does not extend capacity",
          PerfReport(concurrency=[
              _sweep_point(1, 400.0, 100.0, 60.0),
              _sweep_point(2, 3000.0, 2500.0, 8.0),
              _sweep_point(4, 400.0, 100.0, 60.0),
          ]).slo_capacity == 1)

    # The note _knee_cause writes names the tripped criterion. It is computed
    # on the *failing* level, not on the capacity.
    knee = report.concurrency[2]
    cause = perf._knee_cause(report, knee)
    check("knee cause names TTFT and per-stream decode",
          "p95 TTFT" in cause and "per-stream decode" in cause, detail=cause)
    check("to_dict exposes the verdict for the web UI",
          report.to_dict()["slo_capacity"] == 2
          and "capacity_users" in report.to_dict()
          and "stream_tps" in report.concurrency[0].to_dict())


def test_capacity_users_unbounded_when_requests_dominate() -> None:
    print("test_capacity_users_unbounded_when_requests_dominate")
    # At the knee the request duration alone consumes the whole per-pair time
    # budget, so no headroom for "other users" exists: the estimate must
    # decline to report rather than return a four-figure fantasy.
    point = ConcurrencyPoint(
        concurrency=4, requests=8, errors=0, wall_ms=8000.0,
        latency=LatencyStats.from_samples([15000.0] * 8),
        ttft=LatencyStats.from_samples([100.0] * 8),
        output_tokens=800, prompt_tokens=1000,
        stream_tps=LatencyStats.from_samples([30.0] * 4),
    )
    report = PerfReport(concurrency=[point], requests_per_user_hour=60.0)
    check("users is None when measured requests would occupy the whole think time",
          report.capacity_users is None, f"got {report.capacity_users}")

    fast = PerfReport(
        concurrency=[ConcurrencyPoint(
            concurrency=4, requests=8, errors=0, wall_ms=2000.0,
            latency=LatencyStats.from_samples([500.0] * 8),
            ttft=LatencyStats.from_samples([100.0] * 8),
            output_tokens=800, prompt_tokens=1000,
            stream_tps=LatencyStats.from_samples([30.0] * 4),
        )],
    )
    check("users tracks concurrency when requests are only a small slice of the hour",
          fast.capacity_users is not None and abs(fast.capacity_users - 4.1) < 0.2,
          f"got {fast.capacity_users}")


def test_full_suite_populates_capacity_stats() -> None:
    print("test_full_suite_populates_capacity_stats")

    class InstantStub:
        """_probe that returns immediately with canned metrics; every level
        measures, SLO passes, and the report carries the capacity fields the
        web UI reads."""

        def __init__(self):
            self.calls = 0
            self._real = perf._probe

        def __enter__(self) -> InstantStub:
            def fake(_client, prompt: str, max_tokens: int, _attempts: int) -> perf._Sample:
                self.calls += 1
                return perf._Sample(metrics=RequestMetrics(
                    ok=True, latency_ms=200.0, ttft_ms=80.0,
                    prompt_tokens=120, completion_tokens=max_tokens,
                ))

            perf._probe = fake
            return self

        def __exit__(self, *exc) -> None:
            perf._probe = self._real

    sweep = perf.PerfConfig(
        serial_samples=4,
        measure_decode=False,
        measure_prefill=False,
        concurrency_levels=(1, 2),
        requests_per_level=8,
    )
    config = _fake_config()
    with InstantStub():
        report = perf.run_perf_suite(config, sweep)

    by_level = {p.concurrency: p for p in report.concurrency}
    check("all requested levels were measured",
          sorted(by_level) == [1, 2], f"got {[p.concurrency for p in report.concurrency]}")
    check("each level carries per-stream decode stats",
          all(p.stream_tps and p.stream_tps.p50 is not None for p in report.concurrency))
    check("SLO capacity is set and notes a highest-measured level",
          report.slo_capacity == 2
          and any("highest measured level" in n for n in report.notes),
          detail=f"capacity={report.slo_capacity} notes={report.notes}")
    check("markdown and console render without crashing",
          "Meets SLO up to" in perf.format_perf_markdown(report)
          and "SLO capacity" in perf.format_perf_console(report))


def test_workload_mix_and_shared_prefix() -> None:
    print("test_workload_mix_and_shared_prefix")

    captured: list[tuple[str, str, int]] = []

    class MixStub:
        """Records (task-name, prompt, max_tokens) per load probe.

        Only requests whose prompt carries a ``[task]`` marker are load
        probes; the warmup/serial probes use their own prompt shapes.
        """

        def __init__(self):
            self._real = perf._probe

        def __enter__(self) -> MixStub:
            def fake(_client, prompt: str, max_tokens: int, _attempts: int) -> perf._Sample:
                for task in perf.WORKLOAD_MIX:
                    if f"[{task.name}] " in prompt:
                        captured.append((task.name, prompt, max_tokens))
                        break
                return perf._Sample(metrics=RequestMetrics(
                    ok=True, latency_ms=200.0, ttft_ms=80.0,
                    prompt_tokens=120, completion_tokens=max_tokens,
                ))

            perf._probe = fake
            return self

        def __exit__(self, *exc) -> None:
            perf._probe = self._real

    mixer = perf.PerfConfig(
        serial_samples=2,
        measure_decode=False,
        measure_prefill=False,
        concurrency_levels=(1,),
        requests_per_level=30,
        workload_mix=perf.WORKLOAD_MIX,
        shared_prefix=True,
    )
    with MixStub():
        report = perf.run_perf_suite(_fake_config(), mixer)

    check("the load phase drew probes for the mix",
          len(captured) >= 30, f"got {len(captured)} load probes")
    names = {t[0] for t in captured}
    check("the load phase drew from every workload task",
          {"chat", "summarize", "codegen"} <= names, f"got {sorted(names)}")
    check("each task's probe carries its own max_tokens",
          all(mt in {t.max_tokens for t in perf.WORKLOAD_MIX} for _, _, mt in captured),
          f"got {sorted({mt for _, _, mt in captured})}")
    check("probes carry the shared prefix when enabled",
          all(perf._FILLER.split(".")[0] in p for _, p, _ in captured),
          detail="sample probe: " + (captured[0][1][:80] if captured else "<none>"))
    point = report.concurrency[0]
    check("task_stats is populated per workload task",
          set(point.task_stats) >= {"chat", "summarize", "codegen"},
          f"got {sorted(point.task_stats)}")
    check("task_stats survive to_dict for the web UI",
          "chat" in point.to_dict()["task_stats"])

    # With the default uniform mix the classic probes are unchanged; the
    # warmup requests also flow through _probe, so only the load-phase
    # count-prompts are captured here.
    load_prompts: list[str] = []
    real = perf._probe

    def classic(_client, prompt: str, max_tokens: int, _attempts: int) -> perf._Sample:
        if "list the integers" in prompt:
            load_prompts.append(prompt)
        return perf._Sample(metrics=RequestMetrics(
            ok=True, latency_ms=200.0, ttft_ms=80.0,
            prompt_tokens=120, completion_tokens=max_tokens,
        ))

    perf._probe = classic
    try:
        classic_cfg = perf.PerfConfig(
            serial_samples=2, measure_decode=False, measure_prefill=False,
            concurrency_levels=(1,), requests_per_level=4,
        )
        perf.run_perf_suite(_fake_config(), classic_cfg)
    finally:
        perf._probe = real
    check("uniform mix keeps the classic count-prompt probes",
          len(load_prompts) >= 4 and all("list the integers" in p for p in load_prompts),
          f"got {len(load_prompts)} load probes")


def test_runner_builds_perf_config_from_form_values() -> None:
    print("test_runner_builds_perf_config_from_form_values")
    from app.services import benchmark_runner as br

    defaults = br.build_perf_config((1, 2))
    check("empty form values fall back to PerfConfig defaults",
          defaults.workload_mix is None
          and defaults.shared_prefix is False
          and defaults.slo_ttft_p95_ms == perf.PerfConfig().slo_ttft_p95_ms)

    cfg = br.build_perf_config(
        (1, 2),
        workload_mix="mixed", shared_prefix=True,
        slo_ttft_ms=500, slo_tps=25, slo_errors=0.02, req_per_user_h=90,
    )
    check("form values land on the PerfConfig",
          cfg.workload_mix == perf.WORKLOAD_MIX
          and cfg.shared_prefix is True
          and cfg.slo_ttft_p95_ms == 500
          and cfg.slo_stream_tps_p50 == 25
          and cfg.slo_error_rate == 0.02
          and cfg.requests_per_user_hour == 90)
    check("unknown mix values keep the classic uniform probes",
          br.build_perf_config((1,), workload_mix="uniform").workload_mix is None)


def test_parse_usage_cache_shapes() -> None:
    print("test_parse_usage_cache_shapes")
    # vLLM-style top-level field wins over the OpenAI-style nested detail.
    v = llm_client._parse_usage(
        {"prompt_tokens": 1000, "completion_tokens": 8, "prompt_cache_hit_tokens": 950})
    check("vLLM prompt_cache_hit_tokens parsed", v.cached_tokens == 950
          and v.prompt_tokens == 1000, f"got {v.cached_tokens}")
    o = llm_client._parse_usage(
        {"prompt_tokens": 1000, "completion_tokens": 8,
         "prompt_tokens_details": {"cached_tokens": 800}})
    check("OpenAI prompt_tokens_details.cached_tokens parsed", o.cached_tokens == 800)
    both = llm_client._parse_usage(
        {"prompt_tokens": 1000, "completion_tokens": 8,
         "prompt_cache_hit_tokens": 950,
         "prompt_tokens_details": {"cached_tokens": 800}})
    check("explicit top-level field beats the nested detail", both.cached_tokens == 950)
    none = llm_client._parse_usage({"prompt_tokens": 1000, "completion_tokens": 8})
    check("absent cache info coerces to 0, not a crash", none.cached_tokens == 0)
    bad = llm_client._parse_usage(
        {"prompt_tokens": 1000, "prompt_tokens_details": "garbage"})
    check("malformed prompt_tokens_details coerces to 0", bad.cached_tokens == 0)


def test_context_warm_probe_pairing() -> None:
    print("test_context_warm_probe_pairing")

    class WarmStub:
        """Records each prompt; reports a fixed cache hit on warm repeats.

        A prompt whose document id reappears is the warm probe (same string as
        cold probe 0 of its cell); the stub reports cached_tokens on it so the
        aggregation can be checked.
        """

        def __init__(self):
            self.prompts: list[str] = []
            self._seen: set[str] = set()
            self._real = perf._probe

        def __enter__(self) -> WarmStub:
            def fake(_client, prompt: str, max_tokens: int, _attempts: int) -> perf._Sample:
                self.prompts.append(prompt)
                repeat = prompt in self._seen
                self._seen.add(prompt)
                return perf._Sample(metrics=RequestMetrics(
                    ok=True, latency_ms=200.0 if not repeat else 60.0,
                    ttft_ms=80.0 if not repeat else 20.0,
                    prompt_tokens=1000, completion_tokens=max_tokens,
                    cached_tokens=950 if repeat else 0,
                ))

            perf._probe = fake
            return self

        def __exit__(self, *exc) -> None:
            perf._probe = self._real

    sweep = perf.ContextSweepConfig(context_sizes=(2000,), concurrency_levels=(1,),
                                    probes_per_size_small=3, warm_probes=2)
    with WarmStub() as stub:
        points = perf.run_context_sweep(_fake_config(), sweep)

    check("total_requests counts cold + warm probes",
          sweep.total_requests() == len(stub.prompts),
          f"total={sweep.total_requests()} fired={len(stub.prompts)}")
    point = points[0]
    check("each warm probe re-ran cold probe 0's exact prompt",
          stub.prompts[3] == stub.prompts[0] and stub.prompts[4] == stub.prompts[0])
    check("warm_ttft aggregates the warm pair",
          point.warm_ttft.count == 2 and point.warm_ttft.mean == 20.0,
          f"got {point.warm_ttft}")
    check("warm prefill rate reflects the cache-hit TTFT",
          point.warm_prompt_tokens_per_sec
          and abs(point.warm_prompt_tokens_per_sec - 1000 / (20.0 / 1000.0)) < 1.0,
          f"got {point.warm_prompt_tokens_per_sec}")
    check("server-reported cache hits are summed over warm probes",
          point.cached_tokens == 1900, f"got {point.cached_tokens}")
    check("to_dict carries the warm fields for the UI",
          all(k in point.to_dict() for k in (
              "warm_ttft", "warm_prompt_tokens", "cached_tokens",
              "warm_prompt_tokens_per_sec")))

    # warm_probes=0 disables the pairing entirely (backward compatible).
    dry = perf.ContextSweepConfig(context_sizes=(2000,), concurrency_levels=(1,),
                                  probes_per_size_small=3, warm_probes=0)
    with WarmStub() as stub2:
        dry_points = perf.run_context_sweep(_fake_config(), dry)
    check("warm_probes=0 leaves the cold sweep unchanged",
          len(stub2.prompts) == 3 and dry_points[0].warm_ttft.count == 0)


def test_prefix_cache_probe() -> None:
    print("test_prefix_cache_probe")

    class CacheStub:
        """Cold first call per unique prompt is slow; repeats are cache-warm."""

        def __init__(self):
            self.seen: dict[str, int] = {}
            self._real = perf._probe

        def __enter__(self) -> CacheStub:
            def fake(_client, prompt: str, max_tokens: int, _attempts: int) -> perf._Sample:
                n = self.seen.get(prompt, 0)
                self.seen[prompt] = n + 1
                is_cache_probe = prompt.startswith("Cache probe ")
                warm = is_cache_probe and n > 0
                return perf._Sample(metrics=RequestMetrics(
                    ok=True, latency_ms=(60.0 if warm else 200.0),
                    ttft_ms=(20.0 if warm else 80.0),
                    prompt_tokens=1000, completion_tokens=max_tokens,
                    cached_tokens=(950 if warm else 0),
                ))

            perf._probe = fake
            return self

        def __exit__(self, *exc) -> None:
            perf._probe = self._real

    sweep = perf.PerfConfig(
        serial_samples=2, measure_decode=False, measure_prefill=False,
        measure_concurrency=False, prefill_prompt_tokens=8000,
        prefix_cache_warm_probes=2,
    )
    with CacheStub() as stub:
        report = perf.run_perf_suite(_fake_config(), sweep)

    cp = report.cache_probe
    check("cache_probe is populated", cp is not None)
    if cp:
        check("cold and warm probes reuse the identical prompt",
              sum(1 for p in stub.seen if p.startswith("Cache probe ")) == 1
              and max(stub.seen[p] for p in stub.seen if p.startswith("Cache probe ")) == 3,
              f"counts={ {p[:20]: c for p, c in stub.seen.items() if p.startswith('Cache probe')} }")
        check("hit ratio is cached/warm-prompt", cp.cache_hit_ratio is not None
              and abs(cp.cache_hit_ratio - 1900 / 2000) < 0.01,
              f"got {cp.cache_hit_ratio}")
        check("TTFT speedup is cold/warm", cp.ttft_speedup is not None
              and abs(cp.ttft_speedup - 80.0 / 20.0) < 0.01, f"got {cp.ttft_speedup}")
        check("prefill gain is warm/cold rate", cp.prefill_gain is not None
              and cp.prefill_gain > 1.0, f"got {cp.prefill_gain}")
        check("to_dict round-trips the cache probe", report.to_dict()["cache_probe"]["cache_hit_ratio"] is not None)
        check("markdown + console carry the cache lines",
              "Prefix cache" in perf.format_perf_markdown(report)
              and "cache" in perf.format_perf_console(report))

    # The probe prompt carries its own nonce at the head and the distinct
    # _CACHE_FILLER body, so it is disjoint from the _FILLER-based prompts of
    # the other phases (the contamination the nonce exists to prevent).
    heads = [p.split(":", 1)[0] for p in stub.seen if p.startswith("Cache probe ")]
    check("cache probe prompt is nonce-headed and a single string per run",
          len(heads) == 1 and heads[0].startswith("Cache probe ")
          and perf._CACHE_FILLER.split(".")[0]
              in [p for p in stub.seen if p.startswith("Cache probe ")][0])

    # Cold failure: note, not abort, cache_probe stays None.
    real = perf._probe

    def failing(_client, prompt: str, max_tokens: int, _attempts: int) -> perf._Sample:
        return perf._Sample(metrics=RequestMetrics(ok=False, error="boom"))

    perf._probe = failing
    try:
        report2 = perf.run_perf_suite(_fake_config(), sweep)
    finally:
        perf._probe = real
    check("cold failure leaves cache_probe None with a note",
          report2.cache_probe is None
          and any("Prefix-cache probe" in n for n in report2.notes))


def test_run_detail_template_renders() -> None:
    print("test_run_detail_template_renders")
    import jinja2

    from app.templates_config import templates

    grid = [
        _point(2000, 1, 50.0, 2000),
        _point(2000, 2, 80.0, 2000),
        _point(300000, 1, 500.0, 300000),
        _point(300000, 2, 900.0, 300000, skipped=True),
    ]
    context_points = [p.to_dict() for p in grid]

    run = {
        "id": 1, "model_name": "Stub", "model_id": "stub-model",
        "base_url": "http://stub.invalid", "status": "completed",
        "passed_questions": 1, "scored_questions": 1, "total_questions": 1,
        "avg_score": 1.0, "weighted_score": 1.0, "error_count": 0,
        "started_at": "2026-01-01T00:00:00", "completed_at": "2026-01-01T00:01:00",
        "test_suite_hash": "deadbeef", "total_prompt_tokens": 2100,
        "total_completion_tokens": 16, "error_message": "",
        "duration_ms": 60_000.0, "workers": 1,
        "latency_p50_ms": 100.0, "latency_p95_ms": 200.0, "latency_p99_ms": 250.0,
        "ttft_p50_ms": 40.0, "ttft_p95_ms": 90.0, "output_tokens_per_sec": 0.3,
    }
    categories = [{
        "category": "general", "total": 1, "scored": 1, "passed": 1,
        "avg_score": 1.0, "weighted_score": 1.0,
        "avg_latency_ms": 100.0, "max_latency_ms": 100.0,
    }]
    result_row = {
        "test_id": "general-001", "category": "general", "prompt": "ping",
        "response": "pong", "score": 1.0, "detail": "ok", "evaluator": "exact_match",
        "prompt_tokens": 1000, "completion_tokens": 8, "passed": 1,
        "pass_threshold": 1.0, "difficulty": "easy", "latency_ms": 100.0,
        "ttft_ms": 40.0, "request_ok": 1,
    }
    env = templates.env
    template = env.get_template("run_detail.html")

    from starlette.requests import Request

    fake_request = Request(scope={
        "type": "http", "method": "GET", "path": "/runs/1",
        "scheme": "http", "server": ("stub", 80), "headers": [],
        "query_string": b"", "client": ("127.0.0.1", 12345),
        "state": {},
    })
    fake_request.state.user = None
    try:
        html = template.render(
            request=fake_request,
            run=run,
            categories=categories,
            results_by_category={"general": [result_row]},
            evaluator_errors=[],
            transport_errors=[],
            difficulty_rows=[{
                "difficulty": "easy", "total": 1, "scored": 1,
                "passed": 1, "avg_score": 1.0,
            }],
            perf=None,
            context_points=context_points,
        )
    except jinja2.TemplateError as e:
        check("run_detail.html renders", False, str(e))
        return
    check("run_detail.html renders", "<canvas" in html and len(html) > 1000)
    check("both context charts present",
          'id="contextChart"' in html and 'id="contextPrefillChart"' in html)
    check("grid points serialised into the chart script",
          '"context_tokens": 300000' in html or '"context_tokens":300000' in html)


def main() -> int:
    test_grid_emits_one_point_per_cell()
    test_skip_short_circuits_remaining_levels()
    test_levels_for_size_clamping()
    test_markdown_summary_lines()
    test_capacity_verdict_and_knee_note()
    test_capacity_users_unbounded_when_requests_dominate()
    test_full_suite_populates_capacity_stats()
    test_workload_mix_and_shared_prefix()
    test_runner_builds_perf_config_from_form_values()
    test_parse_usage_cache_shapes()
    test_context_warm_probe_pairing()
    test_prefix_cache_probe()
    test_run_detail_template_renders()
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed.")
        return 1
    print("\nAll context-grid selftests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
