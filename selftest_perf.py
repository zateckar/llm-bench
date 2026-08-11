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
from llm_client import ClientConfig
from models import ContextPoint, LatencyStats, RequestMetrics

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
    # 2000 (<192k): 3 probes per level; 300000 (>=192k): max(level, 1) = 1, 2.
    expected = 3 + 3 + 1 + 2
    check("probe counts follow the docs", sweep.total_requests() == expected,
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
    expected = 3 + 3 + 1
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
    test_run_detail_template_renders()
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed.")
        return 1
    print("\nAll context-grid selftests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
