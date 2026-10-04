"""Cache coverage and client-observed prefill/decode rate proxies."""

from app.benchmarking.models import LatencyStats


def rates(values):
    return {key.removesuffix("_ms"): value
            for key, value in LatencyStats.from_samples(values).to_dict().items()}


def cache_metrics(samples, wall_ms):
    good = [m for m in samples if m.ok]
    reported = [m for m in good if (m.cached_tokens_reported or m.cached_tokens > 0)
                and not m.prompt_tokens_estimated and 0 <= m.cached_tokens <= m.prompt_tokens]
    prompt = sum(m.prompt_tokens for m in reported)
    cached = sum(m.cached_tokens for m in reported)
    timed = [m for m in reported if m.streamed and m.attempts == 1
             and m.ttft_ms is not None and m.ttft_ms > 0]
    return {
        "cache_reporting_requests": len(reported),
        "successful_requests": len(good),
        "reported_prompt_tokens": prompt,
        "reported_cached_tokens": cached,
        "cache_hit_fraction": cached / prompt if prompt else None,
        "uncached_prefill_tokens_per_sec": rates([
            (m.prompt_tokens - m.cached_tokens) * 1000 / m.ttft_ms for m in timed
            if m.prompt_tokens > m.cached_tokens
        ]),
        "effective_prompt_tokens_per_sec": rates([
            m.prompt_tokens * 1000 / m.ttft_ms for m in timed
        ]),
        "decode_tokens_per_sec": rates([
            1000 / m.output_token_time_ms for m in good
            if m.attempts == 1 and m.output_token_time_ms is not None and m.output_token_time_ms > 0
        ]),
        "cached_tokens_per_wall_sec": cached * 1000 / wall_ms if wall_ms and reported else None,
    }
