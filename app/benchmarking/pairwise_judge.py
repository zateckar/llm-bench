"""Pairwise LLM judge for blind A/B studies (``pairwise-judge-v1``).

Every pair is judged twice, once in each order, so position bias cancels: A
scores 1, a tie ½ and B 0 per order, and the mean decides. Malformed output or
a failed request is a recorded judge error, never a verdict.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import logging
import re
import threading

from app.benchmarking.evaluators import extract_json, strip_think_blocks

logger = logging.getLogger(__name__)

REVISION = "pairwise-judge-v1"
ANSWER_LIMIT = 24_000
WORKERS = 4
MAX_TOKENS = 8192
STOP_AFTER_ERRORS = 8
REASON_LIMIT = 2000
DEFAULT_CRITERIA = (
    "Does what the request asks and follows the assistant instructions, including language and format",
    "Is accurate and does not invent facts",
    "Is complete for the request without padding or unnecessary content",
    "Is clear and well organised",
)
SYSTEM = (
    "You are an impartial expert reviewer. You compare two AI assistant answers to the same "
    "request and decide which one better serves the person who asked. Judge only by the criteria "
    "given. Do not prefer an answer because it is longer, comes first, or sounds confident. If both "
    "answers are equally good or equally bad, the result is a tie."
)
VALUE = {"a": 1.0, "tie": 0.5, "b": 0.0}


def _clip(text):
    text = text or ""
    if len(text) <= ANSWER_LIMIT:
        return text, False
    return text[:ANSWER_LIMIT], True


def judge_prompt(pair, order):
    """The judge request for one order: ``"ab"`` shows A first, ``"ba"`` B first."""
    first, second = (pair["answer_a"], pair["answer_b"]) if order == "ab" else (pair["answer_b"], pair["answer_a"])
    criteria = pair.get("criteria") or list(DEFAULT_CRITERIA)
    parts = []
    if pair.get("system_prompt"):
        parts.append(f"## Assistant instructions\n{pair['system_prompt']}")
    parts.append(f"## User request\n{pair['prompt']}")
    parts.append("## Criteria\n" + "\n".join(f"- {c}" for c in criteria))
    if pair.get("reference"):
        parts.append("## Reference answer (one good answer; others can also be good)\n" + pair["reference"])
    for number, answer in ((1, first), (2, second)):
        text, clipped = _clip(answer)
        note = f"\n[Answer {number} was truncated to {ANSWER_LIMIT:,} characters for review.]" if clipped else ""
        parts.append(f"## Answer {number}\n<answer_{number}>\n{text}\n</answer_{number}>{note}")
    parts.append(
        "## Your task\nCompare the two answers against the criteria. Reply with exactly one JSON "
        'object and nothing else: {"reason": "<one to three sentences>", "winner": "1" | "2" | "tie"}'
    )
    return "\n\n".join(parts)


def parse_verdict(text):
    """Return ``(winner, reason)`` with winner "1", "2" or "tie"; raise ValueError otherwise."""
    answer = strip_think_blocks(text or "").strip()
    data, _error = extract_json(answer, strict=False, allow_fence=True)
    if not isinstance(data, dict):
        raise ValueError("The judge did not return a JSON object")
    winner = data.get("winner")
    if isinstance(winner, int) and not isinstance(winner, bool):
        winner = str(winner)
    if not isinstance(winner, str):
        raise ValueError("The judge's JSON has no winner")
    winner = re.sub(r"^answer\s*", "", winner.strip().lower())
    if winner not in {"1", "2", "tie"}:
        raise ValueError(f"Unknown winner {data.get('winner')!r}")
    reason = data.get("reason")
    return winner, (reason if isinstance(reason, str) else "")[:REASON_LIMIT]


def to_side(winner, order):
    if winner == "tie":
        return "tie"
    shown_first = order[0]
    return shown_first if winner == "1" else ("b" if shown_first == "a" else "a")


def combine(first, second):
    """Verdict and position consistency from the two orders' sides."""
    mean = (VALUE[first] + VALUE[second]) / 2
    verdict = "a" if mean > 0.5 else "b" if mean < 0.5 else "tie"
    return verdict, first == second


def judge_pair(client, pair):
    """Judge one pair in both orders; a failure is returned as ``error``."""
    out = {"first": None, "second": None, "verdict": None, "consistent": None,
           "reason_first": None, "reason_second": None, "error": None}
    for order, key in (("ab", "first"), ("ba", "second")):
        text, _tokens, metrics = client.complete(judge_prompt(pair, order), SYSTEM, max_tokens=MAX_TOKENS)
        if not metrics.ok:
            out["error"] = f"Judge request failed: {metrics.error}"[:500]
            return out
        try:
            winner, reason = parse_verdict(text)
        except ValueError as error:
            out["error"] = f"Unusable judge output ({order}): {error}"[:500]
            return out
        out[key] = to_side(winner, order)
        out[f"reason_{key}"] = reason
    out["verdict"], out["consistent"] = combine(out["first"], out["second"])
    return out


def judge_config(model, timeout=300.0):
    """Client settings for a judge model: deterministic, bounded output."""
    from app.benchmarking.llm_client import ClientConfig

    return ClientConfig(
        base_url=model["base_url"], api_key=model["api_key"], model=model["model_id"],
        max_tokens=MAX_TOKENS, temperature=0, seed=0, timeout=timeout, stream_deadline=900.0,
        stream=True, reasoning_effort=model.get("reasoning_effort"),
    )


def run_judge(judge_id, config, *, client_factory=None, workers=WORKERS):
    """Judge every pair of the study that this judge has not judged successfully.

    Each judgment is stored as it finishes, so a cancelled or interrupted judge
    resumes where it stopped. The status row is the cancellation signal."""
    from app.benchmarking.llm_client import ChatClient
    from app.services import ab_studies

    factory = client_factory or ChatClient
    config = replace(config, detect_repetition=False)
    todo = ab_studies.pending_pairs(judge_id)
    lock = threading.Lock()
    local = threading.local()
    clients = []
    state = {"consecutive": 0, "stopped": None, "last_error": None}

    def active():
        return state["stopped"] is None and ab_studies.judge_status(judge_id) == "running"

    def job(pair):
        if not active():
            return
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = factory(config)
            with lock:
                clients.append(client)
        try:
            outcome = judge_pair(client, pair)
        except Exception as error:  # noqa: BLE001 - one bad pair must not stop the judge
            logger.exception("Judging pair %s failed", pair["id"])
            outcome = {"error": f"Judge error: {error}"[:500]}
        with lock:
            ab_studies.store_judgment(judge_id, pair["id"], outcome)
            if outcome.get("error"):
                state["last_error"] = outcome["error"]
                state["consecutive"] += 1
                if state["consecutive"] >= STOP_AFTER_ERRORS:
                    state["stopped"] = f"Stopped after {STOP_AFTER_ERRORS} consecutive judge errors. {outcome['error']}"
            else:
                state["consecutive"] = 0

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for future in [pool.submit(job, pair) for pair in todo]:
                future.result()
    except Exception as error:  # noqa: BLE001 - recorded on the judge row
        logger.exception("Judge %d failed", judge_id)
        state["stopped"] = f"Judge failed: {error}"
    finally:
        for client in clients:
            session = getattr(client, "session", None)
            if session is not None:
                session.close()
    ab_studies.finish_judge(judge_id, state["stopped"])


def start_judge_thread(judge_id, config):
    thread = threading.Thread(target=run_judge, args=(judge_id, config), daemon=True,
                              name=f"ab-judge-{judge_id}")
    thread.start()
    return thread
