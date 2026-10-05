"""Statistics for blind A/B studies (``ab-stats-v1``).

A pair's value is the preference for B: a B win is 1, a tie ½ and an A win 0.
Several human votes on one pair are averaged first, so one pair is one
observation. Categories weigh equally and families form clusters, as in the
quality reports.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import comb
import random
import statistics

REVISION = "ab-stats-v1"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 4242
ALPHA = 0.05
VOTE_VALUE = {"a": 0.0, "b": 1.0, "tie": 0.5, "both_bad": 0.5}
JUDGE_VALUE = {"a": 0.0, "b": 1.0, "tie": 0.5}


def _groups(items):
    groups = defaultdict(lambda: defaultdict(list))
    for item in items:
        groups[item["category"]][item["family"]].append(item["value"])
    return {category: [statistics.mean(v) for v in families.values()] for category, families in groups.items()}


def balanced(groups):
    return statistics.mean(statistics.mean(v) for v in groups.values()) if groups else None


def bootstrap(groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    if not groups or sum(len(v) for v in groups.values()) < 2:
        return None
    rng = random.Random(seed)
    values = sorted(
        statistics.mean(statistics.mean(rng.choices(v, k=len(v))) for v in groups.values())
        for _ in range(draws)
    )
    return [values[int(0.025 * (draws - 1))], values[int(round(0.975 * (draws - 1)))]]


def sign_test(wins_a, wins_b):
    """Exact two-sided binomial test of decisive pairs against ½."""
    n = wins_a + wins_b
    if n == 0:
        return None
    k = min(wins_a, wins_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def side(value):
    return "b" if value > 0.5 else "a" if value < 0.5 else "tie"


def preference(items):
    """``items``: dicts with category, family and value (preference for B)."""
    items = list(items)
    groups = _groups(items)
    sides = Counter(side(i["value"]) for i in items)
    p = sign_test(sides["a"], sides["b"])
    ci = bootstrap(groups)
    point = balanced(groups)
    if not items:
        verdict = "no_data"
    elif ci and p is not None and p < ALPHA and ci[0] > 0.5:
        verdict = "b"
    elif ci and p is not None and p < ALPHA and ci[1] < 0.5:
        verdict = "a"
    else:
        verdict = "unclear"
    categories = {}
    for category in sorted(groups):
        rows = [i for i in items if i["category"] == category]
        counts = Counter(side(i["value"]) for i in rows)
        categories[category] = {"pairs": len(rows), "preference_b": statistics.mean(groups[category]),
                                "a": counts["a"], "b": counts["b"], "tie": counts["tie"]}
    return {"pairs": len(items), "preference_b": point, "ci95": ci, "p_value": p,
            "a": sides["a"], "b": sides["b"], "tie": sides["tie"], "verdict": verdict,
            "categories": categories}


def human_items(pairs, votes):
    """One item per voted pair; ``votes`` are rows with pair_id and verdict."""
    by_pair = defaultdict(list)
    for vote in votes:
        by_pair[vote["pair_id"]].append(vote["verdict"])
    items = []
    for pair in pairs:
        verdicts = by_pair.get(pair["id"])
        if verdicts:
            items.append({"pair_id": pair["id"], "category": pair["category"], "family": pair["family"],
                          "value": statistics.mean(VOTE_VALUE[v] for v in verdicts),
                          "label": human_label(verdicts), "votes": len(verdicts)})
    return items


def human_label(verdicts):
    """Majority over {a, b, tie}; both-bad counts as a tie and a split is a tie."""
    counts = Counter("tie" if v == "both_bad" else v for v in verdicts).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "tie"
    return counts[0][0]


def judge_items(pairs, judgments):
    by_pair = {j["pair_id"]: j for j in judgments if j.get("verdict")}
    return [{"pair_id": p["id"], "category": p["category"], "family": p["family"],
             "value": JUDGE_VALUE[by_pair[p["id"]]["verdict"]], "label": by_pair[p["id"]]["verdict"],
             "consistent": bool(by_pair[p["id"]]["consistent"])}
            for p in pairs if p["id"] in by_pair]


def judge_diagnostics(items, lengths):
    """Position consistency and how often the longer answer wins a decisive verdict.

    ``lengths`` maps pair id to (len(answer_a), len(answer_b))."""
    consistency = statistics.mean(i["consistent"] for i in items) if items else None
    decisive = [i for i in items if i["label"] in {"a", "b"} and lengths[i["pair_id"]][0] != lengths[i["pair_id"]][1]]
    longer = sum(
        (i["label"] == "a") == (lengths[i["pair_id"]][0] > lengths[i["pair_id"]][1]) for i in decisive
    )
    return {"position_consistency": consistency, "decisive_with_length_difference": len(decisive),
            "longer_answer_win_rate": longer / len(decisive) if decisive else None}


def agreement(judge, human):
    """Raw agreement and Cohen's kappa over {a, b, tie} on pairs both judged."""
    human_by_pair = {i["pair_id"]: i["label"] for i in human}
    shared = [(i["label"], human_by_pair[i["pair_id"]]) for i in judge if i["pair_id"] in human_by_pair]
    if not shared:
        return {"pairs": 0, "agreement": None, "kappa": None}
    n = len(shared)
    observed = sum(a == b for a, b in shared) / n
    left, right = Counter(a for a, _ in shared), Counter(b for _, b in shared)
    expected = sum(left[k] * right[k] for k in ("a", "b", "tie")) / n ** 2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {"pairs": n, "agreement": observed, "kappa": kappa}
