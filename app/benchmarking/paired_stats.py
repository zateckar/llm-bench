"""Dependency-free statistics for repeated runs and paired model comparisons.

Every random procedure uses a fixed seed so a saved report reproduces exactly.
Inputs are nested ``{category: {family: [task, ...]}}`` structures whose tasks
carry per-repeat values; categories keep equal weight, as in the headline score.
"""

from math import comb
import random
import statistics

from app.benchmarking.models import percentile

BOOTSTRAP_SAMPLES = 1000
PERMUTATIONS = 10000
BOOTSTRAP_SEED = 90210
PERMUTATION_SEED = 20261005

# Two-sided 95% Student t critical values for 1..30 degrees of freedom.
_T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def t_interval_95(values):
    """Mean interval across independent runs; unavailable below three runs."""
    if len(values) < 3:
        return None
    centre = statistics.mean(values)
    df = len(values) - 1
    half = (_T95[df - 1] if df <= len(_T95) else 1.96) * statistics.stdev(values) / len(values) ** 0.5
    return [centre - half, centre + half]


def pass_hat_k(n, c, k):
    """Probability that k attempts drawn without replacement from n all pass."""
    if not 0 < k <= n or not 0 <= c <= n:
        raise ValueError("pass^k requires 0 < k <= n and 0 <= c <= n")
    return comb(c, k) / comb(n, k)


def pass_at_k(n, c, k):
    """Probability that at least one of k attempts drawn from n passes."""
    if not 0 < k <= n or not 0 <= c <= n:
        raise ValueError("pass@k requires 0 < k <= n and 0 <= c <= n")
    return 1 - comb(n - c, k) / comb(n, k)


def family_differences(tasks):
    """Collapse ``{category: {family: [(left_values, right_values), ...]}}`` to
    per-family mean differences (right minus left)."""
    return {
        category: [
            statistics.mean(statistics.mean(right) - statistics.mean(left) for left, right in pairs)
            for pairs in families.values()
        ]
        for category, families in sorted(tasks.items())
    }


def hierarchical_bootstrap(tasks, samples=BOOTSTRAP_SAMPLES):
    """Percentile interval for the balanced difference covering task-family
    sampling and, where repeats exist, run-to-run variation within each task.

    Families are resampled within categories; for each drawn family every task
    resamples its left and right repeats. With one repeat per side this draws
    exactly the same random sequence as the single-run family bootstrap."""
    if not tasks or any(len(families) < 2 for families in tasks.values()):
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    structure = [list(families.values()) for _, families in sorted(tasks.items())]
    repeated = any(len(left) > 1 or len(right) > 1
                   for families in structure for pairs in families for left, right in pairs)
    if not repeated:
        groups = [[statistics.mean(right[0] - left[0] for left, right in pairs) for pairs in families]
                  for families in structure]
        values = [statistics.mean(statistics.mean(rng.choices(group, k=len(group))) for group in groups)
                  for _ in range(samples)]
        return [percentile(values, 2.5), percentile(values, 97.5)]

    def resampled(values):
        return statistics.mean(rng.choices(values, k=len(values))) if len(values) > 1 else values[0]

    values = []
    for _ in range(samples):
        category_means = []
        for families in structure:
            drawn = rng.choices(families, k=len(families))
            category_means.append(statistics.mean(
                statistics.mean(resampled(right) - resampled(left) for left, right in pairs)
                for pairs in drawn
            ))
        values.append(statistics.mean(category_means))
    return [percentile(values, 2.5), percentile(values, 97.5)]


def sign_flip_test(groups, permutations=PERMUTATIONS):
    """Two-sided paired randomization test of a balanced mean difference.

    ``groups`` maps category to per-family differences. Under the null
    hypothesis each family's difference is equally likely to have either sign;
    flipping signs keeps the category-balanced weighting of the statistic."""
    if not groups:
        return None
    weights = [(1 / len(groups)) / len(values) for values in groups.values()]
    terms = [weight * value
             for weight, values in zip(weights, groups.values()) for value in values]
    observed = abs(sum(terms))
    if observed == 0:
        return 1.0
    nonzero = [term for term in terms if term]
    rng = random.Random(PERMUTATION_SEED)
    tolerance = 1e-12 * max(1.0, observed)
    extreme = 0
    for _ in range(permutations):
        bits = rng.getrandbits(len(nonzero))
        total = 0.0
        for index, term in enumerate(nonzero):
            total += -term if bits >> index & 1 else term
        if abs(total) >= observed - tolerance:
            extreme += 1
    return (1 + extreme) / (1 + permutations)


def mcnemar_exact(left_only, right_only):
    """Exact two-sided McNemar p-value from discordant pair counts."""
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(left_only, right_only) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(p_values):
    """Holm step-down adjustment; ``None`` entries are passed through."""
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted = list(p_values)
    running = 0.0
    m = len(indexed)
    for rank, (p, index) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[index] = running
    return adjusted


def verdict(difference, interval, p_value, alpha=0.05):
    """Plain-language conclusion that never claims more than the evidence."""
    if difference is None:
        return "unavailable"
    if p_value is not None and p_value < alpha and interval and (interval[0] > 0 or interval[1] < 0):
        return "right_higher" if difference > 0 else "left_higher"
    return "no_detectable_difference"
