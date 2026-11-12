"""P8 long context, stages 2-3: scoring and the registered statistics (research/prereg/2026-11-01-p8-longctx-stages2-3.md,
Amendment 1, item 10).

  - score: P6's visible_answer then score_longctx (testsuite/paper6/p6common.py, imported unchanged): 1 when every
    required answer appears, case-insensitive, in the answer with any think block removed.
  - select_level: the calibration rule, the level whose 8,8 accuracy is closest to 65%, the harder on a tie.
  - paired_interval: a 90% percentile bootstrap of the mean paired difference in points, resampling items with
    replacement, 10,000 resamples, Python's random.Random(20261102); quantiles by linear interpolation between order
    statistics (numpy's default).
  - mcnemar: the exact two-sided binomial test on the discordant pairs.
  - holm: Holm-adjusted p-values. wilson: the Wilson 95% interval.
"""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "testsuite", "paper6"))
import p6common as P6  # noqa: E402  (unchanged)

TARGET_ACC = 0.65
BOOT_N, BOOT_SEED, BOOT_LEVEL = 10_000, 20261102, 0.90
USABLE_LB, EQUIV = -10.0, 5.0


def score(text, item):
    return int(P6.score_longctx(P6.visible_answer({"content": text}), item))


def select_level(acc_by_level):
    """acc_by_level: {level: accuracy in 0..1}. Closest to 65%; on an exact tie, the larger (harder) level."""
    return min(acc_by_level, key=lambda lv: (round(abs(acc_by_level[lv] - TARGET_ACC), 12), -lv))


def quantile(sorted_xs, q):
    pos = (len(sorted_xs) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def paired_interval(a, b, n_boot=BOOT_N, seed=BOOT_SEED, level=BOOT_LEVEL):
    """a, b: 0/1 scores of the same items under two arms. Mean of a - b in points, and its percentile interval."""
    if len(a) != len(b) or not a:
        raise ValueError("paired scores of equal, non-zero length needed")
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(100.0 * s / n)
    means.sort()
    alpha = (1 - level) / 2
    return 100.0 * sum(d) / n, quantile(means, alpha), quantile(means, 1 - alpha)


def verdict(lo, hi):
    return {"usable": lo > USABLE_LB, "equivalent": lo >= -EQUIV and hi <= EQUIV}


def mcnemar(a, b):
    """Exact two-sided McNemar on paired 0/1 scores: (n10, n01, p) with n10 = a right and b wrong."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    tail = sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n
    return n10, n01, min(1.0, 2 * tail)


def holm(pvals):
    """{name: p} -> {name: Holm-adjusted p}."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)
