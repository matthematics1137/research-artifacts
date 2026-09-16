"""Verbatim statistics only; no source-data or machine-local loading."""
from __future__ import annotations
import math
import statistics
from fractions import Fraction
Z_95 = statistics.NormalDist().inv_cdf(.975)

# Full source SHA256: fccf114d57b887aff1350de6ffa0cb147c417c65932ae8c1848cd26812668fb6
def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"invalid binomial count {k}/{n}")
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact_two_sided(low_only: int, healthy_only: int) -> float:
    """Exact two-sided McNemar p (binomial test on discordant pairs)."""
    if low_only < 0 or healthy_only < 0:
        raise ValueError("negative discordance count")
    discordant = low_only + healthy_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k)
               for k in range(min(low_only, healthy_only) + 1))
    return float(min(Fraction(1, 1), Fraction(2 * tail, 2 ** discordant)))


def newcombe_paired_risk_difference_interval(
    both_correct: int,
    low_only: int,
    healthy_only: int,
    both_wrong: int,
    z: float = Z_95,
) -> tuple[float, float, float, float]:
    """Newcombe method-10 hybrid-score CI for healthy-minus-low accuracy.

    The two marginal Wilson intervals are combined with the observed phi
    correlation.  This is deterministic, respects pairing, and behaves much
    better than the paired Wald interval at the near-ceiling cells in this study.
    Returns (risk_difference, lower, upper, phi).
    """
    counts = (both_correct, low_only, healthy_only, both_wrong)
    if any(value < 0 for value in counts):
        raise ValueError("negative paired-table count")
    n = sum(counts)
    if n <= 0:
        raise ValueError("empty paired table")
    healthy_correct = both_correct + healthy_only
    low_correct = both_correct + low_only
    p_healthy = healthy_correct / n
    p_low = low_correct / n
    difference = p_healthy - p_low
    healthy_lo, healthy_hi = wilson_interval(healthy_correct, n, z)
    low_lo, low_hi = wilson_interval(low_correct, n, z)
    phi_denominator = math.sqrt(
        p_healthy * (1.0 - p_healthy) * p_low * (1.0 - p_low)
    )
    phi = ((both_correct / n) - p_healthy * p_low) / phi_denominator \
        if phi_denominator else 0.0
    lower_term = (
        (p_healthy - healthy_lo) ** 2
        + (low_hi - p_low) ** 2
        - 2.0 * phi * (p_healthy - healthy_lo) * (low_hi - p_low)
    )
    upper_term = (
        (healthy_hi - p_healthy) ** 2
        + (p_low - low_lo) ** 2
        - 2.0 * phi * (healthy_hi - p_healthy) * (p_low - low_lo)
    )
    lower = max(-1.0, difference - math.sqrt(max(0.0, lower_term)))
    upper = min(1.0, difference + math.sqrt(max(0.0, upper_term)))
    return difference, lower, upper, phi


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down family-wise-error adjustment."""
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted

