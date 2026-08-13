"""Distributional statistics for the null experiments.

Total variation is used HERE ONLY, as a noise-floor diagnostic (§5). It is
never a Shapley value function: TV is an unsigned distance between
distributions, not a coalition characteristic function, so a listener that
correctly identifies a liar and inverts its reports would register the same
large value as one that credulously follows it. The Shapley value function is
v(C) = P(correct action | C) (§3.2).
"""

from __future__ import annotations

import numpy as np


def distribution(actions: list[str | None], support: tuple[str, ...]) -> np.ndarray:
    """Categorical distribution over the action set.

    Unparseable samples (None) are EXCLUDED from the distribution but are
    counted by the caller. Silently folding them into a bucket would let a
    drifting parse rate masquerade as a distribution shift.
    """
    counts = np.array([sum(1 for a in actions if a == s) for s in support], dtype=float)
    total = counts.sum()
    if total == 0:
        return np.full(len(support), np.nan)
    return counts / total


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def bootstrap_tv_ci(
    a: list[str | None],
    b: list[str | None],
    support: tuple[str, ...],
    iters: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI for TV between two sample sets.

    Returns (tv, lo, hi). Resamples each side independently with replacement.
    """
    rng = np.random.default_rng(seed)
    aa = [x for x in a if x is not None]
    bb = [x for x in b if x is not None]
    point = total_variation(distribution(aa, support), distribution(bb, support))
    if not aa or not bb:
        return point, float("nan"), float("nan")
    idx_a = rng.integers(0, len(aa), size=(iters, len(aa)))
    idx_b = rng.integers(0, len(bb), size=(iters, len(bb)))
    arr_a = np.array(aa)
    arr_b = np.array(bb)
    tvs = np.empty(iters)
    for i in range(iters):
        tvs[i] = total_variation(
            distribution(list(arr_a[idx_a[i]]), support),
            distribution(list(arr_b[idx_b[i]]), support),
        )
    lo, hi = np.percentile(tvs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile over the full array. Returns None when empty."""
    vs = sorted(v for v in values if v == v)  # drop NaN
    if not vs:
        return None
    rank = max(1, int(np.ceil(p / 100 * len(vs))))
    return vs[min(rank, len(vs)) - 1]
