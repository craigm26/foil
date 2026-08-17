"""Gate decision rules that know their own resolution.

THE FAILURE THIS FIXES
----------------------
TURN-3's construct gate failed at 32/36 = 0.889 against a threshold of 0.90.
One more correct run would have passed. The Wilson interval [0.747, 0.956]
straddled the threshold, so the verdict was substantially decided by sampling
luck -- and the pre-registration, correctly, forbade arguing with it after the
fact. The place to fix that is BEFORE the run: a gate at n=36 simply cannot
distinguish 0.85 from 0.95, and nobody computed that until the money was spent.

This is the third appearance of the same error class in this project:
TURN-1's MIN_UNTIED was unsatisfiable in expectation, the SWEEP's n=12 made
every comparison indistinguishable, and TURN-3's gate hinged on one run.

THE RULE
--------
A gate is pre-registered as (n, threshold, alpha) with THREE outcomes:

    PASS    observed rate >= threshold
    FAIL    the data are statistically inconsistent with rate >= threshold:
            an exact binomial test rejects H0: p >= threshold at alpha
    EXTEND  neither -- the point estimate is below threshold but the data
            cannot rule out being above it. Run the pre-registered extension
            batch and re-apply the rule to the POOLED counts.

EXTEND is not a loophole. The extension size and the maximum number of
extensions are fixed in the pre-registration alongside n; after the last
extension an unresolved gate resolves to FAIL. What EXTEND removes is the
one-run cliff: a 32/36 stops being a coin-flip verdict and becomes "buy more
information, by a rule written before you knew you would want to".

Under this rule TURN-3's 32/36 is EXTEND (exact binomial p = 0.48 against
p >= 0.90 -- nowhere near rejection), which is what a near-miss should be.

Use `plan()` before pre-registering, so n is chosen for the resolution the
decision needs rather than for how many runs feel like enough.

Zero dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Verdict", "GateDesign", "verdict", "plan", "binom_cdf"]


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p). Exact, stdlib only."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


@dataclass(frozen=True)
class Verdict:
    outcome: str          # "PASS" | "FAIL" | "EXTEND"
    k: int
    n: int
    threshold: float
    alpha: float
    #: exact binomial P(X <= k | p = threshold): small means the data are
    #: inconsistent with the rate actually meeting the threshold
    p_value: float

    @property
    def rate(self) -> float:
        return self.k / self.n if self.n else 0.0

    def __str__(self) -> str:
        return (f"{self.outcome}  {self.k}/{self.n} = {self.rate:.3f} "
                f"vs threshold {self.threshold} "
                f"(exact binomial p = {self.p_value:.3f}, alpha = {self.alpha})")


def verdict(k: int, n: int, threshold: float, alpha: float = 0.05) -> Verdict:
    """Apply the three-outcome rule to observed counts.

    FAIL requires statistical evidence, not a near-miss: the exact binomial
    test must reject H0: p >= threshold. A point estimate below threshold
    without that evidence is EXTEND.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p = binom_cdf(k, n, threshold)
    if k / n >= threshold:
        return Verdict("PASS", k, n, threshold, alpha, p)
    if p < alpha:
        return Verdict("FAIL", k, n, threshold, alpha, p)
    return Verdict("EXTEND", k, n, threshold, alpha, p)


@dataclass(frozen=True)
class GateDesign:
    n: int
    threshold: float
    alpha: float
    #: true rate the design is powered to FAIL
    detectable_rate: float
    #: P(FAIL) when the true rate is `detectable_rate`
    power: float
    #: largest k that still FAILs at this n
    fail_at_or_below: int

    def __str__(self) -> str:
        return (f"GateDesign(n={self.n}, threshold={self.threshold}: "
                f"FAIL at <= {self.fail_at_or_below}/{self.n}; "
                f"power {self.power:.2f} against true rate "
                f"{self.detectable_rate})")


def plan(threshold: float, detectable_rate: float,
         alpha: float = 0.05, power: float = 0.80,
         max_n: int = 2000) -> GateDesign:
    """Smallest n at which a true rate of `detectable_rate` FAILs with the
    requested probability.

    Run this BEFORE pre-registering the gate. If the n it returns is more than
    you will pay for, that is a fact about the decision you are able to make,
    and the pre-registration should say so rather than run an n that cannot
    resolve the question.

        plan(0.90, 0.80)   ->  n where a truly-0.80 environment reliably FAILs
    """
    if not 0.0 < detectable_rate < threshold <= 1.0:
        raise ValueError("need 0 < detectable_rate < threshold <= 1")
    for n in range(5, max_n + 1):
        # largest k with binom_cdf(k; n, threshold) < alpha
        crit = -1
        for k in range(n + 1):
            if binom_cdf(k, n, threshold) < alpha:
                crit = k
            else:
                break
        if crit < 0:
            continue
        achieved = binom_cdf(crit, n, detectable_rate)
        if achieved >= power:
            return GateDesign(n, threshold, alpha, detectable_rate,
                              achieved, crit)
    raise ValueError(f"no n <= {max_n} achieves the requested power")
