"""Gate 3 — power. Can the design detect the effect you are looking for? Free.

Catches the TURN-1 failure: 24 units x 3 replicates, analysed with a sign test.
At the observed accuracy that design had **40% power**, and its minimum-untied
criterion was unsatisfiable *in expectation* -- P(tie) was 0.49, so ~12 units
would be untied against a required 16. The experiment could not have passed its
own gate. Two lines of arithmetic, available before spending anything.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class PowerReport:
    power: float
    false_positive_rate: float
    tie_rate: float
    units: int
    reps: int
    effect: float

    @property
    def ok(self) -> bool:
        return self.power >= 0.80

    def __repr__(self) -> str:
        return (f"PowerReport(power={self.power:.2f}, fpr={self.false_positive_rate:.3f}, "
                f"tie_rate={self.tie_rate:.2f}, units={self.units}, reps={self.reps})")


def tie_rate(base: float, reps: int) -> float:
    """P(two conditions give identical per-unit rates) under the null.

    This is the number that killed TURN-1 and the cheapest thing to check: with
    a binary outcome and few replicates, most units tie, and a sign test throws
    every tied unit away.
    """
    p = [math.comb(reps, k) * base**k * (1 - base) ** (reps - k) for k in range(reps + 1)]
    return sum(x * x for x in p)


def _perm_p(deltas: list[float], iters: int, rng: random.Random) -> float:
    obs = sum(deltas) / len(deltas)
    ge = sum(1 for _ in range(iters)
             if sum(d if rng.random() < 0.5 else -d for d in deltas) / len(deltas) >= obs)
    return (ge + 1) / (iters + 1)


def paired(
    effect: float = 0.20,
    units: int = 32,
    reps: int = 5,
    base: float = 0.70,
    spread: float = 0.12,
    alpha: float = 0.01,
    trials: int = 300,
    iters: int = 4000,
    seed: int = 0,
) -> PowerReport:
    """Simulate a paired two-condition design analysed by permutation test.

    Simulates the ACTUAL pipeline -- per-unit difficulty variation, binomial
    replicate sampling, and the test you will run -- rather than applying a
    closed-form formula to a design that does not match its assumptions.

    effect -- the true difference in rate you want to be able to detect.
    base   -- expected outcome rate. Measure it with a pilot; do not guess.
    spread -- SD of per-unit difficulty around `base`.
    """
    rng = random.Random(seed)

    def once(delta: float) -> float:
        ds = []
        for _ in range(units):
            p = min(0.97, max(0.03, rng.gauss(base, spread)))
            a = min(1.0, p + delta / 2)
            b = max(0.0, p - delta / 2)
            fa = sum(rng.random() < a for _ in range(reps)) / reps
            fb = sum(rng.random() < b for _ in range(reps)) / reps
            ds.append(fa - fb)
        return _perm_p(ds, iters, rng)

    hits = sum(1 for _ in range(trials) if once(effect) < alpha)
    fp = sum(1 for _ in range(trials) if once(0.0) < alpha)
    return PowerReport(hits / trials, fp / trials, tie_rate(base, reps),
                       units, reps, effect)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate. The canonical copy.

    sweep_run.py imports this. Do not reimplement it elsewhere -- the price
    table drifted twice because two copies of one fact existed.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _z(q: float) -> float:
    """Standard normal quantile (Acklam's approximation, ~1e-9 accurate)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        u = math.sqrt(-2 * math.log(q))
        return (((((c[0]*u+c[1])*u+c[2])*u+c[3])*u+c[4])*u+c[5]) / \
               ((((d[0]*u+d[1])*u+d[2])*u+d[3])*u+1)
    if q > phigh:
        u = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*u+c[1])*u+c[2])*u+c[3])*u+c[4])*u+c[5]) / \
               ((((d[0]*u+d[1])*u+d[2])*u+d[3])*u+1)
    u = q - 0.5
    t = u * u
    return (((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4])*t+a[5])*u / \
           (((((b[0]*t+b[1])*t+b[2])*t+b[3])*t+b[4])*t+1)


def separate(p1: float, p2: float, alpha: float = 0.05,
             power: float = 0.80) -> int:
    """Units PER GROUP needed for a two-sided two-proportion test to separate
    two rates -- the resolution planner the SWEEP should have run first.

    The SWEEP measured bistability on 12 episodes per model and found every
    pair of models indistinguishable, because at n=12 the intervals are ~0.25
    wide. This function would have said so for free:

        separate(0.42, 0.58)  ->  ~150 episodes per model

    That is the study you have to be willing to buy. If you are not, the
    honest pre-registration says the design cannot rank models, before the
    run rather than after it.
    """
    if not (0 < p1 < 1 and 0 < p2 < 1) or p1 == p2:
        raise ValueError("need distinct rates strictly inside (0, 1)")
    za, zb = _z(1 - alpha / 2), _z(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar))
           + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)
