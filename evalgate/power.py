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
