"""Gate 2 — construct validity. Does the MODEL share your ground truth?

This is the gate that costs money and the only one that catches the failure
that matters most.

TURN-1 skipped it. Its structural gate passed 200/200 seeds, re-derived from
rendered text. The environment was still invalid: all 21 "incorrect" runs were
UNANIMOUS votes for the option the reference scorer ranked last. The panel was
reasoning defensibly against a scorer that counted facts at equal weight. The
15% "error rate" measured disagreement with the experimenter.

The check: give the model everything needed to reach the intended answer, with
no manipulation applied. If it does not reliably get there, your labels are not
the thing being measured, and no amount of replication fixes that.

TURN-2 ran this gate and scored 36/36 before its main study was allowed to start.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class ConstructReport:
    n: int
    n_intended: int
    min_rate: float
    disagreements: list[tuple[Any, Any, Any]] = field(default_factory=list)
    errors: int = 0

    @property
    def rate(self) -> float:
        return self.n_intended / self.n if self.n else 0.0

    @property
    def ok(self) -> bool:
        return self.n > 0 and self.rate >= self.min_rate

    def __repr__(self) -> str:
        return (f"ConstructReport(rate={self.rate:.3f} of {self.n}, "
                f"required={self.min_rate}, ok={self.ok})")


def verify(
    oracle: Callable[[Any], Any],
    cases: Sequence[Any],
    truth: Callable[[Any], Any],
    *,
    min_rate: float = 0.90,
    reps: int = 3,
    max_workers: int = 4,
    max_examples: int = 10,
) -> ConstructReport:
    """Run `oracle` on each case `reps` times and compare against `truth`.

    oracle -- case -> answer, with FULL information and NO manipulation. This is
              the best case your design can produce. If it cannot reach the
              intended answer, nothing downstream can.
    min_rate -- required agreement. Below it, the environment is rejected and
              the main study should not run. Enforce that in code: a gate you
              have to remember to honour is the gate TURN-1 had.
    """
    jobs = [(c, r) for c in cases for r in range(reps)]

    def run(job):
        case, _ = job
        try:
            return case, oracle(case), None
        except Exception as e:
            return case, None, e

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        got = list(pool.map(run, jobs))

    n = intended = errors = 0
    bad: list[tuple[Any, Any, Any]] = []
    for case, got_answer, err in got:
        if err is not None:
            errors += 1
            continue
        n += 1
        want = truth(case)
        if got_answer == want:
            intended += 1
        elif len(bad) < max_examples:
            bad.append((case, want, got_answer))
    return ConstructReport(n, intended, min_rate, bad, errors)
