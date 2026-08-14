"""Gate 3 — error variance. Does your environment produce anything to detect?

A detector study needs errors. If the model is right every time under the
condition you are actually studying, there is no signal, and no amount of
replication or statistical care creates one.

This is the gate FOIL PID-1 needed and did not have. The environment was
structurally sound, the model shared the ground truth, and the model scored
40 out of 40. The hypothesis -- that instability predicts error -- could not be
tested at all, because there were no errors. The run was paid for in full before
anyone noticed.

RUN THIS ON THE EXPERIMENTAL CONDITION, NOT THE CONTROL
-------------------------------------------------------
`construct.verify` asks whether the model reaches the intended answer given FULL
information and NO manipulation. It should score near 1.0; a low score there
means your labels are wrong.

This gate asks the opposite question about the condition you are studying: with
the manipulation applied, does the model ever get it wrong? A score near 1.0
HERE means there is nothing to measure.

The two gates want opposite answers from different oracles. Passing one tells
you nothing about the other.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class VarianceReport:
    n: int
    n_wrong: int
    min_error_rate: float
    errors: int = 0
    examples: list[Any] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return self.n_wrong / self.n if self.n else 0.0

    @property
    def at_ceiling(self) -> bool:
        """The model never missed. Nothing to detect."""
        return self.n > 0 and self.n_wrong == 0

    @property
    def at_floor(self) -> bool:
        """The model never succeeded. Usually a broken harness, not a hard task."""
        return self.n > 0 and self.n_wrong == self.n

    @property
    def ok(self) -> bool:
        return (self.n > 0
                and self.error_rate >= self.min_error_rate
                and not self.at_floor)

    def __repr__(self) -> str:
        return (f"VarianceReport(error_rate={self.error_rate:.3f} of {self.n}, "
                f"required>={self.min_error_rate}, ok={self.ok})")

    def explain(self) -> str:
        if self.at_ceiling:
            return ("REJECT: the model was correct on every case. There are no "
                    "errors for a detector to detect, so the study cannot test "
                    "its hypothesis however many samples you buy. This is the "
                    "PID-1 failure, and it cost a full run to discover.")
        if self.at_floor:
            return ("REJECT: the model was wrong on every case. That is almost "
                    "always a broken scorer or a mis-specified prompt rather "
                    "than a hard task. Check construct validity first.")
        if not self.ok:
            return (f"REJECT: error rate {self.error_rate:.3f} is below the "
                    f"required {self.min_error_rate}. There may be too little "
                    "signal to power the comparison you intend.")
        return (f"OK: error rate {self.error_rate:.3f} on {self.n} cases. "
                "There is variance to detect.")


def verify(
    oracle: Callable[[Any], Any],
    cases: Sequence[Any],
    truth: Callable[[Any], Any],
    *,
    min_error_rate: float = 0.05,
    reps: int = 1,
    max_workers: int = 4,
    max_examples: int = 10,
) -> VarianceReport:
    """Measure the error rate under the condition you intend to study.

    oracle -- case -> answer, WITH the manipulation applied. Not the
              full-information control; that is `construct.verify`.
    min_error_rate -- below this, the environment is rejected as having too
              little signal. The default 0.05 is a floor, not a target: a rate
              that low still needs a large sample to detect anything.
    """
    jobs = [(c, r) for c in cases for r in range(reps)]
    n = n_wrong = errs = 0
    examples: list[Any] = []

    def run(job):
        case, _ = job
        return case, oracle(case)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for case, got in pool.map(run, jobs):
            try:
                want = truth(case)
            except Exception:
                errs += 1
                continue
            n += 1
            if got != want:
                n_wrong += 1
                if len(examples) < max_examples:
                    examples.append((case, got, want))
    return VarianceReport(n, n_wrong, min_error_rate, errs, examples)
