"""Gate 1 — structural. Does the generator produce instances with the property
the experiment depends on? Free: no model calls.

Catches the FOIL v1 and v2 failures, where the environment was under- and then
over-determined and neither was noticed until the results were meaningless.

IMPORTANT: passing is necessary and NOT sufficient. TURN-1 passed this gate over
200 seeds and was still invalid, because the property it verified was defined by
a scorer nobody had checked against the model. Run `construct.verify` too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StructuralReport:
    n_requested: int
    n_generated: int
    n_holding: int
    failures: list[tuple[int, str]] = field(default_factory=list)

    @property
    def admission_rate(self) -> float:
        return self.n_generated / self.n_requested if self.n_requested else 0.0

    @property
    def hold_rate(self) -> float:
        return self.n_holding / self.n_generated if self.n_generated else 0.0

    @property
    def ok(self) -> bool:
        return self.n_generated > 0 and self.n_holding == self.n_generated

    def __repr__(self) -> str:
        return (f"StructuralReport(generated={self.n_generated}/{self.n_requested}, "
                f"holding={self.n_holding}, ok={self.ok})")


def verify(
    make_case: Callable[[int], Any | None],
    holds: Callable[[Any], bool],
    n: int = 200,
    seed_base: int = 0,
    max_failures: int = 10,
) -> StructuralReport:
    """Generate `n` cases and check `holds` on each.

    make_case -- seed -> case, or None if that seed yields nothing admissible.
    holds     -- case -> bool. Re-derive the property from the case's RENDERED
                 content wherever you can, not from fields the generator set.
                 A generator that records `truth="B"` and a checker that reads
                 `case.truth` verify nothing; they agree by construction.
    """
    generated = holding = 0
    failures: list[tuple[int, str]] = []
    for i in range(n):
        try:
            case = make_case(seed_base + i)
        except Exception as e:
            if len(failures) < max_failures:
                failures.append((seed_base + i, f"generator raised: {e!r}"))
            continue
        if case is None:
            continue
        generated += 1
        try:
            good = bool(holds(case))
        except Exception as e:
            good = False
            if len(failures) < max_failures:
                failures.append((seed_base + i, f"predicate raised: {e!r}"))
        if good:
            holding += 1
        elif len(failures) < max_failures:
            failures.append((seed_base + i, "property does not hold"))
    return StructuralReport(n, generated, holding, failures)
