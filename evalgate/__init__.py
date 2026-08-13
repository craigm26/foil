"""evalgate — prove your eval environment can measure what you claim, before
you pay to run it.

Building a multiagent eval is easy. Building one that measures what you think is
not. Across seven environments built for the FOIL project, **four were invalid**
— and every invalid one produced results that looked perfectly analyzable.

    v1   under-determined: an unruled-out option competed with the evidence
    v2   over-determined:  81% of ablations moved the answer by exactly zero
    PID-1 no error variance: the model scored 40/40, nothing to detect
    TURN-1 ground truth the model did not share -- and the analytic gate PASSED

TURN-1 is the cautionary one. Its structural gate ran over 200 seeds, re-derived
from rendered text, and passed. The environment was still invalid: every
"incorrect" run was a *unanimous* vote for the option the reference scorer
ranked last. The scorer was wrong, not the agents. A gate can be rigorous,
reproducible, and confirm the wrong thing.

These three gates are what would have caught all four. Two are free.

    from evalgate import structural, construct, power

    structural.verify(make_case, holds, n=200)     # free
    power.paired(effect=0.20, units=32, reps=5)    # free
    construct.verify(oracle, cases, min_rate=0.90) # costs model calls

Zero dependencies. Python 3.10+.
"""

from __future__ import annotations

from . import construct, power, structural

__all__ = ["structural", "construct", "power"]
__version__ = "0.1.0"
