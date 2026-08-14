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

Four gates, three of them free:

    from evalgate import structural, construct, variance, power

    structural.verify(make_case, holds, n=200)      # free
    power.paired(effect=0.20, units=32, reps=5)     # free
    construct.verify(oracle, cases, min_rate=0.90)  # costs model calls
    variance.verify(exp_oracle, cases, truth)       # costs model calls

Which gate catches which failure:

    under_determined    structural
    over_determined     structural
    scorer_disagrees    construct   <-- nothing free reaches this
    no_error_variance   variance    <-- nothing free reaches this

`construct` and `variance` want OPPOSITE answers from DIFFERENT oracles.
construct asks whether the model reaches your answer given full information and
no manipulation; it should score near 1.0. variance asks whether the model ever
gets it wrong WITH the manipulation applied; near 1.0 there means there is
nothing to detect. Passing one tells you nothing about the other.

TEST YOUR OWN GATE
------------------
A validity check is untested code until you run it against an environment that
deserves to fail. All four of ours are shipped as fixtures:

    from evalgate import fixtures
    print(fixtures.audit(my_check))

Zero dependencies. Python 3.10+.
"""

from __future__ import annotations

from . import construct, fixtures, power, structural, variance

__all__ = ["structural", "construct", "variance", "power", "fixtures"]
__version__ = "0.2.0"
