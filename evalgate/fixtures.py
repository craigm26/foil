"""Known-invalid eval environments, so you can test your validity check.

A validity gate is itself untested code. You write one, it passes, and you have
learned nothing unless you know it would have failed on an environment that
deserved to fail. Almost nobody publishes their invalid environments, so there
is nothing to test a gate against.

These are ours. Four environments built in earnest for the FOIL project, each of
which produced results that looked perfectly analyzable, and each of which was
invalid:

    under_determined     a route nobody attested competes with the answer
    over_determined      no single source mattered; ablation moved nothing
    no_error_variance    the model scored 100%, leaving nothing to detect
    scorer_disagrees     ground truth the model did not share

plus three valid controls, so a check that rejects everything scores zero:

    decisive             >= 3 individually decisive sources  (Phase 0 v3)
    two_candidate        forced choice between two live options  (TURN-2)
    calibrated_variance  answerable, but with real error variance  (PID-2)

Usage:

    from evalgate import fixtures

    print(fixtures.audit(my_check))     # my_check(make_case) -> bool

THE POINT OF `scorer_disagrees`
-------------------------------
Two of the four are unreachable by any check that does not call a model, and
`audit` reports them separately rather than counting them against you.

`scorer_disagrees` is the one worth studying. It satisfies every structural
property the valid fixtures satisfy -- unique reference answer, full coverage,
decisive sources -- over any number of seeds, re-derived from rendered text. It
is invalid anyway, because the scorer and the model disagree about the answer.
In the original run every episode marked incorrect was a *unanimous* four-agent
vote for the option the scorer ranked last. The agents were right.

If your free check flags it, your check is wrong about something else, because
nothing in the generated text distinguishes it from a valid environment. That
is the finding: a gate can be rigorous, reproducible, and confirm the wrong
thing. Only `construct.verify` against a real model reaches it.

Zero dependencies. No model calls. Deterministic given a seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

ROUTES = ("north", "south", "east", "west")
SOURCES = ("Alpha", "Bravo", "Charlie", "Delta")

__all__ = [
    "Fixture", "AuditReport", "Case", "FIXTURES",
    "get", "names", "audit",
    "reference_argmax", "decisive_sources", "attested_routes",
    "default_structural_check", "simulated_oracle",
    "simulated_experimental_oracle",
]


# ── the reference reasoner ───────────────────────────────────────────────


def reference_argmax(claims: dict[str, dict[str, bool]]) -> tuple[str | None, bool]:
    """Tally clear/blocked votes per route. Returns (best_route, ambiguous)."""
    score = {r: 0 for r in ROUTES}
    for reports in claims.values():
        for route, clear in reports.items():
            score[route] += 1 if clear else -1
    best = max(score.values())
    winners = [r for r in ROUTES if score[r] == best]
    return (winners[0] if len(winners) == 1 else None, len(winners) != 1)


def decisive_sources(claims: dict[str, dict[str, bool]]) -> list[str]:
    """Sources whose removal changes the reference conclusion."""
    full = reference_argmax(claims)
    return [s for s in claims
            if reference_argmax({k: v for k, v in claims.items() if k != s}) != full]


def attested_routes(claims: dict[str, dict[str, bool]]) -> set[str]:
    """Routes some source actually reported on. Anything missing is a rival
    hypothesis the evidence never rules out -- the v1 failure."""
    return {r for reports in claims.values() for r in reports}


# ── types ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    """One generated episode.

    A checker should reason over `claims` and `text`. `intended` is what the
    environment's own scorer calls correct.

    `_model_answer` is what a competent model actually concludes. It is NOT
    visible in the rendered text and no structural check may read it -- it
    exists so `simulated_oracle` can stand in for a real model call in tests
    and in the doctest below. On every fixture but `scorer_disagrees` it equals
    `intended`.
    """

    seed: int
    claims: dict[str, dict[str, bool]]
    intended: str
    text: str = ""
    #: what a competent model concludes given FULL information, no manipulation.
    #: Equals `intended` everywhere except `scorer_disagrees`.
    _model_answer: str = ""
    #: what it concludes WITH the manipulation applied. Equals `_model_answer`
    #: everywhere except `calibrated_variance`, which has real error variance.
    _experimental_answer: str = ""


@dataclass(frozen=True)
class Fixture:
    name: str
    valid: bool
    catchable_by: str | None      # "structural" | "construct" | "power" | None
    why: str
    origin: str
    make_case: Callable[[int], Case]

    @property
    def free_to_catch(self) -> bool:
        return self.catchable_by in ("structural", "power")


def _render(claims: dict[str, dict[str, bool]]) -> str:
    return "\n".join(
        f"{src} reports {route} is {'clear' if claims[src][route] else 'blocked'}."
        for src in sorted(claims) for route in ROUTES if route in claims[src]
    )


def _case(seed, claims, intended, model_answer=None, experimental=None) -> Case:
    m = model_answer if model_answer is not None else intended
    return Case(seed, claims, intended, _render(claims), m,
                experimental if experimental is not None else m)


def _structurally_sound(claims: dict[str, dict[str, bool]]) -> bool:
    """The three properties every VALID fixture here satisfies, and which
    `scorer_disagrees` also satisfies. This is what a free check can see."""
    arg, amb = reference_argmax(claims)
    return (arg is not None and not amb
            and attested_routes(claims) == set(ROUTES)
            and len(decisive_sources(claims)) >= 1)


def _sound_claims(rng: random.Random) -> tuple[dict[str, dict[str, bool]], str]:
    """Rejection-sample a structurally sound episode."""
    for _ in range(4000):
        intended = rng.choice(ROUTES)
        claims = {}
        for src in SOURCES:
            picks = rng.sample(ROUTES, rng.choice((2, 3)))
            claims[src] = {r: (r == intended) for r in picks}
        arg, _amb = reference_argmax(claims)
        if arg == intended and _structurally_sound(claims):
            return claims, intended
    raise RuntimeError("could not sample a sound episode")


# ── the four invalid environments ────────────────────────────────────────


def _make_under_determined(seed: int) -> Case:
    """v1. A route nobody reported on is never ruled out, so an
    'unruled-out therefore clear' reading competes with the evidence.
    A quarter of episodes went bistable; T_null exceeded the kill threshold 36x.
    """
    rng = random.Random(seed)
    intended = rng.choice(ROUTES)
    unreported = rng.choice([r for r in ROUTES if r != intended])
    covered = [r for r in ROUTES if r != unreported]
    claims = {src: {r: (r == intended) for r in rng.sample(covered, 2)}
              for src in SOURCES}
    return _case(seed, claims, intended)


def _make_over_determined(seed: int) -> Case:
    """v2. Every source carries the whole answer, so removing any one changes
    nothing. 81% of single-source ablations moved the distribution by exactly
    zero and the comparison denominator collapsed."""
    rng = random.Random(seed)
    intended = rng.choice(ROUTES)
    claims = {src: {r: (r == intended) for r in ROUTES} for src in SOURCES}
    return _case(seed, claims, intended)


def _make_no_error_variance(seed: int) -> Case:
    """PID-1. Structurally sound and far too easy: the model scored 40/40.

    A detector needs errors to detect. Nothing in the structure reveals this --
    only calling a model and finding no spread does."""
    rng = random.Random(seed)
    claims, intended = _sound_claims(rng)
    return _case(seed, claims, intended)


def _make_scorer_disagrees(seed: int) -> Case:
    """TURN-1. Structurally indistinguishable from a valid environment.

    `intended` is what the scorer declares correct, and it is self-consistent
    with the reference reasoner, so every free check passes. A real model
    concludes something else -- unanimously, in the original run.
    """
    rng = random.Random(seed)
    claims, intended = _sound_claims(rng)
    # what a competent model actually concludes; invisible to the text
    model_says = rng.choice([r for r in ROUTES if r != intended])
    return _case(seed, claims, intended, model_answer=model_says)


# ── valid controls ───────────────────────────────────────────────────────


def _make_decisive(seed: int) -> Case:
    """v3. Multiple individually decisive sources, verified analytically before
    any model call. Ablation moves the answer by a median of 0.320.

    Constructed rather than rejection-sampled, on a deliberately thin margin:
    the answer leads by one vote, so removing either supporter makes the result
    ambiguous. Under this module's tally reasoner that yields 2 of 4 decisive.
    (FOIL v3 required >= 3 of 4 under its own reference reasoner; the threshold
    is reasoner-specific, the property is not.)
    """
    rng = random.Random(seed)
    intended = rng.choice(ROUTES)
    decoy = rng.choice([r for r in ROUTES if r != intended])
    rest = [r for r in ROUTES if r not in (intended, decoy)]
    claims = {
        "Alpha":   {intended: True},
        "Bravo":   {intended: True},
        "Charlie": {decoy: True},
        "Delta":   {rest[0]: False, rest[1]: False},
    }
    # v3 measured six orderings; one of the six inverted the answer.
    inverted = rng.random() < 1 / 6
    return _case(seed, claims, intended,
                 experimental=decoy if inverted else intended)


def _make_two_candidate(seed: int) -> Case:
    """TURN-2. Forced choice between two live options; the other two are
    attested as blocked. Its control panel reached the intended answer 36/36."""
    rng = random.Random(seed)
    a, b = rng.sample(ROUTES, 2)
    dead = [r for r in ROUTES if r not in (a, b)]
    claims = {
        "Alpha":   {a: True, dead[0]: False},
        "Bravo":   {a: True, dead[1]: False},
        "Charlie": {b: True},                      # the dissenter
        "Delta":   {dead[0]: False, dead[1]: False},
    }
    # TURN-2 measured 0.906 group accuracy with the holder speaking first.
    wrong = rng.random() < 1 - 0.906
    return _case(seed, claims, a, experimental=b if wrong else a)


def _make_calibrated_variance(seed: int) -> Case:
    """PID-2. Decisive, non-trivial, with real error variance -- the
    environment the one supported result was measured on."""
    rng = random.Random(seed)
    for _ in range(4000):
        intended = rng.choice(ROUTES)
        decoy = rng.choice([r for r in ROUTES if r != intended])
        rest = [r for r in ROUTES if r not in (intended, decoy)]
        claims = {
            "Alpha":   {intended: True,  decoy: False, rest[0]: False},
            "Bravo":   {intended: True,  rest[1]: False},
            "Charlie": {decoy: True,     intended: False},        # the liar
            "Delta":   {decoy: False,    rest[0]: False, rest[1]: False},
        }
        if _structurally_sound(claims) and reference_argmax(claims)[0] == intended:
            # PID-2 measured 13 errors in 48 episodes under the manipulation.
            # Reproduce that rate so the variance gate has something to accept.
            wrong = rng.random() < 13 / 48
            return _case(seed, claims, intended,
                         experimental=decoy if wrong else intended)
    raise RuntimeError("calibrated_variance: no admissible episode for this seed")


FIXTURES: tuple[Fixture, ...] = (
    Fixture("under_determined", False, "structural",
            "A route nobody attested is never ruled out, so it competes with "
            "the intended answer.",
            "FOIL Phase 0 v1 -- KILL, T_null 36x the threshold",
            _make_under_determined),
    Fixture("over_determined", False, "structural",
            "Every source carries the full answer, so no single source is "
            "decisive and ablation has nothing to move.",
            "FOIL Phase 0 v2 -- INDETERMINATE, 81% of ablations moved zero",
            _make_over_determined),
    Fixture("no_error_variance", False, "construct",
            "Structurally sound and far too easy. The model scored 40/40, "
            "leaving no errors for the detector to detect.",
            "FOIL PID-1 -- DEGENERATE",
            _make_no_error_variance),
    Fixture("scorer_disagrees", False, "construct",
            "The scorer's answer is self-consistent but not the one a real "
            "model reaches. Structurally indistinguishable from a valid "
            "environment, which is exactly why it is dangerous.",
            "FOIL TURN-1 -- DEGENERATE, structural gate passed over 200 seeds",
            _make_scorer_disagrees),
    Fixture("decisive", True, None,
            ">= 3 of 4 sources individually decisive, verified analytically.",
            "FOIL Phase 0 v3 -- valid",
            _make_decisive),
    Fixture("two_candidate", True, None,
            "Forced choice between two live options; control panel 36/36.",
            "FOIL TURN-2 -- valid",
            _make_two_candidate),
    Fixture("calibrated_variance", True, None,
            "Decisive, non-trivial, with real error variance.",
            "FOIL PID-2 -- valid, LR 8.1 at p = 0.00003",
            _make_calibrated_variance),
)


def names() -> list[str]:
    return [f.name for f in FIXTURES]


def get(name: str) -> Fixture:
    for f in FIXTURES:
        if f.name == name:
            return f
    raise KeyError(f"no fixture {name!r}; have {names()}")


# ── the audit ────────────────────────────────────────────────────────────


@dataclass
class AuditReport:
    caught: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    false_alarms: list[str] = field(default_factory=list)
    passed_valid: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    unreachable_free: list[str] = field(default_factory=list)

    @property
    def reachable_missed(self) -> list[str]:
        return [m for m in self.missed if m not in self.unreachable_free]

    @property
    def n_free_catchable(self) -> int:
        return sum(1 for f in FIXTURES if not f.valid and f.free_to_catch)

    @property
    def ok(self) -> bool:
        return not self.reachable_missed and not self.false_alarms and not self.errors

    def __str__(self) -> str:
        L = [f"caught       {len(self.caught)}/{self.n_free_catchable} "
             f"free-catchable   {self.caught or '[]'}"]
        if self.reachable_missed:
            L.append(f"MISSED       {self.reachable_missed}"
                     "   <-- a free check should catch these")
        if self.false_alarms:
            L.append(f"FALSE ALARM  {self.false_alarms}"
                     "   <-- these environments are valid")
        if self.errors:
            L.append(f"raised       {self.errors}")
        if self.unreachable_free:
            L.append("")
            L.append(f"not reachable without model calls: {self.unreachable_free}")
            L.append("  scorer_disagrees is structurally identical to a valid")
            L.append("  environment. If your check flagged it, that is luck or a")
            L.append("  bug, not detection. Use construct.verify with a real model.")
        L.append("")
        L.append(f"verdict      {'OK' if self.ok else 'INCOMPLETE'}")
        return "\n".join(L)


def audit(check: Callable[[Callable[[int], Case]], bool], n: int = 200) -> AuditReport:
    """Run your validity check against every fixture.

    check -- takes a `make_case(seed) -> Case` generator, returns True if it
             considers that environment VALID. It must not read `_model_answer`.
    """
    r = AuditReport()
    for f in FIXTURES:
        if not f.valid and not f.free_to_catch:
            r.unreachable_free.append(f.name)
        try:
            said_valid = bool(check(f.make_case))
        except Exception as e:
            r.errors.append((f.name, repr(e)))
            continue
        if f.valid:
            (r.passed_valid if said_valid else r.false_alarms).append(f.name)
        else:
            (r.missed if said_valid else r.caught).append(f.name)
    return r


def default_structural_check(make_case: Callable[[int], Case], n: int = 200) -> bool:
    """A reference free check: unique reference answer, full coverage, and at
    least one decisive source. Rejects both structural failures and passes all
    three valid controls -- and passes `scorer_disagrees`, as any free check
    must."""
    from . import structural

    return structural.verify(
        make_case, lambda c: _structurally_sound(c.claims), n=n
    ).ok


def simulated_oracle(case: Case) -> str:
    """Full-information stand-in for a real model, for `construct.verify`.

    No manipulation applied: this is the best case the design can produce.
    Replace with a real model call to reach the construct fixtures for real.
    """
    return case._model_answer


def simulated_experimental_oracle(case: Case) -> str:
    """Stand-in for a real model WITH the manipulation applied, for
    `variance.verify`.

    Differs from `simulated_oracle` only on `calibrated_variance`, which
    reproduces PID-2's measured 13-in-48 error rate. `no_error_variance` returns
    the intended answer every time, which is precisely why it is invalid.
    """
    return case._experimental_answer
