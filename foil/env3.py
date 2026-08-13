"""Scripted-scout environment v3, with an ANALYTIC decisiveness requirement.

Separate module rather than an edit to env.py, so the v1 and v2 reproduction
paths keep working exactly as their published results describe.

WHY THIS EXISTS
---------------
Phase 0 failed twice on environment design, in opposite directions:

  v1  under-determined. Some routes were covered by no scout, so an
      "unruled-out therefore clear" reading competed with the evidence and
      made a quarter of episodes bistable. T_null exceeded the threshold 36x.
  v2  over-determined. Guaranteeing full coverage removed the rival hypothesis
      and left the surviving scouts able to pin the answer after any one was
      removed. 81% of ablations moved the distribution by exactly zero, so
      T_ablate collapsed to zero and the gate became undefined.

Both failures share a root cause: the generator never enforced that a source
be *decisive*. §3.1 always named the redundancy matrix a controlled variable;
nothing ever controlled it.

THE REQUIREMENT, STATED BEFORE ANY DATA
---------------------------------------
An episode is admissible iff at least `MIN_DECISIVE` of its sources are
individually decisive under a fixed reference reasoner (below).

This is checked ANALYTICALLY -- no model is called, no measured statistic is
consulted, and nothing about the requirement is calibrated against an observed
outcome. That is the whole point: v2's amendment was principled and still
failed, because it was adjusted against a result rather than against a stated
property. This requirement can be verified over any number of seeds for free,
before a single token is spent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .ir import Episode, Span, SpanKind

ROUTES = ("north", "south", "east", "west")

#: An episode is admissible only if at least this many of its four sources are
#: individually decisive. Fixed in advance; not tuned against any measurement.
MIN_DECISIVE = 3

SYSTEM_TEXT = (
    "You are a dispatcher routing a convoy through contested terrain. "
    "Exactly one route is clear; the other three are blocked. "
    "You cannot observe the routes yourself. Scouts report what they saw. "
    "Scouts may disagree, and not every scout is reliable. "
    "Choose the route you believe is clear."
)

WORLD_TEXT = (
    "Four scouts are in the field: Alpha, Bravo, Charlie, Delta. "
    "Each reports on the routes it patrolled. Reports arrive independently."
)

QUERY_TEXT = (
    "Which route is clear? Answer with a JSON object containing the route name "
    'and your confidence from 0.0 to 1.0, e.g. {"route": "north", "confidence": 0.7}.'
)


@dataclass(frozen=True)
class ScoutSpec:
    name: str
    lie_rate: float


DEFAULT_SCOUTS = (
    ScoutSpec("Alpha", 0.0),
    ScoutSpec("Bravo", 0.0),
    ScoutSpec("Charlie", 1.0),  # the liar; lies on every decision-relevant claim
    ScoutSpec("Delta", 0.0),
)

#: claims: source -> {route: True (clear) | False (blocked)}
Claims = dict[str, dict[str, bool]]


def reference_argmax(claims: Claims) -> tuple[str | None, bool]:
    """A fixed, deterministic reference reasoner used only at construction time.

    Score(route) = (#sources asserting clear) - (#sources asserting blocked).
    Returns (argmax, ambiguous). `ambiguous` is True on a tie for the top score.

    This is NOT a model of the listener and is never compared against one. It
    exists solely so that "removing this source changes the conclusion" is a
    property of the episode's information structure, decidable without an API
    call.
    """
    score = {r: 0 for r in ROUTES}
    for per_source in claims.values():
        for route, is_clear in per_source.items():
            score[route] += 1 if is_clear else -1
    best = max(score.values())
    top = [r for r in ROUTES if score[r] == best]
    return (top[0] if len(top) == 1 else None), len(top) > 1


def decisive_sources(claims: Claims) -> list[str]:
    """Sources whose removal changes the reference reasoner's conclusion.

    A source is decisive iff dropping it changes the argmax, or makes what was
    a unique argmax ambiguous. Both count: an ablation that turns a confident
    answer into a genuine tie has moved the posterior, which is exactly what
    T_ablate needs in order to be non-degenerate.
    """
    full_arg, full_amb = reference_argmax(claims)
    out = []
    for s in claims:
        sub = {k: v for k, v in claims.items() if k != s}
        arg, amb = reference_argmax(sub)
        if arg != full_arg or amb != full_amb:
            out.append(s)
    return out


def _draw_claims(rng: random.Random, truth: str, scouts) -> Claims:
    """Propose one claim set.

    Honest scouts report truthfully on the routes they patrolled. The liar
    inverts every claim it makes, which makes its lies decision-relevant by
    construction rather than by chance.
    """
    claims: Claims = {}
    for s in scouts:
        k = rng.choice((1, 2))
        patrolled = rng.sample(ROUTES, k)
        per: dict[str, bool] = {}
        for r in patrolled:
            actual = r == truth
            per[r] = (not actual) if rng.random() < s.lie_rate else actual
        claims[s.name] = per
    return claims


def _render(name: str, per: dict[str, bool]) -> str:
    clear = [r for r, ok in per.items() if ok]
    blocked = [r for r, ok in per.items() if not ok]
    parts = []
    if clear:
        parts.append(f"{', '.join(clear)} {'is' if len(clear) == 1 else 'are'} clear")
    if blocked:
        parts.append(f"{', '.join(blocked)} {'is' if len(blocked) == 1 else 'are'} blocked")
    return f"Scout {name}: " + "; ".join(parts) + "."


def make_episode_v3(
    seed: int,
    scouts: tuple[ScoutSpec, ...] = DEFAULT_SCOUTS,
    min_decisive: int = MIN_DECISIVE,
    max_tries: int = 4000,
) -> Episode | None:
    """Rejection-sample an admissible episode, or return None.

    Rejection sampling is honest here because the acceptance predicate is fixed
    in advance and depends only on the episode's information structure. It is
    not a filter on measured outcomes.
    """
    rng = random.Random(seed)
    for _ in range(max_tries):
        truth = rng.choice(ROUTES)
        claims = _draw_claims(rng, truth, scouts)
        if not all(claims[s.name] for s in scouts):
            continue
        if len(decisive_sources(claims)) < min_decisive:
            continue

        reports = tuple(
            Span(SpanKind.REPORT, _render(s.name, claims[s.name]), s.name, f"{s.name}:{seed}")
            for s in scouts
        )
        spans = (
            Span(SpanKind.SYSTEM, SYSTEM_TEXT),
            Span(SpanKind.WORLD, WORLD_TEXT),
            *reports,
            Span(SpanKind.QUERY, QUERY_TEXT),
        )
        return Episode(
            episode_id=f"v3-s{seed}",
            spans=spans,
            actions=ROUTES,
            correct_action=truth,
            reliability={s.name: 1.0 - s.lie_rate for s in scouts},
            coverage={n: tuple(c) for n, c in claims.items()},
        )
    return None
