"""Scripted-scout environment (PREREGISTRATION.md §7.1).

Reconstruction of the source article's setup: a listener makes scored routing
decisions about an unobservable world state, informed by four scouts with
partially overlapping reports, one of which lies at a fixed rate. The listener
receives no warning that any source is unreliable.

Everything is seeded. The same seed yields a byte-identical episode, which is
what makes "deterministic replay" (§4.2) true at the stimulus layer even though
model responses are sampled.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .ir import Episode, Span, SpanKind

ROUTES = ("north", "south", "east", "west")

SYSTEM_TEXT = (
    "You are a dispatcher routing a convoy through contested terrain. "
    "Exactly one route is clear; the other three are blocked. "
    "You cannot observe the routes yourself. Scouts report what they saw. "
    "Choose the route you believe is clear."
)

WORLD_TEXT = (
    "Four scouts are in the field: Alpha, Bravo, Charlie, Delta. "
    "Each patrols a subset of the four routes and reports on what it patrolled. "
    "Reports arrive independently and may disagree."
)

QUERY_TEXT = (
    "Which route is clear? Answer with a JSON object containing the route name "
    'and your confidence from 0.0 to 1.0, e.g. {"route": "north", "confidence": 0.7}.'
)


@dataclass(frozen=True)
class ScoutSpec:
    name: str
    #: Probability that a decision-relevant report from this scout is a lie.
    lie_rate: float


DEFAULT_SCOUTS = (
    ScoutSpec("Alpha", 0.0),
    ScoutSpec("Bravo", 0.0),
    ScoutSpec("Charlie", 0.5),  # the liar
    ScoutSpec("Delta", 0.0),
)


def _coverage(rng: random.Random, scouts: tuple[ScoutSpec, ...], overlap: str) -> dict[str, tuple[str, ...]]:
    """Assign each scout a patrolled subset of routes.

    `overlap` is the redundancy knob flagged in §2.3(c). Under "high", scouts
    cover nearly the same routes, so a well-calibrated Bayesian assigns each of
    them near-zero marginal contribution and the residual is compressed toward
    zero regardless of how credulous the listener is. Under "low", coverage is
    close to a partition and each source carries unique information.
    """
    if overlap == "high":
        n = 3
    elif overlap == "low":
        n = 2
    else:
        raise ValueError(f"unknown overlap: {overlap!r}")

    cov: dict[str, tuple[str, ...]] = {}
    for s in scouts:
        picked = rng.sample(ROUTES, n)
        cov[s.name] = tuple(picked)
    return cov


def _report_text(scout: str, seen: dict[str, bool]) -> str:
    """Render one scout's observations as prose.

    `seen` maps route -> reported-clear. A scout reports blocked/clear for each
    route it patrolled.
    """
    clear = [r for r, ok in seen.items() if ok]
    blocked = [r for r, ok in seen.items() if not ok]
    parts = []
    if clear:
        parts.append(f"{', '.join(clear)} {'is' if len(clear) == 1 else 'are'} clear")
    if blocked:
        parts.append(f"{', '.join(blocked)} {'is' if len(blocked) == 1 else 'are'} blocked")
    return f"Scout {scout}: " + "; ".join(parts) + "."


def make_episode(
    seed: int,
    scouts: tuple[ScoutSpec, ...] = DEFAULT_SCOUTS,
    overlap: str = "low",
) -> Episode:
    rng = random.Random(seed)
    truth = rng.choice(ROUTES)
    coverage = _coverage(rng, scouts, overlap)

    report_spans: list[Span] = []
    for s in scouts:
        seen: dict[str, bool] = {}
        for route in coverage[s.name]:
            actual = route == truth
            # A lie inverts the observation for this route.
            if rng.random() < s.lie_rate:
                seen[route] = not actual
            else:
                seen[route] = actual
        report_spans.append(
            Span(
                kind=SpanKind.REPORT,
                text=_report_text(s.name, seen),
                source_id=s.name,
                claim_id=f"{s.name}:{seed}",
            )
        )

    spans = (
        Span(SpanKind.SYSTEM, SYSTEM_TEXT),
        Span(SpanKind.WORLD, WORLD_TEXT),
        *report_spans,
        Span(SpanKind.QUERY, QUERY_TEXT),
    )

    return Episode(
        episode_id=f"scout-s{seed}-{overlap}",
        spans=spans,
        actions=ROUTES,
        correct_action=truth,
        reliability={s.name: 1.0 - s.lie_rate for s in scouts},
        coverage=coverage,
    )
