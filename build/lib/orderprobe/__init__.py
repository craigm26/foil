"""orderprobe — detect when a decision depends on the order of its inputs.

Permute the context, re-run, see whether the answer moves. An answer that
changes under reordering is substantially more likely to be wrong.

    from orderprobe import probe

    @probe(k=6, samples=5)
    def decide(items: list[str]) -> str:
        return call_your_model(items)

    r = decide(tool_results)
    if r.unstable:
        escalate(r)

WHAT THIS TAKES: a callable. Not a model client, not a provider SDK, not a
config. `orderprobe` never learns how you call a model, which is what lets it
drop into any harness without a port. Zero dependencies, stdlib only.

EVIDENCE. Pre-registered test on `claude-opus-5`, 48 episodes, disjoint from
the data the effect was noticed on:

    P(wrong | unstable) = 0.714      P(wrong | stable) = 0.088
    likelihood ratio 8.1, Fisher one-sided p = 0.00003

Robust at every instability cutoff from 0.1 to 0.9.

LIMITS, which matter as much as the evidence:

  * Validated on SYNTHETIC tasks on two models. Real agent traffic is untested.
  * NOT CALIBRATED. Those rates are point estimates on 48 episodes in one
    environment. No production threshold is licensed by them.
  * STABILITY IS NOT A GUARANTEE. In the same run, 3 of 34 stable episodes were
    wrong -- two of them perfectly stable across all six orderings and
    confidently wrong. Treating stability as proof of correctness is wrong about
    one case in eleven.
  * Costs k x samples calls per decision.

An earlier exploratory reading of this effect claimed no false negatives. The
pre-registered test refuted that half of it. Do not build a guarantee on this.

Full method and every negative result: https://foil-9vg.pages.dev
"""

from __future__ import annotations

import functools
import itertools
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Sequence

__all__ = ["probe", "probe_call", "ProbeResult", "Verdict"]

__version__ = "0.1.0"


class Verdict:
    STABLE = "stable"
    UNSTABLE = "unstable"
    #: Fewer than two distinct orderings were possible (0 or 1 items), so the
    #: probe could not run. Reported rather than silently returning "stable",
    #: because "we could not check" is not the same claim as "it is fine".
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ProbeResult:
    """Outcome of an order-sensitivity probe."""

    value: Any
    """The answer at the order you actually passed in -- the one you would have
    shipped without probing."""

    verdict: str
    dispersion: float
    """Maximum pairwise total variation between the answer distributions of any
    two orderings. 0.0 = every ordering agreed exactly; 1.0 = two orderings
    shared no answer at all."""

    by_ordering: list[tuple[tuple[int, ...], Counter]] = field(default_factory=list)
    """(permutation of input indices, distribution of answers) per ordering.
    The first entry is always the order you passed in."""

    calls: int = 0
    errors: int = 0
    """Calls that raised. Their orderings are excluded from the statistics and
    counted here rather than being silently dropped."""

    @property
    def unstable(self) -> bool:
        return self.verdict == Verdict.UNSTABLE

    @property
    def answers(self) -> set:
        return {a for _, c in self.by_ordering for a in c}

    def __repr__(self) -> str:
        return (f"ProbeResult(value={self.value!r}, verdict={self.verdict!r}, "
                f"dispersion={self.dispersion:.3f}, orderings={len(self.by_ordering)}, "
                f"calls={self.calls}, errors={self.errors})")


def _tv(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return 0.0
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)


def _orderings(n: int, k: int, seed: int) -> list[tuple[int, ...]]:
    """Canonical order first, then distinct random permutations.

    Capped at n! so a small input cannot request more distinct orderings than
    exist -- asking for 6 orderings of 3 items would otherwise either loop or
    silently repeat, and a repeated ordering is a fake agreement.
    """
    canonical = tuple(range(n))
    total = math.factorial(n)
    if n < 2 or k < 2 or total < 2:
        return [canonical]
    k = min(k, total)
    if total <= 5040:                       # cheap to enumerate exactly
        pool = [p for p in itertools.permutations(range(n)) if p != canonical]
        random.Random(seed).shuffle(pool)
        return [canonical] + pool[: k - 1]
    rng = random.Random(seed)
    seen, out = {canonical}, [canonical]
    while len(out) < k:
        p = tuple(rng.sample(range(n), n))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def probe_call(
    fn: Callable[[Sequence], Any],
    items: Sequence,
    *,
    k: int = 6,
    samples: int = 5,
    seed: int = 0,
    key: Callable[[Any], Hashable] | None = None,
    threshold: float | None = None,
    max_workers: int = 1,
) -> ProbeResult:
    """Run `fn` over `k` orderings of `items`, `samples` times each.

    key       -- maps a return value to something hashable for comparison.
                 Defaults to the value itself, falling back to `repr` when the
                 value is unhashable.
    threshold -- if given, `unstable` means dispersion > threshold. If omitted,
                 `unstable` means the modal answers of two orderings differ,
                 which is the definition the published evidence used.
    max_workers -- >1 runs orderings concurrently. `fn` must then be
                 thread-safe.
    """
    def _key(v: Any) -> Hashable:
        if key is not None:
            return key(v)
        try:
            hash(v)
            return v
        except TypeError:
            return repr(v)

    n = len(items)
    orders = _orderings(n, k, seed)

    def run(order: tuple[int, ...]) -> tuple[tuple[int, ...], Counter, int, int]:
        c: Counter = Counter()
        calls = errs = 0
        arg = [items[i] for i in order]
        for _ in range(samples):
            calls += 1
            try:
                c[_key(fn(arg))] += 1
            except Exception:
                errs += 1
        return order, c, calls, errs

    if max_workers > 1 and len(orders) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            got = list(pool.map(run, orders))
        got.sort(key=lambda t: orders.index(t[0]))     # keep canonical first
    else:
        got = [run(o) for o in orders]

    by = [(o, c) for o, c, _, _ in got if sum(c.values())]
    calls = sum(x[2] for x in got)
    errors = sum(x[3] for x in got)

    if not by:
        return ProbeResult(None, Verdict.NOT_APPLICABLE, 0.0, [], calls, errors)

    canonical_dist = by[0][1]
    value = canonical_dist.most_common(1)[0][0]

    if len(by) < 2:
        return ProbeResult(value, Verdict.NOT_APPLICABLE, 0.0, by, calls, errors)

    dispersion = max(_tv(a, b) for (_, a), (_, b) in itertools.combinations(by, 2))
    if threshold is None:
        modals = {c.most_common(1)[0][0] for _, c in by}
        unstable = len(modals) > 1
    else:
        unstable = dispersion > threshold

    return ProbeResult(value, Verdict.UNSTABLE if unstable else Verdict.STABLE,
                       dispersion, by, calls, errors)


def probe(k: int = 6, samples: int = 5, *, seed: int = 0,
          key: Callable[[Any], Hashable] | None = None,
          threshold: float | None = None, max_workers: int = 1):
    """Decorator form. The wrapped function must take one positional argument:
    the ordered sequence whose order you want to test."""
    def deco(fn: Callable[[Sequence], Any]):
        @functools.wraps(fn)
        def wrapper(items: Sequence, *a, **kw) -> ProbeResult:
            bound = fn if not (a or kw) else (lambda xs: fn(xs, *a, **kw))
            return probe_call(bound, items, k=k, samples=samples, seed=seed,
                              key=key, threshold=threshold, max_workers=max_workers)
        wrapper.unprobed = fn
        return wrapper
    return deco
