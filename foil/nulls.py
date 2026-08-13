"""Phase 0 null experiments and the kill rule (PREREGISTRATION.md §5).

No attribution machinery is built until these run. Each can terminate the
project, and that outcome is a publishable finding about the fragility of
ablation-based attribution in language models generally.

    N1  identical reports, permuted presentation order   -> order noise floor
    N2  semantically identical paraphrase, order fixed   -> paraphrase noise floor
    REF full coalition vs each single-source ablation    -> the signal to beat

KILL RULE: let T_null be the 95th percentile TV across N1 and N2, and T_ablate
the median TV produced by ablating one source under the primary operator. If
T_null >= 0.5 * T_ablate the project stops.
"""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .execute import Executor
from .ir import Episode, Span, SpanKind
from .render import ForkKey, Operator, parse_action, render
from .stats import bootstrap_tv_ci, distribution, percentile, total_variation


def paraphrase(span: Span) -> Span:
    """Semantically identical rewrite of a report.

    Deliberately mechanical and content-preserving: it changes wording, not
    which routes are asserted clear or blocked. If the action distribution
    moves under this, the model is responding to surface form and the
    instrument's noise floor is doing the work.
    """
    t = span.text
    m = re.match(r"Scout (\w+): (.*)\.$", t)
    if not m:
        return span
    name, body = m.group(1), m.group(2)
    body = body.replace(" are clear", " passable").replace(" is clear", " passable")
    body = body.replace(" are blocked", " impassable").replace(" is blocked", " impassable")
    body = body.replace("; ", ", and ")
    return Span(SpanKind.REPORT, f"Scout {name} reports {body}.", span.source_id, span.claim_id)


def paraphrased_episode(ep: Episode) -> Episode:
    spans = tuple(paraphrase(s) if s.kind is SpanKind.REPORT else s for s in ep.spans)
    return Episode(
        episode_id=ep.episode_id + "-para",
        spans=spans,
        actions=ep.actions,
        correct_action=ep.correct_action,
        reliability=ep.reliability,
        coverage=ep.coverage,
    )


@dataclass
class Arm:
    """One measured configuration: a fork plus its drawn samples."""

    label: str
    key: ForkKey
    actions: list[str | None] = field(default_factory=list)
    unparseable: int = 0

    def dist(self, support: tuple[str, ...]):
        return distribution(self.actions, support)


def _draw(ex: Executor, ep: Episode, key: ForkKey, n: int, label: str, dry_run: bool) -> Arm:
    body = render(ep, key)
    arm = Arm(label=label, key=key)
    recs = ex.sample(body, n, dry_run=dry_run)
    if dry_run:
        return arm
    for r in recs:
        act, _conf = parse_action(r["text"], ep.actions)
        if act is None:
            arm.unparseable += 1
            ex.ledger.unparseable += 1
        arm.actions.append(act)
    return arm


def run_nulls(
    ex: Executor,
    ep: Episode,
    n: int = 30,
    orders: int = 6,
    seed: int = 0,
    dry_run: bool = False,
) -> dict:
    sources = ep.source_ids
    full = frozenset(sources)
    canonical = tuple(sources)

    rng = random.Random(seed)
    perms = [canonical]
    pool = [p for p in itertools.permutations(sources) if p != canonical]
    rng.shuffle(pool)
    perms += pool[: max(0, orders - 1)]

    def key(coalition, order, operator=Operator.NULL, episode=ep):
        return ForkKey(
            episode_id=episode.episode_id,
            decision_index=0,
            coalition=coalition,
            operator=operator,
            render_order=order,
            model=ex.model,
        )

    # ---- N1: order sensitivity, full coalition ----
    n1_arms = [
        _draw(ex, ep, key(full, p), n, f"N1:order{i}", dry_run)
        for i, p in enumerate(perms)
    ]

    # ---- N2: paraphrase sensitivity, canonical order ----
    ep_para = paraphrased_episode(ep)
    n2_arms = [
        n1_arms[0],
        _draw(ex, ep_para, key(full, canonical, episode=ep_para), n, "N2:paraphrase", dry_run),
    ]

    # ---- REF: single-source ablation under the primary operator ----
    ref_arms = [
        _draw(ex, ep, key(full - {s}, canonical), n, f"REF:ablate-{s}", dry_run)
        for s in sources
    ]

    if dry_run:
        total = (len(n1_arms) + 1 + len(ref_arms)) * n
        return {"dry_run": True, "planned_calls": total, "arms": len(n1_arms) + 1 + len(ref_arms)}

    base = n1_arms[0]

    n1_tvs, n1_detail = [], []
    for a, b in itertools.combinations(n1_arms, 2):
        tv, lo, hi = bootstrap_tv_ci(a.actions, b.actions, ep.actions, seed=seed)
        n1_tvs.append(tv)
        n1_detail.append({"a": a.label, "b": b.label, "tv": tv, "ci": [lo, hi]})

    tv2, lo2, hi2 = bootstrap_tv_ci(n2_arms[0].actions, n2_arms[1].actions, ep.actions, seed=seed)
    n2_tvs = [tv2]

    ref_tvs, ref_detail = [], []
    for arm in ref_arms:
        tv, lo, hi = bootstrap_tv_ci(base.actions, arm.actions, ep.actions, seed=seed)
        ref_tvs.append(tv)
        ref_detail.append({"arm": arm.label, "tv": tv, "ci": [lo, hi]})

    t_null = percentile(n1_tvs + n2_tvs, 95)
    t_ablate = percentile(ref_tvs, 50)

    verdict = "INDETERMINATE"
    if t_null is not None and t_ablate is not None:
        verdict = "KILL" if (t_ablate == 0 or t_null >= 0.5 * t_ablate) else "PROCEED"

    all_arms = n1_arms + n2_arms[1:] + ref_arms
    short = {a.label: len(a.actions) for a in all_arms if len(a.actions) < n}

    return {
        "episode_id": ep.episode_id,
        "model": ex.model,
        "correct_action": ep.correct_action,
        "samples_per_arm_requested": n,
        "arm_sizes": {a.label: len(a.actions) for a in all_arms},
        # An arm that did not reach n is a caveat on every statistic derived
        # from it, so it is surfaced rather than left to be inferred.
        "short_arms": short,
        "N1": {"pairs": n1_detail, "tvs": n1_tvs},
        "N2": {"tv": tv2, "ci": [lo2, hi2]},
        "REF": {"arms": ref_detail, "tvs": ref_tvs},
        "T_null_p95": t_null,
        "T_ablate_median": t_ablate,
        "kill_threshold": None if t_ablate is None else 0.5 * t_ablate,
        "verdict": verdict,
        "base_distribution": {a: float(v) for a, v in zip(ep.actions, base.dist(ep.actions))},
        "unparseable": sum(a.unparseable for a in n1_arms + n2_arms[1:] + ref_arms),
        "ledger": ex.ledger.summary(),
    }
