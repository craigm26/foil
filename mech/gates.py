"""MECH-1 gates, in the order METHODS §2 requires. Each one blocks.

    G0  eligibility   free   pure softmax attention, gated MLP      (whitebox.eligibility)
    G1  structural    free   every arm renders, maps and reads out  (tokenizer only)
    G2  leak          paid*  knockouts cut every path on these weights
    G3  construct     paid*  the model reaches the truth with full, honest information

    * paid in compute, not dollars. Nothing here calls an API.

G2 exists because "cuts every path" is a claim about an architecture, and the
weights should be made to confirm it: overwrite a knocked-out span with random
tokens and nothing after it may move. It also checks it CAN fail -- the same
overwrite without the knockout must move the logits, or the test was vacuous.

G3 uses evalgate.gates.verdict, so a near-miss is EXTEND, not a one-run cliff
(TURN-3). The extension is fixed here, before any model has run.
"""

from __future__ import annotations

import numpy as np

from evalgate import structural
from evalgate.gates import verdict as gate_verdict
from foil.env3 import make_episode_v3
from foil.nulls import paraphrased_episode
from foil.render import Operator

from .local import (ReadoutError, RenderMismatch, claims_from_text, honest_variant,
                    identified_route, knock, render_local)

#: Registered seeds (PREREGISTRATION-MECH.md §5). Disjoint from every prior
#: FOIL study: v1-v3 used 1-12; PID-2 1000+; TURN 2000+; TURN-2 3000+;
#: TURN-3 4000+; SWEEP 7000+.
STUDY_SEEDS = tuple(range(9000, 9048))
CONSTRUCT_SEED_BASE = 9500
CONSTRUCT_N = 48
CONSTRUCT_EXTENSION = 24
CONSTRUCT_THRESHOLD = 0.90
CONSTRUCT_ALPHA = 0.05
#: Median next-token mass on the four route tokens after the prefill. Below
#: this the renormalised distribution is a reading of the tail, not the answer.
MIN_ON_FORMAT = 0.90
LEAK_SEEDS = tuple(range(9600, 9608))
#: fp32 tolerance on any decision-position logit. Masked keys get exactly zero
#: weight, so the true difference is 0; this allows only reduction-order noise.
LEAK_TOL = 1e-4


def _arms(tok, ep, seed: int):
    """Every prompt MECH-1 renders for one episode, exactly as study.py will:
    N1 at six orderings, N2 and REF at the canonical order, and the edited
    per-source arms at the episode's arms ordering. (The knockout, partial
    knockout and shift arms reuse the full prompt at that ordering, which N1
    already contains.)"""
    from .study import ORDERINGS, arms_ordering, orderings
    src = tuple(ep.source_ids)
    full = frozenset(src)
    out = [render_local(tok, ep, o, full, Operator.NULL) for o in orderings(src, ORDERINGS, seed)]
    out.append(render_local(tok, paraphrased_episode(ep), src, full, Operator.NULL))
    ao = arms_ordering(src, seed)
    for s in src:
        out.append(render_local(tok, ep, src, full - {s}, Operator.NULL))
        out.append(render_local(tok, ep, ao, full - {s}, Operator.NULL))
        out.append(render_local(tok, ep, ao, full - {s}, Operator.DEL))
    return out


def _span_problems(prompt) -> list[str]:
    bad = []
    owner: dict[int, str] = {}
    for sid, sm in prompt.spans.items():
        if not sm.tokens:
            bad.append(f"{sid}: no tokens")
            continue
        covered = prompt.text[prompt.offsets[sm.first][0]:prompt.offsets[sm.last][1]]
        if sm.text not in covered:
            bad.append(f"{sid}: tokens do not cover the line text")
        if not sm.summary:
            bad.append(f"{sid}: no punctuation token")
        if not sm.content:
            bad.append(f"{sid}: no content token")
        for i in sm.tokens:
            if i in owner:
                bad.append(f"token {i} shared by {owner[i]} and {sid}")
            owner[i] = sid
        if sm.last >= prompt.decision_pos:
            bad.append(f"{sid}: span reaches the decision position")
    return bad


def structural_gate(tok, study_seeds=STUDY_SEEDS) -> dict:
    """G1, free. Every property re-derived from rendered text and tokens.

    Study episodes: every arm renders through the model's template, every
    report maps to its own tokens with at least one punctuation and one
    content token, the readout is one token per action, and the env3
    decisiveness requirement still holds when re-derived from the rendered
    lines rather than read from the generator.
    Construct cases: every honest variant renders and pins the labelled truth
    from its own text (`construct_seeds` reports how many seeds that skipped).
    """
    from foil.env3 import MIN_DECISIVE, decisive_sources

    failures: list[tuple[str, str]] = []

    def make(seed):
        ep = make_episode_v3(seed)
        return None if ep is None else (seed, ep)

    def holds(case) -> bool:
        seed, ep = case
        try:
            arms = _arms(tok, ep, seed)
            for pr in arms:
                probs = _span_problems(pr)
                if probs:
                    failures.append((ep.episode_id, "; ".join(probs)))
                    return False
            lines = [sm.text for sm in arms[0].spans.values()]
            if len(decisive_sources(claims_from_text(lines))) < MIN_DECISIVE:
                failures.append((ep.episode_id, "fewer than MIN_DECISIVE decisive sources in the rendered text"))
                return False
            return True
        except (ReadoutError, RenderMismatch, ValueError) as e:
            failures.append((ep.episode_id, repr(e)))
            return False

    rep = structural.verify(make, holds, n=len(study_seeds), seed_base=study_seeds[0])

    seeds, skipped = construct_seeds(tok, CONSTRUCT_N + CONSTRUCT_EXTENSION, report_skips=True)
    construct_bad = 0
    for seed in seeds:
        try:
            hv = honest_variant(tok, make_episode_v3(seed))
            if _span_problems(hv):
                construct_bad += 1
        except (ReadoutError, RenderMismatch, ValueError):
            construct_bad += 1
    ok = rep.ok and rep.n_generated == len(study_seeds) and construct_bad == 0
    return {
        "gate": "G1 structural",
        "outcome": "PASS" if ok else "FAIL",
        "study_episodes": {"requested": rep.n_requested, "generated": rep.n_generated,
                           "holding": rep.n_holding},
        "construct_cases": len(seeds),
        "construct_seeds_skipped_unidentifiable": skipped,
        "construct_cases_failing": construct_bad,
        "failures": failures[:10],
    }


def construct_seeds(tok, n: int, report_skips: bool = False):
    """The first n seeds from CONSTRUCT_SEED_BASE whose honest variant pins the
    truth from its rendered text. Decided analytically, before any model."""
    out, seed, skipped = [], CONSTRUCT_SEED_BASE, 0
    while len(out) < n:
        ep = make_episode_v3(seed)
        if ep is not None:
            hv = honest_variant(tok, ep)
            lines = [sm.text for sm in hv.spans.values()]
            if identified_route(claims_from_text(lines), tuple(ep.actions)) == ep.correct_action:
                out.append(seed)
            else:
                skipped += 1
        seed += 1
        if seed > CONSTRUCT_SEED_BASE + 10_000:
            raise RuntimeError("could not find enough identifiable construct cases")
    return (out, skipped) if report_skips else out


def leak_gate(wb, seeds=LEAK_SEEDS, tol=LEAK_TOL) -> dict:
    """G2. For every source: knockout + random overwrite of the span must leave
    the decision-position logits unchanged; the overwrite alone must not."""
    rng = np.random.default_rng(9600)
    specials = set(getattr(wb.tok, "all_special_ids", []) or [])
    vocab = [i for i in range(len(wb.tok)) if i not in specials][:50_000]
    worst, weakest_power, n = 0.0, float("inf"), 0
    for seed in seeds:
        ep = make_episode_v3(seed)
        base = render_local(wb.tok, ep)
        plain = wb.run(base, keep_full_logits=True)
        for s in ep.source_ids:
            keys, q0 = knock(base, s, "all")
            ids = list(base.input_ids)
            for k in keys:
                ids[k] = int(rng.choice(vocab))
            edited = base.with_ids(tuple(ids))
            a = wb.run(base, knockouts=[(keys, q0)], keep_full_logits=True)
            b = wb.run(edited, knockouts=[(keys, q0)], keep_full_logits=True)
            c = wb.run(edited, keep_full_logits=True)
            worst = max(worst, float(np.max(np.abs(a.full_logits - b.full_logits))))
            weakest_power = min(weakest_power, float(np.max(np.abs(c.full_logits - plain.full_logits))))
            n += 1
    vacuous = weakest_power <= 100 * tol
    return {
        "gate": "G2 leak",
        "outcome": "PASS" if worst <= tol and not vacuous else "FAIL",
        "tests": n,
        "max_abs_logit_diff_with_knockout": worst,
        "min_abs_logit_diff_without_knockout": weakest_power,
        "tolerance": tol,
        "vacuous": vacuous,
    }


def construct_gate(wb, log=print) -> dict:
    """G3. Honest variant, canonical order: argmax must be the truth.

    Three outcomes via evalgate.gates.verdict. EXTEND runs the registered
    extension once and re-applies the rule to the pooled counts; still EXTEND
    afterwards resolves to FAIL. The readout must also be on-format.
    """
    seeds = construct_seeds(wb.tok, CONSTRUCT_N + CONSTRUCT_EXTENSION)

    def batch(ss):
        k, masses, misses = 0, [], []
        for seed in ss:
            ep = make_episode_v3(seed)
            hv = honest_variant(wb.tok, ep)
            r = wb.run(hv)
            masses.append(r.on_format)
            if int(np.argmax(r.probs)) == hv.correct_index:
                k += 1
            elif len(misses) < 10:
                misses.append({"seed": seed, "truth": ep.correct_action,
                               "answered": ep.actions[int(np.argmax(r.probs))],
                               "p_truth": float(r.probs[hv.correct_index])})
        return k, masses, misses

    k, masses, misses = batch(seeds[:CONSTRUCT_N])
    v = gate_verdict(k, CONSTRUCT_N, CONSTRUCT_THRESHOLD, CONSTRUCT_ALPHA)
    log(f"  construct: {v}")
    n = CONSTRUCT_N
    extended = False
    if v.outcome == "EXTEND":
        extended = True
        k2, m2, x2 = batch(seeds[CONSTRUCT_N:])
        k, n = k + k2, n + CONSTRUCT_EXTENSION
        masses += m2
        misses += x2
        v = gate_verdict(k, n, CONSTRUCT_THRESHOLD, CONSTRUCT_ALPHA)
        log(f"  construct after extension: {v}")
    outcome = v.outcome
    if outcome == "EXTEND":
        outcome = "FAIL"  # unresolved after the registered extension (§5)
    med_mass = float(np.median(masses))
    if med_mass < MIN_ON_FORMAT:
        outcome = "FAIL"
    return {
        "gate": "G3 construct",
        "outcome": outcome,
        "k": k, "n": n, "rate": k / n, "threshold": CONSTRUCT_THRESHOLD,
        "binomial_p": v.p_value, "extended": extended,
        "median_on_format_mass": med_mass, "min_on_format_mass": float(min(masses)),
        "misses": misses[:10],
    }
