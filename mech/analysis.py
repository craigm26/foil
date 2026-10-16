"""MECH-1 statistics and decision rules. numpy only: no torch, no model.

Kept apart from the model code so the registered analysis can be read, tested
and re-run from a results file on any machine, the way the core harness can.
Every constant below is the one fixed in PREREGISTRATION-MECH.md. Changing one
is an amendment with a date and a reason, not a refactor.

THE DECOMPOSITION
-----------------
FOIL's Finding 1 says removing a source changes two things at once: the
information available, and the arrangement of everything after it. With the
weights in hand the two can be pulled apart, because a span can be made
invisible without moving a single token (attention knockout, `mech.whitebox`).
Four contexts per source `s`, each read as P(correct):

    full        all reports visible                         (the factual)
    ko          s's span present but invisible downstream   (information gone, arrangement kept)
    null_ko     s replaced by Ω_null's text, also invisible (information gone, slot resized)
    null        s replaced by Ω_null's text, visible         (FOIL's primary operator)

and the single-source Ω_null effect telescopes:

    Δ_null = (ko - full) + (null_ko - ko) + (null - null_ko)
           =   info      +   arr_null     +   text_null

This is an accounting identity along one PATH, not a unique attribution: any
interaction between information and position is credited to the later step.
So a second path is measured too -- `shift` keeps s visible and moves later
tokens to Ω_null's positions -- giving

    Δ_null = (shift - full) + (null_ko - shift) + (null - null_ko)
           =   arr_rev      +   info_rev        +   text_null

and both arrangement shares are reported. Two caveats the registration
states: an invisible slot still reveals its LENGTH through position indices
(reports are 8, 10 or 12 tokens, which encodes their structure), so `info`
excludes whatever a listener reads from a gap's size; and a shifted position
can coincide with a span token's, which is off-distribution as a gap is.

Ω_del decomposes in two terms: Δ_del = info + arr_del, arr_del = del - ko.
For the last report, deletion also re-tokenises the previous report's final
token (".\\n" becomes ".\\n\\n"), so arr_del there includes a visible change.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from foil.nulls import BISTABLE_TV, _verdict  # one copy of each, imported not restated
from foil.stats import percentile, total_variation

# ---------------------------------------------------------------------------
# Registered constants (PREREGISTRATION-MECH.md §5-§7). Fixed before any data.
# ---------------------------------------------------------------------------

#: |ΔP(correct)| at which a source counts as having moved the decision.
ACTIVE = 0.10
#: H1 decision line on the arrangement share of Ω_null's effect.
ARR_SHARE_LINE = 0.20
#: H2 decision line on the routing ratios.
ROUTE_LINE = 0.50
#: H3 decision line on carrier fidelity, and the registered primary k.
FIDELITY_LINE = 0.50
PRIMARY_K = 16
K_LADDER = (4, 16, 64, 256)
#: A pass whose next-token mass on the four route tokens falls below this is
#: not reading an answer. A unit with any such pass is a dropout.
ON_FORMAT_FLOOR = 0.50
#: METHODS §3: dropouts above 5% label the result compromised.
COMPROMISED_FRACTION = 0.05
#: Cluster bootstrap over episodes.
BOOT_ITERS = 2000
BOOT_SEED = 0
#: Below these, a hypothesis is DEGENERATE rather than tested.
MIN_UNITS = 20
MIN_EPISODES = 10
#: The two-path identity must hold to float precision, or the record is corrupt.
IDENTITY_TOL = 1e-9


# ---------------------------------------------------------------------------
# Per-unit quantities
# ---------------------------------------------------------------------------

def components(unit: dict) -> dict[str, float]:
    """The decomposition for one (episode, source) unit, both paths."""
    p = unit["p"]
    out = {
        "info": p["ko"] - p["full"],
        "arr_null": p["null_ko"] - p["ko"],
        "text_null": p["null"] - p["null_ko"],
        "arr_del": p["del"] - p["ko"],
        "d_null": p["null"] - p["full"],
        "d_del": p["del"] - p["full"],
    }
    if "shift" in p:
        out["arr_rev"] = p["shift"] - p["full"]
        out["info_rev"] = p["null_ko"] - p["shift"]
    resid = out["d_null"] - (out["info"] + out["arr_null"] + out["text_null"])
    if abs(resid) > IDENTITY_TOL:
        raise ValueError(f"decomposition identity violated by {resid:g} for "
                         f"{unit.get('episode')}/{unit.get('source')}")
    return out


#: The passes each registered quantity rests on (§3.1). A unit is a dropout for
#: a quantity only if one of THAT quantity's passes fell below the floor, so a
#: failing carrier patch cannot remove a unit from H1, and exclusion never
#: depends on another hypothesis's outcome.
ARMS = {
    "H1": ("full", "ko", "null_ko", "null"),
    "H1_rev": ("full", "ko", "null_ko", "null", "shift"),
    "H1_del": ("full", "ko", "del"),
    "H1_any": ("full", "ko", "null_ko", "null", "del"),
    "H2": ("full", "ko", "ko_sum", "ko_con"),
    "H2_inner": ("full", "ko", "ko_inner"),
    "H2_final": ("full", "ko", "ko_final"),
    "H3": ("full", "ko", "car_k16", "car_matched"),
    "H3b": ("full", "ko", "pre", "pre_matched"),
    "KO": ("full", "ko"),
}


def valid_for(unit: dict, quantity: str) -> bool:
    """Every pass behind `quantity` read an answer. A pass that was never run
    (an arm the unit does not have) makes the quantity unavailable."""
    m = unit.get("on_format", {})
    return all(a in m and m[a] >= ON_FORMAT_FLOOR for a in ARMS[quantity])


def min_on_format(unit: dict) -> float:
    vals = list(unit.get("on_format", {}).values())
    return min(vals) if vals else 1.0


def is_active(unit: dict) -> bool:
    """H1 population: some operator moved P(correct) by at least ACTIVE.
    Activity is judged on the arms H1 reads (full, ko, null, del), all of
    which must be on-format for the judgement to be made."""
    c = components(unit)
    return max(abs(c["d_null"]), abs(c["d_del"]), abs(c["info"])) >= ACTIVE


def is_ko_active(unit: dict) -> bool:
    """H2/H3 population: hiding the span in place moved P(correct)."""
    return abs(components(unit)["info"]) >= ACTIVE


def route_ratios(unit: dict) -> dict[str, float]:
    """Share of the in-place effect reproduced by hiding only part of the span.

    1 means hiding that part alone removes the whole effect; 0 means the rest
    of the span still carries it. Ratios outside [0, 1] are kept, not clipped.
    """
    p = unit["p"]
    info = p["ko"] - p["full"]
    if info == 0:
        raise ZeroDivisionError("route ratios are undefined when the knockout moved nothing")
    out = {"sum": (p["ko_sum"] - p["full"]) / info, "con": (p["ko_con"] - p["full"]) / info}
    for part in ("inner", "final"):
        if f"ko_{part}" in p:
            out[part] = (p[f"ko_{part}"] - p["full"]) / info
    return out


def fidelities(unit: dict) -> dict:
    """Carrier fidelity, (P_patched - P_full) / (P_ko - P_full).

    Keys: each k of the ladder, "matched" and "top_abs" (H3's controls), and
    for H3b "pre", "pre_matched", "pre_top_abs".
    """
    car = unit.get("carriers")
    if not car:
        return {}
    p = unit["p"]
    info = p["ko"] - p["full"]

    def f(v: float) -> float:
        return (v - p["full"]) / info

    out: dict = {int(k): f(v) for k, v in car["by_k"].items()}
    out["matched"] = f(car["matched"])
    out["top_abs"] = f(car["top_abs"])
    pre = car.get("pre_answer")
    if pre:
        out["pre"] = f(pre["p"])
        out["pre_matched"] = f(pre["matched"])
        out["pre_top_abs"] = f(pre["top_abs"])
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _by_episode(units: list[dict]) -> "OrderedDict[str, list[dict]]":
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for u in units:
        groups.setdefault(u["episode"], []).append(u)
    return groups


def cluster_bootstrap(
    units: list[dict],
    stat: Callable[[list[dict]], float | None],
    iters: int = BOOT_ITERS,
    seed: int = BOOT_SEED,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    """Point estimate and percentile CI, resampling whole EPISODES.

    The four sources of one episode share a listener state and are not
    independent; resampling units would overstate the evidence. Episodes are
    the units of replication here, so episodes are what gets resampled.
    """
    point = stat(units)
    groups = _by_episode(units)
    keys = list(groups)
    if point is None or not keys:
        return point, None, None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iters):
        pick = rng.integers(0, len(keys), size=len(keys))
        sample = [u for i in pick for u in groups[keys[i]]]
        v = stat(sample)
        if v is not None and v == v:
            vals.append(v)
    if not vals:
        return point, None, None
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def _degenerate(units: list[dict]) -> str | None:
    eps = {u["episode"] for u in units}
    if len(units) < MIN_UNITS or len(eps) < MIN_EPISODES:
        return (f"{len(units)} units from {len(eps)} episodes; the registration "
                f"requires >= {MIN_UNITS} from >= {MIN_EPISODES}")
    return None


def _ci(t: tuple) -> dict:
    point, lo, hi = t
    return {"point": point, "lo": lo, "hi": hi}


def _median_of(fn: Callable[[dict], float]) -> Callable[[list[dict]], float | None]:
    def stat(us: list[dict]) -> float | None:
        vals = [v for v in (fn(u) for u in us) if v is not None and v == v]
        return float(np.median(vals)) if vals else None
    return stat


def _fraction_of(pred: Callable[[dict], bool]) -> Callable[[list[dict]], float | None]:
    def stat(us: list[dict]) -> float | None:
        return float(np.mean([bool(pred(u)) for u in us])) if us else None
    return stat


# ---------------------------------------------------------------------------
# H1 -- how much of a single-source ablation effect is arrangement?
# ---------------------------------------------------------------------------

def share(units: list[dict], arr_key: str, parts: tuple[str, ...]) -> float | None:
    """Σ|arr| / Σ Σ|part| over units. Absolute values, deliberately: an
    arrangement term that cancels an information term is contamination even
    when the net effect looks small."""
    num = den = 0.0
    for u in units:
        c = components(u)
        if any(k not in c for k in parts):
            continue
        num += abs(c[arr_key])
        den += sum(abs(c[k]) for k in parts)
    return num / den if den > 0 else None


NULL_PATH = ("info", "arr_null", "text_null")
REV_PATH = ("info_rev", "arr_rev", "text_null")
DEL_PATH = ("info", "arr_del")


def arrangement_share(units: list[dict], operator: str = "null") -> float | None:
    if operator == "null":
        return share(units, "arr_null", NULL_PATH)
    if operator == "rev":
        return share(units, "arr_rev", REV_PATH)
    if operator == "del":
        return share(units, "arr_del", DEL_PATH)
    raise ValueError(operator)


def h1(units: list[dict], bistable: dict[str, bool] | None = None) -> dict:
    """H1 on units whose H1 passes are all on-format (§3.1)."""
    ok = [u for u in units if valid_for(u, "H1_any")]
    active = [u for u in ok if is_active(u)]
    out: dict = {"hypothesis": "H1", "n_units": len(units), "n_valid": len(ok),
                 "n_active": len(active), "n_active_episodes": len({u["episode"] for u in active})}
    why = _degenerate(active)
    if why:
        out.update(outcome="DEGENERATE", reason=why)
    else:
        s = cluster_bootstrap(active, lambda us: arrangement_share(us, "null"))
        out["arr_share_null"] = _ci(s)
        if s[2] is not None and s[2] < ARR_SHARE_LINE:
            out["outcome"] = "ARRANGEMENT MINOR"
        elif s[1] is not None and s[1] > ARR_SHARE_LINE:
            out["outcome"] = "ARRANGEMENT MATERIAL"
        else:
            out["outcome"] = "UNRESOLVED"
    if not active:
        return out

    # Reported regardless of the outcome (§7.1), DEGENERATE included.
    boot = lambda f, us=active: _ci(cluster_bootstrap(us, f)) if us else None  # noqa: E731
    rev = [u for u in active if valid_for(u, "H1_rev")]
    not_last = [u for u in active if not u.get("last_slot")]
    out["regardless"] = {
        "arr_share_null": boot(lambda us: arrangement_share(us, "null")),
        "arr_share_reverse_path": boot(lambda us: arrangement_share(us, "rev"), rev),
        "reverse_path_units": len(rev),
        "arr_share_del": boot(lambda us: arrangement_share(us, "del")),
        "arr_share_del_excluding_last_slot": boot(lambda us: arrangement_share(us, "del"), not_last),
        "arr_share_null_excluding_last_slot": boot(lambda us: arrangement_share(us, "null"), not_last),
        "info_share_null": boot(lambda us: share(us, "info", NULL_PATH)),
        "text_share_null": boot(lambda us: share(us, "text_null", NULL_PATH)),
        "concordant_fraction": boot(_fraction_of(
            lambda u: abs(components(u)["d_null"] - components(u)["info"]) <= ACTIVE)),
        "material_arrangement_fraction": boot(_fraction_of(
            lambda u: abs(components(u)["arr_null"]) >= ACTIVE)),
        "opposite_direction_fraction": boot(_fraction_of(
            lambda u: (components(u)["d_null"] * components(u)["info"] < 0
                       and abs(components(u)["d_null"]) >= ACTIVE
                       and abs(components(u)["info"]) >= ACTIVE))),
        # Report length maps one-to-one onto the shift here (8 -> +1,
        # 10 -> -1, 12 -> -3): this split cannot separate reading the gap's
        # size from the size of the shift.
        "arr_share_null_by_span_tokens": {
            str(n): {"n_units": len(g), "share": arrangement_share(g, "null")}
            for n, g in sorted(_group(active, lambda u: u.get("span_tokens")).items())
        },
    }
    if bistable is not None:
        strata = {}
        for label, flag in (("bistable", True), ("stable", False)):
            sub = [u for u in active if bistable.get(u["episode"]) is flag]
            strata[label] = {
                "n_units": len(sub), "n_episodes": len({u["episode"] for u in sub}),
                "arr_share_null": boot(lambda us: arrangement_share(us, "null"), sub),
            }
        strata["unclassified_units"] = sum(1 for u in active if u["episode"] not in bistable)
        out["regardless"]["by_order_stability"] = strata
    return out


def _group(units, key) -> dict:
    out: dict = {}
    for u in units:
        out.setdefault(key(u), []).append(u)
    return out


# ---------------------------------------------------------------------------
# H2 -- is a source read through its summary tokens?
# ---------------------------------------------------------------------------

def route_outcome(s: tuple, c: tuple) -> str:
    """H2's outcomes (§7.2). Each argument is (point, lo, hi) for the median
    ratio. All four named outcomes are interval statements, so they are
    mutually exclusive and none is easier to reach than its rivals."""
    if any(x is None for x in (*s, *c)):
        return "UNRESOLVED"
    L = ROUTE_LINE
    if s[1] >= L and c[1] >= L:
        return "JOINT"
    if s[2] < L and c[2] < L:
        return "REDUNDANT"
    if s[1] >= L and c[2] < L:
        return "SUMMARY-ROUTED"
    if c[1] >= L and s[2] < L:
        return "CONTENT-ROUTED"
    return "UNRESOLVED"


def h2(units: list[dict]) -> dict:
    """H2 on ko-active units whose H2 passes are all on-format."""
    act = [u for u in units if valid_for(u, "H2") and is_ko_active(u)]
    out: dict = {"hypothesis": "H2", "n_ko_active": len(act),
                 "n_ko_active_episodes": len({u["episode"] for u in act})}
    why = _degenerate(act)
    s = cluster_bootstrap(act, _median_of(lambda u: route_ratios(u)["sum"])) if act else (None,) * 3
    c = cluster_bootstrap(act, _median_of(lambda u: route_ratios(u)["con"])) if act else (None,) * 3
    inner_units = [u for u in act if valid_for(u, "H2_inner")]
    inner = (cluster_bootstrap(inner_units, _median_of(lambda u: route_ratios(u).get("inner")))
             if inner_units else (None,) * 3)
    if why:
        out.update(outcome="DEGENERATE", reason=why)
    else:
        out.update(r_sum=_ci(s), r_con=_ci(c), outcome=route_outcome(s, c))
    out["motif_reading"] = motif_reading(out["outcome"], inner, len(inner_units))
    if not act:
        return out
    boot = lambda f, us=act: _ci(cluster_bootstrap(us, f)) if us else None  # noqa: E731
    final_units = [u for u in act if valid_for(u, "H2_final")]
    out["regardless"] = {
        "r_sum": _ci(s), "r_con": _ci(c),
        "r_diff": boot(_median_of(lambda u: route_ratios(u)["sum"] - route_ratios(u)["con"])),
        "fraction_r_sum_at_or_above_line": boot(_fraction_of(lambda u: route_ratios(u)["sum"] >= ROUTE_LINE)),
        "fraction_r_con_at_or_above_line": boot(_fraction_of(lambda u: route_ratios(u)["con"] >= ROUTE_LINE)),
        "r_inner": _ci(inner), "inner_units": len(inner_units),
        "r_final": boot(_median_of(lambda u: route_ratios(u).get("final")), final_units),
        "last_slot_units": sum(1 for u in act if u.get("last_slot")),
        "r_sum_excluding_last_slot": boot(_median_of(lambda u: route_ratios(u)["sum"]),
                                          [u for u in act if not u.get("last_slot")]),
    }
    return out


def motif_reading(outcome: str, inner: tuple, n_inner: int) -> str:
    """What an H2 outcome says about the essay's summary motif (§2).

    Qwen3 fuses a report's closing period with its line break, so hiding the
    punctuation also hides the line structure. SUMMARY-ROUTED counts as
    support only when the inner punctuation -- the marks that carry no line
    break -- removes at least half the effect on its own (interval lower
    bound, on at least MIN_UNITS units); otherwise the result is routed
    through tokens that cannot be separated from line structure.
    """
    if outcome == "SUMMARY-ROUTED":
        if n_inner >= MIN_UNITS and inner[1] is not None and inner[1] >= ROUTE_LINE:
            return "SUPPORTS THE MOTIF"
        return "CONFOUNDED WITH LINE STRUCTURE"
    if outcome in ("CONTENT-ROUTED", "JOINT"):
        return "AGAINST THE MOTIF"
    return "UNINFORMATIVE ABOUT THE MOTIF"


# ---------------------------------------------------------------------------
# H3 / H3b -- do a few MLP neuron-sites carry a source into the decision?
# ---------------------------------------------------------------------------

#: The matched control must change about as much as the units it stands in
#: for: median |Δ_substitute| / |Δ_selected| at least this, or a win over it
#: cannot establish that the chosen units are special (§7.3).
MATCHED_SIZE_FLOOR = 0.50


def h3(units: list[dict], k: int = PRIMARY_K, variant: str = "all") -> dict:
    """H3 (variant "all") and H3b (variant "pre": assistant turn excluded)."""
    if variant == "all":
        key, ctrl, absk, quantity = k, "matched", "top_abs", "H3"
        ratio_of = lambda u: u["carriers"].get("matched_size_ratio")  # noqa: E731
    else:
        key, ctrl, absk, quantity = "pre", "pre_matched", "pre_top_abs", "H3b"
        ratio_of = lambda u: u["carriers"]["pre_answer"].get("matched_size_ratio")  # noqa: E731
    act = [u for u in units if is_ko_active(u) and u.get("carriers")
           and valid_for(u, quantity) and key in fidelities(u)]
    out: dict = {"hypothesis": quantity, "k": k,
                 "n_units": len(act), "n_episodes": len({u["episode"] for u in act})}
    why = _degenerate(act)
    f = cluster_bootstrap(act, _median_of(lambda u: fidelities(u)[key])) if act else (None,) * 3
    d = cluster_bootstrap(act, _median_of(lambda u: fidelities(u)[key] - fidelities(u)[ctrl])) \
        if act else (None,) * 3
    ratios = [r for r in (ratio_of(u) for u in act) if r is not None]
    size = float(np.median(ratios)) if ratios else None
    if why:
        out.update(outcome="DEGENERATE", reason=why)
    else:
        out.update(fidelity=_ci(f), minus_matched=_ci(d), matched_size_ratio=size)
        if f[1] is not None and f[1] >= FIDELITY_LINE:
            if d[1] is not None and d[1] > 0:
                out["outcome"] = ("FEW-NEURON MEDIATION" if size is not None and size >= MATCHED_SIZE_FLOOR
                                  else "MEDIATION, SPECIFICITY UNRESOLVED")
            else:
                out["outcome"] = "MEDIATION, NOT UNIT-SPECIFIC"
        elif f[2] is not None and f[2] < FIDELITY_LINE:
            out["outcome"] = "NOT FEW"
        else:
            out["outcome"] = "UNRESOLVED"
    if not act:
        return out
    boot = lambda g: _ci(cluster_bootstrap(act, g))  # noqa: E731
    reg = {
        "fidelity": _ci(f), "minus_matched": _ci(d),
        "matched": boot(_median_of(lambda u: fidelities(u)[ctrl])),
        "matched_size_ratio": size,
        "top_abs": boot(_median_of(lambda u: fidelities(u)[absk])),
        "fraction_at_or_above_line": boot(_fraction_of(lambda u: fidelities(u)[key] >= FIDELITY_LINE)),
        "fraction_overshoot_above_1": boot(_fraction_of(lambda u: fidelities(u)[key] > 1.0)),
    }
    if variant == "all":
        reg["ladder"] = {}
        for kk in K_LADDER:
            vals = [fidelities(u)[kk] for u in act
                    if u.get("on_format", {}).get(f"car_k{kk}", 0.0) >= ON_FORMAT_FLOOR]
            reg["ladder"][str(kk)] = float(np.median(vals)) if vals else None
        reg["units_short_by_k"] = {str(kk): sum(1 for u in act if kk in {int(x) for x in u["carriers"]["shortfall"]})
                                   for kk in K_LADDER}
    else:
        reg["units_short_of_k"] = sum(1 for u in act if u["carriers"]["pre_answer"].get("shortfall"))
    out["regardless"] = reg
    return out


def carrier_locations(units: list[dict], k: int = PRIMARY_K, variant: str = "all") -> dict:
    """Descriptive only: where the top-k carriers sit -- position class, depth
    within each class, and SwiGLU sign regime. For H3b, the "turn" class is
    the end of the user's turn, the two tokens its caveat is about (§2)."""
    by_class: dict[str, list[float]] = {}
    regime_counts: dict[str, int] = {}
    total = 0
    for u in units:
        car = u.get("carriers")
        if not car:
            continue
        top = car["top"] if variant == "all" else car.get("pre_answer", {}).get("top", [])
        for c in top[:k]:
            total += 1
            by_class.setdefault(c["position_class"], []).append(c["layer"] / max(1, car["n_layers"] - 1))
            if c.get("regime_full") is not None:
                regime_counts[c["regime_full"]] = regime_counts.get(c["regime_full"], 0) + 1
    return {
        "n_carriers": total,
        "by_position_class": {
            cls: {"fraction": len(d) / total, "count": len(d),
                  "depth_quartiles": np.percentile(d, [25, 50, 75]).tolist()}
            for cls, d in sorted(by_class.items())
        } if total else {},
        "regime_full": {k_: v / total for k_, v in sorted(regime_counts.items())} if total else {},
    }


# ---------------------------------------------------------------------------
# R0 -- Phase 0, replicated on an open model with exact distributions
# ---------------------------------------------------------------------------

def episode_order_tvs(episode: dict) -> list[float]:
    dists = [np.asarray(x["probs"], dtype=float) for x in episode["n1"]]
    return [total_variation(a, b) for a, b in itertools.combinations(dists, 2)]


def bistable_flags(episodes: list[dict]) -> dict[str, bool]:
    return {e["episode"]: (max(episode_order_tvs(e)) if e["n1"] else 0.0) > BISTABLE_TV
            for e in episodes}


def phase0(episodes: list[dict]) -> dict:
    """FOIL's kill rule, unchanged, on exact action distributions.

    T_null is the 95th percentile of pooled N1 (order) and N2 (paraphrase)
    TVs; T_ablate the median of pooled single-source Ω_null TVs at the
    canonical order; the verdict comes from foil.nulls._verdict.
    """
    pooled_null: list[float] = []
    pooled_ref: list[float] = []
    per = []
    for e in episodes:
        full = np.asarray(e["n1"][0]["probs"], dtype=float)
        order_tvs = episode_order_tvs(e)
        n2 = total_variation(full, np.asarray(e["n2"]["probs"], dtype=float))
        ref = [total_variation(full, np.asarray(v, dtype=float)) for v in e["ref"].values()]
        pooled_null.extend(order_tvs + [n2])
        pooled_ref.extend(ref)
        per.append({"episode": e["episode"], "max_order_tv": max(order_tvs) if order_tvs else 0.0,
                    "n2_tv": n2, "ref_tvs": ref})
    t_null = percentile(pooled_null, 95)
    t_ablate = percentile(pooled_ref, 50)
    flags = bistable_flags(episodes)
    return {
        "n_episodes": len(episodes),
        "T_null_p95": t_null,
        "T_ablate_median": t_ablate,
        "kill_threshold": None if t_ablate is None else 0.5 * t_ablate,
        "verdict": _verdict(t_null, t_ablate),
        "pooled_null_n": len(pooled_null),
        "pooled_ref_n": len(pooled_ref),
        "bistable_fraction": (sum(flags.values()) / len(flags)) if flags else None,
        "per_episode": per,
    }


def accuracy(episodes: list[dict]) -> dict:
    """Canonical-ordering accuracy (argmax of the exact distribution)."""
    right = [int(np.argmax(e["n1"][0]["probs"]) == e["correct_index"]) for e in episodes]
    pc = [e["n1"][0]["probs"][e["correct_index"]] for e in episodes]
    return {"n": len(right), "argmax_correct": int(sum(right)),
            "rate": (sum(right) / len(right)) if right else None,
            "mean_p_correct": float(np.mean(pc)) if pc else None}


def by_role(units: list[dict]) -> dict:
    """Descriptive: mean in-place effect for the liar versus honest sources."""
    out = {}
    for role in sorted({u["role"] for u in units}):
        vals = [components(u)["info"] for u in units if u["role"] == role]
        out[role] = {"n": len(vals), "mean_info": float(np.mean(vals)),
                     "mean_abs_info": float(np.mean(np.abs(vals)))}
    return out


# ---------------------------------------------------------------------------
# Everything §7 says to report
# ---------------------------------------------------------------------------

def integrity(episodes: list[dict]) -> dict:
    """§7.5. VOID if any episode's hooks altered the computation. Otherwise a
    per-hypothesis dropout count, over ALL units, of those the hypothesis lost
    to off-format passes, with COMPROMISED above 5%; the same for R0 over
    episodes."""
    units = [u for e in episodes for u in e["units"]]
    void = any(not e["consistency"].get("ok", True) for e in episodes)
    drift = max((e["consistency"]["capture_vs_plain_max_abs"] for e in episodes), default=0.0)

    def lost(quantity: str, needs_activity: bool) -> int:
        n = 0
        for u in units:
            if needs_activity:
                if not valid_for(u, "KO"):
                    n += 1
                elif is_ko_active(u) and not valid_for(u, quantity):
                    n += 1
            elif not valid_for(u, quantity):
                n += 1
        return n

    per = {}
    for name, quantity, act in (("H1", "H1_any", False), ("H2", "H2", True),
                                ("H3", "H3", True), ("H3b", "H3b", True)):
        n = lost(quantity, act)
        frac = n / len(units) if units else 0.0
        per[name] = {"dropouts": n, "fraction": frac,
                     "label": "COMPROMISED" if frac > COMPROMISED_FRACTION else None}
    r0_out = sum(1 for e in episodes if e.get("r0_min_on_format", 1.0) < ON_FORMAT_FLOOR)
    r0_frac = r0_out / len(episodes) if episodes else 0.0
    per["R0"] = {"dropouts": r0_out, "fraction": r0_frac,
                 "label": "COMPROMISED" if r0_frac > COMPROMISED_FRACTION else None}
    return {
        "status": "VOID" if void else "OK",
        "max_capture_vs_plain": drift,
        "units": len(units),
        "dropouts": per,
        "min_on_format_any_pass": min((min_on_format(u) for u in units), default=None),
    }


def analyze(episodes: list[dict]) -> dict:
    integ = integrity(episodes)
    units = [u for e in episodes for u in e["units"]]
    r0_eps = [e for e in episodes if e.get("r0_min_on_format", 1.0) >= ON_FORMAT_FLOOR]
    flags = bistable_flags(r0_eps)
    ko_units = [u for u in units if valid_for(u, "KO")]
    res = {
        "n_episodes": len(episodes),
        "n_units": len(units),
        "integrity": integ,
        "accuracy": accuracy(episodes),
        "R0": phase0(r0_eps),
        "H1": h1(units, flags),
        "H2": h2(units),
        "H3": h3(units),
        "H3b": h3(units, variant="pre"),
        "carrier_locations": carrier_locations(
            [u for u in units if u.get("carriers") and valid_for(u, "H3")]),
        "carrier_locations_pre_answer": carrier_locations(
            [u for u in units if u.get("carriers") and valid_for(u, "H3b")], variant="pre"),
        "by_role": by_role(ko_units) if ko_units else {},
    }
    for h in ("R0", "H1", "H2", "H3", "H3b"):
        if integ["status"] == "VOID":
            res[h]["computed_outcome"] = res[h].get("outcome", res[h].get("verdict"))
            res[h]["outcome"] = "VOID"
        elif integ["dropouts"][h]["label"]:
            res[h]["label"] = integ["dropouts"][h]["label"]
    return res


def void_result(episodes: list[dict]) -> dict:
    """What a void-terminated rung reports: the failure, and nothing else."""
    bad = [e["episode"] for e in episodes if not e["consistency"].get("ok", True)]
    return {"n_episodes": len(episodes), "integrity": {"status": "VOID", "void_episodes": bad},
            **{h: {"outcome": "VOID"} for h in ("R0", "H1", "H2", "H3", "H3b")}}


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    import json
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


__all__ = [
    "ACTIVE", "ARR_SHARE_LINE", "ROUTE_LINE", "FIDELITY_LINE", "PRIMARY_K", "K_LADDER",
    "ON_FORMAT_FLOOR", "components", "valid_for", "is_active", "is_ko_active", "route_ratios",
    "fidelities", "cluster_bootstrap", "arrangement_share", "h1", "h2", "h3", "phase0", "analyze",
]
