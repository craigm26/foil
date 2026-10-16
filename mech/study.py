"""MECH-1 measurement: every arm the pre-registration names, for one episode.

Per episode:

    N1      full coalition at 6 orderings (canonical + 5 seeded)   R0
    N2      paraphrased reports, canonical order                     R0
    REF     Ω_null per source, canonical order                       R0
    per source s, at the episode's ARMS ORDERING (the canonical order rotated
    by seed mod 4, so across the 48 episodes every source sits in every slot
    exactly 12 times):
      null      Ω_null, visible                    FOIL's primary operator
      null_ko   Ω_null, its slot knocked out       information gone, slot resized
      shift     s visible, later tokens at Ω_null's positions   arrangement only
      del       Ω_del                              slot removed
      ko        s knocked out in place             information gone, arrangement fixed
      ko_sum    s's punctuation knocked out        H2
      ko_con    s's other tokens knocked out       H2
      ko_inner  punctuation except the last token  H2, reported regardless
      ko_final  the last token only                H2, reported regardless
      carriers  when |ko - full| >= ACTIVE         H3 / H3b

The arms ordering exists because the review of this design found canonical
order confounds role with slot: the liar was always third and the last
honest scout always absorbed the paragraph break before the query. A cyclic
rotation is a Latin square over (source, slot), so the confound is balanced
out rather than merely varied.

Orderings are chosen by foil.nulls.run_nulls's procedure. Nothing is sampled:
every arm is one exact forward pass.
"""

from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import numpy as np

from foil.env3 import make_episode_v3
from foil.ir import Episode
from foil.nulls import paraphrased_episode
from foil.render import Operator

from .analysis import ACTIVE, K_LADDER, PRIMARY_K
from .basis import NeuronBasis
from .local import LocalPrompt, knock, position_class, render_local, shifted_positions
from .whitebox import WhiteBox, regime_name

ORDERINGS = 6
#: The capture run and the plain run of the same prompt must agree this
#: closely, or hooks are changing the computation they claim only to observe.
CONSISTENCY_TOL = 1e-5


def orderings(sources: tuple[str, ...], m: int, seed: int) -> list[tuple[str, ...]]:
    """Canonical first, then m-1 distinct permutations: foil.nulls.run_nulls's
    procedure, restated only because that function also samples."""
    canonical = tuple(sources)
    rng = random.Random(seed)
    pool = [p for p in itertools.permutations(sources) if p != canonical]
    rng.shuffle(pool)
    return [canonical] + pool[: max(0, m - 1)]


ROTATIONS = 4


def arms_ordering(sources: tuple[str, ...], seed: int) -> tuple[str, ...]:
    """The registered ordering for the per-source arms: the canonical order
    rotated left by seed mod 4."""
    r = seed % ROTATIONS
    return tuple(sources[r:]) + tuple(sources[:r])


def _p(r) -> list[float]:
    return [float(x) for x in r.probs]


def _candidates(prompt: LocalPrompt, source_id: str) -> np.ndarray:
    """Positions a carrier may occupy: strictly after the span, through the
    decision position. Under a knockout of s nothing at or before the span
    changes, so nothing there can carry s anywhere."""
    span = prompt.spans[source_id]
    return np.arange(span.last + 1, prompt.decision_pos + 1)


def find_carriers(
    wb: WhiteBox,
    run,
    base: LocalPrompt,
    source_id: str,
    full_run,
    ko_run,
    info: float,
    basis,
    k_ladder=K_LADDER,
) -> dict:
    """H3 and H3b, with their controls. Fully deterministic.

    Selection: rank sites (layer, position, unit) by attribution --
    Δcoefficient (ko - full) x d P(correct)/d coefficient at the factual run --
    keeping only sites pushing P(correct) the way the knockout did. Test:
    patch the top k from the knockout run into the factual run. Selection is a
    linear estimate; the patch is the test, so a poor estimate costs power,
    never validity.

    Matched control (registered, §3.3): each selected site swapped for the
    unselected unit in the SAME (layer, position) cell whose |Δ| is nearest.
    Location and size of change held fixed; only which unit varies. If the
    chosen units beat it, the units themselves matter, not merely where and
    how much things changed.
    Top-|Δ| control (reported): the 16 sites that changed most, ignoring the
    gradient.

    H3 searches every position after the span. H3b repeats the k = PRIMARY_K
    search with the assistant turn excluded (prefill and decision position).
    """
    pos = _candidates(base, source_id)
    pre_answer = np.array([base.region[int(p)] != "answer" for p in pos])
    direction = 1.0 if info > 0 else -1.0
    layers = basis.layers(wb.n_layers)
    f_full, f_ko, scores = {}, {}, {}
    for l in layers:
        ff = basis.coeffs(l, full_run.captures[(basis.read_site, l)][pos]).astype(np.float32)
        fk = basis.coeffs(l, ko_run.captures[(basis.read_site, l)][pos]).astype(np.float32)
        g = basis.coeff_grads(l, full_run.grads[(basis.write_site, l)][pos]).astype(np.float32)
        f_full[l], f_ko[l] = ff, fk
        scores[l] = (fk - ff) * g
    n_units = scores[layers[0]].shape[1]
    flat = np.concatenate([scores[l].ravel() for l in layers])
    absd = np.abs(np.concatenate([(f_ko[l] - f_full[l]).ravel() for l in layers]))
    per_layer = len(pos) * n_units
    allowed_pre = np.tile(np.repeat(pre_answer, n_units), len(layers))
    aligned = flat * direction

    def unflat(i: int) -> tuple[int, int, int]:
        li, rem = divmod(int(i), per_layer)
        pi, u = divmod(rem, n_units)
        return layers[li], pi, u

    def top_by(values, mask, kmax, positive_only):
        cand = values if mask is None else np.where(mask, values, -np.inf)
        n_ok = int((cand > 0).sum()) if positive_only else int(np.isfinite(cand).sum())
        take = min(kmax, n_ok)
        if not take:
            return np.array([], dtype=int), n_ok
        idx = np.argpartition(-cand, take - 1)[:take]
        return idx[np.argsort(-cand[idx], kind="stable")], n_ok

    def matched(selected) -> tuple[list[int], float | None]:
        """Same-cell, nearest-|Δ| substitutes, and the median ratio
        |Δ_substitute| / |Δ_selected|. Attribution favours each cell's
        largest change, so substitutes tend to be smaller; the ratio is
        recorded because the outcome rule needs it (§7.3)."""
        used = set(int(i) for i in selected)
        out, ratios = [], []
        for i in selected:
            cell = (int(i) // n_units) * n_units
            gap = np.abs(absd[cell:cell + n_units] - absd[int(i)])
            for j in np.argsort(gap, kind="stable"):
                j = cell + int(j)
                if j not in used:
                    used.add(j)
                    out.append(j)
                    if absd[int(i)] > 0:
                        ratios.append(float(absd[j] / absd[int(i)]))
                    break
        return out, (float(np.median(ratios)) if ratios else None)

    def patch_for(idx) -> list:
        groups: dict[int, tuple[list, list, list, list]] = {}
        for i in idx:
            l, pi, u = unflat(i)
            g = groups.setdefault(l, ([], [], [], []))
            g[0].append(int(pos[pi])); g[1].append(u)
            g[2].append(float(f_ko[l][pi, u])); g[3].append(float(f_full[l][pi, u]))
        out = []
        for l, (ps, us, src, dst) in groups.items():
            out.extend(basis.patches(l, ps, us, src, dst))
        return out

    c = base.correct_index
    masses: dict[str, float] = {}

    def p_after(idx, label: str) -> float:
        if not len(idx):
            masses[label] = full_run.on_format
            return float(full_run.probs[c])
        r = run(base, patches=patch_for(idx))
        masses[label] = r.on_format
        return float(r.probs[c])

    toks = wb.tok.convert_ids_to_tokens([base.input_ids[int(p)] for p in pos])

    def describe(idx) -> list[dict]:
        out = []
        for i in idx:
            l, pi, u = unflat(i)
            p_abs = int(pos[pi])
            text = base.text[base.offsets[p_abs][0]:base.offsets[p_abs][1]]
            d = {
                "layer": l, "position": p_abs, "token": toks[pi],
                "position_class": position_class(base, p_abs, text),
                "unit": u, "delta": float(f_ko[l][pi, u] - f_full[l][pi, u]),
                "score": float(scores[l][pi, u]),
            }
            if basis.name == "neurons" and full_run.regimes and ko_run.regimes:
                d["regime_full"] = regime_name(full_run.regimes[l][p_abs, u])
                d["regime_ko"] = regime_name(ko_run.regimes[l][p_abs, u])
            out.append(d)
        return out

    top, n_aligned = top_by(aligned, None, max(k_ladder), True)
    by_k = {k: p_after(top[:k], f"car_k{k}") for k in k_ladder}
    shortfall = {k: int(len(top)) for k in k_ladder if len(top) < k}
    top16 = top[:PRIMARY_K]
    m16, ratio16 = matched(top16)
    top_abs, _ = top_by(absd, None, PRIMARY_K, False)

    top_pre, n_aligned_pre = top_by(aligned, allowed_pre, PRIMARY_K, True)
    m_pre, ratio_pre = matched(top_pre)
    top_abs_pre, _ = top_by(absd, allowed_pre, PRIMARY_K, False)
    pre = {
        "k": PRIMARY_K,
        "p": p_after(top_pre, "pre"),
        "matched": p_after(m_pre, "pre_matched"),
        "matched_size_ratio": ratio_pre,
        "top_abs": p_after(top_abs_pre, "pre_top_abs"),
        "n_candidates": int(allowed_pre.sum()),
        "n_aligned": n_aligned_pre,
        "shortfall": int(len(top_pre)) if len(top_pre) < PRIMARY_K else None,
        "top": describe(top_pre),
    }
    return {
        "basis": basis.name,
        "n_layers": wb.n_layers,
        "n_candidates": int(flat.size),
        "n_aligned": n_aligned,
        "by_k": by_k,
        "matched": p_after(m16, "car_matched"),
        "matched_size_ratio": ratio16,
        "top_abs": p_after(top_abs, "car_top_abs"),
        "shortfall": shortfall,
        "top": describe(top16),
        "pre_answer": pre,
        "on_format": masses,
    }


def measure_episode(wb: WhiteBox, ep: Episode, seed: int, basis=None) -> dict:
    t0 = time.time()
    tok = wb.tok
    basis = basis or NeuronBasis()
    sources = tuple(ep.source_ids)
    full = frozenset(sources)
    passes = 0

    def run(prompt, **kw):
        nonlocal passes
        r = wb.run(prompt, **kw)
        passes += 1
        return r

    rec: dict = {"episode": ep.episode_id, "seed": seed,
                 "actions": list(ep.actions), "correct_index": ep.actions.index(ep.correct_action)}
    c = rec["correct_index"]
    r0_masses: list[float] = []

    orders = orderings(sources, ORDERINGS, seed)
    plain_by_order = {}
    rec["n1"] = []
    for o in orders:
        pr = render_local(tok, ep, o, full, Operator.NULL)
        r = run(pr)
        r0_masses.append(r.on_format)
        plain_by_order[o] = r
        rec["n1"].append({"order": list(o), "probs": _p(r), "on_format": r.on_format, "digest": pr.digest()})
    para = render_local(tok, paraphrased_episode(ep), sources, full, Operator.NULL)
    r = run(para)
    r0_masses.append(r.on_format)
    rec["n2"] = {"probs": _p(r), "on_format": r.on_format, "digest": para.digest()}

    arms_order = arms_ordering(sources, seed)
    rec["arms_order"] = list(arms_order)
    base = render_local(tok, ep, arms_order, full, Operator.NULL)
    plain = plain_by_order.get(arms_order) or run(base)
    sites = {basis.read_site, basis.write_site}
    want_regimes = basis.name == "neurons"
    full_run = run(base, capture=sites, grad=True, regimes=want_regimes)
    drift = float(np.max(np.abs(full_run.probs - plain.probs)))
    rec["consistency"] = {"capture_vs_plain_max_abs": drift, "ok": drift <= CONSISTENCY_TOL}
    pf = float(full_run.probs[c])

    rec["ref"] = {}
    rec["units"] = []
    for slot, s in enumerate(arms_order):
        null_p = render_local(tok, ep, arms_order, full - {s}, Operator.NULL)
        del_p = render_local(tok, ep, arms_order, full - {s}, Operator.DEL)
        sm = base.spans[s]
        delta = len(null_p.spans[s].tokens) - len(sm.tokens)
        arms = {
            "null": run(null_p),
            "null_ko": run(null_p, knockouts=[knock(null_p, s, "all")]),
            "shift": run(base, position_ids=shifted_positions(base, s, delta)),
            "del": run(del_p),
            "ko": run(base, knockouts=[knock(base, s, "all")], capture={basis.read_site},
                      regimes=want_regimes),
            "ko_sum": run(base, knockouts=[knock(base, s, "summary")]),
            "ko_con": run(base, knockouts=[knock(base, s, "content")]),
            "ko_inner": run(base, knockouts=[knock(base, s, "inner")]) if len(sm.summary) > 1 else None,
            "ko_final": run(base, knockouts=[knock(base, s, "final")]),
        }
        live = {k: v for k, v in arms.items() if v is not None}
        unit = {
            "episode": ep.episode_id, "source": s, "slot": slot,
            "last_slot": slot == len(arms_order) - 1,
            "role": "liar" if ep.reliability.get(s, 1.0) <= 0.0 else "honest",
            "span_tokens": len(sm.tokens),
            "null_slot_tokens": len(null_p.spans[s].tokens),
            "position_shift": delta,
            "p": {"full": pf, **{k: float(v.probs[c]) for k, v in live.items()}},
            "dist": {k: _p(v) for k, v in live.items()},
            "on_format": {"full": full_run.on_format, **{k: v.on_format for k, v in live.items()}},
            "carriers": None,
        }
        info = unit["p"]["ko"] - pf
        if abs(info) >= ACTIVE:
            unit["carriers"] = find_carriers(wb, run, base, s, full_run, arms["ko"], info, basis)
            unit["on_format"].update(unit["carriers"].pop("on_format"))
        rec["units"].append(unit)
        del arms, live

    # REF for R0 is FOIL's: single-source Ω_null at the CANONICAL order.
    for s in sources:
        if arms_order == sources:
            u = next(u for u in rec["units"] if u["source"] == s)
            rec["ref"][s] = u["dist"]["null"]
            r0_masses.append(u["on_format"]["null"])
        else:
            r = run(render_local(tok, ep, sources, full - {s}, Operator.NULL))
            rec["ref"][s] = _p(r)
            r0_masses.append(r.on_format)
    rec["r0_min_on_format"] = min(r0_masses)
    rec["passes"] = passes
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def _code(prov: dict | None) -> tuple:
    """Which code wrote a row: (git commit, dirty flag)."""
    prov = prov or {}
    return prov.get("git_commit"), prov.get("git_dirty")


def run_study(wb: WhiteBox, seeds, out_path: str | Path, *, basis=None, log=print,
              provenance: dict | None = None) -> list[dict]:
    """Measure every registered episode, appending one JSON line each.

    Resumable: an episode already present with the same identity is skipped,
    the idempotence foil.execute gives sampled arms. A line from a different
    model, revision, dtype, device, hardware or library version is an error,
    never silently mixed in; so is one written by different code (another git
    commit, or a dirty tree), and so is a second line for one episode. An
    episode whose hooks altered the computation is written to disk -- the
    void is part of the record -- and then the run stops: every later number
    would share the fault.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ident = wb.identity()
    key = {k: ident[k] for k in WhiteBox.IDENTITY_KEYS}
    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if {k: row["identity"].get(k) for k in key} != key:
                raise RuntimeError(f"{out_path} holds episodes from {row['identity']}; refusing to mix")
            if provenance is not None and _code(row.get("provenance")) != _code(provenance):
                raise RuntimeError(f"{out_path}: {row['episode']} was measured by other code "
                                   f"({_code(row.get('provenance'))}, now {_code(provenance)}); "
                                   "refusing to mix")
            if not row["consistency"].get("ok", True):
                raise RuntimeError(f"{out_path}: {row['episode']} is VOID (hooks altered the "
                                   "computation); the rung does not resume")
            if row["episode"] in done:
                raise RuntimeError(f"{out_path}: two rows for {row['episode']}; refusing to resume")
            done[row["episode"]] = row
    rows = []
    for i, seed in enumerate(seeds):
        ep = make_episode_v3(seed)
        if ep is None:
            raise RuntimeError(f"env3 produced no admissible episode for seed {seed}")
        if ep.episode_id in done:
            rows.append(done[ep.episode_id])
            continue
        rec = measure_episode(wb, ep, seed, basis=basis)
        rec["identity"] = ident
        if provenance is not None:
            rec["provenance"] = provenance
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        rows.append(rec)
        log(f"  episode {i + 1}/{len(seeds)} {ep.episode_id}: {rec['passes']} passes, {rec['seconds']}s")
        if not rec["consistency"]["ok"]:
            raise RuntimeError(f"{ep.episode_id}: hooks altered the computation "
                               f"({rec['consistency']['capture_vs_plain_max_abs']:g}); the rung is void")
    return rows
