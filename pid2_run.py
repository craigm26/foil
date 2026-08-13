#!/usr/bin/env python3
"""PID-2 runner — executes PREREGISTRATION-PID2.md exactly.

All 48 episodes are collected before any statistic is computed (§5, no optional
stopping). The analysis runs once.

Delivery is via the Batches API at 50% of list. Cache seeding is deliberately
NOT attempted: the scout system prompt is ~171 tokens against Opus-class's
512-token minimum, so a `cache_control` marker here would be inert. Claiming a
saving that cannot occur is the exact failure this project already made once.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from foil.batch import BatchExecutor
from foil.env3 import make_episode_v3
from foil.execute import load_prices
from foil.nulls import Arm  # noqa: F401  (kept for parity of vocabulary)
from foil.render import ForkKey, Operator, parse_action, render
from foil.stats import distribution, total_variation

ROOT = Path(__file__).parent

# Fixed by the pre-registration; not overridable from the command line so a run
# cannot quietly become a different experiment.
K, M, N = 48, 6, 10
SEED_BASE = 1000                      # disjoint from v2/v3 (seeds 1-12)
MODEL = "claude-opus-5"
ALPHA, MIN_RISK_DIFF, MIN_MARGIN = 0.01, 0.30, 10
UNSTABLE_TV = 0.5
SENSITIVITY = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def orderings(sources: tuple[str, ...], m: int, seed: int) -> list[tuple[str, ...]]:
    import itertools, random
    canonical = tuple(sources)
    rng = random.Random(seed)
    pool = [p for p in itertools.permutations(sources) if p != canonical]
    rng.shuffle(pool)
    return [canonical] + pool[: m - 1]


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    n, row, col = a + b + c + d, a + b, a + c
    return sum(
        math.comb(col, i) * math.comb(n - col, row - i)
        for i in range(a, min(col, row) + 1)
    ) / math.comb(n, row)


def clopper_pearson_upper(x: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper bound on a binomial rate.

    For x=0 this is the closed form 1 - alpha**(1/n); the general case is
    solved by bisection on the binomial CDF, avoiding a scipy dependency.
    """
    if n == 0:
        return float("nan")
    if x == 0:
        return 1 - alpha ** (1 / n)

    def cdf(p: float) -> float:
        return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(x + 1))

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> int:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("no ANTHROPIC_API_KEY", file=sys.stderr)
        return 2
    prices = load_prices(ROOT / "data" / "prices-anthropic.json")
    ex = BatchExecutor(ROOT / "runs" / f"pid2-{MODEL}.jsonl", MODEL, key, prices)

    episodes = []
    for i in range(K):
        ep = make_episode_v3(SEED_BASE + i)
        if ep is None:
            print(f"no admissible episode for seed {SEED_BASE + i}", file=sys.stderr)
            return 2
        episodes.append(ep)
    print(f"── PID-2: {K} episodes x {M} orderings x {N} samples on {MODEL} ──", flush=True)

    requests, index = [], {}
    for ep in episodes:
        perms = orderings(ep.source_ids, M, seed=SEED_BASE)
        for oi, order in enumerate(perms):
            body = render(
                ep,
                ForkKey(ep.episode_id, 0, frozenset(ep.source_ids),
                        Operator.NULL, order, MODEL),
            )
            for si in range(N):
                # custom_id must match ^[a-zA-Z0-9_-]{1,64}$ -- no pipes.
                cid = f"{ep.episode_id}_{oi}_{si}"
                requests.append((cid, body))
                index[cid] = (ep.episode_id, oi)

    todo = sum(1 for cid, _ in requests if not ex.have(cid))
    print(f"  {len(requests)} requests, {todo} not yet collected", flush=True)
    if todo:
        ids = ex.submit(requests)
        ex.wait(ids)
        ex.collect(ids)

    # ---- assemble ----
    by = {}
    for cid, _ in requests:
        eid, oi = index[cid]
        recs = ex.get(cid)
        if recs:
            by.setdefault((eid, oi), []).append(recs[0]["text"])

    records, dropouts = [], 0
    for ep in episodes:
        modals, dists = [], []
        ok = True
        for oi in range(M):
            texts = by.get((ep.episode_id, oi), [])
            acts = [parse_action(t, ep.actions)[0] for t in texts]
            got = [a for a in acts if a]
            if len(got) < 6:
                ok = False
                break
            c = Counter(got)
            top = max(c.values())
            modals.append(next(a for a in ep.actions if c[a] == top))
            dists.append(distribution(acts, ep.actions))
        if not ok:
            dropouts += 1
            continue
        max_tv = max(total_variation(x, y) for x, y in combinations(dists, 2))
        records.append({
            "episode_id": ep.episode_id, "truth": ep.correct_action,
            "modals": modals, "max_order_tv": max_tv,
            "unstable": max_tv > UNSTABLE_TV,
            "correct": modals[0] == ep.correct_action,
        })

    def table(thr: float):
        a = sum(1 for r in records if r["max_order_tv"] > thr and not r["correct"])
        b = sum(1 for r in records if r["max_order_tv"] > thr and r["correct"])
        c = sum(1 for r in records if r["max_order_tv"] <= thr and not r["correct"])
        d = sum(1 for r in records if r["max_order_tv"] <= thr and r["correct"])
        return a, b, c, d

    a, b, c, d = table(UNSTABLE_TV)
    p = fisher_one_sided(a, b, c, d) if (a + b) and (c + d) else float("nan")
    r_u = a / (a + b) if a + b else float("nan")
    r_s = c / (c + d) if c + d else float("nan")
    rd = r_u - r_s
    degenerate = (a + b) < MIN_MARGIN or (c + d) < MIN_MARGIN
    outcome = ("DEGENERATE" if degenerate
               else "SUPPORTED" if (p < ALPHA and rd >= MIN_RISK_DIFF)
               else "NOT REPLICATED")
    fn_upper = clopper_pearson_upper(c, c + d)

    res = {
        "preregistration": "PREREGISTRATION-PID2.md", "model": MODEL,
        "episodes": K, "orderings": M, "samples": N, "seed_base": SEED_BASE,
        "table": {"unstable_wrong": a, "unstable_right": b,
                  "stable_wrong": c, "stable_right": d},
        "dropouts": dropouts, "p_one_sided": p,
        "err_rate_unstable": r_u, "err_rate_stable": r_s, "risk_difference": rd,
        "false_negative_rate": r_s, "false_negative_upper95": fn_upper,
        "accuracy": (b + d) / len(records) if records else float("nan"),
        "instability_rate": (a + b) / len(records) if records else float("nan"),
        "criteria": {"alpha": ALPHA, "min_risk_difference": MIN_RISK_DIFF,
                     "min_margin": MIN_MARGIN, "unstable_tv": UNSTABLE_TV},
        "outcome": outcome,
        "threshold_sensitivity": [
            {"threshold": t, "table": table(t),
             "p": fisher_one_sided(*table(t))
                  if (table(t)[0] + table(t)[1]) and (table(t)[2] + table(t)[3])
                  else None}
            for t in SENSITIVITY
        ],
        "records": records, "ledger": ex.summary(),
    }
    out = ROOT / "runs" / f"pid2-result-{MODEL}.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"""
── PID-2 result ──────────────────────────────────────
                wrong   right
  unstable      {a:>5}   {b:>5}
  stable        {c:>5}   {d:>5}

  dropouts            {dropouts}
  accuracy            {res['accuracy']:.3f}
  instability rate    {res['instability_rate']:.3f}
  err | unstable      {r_u:.3f}
  err | stable        {r_s:.3f}   (95% upper bound {fn_upper:.3f})
  risk difference     {rd:.3f}   (need >= {MIN_RISK_DIFF})
  p (one-sided)       {p:.5f}   (need < {ALPHA})
  margins             {a + b} unstable / {c + d} stable  (need >= {MIN_MARGIN})

  OUTCOME             {outcome}

threshold sensitivity (a,b,c,d / p):""")
    for row in res["threshold_sensitivity"]:
        pv = "n/a" if row["p"] is None else f"{row['p']:.5f}"
        print(f"   tv>{row['threshold']:.1f}: {row['table']}  p={pv}")
    print("\nledger:", json.dumps(res["ledger"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
