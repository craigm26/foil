#!/usr/bin/env python3
"""PID runner — executes PREREGISTRATION-PID.md exactly.

All 40 items are collected before any test is computed (§5, no optional
stopping). The analysis runs once.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from foil.execute import Executor, load_prices
from foil.pid import make_item, permutations_for, render_item

ROOT = Path(__file__).parent

# Fixed by the pre-registration. Not adjustable from the command line, so a
# run cannot quietly become a different experiment.
K, M, N = 40, 4, 5
MODEL = "claude-opus-5"
ALPHA, MIN_RISK_DIFF, MIN_MARGIN = 0.01, 0.30, 8
SEED = 7


def modal(answers: list[str | None], options: tuple[str, ...]) -> str | None:
    """Most frequent parsed answer; None if fewer than 3 of N parsed."""
    got = [a for a in answers if a]
    if len(got) < 3:
        return None
    c = Counter(got)
    top = max(c.values())
    for o in options:  # pre-registered tie-break: answer-set order
        if c[o] == top:
            return o
    return None


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(X >= a) for the top-left cell under the hypergeometric null."""
    n = a + b + c + d
    row, col = a + b, a + c
    return sum(
        math.comb(col, i) * math.comb(n - col, row - i)
        for i in range(a, min(col, row) + 1)
    ) / math.comb(n, row)


def main() -> int:
    prices = load_prices(ROOT / "data" / "prices-anthropic.json")
    ex = Executor(
        store_path=ROOT / "runs" / f"pid-{MODEL}.jsonl",
        model=MODEL, prices=prices, concurrency=4, max_output_tokens=200_000,
    )

    items = [i for i in (make_item(s) for s in range(1, 400)) if i][:K]
    assert len(items) == K, f"only {len(items)} determinate items"
    print(f"── PID: {K} items x {M} permutations x {N} samples on {MODEL} ──", flush=True)

    records = []
    for idx, it in enumerate(items, 1):
        perms = permutations_for(it, M, seed=SEED + idx)
        modals = []
        for p in perms:
            recs = ex.sample(render_item(it, p, MODEL), N)
            answers = []
            for r in recs:
                try:
                    answers.append(json.loads(r["text"]).get("answer"))
                except Exception:
                    answers.append(None)
                    ex.ledger.unparseable += 1
            modals.append(modal(answers, it.options))
        rec = {
            "item_id": it.item_id,
            "truth": it.answer,
            "modals": modals,
            "dropout": any(m is None for m in modals),
            "unstable": len(set(modals)) > 1,
            "correct": modals[0] == it.answer,
        }
        records.append(rec)
        print(f"  {idx}/{K} {it.item_id}: modal={modals[0]} truth={it.answer} "
              f"unstable={rec['unstable']}", flush=True)

    kept = [r for r in records if not r["dropout"]]
    a = sum(1 for r in kept if r["unstable"] and not r["correct"])      # unstable & wrong
    b = sum(1 for r in kept if r["unstable"] and r["correct"])          # unstable & right
    c = sum(1 for r in kept if not r["unstable"] and not r["correct"])  # stable & wrong
    d = sum(1 for r in kept if not r["unstable"] and r["correct"])      # stable & right

    p = fisher_one_sided(a, b, c, d) if (a + b) and (c + d) else float("nan")
    r_u = a / (a + b) if a + b else float("nan")
    r_s = c / (c + d) if c + d else float("nan")
    rd = r_u - r_s

    degenerate = (a + b) < MIN_MARGIN or (c + d) < MIN_MARGIN
    supported = (not degenerate) and p < ALPHA and rd >= MIN_RISK_DIFF
    outcome = ("DEGENERATE" if degenerate
               else "SUPPORTED" if supported else "NOT REPLICATED")

    res = {
        "preregistration": "PREREGISTRATION-PID.md",
        "model": MODEL, "items": K, "permutations": M, "samples": N,
        "table": {"unstable_wrong": a, "unstable_right": b,
                  "stable_wrong": c, "stable_right": d},
        "dropouts": len(records) - len(kept),
        "p_one_sided": p,
        "err_rate_unstable": r_u, "err_rate_stable": r_s, "risk_difference": rd,
        "accuracy": (b + d) / len(kept) if kept else float("nan"),
        "instability_rate": (a + b) / len(kept) if kept else float("nan"),
        "criteria": {"alpha": ALPHA, "min_risk_difference": MIN_RISK_DIFF,
                     "min_margin": MIN_MARGIN},
        "outcome": outcome,
        "records": records,
        "ledger": ex.ledger.summary(),
    }
    out = ROOT / "runs" / f"pid-result-{MODEL}.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"""
── PID result ────────────────────────────────────────
                wrong   right
  unstable      {a:>5}   {b:>5}
  stable        {c:>5}   {d:>5}

  dropouts            {res['dropouts']}
  accuracy            {res['accuracy']:.3f}
  instability rate    {res['instability_rate']:.3f}
  err rate | unstable {r_u:.3f}
  err rate | stable   {r_s:.3f}
  risk difference     {rd:.3f}   (need >= {MIN_RISK_DIFF})
  p (one-sided)       {p:.5f}   (need < {ALPHA})
  margins             {a + b} unstable / {c + d} stable  (need >= {MIN_MARGIN} each)

  OUTCOME             {outcome}
""")
    print("ledger:", json.dumps(res["ledger"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
