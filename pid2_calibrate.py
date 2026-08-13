#!/usr/bin/env python3
"""Difficulty calibration pilot for PID-2. NOT a hypothesis test.

Purpose: find a task difficulty at which the model errs on a usable fraction of
items, so the detector has error variance to predict. PID-1 landed on the
pre-registered DEGENERATE outcome because `claude-opus-5` answered 2-hop
reading essentially perfectly.

DELIBERATE BLINDNESS. This pilot samples ONE order per item (canonical) and
therefore cannot compute instability at all. It measures accuracy and nothing
else. That is structural, not a promise: the quantity the hypothesis is about
is not observable from this script's data, so running it cannot inform, bias,
or pre-empt the test. Calibrating difficulty is ordinary experimental design;
peeking at the hypothesis would not be.

Items used here are discarded. PID-2 draws from a disjoint seed range.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from foil.execute import Executor, load_prices
from foil.pid2 import make_item2, permutations_for2, render_item2

ROOT = Path(__file__).parent
MODEL = "claude-opus-5"
ITEMS_PER_SETTING = 8
SAMPLES = 3
PILOT_SEEDS = range(1, 200)          # PID-2 will use seeds >= 1000
SETTINGS = [
    {"hops": 3, "near_misses": 2, "distractors": 3},
    {"hops": 3, "near_misses": 3, "distractors": 5},
    {"hops": 3, "near_misses": 3, "distractors": 5, "confusable": True},
    {"hops": 3, "near_misses": 3, "distractors": 8, "confusable": True},
]
#: Target band for base accuracy. Above it there are too few errors to predict;
#: below it the task is measuring something other than tracking.
TARGET = (0.50, 0.90)


def main() -> int:
    prices = load_prices(ROOT / "data" / "prices-anthropic.json")
    ex = Executor(
        store_path=ROOT / "runs" / f"pid2-calib-{MODEL}.jsonl",
        model=MODEL, prices=prices, concurrency=4, max_output_tokens=100_000,
    )
    print(f"── PID-2 calibration (accuracy only, canonical order) on {MODEL} ──", flush=True)
    out = []
    for cfg in SETTINGS:
        items, seen = [], iter(PILOT_SEEDS)
        while len(items) < ITEMS_PER_SETTING:
            s = next(seen)
            it = make_item2(s, **cfg)
            if it:
                items.append(it)
        correct = 0
        for it in items:
            canonical = permutations_for2(it, 1, seed=1)[0]
            recs = ex.sample(render_item2(it, canonical, MODEL), SAMPLES)
            answers = []
            for r in recs:
                try:
                    answers.append(json.loads(r["text"]).get("answer"))
                except Exception:
                    answers.append(None)
            got = [a for a in answers if a]
            modal = Counter(got).most_common(1)[0][0] if got else None
            correct += modal == it.answer
        acc = correct / len(items)
        inband = TARGET[0] <= acc <= TARGET[1]
        out.append({**cfg, "accuracy": acc, "in_band": inband})
        print(f"  hops={cfg['hops']} near={cfg['near_misses']} dist={cfg['distractors']}"
              f"{' confusable' if cfg.get('confusable') else ''}: "
              f"accuracy {acc:.2f} ({correct}/{len(items)})"
              f"{'  <- in band' if inband else ''}", flush=True)

    chosen = next((c for c in out if c["in_band"]), None)
    res = {"model": MODEL, "target_band": TARGET, "settings": out,
           "chosen": chosen, "ledger": ex.ledger.summary(),
           "note": "accuracy only; instability not measurable from this design"}
    p = ROOT / "runs" / "pid2-calibration.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\nchosen setting: {chosen}")
    if chosen is None:
        print("No setting landed in the target band. PID-2 is NOT pre-registered "
              "until one does; widen the difficulty ladder rather than the band.")
    print("ledger:", json.dumps(res["ledger"], indent=2))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
