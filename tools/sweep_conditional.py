#!/usr/bin/env python3
"""EXPLORATORY: bistability conditional on canonical correctness.

Not pre-registered. Zero model calls -- re-analyses runs/sweep-result.json.

The SWEEP left a confound standing: bistability might just restate task
difficulty. This split asks, per model, whether episodes the model gets RIGHT
at the canonical ordering flip under reordering at a different rate than
episodes it gets WRONG. Episodes are regenerated deterministically from the
sweep's seed base to recover each episode's correct action.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root

from evalgate.power import wilson
from foil.env3 import make_episode_v3

ROOT = Path(__file__).parent.parent
SEED_BASE, EPISODES = 7000, 12   # must match sweep_run.py


def truth_map() -> dict[str, str]:
    out, seed = {}, SEED_BASE
    while len(out) < EPISODES:
        ep = make_episode_v3(seed)
        seed += 1
        if ep:
            out[ep.episode_id] = ep.correct_action
    return out


def main() -> None:
    d = json.loads((ROOT / "runs" / "sweep-result.json").read_text())
    truths = truth_map()

    print(f"{'model':20} {'right: bi/n (rate, CI)':>26} {'wrong: bi/n (rate, CI)':>26}")
    rows = []
    for r in d["results"]:
        right = {"bi": 0, "n": 0}
        wrong = {"bi": 0, "n": 0}
        for e in r["per_episode"]:
            canonical_mode = e["modes"][0]
            bucket = right if canonical_mode == truths[e["episode"]] else wrong
            bucket["n"] += 1
            bucket["bi"] += bool(e["bistable"])
        def fmt(b):
            if not b["n"]:
                return "-- 0 episodes --"
            lo, hi = wilson(b["bi"], b["n"])
            return f"{b['bi']}/{b['n']} ({b['bi']/b['n']:.2f} [{lo:.2f},{hi:.2f}])"
        print(f"{r['model']:20} {fmt(right):>26} {fmt(wrong):>26}")
        rows.append({"model": r["model"], "canonical_right": right,
                     "canonical_wrong": wrong})

    pooled_r = {"bi": sum(x["canonical_right"]["bi"] for x in rows),
                "n": sum(x["canonical_right"]["n"] for x in rows)}
    pooled_w = {"bi": sum(x["canonical_wrong"]["bi"] for x in rows),
                "n": sum(x["canonical_wrong"]["n"] for x in rows)}
    lr, hr = wilson(pooled_r["bi"], pooled_r["n"])
    lw, hw = wilson(pooled_w["bi"], pooled_w["n"])
    print(f"\npooled  canonical-right {pooled_r['bi']}/{pooled_r['n']} "
          f"= {pooled_r['bi']/pooled_r['n']:.2f} [{lr:.2f},{hr:.2f}]")
    print(f"pooled  canonical-wrong {pooled_w['bi']}/{pooled_w['n']} "
          f"= {pooled_w['bi']/pooled_w['n']:.2f} [{lw:.2f},{hw:.2f}]")
    print("\nExploratory. Pooling crosses models; per-model cells are tiny.")


if __name__ == "__main__":
    main()
