#!/usr/bin/env python3
"""SWEEP — order sensitivity across model generations, on a fixed environment.

Pre-registered in PREREGISTRATION-SWEEP.md, committed before any data existed.
Descriptive measurement. There is no SUPPORTED outcome.

    ./run.sh sweep_run.py                  # all models
    ./run.sh sweep_run.py --models a,b     # a subset
    ./run.sh sweep_run.py --dry-run        # cost projection, zero calls
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

from evalgate.power import wilson

from foil.env3 import make_episode_v3
from foil.execute import Executor, load_prices, cost_usd
from foil.render import ForkKey, Operator, render, parse_action
from foil.stats import distribution, total_variation

ROOT = Path(__file__).parent

# Fixed by the pre-registration. Not tuned per model.
EPISODES, ORDERS, SAMPLES = 12, 6, 20
SEED_BASE = 7000
PERM_SEED = 0

MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5", "claude-sonnet-4-6", "claude-sonnet-5",
    "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7",
    "claude-opus-4-8", "claude-opus-5",
]


def orderings(sources: tuple[str, ...], k: int, seed: int) -> list[tuple[str, ...]]:
    """Canonical first, then a seeded sample. Identical across models."""
    canonical = tuple(sources)
    pool = [p for p in itertools.permutations(sources) if p != canonical]
    random.Random(seed).shuffle(pool)
    return [canonical] + pool[: max(0, k - 1)]


def measure_model(model: str, prices, key: str, dry_run: bool) -> dict:
    ex = Executor(store_path=ROOT / "runs" / f"sweep-{model}.jsonl",
                  model=model, prices=prices, api_key=key)
    episodes, seed = [], SEED_BASE
    while len(episodes) < EPISODES:          # identical episodes for every model
        ep = make_episode_v3(seed)
        seed += 1
        if ep:
            episodes.append(ep)

    per_episode, bistable, invert_max = [], 0, 0.0
    tv_all, correct, total = [], 0, 0

    for ep in episodes:
        perms = orderings(ep.source_ids, ORDERS, PERM_SEED)
        dists, modes = [], []
        for oi, order in enumerate(perms):
            fk = ForkKey(episode_id=ep.episode_id, decision_index=0,
                         coalition=frozenset(ep.source_ids),
                         operator=Operator.NULL, render_order=order, model=model)
            recs = ex.sample(render(ep, fk), SAMPLES, dry_run=dry_run)
            if dry_run:
                continue
            acts = []
            for r in recs:
                a, _ = parse_action(r["text"], ep.actions)
                acts.append(a)
            d = distribution(acts, ep.actions)
            dists.append(d)
            modes.append(ep.actions[int(d.argmax())] if d.sum() else None)
            if oi == 0:
                correct += sum(1 for a in acts if a == ep.correct_action)
                total += len(acts)
        if dry_run:
            continue
        pair_tv = [total_variation(dists[i], dists[j])
                   for i in range(len(dists)) for j in range(i + 1, len(dists))]
        tv_all.extend(pair_tv)
        mx = max(pair_tv) if pair_tv else 0.0
        invert_max = max(invert_max, mx)
        bi = len(set(m for m in modes if m is not None)) > 1
        bistable += bi
        per_episode.append({"episode": ep.episode_id, "bistable": bi,
                            "max_tv": mx, "modes": modes})

    if dry_run:
        return {"model": model, "projected_calls": EPISODES * ORDERS * SAMPLES}

    tv_all.sort()
    n = len(per_episode)
    lo, hi = wilson(bistable, n)
    return {
        "model": model,
        "episodes": n,
        "bistability_rate": bistable / n if n else 0.0,
        "bistability_ci95": [lo, hi],
        "t_null_p95": tv_all[min(len(tv_all) - 1, int(len(tv_all) * 0.95))] if tv_all else 0.0,
        "max_inversion": invert_max,
        "accuracy_canonical": correct / total if total else 0.0,
        "per_episode": per_episode,
        "usage": ex.ledger.summary(),
        "cost_usd": cost_usd(ex.ledger.usage, model, prices),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and not args.dry_run:
        print("no ANTHROPIC_API_KEY", file=sys.stderr)
        return 2
    prices = load_prices(ROOT / "data" / "prices-anthropic.json")

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"── SWEEP: {len(wanted)} models x {EPISODES} episodes x {ORDERS} "
          f"orderings x {SAMPLES} samples ──", flush=True)
    print(f"   {EPISODES*ORDERS*SAMPLES} calls per model, "
          f"{len(wanted)*EPISODES*ORDERS*SAMPLES} total\n", flush=True)

    results, unavailable = [], []
    for m in wanted:
        print(f"  {m} ...", end="", flush=True)
        try:
            r = measure_model(m, prices, key, args.dry_run)
            results.append(r)
            if args.dry_run:
                print(f" projected {r['projected_calls']} calls", flush=True)
            else:
                lo, hi = r["bistability_ci95"]
                print(f" bistable {r['bistability_rate']:.2f} "
                      f"[{lo:.2f},{hi:.2f}]  T_null {r['t_null_p95']:.3f}  "
                      f"acc {r['accuracy_canonical']:.3f}  "
                      f"${r['cost_usd']:.2f}", flush=True)
        except Exception as e:
            # Declared in the pre-registration: unavailable models are reported
            # with the error, never silently dropped.
            unavailable.append({"model": m, "error": repr(e)[:300]})
            print(f" UNAVAILABLE  {type(e).__name__}", flush=True)

    if args.dry_run:
        return 0

    out = {"preregistration": "PREREGISTRATION-SWEEP.md",
           "environment": "Phase 0 v3, unchanged",
           "config": {"episodes": EPISODES, "orders": ORDERS,
                      "samples": SAMPLES, "seed_base": SEED_BASE},
           "results": results, "unavailable": unavailable,
           "total_cost_usd": sum(r["cost_usd"] for r in results)}
    (ROOT / "runs" / "sweep-result.json").write_text(json.dumps(out, indent=2))

    print("\n── SWEEP result ──────────────────────────────────────")
    print(f"  {'model':22} {'bistable':>9}  {'95% CI':>13}  {'T_null':>7}  {'acc':>6}")
    for r in sorted(results, key=lambda x: x["bistability_rate"]):
        lo, hi = r["bistability_ci95"]
        print(f"  {r['model']:22} {r['bistability_rate']:>9.2f}  "
              f"[{lo:.2f},{hi:.2f}]  {r['t_null_p95']:>7.3f}  "
              f"{r['accuracy_canonical']:>6.3f}")
    if unavailable:
        print(f"\n  unavailable: {[u['model'] for u in unavailable]}")
    print(f"\n  total cost ${out['total_cost_usd']:.2f}")
    print("  Overlapping intervals are indistinguishable, not a ranking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
