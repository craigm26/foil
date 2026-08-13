#!/usr/bin/env python3
"""TURN-2 runner — executes PREREGISTRATION-TURN2.md exactly.

Gate 2 (construct validity) runs FIRST and blocks the main study in code, not by
discipline. TURN-1 failed because ground truth was never validated against the
model; here the main run is unreachable unless a full-information panel reaches
the intended answer in >= 90% of runs.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from foil.execute import cost_usd, load_prices
from foil.turn import (AGENTS, make_scenario2, speak_body, speaking_order,
                       vote_body)
from turn_run import Store

ROOT = Path(__file__).parent

# Fixed by the pre-registration.
S, R, ROUNDS = 32, 5, 2
SEED_BASE = 3000
MODEL = "claude-sonnet-5"
CONDITIONS = ("first", "last")
ALPHA, MIN_MEAN_DELTA = 0.01, 0.15
PERM_ITERS, PERM_SEED = 200_000, 0
GATE_SCENARIOS, GATE_REPS, GATE_MIN = 12, 3, 0.90
WORKERS = 6


def deliberate(store: Store, sc, order, holders: set[str], run_id: str) -> dict:
    transcript: list[tuple[str, str]] = []
    for rnd in range(1, ROUNDS + 1):
        for agent in order:
            body = speak_body(sc, agent, agent in holders, transcript, rnd, MODEL)
            transcript.append((agent, store.call(f"{run_id}_r{rnd}_{agent}", body).strip()))
    votes = {}
    for agent in AGENTS:
        raw = store.call(f"{run_id}_vote_{agent}",
                         vote_body(sc, agent, agent in holders, transcript, MODEL))
        try:
            votes[agent] = json.loads(raw).get("vote")
        except Exception:
            votes[agent] = None
    cast = [v for v in votes.values() if v in sc.candidates]
    if not cast:
        return {"run_id": run_id, "dropout": True}
    tally = Counter(cast)
    top = max(tally.values())
    group = next(c for c in sc.candidates if tally[c] == top)
    joined = " ".join(m for _, m in transcript)
    return {"run_id": run_id, "dropout": False, "scenario": sc.scenario_id,
            "group_answer": group, "truth": sc.truth, "correct": group == sc.truth,
            "votes": votes, "uttered": sc.private_token in joined}


def perm_test(deltas: list[float], iters: int = PERM_ITERS, seed: int = PERM_SEED) -> float:
    """Paired permutation test, one-sided. Under H0 the condition labels are
    exchangeable within a scenario, so each delta may flip sign. Uses the
    magnitudes, unlike the sign test that left TURN-1 unable to clear its own
    margin."""
    rng = random.Random(seed)
    obs = sum(deltas) / len(deltas)
    ge = 0
    for _ in range(iters):
        m = sum(d if rng.random() < 0.5 else -d for d in deltas) / len(deltas)
        if m >= obs:
            ge += 1
    return (ge + 1) / (iters + 1)


def main() -> int:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("no ANTHROPIC_API_KEY", file=sys.stderr)
        return 2
    prices = load_prices(ROOT / "data" / "prices-anthropic.json")
    store = Store(ROOT / "runs" / f"turn2-{MODEL}.jsonl", key, prices)

    scenarios, seed = [], SEED_BASE
    while len(scenarios) < S:
        sc = make_scenario2(seed)
        seed += 1
        if sc:
            scenarios.append(sc)

    # ─────────────── GATE 2: construct validity ───────────────
    print(f"── TURN-2 Gate 2: full-information panel, {GATE_SCENARIOS} scenarios "
          f"x {GATE_REPS} reps ──", flush=True)
    print("   (all four agents hold the private fact; no order manipulation)", flush=True)
    gate_jobs = [(sc, r) for sc in scenarios[:GATE_SCENARIOS] for r in range(GATE_REPS)]
    gate_res = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(deliberate, store, sc,
                            speaking_order("first")[0], set(AGENTS),
                            f"gate_{sc.scenario_id}_{r}")
                for sc, r in gate_jobs]
        for f in as_completed(futs):
            gate_res.append(f.result())
    kept = [r for r in gate_res if not r["dropout"]]
    gate_rate = sum(r["correct"] for r in kept) / max(1, len(kept))
    print(f"   full-information panel reached the intended answer in "
          f"{sum(r['correct'] for r in kept)}/{len(kept)} runs = {gate_rate:.3f}", flush=True)
    print(f"   requirement >= {GATE_MIN}", flush=True)

    if gate_rate < GATE_MIN:
        res = {"preregistration": "PREREGISTRATION-TURN2.md", "outcome": "GATE 2 FAILED",
               "gate_rate": gate_rate, "gate_required": GATE_MIN,
               "gate_runs": kept, "ledger": store.ledger.summary(),
               "note": ("The scenario family does not encode the intended judgement. "
                        "Per the pre-registration the main run does not happen and the "
                        "family is rejected; a revised family is a new attempt.")}
        out = ROOT / "runs" / f"turn2-result-{MODEL}.json"
        out.write_text(json.dumps(res, indent=2))
        print(f"\n  OUTCOME  GATE 2 FAILED — main run blocked, {len(kept)*12} calls spent")
        print(f"  wrote {out}")
        return 3

    print("   gate PASSED — proceeding to the main study\n", flush=True)

    # ─────────────── MAIN STUDY ───────────────
    print(f"── TURN-2 main: {S} scenarios x 2 conditions x {R} reps "
          f"= {S*2*R} deliberations ──", flush=True)
    jobs = [(sc, c, r) for sc in scenarios for c in CONDITIONS for r in range(R)]
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = []
        for sc, c, r in jobs:
            order, holder = speaking_order(c)
            futs.append(pool.submit(deliberate, store, sc, order, {holder},
                                    f"{sc.scenario_id}_{c}_{r}"))
        meta = {id(f): (sc, c, r) for f, (sc, c, r) in zip(futs, jobs)}
        for f in as_completed(futs):
            out = f.result()
            sc, c, r = meta[id(f)]
            out["condition"] = c
            results.append(out)
            done += 1
            if done % 32 == 0:
                print(f"  {done}/{len(jobs)} runs", flush=True)

    kept = [r for r in results if not r["dropout"]]
    by: dict[str, dict[str, list]] = {}
    for r in kept:
        by.setdefault(r["scenario"], {}).setdefault(r["condition"], []).append(r)

    per_scenario, dc, du = [], [], []
    for sid, cond in sorted(by.items()):
        if not all(c in cond and cond[c] for c in CONDITIONS):
            continue
        pf = sum(x["correct"] for x in cond["first"]) / len(cond["first"])
        pl = sum(x["correct"] for x in cond["last"]) / len(cond["last"])
        uf = sum(x["uttered"] for x in cond["first"]) / len(cond["first"])
        ul = sum(x["uttered"] for x in cond["last"]) / len(cond["last"])
        per_scenario.append({"scenario": sid, "delta_correct": pf - pl,
                             "p_correct_first": pf, "p_correct_last": pl,
                             "delta_uttered": uf - ul})
        dc.append(pf - pl)
        du.append(uf - ul)

    p1 = perm_test(dc)
    p2 = perm_test(du)
    mean_d = sum(dc) / len(dc)
    pos = sum(1 for d in dc if d > 0)
    neg = sum(1 for d in dc if d < 0)
    acc = {c: sum(r["correct"] for r in kept if r["condition"] == c)
              / max(1, sum(1 for r in kept if r["condition"] == c)) for c in CONDITIONS}
    utt = {c: sum(r["uttered"] for r in kept if r["condition"] == c)
              / max(1, sum(1 for r in kept if r["condition"] == c)) for c in CONDITIONS}
    outcome = "SUPPORTED" if (p1 < ALPHA and mean_d >= MIN_MEAN_DELTA) else "NOT SUPPORTED"

    led = store.ledger.summary()
    led["cost_usd"] = cost_usd(store.ledger.usage, MODEL, prices)
    res = {"preregistration": "PREREGISTRATION-TURN2.md", "model": MODEL,
           "gate_rate": gate_rate, "scenarios": S, "reps": R,
           "runs": len(results), "dropouts": len(results) - len(kept),
           "H1": {"p_permutation": p1, "mean_delta": mean_d,
                  "scenarios_first_better": pos, "scenarios_last_better": neg},
           "H2": {"p_permutation": p2, "mean_delta": sum(du) / len(du)},
           "accuracy_by_condition": acc, "utterance_by_condition": utt,
           "pressed_and_ignored": sum(1 for r in kept if r["uttered"] and not r["correct"]),
           "criteria": {"alpha": ALPHA, "min_mean_delta": MIN_MEAN_DELTA},
           "outcome": outcome, "per_scenario": per_scenario,
           "runs_detail": kept, "ledger": led}
    out = ROOT / "runs" / f"turn2-result-{MODEL}.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"""
── TURN-2 result ─────────────────────────────────────
  gate 2 (full-info panel)  {gate_rate:.3f}
  runs {len(kept)} kept / {res['dropouts']} dropouts

  accuracy   holder first {acc['first']:.3f}   holder last {acc['last']:.3f}
  uttered    holder first {utt['first']:.3f}   holder last {utt['last']:.3f}
  uttered but group wrong: {res['pressed_and_ignored']}

  H1  mean delta {mean_d:+.3f}   (need >= {MIN_MEAN_DELTA})
      permutation p = {p1:.5f}   (need < {ALPHA})
      first better in {pos} scenarios, last better in {neg}

  H2  mean delta {sum(du)/len(du):+.3f}   permutation p = {p2:.5f}

  OUTCOME  {outcome}
""")
    print("ledger:", json.dumps(led, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
