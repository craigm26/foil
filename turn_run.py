#!/usr/bin/env python3
"""TURN runner — executes PREREGISTRATION-TURN.md exactly.

All 144 deliberations complete before any statistic is computed (§5, no optional
stopping). The analysis runs once.

Storage is keyed by an explicit call id rather than a request hash: three
replicates of a scenario issue byte-identical first turns, and hash keying would
hand all three the same cached response instead of three independent draws.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from foil.execute import ApiError, Ledger, Usage, cost_usd, load_prices
from foil.turn import (AGENTS, make_scenario, speak_body, speaking_order,
                       vote_body)

ROOT = Path(__file__).parent

# Fixed by the pre-registration.
S, R, ROUNDS = 24, 3, 2
SEED_BASE = 2000
MODEL = "claude-sonnet-5"
CONDITIONS = ("first", "last")
ALPHA, MIN_MEAN_DELTA, MIN_UNTIED = 0.01, 0.20, 16
WORKERS = 6

_lock = threading.Lock()


class Store:
    """Append-only call store keyed by explicit call id."""

    def __init__(self, path: Path, key: str, prices: list[dict]):
        self.path, self.key = path, key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(model=MODEL, prices=prices)
        self.cache: dict[str, str] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["call_id"]] = r["text"]

    def call(self, call_id: str, body: dict) -> str:
        with _lock:
            if call_id in self.cache:
                return self.cache[call_id]
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json",
                     "anthropic-version": "2023-06-01", "x-api-key": self.key},
            method="POST",
        )
        import time
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode()[:400]
                if e.code in (429, 500, 502, 503, 529) or (e.code == 400 and attempt < 2):
                    if attempt < 4:
                        time.sleep(2 ** attempt)
                        continue
                raise ApiError(e.code, detail) from None
            except urllib.error.URLError:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise
        u = data.get("usage", {}) or {}
        usage = Usage(u.get("input_tokens", 0) or 0, u.get("output_tokens", 0) or 0,
                      u.get("cache_read_input_tokens", 0) or 0,
                      u.get("cache_creation_input_tokens", 0) or 0)
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        with _lock:
            self.ledger.record(usage)
            self.cache[call_id] = text
            with self.path.open("a") as fh:
                fh.write(json.dumps({"call_id": call_id, "text": text}) + "\n")
        return text


def run_one(store: Store, sc, cond: str, rep: int) -> dict:
    order, holder = speaking_order(cond)
    run_id = f"{sc.scenario_id}_{cond}_{rep}"
    transcript: list[tuple[str, str]] = []

    for rnd in range(1, ROUNDS + 1):
        for agent in order:
            body = speak_body(sc, agent, agent == holder, transcript, rnd, MODEL)
            msg = store.call(f"{run_id}_r{rnd}_{agent}", body).strip()
            transcript.append((agent, msg))

    votes = {}
    for agent in AGENTS:
        body = vote_body(sc, agent, agent == holder, transcript, MODEL)
        raw = store.call(f"{run_id}_vote_{agent}", body)
        try:
            votes[agent] = json.loads(raw).get("vote")
        except Exception:
            votes[agent] = None

    cast = [v for v in votes.values() if v in sc.candidates]
    if not cast:
        return {"run_id": run_id, "dropout": True}
    tally = Counter(cast)
    top = max(tally.values())
    group = next(c for c in sc.candidates if tally[c] == top)  # tie -> option order

    joined = " ".join(m for _, m in transcript)
    holder_msgs = " ".join(m for who, m in transcript if who == holder)
    return {
        "run_id": run_id, "scenario": sc.scenario_id, "condition": cond,
        "rep": rep, "dropout": False,
        "group_answer": group, "truth": sc.truth, "decoy": sc.decoy,
        "correct": group == sc.truth,
        "votes": votes,
        "uttered": sc.private_token in joined,
        "uttered_by_holder": sc.private_token in holder_msgs,
    }


def sign_test(deltas: list[float]) -> tuple[float, int, int]:
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return float("nan"), pos, neg
    p = sum(math.comb(n, k) for k in range(pos, n + 1)) / (2 ** n)
    return p, pos, neg


def main() -> int:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("no ANTHROPIC_API_KEY", file=sys.stderr)
        return 2
    prices = load_prices(ROOT / "data" / "prices-anthropic.json")
    store = Store(ROOT / "runs" / f"turn-{MODEL}.jsonl", key, prices)

    scenarios, seed, tried = [], SEED_BASE, 0
    while len(scenarios) < S:
        sc = make_scenario(seed)
        tried += 1
        seed += 1
        if sc:
            scenarios.append(sc)
    print(f"── TURN: {S} scenarios x {len(CONDITIONS)} conditions x {R} reps "
          f"= {S * 2 * R} deliberations on {MODEL} ──", flush=True)
    print(f"  scenario admission rate {S}/{tried}", flush=True)

    jobs = [(sc, cond, rep) for sc in scenarios for cond in CONDITIONS for rep in range(R)]
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(run_one, store, sc, c, r): (sc, c, r) for sc, c, r in jobs}
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 12 == 0:
                print(f"  {done}/{len(jobs)} runs", flush=True)

    kept = [r for r in results if not r["dropout"]]
    by_sc: dict[str, dict[str, list[dict]]] = {}
    for r in kept:
        by_sc.setdefault(r["scenario"], {}).setdefault(r["condition"], []).append(r)

    per_scenario, d_correct, d_utter = [], [], []
    for sid, conds in sorted(by_sc.items()):
        if not all(c in conds and conds[c] for c in CONDITIONS):
            continue
        pf = sum(x["correct"] for x in conds["first"]) / len(conds["first"])
        pl = sum(x["correct"] for x in conds["last"]) / len(conds["last"])
        uf = sum(x["uttered"] for x in conds["first"]) / len(conds["first"])
        ul = sum(x["uttered"] for x in conds["last"]) / len(conds["last"])
        per_scenario.append({"scenario": sid, "p_correct_first": pf,
                             "p_correct_last": pl, "delta_correct": pf - pl,
                             "p_uttered_first": uf, "p_uttered_last": ul,
                             "delta_uttered": uf - ul})
        d_correct.append(pf - pl)
        d_utter.append(uf - ul)

    p1, pos1, neg1 = sign_test(d_correct)
    p2, pos2, neg2 = sign_test(d_utter)
    mean_d = sum(d_correct) / len(d_correct) if d_correct else float("nan")
    mean_u = sum(d_utter) / len(d_utter) if d_utter else float("nan")
    untied1, untied2 = pos1 + neg1, pos2 + neg2

    acc = {c: sum(r["correct"] for r in kept if r["condition"] == c)
              / max(1, sum(1 for r in kept if r["condition"] == c)) for c in CONDITIONS}
    utt = {c: sum(r["uttered"] for r in kept if r["condition"] == c)
              / max(1, sum(1 for r in kept if r["condition"] == c)) for c in CONDITIONS}
    pressed_ignored = sum(1 for r in kept if r["uttered"] and not r["correct"])

    ceiling = all(v >= 0.99 for v in acc.values()) or all(v <= 0.01 for v in acc.values())
    if untied1 < MIN_UNTIED or ceiling:
        outcome = "DEGENERATE"
    elif p1 < ALPHA and mean_d >= MIN_MEAN_DELTA:
        outcome = "SUPPORTED"
    else:
        outcome = "NOT SUPPORTED"

    led = store.ledger.summary()
    led["cost_usd"] = cost_usd(store.ledger.usage, MODEL, prices)
    res = {
        "preregistration": "PREREGISTRATION-TURN.md", "model": MODEL,
        "scenarios": S, "reps": R, "rounds": ROUNDS,
        "runs": len(results), "dropouts": len(results) - len(kept),
        "scenario_admission": f"{S}/{tried}",
        "H1": {"p_sign_test": p1, "scenarios_first_better": pos1,
               "scenarios_last_better": neg1, "untied": untied1,
               "mean_delta": mean_d},
        "H2": {"p_sign_test": p2, "scenarios_first_more": pos2,
               "scenarios_last_more": neg2, "untied": untied2,
               "mean_delta": mean_u},
        "accuracy_by_condition": acc, "utterance_by_condition": utt,
        "pressed_and_ignored": pressed_ignored,
        "criteria": {"alpha": ALPHA, "min_mean_delta": MIN_MEAN_DELTA,
                     "min_untied": MIN_UNTIED},
        "outcome": outcome,
        "per_scenario": per_scenario, "runs_detail": kept, "ledger": led,
    }
    out = ROOT / "runs" / f"turn-result-{MODEL}.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"""
── TURN result ───────────────────────────────────────
  runs {len(kept)} kept / {res['dropouts']} dropouts

  accuracy   holder first {acc['first']:.3f}   holder last {acc['last']:.3f}
  uttered    holder first {utt['first']:.3f}   holder last {utt['last']:.3f}
  uttered but group still wrong: {pressed_ignored} runs

  H1  first better in {pos1} scenarios, last better in {neg1}, untied {untied1}
      mean delta {mean_d:+.3f}   (need >= {MIN_MEAN_DELTA})
      sign test p = {p1:.5f}     (need < {ALPHA})

  H2  first more in {pos2}, last more in {neg2}, untied {untied2}
      mean delta {mean_u:+.3f}   sign test p = {p2:.5f}

  OUTCOME  {outcome}
""")
    print("ledger:", json.dumps(led, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
