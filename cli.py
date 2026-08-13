#!/usr/bin/env python3
"""FOIL Phase 0 runner.

    python3 cli.py plan                 # render + cost projection, zero API calls
    python3 cli.py run --n 30           # execute the nulls, apply the kill rule

Route through mcp-tape for free request/response capture and cost accounting:

    mcp-tape llm --port 4141 &
    ANTHROPIC_BASE_URL=http://127.0.0.1:4141 python3 cli.py run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from foil.env import DEFAULT_SCOUTS, make_episode
from foil.execute import DEFAULT_MIN_CACHEABLE, Executor, load_prices, resolve_price
from foil.env3 import make_episode_v3
from foil.nulls import run_nulls, run_nulls_multi
from foil.render import ForkKey, Operator, render, request_hash

ROOT = Path(__file__).parent
PRICES = ROOT / "data" / "prices-anthropic.json"


#: Calibration for the pre-flight estimator, measured 2026-08-13 against
#: claude-sonnet-5. Two corrections to the original prose-only char/4 rule:
#:
#:  1. Estimate over the WHOLE serialized body. Summing only the prose text
#:     fields projected ~187 tok/call where the API reported ~520, because it
#:     silently omitted the JSON output schema.
#:  2. Use ~2.4 chars/token, not 4. The 4:1 rule is for English prose; a
#:     request body is punctuation-dense JSON, which tokenizes far worse.
#:
#: Fitted to measured usage (7286 input tokens over 14 calls). Re-measure when
#: the request shape changes -- this is a calibration, not a law.
CHARS_PER_TOKEN = 2.4

#: Measured typical output for the two-field structured action (252 output
#: tokens over 14 calls, 2026-08-13). max_tokens is a truncation guard, not a
#: spend estimate.
TYPICAL_OUTPUT_TOKENS = 18


def est_tokens(body: dict) -> int:
    """Pre-flight input-token estimate over the full serialized request.

    Explicitly an ESTIMATE and never the figure of record: the ledger reports
    measured usage returned by the API, and the §8 G4 cost gate is set from
    that. Summing only the prose fields (the first implementation) understated
    input by ~2.8x because it silently omitted the output schema.
    """
    return int(len(json.dumps(body)) / CHARS_PER_TOKEN)


def cmd_plan(args) -> int:
    prices = load_prices(PRICES)
    ep = make_episode(seed=args.seed, overlap=args.overlap)
    key = ForkKey(ep.episode_id, 0, frozenset(ep.source_ids), Operator.NULL,
                  tuple(ep.source_ids), args.model)
    body = render(ep, key)

    prefix_tokens = sum(len(b["text"]) for b in body["system"]) // 4
    per_call_in = est_tokens(body)
    arms = args.orders + 1 + len(ep.source_ids)   # N1 orders + N2 + one ablation per source
    calls = arms * args.n

    row = resolve_price(args.model, prices)
    print("── FOIL Phase 0 plan ─────────────────────────────────")
    print(f"episode          {ep.episode_id}   (correct: {ep.correct_action})")
    print(f"model            {args.model}")
    print(f"sources          {', '.join(ep.source_ids)}")
    print(f"liar             {', '.join(s.name for s in DEFAULT_SCOUTS if s.lie_rate > 0)}"
          f" @ {[s.lie_rate for s in DEFAULT_SCOUTS if s.lie_rate > 0][0]}")
    print(f"arms             {arms}  (N1 orders {args.orders} + N2 1 + REF {len(ep.source_ids)})")
    print(f"samples/arm      {args.n}")
    print(f"total calls      {calls}")
    print(f"est input/call   ~{per_call_in} tok (calibrated estimate, +/-a few %)")
    print(f"est output/call  ~{TYPICAL_OUTPUT_TOKENS} tok typical "
          f"({body['max_tokens']} max, rarely approached)")
    print()
    print(f"invariant prefix ~{prefix_tokens} tok")
    if prefix_tokens < DEFAULT_MIN_CACHEABLE:
        print(f"  ⚠ below the ~{DEFAULT_MIN_CACHEABLE}-token minimum cacheable prefix.")
        print("    Prompt caching will NOT engage at this episode size. The §4.3 cost")
        print("    law still holds, but the cache term is zero -- Phase 1 must either")
        print("    grow the invariant prefix (history) or drop caching from the cost model.")
    print()
    if row is None:
        print(f"cost             UNKNOWN -- no price row for {args.model!r}")
        print("                 fails closed by design; add a row to data/prices-anthropic.json")
    else:
        # The structured action is a two-field JSON object, so output sits far
        # below max_tokens. Costing at max_tokens produced a ceiling ~3x the
        # real figure, which is not a useful bound for a cost gate.
        expected = (
            calls * per_call_in * row["inPerMtok"]
            + calls * TYPICAL_OUTPUT_TOKENS * row["outPerMtok"]
        ) / 1e6
        ceiling = (
            calls * per_call_in * row["inPerMtok"]
            + calls * body["max_tokens"] * row["outPerMtok"]
        ) / 1e6
        print(f"cost (est)       ${expected:.2f} expected   (${ceiling:.2f} absolute ceiling)")
        print(f"                 [rates asOf {row['asOf']}; estimates, never billing truth]")
    print()
    print(f"request hash     {request_hash(body)}")
    if args.show_prompt:
        print("── rendered request ──────────────────────────────────")
        print(json.dumps(body, indent=2))
    return 0


def cmd_run(args) -> int:
    prices = load_prices(PRICES)
    ep = make_episode(seed=args.seed, overlap=args.overlap)
    ex = Executor(
        store_path=ROOT / "runs" / f"{ep.episode_id}-{args.model}.jsonl",
        model=args.model,
        prices=prices,
        base_url=args.base_url,
        max_output_tokens=args.max_output_tokens,
    )
    res = run_nulls(ex, ep, n=args.n, orders=args.orders, seed=args.seed)
    out = ROOT / "runs" / f"nulls-{ep.episode_id}-{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    print("── Phase 0 nulls ─────────────────────────────────────")
    print(f"episode {res['episode_id']}  model {res['model']}  n={res['samples_per_arm_requested']}/arm")
    print(f"base distribution: {res['base_distribution']}")
    if res["short_arms"]:
        print(f"⚠ SHORT ARMS (fewer samples than requested): {res['short_arms']}")
    print()
    print(f"N1 order    p95 TV over {len(res['N1']['tvs'])} pairs")
    print(f"N2 paraphrase  TV = {res['N2']['tv']:.3f}  CI {res['N2']['ci']}")
    print(f"REF ablation median TV = {res['T_ablate_median']}")
    print()
    print(f"T_null (p95)      = {res['T_null_p95']}")
    print(f"kill threshold    = {res['kill_threshold']}  (0.5 x T_ablate)")
    print(f"VERDICT           = {res['verdict']}")
    print()
    print("ledger:", json.dumps(res["ledger"], indent=2))
    print(f"\nwrote {out}")
    return 0 if res["verdict"] != "KILL" else 3


def cmd_run2(args) -> int:
    """Phase 0 protocol v2 (PREREGISTRATION.md §12, post-hoc amendment)."""
    prices = load_prices(PRICES)
    # Versioned store: v1 and v2 episodes differ in coverage construction, so
    # their samples must never pool into one file even though request_hash
    # would keep them distinct.
    ex = Executor(
        store_path=ROOT / "runs" / f"v2-{args.model}.jsonl",
        model=args.model,
        prices=prices,
        base_url=args.base_url,
        max_output_tokens=args.max_output_tokens,
    )
    print(f"── Phase 0 v2: {args.episodes} episodes x 11 arms x {args.n} samples ──")
    res = run_nulls_multi(
        ex, n=args.n, episodes=args.episodes, orders=args.orders,
        overlap=args.overlap, seed=args.seed,
    )
    out = ROOT / "runs" / f"nulls-v2-{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    print()
    print("── Phase 0 v2 pooled result ──────────────────────────")
    print(f"episodes           {res['episodes']}   samples/arm {res['samples_per_arm']}")
    print(f"pooled null TVs    {res['pooled_null_n']}  (p95 is rank "
          f"{int(-(-0.95 * res['pooled_null_n'] // 1))}, an actual percentile)")
    print(f"pooled ablation    {res['pooled_ref_n']}")
    print()
    print(f"T_null (p95)       = {res['T_null_p95']:.4f}   "
          f"[median {res['null_tv_median']:.4f}, max {res['null_tv_max']:.4f}]")
    print(f"T_ablate (median)  = {res['T_ablate_median']:.4f}")
    print(f"kill threshold     = {res['kill_threshold']:.4f}  (0.5 x T_ablate)")
    print(f"VERDICT            = {res['verdict']}")
    print()
    print(f"bistable episodes  {len(res['bistable_episodes'])}/{res['episodes']} "
          f"({res['bistable_fraction']:.0%})  [max order TV > 0.5]")
    for e in res["bistable_episodes"]:
        print(f"   ⚠ {e}")
    shorts = {e["episode_id"]: e["short_arms"] for e in res["per_episode"] if e["short_arms"]}
    if shorts:
        print(f"⚠ SHORT ARMS: {shorts}")
    print()
    print("ledger:", json.dumps(res["ledger"], indent=2))
    print(f"\nwrote {out}")
    return 0 if res["verdict"] != "KILL" else 3


def cmd_run3(args) -> int:
    """Protocol v3: same gate, environment constrained analytically in advance."""
    prices = load_prices(PRICES)
    ex = Executor(
        store_path=ROOT / "runs" / f"v3-{args.model}.jsonl",
        model=args.model, prices=prices,
        base_url=args.base_url, max_output_tokens=args.max_output_tokens,
    )
    print(f"── Phase 0 v3: {args.episodes} episodes, >=3/4 sources decisive by construction ──")
    res = run_nulls_multi(
        ex, n=args.n, episodes=args.episodes, orders=args.orders,
        seed=args.seed, make=lambda seed, overlap="low": make_episode_v3(seed),
    )
    res["protocol"] = "v3 (analytic decisiveness requirement)"
    out = ROOT / "runs" / f"nulls-v3-{args.model}.json"
    out.write_text(json.dumps(res, indent=2))
    print()
    print("── Phase 0 v3 pooled result ──────────────────────────")
    print(f"T_null (p95)       = {res['T_null_p95']:.4f}   [median {res['null_tv_median']:.4f}]")
    print(f"T_ablate (median)  = {res['T_ablate_median']:.4f}")
    print(f"kill threshold     = {res['kill_threshold']:.4f}")
    print(f"VERDICT            = {res['verdict']}")
    print(f"bistable episodes  {len(res['bistable_episodes'])}/{res['episodes']}")
    print("ledger:", json.dumps(res["ledger"], indent=2))
    print(f"\nwrote {out}")
    return 0 if res["verdict"] == "PROCEED" else 3


def main() -> int:
    p = argparse.ArgumentParser(prog="foil")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--overlap", default="low", choices=["low", "high"])
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--orders", type=int, default=6)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="render and project cost; zero API calls")
    sp.add_argument("--show-prompt", action="store_true")
    sp.set_defaults(fn=cmd_plan)

    sr = sub.add_parser("run", help="execute the nulls (protocol v1, single episode)")
    sr.add_argument("--base-url")
    sr.add_argument("--max-output-tokens", type=int, default=50_000)
    sr.set_defaults(fn=cmd_run)

    s3 = sub.add_parser("run3", help="protocol v3: decisiveness-constrained environment")
    s3.add_argument("--base-url")
    s3.add_argument("--max-output-tokens", type=int, default=200_000)
    s3.add_argument("--episodes", type=int, default=12)
    s3.set_defaults(fn=cmd_run3)

    s2 = sub.add_parser("run2", help="protocol v2: pooled multi-episode nulls")
    s2.add_argument("--base-url")
    s2.add_argument("--max-output-tokens", type=int, default=200_000)
    s2.add_argument("--episodes", type=int, default=12)
    s2.set_defaults(fn=cmd_run2)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
