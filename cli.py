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
from foil.nulls import run_nulls
from foil.render import ForkKey, Operator, render, request_hash

ROOT = Path(__file__).parent
PRICES = ROOT / "data" / "prices-anthropic.json"


def est_tokens(body: dict) -> int:
    """Character/4 estimate. Explicitly an ESTIMATE: it is used only for the
    pre-flight projection, never for the ledger, which reports measured usage
    returned by the API."""
    n = sum(len(b["text"]) for b in body["system"])
    for m in body["messages"]:
        n += sum(len(c["text"]) for c in m["content"])
    return n // 4


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
    print(f"est input/call   ~{per_call_in} tok (estimate, char/4)")
    print(f"est output/call  ~{body['max_tokens']} tok max")
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
        lo = calls * per_call_in * row["inPerMtok"] / 1e6
        hi = lo + calls * body["max_tokens"] * row["outPerMtok"] / 1e6
        print(f"cost (est)       ${lo:.4f} .. ${hi:.4f}   [rates asOf {row['asOf']}]")
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
    print(f"episode {res['episode_id']}  model {res['model']}  n={res['samples_per_arm']}/arm")
    print(f"base distribution: {res['base_distribution']}")
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

    sr = sub.add_parser("run", help="execute the nulls")
    sr.add_argument("--base-url")
    sr.add_argument("--max-output-tokens", type=int, default=50_000)
    sr.set_defaults(fn=cmd_run)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
