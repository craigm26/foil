#!/usr/bin/env python3
"""VERIFICATION RUN — evalgate.fixtures against a real model.

The fixtures' gate matrix is verified in tests against a SIMULATED oracle.
This run replaces it with `claude-sonnet-5` and checks the wiring against
reality. It is a verification, not a hypothesis test, but the expectations are
still committed before the data exists, because the fake-feed near-miss taught
us to verify the wiring rather than the capability.

PREDICTIONS, DECLARED BEFORE ANY CALL
-------------------------------------
20 cases per fixture, 3 samples each, full information, no manipulation,
scored against each fixture's `intended`:

  P1  The three VALID fixtures score >= 0.90.
  P2  `no_error_variance` scores >= 0.90 -- that near-perfection is exactly
      its invalidity, and only a variance-style check sees it.
  P3  UNCOMFORTABLE, AND THE POINT: `scorer_disagrees` ALSO scores >= 0.90.
      Its `intended` is self-consistent with the reference reasoner, and the
      real disagreement lives in a hidden field only the simulated oracle
      reads. A real model reading the rendered text should AGREE with the
      scorer. If P3 holds, the fixture CANNOT enact its invalidity against a
      real oracle -- which demonstrates the limitation honestly: a genuine
      scorer-model disagreement cannot be synthesized analytically. TURN-1's
      was discovered, not designed. The fixture therefore tests your gate's
      WIRING (does it call the oracle, compare, and reject on mismatch), not
      your model's judgement, and the docs must say so.
  P4  The two structurally-invalid fixtures are NOT expected to fail here;
      construct is the wrong gate for them, which is the point of having four.

If P3 FAILS (the model disagrees with the scorer on sound, fully-attested
text), that is more interesting: a real construct-gap inside a fixture we
believed self-consistent, and it gets its own write-up.

~420 calls on claude-sonnet-5. Estimated ~$1.50.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from evalgate.fixtures import FIXTURES
from foil.execute import Executor, load_prices, cost_usd
from foil.provenance import stamp
from foil.render import action_schema, parse_action

ROOT = Path(__file__).parent
MODEL = "claude-sonnet-5"
CASES, SAMPLES = 20, 3
ROUTES = ("north", "south", "east", "west")

SYSTEM = (
    "You are a dispatcher routing a convoy through contested terrain. "
    "Exactly one route is clear; the other three are blocked. "
    "You cannot observe the routes yourself. Scouts report what they saw. "
    "Scouts may disagree, and not every scout is reliable. "
    "Choose the route you believe is clear."
)
QUERY = (
    "Which route is clear? Answer with a JSON object containing the route name "
    'and your confidence from 0.0 to 1.0, e.g. {"route": "north", "confidence": 0.7}.'
)


def body(text: str) -> dict:
    return {
        "model": MODEL,
        "max_tokens": 256,
        "thinking": {"type": "disabled"},
        "output_config": {"format": {"type": "json_schema",
                                     "schema": action_schema(ROUTES)}},
        "system": [{"type": "text", "text": SYSTEM,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user",
                      "content": [{"type": "text",
                                   "text": text + "\n\n" + QUERY}]}],
    }


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("no ANTHROPIC_API_KEY", file=sys.stderr)
        return 2
    prices = load_prices(ROOT / "data" / "prices-anthropic.json")
    ex = Executor(store_path=ROOT / "runs" / f"fixtures-verify-{MODEL}.jsonl",
                  model=MODEL, prices=prices)

    rows = []
    print(f"── fixtures verification: {len(FIXTURES)} fixtures x {CASES} cases "
          f"x {SAMPLES} samples on {MODEL} ──", flush=True)
    for f in FIXTURES:
        agree = total = 0
        disagreements = []
        for seed in range(CASES):
            case = f.make_case(seed)
            recs = ex.sample(body(case.text), SAMPLES)
            for r in recs:
                act, _ = parse_action(r["text"], ROUTES)
                if act is None:
                    continue
                total += 1
                if act == case.intended:
                    agree += 1
                elif len(disagreements) < 5:
                    disagreements.append({"seed": seed, "model": act,
                                          "intended": case.intended})
        rate = agree / total if total else 0.0
        rows.append({"fixture": f.name, "valid": f.valid, "agree": agree,
                     "total": total, "rate": rate,
                     "disagreements": disagreements})
        print(f"  {f.name:22} agrees with scorer {agree}/{total} = {rate:.3f}",
              flush=True)

    led = ex.ledger.summary()
    led["cost_usd"] = cost_usd(ex.ledger.usage, MODEL, prices)
    out = {"purpose": "verification of evalgate.fixtures against a real model",
           "predictions": "P1-P4 in the module docstring, committed pre-run",
           "model": MODEL, "cases": CASES, "samples": SAMPLES,
           "provenance": stamp(__file__), "results": rows, "ledger": led}
    (ROOT / "runs" / "fixtures-verify-result.json").write_text(
        json.dumps(out, indent=2))
    print(f"\n  cost ${led['cost_usd']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
