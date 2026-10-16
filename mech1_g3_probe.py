#!/usr/bin/env python3
"""EXPLORATORY: does env3's format example steer the construct-gate answers?

    python3 mech1_g3_probe.py --model Qwen/Qwen3-0.6B

Post hoc. Not one of PREREGISTRATION-MECH.md §9's named explorations; written
on 2026-10-16 after rung A failed G3. It changes no registered outcome and
supports no claim. A claim from it would need its own registration.

env3's query ends `e.g. {"route": "north", "confidence": 0.7}`, and rung A
answered "north" in 41 of its 48 construct cases. This re-runs exactly those
cases: the first 48 identifiable seeds from 9500, the honest variant, the
canonical order and the registered weights and precision. The only change is
that the example's route token is swapped for each of the four routes in turn.
The token count stays the same, so every other token keeps its position.

It refuses to run on a rung whose registered gates have not run yet. Without
that check, the probe could be used to preview a rung's construct gate.
Output goes to runs/exploratory/, as with every unregistered result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mech_run  # noqa: E402

ROOT = Path(__file__).parent


def example_position(prompt, prefill_ids) -> int:
    """Index of the format example's route token in the user turn.

    In the example, `{"` carries a leading space and so is its own token. The
    match therefore uses the prefill without its first token, and it excludes
    the readout at the end of the prompt.
    """
    tail = list(prefill_ids[1:])
    ids = list(prompt.input_ids)
    found = [i for i in range(len(tail), len(ids) - 1)
             if ids[i - len(tail):i] == tail and ids[i] in prompt.action_ids]
    if len(found) != 1:
        raise RuntimeError(f"expected one format example, found {len(found)}")
    return found[0]


def main() -> int:
    p = argparse.ArgumentParser(prog="mech1_g3_probe")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", choices=sorted(mech_run.REGISTERED))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    gates = ROOT / "runs" / f"mech1-gates-{mech_run.slug(args.model)}.json"
    if not gates.exists():
        sys.exit(f"{gates.name} is missing: run this rung's registered gates first; "
                 "this probe never runs ahead of them")

    import numpy as np
    from foil.env3 import make_episode_v3
    from foil.provenance import stamp
    from mech.gates import CONSTRUCT_N, construct_seeds
    from mech.local import PREFILL, honest_variant
    from mech.whitebox import WhiteBox

    t0 = time.time()
    wb = WhiteBox(args.model, revision=mech_run.REGISTERED[args.model], device=args.device,
                  dtype=mech_run.REGISTERED_DTYPE)
    seeds = construct_seeds(wb.tok, CONSTRUCT_N)
    prefill_ids = wb.tok.encode(PREFILL, add_special_tokens=False)
    conditions = {}
    for example in ("north", "south", "east", "west"):
        rows = []
        for seed in seeds:
            ep = make_episode_v3(seed)
            hv = honest_variant(wb.tok, ep)
            at = example_position(hv, prefill_ids)
            if wb.tok.decode([hv.input_ids[at]]) != "north":
                raise RuntimeError("the registered example no longer names north")
            ids = list(hv.input_ids)
            ids[at] = hv.action_ids[list(ep.actions).index(example)]
            r = wb.run(hv.with_ids(tuple(ids)))
            probs = [float(x) for x in r.probs]
            rows.append({"seed": seed, "truth": ep.correct_action,
                         "answered": ep.actions[int(np.argmax(r.probs))],
                         "probs": dict(zip(ep.actions, probs)),
                         "on_format": float(r.on_format)})
        same = [r for r in rows if r["truth"] == example]
        other = [r for r in rows if r["truth"] != example]
        c = {
            "correct": sum(r["answered"] == r["truth"] for r in rows),
            "n": len(rows),
            "answered": dict(Counter(r["answered"] for r in rows)),
            "correct_when_truth_is_example": sum(r["answered"] == r["truth"] for r in same),
            "n_truth_is_example": len(same),
            "correct_otherwise": sum(r["answered"] == r["truth"] for r in other),
            "n_otherwise": len(other),
            "mean_p_truth": float(np.mean([r["probs"][r["truth"]] for r in rows])),
            "median_on_format": float(np.median([r["on_format"] for r in rows])),
            "rows": rows,
        }
        conditions[example] = c
        print(f"example={example:5s}  correct {c['correct']:2d}/{c['n']}  answered {c['answered']}  "
              f"truth=example {c['correct_when_truth_is_example']}/{c['n_truth_is_example']}  "
              f"otherwise {c['correct_otherwise']}/{c['n_otherwise']}  "
              f"mean P(truth) {c['mean_p_truth']:.3f}", flush=True)

    out_dir = ROOT / "runs" / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"mech1-g3-example-probe-{mech_run.slug(args.model)}.json"
    record = {
        "study": "MECH-1",
        "label": "EXPLORATORY: post hoc, not registered, not one of §9's named items; no claim",
        "question": "does the query's format example steer the construct-gate answers?",
        "registered_example": "north",
        "seeds": seeds,
        "identity": wb.identity(),
        "provenance": stamp(__file__),
        "forward_passes": 4 * len(seeds),
        "seconds": round(time.time() - t0, 1),
        "conditions": conditions,
    }
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
