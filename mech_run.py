#!/usr/bin/env python3
"""MECH-1 runner: FOIL's white-box layer on an open-weight listener.

    python3 mech_run.py plan    --model Qwen/Qwen3-1.7B   # tokenizer only: G1 + pass budget
    python3 mech_run.py gates   --model Qwen/Qwen3-1.7B   # G0-G3 on the weights
    python3 mech_run.py run     --model Qwen/Qwen3-1.7B   # refuses unless every gate passed
    python3 mech_run.py analyze --model Qwen/Qwen3-1.7B   # every decision, re-derived from the jsonl

No API calls and no key: a Claude subscription is irrelevant here and so is a
price table. The cost is compute, reported as forward passes and wall-clock.
Float32 is the registered precision (PREREGISTRATION-MECH.md §5). The device
is not registered, but it is recorded, TF32 is disabled on GPUs, and a study
must run on the same device and library versions its gates passed on.

The registration is enforced in code, the way TURN-2's calibration gate was:
`run` will not start for a model, revision or dtype the gates did not pass,
and `analyze` will not decide anything until all 48 episodes exist. Anything
else needs `--exploratory`, which writes under runs/exploratory/ and says so
in every file it touches.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent

#: The registered rungs (PREREGISTRATION-MECH.md §5.1): Hub commit hashes
#: pinned so "Qwen3-1.7B" means one set of weights forever. B is the
#: confirmatory rung; A is run first and reported as a replication.
REGISTERED = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
CONFIRMATORY = "Qwen/Qwen3-1.7B"
REGISTERED_DTYPE = "float32"
#: Upper bound on forward passes per episode: N1 6, N2 1, REF at canonical 4,
#: a plain pass at the arms ordering when N1 lacks it 1, factual-with-gradient
#: 1, nine arms x 4 sources, and per active source the carrier searches --
#: H3's k-ladder (4) + matched + top-|Δ|, H3b's k=16 + matched + top-|Δ| -- 9.
MAX_PASSES_PER_EPISODE = 6 + 1 + 4 + 1 + 1 + 9 * 4 + 9 * 4


def slug(model: str) -> str:
    return model.replace("/", "--")


def out_dir(args) -> Path:
    d = ROOT / "runs" / ("exploratory" if args.exploratory else "")
    d.mkdir(parents=True, exist_ok=True)
    return d


def check_registered(args, need_clean: bool = True) -> None:
    if args.exploratory:
        return
    if args.model not in REGISTERED:
        sys.exit(f"{args.model} is not in the registered ladder; use --exploratory")
    if args.revision not in (None, REGISTERED[args.model]):
        sys.exit("the registered revision is pinned; use --exploratory to run another")
    if args.dtype != REGISTERED_DTYPE:
        sys.exit(f"the registered dtype is {REGISTERED_DTYPE}; use --exploratory for {args.dtype}")
    if not need_clean:
        return
    from foil.provenance import stamp
    prov = stamp(__file__)
    if prov["git_dirty"] or not prov["git_commit"]:
        sys.exit("registered runs need a clean, committed tree: a result must name the code "
                 "that produced it (git status shows uncommitted changes)")


def revision(args) -> str | None:
    return args.revision or REGISTERED.get(args.model)


def load_tokenizer(args):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(args.model, revision=revision(args))


def cmd_plan(args) -> int:
    from foil.env3 import make_episode_v3
    from mech.gates import STUDY_SEEDS, structural_gate
    from mech.local import render_local

    check_registered(args, need_clean=False)  # tokenizer only: no data, no provenance needed
    tok = load_tokenizer(args)
    t0 = time.time()
    g1 = structural_gate(tok)
    base = render_local(tok, make_episode_v3(STUDY_SEEDS[0]))
    print("── MECH-1 plan ───────────────────────────────────────")
    print(f"model            {args.model} @ {revision(args)}")
    print(f"prompt length    {len(base.input_ids)} tokens (episode {base.episode_id})")
    print(f"episodes         {len(STUDY_SEEDS)}  (seeds {STUDY_SEEDS[0]}-{STUDY_SEEDS[-1]})")
    print(f"forward passes   <= {MAX_PASSES_PER_EPISODE * len(STUDY_SEEDS)} "
          f"(+{len(STUDY_SEEDS)} backward) for the study; gates add <= {8 * 4 * 3 + 8 + 72}")
    print(f"role             {'confirmatory' if args.model == CONFIRMATORY else 'replication'} rung")
    print(f"G1 structural    {g1['outcome']}  {g1['study_episodes']}  "
          f"construct cases {g1['construct_cases']} (skipped {g1['construct_seeds_skipped_unidentifiable']})")
    for f in g1["failures"]:
        print(f"   ✗ {f}")
    print(f"({time.time() - t0:.1f}s, zero model calls)")
    return 0 if g1["outcome"] == "PASS" else 3


def load_whitebox(args):
    from mech.whitebox import WhiteBox
    return WhiteBox(args.model, revision=revision(args), device=args.device, dtype=args.dtype)


def cmd_gates(args) -> int:
    from foil.provenance import stamp
    from mech.gates import construct_gate, leak_gate, structural_gate
    from mech.whitebox import Ineligible

    check_registered(args)
    path = out_dir(args) / f"mech1-gates-{slug(args.model)}.json"
    rec: dict = {"study": "MECH-1", "exploratory": args.exploratory, "gates": [],
                 "provenance": stamp(__file__)}
    t0 = time.time()
    loaded: dict = {}

    def finish(code: int) -> int:
        rec["seconds"] = round(time.time() - t0, 1)
        if "wb" in loaded:
            rec["forward_passes"] = loaded["wb"].passes
        rec["all_pass"] = code == 0
        path.write_text(json.dumps(rec, indent=2))
        print(f"\nwrote {path}")
        return code

    print(f"── MECH-1 gates: {args.model} ──")
    try:
        wb = load_whitebox(args)
    except Ineligible as e:
        rec["gates"].append({"gate": "G0 eligibility", "outcome": "FAIL", "reason": str(e)})
        print(f"G0 eligibility   FAIL  {e}")
        return finish(3)
    loaded["wb"] = wb
    rec["identity"] = wb.identity()
    rec["gates"].append({"gate": "G0 eligibility", "outcome": "PASS"})
    print("G0 eligibility   PASS")

    for name, fn in (("G1", lambda: structural_gate(wb.tok)),
                     ("G2", lambda: leak_gate(wb)),
                     ("G3", lambda: construct_gate(wb))):
        t1, p1 = time.time(), wb.passes
        g = fn()
        g["seconds"] = round(time.time() - t1, 1)
        g["forward_passes"] = wb.passes - p1
        rec["gates"].append(g)
        detail = {k: v for k, v in g.items() if k not in ("gate", "outcome", "failures", "misses")}
        print(f"{g['gate']:<16} {g['outcome']}  {json.dumps(detail)}")
        if g["outcome"] != "PASS":
            print("  a gate blocks: nothing after it runs")
            return finish(3)
    return finish(0)


def _gates_passed(args, ident: dict) -> bool:
    path = out_dir(args) / f"mech1-gates-{slug(args.model)}.json"
    if not path.exists():
        print(f"no gates file at {path}; run `mech_run.py gates` first")
        return False
    rec = json.loads(path.read_text())
    if not rec.get("all_pass"):
        print("the gates did not all pass for this model; the study does not run")
        return False
    from mech.whitebox import WhiteBox
    want = {k: ident[k] for k in WhiteBox.IDENTITY_KEYS}
    got = {k: rec["identity"].get(k) for k in WhiteBox.IDENTITY_KEYS}
    if want != got:
        print(f"gates passed for {got}, not {want}")
        return False
    return True


def validate_rows(rows: list[dict], gates: dict | None, *, exploratory: bool) -> str | None:
    """Why these rows may not be analysed, or None. Checked before any
    statistic exists, so a refusal can never depend on an outcome."""
    from collections import Counter
    from mech.gates import STUDY_SEEDS
    from mech.whitebox import WhiteBox

    if not rows:
        return "no episodes"
    dup = [e for e, n in Counter(r["episode"] for r in rows).items() if n > 1]
    if dup:
        return f"duplicate rows for {dup[:5]}: each registered episode must appear exactly once"
    ident = {k: rows[0]["identity"].get(k) for k in WhiteBox.IDENTITY_KEYS}
    if any({k: r["identity"].get(k) for k in WhiteBox.IDENTITY_KEYS} != ident for r in rows):
        return "rows come from more than one model identity"
    code = {(r.get("provenance") or {}).get("git_commit") for r in rows}
    dirty = any((r.get("provenance") or {}).get("git_dirty") for r in rows)
    if exploratory:
        return None
    if len(code) != 1 or None in code or dirty:
        return "rows do not all come from one clean, committed version of the code"
    want = {f"v3-s{s}" for s in STUDY_SEEDS}
    have = {r["episode"] for r in rows}
    if have != want:
        return (f"{len(have & want)}/{len(want)} registered episodes present; no optional "
                "stopping, so nothing is decided until all exist")
    if gates is None:
        return "no gates file: the gates block, so nothing is decided without them"
    if not gates.get("all_pass"):
        return "the gates did not all pass: the gate failure is this rung's result"
    if {k: gates.get("identity", {}).get(k) for k in WhiteBox.IDENTITY_KEYS} != ident:
        return "the gates passed for a different identity than the one that measured the episodes"
    return None


def cmd_run(args) -> int:
    from foil.provenance import stamp
    from mech.gates import STUDY_SEEDS
    from mech.study import run_study

    check_registered(args)
    wb = load_whitebox(args)
    if not _gates_passed(args, wb.identity()):
        return 3
    jsonl = out_dir(args) / f"mech1-{slug(args.model)}.jsonl"
    print(f"── MECH-1 study: {args.model}, {len(STUDY_SEEDS)} episodes → {jsonl} ──")
    t0 = time.time()
    run_study(wb, STUDY_SEEDS, jsonl, provenance=stamp(__file__))
    print(f"measured in {time.time() - t0:.0f}s")
    return cmd_analyze(args)


def cmd_analyze(args) -> int:
    from foil.provenance import stamp
    from mech import analysis as A
    from mech.gates import STUDY_SEEDS

    check_registered(args)
    jsonl = out_dir(args) / f"mech1-{slug(args.model)}.jsonl"
    rows = list(A.iter_jsonl(jsonl))
    out = out_dir(args) / f"mech1-{slug(args.model)}.json"
    if any(not r["consistency"].get("ok", True) for r in rows):
        res = A.void_result(rows)
        res.update(study="MECH-1", provenance=stamp(__file__),
                   identity=rows[0]["identity"] if rows else None)
        out.write_text(json.dumps(res, indent=2))
        print(f"VOID: hooks altered the computation in {res['integrity']['void_episodes']}; wrote {out}")
        return 3
    gates_path = out_dir(args) / f"mech1-gates-{slug(args.model)}.json"
    gates = json.loads(gates_path.read_text()) if gates_path.exists() else None
    why = validate_rows(rows, gates, exploratory=args.exploratory)
    if why:
        print(f"not analysed: {why}")
        return 3
    rows.sort(key=lambda r: r["seed"])
    res = A.analyze(rows)
    res["study"] = "MECH-1"
    res["exploratory"] = args.exploratory
    res["role"] = ("exploratory" if args.exploratory else
                   "confirmatory" if args.model == CONFIRMATORY else "replication")
    res["identity"] = rows[0]["identity"] if rows else None
    if gates is not None:
        res["gates"] = {x["gate"]: x["outcome"] for x in gates["gates"]}
        leak = next((x for x in gates["gates"] if x["gate"] == "G2 leak"), None)
        if leak:
            res["integrity"]["g2_max_leak"] = leak["max_abs_logit_diff_with_knockout"]
    res["compute"] = {"forward_passes": sum(r["passes"] for r in rows),
                      "seconds": round(sum(r["seconds"] for r in rows), 1)}
    res["provenance"] = stamp(__file__)
    out.write_text(json.dumps(res, indent=2))

    p0 = res["R0"]

    def f3(x):
        return "n/a" if x is None else f"{x:.3f}"

    print(f"── MECH-1 result ({res['role']}) ─────────────────────────")
    drops = {h: d["dropouts"] for h, d in res["integrity"]["dropouts"].items()}
    print(f"integrity             {res['integrity']['status']}  dropouts {drops}")
    print(f"accuracy (canonical)  {res['accuracy']['argmax_correct']}/{res['accuracy']['n']}  "
          f"mean P(correct) {f3(res['accuracy']['mean_p_correct'])}")
    print(f"R0 Phase 0 replica    T_null {f3(p0['T_null_p95'])}  T_ablate {f3(p0['T_ablate_median'])}  "
          f"→ {p0['verdict']}   bistable {f3(p0['bistable_fraction'])}")
    for h in ("H1", "H2", "H3", "H3b"):
        r = res[h]
        extra = f"   [{r['label']}]" if r.get("label") else ""
        extra += f"   ({r['reason']})" if r.get("reason") else ""
        extra += f"   motif: {r['motif_reading']}" if h == "H2" else ""
        print(f"{h:<21} {r['outcome']}{extra}")
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="mech_run")
    p.add_argument("cmd", choices=["plan", "gates", "run", "analyze"])
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default=REGISTERED_DTYPE)
    p.add_argument("--exploratory", action="store_true",
                   help="allow unregistered models/settings; outputs are labelled and kept apart")
    args = p.parse_args()
    return {"plan": cmd_plan, "gates": cmd_gates, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
