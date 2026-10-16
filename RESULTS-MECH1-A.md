# MECH-1, rung A: G3 FAILED — Qwen3-0.6B does not do the task

**Date:** 2026-10-16
**Model:** `Qwen/Qwen3-0.6B` at Hub revision `c1899de289a04d12100db370d81485cdf75e47ca`,
float32, CPU (Intel Xeon @ 2.80 GHz, 2 cores), torch 2.14.1+cpu, transformers 5.20.1.
**Pre-registration:** [PREREGISTRATION-MECH.md](PREREGISTRATION-MECH.md), committed in
`fb57198` before any MECH-1 data.
**Code:** `78e0e44`. This is the registration plus runner hardening from the
review of #1 (#2): `analyze` now enforces the gates, and the runner checks one
row per episode, code provenance and full device identity. It landed before
any registered data and changed no hypothesis, threshold or arm.
**Role:** replication rung. MECH-1's claims come from rung B (`Qwen/Qwen3-1.7B`),
which has not been run.
**Volume:** gates only: 152 forward passes, 296 s. No API calls, $0. The
48-episode run never started.

---

## 1. Outcome

**G3 construct FAILED.** The model got full information with no manipulation:
the liar's report inverted to the truth, the reports in canonical order. It
picked the true route in **19 of 48** cases, **0.396**, against a registered
requirement of ≥ 0.90. The exact binomial test rejects 0.90 at p = 1.7 × 10⁻¹⁷,
so the outcome is FAIL outright, not EXTEND, and the extension batch did not
run.

The readout was on format. Median on-format mass was 0.989, against G3's
separate requirement of a median ≥ 0.90. The minimum was 0.971, against a
per-pass floor of 0.50. The failure is in which route the model chose, not in
whether it answered.

Per §6, a rung that fails a gate stops there, and the failure is that rung's
published result. Rung A's 48 registered episodes were not run, and H1, H2,
H3, H3b and R0 have no rung A outcome. Per §5.1, rung B runs regardless.

## 2. The gates

| gate | outcome | detail |
|---|---|---|
| G0 eligibility | PASS | pure softmax attention, gated MLP with `down_proj`, eager attention |
| G1 structural | PASS | 48/48 study episodes and 72/72 construct cases render, map and read out; env3's decisiveness holds on the rendered text |
| G2 leak | PASS | 32 tests (8 episodes × 4 sources). With a report knocked out, overwriting its tokens moved no decision logit at all: max \|Δ\| = **0.0**, tolerance 1e-4. Without the knockout, the same overwrite moved the logits by at least 1.58 in every test, so the test had power |
| G3 construct | **FAIL** | 19/48 = 0.396 against 0.90, p = 1.7 × 10⁻¹⁷ |

G2 is the instrument's own test, and it passed exactly. On these weights at
float32, the in-place knockout cuts every path from a report to the decision,
and the tolerance never had to excuse a leak. The model's answers failed, not
the knockout.

## 3. What the failure looks like (descriptive)

The gates file keeps only the count, the on-format statistics and the first
ten misses. To see the whole readout, the 48 cases were re-run in §4's probe.
Its registered-example condition is exactly the G3 prompt: condition `north`
in `data/mech1-A-g3-example-probe-EXPLORATORY.json`. The passes are exact. The
re-run reproduces the 19/48, the on-format median and minimum, and the ten
recorded misses, all to the bit. Every number in this section comes from that
re-run.

- **The model answers "north" 41 times out of 48.** All 29 errors answered
  north, and all 7 non-north answers were correct.
- By true route: north **12/12**, east 4/10, south 2/11, west 1/15.
- Mean probability over the 48 cases: north 0.650, east 0.152, south 0.143,
  west 0.056. The truths were 12 north, 11 south, 10 east and 15 west.

The reports are read. Each route gets between 1.5× and 4.8× more probability
when it is the truth than when it is not:

| route | mean P(route) when it is the truth | when it is not |
|---|---|---|
| north | 0.884 | 0.571 |
| south | 0.289 | 0.099 |
| east | 0.408 | 0.084 |
| west | 0.103 | 0.034 |

That shift is rarely big enough to overcome a default of about 0.57 on north,
which holds even when north is blocked.

Every episode's query ends the same way. This is env3's `QUERY_TEXT`. The same
example is in every scout-task study's prompt: Phase 0 v1–v3, PID-2 and SWEEP.

> Which route is clear? Answer with a JSON object containing the route name and
> your confidence from 0.0 to 1.0, e.g. {"route": "north", "confidence": 0.7}.

## 4. Exploratory: the format example

**This probe is post hoc and unregistered, and it is not one of §9's named
explorations. It was run after the failure and makes no claim.**
[`mech1_g3_probe.py`](mech1_g3_probe.py) re-runs the same 48 cases and swaps
only the example's route token. The token count does not change, so every
other token keeps its position.

| example names | correct | answered the example's route | correct, truth = example | correct, otherwise | mean P(truth) |
|---|---|---|---|---|---|
| north (registered) | 19/48 | 41 | 12/12 | 7/36 | 0.404 |
| south | 21/48 | 25 | 11/11 | 10/37 | 0.420 |
| east | 20/48 | 33 | 10/10 | 10/38 | 0.376 |
| west | 32/48 | 14 | 13/15 | 19/33 | 0.533 |

Two readings, both descriptive:

1. **The example is a strong attractor.** Naming a route in it multiplies how
   often that route is answered. Against the registered example, south goes
   from 2 answers to 25, east from 4 to 33 and west from 1 to 14. Across the
   four conditions, the model is right on 46 of the 48 cases whose truth is
   the route the example names.
2. **It does not account for the whole failure.** No choice of example brings
   the model near 0.90; the best is 32/48 = 0.67. North also keeps
   probability it should not. Under the three swapped examples, it holds a
   mean of 0.21–0.34 on cases where it is blocked. Routes that are neither
   north nor the example's hold 0.02–0.18 on cases where they are blocked.

These numbers suggest a reading but do not establish it. On this stimulus,
Qwen3-0.6B with thinking off may not reliably combine four scout reports,
leaving its answer set mostly by defaults, one of which the prompt's own
example supplies. Testing that would need its own registration.

## 5. What this does not say

- **Nothing about R0 or H1–H3b**, on this model or any other, and nothing
  about rung B. B's G3 is a separate test, and the larger model may pass or
  fail it. §5.1 already fixes what each outcome means.
- **Nothing against the instrument.** G2's exact 0.0 and the 0.99 on-format
  mass show that the knockout and the readout work on these weights. G3
  caught the case it exists for: a model that does not do the task. On such
  a model, measuring how the answer depends on each source would mostly
  measure how the sources nudge a default.
- **No licence to change the stimulus inside MECH-1.** The example is part of
  the registered design. A neutral example is a new study with its own
  registration, just as a larger rung would be (§5.1).
- **Little about Claude.** The example is in every scout-task study's prompt.
  PID-1 and the TURN studies use other prompts. Whether the example pulls
  Claude toward north is untested, but the archived per-episode answers do
  not suggest it:
  - In Phase 0 v2, Sonnet 5's base distributions put 3.0 of 12 episodes'
    mass on north, with 3 north truths.
  - In Phase 0 v3, they put 3.76 on north, with 5 north truths.
  - In PID-2, Opus 5 answered north 12 times, with 16 north truths, and 2 of
    its 13 errors were north.

  These samples are small, and no swap like §4's has been run on Claude.

## 6. Status of MECH-1

| rung | model | status |
|---|---|---|
| A (replication) | Qwen3-0.6B | **G3 FAILED** (this document). Descriptive only, as every A outcome would have been |
| **B (confirmatory)** | Qwen3-1.7B | **not run.** It needs about 12 GB of RAM in float32, more than the machine that ran A has (7 GB). Per §5.1 it runs whatever A's outcome |

If B passes its gates, its outcomes (R0, H1–H3b) are MECH-1's result. A's
failure is reported alongside them and never pooled with them. If B fails any
gate, that failure is MECH-1's confirmatory result. A larger model or a
neutral example would then be a new study.

## 7. Cost and data

| | forward passes | wall-clock |
|---|---|---|
| G2 leak | 104 | 112.5 s |
| G3 construct | 48 | 52.7 s |
| gates, total (with model load and G1) | 152 | 295.7 s |
| exploratory probe (§4) | 192 | 222.2 s |

No API calls, $0.

The gates file records wall-clock but not forward passes, although §10 says
every result file records both. The pass counts for G2 and G3 above come from
`gates.py`: G2 runs 8 × (1 + 4 × 3) = 104 and G3 runs 48. The probe records
its own count. Since `884ac28` the runner counts passes and writes them into
the gates file, so rung B's gates file will carry them.

- `data/mech1-A-gates-result.json` is the registered gates file, verbatim
  (`runs/mech1-gates-Qwen--Qwen3-0.6B.json`). Provenance: `78e0e44`, clean
  tree.
- `data/mech1-A-g3-example-probe-EXPLORATORY.json` holds the probe's four
  conditions, with every case's full distribution. Provenance: `e3795bc`,
  clean tree.

Reproduce:

```bash
pip install -r requirements-mech.txt
python3 mech_run.py gates --model Qwen/Qwen3-0.6B   # exits 3 at G3
python3 mech1_g3_probe.py --model Qwen/Qwen3-0.6B   # exploratory; needs the gates file
```

Every pass is exact. On the same weights, precision and libraries, a rerun
gives the same numbers. The re-run in §3 matched every figure the gates file
records.
