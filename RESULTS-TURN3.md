# TURN-3: GATE 2 FAILED — the design has no workable middle

**Date:** 2026-08-13
**Model:** `claude-sonnet-5`
**Pre-registration:** [PREREGISTRATION-TURN3.md](PREREGISTRATION-TURN3.md),
committed before the generator was written.
**Volume:** 36 gate runs, 432 calls. Cost **$1.56**. The main run never started.

---

## 1. Outcome

**GATE 2 FAILED.** The full-information panel reached the intended answer in
**32 of 36** runs = **0.889**, against a pre-registered requirement of ≥ 0.90.

Per the pre-registration, the scenario family is rejected and the main run does
not happen.

## 2. This failed by one run, and that changes nothing

33 of 36 would have passed. The Wilson 95% interval on 32/36 is
**[0.747, 0.956]**, which straddles the threshold, so the true rate may well be
above 0.90.

None of that is grounds to proceed. A pre-registered threshold is a **decision
rule fixed in advance**, not an estimate to be argued with after seeing which
side of it the data landed on. "It was nearly 0.90" and "the CI includes 0.90"
are the two most natural ways to talk oneself past a gate, and both were
available to TURN-1, which passed its structural gate and was invalid anyway.

Nor is the family being retuned. The pre-registration says a revised family is a
**new attempt**, requiring its own pre-registration. Nudging the private fact's
weight until the gate clears is precisely the search-until-it-passes procedure
this project exists to avoid, and it would invalidate everything downstream.

## 3. What the failure actually says

This is informative, not just a null.

TURN-2 used a private fact of weight −4 against a shared profile favouring the
decoy 3–2. Once uttered it settled the question outright, and the last-speaker
arm scored **1.000** — the ceiling that made TURN-2's hypothesis untestable.

TURN-3 weakened it to −2, leaving the truth ahead by one point. That was enough
to break construct validity: with **full information and no manipulation
whatsoever**, four agents reach the intended answer only 89% of the time. The
scorer says the truth leads by one; the model does not reliably agree.

So on this scenario family the two failure modes are adjacent:

| private fact weight | consequence |
|---|---|
| −4 | comparison arm at ceiling (1.000). Δ bounded, hypothesis untestable. |
| −2 | ground truth no longer shared (0.889). Labels are not what is measured. |

There may be no setting between them that both leaves room to move and keeps the
model's answer aligned with the scorer's. If so, **this design cannot test the
TURN-2 reversal**, and a different design is needed rather than a different
constant.

## 4. Status of the TURN-2 reversal

**Unreplicated, and not claimed.** TURN-2 observed that groups did better when
the private-fact holder spoke last (1.000 against 0.906, CI on the mean
Δ excluding zero). That was never claimed, because the test was one-sided in the
opposite direction. TURN-3 was the attempt to test it properly and did not get
far enough to try.

The question stands where TURN-2 left it: a suggestive point estimate with no
confirmatory test behind it.

## 5. The gate paid for itself

The main run was budgeted at roughly $18 and 4,272 calls. Gate 2 cost **$1.56**
and 432 calls, and it blocked a study that would have measured group accuracy
against labels the model does not share — the exact failure that made TURN-1
worthless after full payment.

That is the case for `evalgate.construct` in one line: **11% of the cost, spent
before the mistake instead of after it.**

## 6. Data

- `runs/turn3-result-claude-sonnet-5.json` — the 36 gate runs, ledger, verdict.

Reproduce with `./run.sh turn3_run.py`. It exits 2 at this gate and spends
nothing further.
