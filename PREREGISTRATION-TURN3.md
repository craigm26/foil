# TURN-3 pre-registration — does a dissenter benefit from the last word?

**Committed before any TURN-3 data exists.** Model: `claude-sonnet-5`.

---

## 1. Why this study exists

TURN-2 tested whether a group decides better when the holder of the decisive
private fact speaks **first**. It does not. The pre-registered outcome was NOT
SUPPORTED, and the point estimate ran the other way: accuracy 0.906 with the
holder first against 1.000 with the holder last, mean Δ = −0.094, bootstrap 95%
CI [−0.156, −0.044], zero of 32 scenarios favouring first and eleven favouring
last.

That reversal was **not claimed**, and is still not claimed. TURN-2's test was
one-sided in the opposite direction, and flipping a one-sided test after seeing
the sign is the manoeuvre pre-registration exists to prevent. This study tests
the reverse direction properly, on fresh data, with the hypothesis fixed in
advance.

TURN-2's data is prior evidence motivating this hypothesis. It is **not**
evidence for it, and will not be pooled with TURN-3's.

## 2. The flaw in TURN-2 this design must fix

TURN-2 declared DEGENERATE only if accuracy was at ceiling or floor **in both
conditions**. The last-speaker arm scored **1.000**.

With one arm at ceiling, Δ = last − first is bounded above by 1 − first, and any
hypothesis predicting a large positive Δ is testing against a wall. TURN-2's
power analysis assumed a base rate the comparison arm exceeded, so the design
was weaker than the stated ≥90% power implied.

**Both fixes are structural, not statistical:**

1. **The degeneracy rule now applies per arm.** If *either* arm scores ≥ 0.95 or
   ≤ 0.05 in the pilot, the design is DEGENERATE and the main run does not start.
2. **The task is made harder** so neither arm can ceiling. TURN-2's private fact
   carried weight −4 against a shared profile favouring the decoy 3–2, so once
   uttered it settled the question outright. TURN-3 reduces the private fact to
   weight −2, leaving the truth ahead by one point rather than three.

## 3. Design

Four agents deliberate over two rounds on a two-candidate hidden profile, then
vote. Exactly one agent holds a private disqualifying fact about the
shared-information leader.

| | |
|---|---|
| Conditions | holder speaks **first** vs holder speaks **last** |
| Scenarios | 32, paired (both conditions per scenario) |
| Replicates | 5 per condition per scenario |
| Runs | 32 × 2 × 5 = 320 |
| Agents per run | 4, two rounds, then a vote |

Assignment of speaking position is the only difference between arms. Scenario
content, agent prompts, and sampling settings are identical.

## 4. Gates, run in order, each blocking

**Gate 1 — structural.** Over 200 seeds: exactly two candidates, the shared
profile alone must favour the decoy, the full profile must favour the truth, and
both must be unique argmaxes. Re-derived from rendered text. Free.

**Gate 2 — construct validity.** A full-information panel, no manipulation, must
reach the intended answer in **≥ 90%** of 36 runs. Below that, the labels are not
what is being measured and the study stops. TURN-2 scored 36/36 here.

**Gate 3 — difficulty pilot.** NEW, and the fix for TURN-2's hole. Run 8
scenarios × 2 conditions × 3 replicates = 48 runs. If **either** arm's accuracy
is ≥ 0.95 or ≤ 0.05, declare **DEGENERATE** and stop. The main run does not
start. Pilot data is reported and is **not** pooled into the main analysis.

## 5. Hypothesis and test

**H1.** Group accuracy is higher when the private-fact holder speaks **last**.

- Statistic: per-scenario Δ = accuracy(last) − accuracy(first), averaged.
- Test: paired permutation, sign-flipping per scenario, 20,000 permutations,
  **one-sided** in the direction last > first.
- SUPPORTED requires **both**: mean Δ ≥ **0.15** and permutation **p < 0.01**.
- Anything else is NOT SUPPORTED. There is no third outcome and no rescue
  analysis.

## 6. Declared before data

- **Optional stopping is forbidden.** The run completes 320 runs or it is void.
- **No pooling** with TURN-1 or TURN-2 under any circumstance.
- **If H1 fails**, the result is reported as evidence that speaking position does
  not materially govern group accuracy in either direction on this task, and
  the TURN-2 reversal is recorded as unreplicated.
- **If the pilot declares DEGENERATE**, that is published as its own outcome. It
  is not an excuse to retune the task until an arm behaves.
- The bootstrap CI on mean Δ is **descriptive** and decides nothing.
- Dropouts (malformed votes, API failures) are reported. If more than 5% of runs
  drop, the result is reported as compromised.

## 7. What a SUPPORTED result would and would not mean

It would mean that on this synthetic two-candidate hidden profile, with this
model, giving the dissenting minority the final word before a vote measurably
improves group accuracy. That is a scheduling lever, and close to the opposite of
what a naive reading of the consensus-suppression literature suggests.

It would **not** establish a mechanism. The plausible story — that a fact
delivered immediately before the vote is not diluted by two rounds of discussion
— is untested by this design, which cannot separate recency from any other
consequence of speaking last. It would not generalise to other group sizes,
round counts, models, or non-synthetic tasks.
