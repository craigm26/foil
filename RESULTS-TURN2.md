# TURN-2: NOT SUPPORTED — and the effect points the other way

**Date:** 2026-08-13
**Model:** `claude-sonnet-5`
**Pre-registration:** [PREREGISTRATION-TURN2.md](PREREGISTRATION-TURN2.md),
committed before any data existed.
**Volume:** Gate 2 (36 runs) + 320 deliberations = 4,272 calls. 0 dropouts,
0 API failures. Cost $15.71.

---

## 1. Outcome

**NOT SUPPORTED.** H1 predicted that a group does better when the holder of the
decisive private fact speaks **first**. It does not.

| | holder first | holder last |
|---|---|---|
| Group correct | 0.906 | **1.000** |
| Private fact uttered | 0.975 | 0.994 |

| H1 | value |
|---|---|
| Mean Δ (first − last) | **−0.094** (needed ≥ +0.15) |
| Permutation test, one-sided | p = 1.00 |
| Scenarios favouring *first* | **0** |
| Scenarios favouring *last* | **11** |
| Bootstrap 95% CI on mean Δ | **[−0.156, −0.044]** |

**Gate 2 passed at 36/36 (1.000)** — the full-information panel reached the
intended answer every time, so unlike TURN-1 the ground truth was the model's
too. The two-candidate fix worked: ties fell from 16/24 to 21/32 of a larger
sample, and the design ran at ≥90% power rather than 40%.

## 2. The direction is reversed, and the one-sided test cannot claim it

Speaking **last** was better, not worse. Zero of 32 scenarios favoured first;
eleven favoured last; the CI on the mean excludes zero on the negative side.

**This cannot be claimed as a finding from this test.** The pre-registered
hypothesis and test were one-sided in the *opposite* direction. A one-sided test
that fails does not license a significance claim for the reverse — flipping it
after seeing the sign is exactly the manoeuvre pre-registration exists to
prevent. The reversal is reported as a **descriptive observation** and would
need its own pre-registered test.

A plausible mechanism, offered as a hypothesis and nothing more: a holder who
speaks last delivers the disqualifying fact after the others have converged on
the decoy, as a decisive correction immediately before the vote. Speaking first,
the same fact is discussed for two rounds and may be diluted. If that is right,
the scheduling intervention worth testing is *giving a dissenter the last word*
— which is close to the opposite of what a naive reading of the suppression
literature would suggest.

## 3. A gap in my own degeneracy check

The pre-registration declared DEGENERATE if accuracy was at ceiling or floor **in
both conditions**. The *last* condition scored **1.000**.

With one arm at ceiling, Δ = first − last is **≤ 0 by construction**, so H1 was
structurally unable to be supported however the model behaved. The check should
have been *ceiling in the comparison arm*, not *ceiling in both*.

This does not change the verdict — H1 is not supported, and the reversal is real
enough to be worth its own study — but the test was weaker than the power
analysis implied, because power was computed against a base rate that the *last*
arm exceeded.

## 4. What this establishes

**Establishes:** on this synthetic two-candidate hidden profile with
`claude-sonnet-5`, moving the decisive holder to the front of the speaking order
does **not** improve group accuracy, at ≥90% power to detect a 0.15 effect. On
this task, deliberation is not damaged by putting a dissenter last.

**This limits the reach of this project's own earlier findings.** Phase 0 showed
a *single listener* can be inverted by input order. TURN-2 finds no comparable
positional harm once four agents deliberate over two rounds. Group deliberation
appears more robust to arrangement than single-shot reading — which is a
constraint on how far the order-sensitivity result generalises, and it is being
published as such.

**Does not establish:** that speaking last *helps* (one-sided test, wrong
direction); that this holds at other group sizes, round counts, models, or on
non-synthetic tasks; that suppression does not occur by other mechanisms. Note
15 runs had the fact uttered and still answered wrongly, all in the *first*
condition.

## 5. Data availability

- `runs/turn2-result-claude-sonnet-5.json` — per-run votes, per-scenario deltas,
  gate results, ledger.

Reproduce with `./run.sh turn2_run.py` (4,272 calls, ~$15.71).
