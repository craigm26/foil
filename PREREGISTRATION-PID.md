# PID: Permutation-Instability Detection — Pre-Registration

**Date written:** 2026-08-13
**Status:** Written before any PID data was collected. No item in the task set
below has been shown to a model at the time of writing.

---

## 1. Where the hypothesis came from, and why that matters

FOIL's Phase 0 v2 run produced an unplanned observation: across 12 scripted-scout
episodes on `claude-sonnet-5`, the three episodes whose answer inverted under
reordering were exactly the three the listener got wrong, at ~1.00 confidence
(Fisher exact one-sided p = 0.0045; `RESULTS-phase0-v2.md` §4).

**That was exploratory.** The hypothesis was noticed after the data, and the
test was selected after the pattern was visible. It cannot be confirmed by the
data that generated it. This document exists to state the hypothesis, the task,
the model, the statistic, and the decision rule **in advance** of a test on
material the effect was not discovered on.

Two changes are deliberate and non-negotiable:

| | discovery run | this test |
|---|---|---|
| Model | `claude-sonnet-5` | **`claude-opus-5`** |
| Task family | scripted-scout routing (4 sources, elimination) | **evidence-passage QA** (5 passages, multi-hop reading) |

An effect that only exists on the model and task it was found on is a property
of that setup, not a detector.

---

## 2. Hypothesis

**H_PID.** An item whose answer changes under permutation of its evidence
passages is more likely to be answered incorrectly than an item whose answer is
stable under the same permutations.

Formally, with `U` = unstable and `C` = correct at canonical order:

> P(¬C | U) > P(¬C | ¬U)

One-sided. The reverse direction, or no difference, falsifies it.

### Why it would matter

The detector requires **no ground truth, no normative baseline, no attribution,
and no known source reliabilities**. At runtime you cannot observe whether an
answer is right, but you can permute the inputs and observe whether it moves. It
therefore sidesteps every limitation recorded in PREREGISTRATION.md §2.3, which
were the strongest arguments against FOIL's original residual metric.

### The most likely confound, stated up front

Item ambiguity plausibly causes **both** the instability and the error. If so,
instability is a *symptom*, not a *mechanism*. That is acceptable for a
detector — ambiguity is unobservable at runtime and instability is not — but it
means a positive result licenses "this predicts errors", never "order
sensitivity causes errors". No causal claim will be made from this design.

---

## 3. Task family

Synthetic evidence-passage QA. Each item is:

- a fictional entity (no real-world referent, so parametric knowledge cannot
  answer it),
- **5 short evidence passages**, exactly one of which is decisive when combined
  with exactly one other (a 2-hop chain),
- 2 distractor passages that are true but irrelevant, and 1 that is
  superficially relevant but does not bear on the question,
- a question with **one determinate answer** entailed by the passages,
- a fixed answer set of 4 options.

Ground truth is by construction. The permutable units are the passages.

**Determinacy is verified analytically before any model call:** the generator
asserts that the 2-hop chain entails exactly one option and that no other
option is entailed. Any item failing that assertion is discarded at generation
time, not after seeing a model's answer.

---

## 4. Design

| Parameter | Value |
|---|---|
| Items | **K = 40** |
| Permutations per item | **m = 4** (canonical + 3 random, fixed by seed) |
| Samples per (item, permutation) | **n = 5** |
| Total calls | 40 × 4 × 5 = **800** |
| Model | `claude-opus-5` |
| Thinking | disabled (matches the discovery condition) |
| Sampling | model default (temperature is not settable on this model) |

### Operationalisation, fixed now

- `modal(item, perm)` = the most frequent parsed answer over the 5 samples.
  Ties broken by the answer-set order; if fewer than 3 of 5 samples parse, the
  item is dropped and counted as a dropout.
- **`U(item)` (unstable)** = `modal` is not identical across all 4 permutations.
- **`C(item)` (correct)** = `modal(item, canonical) == truth`.

Note `C` is evaluated at the canonical order only. A detector must be usable
from the answer you would actually have shipped.

---

## 5. Analysis plan, fixed now

**Primary test.** Fisher exact, one-sided, on the 2×2 table of `U` × `¬C`.

**Decision rule — all three must hold to support H_PID:**

1. `p < 0.01`
2. Risk difference `P(¬C|U) − P(¬C|¬U) ≥ 0.30`
3. At least **8** unstable items and at least **8** stable items, so neither
   margin is degenerate

**Reported regardless:** the full 2×2 table, both conditional error rates, the
risk difference with a 95% bootstrap CI, overall accuracy, the instability
rate, and the dropout count.

**Pre-specified failure outcomes, each of which is a real result:**

- **Not replicated** — `p ≥ 0.01` or risk difference `< 0.30`. Reported as a
  failure to replicate on a new model and task family, which substantially
  weakens the discovery-run observation.
- **Degenerate** — fewer than 8 items in either margin. Reported as
  uninformative; the task was too easy or too hard to test the hypothesis, and
  no claim is made in either direction. This is not a licence to retune the
  task and rerun: any such rerun is a new, separately labelled attempt.

**No optional stopping.** All 40 items are collected before any test is run.
The analysis is executed once.

---

## 6. What a positive result would and would not establish

**Would:** that on two models and two task families, permutation instability
carries information about correctness, at an effect size worth acting on.

**Would not:** that instability causes error; that the effect holds on
non-synthetic tasks; that any particular threshold generalises; or that the
detector is calibrated. Calibration would need a separate study with its own
pre-registration.

---

## 7. Relationship to FOIL Phase 0

Independent. FOIL's Phase 0 gate has not been passed under any protocol
(`RESULTS-phase0.md`, `RESULTS-phase0-v2.md`, and the v3 attempt). PID does not
depend on that gate and does not revive it — it uses the same harness, the same
executor, and the same cost discipline, but tests a different claim that needs
none of the machinery Phase 0 was gating.

A positive PID result does **not** reopen Phase 1.

---

## 8. Amendment log

Amendments appended below with date and reason. Nothing above is edited after
first data collection.

| Date | Section | Change | Reason |
|---|---|---|---|
| _(none)_ | | | |
