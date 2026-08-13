# PID-2: Permutation-Instability Detection — Pre-Registration

**Written:** 2026-08-13, before any PID-2 data was collected. No episode in the
seed range below has been shown to a model at the time of writing.
**Supersedes:** nothing. [PREREGISTRATION-PID.md](PREREGISTRATION-PID.md) and
its DEGENERATE outcome ([RESULTS-PID.md](RESULTS-PID.md)) stand unrevised; this
is a separately labelled attempt, as §5 of that document requires.

---

## 1. Why this design differs from PID-1

PID-1 required a different model **and** a different task family. That is the
stronger test in principle, and it failed in practice: the new task's difficulty
had never been measured, `claude-opus-5` answered 40 of 40 items correctly under
every permutation, and the run could not test anything. A calibration ladder on
that family has since returned 1.00 on its first two rungs.

The scripted-scout family is the one environment with a **measured** error rate:
3/12 wrong in Phase 0 v2 and 4/12 in v3. It cannot land on DEGENERATE for lack
of errors.

This is a deliberate trade, stated plainly: **one task family instead of two, in
exchange for a test that can actually run.** A pre-registered result on one
family is worth more than an untestable one on two. It buys a weaker
generalisation claim, and §6 records exactly how much weaker.

| | discovery (v2/v3) | PID-2 |
|---|---|---|
| Model | `claude-sonnet-5` | **`claude-opus-5`** |
| Environment | scout v3 (decisive), seeds 1–12 | scout v3 (decisive), **seeds 1000–1047** |
| Status | exploratory, post-hoc | **pre-registered** |

The model axis changes and the episodes are disjoint. The task family does not
change, and no claim beyond it will be made.

---

## 2. Hypothesis

**H_PID.** An episode whose answer changes under permutation of its source
reports is more likely to be answered incorrectly than an episode whose answer
is stable under the same permutations.

> P(¬C | U) > P(¬C | ¬U), one-sided.

**Secondary, and the one most at risk.** Across v2 and v3, no stable episode was
ever answered wrong (0 of 15). PID-2 is designed to *break* that claim if it is
breakable, not to confirm it: §4 sizes the stable margin so that a genuine
false-negative rate above roughly 10% would be expected to show at least one
case.

### The confound, stated up front

Episode ambiguity plausibly causes both the instability and the error, making
instability a symptom rather than a mechanism. That is acceptable for a
detector — ambiguity is unobservable at runtime, instability is not — but **no
causal claim will be made from this design**, whatever the result.

---

## 3. What is measured, and what is not

PID-2 needs only orderings and correctness. The ablation battery (REF) and the
paraphrase arm (N2) belong to FOIL's kill rule and are **not run**: they would
triple the cost and answer a different question.

Per episode: `m` orderings × `n` samples. No ablations.

---

## 4. Design, fixed now

| Parameter | Value |
|---|---|
| Episodes | **K = 48** (`env3` generator, seeds 1000–1047) |
| Orderings per episode | **m = 6** (canonical + 5 random, seeded) |
| Samples per (episode, ordering) | **n = 10** |
| Total calls | 48 × 6 × 10 = **2,880** |
| Model | `claude-opus-5` |
| Thinking | disabled |
| Delivery | Batches API (50% of list) |

`n = 10` is justified by measurement, not convenience: across v1, v2 and v3,
within-ordering sampling variance was near zero — arms returned P(correct) at
0.000, 0.005, 1.000 and similar extremes, essentially never near 0.5. Ten
samples identify a modal answer at those extremes comfortably.

`K = 48` is sized for the secondary claim. At the ~50% stability rate observed
in v3, 48 episodes yields ~24 stable ones; observing 0/24 failures bounds the
false-negative rate at roughly 12% (rule of three), against ~20% from the 15
episodes available now.

### Operationalisation

- `modal(ep, order)` — most frequent parsed action over the 10 samples; ties
  broken by action-set order. Fewer than 6 of 10 parsing ⇒ the episode is
  dropped and counted as a dropout.
- **`U(ep)`** — `max` pairwise total variation across the 6 orderings **> 0.5**.
- **`C(ep)`** — `modal(ep, canonical) == truth`.

**On the 0.5 threshold.** It is arbitrary in value and near-irrelevant in
effect: across three protocols the ordering effect has been bimodal, clustering
at 0.000 and ~1.000 with almost nothing between, so any cutoff in (0.1, 0.9)
classifies identically. It is fixed at 0.5 now, and a **sensitivity table across
cutoffs 0.1–0.9 is reported regardless of outcome** so the claim cannot rest on
the choice.

`C` is evaluated at the canonical ordering only: a detector must be usable from
the answer you would actually have shipped.

---

## 5. Analysis plan, fixed now

**Primary.** Fisher exact, one-sided, on the 2×2 of `U` × `¬C`.

**Supported requires all three:**

1. `p < 0.01`
2. Risk difference `P(¬C|U) − P(¬C|¬U) ≥ 0.30`
3. At least **10** unstable and **10** stable episodes

**Secondary, reported regardless of the primary outcome:** the false-negative
count `P(¬C | ¬U)` with an exact (Clopper–Pearson) 95% upper bound.

**Also reported regardless:** the full 2×2, both conditional error rates, risk
difference with a bootstrap CI, overall accuracy, instability rate, dropouts,
and the threshold-sensitivity table.

**Pre-specified outcomes:**

- **SUPPORTED** — all three criteria met.
- **NOT REPLICATED** — margins adequate but `p ≥ 0.01` or risk difference
  `< 0.30`. This substantially weakens the v2/v3 observations and will be
  reported as such.
- **DEGENERATE** — either margin under 10. Uninformative; no claim either way.

**No optional stopping.** All 48 episodes are collected before any statistic is
computed. The analysis runs once. If the batch returns partial results, the
missing episodes are reported as dropouts rather than replaced.

---

## 6. What a positive result would and would not establish

**Would:** that on two models, in the scripted-scout family, permutation
instability carries information about correctness at an actionable effect size,
under a pre-registered test.

**Would not:** that instability causes error; that the effect holds in any other
task family (PID-1 tried and could not test it); that it holds on
non-synthetic tasks; that any threshold generalises; or that the detector is
calibrated. Each needs its own study.

**Relationship to FOIL.** Independent. Phase 0 was not passed under any protocol
and a positive PID-2 result does not reopen Phase 1.

---

## 7. Amendment log

| Date | Section | Change | Reason |
|---|---|---|---|
| _(none)_ | | | |
