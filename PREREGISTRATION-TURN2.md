# TURN-2: Does speaking order determine group consensus?

## Pre-Registration — written, verified, and NOT run

**Written:** 2026-08-13, before any TURN-2 data exists.
**Run status:** **NOT RUN.** Everything in this document that can be verified
without an API call has been. The experiment itself is unexecuted and awaits a
decision to spend.
**Supersedes:** nothing. [PREREGISTRATION-TURN.md](PREREGISTRATION-TURN.md) and
its DEGENERATE outcome ([RESULTS-TURN.md](RESULTS-TURN.md)) stand unrevised.
This is a separately labelled attempt, as §5 of that document requires.

---

## 1. Why TURN-1 failed, and what changes

TURN-1 returned DEGENERATE for two independent reasons, both mine:

| defect | consequence | fix here |
|---|---|---|
| **Ground truth mis-specified.** The reference scorer counted positive facts at equal weight; the panel weighed what facts *said*. All 21 "incorrect" runs were unanimous 4–0 votes for the option the scorer ranked **last**. | The 15% "error rate" measured disagreement with the experimenter, not group failure. | §3: two candidates, and a **mandatory calibration gate** that validates ground truth against the model. |
| **Underpowered by construction.** 3 replicates gave P(tie) = 0.49, so expected untied scenarios was 12.2 against a required minimum of 16. | The margin criterion was unsatisfiable in expectation, under the null *or* a real effect. | §4: design sized by simulation; §5: a test that uses magnitudes rather than discarding ties. |

The deeper lesson, recorded because it generalises: **TURN-1 did run an analytic
admissibility gate, over 200 seeds, re-derived from rendered text.** It passed,
and it was worthless — it confirmed internal consistency with a scorer nobody
had validated. An analytic gate confers no construct validity. Both must be
checked, and only the first is cheap.

---

## 2. Hypotheses

**H1 (primary).** A hidden-profile group is more likely to reach the correct
answer when the holder of the decisive private fact speaks **first** than when
it speaks **last**.

> P(correct | holder first) > P(correct | holder last), one-sided.

**H2 (secondary).** The private fact is uttered less often when its holder
speaks later — separating *never volunteered* from *volunteered and ignored*.

**A null is a real result.** If position does not matter, deliberation is robust
to arrangement, which would materially limit the reach of this project's
single-listener findings (Phase 0, PID-2). That is worth publishing and is
pre-specified as such.

---

## 3. Task, and the calibration gate that must pass first

**Two candidates, not three.** TURN-1 used three, and every failure escaped to
the third option — a defensible disagreement had somewhere to land. With two,
the private fact disqualifies the leader and the only alternative is the
intended answer.

| | content |
|---|---|
| Shared facts | Held by all four agents; favour candidate **A**. |
| Private fact | Held by exactly one agent; disqualifies **A**, leaving **B**. |

### Gate 1 — analytic (free, run before anything)

Shared facts alone must uniquely favour A; shared plus private must uniquely
favour B; the private fact's unique token must appear nowhere in the shared
record. Verified across 200 seeds, re-derived from rendered text.

### Gate 2 — construct validity (costs money, and is the point)

**This is the step TURN-1 skipped and the reason it failed.**

Before the main run, a **full-information panel** — all four agents holding the
private fact, no order manipulation — deliberates on 12 scenarios, 3 replicates.

> **Requirement: the full-information panel reaches candidate B in ≥ 90% of
> runs.**

If it does not, the scenarios do not encode the intended judgement, and **the
main run does not happen.** The scenario family is rejected and reported as
rejected. No retuning under this pre-registration; a revised family is a new
attempt.

Cost: 12 × 3 × 12 calls ≈ 432 calls, roughly $1.50. It is cheap precisely
because it is the check that matters most.

---

## 4. Design, sized by simulation rather than hope

Power was estimated by simulating the full analysis pipeline — scenario-level
difficulty variation, binomial replicate sampling, and the actual permutation
test — at α = 0.01 and a true effect of δ = 0.20:

| S | R | power (base 0.60) | power (base 0.75) | calls |
|---|---|---|---|---|
| 24 | 3 | 0.40 | 0.62 | 1,728 *(TURN-1's design)* |
| 24 | 5 | 0.78 | 0.82 | 2,880 |
| **32** | **5** | **0.90** | **0.96** | **3,840** |
| 40 | 7 | 0.99 | 1.00 | 6,720 |

False-positive rate at δ = 0 was 0.003, within the nominal 0.01.

**Chosen: S = 32 scenarios, R = 5 replicates per condition.** TURN-1 ran at 40%
power; this runs at ≥ 90%.

| Parameter | Value |
|---|---|
| Scenarios | 32 (seeds 3000+) |
| Conditions | 2 — holder speaks first / last |
| Replicates | 5 per (scenario, condition) |
| Deliberation | 4 agents × 2 rounds, then one vote each |
| Runs | 320 |
| Calls | 320 × 12 = **3,840** |
| Model | `claude-sonnet-5` |
| **Estimated cost** | **~$13**, plus ~$1.50 for Gate 2 |

---

## 5. Analysis plan, fixed now

**Unit of analysis is the scenario.** Runs within a scenario share a
construction and are not independent.

**Primary — paired permutation test on mean Δ**, one-sided, 200,000 resamples,
seed 0. Under H0 the condition labels are exchangeable within a scenario, so
each scenario's Δ may flip sign.

Chosen over TURN-1's sign test deliberately: the sign test discards tie
magnitude *and* every tied scenario, which is exactly what left TURN-1 unable to
clear its own margin. The permutation test uses the magnitudes and needs no
minimum-untied criterion at all.

**H1 supported requires both:**

1. `p < 0.01`
2. Mean Δ ≥ **0.15**

**Reported regardless:** the sign test as a secondary robustness check, both Δ
distributions, accuracy and utterance rate by condition, runs where the fact was
uttered but the group answered wrongly, the Gate-2 pass rate, dropouts, and the
ledger.

**Pre-specified outcomes:**

- **SUPPORTED** — both criteria met.
- **NOT SUPPORTED** — criteria unmet at ≥ 90% power. Reported as evidence that
  deliberation is comparatively robust to speaking order.
- **GATE 2 FAILED** — the scenario family does not encode the intended
  judgement. No main run. Reported as a task-construction result.

**No optional stopping.** All 320 runs complete before any statistic is
computed.

---

## 6. The intervention arm remains quarantined

Simultaneous reveal runs **only if H1 is SUPPORTED**, with its prediction
registered separately beforehand. If there is nothing to intervene on, an
intervention that appears to help is measuring noise.

---

## 7. What a positive result would and would not establish

**Would:** that in a synthetic two-candidate hidden profile on
`claude-sonnet-5`, the position of a decisive minority holder measurably changes
what a group concludes — making consensus partly a property of the schedule, and
turn order a lever an orchestrator or adversary can pull without touching
content.

**Would not:** that this holds on other models, other group sizes, other round
counts, or non-synthetic tasks; that position causes the effect by any
particular mechanism; or that any intervention fixes it.

---

## 8. Amendment log

| Date | Section | Change | Reason |
|---|---|---|---|
| _(none)_ | | | |
