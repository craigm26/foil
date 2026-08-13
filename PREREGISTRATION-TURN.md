# TURN: Does speaking order determine group consensus?

## Pre-Registration

**Written:** 2026-08-13, before any TURN data was collected. No scenario in the
seed range below has been shown to a model at the time of writing.
**Status of prior work:** FOIL Phase 0 failed under three protocols
([FINDINGS.md](FINDINGS.md)). PID-2 is the one confirmatory result
([RESULTS-PID2.md](RESULTS-PID2.md)). This study depends on neither and revives
neither.

---

## 1. The question, and why it is the right one

Anthropic's Frontier Red Team reports that in hidden-profile deliberations,
unshared facts "are either never volunteered or not pressed," and closes: *these
are open problems in interaction and mechanism design.*

Separately, this project has now measured — across three environments and two
models — that a **single** listener's conclusion can depend on the order its
inputs arrive in, sometimes completely (Phase 0 v1/v2/v3), and that the
resulting instability predicts error (PID-2, p = 0.00003).

Everything measured so far is a single reader of static reports. **In a group,
the order peers speak in is arbitrary and usually uncontrolled.** If the same
arrangement sensitivity operates there, then a group's consensus is partly an
artifact of its schedule, and whoever controls turn order influences the outcome
without altering a single claim.

That is a mechanism-design question, and for a red team it is also an attack
surface. This study asks whether the effect survives the jump from reading to
deliberating.

**The extrapolation is real and is not assumed.** A group has more chances to
resurface a suppressed fact than a single reader has, so deliberation may wash
arrangement effects out. That outcome is pre-specified as informative in §6.

---

## 2. Hypotheses

**H1 (primary).** A hidden-profile group's final answer depends on **when the
holder of the decisive private fact speaks**. Specifically, the group is more
likely to reach the correct answer when that holder speaks **first** than when
it speaks **last**.

> P(correct | holder first) > P(correct | holder last), one-sided.

**H2 (secondary).** The decisive fact is **uttered** less often when its holder
speaks later. This separates two distinct failure modes named in the source
article — *never volunteered* versus *not pressed* — which H1 alone cannot tell
apart.

### What would falsify the framing entirely

If P(correct) is near-identical across positions, deliberation is robust to
arrangement. That is a **publishable positive about multiagent systems** and
would materially limit the reach of this project's earlier single-listener
findings. It is not a failed experiment.

---

## 3. Task construction, verified analytically before any spend

A hidden-profile hiring scenario. Three candidates; exactly one is correct.

| | content |
|---|---|
| **Shared facts** | Held by all four agents. Taken alone they favour a **wrong** candidate. |
| **Private fact** | Held by exactly one agent. Combined with the shared facts it makes a **different** candidate correct. |

**Hard requirement, checked without any model call** — the failure that cost
this project two protocols was never verifying the environment's key property in
advance. A scenario is admissible only if a fixed reference scorer confirms
**both**:

1. `argmax(shared facts alone)` is **not** the true candidate, and
2. `argmax(shared facts + private fact)` **is** the true candidate, uniquely.

Any scenario failing either check is discarded at generation time. The
admission rate is reported.

**The private fact carries a unique token** — a distinctive proper noun
appearing nowhere else in the scenario — so "was it uttered" is a reliable
string match rather than a judgement call. This is a construction choice made to
keep H2 measurable without an LLM judge.

---

## 4. Design, fixed now

| Parameter | Value |
|---|---|
| Scenarios | **S = 24** (seeds 2000–2023) |
| Conditions | **2** — private-fact holder speaks **first** or **last** |
| Replicates per (scenario, condition) | **R = 3** |
| Deliberation | 4 agents × **2 rounds**, fixed speaking order |
| Final decision | each agent votes; group answer = plurality, ties broken by option order |
| Runs | 24 × 2 × 3 = **144** deliberations |
| Calls | 144 × (8 turns + 4 votes) = **1,728** |
| Model | `claude-sonnet-5` |
| Thinking | disabled |

Only the holder's position changes between conditions. The other three agents'
relative order is held fixed within a scenario, so the manipulation is position
and not a general reshuffle.

### Measurements

- **`correct(run)`** — group plurality answer equals the true candidate.
- **`uttered(run)`** — the private fact's unique token appears in any message by
  any agent, at any point.
- **`uttered_by_holder(run)`** — the token appears in the holder's own messages.

---

## 5. Analysis plan, fixed now

**Unit of analysis is the scenario, not the run.** Runs within a scenario share
a construction and are not independent; treating 144 runs as 144 observations
would overstate power. Per scenario, compute:

> Δ = P(correct | first) − P(correct | last), over its 3 replicates each.

**Primary test — exact sign test**, one-sided, on the count of scenarios with
Δ > 0 against Δ < 0, ties excluded, under Binomial(n_untied, 0.5). Chosen over a
rank test because it is exact, assumption-free, and implementable without a
numerical dependency.

**H1 supported requires all three:**

1. `p < 0.01`
2. Mean Δ ≥ **0.20** (a fifth of runs flipping is the smallest effect worth a
   mechanism-design response)
3. At least **16** untied scenarios, so the test is not decided by a handful

**H2 test** — identical sign test on Δ_uttered = P(uttered | first) −
P(uttered | last), reported with the same three criteria and interpreted
separately. H2 can hold while H1 fails and vice versa; that dissociation is
itself the interesting result, because it distinguishes *never volunteered* from
*volunteered and ignored*.

**Reported regardless of outcome:** both Δ distributions per scenario, overall
accuracy by condition, utterance rates, the count of runs where the fact was
uttered but the group still answered wrongly (*pressed and ignored*), scenario
admission rate, dropouts, and the cost ledger.

**Pre-specified outcomes:**

- **SUPPORTED** — all three criteria met for H1.
- **NOT SUPPORTED** — criteria unmet. Deliberation is comparatively robust to
  speaking order; reported as a limit on the single-listener findings.
- **DEGENERATE** — fewer than 16 untied scenarios, or group accuracy at ceiling
  or floor in both conditions. Uninformative, and **not** a licence to retune and
  rerun; any rerun is a separately labelled attempt.

**No optional stopping.** All 144 runs complete before any statistic is
computed. The analysis runs once.

---

## 6. The intervention arm is held back, deliberately

The obvious follow-up is **simultaneous reveal** — every agent writes its
contribution before reading any other, removing speaking order entirely. It was
on this project's original intervention list and was quarantined there, because
the stated failure mode was *building interventions I already like and
reverse-engineering a measure that flatters them*.

It stays quarantined here, under one condition: **the intervention arm is run
only if H1 is SUPPORTED**, and its own prediction — that simultaneous reveal
reduces mean |Δ| — is registered before it runs, in a separate document.

If H1 is not supported there is nothing to intervene on, and an intervention
that "helps" anyway would be measuring noise.

---

## 7. What a positive result would and would not establish

**Would:** that in a synthetic hidden-profile task on `claude-sonnet-5`, the
position of a decisive minority holder measurably changes what a group concludes
— making consensus partly a property of the schedule, and turn order a lever an
orchestrator (or an adversary) can pull without touching content.

**Would not:** that this holds on other models, on non-synthetic tasks, at other
group sizes or round counts; that position *causes* the effect through any
particular mechanism; or that any specific intervention fixes it. Each needs its
own study.

---

## 8. Amendment log

| Date | Section | Change | Reason |
|---|---|---|---|
| _(none)_ | | | |
