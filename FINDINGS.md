# FOIL — findings index

A measurement harness for multiagent epistemics, and the results it returned.
Every result below is published whether or not it was the one hoped for; the
pre-registration required that in advance.

**Headline: FOIL's Phase 0 gate has not been passed under any protocol.** The
residual metric the project was built to test has never been computed, because
the harness never cleared the noise-floor check designed to gate it.

Two of the three failures were faults in the environment rather than the
instrument. **The third was not.** With decisiveness guaranteed analytically
before any model call — so that ablation demonstrably moves the answer
(`T_ablate` = 0.320, against 0.000 in the prior protocol) — worst-case
presentation-order noise still exceeded half the typical ablation effect by
6.25×. That is a verdict about ablation-based attribution, not about a
misbuilt scout scenario.

---

## The results

| # | Result | Verdict | Document |
|---|---|---|---|
| 1 | Phase 0 v1 — single episode | **KILL**, 36× over threshold | [RESULTS-phase0.md](RESULTS-phase0.md) |
| 2 | Phase 0 v2 — 12 episodes, full coverage | **INDETERMINATE** | [RESULTS-phase0-v2.md](RESULTS-phase0-v2.md) |
| 3 | Phase 0 v3 — analytic decisiveness requirement | **KILL**, 6.25× over threshold | [RESULTS-phase0-v3.md](RESULTS-phase0-v3.md) |
| 4 | PID-1 — detection, new task family | **DEGENERATE** (untestable) | [RESULTS-PID.md](RESULTS-PID.md) |
| 5 | PID-2 — detection, pre-registered | **SUPPORTED** (p = 0.00003) | [RESULTS-PID2.md](RESULTS-PID2.md) |
| 6 | TURN — speaking order in group deliberation | **DEGENERATE** (invalid task) | [RESULTS-TURN.md](RESULTS-TURN.md) |
| 7 | TURN-2 — same question, repaired design | **NOT SUPPORTED** | [RESULTS-TURN2.md](RESULTS-TURN2.md) |
| 8 | Real traffic — does the phenomenon occur? | **NOT MEASURABLE** — 3 eligible points in 1,855 sessions | [RESULTS-REALTRAFFIC.md](RESULTS-REALTRAFFIC.md) |
| 9 | TURN-3 — test the TURN-2 reversal properly | **GATE 2 FAILED** — blocked for $1.56 | [RESULTS-TURN3.md](RESULTS-TURN3.md) |

Pre-registrations: [PREREGISTRATION.md](PREREGISTRATION.md) (FOIL),
[PREREGISTRATION-PID.md](PREREGISTRATION-PID.md) (PID-1),
[PREREGISTRATION-PID2.md](PREREGISTRATION-PID2.md) (PID-2),
[PREREGISTRATION-TURN.md](PREREGISTRATION-TURN.md) (TURN).
Interactive site: **https://foil-9vg.pages.dev**

---

## 1. Presentation order can invert a decision

Reordering four byte-identical sentences moved a scored decision from 99%
correct to 99.5% wrong, with near-zero sampling variance *within* each
ordering. Neither recency nor primacy explains it: two orderings differing by a
single adjacent swap of two honest sources gave 1.000 and 0.005.

**Consequence for anyone running ablation studies on LM agents.** Removing a
source changes two things at once — the information available, and the
arrangement of everything after it. If arrangement alone moves the outcome
further than removal does, an attribution score computed over
differently-arranged coalitions is partly measuring the arrangement. FOIL's own
principal cost optimisation (evaluating coalitions in a rearranged order to
share a cached prefix) was killed by this result. It would have looked like it
was working.

## 2. Order sensitivity is bimodal, and rises as the environment improves

Across three protocols the same signature held: episodes are immune or they
invert, with almost nothing between. Bistable episodes went 25% → 50% as the
environment was made *more* suitable for attribution, because when every source
is decisive there is more for a reordering to disturb. Episodes are not *somewhat* order-sensitive; they
are immune or they flip. Any central summary — a mean, a median — hides this
by construction, which is why inverted orderings are now counted as a
first-class statistic rather than averaged over.

## 3. Redundant sources are not interchangeable

Two sources emitting byte-identical text contributed unequally: ablating one
moved the distribution 0.830, ablating the other 0.055. Meanwhile ablating the
*liar* moved it 0.010 — the listener had already discounted it. Attribution
schemes that assume identical evidence is fungible are assuming something
false.

## 4. Environment design dominated every result

Both Phase 0 failures were faults in the environment, in opposite directions:

- **v1, under-determined.** An action covered by no source gave the listener a
  coherent "unruled-out therefore clear" rival hypothesis, and a quarter of
  episodes sat on the resulting knife edge.
- **v2, over-determined.** Guaranteeing full coverage removed the rival
  hypothesis and left the survivors able to pin the answer after any single
  removal. 81% of ablations moved the distribution by exactly zero.

The generator never enforced that a source be *decisive*, though the
pre-registration always named the redundancy matrix a controlled variable.
Protocol v3 fixes this with a requirement checked **analytically, before any
model call** — deliberately not calibrated against a measured outcome, because
v2's amendment was principled and still failed for exactly that reason.

## 5. Permutation instability predicts error — but does not guarantee its absence

Noticed exploratorily in v2/v3, then tested under its own pre-registration on a
second model and disjoint episodes. **Supported**, with a large effect:

| | wrong | right |
|---|---|---|
| unstable | 10 | 4 |
| stable | 3 | 31 |

P(wrong \| unstable) = 0.714 against P(wrong \| stable) = 0.088 — a likelihood
ratio of 8.1, Fisher one-sided p = 0.00003, robust across every instability
cutoff from 0.1 to 0.9. As a screen it flags 29% of episodes and catches 77% of
errors, needing no ground truth, no normative baseline and no attribution.

**And it refuted the stronger claim.** The exploratory data showed no stable
episode ever answered wrong (0 of 15). The pre-registered run, sized
deliberately to break that, found 3 of 34 — a false-negative rate of 8.8% (95%
upper bound 21%). Two of the three were *perfectly* stable across all six
orderings and confidently wrong.

Stability is a useful signal, not a proof of correctness. The direction
replicated; the absolute claim did not.

---

## On method

Three things this project did that are worth copying, and one worth avoiding.

**Worth copying.** The kill rule was written before the data and honoured when
it fired. Every amendment made after seeing a result is fenced and labelled
post-hoc, so a reader can tell which decisions preceded the data. Exploratory
findings are never reported with confirmatory language, however tempting the
p-value.

**Worth avoiding, one.** Phase 0 was amended once after a failure, and the
amended protocol failed too. Adjusting a design against observed outcomes is how
you end up tuning until the gate passes. The v3 requirement is stated
analytically and verified over 200 seeds at zero cost precisely to break that
loop.

**Worth avoiding, two — and this one caught us.** An analytic gate confers no
construct validity. TURN verified its scenarios over 200 seeds, re-derived from
rendered text rather than internals, and the gate was still worthless: it
confirmed consistency with a reference scorer nobody had checked against the
judgement being made. Every incorrect run was a *unanimous* vote for the option
that scorer ranked last — the panel was reasoning defensibly and the ground
truth was wrong. A gate can be rigorous, reproducible, and confirm the wrong
thing. Validate the ground truth against the model before running the design
that depends on it.

## Cost

Every result above was produced with metered Messages API calls; a Claude Code
subscription cannot serve them. Total spend for the six executed studies:

| study | calls | cost |
|---|---|---|
| Phase 0 v1 | 1,793 | $2.19 |
| Phase 0 v2 | 6,250 | $7.67 |
| Phase 0 v3 | 6,600 | $8.04 |
| PID-1 | 800 | $2.53 |
| PID-2 | 2,880 | $4.40 |
| TURN | 1,728 | $6.47 |
| TURN-2 | 4,272 | $15.71 |
| **total** | **24,323** | **~$47.01** |

TURN-2 has since run: its calibration gate passed 36/36 and the main study
returned NOT SUPPORTED.

## Reproduction

```bash
python3 cli.py plan                       # cost projection, zero API calls
python3 cli.py --n 200 run                # v1  (~2,200 calls, ~$2.19)
python3 cli.py --n 50 run2 --episodes 12  # v2  (~6,250 calls, ~$7.67)
python3 cli.py --n 50 run3 --episodes 12  # v3  (~6,600 calls, ~$8)
python3 pid_run.py                        # PID (800 calls on opus, ~$4.50)
```

Python 3.11 and numpy; no other dependencies, so the harness runs unchanged
inside a lab. Determinism is at the request layer only — environments,
payloads and fork keys reproduce exactly; sampled responses do not. MIT.
