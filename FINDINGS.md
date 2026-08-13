# FOIL — findings index

A measurement harness for multiagent epistemics, and the results it returned.
Every result below is published whether or not it was the one hoped for; the
pre-registration required that in advance.

**Headline: FOIL's Phase 0 gate has not been passed under any protocol.** The
residual metric the project was built to test has never been computed, because
the harness never cleared the noise-floor check that was designed to gate it.
Two of the three failures were faults in the environment, not the instrument —
which is itself the most useful thing learned, and the least expected.

---

## The results

| # | Result | Verdict | Document |
|---|---|---|---|
| 1 | Phase 0 v1 — single episode | **KILL**, 36× over threshold | [RESULTS-phase0.md](RESULTS-phase0.md) |
| 2 | Phase 0 v2 — 12 episodes, full coverage | **INDETERMINATE** | [RESULTS-phase0-v2.md](RESULTS-phase0-v2.md) |
| 3 | Phase 0 v3 — analytic decisiveness requirement | see document | [RESULTS-phase0-v3.md](RESULTS-phase0-v3.md) |
| 4 | PID — permutation-instability detection | see document | [RESULTS-PID.md](RESULTS-PID.md) |

Pre-registrations: [PREREGISTRATION.md](PREREGISTRATION.md) (FOIL),
[PREREGISTRATION-PID.md](PREREGISTRATION-PID.md) (PID).
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

## 2. Order sensitivity is bimodal, not graded

Across 12 episodes, nine were perfectly stable across all six orderings and
three inverted completely. Episodes are not *somewhat* order-sensitive; they
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

## 5. A lead worth testing separately

In v2, the three episodes that inverted under reordering were exactly the three
answered incorrectly, at ~1.00 confidence (Fisher exact one-sided p = 0.0045,
n = 12). If it holds, it is an error signal needing no ground truth, no
normative baseline, no attribution, and no known reliabilities — sidestepping
every limitation that made FOIL's original metric hard to defend.

It was noticed after the data and the test was chosen after the pattern, so it
is a lead, not a result. It is being tested under its own pre-registration, on
a model and a task family it was **not** discovered on.

---

## On method

Three things this project did that are worth copying, and one worth avoiding.

**Worth copying.** The kill rule was written before the data and honoured when
it fired. Every amendment made after seeing a result is fenced and labelled
post-hoc, so a reader can tell which decisions preceded the data. Exploratory
findings are never reported with confirmatory language, however tempting the
p-value.

**Worth avoiding.** Phase 0 was amended once after a failure, and the amended
protocol failed too. Adjusting a design against observed outcomes is how you end
up tuning until the gate passes. The v3 requirement is stated analytically and
verified over 200 seeds at zero cost precisely to break that loop — the
requirement is a property of the episode's information structure, not of any
number a model produced.

## Reproduction

```bash
python3 cli.py plan                       # cost projection, zero API calls
python3 cli.py --n 200 run                # v1  (~2,200 calls, ~$3.30)
python3 cli.py --n 50 run2 --episodes 12  # v2  (~6,250 calls, ~$11.50)
python3 cli.py --n 50 run3 --episodes 12  # v3  (~6,600 calls, ~$12)
python3 pid_run.py                        # PID (800 calls on opus, ~$4)
```

Python 3.11 and numpy; no other dependencies, so the harness runs unchanged
inside a lab. Determinism is at the request layer only — environments,
payloads and fork keys reproduce exactly; sampled responses do not. MIT.
