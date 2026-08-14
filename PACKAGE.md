# orderprobe + evalgate

Two zero-dependency tools for people building or running multiagent systems.

- **`orderprobe`** — a runtime brake trigger that needs no ground truth.
  Permute a decision's inputs; if the answer moves, stop and escalate.
  Pre-registered test: catches 77% of errors while stopping 29% of decisions
  (likelihood ratio 8.1, p = 0.00003). **Stability is not a guarantee** — 3 of
  34 stable episodes were wrong. It is triage, not proof. Point it at
  concurrently-gathered sources, **not** sequential agent loops; see
  [RESULTS-REALTRAFFIC.md](RESULTS-REALTRAFFIC.md) for why that distinction is
  load-bearing.
- **`evalgate`** — four checks that prove an eval environment can measure what
  you claim, before you pay to run it. Two are free. Ships with **five known-
  invalid environments as fixtures**, so you can test your own gate against
  environments that deserve to fail.

```bash
pip install git+https://github.com/craigm26/foil
```

Or vendor them — each package is a few hundred lines of stdlib Python with no
dependencies, and copying the directory into your repo works.

## Why the fixtures exist

A validity gate is untested code until you run it against something that should
fail it. We built eight environments for this project and **five were invalid**,
each producing results that looked perfectly analyzable:

```python
from evalgate import fixtures
print(fixtures.audit(my_check))
```

Two of the five are unreachable by any check that does not call a model, and the
audit says so rather than counting them against you. `scorer_disagrees` is the
one worth studying: it satisfies every structural property a valid environment
satisfies, over any number of seeds, re-derived from rendered text, and is
invalid anyway. A gate can be rigorous, reproducible, and confirm the wrong
thing.

## The two paid gates want opposite answers

`construct.verify` asks whether the model reaches your answer given full
information and no manipulation. It should score near 1.0.

`variance.verify` asks whether the model *ever* gets it wrong with the
manipulation applied. Near 1.0 there means there is nothing to detect.

PID-1 scored 40/40 and passed construct. It was invalid: no errors, so nothing
to measure. Passing one gate tells you nothing about the other.

Evidence, limits, every negative result, and live demos:
**https://foil-9vg.pages.dev**

MIT.
