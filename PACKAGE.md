# orderprobe + evalgate

Two zero-dependency tools for people building or running multiagent systems.

- **`orderprobe`** — a runtime brake trigger that needs no ground truth.
  Permute a decision's inputs; if the answer moves, stop and escalate.
  Pre-registered test: catches 77% of errors while stopping 29% of decisions
  (likelihood ratio 8.1, p = 0.00003).
- **`evalgate`** — three checks that prove an eval environment can measure what
  you claim, before you pay to run it. Two are free.

```bash
pip install orderprobe
```

Or vendor them — each package is a few hundred lines of stdlib Python with no
dependencies, and copying the directory into your repo works.

Evidence, limits, every negative result, and live demos:
**https://foil-9vg.pages.dev**

MIT.
