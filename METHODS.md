# METHODS — the standing protocol

One copy of the conventions every FOIL study follows. Pre-registrations cite
this file and state only what is specific to their study; restating a rule here
verbatim is how documents drift apart, and drift is how the price table went
wrong twice.

Each rule names the study that taught it. None of them arrived by foresight.

## 1. Pre-registration

- Hypothesis, statistics, thresholds, and every failure outcome are committed
  to version control **before any data exists**. The commit hash is the proof.
- A revised environment or scenario family is a **new attempt** with its own
  pre-registration. Retuning a constant until a gate clears is a search
  procedure, and its result is the search, not the world. *(TURN-3: the gate
  failed by one run and the family was rejected, not nudged.)*
- One-sided tests do not flip after seeing the sign. An effect opposite to the
  registered direction is reported as descriptive, and claiming it requires a
  fresh pre-registered study. *(TURN-2: last-speaker advantage, still
  unclaimed; the confirmatory attempt, TURN-3, died at its gate.)*
- Exploratory analyses are welcome, labelled **EXPLORATORY**, dated, and never
  pooled with confirmatory data. Where an exploratory split partly restates a
  prior finding, the write-up says so before the number. *(SWEEP §7 is the
  worked example, circularity caveat first.)*

## 2. Gates, in order, each blocking

1. **Structural** (free): the generator's property, re-derived from **rendered
   text**, never from fields the generator set — a checker that reads
   `case.truth` agrees with the generator by construction. *(v1/v2.)*
2. **Power / resolution** (free): `evalgate.gates.plan()` sizes accept/reject
   gates; `evalgate.power.separate()` sizes rate comparisons; `tie_rate()`
   checks that a sign test can even be fed. Run before registering n, not
   after spending it. *(TURN-1 at 40% power; SWEEP at n=12 where separating
   its rates needed n≈153.)*
3. **Construct** (paid): a full-information, no-manipulation panel must reach
   the intended answer. Passing structural confers nothing here. *(TURN-1:
   structural gate passed 200/200 and every "wrong" run was a unanimous vote
   against our scorer.)*
4. **Variance** (paid): under the manipulation, the model must actually err.
   Construct and variance want opposite answers from different oracles.
   *(PID-1: 40/40 correct, nothing to detect.)* Degeneracy checks apply **per
   arm** — one arm at ceiling already bounds the contrast. *(TURN-2's hole,
   found by TURN-3.)*

Gate verdicts use the three-outcome rule in `evalgate.gates`: PASS, FAIL only
on an exact binomial rejection, EXTEND otherwise — extension size and count
fixed in advance, unresolved gates resolve to FAIL. A point-estimate cliff at
small n is a coin flip wearing a rule's clothes. *(TURN-3's 32/36, p = 0.49.)*

## 3. Running

- No optional stopping. A run completes its registered size or is void.
- Dropouts and API failures are counted and reported; above 5% the result is
  labelled compromised.
- Every rate carries a **Wilson 95% interval** (`evalgate.power.wilson` — the
  only copy). Overlapping intervals are *indistinguishable*, never a ranking.
- Models that fail to serve are reported with the error, not dropped. Models
  skipped for cost are disclosed in the results, not removed from the table.
  *(SWEEP §5.)*
- Determinism is at the request layer: environments, payloads, fork keys, and
  orderings reproduce exactly; sampled responses do not.

## 4. Reporting

- Negative results, killed studies, and invalid environments are published at
  the same standard as positive ones. The project's ledger lives in
  [FINDINGS.md](FINDINGS.md); its invalid environments ship as importable
  fixtures (`evalgate.fixtures`).
- Trend claims require monotone movement across at least three consecutive
  releases in one family. Two points are not a trend, and a window chosen
  after seeing the data is not a series. *(SWEEP §2.)*
- Corrections are made in place with the reason stated, and a claim retracted
  is retracted plainly. *(The price table, twice; orderprobe's agent-loop
  claim, RESULTS-REALTRAFFIC.md.)*
- Every result file records provenance: git commit, dirty flag, and the
  generating script (`foil/provenance.py`, from item 5 of the 2026-08-17
  tooling review).

## 5. Money

- Costs are computed from `data/prices-anthropic.json` and reported per study.
- The cheap gates run first because they are cheap: TURN-3's construct gate
  cost $1.56 and blocked an $18 run that would have measured nothing.
