# TURN: DEGENERATE — and the task was invalid in a way the gate could not see

**Date:** 2026-08-13
**Model:** `claude-sonnet-5`
**Pre-registration:** [PREREGISTRATION-TURN.md](PREREGISTRATION-TURN.md),
committed before any data existed (commit `4298a58`).
**Volume:** 24 scenarios × 2 conditions × 3 replicates = 144 deliberations,
1,728 calls. 0 dropouts, 0 API failures. Cost $6.47.

---

## 1. Outcome

**DEGENERATE.** Only 8 of 24 scenarios were untied, against a pre-registered
minimum of 16. No claim is made about H1 in either direction.

| | holder first | holder last |
|---|---|---|
| Group correct | 62/72 (0.861) | 61/72 (0.847) |
| Private fact uttered | 71/72 (0.986) | 69/72 (0.958) |

| H1 | value |
|---|---|
| Scenarios favouring *first* | 4 |
| Scenarios favouring *last* | 4 |
| Untied | **8** (needed ≥ 16) |
| Mean Δ | +0.014 (needed ≥ 0.20) |
| Sign test | p = 0.64 |

There are two separate defects here. The second is much worse than the first.

---

## 2. Defect one: the margin criterion was unsatisfiable in expectation

With 3 replicates per condition and a binary outcome, per-scenario Δ can only
take 7 values, and ties dominate. At the observed accuracy (p ≈ 0.854):

> P(Δ = 0) = Σ P(X=k)² = **0.493**, so the expected number of untied scenarios
> out of 24 is **12.2**.

**The pre-registered minimum was 16.** The experiment could not have met its own
margin criterion in expectation, under the null *or* under a moderate effect. I
sized `S = 24` against the effect I hoped to detect and never computed the tie
rate the sign test depends on. That is a power-analysis failure, fixed in
advance in the wrong direction, and it is mine.

The arithmetic was available before spending anything. Two lines of it would
have shown that 24 scenarios at 3 replicates cannot clear 16 untied — either
`S ≈ 32`, or `R = 5`+, or a clustered run-level test was needed.

---

## 3. Defect two: the ground truth did not survive contact

This is the one that matters, and it nearly got reported as a finding.

**All 21 incorrect runs were unanimous 4–0 votes for the *third* candidate** —
the option the reference scorer ranks **last**. Not one wrong run picked the
decoy. Not one was a split vote.

```
group_answer == decoy : 0
group_answer == truth : 0
group_answer == third : 21
```

A group that fails through consensus suppression falls for the decoy the shared
record favours, or splits. Unanimous convergence on the option my scorer ranks
bottom is not a group failure. **It is my scorer being wrong.**

### Why

The reference scorer counts positive facts at **equal weight**: decoy 3, truth
2, third 1. The agents weigh what the facts *say*. Given

- *"Bellweather published the reference work on failover"* (third, 1 fact), and
- *"Renwick shipped the telemetry platform end to end"* + *"turned around the
  failing indexing programme"* (truth, 2 facts),

the panel repeatedly judged the first candidate stronger. That is a defensible
reading. My weighting was arbitrary and I never checked it against the judgement
actually being made.

### The lesson, which generalises past this experiment

Phase 0 v3 established a discipline that worked: **verify the environment's
defining property analytically, before spending.** TURN did exactly that. The
gate passed 200/200 seeds and I re-derived it from the rendered text rather than
internals.

**The gate was still worthless here**, because it verified internal consistency
with a scorer whose validity nobody had checked. Analytic verification is only
as strong as the property being the right property. A gate can be rigorous,
reproducible, and confirm the wrong thing.

Concretely: the 15% "error rate" is not an error rate, and the 21
"pressed-and-ignored" runs are not evidence about consensus suppression. They
are 21 cases of a mis-specified ground truth. Reporting them as a finding about
multiagent epistemics would have been the worst outcome of this session.

---

## 4. What this does and does not establish

**Establishes:** nothing about H1 or H2. The experiment did not test what it
claimed to test.

**Suggestive but not licensed:** the position effect appears small — 62/72
versus 61/72 at the run level, mean Δ = +0.014. But since "correct" was
mis-specified, even this descriptive reading is unreliable; it compares
conditions against a label that does not track what the group was doing.

**Does establish, about method:** an analytic admissibility gate does not confer
construct validity. Both must be checked, and only the first is cheap.

---

## 5. What a valid attempt would change

Per §5 of the pre-registration a degenerate outcome "is not a licence to retune
and rerun: any such rerun is a separately labelled attempt." Recorded here for
whoever runs it:

1. **Remove the escape hatch.** Two candidates, not three. The private fact
   disqualifies the leader; the only alternative is the true answer. There is
   then no third option for a defensible disagreement to land on.
2. **Validate the ground truth against the model, not the scorer.** Before the
   real run, check that a panel with *full* information (every agent holding the
   private fact) reaches the intended answer at a high rate. If it does not, the
   scenario is not measuring suppression — it is measuring disagreement with the
   experimenter. This is cheap and would have caught the defect for a few
   dollars.
3. **Fix the power arithmetic first.** Compute the tie rate at the expected
   accuracy and size `S` and `R` from it, rather than setting a margin and
   hoping.

Note that (2) is the calibration step PID-2 taught and TURN skipped. The lesson
was available and I did not carry it across.

## 6. Data availability

- `runs/turn-result-claude-sonnet-5.json` — per-run votes, transcripts flags,
  per-scenario deltas, and the ledger.

Reproduce with `python3 turn_run.py` (1,728 calls, ~$6.47).
