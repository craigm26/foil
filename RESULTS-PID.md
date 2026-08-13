# PID-1: DEGENERATE — the task could not test the hypothesis

**Date:** 2026-08-13
**Model:** `claude-opus-5`
**Pre-registration:** [PREREGISTRATION-PID.md](PREREGISTRATION-PID.md), written
before any data was collected.
**Volume:** 40 items × 4 permutations × 5 samples = 800 calls. 0 dropouts,
0 unparseable, 0 API failures. Cost $2.53.

---

## 1. Outcome

**DEGENERATE**, one of the three outcomes fixed in advance (§5). No claim is
made in either direction about the hypothesis.

|  | wrong | right |
|---|---|---|
| **unstable** | 0 | 0 |
| **stable** | 0 | 40 |

| Quantity | Value | Criterion |
|---|---|---|
| Accuracy | 1.000 (40/40) | — |
| Instability rate | 0.000 (0/40) | — |
| Unstable margin | **0** | ≥ 8 |
| Stable margin | 40 | ≥ 8 |
| Risk difference | undefined | ≥ 0.30 |
| p (one-sided) | undefined | < 0.01 |

`claude-opus-5` answered every item correctly under every permutation. With no
errors and no instability, both margins the test needs are empty. The
hypothesis was neither supported nor refuted; it was untestable on this
material.

## 2. The model was reading, not guessing

A perfect score invites the suspicion that the task was answerable without the
evidence. It was not:

- Modal answers spanned **6 distinct handlers** across the 40 items. A constant
  strategy would show one.
- The correct answer sat at option index 0/1/2/3 in 13/8/13/6 items, so
  always-picking-first scores 32% against a 25% chance floor.
- Every option appeared somewhere in each item's passages, so "named in the
  evidence" carries no signal.

The task was well-formed. It was simply easy: a 2-hop read over five short
passages is not a challenge for this model.

## 3. What this does and does not establish

**Establishes:** that 2-hop evidence-passage QA on `claude-opus-5` produces no
error variance, and therefore cannot host a test of an error detector.

**Does not establish:** anything about H_PID. A detector cannot be evaluated
where nothing is missed. In particular this is **not** evidence against the
v2 observation — it is silence, not disagreement.

**Does weaken the discovery-run result in one narrow sense:** the v2 pattern
was found where the listener erred on 3 of 12 episodes. If instability only
ever accompanies error, and this model on this task never errs, then the
detector has nothing to detect here. That is consistent with H_PID, and equally
consistent with the v2 pattern being an artifact of a degenerate environment.
Consistency with both is exactly what "uninformative" means.

## 4. What follows, and the constraint on it

§5 is explicit: a degenerate outcome "is not a licence to retune the task and
rerun: any such rerun is a new, separately labelled attempt."

So PID-2 is a new attempt, not a continuation:

1. **A calibration pilot** locates a difficulty at which the model errs on a
   usable fraction of items. It samples **one order per item** and therefore
   cannot compute instability at all — the quantity the hypothesis concerns is
   not observable from its data, so it cannot bias the test. Its items come
   from a seed range disjoint from PID-2's.
2. **A fresh pre-registration** fixes the hypothesis, statistic, thresholds and
   margins at the calibrated difficulty, before the test run.
3. PID-1 stands published, unrevised.

Calibrating task difficulty so a dependent variable has variance is ordinary
experimental design. Calibrating the hypothesis, the statistic, or the
thresholds would not be, and none of those move.

## 5. A ledger defect worth recording

The run reported `cost_usd: null` with the note *no price row for
'claude-opus-5'*. Two compounding causes:

- The vendored price table had no Opus 5 row at all. Longest-prefix matching
  found nothing and the ledger failed closed, as designed — a missing cost is
  correct behaviour where a guessed one would not be.
- The table was corrected *while this run was in flight*, and the executor
  loads prices at construction. Fixing a data file mid-run does not retroactively
  fix an already-running process.

The $2.53 above is recomputed from the recorded token counts against the
corrected table. Token counts were never in doubt; only the rate applied to them.

## 6. Data availability

- `runs/pid-result-claude-opus-5.json` — per-item modal answers across all four
  permutations, the 2×2 table, and the ledger.

Reproduce with `python3 pid_run.py` (800 calls, ~$2.53).
