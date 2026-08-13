# Phase 0 v3: KILL — and this time it is a verdict about the instrument

**Date:** 2026-08-13
**Model:** `claude-sonnet-5`
**Protocol:** v3 — environment decisiveness required analytically in advance
(PREREGISTRATION.md §12, amendment block)
**Volume:** 12 episodes × 11 arms × 50 samples = 6,600 calls. 0 unparseable,
0 API failures, 0 short arms. Cost $8.04.
*(The run reported $12.06; it began before the price-table correction and the
executor loads prices at construction. Token counts are unchanged.)*

---

## 1. Verdict

**KILL**, by a factor of 6.25.

| Quantity | v1 | v2 | **v3** |
|---|---|---|---|
| `T_null` (95th pct) | 0.995 | 1.000 | **1.000** |
| `T_ablate` (median) | 0.055 | **0.000** | **0.320** |
| Kill threshold | 0.028 | — | **0.160** |
| Verdict | KILL | INDETERMINATE | **KILL** |

**The number that matters is `T_ablate = 0.320`.** In v2 it was exactly zero,
which is why that run was uninformative — there was no signal for a noise floor
to be compared against. The analytic decisiveness requirement did what it was
built to do: ablations that moved nothing fell from 81% to 35%, and the median
ablation now shifts the action distribution by a third.

So there is signal. And the worst-case ordering noise still exceeds half of it
by more than six times.

This is the first of the three runs whose verdict is **not** an artifact of
environment design. v1 was under-determined, v2 over-determined; v3's
environment was constructed so that at least 3 of 4 sources are decisive, and
that property was verified over 200 seeds — re-derived from the rendered report
text, not from internal state — before a single token was spent.

## 2. The statistic decides the verdict, and a reader should know that

The kill rule compares the **95th percentile** of null comparisons against the
**median** ablation. Under that rule v3 fails badly. Under a median-to-median
comparison it would pass cleanly:

| comparison | noise | signal | outcome |
|---|---|---|---|
| p95 null vs median ablation *(pre-registered)* | 1.000 | 0.320 | **KILL** |
| median null vs median ablation | 0.000 | 0.320 | would PASS |

Both numbers are real. The ordering effect is **bimodal**: most reorderings
change nothing at all, and some invert the answer completely. "Typical noise" is
therefore zero and "worst-case noise" is total, and which one you compare
against determines the result.

The pre-registered rule uses the tail deliberately, and the reason survives
inspection: **6 of 12 episodes were bistable, and nothing in the measurement
tells you in advance which 6.** An instrument that is exact on half its inputs
and inverted on the other half, with no way to tell them apart at measurement
time, is not usable — a per-episode attribution is either fine or backwards and
you cannot know which. That is the argument for the tail statistic, and it was
fixed before any data existed. But the median-to-median reading is stated here
rather than buried, because a reader who prefers it should be able to see it.

## 3. Order sensitivity rose as the environment improved

| protocol | environment | bistable episodes |
|---|---|---|
| v1 | under-determined | (single episode, bistable) |
| v2 | over-determined | 3/12 (25%) |
| v3 | decisiveness enforced | **6/12 (50%)** |

This is coherent rather than surprising. When every source is decisive, the
answer genuinely depends on tracking each of them, so there is more for a
reordering to disturb. In v2's over-determined environment most episodes were
pinned regardless of presentation order — which is the same fact that made
`T_ablate` zero.

The bimodal signature held across all three: values cluster at 0.000 and ~1.000,
almost nothing between. Three protocols, three environments, one structure.

## 4. Exploratory: the detector signal replicated in direction

**Still exploratory.** The hypothesis came from v2 and this is a second look at
data collected for another purpose. It is not a confirmatory test, and
PID-1 — the only pre-registered attempt so far — was uninformative.

v3 alone:

|  | wrong | right |
|---|---|---|
| bistable | 4 | 2 |
| stable | **0** | 6 |

Fisher exact one-sided **p = 0.030**.

Pooled with v2 (24 episodes across two environments):

|  | wrong | right |
|---|---|---|
| bistable | 7 | 2 |
| stable | **0** | 15 |

Fisher exact one-sided **p = 0.000104**.

The asymmetry is the interesting part, and it sharpens the claim:

- **Stability implied correctness in 15 of 15 cases.** No stable episode was
  ever answered wrong, across two environments.
- **Instability did not imply error** — 2 of 9 bistable episodes were answered
  correctly.

So in this data instability behaves as a **necessary but not sufficient** marker
of error: a screen with no false negatives and some false positives. That is a
more useful shape than a symmetric predictor, and it is also the shape most
vulnerable to a small sample — one stable-but-wrong episode would break it.

Pooling two exploratory observations does not make either confirmatory. The
pooled p-value is reported because withholding it would be its own distortion,
not because it settles anything.

## 5. What Phase 0 established, across three protocols

**Established:**

- On a decisiveness-guaranteed environment, worst-case presentation-order
  effects exceed half the typical single-source ablation effect by 6.25×.
  Ablation-based attribution on this model and task family is dominated by an
  artifact of arrangement.
- Order sensitivity is bimodal, not graded, and its incidence *rises* as an
  environment is made more suitable for attribution.
- Environment design dominated two of three runs; only the third produced a
  verdict about the method.

**Not established:**

- That this generalises beyond `claude-sonnet-5` and the scripted-scout family.
- That the residual metric fails. It has still never been computed, in any run.
- Anything confirmatory about permutation-instability detection.

**Phase 0 is not passed. The project does not advance to Phase 1.** Three
protocols, three halts, one of which is finally a statement about the
instrument.

## 6. Data availability

- `runs/nulls-v3-claude-sonnet-5.json` — pooled and per-episode results.
- `data/phase0-v3-result.json` — archived copy.

Reproduce with `python3 cli.py --n 50 run3 --episodes 12` (~6,600 calls, ~$8.04).
