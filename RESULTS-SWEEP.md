# SWEEP (partial): accuracy improved across generations. Order sensitivity did not.

**Date:** 2026-08-13
**Pre-registration:** [PREREGISTRATION-SWEEP.md](PREREGISTRATION-SWEEP.md),
committed before any model was called.
**Status:** **4 of 9 models.** The five Opus models are **not run**, for cost, not
because of anything in these numbers. See §5 — this is a disclosure, not an
omission.
**Volume:** 5,760 calls. Cost **$6.57**.

---

## 1. Results

Identical episodes, identical orderings, identical environment (Phase 0 v3,
unchanged) for every model.

| model | bistable | Wilson 95% | T_null | accuracy |
|---|---|---|---|---|
| `claude-haiku-4-5` | 0.83 | [0.55, 0.95] | 1.000 | 0.458 |
| `claude-sonnet-4-5` | 0.50 | [0.25, 0.75] | 1.000 | 0.583 |
| `claude-sonnet-4-6` | 0.58 | [0.32, 0.81] | 1.000 | 0.692 |
| `claude-sonnet-5` | 0.42 | [0.19, 0.68] | 1.000 | 0.738 |

## 2. No trend is claimed, because the rule says so

The pre-registration permits a generational trend claim only if bistability moves
**monotonically across three consecutive releases in one family**.

Sonnet bistability runs **0.50 → 0.58 → 0.42**. Up, then down. Not monotone.
**No trend is claimed.**

It would have been easy to describe this as "order sensitivity fell from 0.58 to
0.42 in the newest release" by quietly starting at 4-6. That is the sentence the
rule exists to prevent, and it is not being written.

**Zero of the six model pairs have non-overlapping intervals.** Every model here
is statistically indistinguishable from every other on bistability. At 12
episodes the intervals are roughly ±0.25 wide, which is far too coarse to
separate 0.42 from 0.58. Per the pre-registration, that is the result and it is
published as such.

## 3. What does move: accuracy

| | 4-5 → 4-6 → 5 |
|---|---|
| Sonnet accuracy | 0.583 → 0.692 → 0.738, **monotone** |
| Sonnet bistability | 0.50 → 0.58 → 0.42, not monotone |

Accuracy rises cleanly across the family and across all four models
(0.458 → 0.738). Order sensitivity does not follow it.

This also **weakens a concern raised mid-run**: after the first two models it
looked as though bistability might simply restate task competence, since haiku
was both least accurate and most bistable. Sonnet-4-6 broke that — it is *more*
bistable than 4-5 while also *more* accurate. Bistability is not merely a
mislabelled accuracy score. That does not make it independent of difficulty
either; this design cannot separate them.

## 4. The finding that survives

**Every model produces complete inversions, and the newest is not exempt.**

| model | episodes with a total inversion (TV = 1.000) |
|---|---|
| `claude-haiku-4-5` | 6 of 12 |
| `claude-sonnet-4-5` | 5 of 12 |
| `claude-sonnet-4-6` | 4 of 12 |
| `claude-sonnet-5` | 5 of 12 |

T_null is **1.000 for every model tested**. On a third to a half of episodes,
reordering byte-identical inputs moves the answer distribution by the maximum
possible amount, and this is as true of the most accurate model in the set as of
the least.

**Stated precisely:** these counts are also mutually indistinguishable at n=12,
so the claim is *not* that inversion frequency is constant across generations.
The claim is narrower and safe: **no model tested is free of complete
inversions**, while accuracy over the same span improved by 28 points. Whatever
the generations bought, it was not immunity to input order.

## 5. Five models not run, and why

`claude-opus-4-5`, `-4-6`, `-4-7`, `-4-8`, and `claude-opus-5` were in the
pre-registered list and were **not run**. Projected cost was $40–60, against
$6.57 for the four here and $47 for the entire project to date.

The pre-registration forbids dropping a model for being inconvenient. This is not
that: the decision was made on price before any Opus number existed, and it is
disclosed here rather than being quietly reflected in a shorter table. Anyone can
run them: `./run.sh sweep_run.py --models claude-opus-5`.

The judgement behind it: at n=12 every interval already overlaps every other, so
five more models at the same sample size would add five more indistinguishable
points. The study is **sample-limited, not model-limited**. Money spent on more
episodes would buy resolution; money spent on more models at this resolution buys
very little.

## 6. Limits

- **Underpowered by design, and it shows.** 12 episodes per model gives ±0.25
  intervals. This can detect only very large differences. It found none.
- Synthetic, single-listener, one environment. TURN-2 already showed group
  deliberation does not inherit the single-listener effect.
- Bistability is not a quality score. A confidently wrong stable model scores
  *better* on it than one wavering toward the right answer, which is why accuracy
  sits beside it in every table above.
- No causal attribution to training, scale, or architecture is possible or
  attempted.

## 7. Data

- `runs/sweep-result.json` — per-episode modes, TVs, ledger, per-model cost.

Reproduce with `./run.sh sweep_run.py --models <list>`.
