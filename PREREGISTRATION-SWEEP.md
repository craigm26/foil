# SWEEP pre-registration — does order sensitivity change across model generations?

**Committed before any sweep data exists.**

---

## 1. Question

Phase 0 v3 established that reordering byte-identical inputs can invert a
single-listener decision, on `claude-sonnet-5`. It also found something
uncomfortable: order sensitivity **rose** as the environment improved, from 25%
of episodes bistable in v2 to 50% in v3, because when every source is decisive
there is more for a reordering to disturb.

That leaves an obvious question the project has not asked: is this a property of
one model, or does it vary across generations? This study measures it on a
**fixed environment** across every model with a price entry.

This is descriptive measurement, not hypothesis testing. There is no SUPPORTED
or NOT SUPPORTED outcome. The pre-registration exists to fix the metric and the
reporting rule **before** any model's number is visible, so no narrative can be
selected after the fact.

## 2. Environment, fixed

Phase 0 **v3**, unchanged: the analytically-verified scripted-scout environment
in `foil/env3.py`, where at least 3 of 4 sources are individually decisive and
the requirement is checked over 200 seeds before any call. Identical episodes,
identical rendered text, identical orderings for every model. Nothing about the
environment is tuned per model.

## 3. Models

Every model in `data/prices-anthropic.json` that the API will serve:

    claude-haiku-4-5
    claude-sonnet-4-5   claude-sonnet-4-6   claude-sonnet-5
    claude-opus-4-5     claude-opus-4-6     claude-opus-4-7
    claude-opus-4-8     claude-opus-5

Any model the API refuses is reported as unavailable with the error, and is not
silently dropped.

## 4. Statistics, fixed in advance

Per model, over **12 episodes × 6 orderings × 20 samples**:

| statistic | definition |
|---|---|
| **bistability rate** | fraction of episodes where two orderings produce different modal answers |
| **T_null** | 95th percentile total-variation shift across reorderings |
| **max inversion** | largest TV shift observed between any two orderings of one episode |
| **accuracy** | fraction of samples landing on the reference-correct route, canonical ordering |

Bistability rate is the headline. It is the statistic PID-2 showed predicts
error, and it needs no ground truth.

## 5. Reporting rule, declared before data

- **Every model that runs is reported.** No model is dropped for being
  inconvenient, an outlier, or off-narrative.
- **No claim of a generational trend** unless bistability moves monotonically
  across at least three consecutive releases in one family. Two points are not a
  trend and will not be described as one.
- Sampling error is real at n=12 episodes. Every rate is reported with a
  **Wilson 95% interval**, and differences whose intervals overlap are described
  as **indistinguishable**, not as a ranking.
- If the models are indistinguishable, that is the result and it is published as
  such.
- The environment is synthetic and single-listener. No result here transfers to
  deployed agent traffic, and TURN-2 already showed that group deliberation does
  not inherit the single-listener effect.

## 6. What this cannot show

It cannot show that any model is better or worse at the underlying task, because
order sensitivity is not task competence: a model that is confidently wrong in a
stable way scores *better* on bistability than one that wavers toward the right
answer. Accuracy is reported alongside precisely so bistability is not read as a
quality score.

It cannot attribute any difference to training, scale, or architecture. It
measures behaviour on one synthetic environment and nothing else.
