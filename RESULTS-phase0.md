# Phase 0 Results: the kill rule fired

**Date:** 2026-08-13
**Model:** `claude-sonnet-5`
**Episode:** `scout-s1-low` (seed 1, low overlap), single decision point
**Volume:** 2,200 samples across 11 arms, 200 per arm. Zero API failures, zero
unparseable samples, zero short arms. Measured cost $3.29.

---

## 1. Verdict

**KILL.** Per the pre-registered rule (PREREGISTRATION.md §5.1):

| Quantity | Value |
|---|---|
| `T_null` (95th pct TV across N1 order pairs + N2 paraphrase) | **0.995** |
| `T_ablate` (median TV of single-source ablation) | **0.055** |
| Kill threshold (`0.5 x T_ablate`) | **0.0275** |

`T_null` exceeds the threshold by a factor of 36. The rule fired, and it did
not fire marginally.

This is the pre-registered outcome and it is reported as such. What follows is
diagnosis, not an attempt to talk the result out of existence.

---

## 2. The finding, verified

**A pure reordering of four identical sentences flips the decision from 99%
correct to 99.5% wrong.**

| order | permutation | P(correct action) |
|---|---|---|
| 0 | Alpha → Bravo → Charlie → Delta | 0.990 |
| 1 | Delta → Bravo → Charlie → Alpha | 0.875 |
| 2 | Bravo → Alpha → Charlie → Delta | 0.970 |
| 3 | Charlie → Delta → Bravo → Alpha | 1.000 |
| 4 | Bravo → Delta → Alpha → Charlie | 1.000 |
| 5 | Delta → Bravo → Alpha → Charlie | **0.005** |

Verified as a genuine permutation: the rendered sentence *sets* for order 0 and
order 5 are identical (`set(lines_0) == set(lines_5)` is `True`). No text
differs. Only sequence.

### It is not recency, and not primacy

Both obvious explanations are refuted by the data:

- **Not recency.** Order 3 places the liar *first* and yields 1.000. Orders 4
  and 5 both place the liar *last* and yield 1.000 and 0.005.
- **Not primacy.** Orders 1 and 5 both open with Delta and yield 0.875 and
  0.005.
- **Orders 4 and 5 differ by one adjacent swap** of two honest scouts in the
  first two positions, and produce 1.000 versus 0.005.

No single positional feature predicts the flip.

### The variance is *between* orderings, not within

This is the sharpest structural fact in the data, and it cuts both ways.

Within any fixed ordering the listener is nearly deterministic: P(correct) is
0.005 or 1.000, not 0.5. Sampling entropy is close to zero. The bimodality
lives entirely in the choice of presentation order.

- **Good news for the instrument:** conditional on a fixed rendering, the
  action distribution is tight and cheaply estimable. `v(C) = P(correct | C)`
  is a well-behaved quantity.
- **Fatal news for the design:** presentation order is an arbitrary
  experimenter degree of freedom that swings the measured value across its
  entire range. Any Shapley attribution computed over coalitions rendered in
  different orders is measuring the ordering, not the sources.

---

## 3. Consequences

### §5.2 (order-aligned coalition evaluation) is dead

The principal cost optimization in the pre-registration was to evaluate
coalitions in inclusion-nested order so consecutive requests share a growing
cached prefix. That requires physically reordering reports. On this evidence it
would have silently destroyed the measurement while appearing to work.

N1 existed precisely to price that decision before committing to it. It did.

### The ablation effects are wildly heterogeneous, and the median hides it

| ablated source | TV vs base | note |
|---|---|---|
| Bravo | 0.830 | honest corroborator |
| Delta | 0.140 | honest |
| Alpha | 0.055 | honest corroborator |
| Charlie | 0.010 | **the liar** |

`T_ablate` is reported as the median (0.055), which is a poor summary of a set
spanning 0.01 to 0.83. Two observations follow:

1. **Removing the liar changes almost nothing** (0.010). The listener has
   already discounted Charlie. On this episode there is little credulity to
   measure — the failure mode the instrument targets is barely present.
2. **Removing an honest corroborator changes everything** (Bravo, 0.830).
   Alpha and Bravo emit byte-identical reports, yet ablating Bravo moves the
   distribution 15x more than ablating Alpha. Redundant, identical sources are
   not interchangeable to the model. This is direct evidence for the
   redundancy-geometry concern in §2.3(c), and it arrived earlier than
   expected.

### The environment is degenerate, and this was flagged before the run

`east` is covered by **no scout** under `--overlap low`. The episode therefore
admits two coherent readings: *south is clear* (Alpha and Bravo assert it,
Charlie is the outlier) or *east is clear* (south is contested; east is
unruled-out). Every flip in the table above is between exactly these two
basins.

This was predicted in advance, before any data was collected, as a
contamination risk for `P(correct)`. It is not a hindsight rationalization. It
is also not a defense of the result — see §4.

---

## 4. Two specification defects in the pre-registration

Owning these matters more than the verdict.

### 4a. The noise floor is estimated from a single episode

§5 specifies N1/N2/REF on one episode and applies a project-terminating kill
rule to the outcome. One episode cannot distinguish:

- *ablation-based attribution is fragile in language models generally*, from
- *this particular episode is bistable*

Those imply completely different next steps, and the pre-registration as
written cannot tell them apart. This is a design error, not a data problem.

### 4b. With 6 orders, the 95th percentile IS the maximum

Six orderings yield 15 pairwise TVs, plus 1 from N2, for 16 values. Nearest-rank
p95 over 16 values is `ceil(0.95 x 16) = 16` — the maximum. The kill rule is
therefore operationally *"does any single pair exceed the threshold"*, which is
far more trigger-happy than "the 95th percentile" implies.

On this data it makes no difference to the verdict — 5 of 6 orderings agree and
one is catastrophic, so a median-based `T_null` would be ~0.06 and the rule
would sit right at the boundary rather than exceeding it 36-fold. But the
statistic does not mean what the pre-registration implies it means, and that
would have mattered on a closer call.

---

## 5. What this does and does not license

**Established, on this episode and this model:**

- Presentation order alone can flip a scored decision from 99% right to 99.5%
  wrong, with near-zero within-order sampling variance.
- Order-aligned coalition evaluation is not viable.
- Identical-content redundant sources are not interchangeable.

**Not established:**

- That ablation-based attribution is fragile in general. One episode, one seed,
  one model, one overlap setting.
- That the residual metric fails. The metric was never computed; Phase 0 tests
  the harness noise floor, not the instrument.

**Explicitly flagged as post-hoc:** any amendment that widens Phase 0 to
multiple episodes is being proposed *after* seeing episode 1 fire the rule.
That is precisely the analytic freedom pre-registration exists to constrain. It
may still be the right call — a kill rule that cannot separate a general
property from a single degenerate instance is under-specified — but the
amendment must be recorded as post-hoc in §12, with the episode-1 result
already known, and it must fix the protocol *before* the next episode is drawn.

The decision of whether to accept the kill or amend is the operator's, not the
harness's.
