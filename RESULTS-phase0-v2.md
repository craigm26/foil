# Phase 0 v2: INDETERMINATE — Phase 0 is not passed

**Date:** 2026-08-13
**Model:** `claude-sonnet-5`
**Protocol:** v2 (post-hoc amendment, PREREGISTRATION.md §12)
**Volume:** 12 episodes × 11 arms × 50 samples = 6,250 calls. 1 unparseable
sample, 0 API failures, 0 short arms. Measured cost $11.50.

---

## 1. Verdict

**INDETERMINATE.** Under amendment A6 this halts exactly as KILL does. Phase 0
has not been passed under either protocol, and the project does not proceed to
Phase 1.

| Quantity | Value |
|---|---|
| `T_null` (95th pct, pooled over 192 null comparisons) | 1.000 |
| `T_null` median | 0.000 |
| `T_ablate` (median, pooled over 48 ablations) | **0.000** |
| Verdict | INDETERMINATE |

The cause is not noise. **39 of 48 single-source ablations (81%) moved the
action distribution by exactly zero.** Removing a whole source usually changed
nothing at all, so there is no signal for the noise floor to be measured
against. A6 exists precisely to keep this from being misreported as "noise
exceeds signal."

## 2. The amendment fixed one failure and caused the opposite one

Amendment A3 guaranteed every action is covered by at least one source, to
remove the unruled-out attractor that made the v1 episode bistable. It worked —
and it made the task over-determined. With full coverage, the surviving scouts
still pin the answer after one is removed, so ablation has nothing to move.

| | v1 | v2 |
|---|---|---|
| Failure mode | **under**-determined: a second coherent hypothesis existed | **over**-determined: no single source is decisive |
| Consequence | one ordering inverted the answer | 81% of ablations changed nothing |
| Verdict | KILL | INDETERMINATE |

Both are environment-design failures, in opposite directions. Neither is a
verdict on the instrument, which has still never been computed.

## 3. Order sensitivity survived the fix, and it is bimodal

Full coverage did **not** eliminate the ordering effect. It concentrated it.

| max order TV | episodes |
|---|---|
| 0.000 (perfectly stable across all six orderings) | 9 |
| 0.100 | 1 |
| 1.000 (complete inversion) | 3 |

Episodes are not "a bit order-sensitive." They are either immune or they invert
completely. This is the same knife-edge structure as v1, now shown to be a
property of a quarter of episodes rather than an artifact of one.

---

## 4. Exploratory: bistability predicted error perfectly

**This was not pre-registered, was noticed after seeing the data, and the test
was chosen after seeing the pattern. Treat the p-value as descriptive.**

The three bistable episodes are exactly the three where the listener's
base-order answer is wrong.

|  | wrong at base | right at base |
|---|---|---|
| **bistable** (max order TV > 0.5) | 3 | 0 |
| **stable** | 0 | 9 |

Fisher exact, one-sided: **p = 0.0045** (n = 12 episodes). Overall listener
accuracy was 9/12, always at ~1.00 confidence — the model is confidently wrong
on exactly the episodes whose answer flips under permutation.

### Why this might matter more than the original thesis

If it holds, it is a **ground-truth-free error detector**. At runtime you cannot
observe whether an answer is correct, but you *can* permute the inputs and
observe whether the answer moves. It needs:

- no normative baseline,
- no Shapley attribution,
- no known reliabilities,
- no ground truth.

It therefore sidesteps every limitation recorded in §2.3 of the
pre-registration, which were the strongest arguments against the original
residual metric.

### Why to distrust it anyway

- **n = 12, with 3 positives.** Perfect separation on three cases is fragile;
  one counterexample moves it a long way.
- **Post-hoc.** The hypothesis and the test were both selected after seeing the
  table. The p-value is not a confirmatory result and must not be reported as
  one.
- **Confounded by construction.** An ambiguous episode is plausibly both more
  likely to flip under permutation *and* more likely to be answered wrong. That
  makes bistability a symptom rather than a cause. For a detector this is
  acceptable — you cannot observe ambiguity at runtime but you can observe
  flipping — but it means the mechanism is unestablished.
- **Single model, single overlap setting, single task family.**

Testing it properly requires a fresh pre-registration with the hypothesis fixed
in advance, more episodes, and at least one model tier and task family it was
not discovered on.

---

## 5. The methodological problem with continuing

Phase 0 has now failed twice. It was amended once, post-hoc, after the first
failure. **Amending the environment a second time, after a second failure,
starts to resemble tuning the setup until the gate passes** — which is the
exact behaviour pre-registration exists to prevent.

That this amendment would be principled (see below) does not by itself make it
safe. The v2 amendment was also principled, and it still produced a failure in
the opposite direction, which is what happens when a design is adjusted against
observed outcomes rather than against a stated requirement.

If a third protocol is run, the environment requirement should be fixed
**analytically and in advance**, not calibrated against a measured result:
construct episodes in which each source is decisive by the coverage-and-truth
structure alone, so that ablation is guaranteed to change the posterior without
any reference to what the model does. §3.1 always specified a redundancy matrix
as a controlled variable; the generator never enforced decisiveness, and that
is the defect — not the value of any statistic.

The decision to stop or to continue is the operator's. The record should show
that Phase 0 was not passed under two protocols.

---

## 6. Data availability

- `runs/nulls-v2-claude-sonnet-5.json` — pooled and per-episode results.
- `RESULTS-phase0.md` — the v1 negative result, unrevised.

Reproduce with `python3 cli.py --n 50 run2 --episodes 12` (~6,250 calls, ~$11.50).
