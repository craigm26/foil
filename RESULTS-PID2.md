# PID-2: SUPPORTED — and the stronger claim is refuted

**Date:** 2026-08-13
**Model:** `claude-opus-5`
**Pre-registration:** [PREREGISTRATION-PID2.md](PREREGISTRATION-PID2.md),
committed before any of this data existed (commit `0a26a13`).
**Volume:** 48 episodes × 6 orderings × 10 samples = 2,880 calls. 0 dropouts,
0 unparseable, 0 API failures. Cost **$4.40** (batched; $8.79 synchronous).

---

## 1. Two results, and the second matters as much as the first

**The primary hypothesis is SUPPORTED.** All three pre-registered criteria met,
by wide margins.

|  | wrong | right |
|---|---|---|
| **unstable** | 10 | 4 |
| **stable** | **3** | 31 |

| Criterion | Required | Observed |
|---|---|---|
| Fisher exact, one-sided | p < 0.01 | **p = 0.00003** |
| Risk difference | ≥ 0.30 | **0.626** |
| Margins | ≥ 10 each | 14 / 34 |
| **Outcome** | | **SUPPORTED** |

**The secondary claim is REFUTED.** Across v2 and v3, no stable episode had ever
been answered wrong — 0 of 15. Here **3 of 34 stable episodes were wrong.**

That claim was the most useful thing the earlier data appeared to show, and it
was the thing this run was sized to break. It broke. `K = 48` was chosen so that
a false-negative rate above roughly 10% would be expected to surface, and one
did: 8.8%, with a 95% upper bound of 21.3%.

**Permutation stability is not a guarantee of correctness.** Any deployment
reading it as one would be wrong about 1 episode in 11.

## 2. What the detector actually buys

| Quantity | Value |
|---|---|
| Base error rate | 13/48 = 0.271 |
| P(wrong \| unstable) | 10/14 = **0.714** |
| P(wrong \| stable) | 3/34 = **0.088** |
| Likelihood ratio | **8.1×** |

As a screen, flagging unstable episodes:

- flags **29%** of episodes (14 of 48),
- catches **77%** of errors (10 of 13),
- **misses 23%** of errors (3 of 13).

That is a genuinely useful triage signal and emphatically not a proof of
correctness. It roughly triples the posterior probability of error when it
fires, and cuts it to a third when it doesn't.

### The three counterexamples

| episode | answered | truth | max order TV |
|---|---|---|---|
| `v3-s1004` | east | north | **0.000** |
| `v3-s1011` | west | south | 0.200 |
| `v3-s1028` | west | north | **0.000** |

Two of the three were *perfectly* stable — identical answers across all six
orderings — and still wrong. These are not marginal cases near the threshold;
they are confident, consistent, incorrect answers. Whatever produces them is
invisible to a permutation probe.

## 3. The threshold does not carry the result

The 0.5 cutoff was fixed in advance, and the pre-registration required this
table regardless of outcome:

| cutoff | (a, b, c, d) | p |
|---|---|---|
| 0.1 | 11, 5, 2, 30 | 0.00001 |
| 0.2 | 10, 5, 3, 30 | 0.00009 |
| 0.3 | 10, 5, 3, 30 | 0.00009 |
| 0.4 | 10, 4, 3, 31 | 0.00003 |
| **0.5** | **10, 4, 3, 31** | **0.00003** |
| 0.6 | 9, 4, 4, 31 | 0.00020 |
| 0.7 | 9, 4, 4, 31 | 0.00020 |
| 0.8 | 9, 3, 4, 32 | 0.00007 |
| 0.9 | 9, 1, 4, 34 | 0.0000004 |

p < 0.001 at every cutoff from 0.1 to 0.9. The finding does not depend on where
the line was drawn — which is what the bimodality of the ordering effect
predicts, and the reason the sensitivity table was pre-registered rather than
produced on request.

## 4. What this establishes, and what it does not

**Establishes:**

- On two models (`claude-sonnet-5`, `claude-opus-5`) in the scripted-scout
  family, permutation instability carries substantial information about
  correctness, under a pre-registered test with a large effect size.
- Stability is **not** a correctness guarantee. The no-false-negative pattern in
  v2/v3 was a small-sample artifact, and this run says so.

**Does not establish:**

- **Any causal claim.** Episode ambiguity plausibly drives both the instability
  and the error, making instability a symptom. This design cannot separate
  those, and the pre-registration said so before the data existed.
- **That it holds in any other task family.** PID-1 tried a second family and
  could not test it — `claude-opus-5` answered 40 of 40 items correctly, so
  there was nothing to detect. That attempt stands published as DEGENERATE.
- **That the detector is calibrated.** 0.714 and 0.088 are point estimates on 48
  episodes in one synthetic environment. Nothing here licenses a threshold for
  production use.
- **Anything about FOIL's Phase 0.** It was not passed under any protocol and
  this does not reopen it.

## 5. Relationship to the rest of the project

This is the first **confirmatory** result in the project. Everything preceding
it was either a pre-registered halt (Phase 0 v1/v2/v3, PID-1) or an exploratory
observation (the v2/v3 detector pattern).

It also partly reverses that exploratory observation. The v2/v3 pattern was
reported here as "instability behaves as a necessary but not sufficient marker
of error — no false negatives, some false positives." **The no-false-negative
half of that is now known to be wrong**, on more data, under a pre-registered
test. The direction of the effect replicated; the absolute claim did not.

That is the intended function of pre-registration and it worked as designed: the
exploratory pattern was strong enough to be worth testing and wrong in one of
its two parts, and the test was sized to find out which.

## 6. Data availability

- `runs/pid2-result-claude-opus-5.json` — per-episode modal answers across all
  six orderings, max order TV, the 2×2, the sensitivity table, and the ledger.
- `data/pid2-result.json` — archived copy.

Reproduce with `python3 pid2_run.py` (2,880 calls, ~$4.40 batched).
