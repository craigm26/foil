# orderprobe

**Detect when a decision depends on the order of its inputs.**

Permute the context, re-run, see whether the answer moves. An answer that
changes under reordering is substantially more likely to be wrong — and you can
check this without knowing the right answer.

```python
from orderprobe import probe

@probe(k=6, samples=5)
def decide(items: list[str]) -> str:
    return call_your_model(items)

r = decide(tool_results)
if r.unstable:
    escalate(r)          # r.value is still the answer you'd have shipped
```

Zero dependencies. Python 3.10+. `pip install` nothing.

---

## Why this exists

Language-model decisions can depend on the arrangement of their inputs, not just
the content. In the study behind this tool, **reordering four byte-identical
sentences moved a decision from 99% correct to 99.5% wrong** — no text changed,
only sequence. The effect is bimodal: most reorderings do nothing, and some
invert the answer completely.

In an agent system the order of context is usually **arbitrary**: tool results
come back in whatever order they finish, subagent reports arrive by scheduling
accident, retrieved chunks are ranked by a scorer that isn't the decision. If
the conclusion depends on that ordering, part of your answer is an artifact of
your scheduler.

`orderprobe` measures it.

## The evidence

Pre-registered test on `claude-opus-5`, 48 episodes, disjoint from the data the
effect was noticed on. The hypothesis, statistic, thresholds and all failure
outcomes were committed to version control before any of the data existed.

|  | wrong | right |
|---|---|---|
| **unstable** | 10 | 4 |
| **stable** | 3 | 31 |

| | |
|---|---|
| P(wrong \| unstable) | **0.714** |
| P(wrong \| stable) | **0.088** |
| Likelihood ratio | **8.1×** |
| Fisher exact, one-sided | **p = 0.00003** |

As a screen it flags 29% of decisions and catches 77% of the errors. It needs no
ground truth, no reference model, no attribution, and no knowledge of which
input is reliable. Robust at every instability cutoff from 0.1 to 0.9.

## The limits — read these before deploying it

**Stability is not a guarantee of correctness.** In the same run, **3 of 34
stable episodes were wrong** — two of them *perfectly* stable across all six
orderings and confidently wrong. Treating a stable answer as verified is wrong
about one case in eleven.

An earlier exploratory reading of this effect showed no false negatives at all
(0 of 15). The pre-registered test refuted that. **Do not build a guarantee on
this signal.** It is triage, not proof.

**It is not calibrated.** 0.714 and 0.088 are point estimates from 48 episodes
in one synthetic environment. They are not a threshold you can port to your
traffic. If you need a threshold, measure it on your own task.

**It is validated on synthetic tasks, on two models.** `claude-sonnet-5` and
`claude-opus-5`, on scripted multi-source decision tasks. Real agent traffic is
untested. An attempt to replicate on a second task family could not test the
question at all — the model answered 40 of 40 items correctly, leaving no errors
to detect.

**No causal claim.** Item ambiguity plausibly drives both the instability and
the error, which makes instability a *symptom* rather than a mechanism. That is
fine for a detector — you cannot observe ambiguity at runtime but you can
observe flipping — but nothing here explains *why*.

**It costs `k × samples` calls per decision.** Reserve it for decisions where
being wrong is expensive. `stable_after` cuts the common stable case roughly in
half: orderings run sequentially and the probe stops the moment the verdict is
decided — at the first modal disagreement (unstable), or once `stable_after`
orderings agree (stable). The verdict is always what the full run would have
returned under the default rule; only `dispersion` becomes partial.

## Where it fits

The test is whether the inputs **could have been gathered in any order without
changing what they say**. That is true when sources are collected concurrently
and then read together. It is false in a sequential agent loop, where each call
is chosen in light of the previous result.

| surface | why order is arbitrary |
|---|---|
| Subagent reports to an orchestrator | arrival order is a scheduling artifact |
| Retrieved chunks in RAG | rank order is a choice, not a fact |
| Multi-source fusion gathered concurrently | completion order carries no meaning |
| Peer messages in multiagent deliberation | turn order is arbitrary |

**Not sequential agent loops.** An earlier version of this table led with "tool
results in an agent loop". A scan of 1,855 real Claude Code sessions found that
99.94% of tool-dispatching turns dispatched exactly one tool, and only **three**
turns in the whole corpus dispatched the three-plus needed to probe at all.
Those results are consumed one at a time, so their order is *causal*: permuting
them does not test order sensitivity, it manufactures an incoherent context.
The measurement is in
[RESULTS-REALTRAFFIC.md](../RESULTS-REALTRAFFIC.md). One corpus, one user, so it
does not show parallel dispatch is rare everywhere -- but it was enough to
retract the claim.

## API

### `probe(k=6, samples=5, *, seed=0, key=None, threshold=None, max_workers=1)`

Decorator. The wrapped function takes the ordered sequence as its first
positional argument and returns a `ProbeResult`.

### `probe_call(fn, items, **same_options) -> ProbeResult`

Function form, for when you can't decorate.

| option | meaning |
|---|---|
| `k` | how many orderings to try. Capped at `len(items)!` |
| `samples` | calls per ordering. Needed when `fn` is stochastic |
| `seed` | permutations are seeded — same seed, same orderings |
| `key` | map a return value to something hashable for comparison |
| `threshold` | if set, `unstable` means `dispersion > threshold`. If unset, it means two orderings produced different modal answers — the definition the evidence above used |
| `max_workers` | `>1` runs orderings concurrently; `fn` must be thread-safe |
| `stable_after` | sequential early stop: quit at the first disagreement, or after this many agreeing orderings. Incompatible with `threshold` and `max_workers>1` |

### `ProbeResult`

| field | meaning |
|---|---|
| `value` | the answer at **the order you passed in** — what you'd have shipped |
| `verdict` | `"stable"`, `"unstable"`, or `"not_applicable"` |
| `unstable` | `verdict == "unstable"` |
| `dispersion` | max pairwise total variation between orderings, 0.0–1.0 |
| `by_ordering` | `(permutation, answer distribution)` per ordering; canonical first |
| `calls`, `errors` | budget actually spent, and calls that raised |

`value` is deliberately **not** a vote across permutations. The probe tells you
whether to trust the answer you already had; it does not silently substitute a
different one.

`not_applicable` means the probe could not run — fewer than two orderings exist,
or every call failed. It is reported separately from `stable` because *"we could
not check"* is a different claim from *"it is fine"*.

## Choosing `k` and `samples`

- `samples=1` is fine when `fn` is deterministic at temperature 0.
- Otherwise `samples=5` was enough in the study: within a fixed ordering the
  model was near-deterministic, and essentially all the variance lived *between*
  orderings.
- `k=6` was the tested value. More orderings raise sensitivity and cost.

## Tests

```bash
python3 -m unittest discover -s orderprobe -t .
```

30 tests, covering the failure modes that would silently corrupt a result:
repeated orderings faking agreement, `k` exceeding `n!`, single-item inputs
reported as stable rather than unchecked, errors being swallowed, and concurrent
execution reordering results.

## Provenance

Built from the FOIL project, which set out to measure something else and
failed — its Phase 0 gate was not passed under three protocols and its original
metric was never computed. Every negative result is published alongside this
one:

**https://foil-9vg.pages.dev**

MIT.
