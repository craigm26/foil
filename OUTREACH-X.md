# X thread — paste-ready

11 posts. Every one verified at or under 280 characters (max 277). No em-dashes.
Paste each block as its own post in the thread composer.

---

## 1/  (269)

I tried to build on Anthropic's Frontier Red Team piece on multiagent systems.

Built 7 eval environments. 4 were invalid.

Every invalid one produced results that looked perfectly analyzable. That's the finding.

Code, data, pre-registrations:
github.com/craigm26/foil

---

## 2/  (277)

The instructive failure: my structural gate ran over 200 seeds, re-derived from rendered text, and passed.

The environment was invalid anyway.

Every run scored "incorrect" was a unanimous 4-agent vote for the option my scorer ranked last.

The agents were right. I was wrong.

---

## 3/  (275)

I was one write-up from publishing that as a consensus-suppression finding.

A gate can be rigorous, reproducible, and confirm the wrong thing.

The other two were the opposite pair: one under-determined, one so over-determined that 81% of ablations moved the answer by zero.

---

## 4/  (265)

What survived:

Two orderings of 4 byte-identical scout reports, differing by ONE adjacent swap, scored 1.000 and 0.005 on the correct route.

Same sentences. Same characters. Only the sequence changed.

5 of 6 orderings agreed and 1 inverted, so any mean hides it.

---

## 5/  (240)

Removing the liar from the context shifted the decision 0.010.

Removing one of two byte-identical honest reports shifted it 0.830.

Redundant sources are not interchangeable. That's a problem for anything doing coalition-based attribution.

---

## 6/  (251)

The usable result: permutation instability predicts error with no ground truth.

LR 8.1, p = 0.00003, pre-registered.

But stability is NOT a guarantee. 3 of 34 stable episodes were wrong, 2 of them stable across all 6 orderings and confidently wrong.

---

## 7/  (183)

An earlier exploratory read showed zero false negatives. The pre-registered run refuted it.

I'd rather you got that from me than from the exploratory number.

It's triage, not proof.

---

## 8/  (262)

And the result that cuts against my own story:

I predicted a group decides better when the agent holding the decisive private fact speaks FIRST.

At 90%+ power, control panel passing 36/36, it did not. Effect ran the other way. 0 of 32 scenarios favoured first.

---

## 9/  (262)

4 agents deliberating over 2 rounds showed no positional harm that a single listener showed dramatically.

I'm not claiming the reverse either. My test was one-sided the other way, and flipping it after seeing the sign is what pre-registration exists to prevent.

---

## 10/  (261)

Two zero-dependency Python libraries came out of the failures, not the findings:

orderprobe: permutes a decision's inputs at runtime, flags it when the answer moves

evalgate: 3 checks that would have caught all 4 invalid environments. 2 are free to run.

MIT.

---

## 11/  (230)

Total spend: ~$47 across ~24,000 calls.

Which is most of the point. The invalid environments were cheap to detect and expensive to trust.

Full write-up, live demos, and all 7 results including the 4 failures:

foil-9vg.pages.dev
