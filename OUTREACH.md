# Cover note to the Frontier Red Team

Subject: **An attempted replication of your scripted-scout setup, and the four environments that turned out to be invalid**

---

I tried to build on *Patterns and problems in emerging multiagent systems*, and
the useful part of what came back is a failure record rather than a finding. It
is all public: code, data, and every pre-registration committed before its own
data existed.

https://github.com/craigm26/foil · https://foil-9vg.pages.dev

**Seven environments, four invalid, and every invalid one produced results that
looked perfectly analyzable.** That is the part I think is worth your time.

The instructive one is TURN-1. Its structural gate ran over 200 seeds,
re-derived from the rendered text, and passed. The environment was still
invalid: every run my scorer marked incorrect was a *unanimous* four-agent vote
for the option that scorer ranked last. The agents were right and the scorer was
wrong. I was one write-up away from publishing that as a consensus-suppression
result. An analytic gate can be rigorous, reproducible, and confirm the wrong
thing. Two of the other three failures were the opposite pair: one
under-determined so a never-ruled-out option competed with the evidence, one
over-determined so 81% of ablations moved the answer by exactly zero.

What survived, with its limits attached:

* Two orderings of four byte-identical scout reports, differing by a single
  adjacent swap, scored 1.000 and 0.005 on the correct route. The canonical
  ordering scored 0.990. Five of six measured orderings agreed and one inverted,
  so any mean or median hides it entirely.
* Removing the liar shifted the decision 0.010. Removing one of two
  byte-identical honest reports shifted it 0.830. Redundant sources are not
  interchangeable, which is a problem for coalition-based attribution.
* Permutation instability predicts error without ground truth: LR 8.1,
  p = 0.00003, pre-registered. **Stability is not a guarantee.** Three of
  thirty-four stable episodes were wrong, two of them stable across all six
  orderings and confidently wrong. An earlier exploratory read showed zero false
  negatives; the pre-registered run refuted that, and I would rather you got
  that from me than from the exploratory number.

And the result that cuts against my own story: I predicted a group would decide
better when the agent holding the decisive private fact spoke first. At 90%+
power, with a full-information control panel passing 36 of 36, it did not.
The effect ran the other way, mean delta -0.094 against a required +0.15, zero
of thirty-two scenarios favouring first. Four agents deliberating over two
rounds showed no positional harm that a single listener showed dramatically. I
am not claiming the reverse either, because the test was one-sided in the
opposite direction. It is published as a limit on everything above it.

Two zero-dependency Python libraries came out of the failures rather than the
findings: `orderprobe`, which permutes a decision's inputs at runtime and flags
it when the answer moves, and `evalgate`, three checks that would have caught
all four invalid environments before they cost anything. Two of the three are
free to run. MIT, no dependencies, small enough to read in an afternoon.

Total spend was about $47 across roughly 24,000 calls, which is most of the
point: the invalid environments were cheap to detect and expensive to trust.

No ask. If the failure record is useful to you, take it.

Craig Merry

---

## Routing

Anthropic publishes no inbound channel for the Frontier Red Team. Checked
2026-08-13: the article page, red.anthropic.com and the FRT team page carry no
email, no form, no submission route, and no named authors. The only FRT-specific
mechanism is an outbound newsletter.

Real options, best first:

1. **Public post, tagging the work rather than a person.** The repo and site are
   already live and the failure record is the hook. FRT reads its own citations.
   Lowest friction, no guessed address, and it survives being ignored.
2. **Direct message to a named FRT researcher** on a public platform. Requires
   identifying one from their published work, not from a guessed address.
3. **Anthropic's general research contact**, if one is reachable through a
   support channel. Slowest and least targeted.

Not doing: guessing at an `@anthropic.com` address.
