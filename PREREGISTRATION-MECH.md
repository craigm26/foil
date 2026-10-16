# MECH-1: Information or arrangement, read from the weights — Pre-Registration

**Written:** 2026-10-16, before any MECH-1 data exists. No registered model has
been run on any registered seed. The one thing already run is the free
structural gate (G1, §6), which loads a tokenizer and makes no model call; it
passed on both registered tokenizers (48/48 study episodes, 72/72 construct
cases, 0 seeds skipped).
**Reviewed before commit**, twice, by an independent agent that had not seen
the work produced. The first round found three blocking problems and eleven
others; the second found six partial or new problems and five minor ones.
Every finding of both rounds is addressed in this text, which is the first
committed version.
**Supersedes:** nothing. FOIL's Phase 0 verdicts stand unrevised and this
study does not reopen Phase 1, whatever it finds.
**Conventions:** [METHODS.md](METHODS.md). Only what is specific to this study
is stated here.

---

## 1. Why this study exists

FOIL's headline finding is a confound it could name but not separate.
Removing a source from a prompt changes two things at once: the information
available, and the position of everything after the edit (FINDINGS §1).
Phase 0 v3 then returned KILL by 6.25×, "a verdict about ablation-based
attribution" — but the API harness can only edit text, so it could not say
which half of an ablation was doing the work.

White-box access separates them. A report can be made **invisible in place**:
every query after it is forbidden from attending to it, in every layer and
head, while every token keeps its position (`mech/whitebox.py`). That is an
arrangement-fixed counterfactual, and with it FOIL's own primary operator
decomposes into an information term, an arrangement term and a null-text term
(§3.2).

Goodfire's essay *Neural Networks Are Already Clean* (Cammarata, 2026) makes
two further claims FOIL's environment can test without taking a side in the
neurons-versus-dictionaries argument: that running summaries of context sit on
punctuation and turn boundaries (the "summary motif"), and that a network's
own neurons are legible units, a handful of which can carry a premise. The
layer is written against a basis interface (`mech/basis.py`), so the same
search runs on neurons or on a learned dictionary. This study registers
neurons as primary and names the dictionary comparison as exploratory (§9).

## 2. Hypotheses

**H1 — arrangement share.** Of the movement in P(correct) produced by FOIL's
primary operator Ω_null, the share attributable to arrangement alone is small.

> Prediction: **ARRANGEMENT MINOR**. Stated so it can fail: Ω_null moves
> every later token by only −3, −1 or +1 positions on these stimuli, so a
> material share would mean decisions move under shifts of one to three
> positions. The API-side finding that one adjacent swap inverted an answer
> says that is possible.

**H2 — summary routing.** Downstream positions read a report through its
punctuation tokens rather than its words.

> Prediction (the essay's): **SUMMARY-ROUTED**. Each outcome carries a fixed
> reading for the motif (§7.2): SUMMARY-ROUTED supports it only if the inner
> punctuation — the marks with no line break fused in — carries the effect on
> its own, and is otherwise read as confounded with line structure;
> CONTENT-ROUTED and JOINT count against it; REDUNDANT and UNRESOLVED are
> uninformative about it.

**H3 — few-neuron mediation.** The in-place effect of a report is reproduced
by patching 16 MLP neuron-sites — (layer, position, neuron) triples — from the
knockout run into the factual run, and by those units specifically: more than
by the matched control (§3.3).

**H3b — the same, upstream of the answer.** As H3, with the assistant turn
excluded from the search: no site at the decision position or in the prefill.

> Prediction (the essay's): **FEW-NEURON MEDIATION** for both. H3b is the
> sharper test: a late-layer unit at the decision position writes the answer
> rather than carrying a premise. H3b reduces that route without removing it —
> the end-of-user-turn tokens remain eligible and late units there could also
> write the answer — so the location table (§7.3) reports how many H3b
> carriers sit on those two tokens and at what depth.

**R0 — Phase 0, replicated on an open model.** FOIL's kill rule, unchanged,
applied to exact action distributions. No prediction is registered. Every
FOIL result so far comes from Claude, and "that this generalises beyond
`claude-sonnet-5`" is listed as not established.

### The confounds, stated up front

- **H1, gap.** An invisible span still occupies positions, and the listener
  can see the gap through the position indices. Reports are 8, 10 or 12
  tokens, which encodes their structure (one route; two routes sharing a
  status; one clear and one blocked), so `info` excludes whatever a listener
  reads from a gap's size, and `arr_null` includes it. Reported: the share by
  report length — which here maps one-to-one onto the shift (8 → +1,
  10 → −1, 12 → −3), so it cannot separate reading the gap from the size of
  the shift. Nothing in this design can.
- **H1, path.** The decomposition is an identity along one path, crediting
  any information × position interaction to the later step. A second path is
  measured (§3.2) and its share reported beside the first.
- **H1, deletion.** For the last report, Ω_del also re-tokenises the previous
  report's final token (".\n" becomes ".\n\n"), so its arrangement term
  includes a visible change. Reported: the Ω_del share with last-slot units
  excluded.
- **H2, line breaks.** Qwen3 fuses a report's closing period with the
  following line break into one token. Hiding "punctuation" therefore also
  hides the line structure, and a SUMMARY-ROUTED result could reflect broken
  parsing. The inner punctuation alone (":", ",", ";", which carry no line
  break) is measured, and the motif reading of SUMMARY-ROUTED depends on it
  (§7.2); the final token alone is reported.
- **H2, token counts.** A report has 2–3 punctuation tokens and 6–9 others.
  The ratios measure which channel the decision depends on, not how much any
  one token matters.
- **H3/H3b.** Selection and test use the same unit. The claim is existential
  — there exist 16 sites that reproduce the effect — and the matched control
  is what separates "these units" from "this place, this much change".
- **Slot.** Per-source arms run at one registered ordering per episode — the
  canonical order rotated by seed mod 4, so across the 48 episodes every
  source sits in every slot exactly 12 times — because canonical order fixes
  the liar in the third slot and the last honest scout in the slot that
  absorbs the paragraph break before the query.
- **All.** The listener is a small open model, not Claude. A result here is a
  fact about these operators on these weights. It is evidence about Claude
  only through a lab running the same operators on its own model, which is
  what the harness is built to allow.

---

## 3. Definitions

### 3.1 Readout

The FOIL stimulus is rendered by `foil.render.render`, unchanged, then through
the model's chat template with thinking disabled (as FOIL disabled it). The
structured-output schema is emulated by the assistant prefill `{"route": "`,
after which every route name must be exactly one token (G1). The action
distribution is the next-token distribution over those four tokens,
renormalised — **exact**, with no sampling and no parse step.
`v = P(correct)`, as in FOIL §3.2.

**On-format floor.** The probability mass on the four route tokens before
renormalising is recorded for every pass. A pass below **0.50** is not
reading an answer. Validity is judged **per hypothesis, on that hypothesis's
own passes**, so a failing pass in one analysis cannot remove a unit from
another, and exclusion never depends on another hypothesis's outcome:

| quantity | passes that must be on-format |
|---|---|
| H1 (population and S_arr) | full, ko, null_ko, null, del |
| H1 reverse path | the above and shift |
| H2 | full, ko, ko_sum, ko_con (ko_inner, ko_final for their own reports) |
| H3 | full, ko, the k = 16 patch, the matched-control patch |
| H3b | full, ko, the pre-answer patch, its matched-control patch |
| R0 | an episode's N1, N2 and REF passes |

A unit failing a hypothesis's passes is a **dropout** for that hypothesis
(for H2/H3/H3b, a unit whose `full` or `ko` is off-format is one too, since
its activity cannot be judged). Each hypothesis's dropouts are counted over
all 192 units, R0's over the 48 episodes; above 5% that result is labelled
**COMPROMISED** (METHODS §3).

### 3.2 Arms and the decomposition

A unit is one (episode, source). Its arms, at the episode's arms ordering:

| arm | context | what changes relative to `full` |
|---|---|---|
| `full` | all reports visible | — |
| `ko` | `s` present, invisible downstream | information |
| `null_ko` | `s` replaced by Ω_null's text, invisible | information; slot length |
| `shift` | `s` visible; later tokens at Ω_null's positions | slot length only |
| `null` | Ω_null, visible (FOIL's primary) | information; slot length; the words "no report this round" |
| `del` | Ω_del | information; slot removed |
| `ko_sum` | only `s`'s punctuation invisible | H2 |
| `ko_con` | only `s`'s other tokens invisible | H2 |
| `ko_inner` | punctuation except the final token | H2, reported |
| `ko_final` | the final token only | H2, reported |

"Invisible" means queries at every position after the span's last token may
not attend to the hidden keys. Queries inside the span are untouched, so the
span's own punctuation still forms normally; only reads from after it are cut.
In `shift`, a −3 or −1 shift makes the first moved tokens share a position
with the span's last ones — off-distribution, as a gap is.

Two paths through the same total:

```
Δ_null = (ko − full) + (null_ko − ko) + (null − null_ko)       registered path
       =    info     +   arr_null    +   text_null
       = (shift − full) + (null_ko − shift) + (null − null_ko)  reverse path
       =    arr_rev     +     info_rev      +   text_null
Δ_del  = info + arr_del                                         arr_del = del − ko
```

### 3.3 Carriers

Units: MLP neurons (one coordinate of the input to `down_proj`), every layer.
Candidate sites for `s`: every position strictly after `s`'s span through the
decision position (H3), or through the last token before the assistant turn
(H3b). Under `ko` nothing at or before the span changes, so nothing there can
carry `s`.

Score: Δactivation (ko − full) × ∂P(correct)/∂activation at `full`. Only sites
whose score has the sign of `info` are eligible; they are ranked by magnitude.
Patch: the top k set to their `ko` values in a `full` pass. Fidelity
`F = (P_patched − P_full) / (P_ko − P_full)`; |info| ≥ 0.10 bounds every
denominator.

**Matched control (registered).** Each of the 16 selected sites is replaced
by the unselected neuron in the **same (layer, position) cell** whose |Δ| is
nearest its own; the 16 replacements are patched the same way. Place is held
fixed exactly; size approximately — attribution favours each cell's largest
change, so substitutes run smaller. The median ratio |Δ_substitute| /
|Δ_selected| is recorded for every unit, and the outcome rule (§7.3) refuses
to confirm specificity against a control that changed less than half as
much.
**Top-|Δ| control (reported).** The 16 sites whose activation changed most,
ignoring the gradient.

Everything in the carrier search is deterministic.

---

## 4. What white-box access changes, and what it cannot

- **Exact distributions.** Every TV, share and fidelity is computed from
  logits; there is no sampling error within an arm. Uncertainty comes only
  from which episodes were drawn, which is what the cluster bootstrap
  resamples.
- **Pure softmax attention only.** The knockout cuts every path only when
  attention is the sole channel between positions. Linear-attention and
  state-space layers carry tokens in a recurrent state no mask reaches, so
  such models are refused (`whitebox.eligibility`). Qwen3.5 interleaves Gated
  DeltaNet layers 3:1 with full attention and is excluded for exactly this
  reason. Mixture-of-experts MLPs are excluded because a token's neurons are
  not a fixed set.
- **Position overrides need an explicit mask.** transformers 5.x infers
  "packed sequences" from any jump in position ids when no attention mask is
  passed, and silently cuts attention into blocks. Every pass therefore
  passes the mask explicitly, and a test checks the mask under a shift.
- **Not Claude.** See §2.

---

## 5. Design, fixed now

| Parameter | Value |
|---|---|
| Environment | `env3` unchanged (scout v3; ≥3 of 4 sources decisive, checked analytically; Charlie lies on every claim) |
| Episodes | **48**: seeds **9000–9047** (`mech/gates.py` `STUDY_SEEDS`) |
| N1 | 6 orderings per episode: canonical + 5, by `foil.nulls.run_nulls`'s procedure, seeded by the episode seed |
| N2, REF | paraphrase and single-source Ω_null at the **canonical** order, as in FOIL |
| Arms ordering | canonical order rotated left by **seed mod 4** (a Latin square: each source in each slot 12 times) |
| Per-unit arms | §3.2 |
| Thinking | disabled |
| Precision | float32, eager attention, no KV cache, explicit attention mask; TF32 off on GPU |
| Carriers | neuron basis; k ladder 4, 16, 64, 256; **primary k = 16**; H3b at k = 16 only |
| Active | \|ΔP(correct)\| ≥ **0.10** (H1: under any of Ω_null, Ω_del, `ko`; H2/H3/H3b: under `ko`) |
| On-format floor | **0.50** per pass (§3.1) |
| Inference | cluster bootstrap over episodes, 2,000 resamples, seed 0, 95% percentile intervals |
| Degeneracy | fewer than 20 qualifying units, or fewer than 10 contributing episodes |

Seeds are disjoint from every prior FOIL study (v1–v3: 1–12; PID-2: 1000+;
TURN: 2000+; TURN-2: 3000+; TURN-3: 4000+; SWEEP: 7000+).

### 5.1 Rungs

| rung | model | Hub revision | role |
|---|---|---|---|
| A | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | replication; runs on any CPU (~3 GB); run first |
| **B** | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | **confirmatory**; the smallest Qwen3 with a published Qwen-Scope SAE, so §9's comparison is possible; ~8 GB in float32 |

Both are dense Qwen3 — the family of the essay's Qwen3-32B audit — with pure
softmax attention.

- **Claims come from B.** A's outcomes are reported as a replication on a
  smaller model: agreement or disagreement with B is described, never pooled,
  and never changes B's outcome. If B fails a gate, MECH-1's confirmatory
  result is that gate failure, and A's outcomes are descriptive only. If B is
  never run, MECH-1 has no confirmatory result, A's outcomes stay
  descriptive, and FINDINGS carries the study as "B not run" — A's result is
  never presented as MECH-1's.
- **Order and commitment.** A runs first, then B. B runs whatever A's outcome,
  including a gate failure. Every rung that passes its gates is run in full.
  The only acceptable reason not to run a rung is the absence of hardware
  that can hold it in float32; a rung not run is disclosed with that reason,
  not dropped (SWEEP §5).
- **No larger rung is registered.** If B fails G3, a larger model is a new
  study with its own registration, not an extension of this one.

The registration is enforced in code: `mech_run.py run` refuses any model,
revision or dtype not in this table, any that did not pass every gate, and
any device or library version other than the one its gates passed on.

---

## 6. Gates, each blocking, in order

| gate | cost | rule |
|---|---|---|
| **G0** eligibility | free | pure softmax attention; gated MLP with `down_proj`; eager attention |
| **G1** structural | free | every arm of every study episode renders, reproduces `foil.render`'s text, maps each report to its own tokens with ≥1 punctuation and ≥1 other token, and reads out one token per action; `env3`'s decisiveness, re-derived from the **rendered** lines, holds; every construct case renders cleanly |
| **G2** leak | compute | 8 episodes (seeds 9600–9607) × 4 sources: with a span knocked out, overwriting it with random tokens may move no decision-position logit by more than **1e-4**; the same overwrite *without* the knockout must move them by more than 1e-2, or the test is vacuous and FAILs |
| **G3** construct | compute | full information, no manipulation: the liar's report inverted to the truth, canonical order. Argmax must equal the truth. `evalgate.gates.verdict` at threshold **0.90**, α 0.05, n = **48** — the first 48 seeds from 9500 whose honest **rendered** text pins the truth. EXTEND runs one extension of the next **24** such seeds and re-applies the rule to the pooled 72; still EXTEND resolves to FAIL. Independently, median on-format mass must be ≥ **0.90** |

`gates.plan(0.90, 0.75)` gives n = 40 for 80% power against a true rate of
0.75; 48 exceeds it. A rung that fails a gate stops there, and the failure is
that rung's published result (PID-1, TURN-3). G2 does not test the slot-length
channel (§2): it overwrites tokens at fixed length, by design.

There is no separate variance gate: the per-hypothesis degeneracy rule plays
that role on the registered data, as PID-2's margin rule did.

---

## 7. Analysis plan, fixed now

Everything below is computed by `mech/analysis.py` from the per-episode
records, which is why it is numpy-only and runs without the model. "Reported
regardless" means reported even when the hypothesis is DEGENERATE, whenever
any unit qualifies.

### 7.1 H1

Population: valid, active units. Statistic, on the registered path:

```
S_arr = Σ|arr_null| / Σ(|info| + |arr_null| + |text_null|)
```

Absolute values, deliberately: an arrangement term that cancels an information
term is contamination even when the net effect looks small.

- **ARRANGEMENT MINOR** — upper 95% bound of S_arr < 0.20
- **ARRANGEMENT MATERIAL** — lower 95% bound > 0.20
- **UNRESOLVED** — otherwise; **DEGENERATE** — §5

**Reported regardless:** S_arr on the reverse path (Σ|arr_rev| /
Σ(|info_rev| + |arr_rev| + |text_null|)); the Ω_del share Σ|arr_del| /
Σ(|info| + |arr_del|), with and without last-slot units; S_arr without
last-slot units; S_arr by report length; the information and null-text
shares; the fraction of units with |Δ_null − info| ≤ 0.10; with
|arr_null| ≥ 0.10; and where Ω_null and the knockout move P(correct) in
opposite directions by ≥ 0.10 each; and S_arr, with intervals, separately for
order-bistable and order-stable episodes (R0's definition).

### 7.2 H2

Population: valid units with |info| ≥ 0.10. Per unit,
`r_sum = (ko_sum − full) / info` and `r_con = (ko_con − full) / info`, kept
unclipped. Medians with cluster-bootstrap intervals. Every named outcome is an
interval statement, so they are mutually exclusive:

- **SUMMARY-ROUTED** — r_sum lower bound ≥ 0.5 and r_con upper bound < 0.5
- **CONTENT-ROUTED** — r_con lower bound ≥ 0.5 and r_sum upper bound < 0.5
- **JOINT** — both lower bounds ≥ 0.5
- **REDUNDANT** — both upper bounds < 0.5
- **UNRESOLVED** — otherwise; **DEGENERATE** — §5

**Motif reading**, fixed with the outcome:

| outcome | reading |
|---|---|
| SUMMARY-ROUTED, and the median `ko_inner` ratio's lower bound ≥ 0.5 on ≥ 20 units | **supports the motif** |
| SUMMARY-ROUTED otherwise | **confounded with line structure** |
| CONTENT-ROUTED or JOINT | **against the motif** |
| REDUNDANT, UNRESOLVED, DEGENERATE | **uninformative about the motif** |

**Reported regardless:** the median of r_sum − r_con; the fraction of units
with r_sum ≥ 0.5 and with r_con ≥ 0.5, with intervals (the ratios are likely
bimodal, as FOIL's effects have been, and a median hides that); the medians
for `ko_inner` and `ko_final`; and r_sum without last-slot units.

### 7.3 H3 and H3b

Population: valid units with |info| ≥ 0.10, all of which get a carrier search.
`F` is F_16 for H3 and the pre-answer F for H3b; `F_m` is the matched
control's fidelity.

- **FEW-NEURON MEDIATION** — lower 95% bound of median F ≥ 0.5, **and** lower
  bound of median (F − F_m) > 0, **and** median size ratio of the matched
  control ≥ 0.5
- **MEDIATION, SPECIFICITY UNRESOLVED** — the first two conditions, but the
  control changed less than half as much as the chosen units, so beating it
  proves nothing about them
- **MEDIATION, NOT UNIT-SPECIFIC** — lower bound of median F ≥ 0.5, but not
  the second condition: the cells carry it, the chosen neurons are not special
- **NOT FEW** — upper bound of median F < 0.5
- **UNRESOLVED** — otherwise; **DEGENERATE** — §5

**Reported regardless:** the full k ladder (H3); the top-|Δ| control; the
fraction of units with F ≥ 0.5 and with F > 1 (overshoot counts toward the
median as it stands, uncapped, and is reported so it can be seen); units with
fewer eligible sites than each k; where the top 16 sit, by position class —
decision, other answer tokens, **turn** (the end-of-user-turn tokens, the
two H3b's caveat is about), punctuation, other words — with the layer depth
of each class; and which SwiGLU sign regime each carrier occupies in the
factual run, including the double-negative regime the essay says
activation-only reading hides. These are descriptive and carry no outcome.

### 7.4 R0

Per FOIL §5.1, with `foil.nulls._verdict` imported rather than restated:
`T_null` the 95th percentile of pooled N1 and N2 TVs, `T_ablate` the median of
pooled single-source Ω_null TVs at the canonical order, verdict KILL /
PROCEED / INDETERMINATE. Bistable: maximum pairwise N1 TV > 0.5
(`foil.nulls.BISTABLE_TV`). Episodes excluded from R0 (§3.1) are counted and
label R0 COMPROMISED above 5%; their units appear in H1's stability strata as
"unclassified". Also reported: canonical accuracy, mean P(correct), and —
descriptive, with slot now balanced — the mean in-place effect of the liar
versus the honest scouts.

### 7.5 Integrity

Reported for every rung:

- **VOID** — any episode whose capture-and-gradient pass disagrees with its
  plain pass by more than 1e-5. Hooks must observe, not alter. The episode is
  written to disk with the failure and the run stops; the runner refuses to
  resume past it, and `analyze` publishes the rung as VOID however few
  episodes exist. Every outcome of the rung is VOID.
- **COMPROMISED** — per hypothesis, dropouts (§3.1) above 5%. Outcomes are
  reported with the label.
- Always: the maximum capture-versus-plain difference, the minimum on-format
  mass across all passes, every dropout count, and G2's measured maximum leak.

**No optional stopping.** All 48 episodes are measured before any statistic is
computed; `mech_run.py analyze` refuses to decide anything on fewer. The
analysis runs once per rung.

---

## 8. What results would and would not establish

**ARRANGEMENT MINOR** would establish that, on these weights and within one
ordering, the change in slot length accounts for little of Ω_null's effect.
Whether Ω_null then measures information also depends on the null-text share,
reported beside it, and on the gap caveat (§2). It would locate FOIL's Phase 0
problem more firmly in the listener's sensitivity to *ordering* (N1), which
no ablation repairs and orderprobe already exploits, and would motivate a
registered test of attribution restricted to order-stable decisions. It would
**not** pass Phase 0, revive the residual metric, or say anything about
Claude.

**ARRANGEMENT MATERIAL** would establish that changing the slot's length —
the positions it shifts and the gap it leaves readable (§2) — accounts for
more than a fifth of Ω_null's effect on these weights. It is attributed to
the **shift** itself only if the reverse-path share's lower bound also
exceeds 0.20. Either way, the remedy need not require the weights: a null
text matched to the report's token count removes the shift with text edits
alone (FOIL's Ω_neutral aims at that), at the price of its own content.

**H2 and H3/H3b** would establish where, on these weights, a source's
information is read and carried, under these operators. They would not
establish that the essay's motif holds in other models, tasks or sizes, or
that neurons beat dictionaries; §9's comparison is exploratory.

## 9. Exploratory, named now so nothing can be promoted later

Labelled **EXPLORATORY** and dated if run; never pooled with the above; any
claim from them needs its own registration.

- H3/H3b repeated in Qwen-Scope's residual SAE basis on rung B
  (`DictionaryBasis.from_qwen_scope`, a subset of layers for memory). The SAE
  was trained on the Base model; that mismatch is part of what is explored.
- Per-layer knockout windows (in which layers a report is read).
- The liar–honest contrast beyond the descriptive means in §7.4.

## 10. Compute

Per episode, at most 85 forward passes — N1 6, N2 1, REF 4 and a plain pass at
the arms ordering (none of these five when the arms ordering is canonical, the
plain pass also not when N1 already holds that ordering), 1 with gradients,
9 arms × 4 sources, and 9 per active source for the carrier searches — plus
one backward pass: at most 4,080 forward passes per rung, plus at most 176 for
G2 and G3. Prompts are 173–198 tokens. Rung A fits a 2-core, 7 GB machine; B
needs ~12 GB of RAM. Wall-clock and pass counts are recorded in every result
file. No API calls are made.

---

## 11. Amendment log

| Date | Section | Change | Reason |
|---|---|---|---|
| _(none)_ | | | |
