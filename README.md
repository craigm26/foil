# FOIL

Counterfactual sensitivity profiling for multiagent epistemics.

A harness for measuring, at a single decision point, the signed gap between how
much a decision actually depended on an information source and how much it
normatively should have. See [PREREGISTRATION.md](PREREGISTRATION.md) for the
hypotheses, the analysis plan, and the kill rules, all fixed before data
collection.

**Status:** Phase 0 was not passed under any protocol. Eleven studies, including
every negative result, are indexed in [FINDINGS.md](FINDINGS.md). MECH-1, the
first white-box study, is pre-registered and not yet run
([PREREGISTRATION-MECH.md](PREREGISTRATION-MECH.md)).

## Why this exists

Anthropic's Frontier Red Team documented two epistemic failure modes in agent
groups: miscalibrated credulity and consensus suppression. Proposals to fix
them are accumulating faster than instruments to measure them. FOIL tries to
build the instrument, and to find out honestly whether it can be built at a
defensible cost. A negative result is a real contribution and the design
pre-commits to publishing one.

## Install

Nothing to install. Python 3.11+ and numpy. Zero third-party dependencies by
design: the harness has to run unchanged inside a lab against models the author
cannot reach, and every dependency is a reason someone does not run it.

The optional white-box layer (`mech/`) needs torch and transformers
(`pip install -r requirements-mech.txt`). Nothing in the core imports it, and
`./run.sh test` passes without it.

## Use

Project cost and inspect the exact rendered request without spending anything:

```bash
python3 cli.py plan --show-prompt
```

Run the Phase 0 nulls and apply the kill rule:

```bash
python3 cli.py --n 200 run
```

Exit code 3 means the kill rule fired.

### Recording through mcp-tape

[mcp-tape](https://github.com/PlatAtlas/mcp-tape) is a byte-transparent
recording proxy. Pointing FOIL at it gives request/response capture, normalized
usage with cache fields, and TTFT for free, and the resulting trace opens in
[mcpreplay.dev](https://mcpreplay.dev):

```bash
mcp-tape llm --port 4141 &
ANTHROPIC_BASE_URL=http://127.0.0.1:4141 python3 cli.py --n 200 run
```

The proxy never retries, so retry policy stays owned by FOIL's executor, where
sampling integrity is decided.

## What Phase 0 measures

| | Manipulation | Purpose |
|---|---|---|
| **N1** | identical reports, permuted order | order-sensitivity noise floor |
| **N2** | semantically identical paraphrase | paraphrase noise floor |
| **REF** | single-source ablation under `Ω_null` | the signal the nulls have to be smaller than |

**Kill rule.** Let `T_null` be the 95th percentile TV across N1 and N2, and
`T_ablate` the median TV from single-source ablation. If
`T_null >= 0.5 · T_ablate`, the project stops and publishes the null result:
merely reordering or rephrasing identical evidence would be moving the action
distribution half as much as removing a source outright, and no amount of
sampling recovers signal from that.

## Cost

FOIL makes metered Messages API calls. A Claude Code subscription cannot serve
them. The Phase 0 nulls are cheap; Phase 1 is not, which is why cost is a gate
criterion (§8 G4) rather than a footnote.

| Sample size | Calls | Estimated cost (Sonnet-class) |
|---|---|---|
| n=30 | 330 | $0.19 – $0.34 |
| n=100 | 1100 | $0.62 – $1.15 |
| n=200 | 2200 | $1.23 – $2.29 |

Rates are estimates from a dated table vendored from mcp-tape
(`data/prices-anthropic.json`), never billing truth. An unknown model resolves
to **no** price rather than a guessed one: a silently wrong cost number is worse
than a missing one when cost gates a decision. `claude-opus-5` has no row yet
and must be added before it can be profiled.

## White-box layer (optional)

The API harness can only edit text, and editing text changes what the listener
knows *and* where everything after the edit sits. With open weights, `mech/`
does what a Messages call cannot:

- reads the action distribution **exactly** from the logits, with no sampling;
- hides a report **in place** (attention knockout): its information is gone,
  every token keeps its position. FOIL's own Ω_null effect then splits, along
  a registered path and a reverse one, into an information term, an
  arrangement term and a null-text term;
- finds the units that carry a report into the decision — in the model's own
  neurons or in a learned dictionary (SAE or transcoder), one interface — and
  tests them by patching.

```bash
python3 mech_run.py plan  --model Qwen/Qwen3-1.7B   # free: tokenizer-only structural gate
python3 mech_run.py gates --model Qwen/Qwen3-1.7B   # eligibility, leak, construct
python3 mech_run.py run   --model Qwen/Qwen3-1.7B   # refuses unless every gate passed
```

No API key and no spend; the cost is forward passes, reported in every result
file. The knockout only cuts every path in a pure softmax-attention decoder, so
hybrid models (linear attention, state-space layers — Qwen3.5, for one) are
refused, and a leak gate makes each model's weights confirm it.

## Layout

```
PREREGISTRATION.md   hypotheses, analysis plan, kill rules, amendment log
cli.py               plan / run
foil/ir.py           transcript IR; ablation addresses span IDs, never strings
foil/env.py          seeded scripted-scout environment
foil/render.py       IR + fork_key -> exact request payload, content-addressed
foil/execute.py      idempotent sampler, cost ledger, hard token budget
foil/stats.py        TV and bootstrap CIs (noise-floor diagnostics only)
foil/nulls.py        N1 / N2 / REF and the kill rule
mech/                white-box layer (optional; torch + transformers)
  local.py           FOIL stimulus -> chat template -> token map of every report
  whitebox.py        exact forward passes with knockouts, patches, captures
  basis.py           neurons or dictionaries (SAE / transcoder), one interface
  study.py           MECH-1 arms per episode; resumable study loop
  gates.py           G0-G3, each blocking
  analysis.py        registered statistics and decision rules (numpy only)
mech_run.py          plan / gates / run / analyze for MECH-1
data/                vendored price table
runs/                sample store (append-only jsonl) and results
```

## Two things worth knowing before you read the code

**Determinism is at the request layer, not the response layer.** The
measurement is a distribution over actions, so `temperature > 0` is mandatory
and responses are never reproducible. Replay reproduces the *stimulus*: the
rendered payload is content-addressed, and a hash already holding enough samples
is never re-executed. Temperature 0 is not a fix; it destroys the measurement.

**TV is a diagnostic, not the metric.** Total variation is unsigned, so a
listener that correctly identifies a liar and inverts its reports registers the
same large value as one that credulously follows it. It is used only for the
noise floor. The Shapley value function is `v(C) = P(correct action | C)`.

## License

MIT.
