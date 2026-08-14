# Real traffic: the phenomenon orderprobe detects is nearly absent from it

**Date:** 2026-08-13
**Corpus:** 1,855 Claude Code session transcripts, 30 projects, one machine,
one user. Scanned for structure only; no message content was read, recorded, or
transmitted.
**Cost:** $0. No model calls.

---

## 1. What this was meant to be

Every result in this project is measured on synthetic environments, and every
write-up says so. `orderprobe`'s README states plainly that real agent traffic
is untested. The plan was to close that gap by replaying real recorded
tool-result sequences in permuted order and measuring how often a real decision
is order-unstable. No ground truth needed: instability is observable without
labels.

That study cannot be run on this corpus, because the thing it measures barely
occurs in it.

## 2. What the corpus contains

Two scans, both structural.

**Tool results never arrive together.**

| tool results in one message | messages |
|---|---|
| 1 | **54,402** |
| 2 or more | **0** |

Every one of 54,402 tool-result messages carries exactly one result. That alone
is not decisive, since it reflects message framing rather than dispatch, so the
second scan looked at how many tools a turn dispatches at once.

**Tools are almost never dispatched in parallel.**

| tools dispatched in one turn | turns |
|---|---|
| 1 | 54,321 |
| 2 | 30 |
| 3 | 1 |
| 4 | 2 |

| | |
|---|---|
| Turns dispatching ≥2 in parallel | **33 of 54,354 (0.06%)** |
| Turns dispatching ≥3, the minimum orderprobe needs | **3** |

Three eligible decision points in 1,855 sessions. You cannot estimate an
instability rate from three points, and no amount of API budget changes that.

The three, in full: `Read, Read, Read`; `Bash, Read, Read, Read`; and
`WebFetch, WebFetch, WebFetch, WebFetch`.

## 3. What this establishes, and what it does not

**Establishes:** in this corpus, real agent execution is overwhelmingly
sequential. Each tool result is consumed before the next call is made, so the
order of results is *causal*, not arbitrary. Permuting a causal sequence does
not test order sensitivity; it manufactures an incoherent context and would
produce a number that looks like a finding and means nothing.

**This contradicts a claim the tool was shipping.** `orderprobe`'s "where it
fits" table led with *"tool results in an agent loop — return order is incidental
to the task."* On the only real traffic available to check, that is false. The
README has been corrected.

**Does not establish** that parallel dispatch is rare in general. This is one
user, one tool, one machine. Claude Code is fully capable of parallel dispatch
and is explicitly prompted toward it; these sessions simply did not use it much.
Frameworks built around fan-out — orchestrators that spawn subagents and
aggregate their reports, retrieval pipelines that assemble ranked chunks,
multi-source fusion — would look completely different, and are exactly where the
premise should hold. None were present here to measure.

**Does not retract the PID-2 result.** That environment had four genuinely
simultaneous, mutually independent sources, which is the structure the detector
is for. What this scan changes is the estimate of *how common* that structure is
in deployed agent loops, and the honest answer from this corpus is: rarer than
the tool's framing implied.

## 4. Consequence for the tool

`orderprobe` should be pointed at decision points where several independent
sources are assembled into one context *before* the decision, not at sequential
agent loops:

- retrieved chunks ranked by a scorer that is not the decision
- subagent or scout reports aggregated by an orchestrator
- multi-source fusion where inputs are gathered concurrently and then read
- peer messages in a single deliberation round

The distinguishing test is whether the inputs could have been gathered in any
order without changing what they say. In a sequential agent loop they could not,
because each call is chosen in light of the last result.

## 5. Reproduce

The scan reads only block types, tool names, and payload sizes. It never reads
or emits message content.

```bash
python3 tools/scan_real_traffic.py            # ~2 min over 1,855 transcripts
```
