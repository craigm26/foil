# Fixtures verification: all four predictions held, including the uncomfortable one

**Date:** 2026-08-17
**Model:** `claude-sonnet-5`
**Protocol:** predictions P1–P4 committed in `fixtures_verify.py` before any
call. 7 fixtures × 20 cases × 3 samples = 420 calls. Cost **$0.27**.

---

## 1. Results

Every fixture, full information, no manipulation, scored against its own
`intended`:

| fixture | agrees with scorer | prediction |
|---|---|---|
| `under_determined` | 60/60 = 1.000 | P4 ✓ construct is the wrong gate for it |
| `over_determined` | 60/60 = 1.000 | P4 ✓ |
| `no_error_variance` | 60/60 = 1.000 | P2 ✓ its perfection **is** its invalidity |
| `scorer_disagrees` | 60/60 = 1.000 | **P3 ✓ — see §2** |
| `decisive` | 60/60 = 1.000 | P1 ✓ |
| `two_candidate` | 60/60 = 1.000 | P1 ✓ |
| `calibrated_variance` | 60/60 = 1.000 | P1 ✓ |

## 2. P3 confirmed: `scorer_disagrees` cannot enact its invalidity against a real oracle

The real model agreed with the fixture's scorer on all 60 runs. That was
predicted, and it is a genuine limitation, now stated in the fixture's own
documentation:

The fixture's disagreement lives in a hidden field that only the simulated
oracle reads. Its rendered text is self-consistent — the evidence genuinely
supports the scorer's answer — so a competent real model *agrees*. The fixture
therefore verifies your gate's **wiring** (does it call the oracle, compare
answers, and reject on mismatch), not your model's **judgement**.

The deeper point stands on its own: **a genuine scorer–model disagreement
cannot be synthesized analytically.** If you could compute, without a model,
which self-consistent scorings a model will reject, you would not need the
construct gate at all. TURN-1's disagreement was discovered by paying for it,
not designed. So the construct gate must be run against *your* scorer with
*your* model on *your* environment, every time — the fixture can only prove
your harness would notice.

Had P3 failed — the model disagreeing with a scorer on sound, fully-attested
text — that would have been a real construct gap inside a fixture we believed
self-consistent, and considerably more alarming.

## 3. What this run bought

- The wiring claim is now measured, not simulated: the request path, parser,
  and scoring loop behave against the real API exactly as the 54-test suite
  says they do against the simulated oracle.
- `no_error_variance` scoring a perfect 60/60 with a real model is the PID-1
  failure reproduced live for 4 cents a fixture: nothing about that perfection
  looks wrong from inside the construct gate.
- The predicted limitation of `scorer_disagrees` is now a documented property
  with data behind it, rather than a caveat discovered by a user.

## 4. Data

- `runs/fixtures-verify-result.json` — per-fixture counts, disagreement
  samples (none occurred), provenance stamp, ledger.

Reproduce with `./run.sh fixtures_verify.py`.
