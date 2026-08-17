"""Tests for evalgate.gates and the power additions.

The load-bearing test is `test_turn3_would_have_extended`: the module exists
because 32/36 against 0.90 should be "buy more information", not a one-run
cliff. If that verdict changes, the design rationale is gone.
"""

import math
import unittest

from . import gates, power


class TestBinomCdf(unittest.TestCase):
    def test_edges(self):
        self.assertEqual(gates.binom_cdf(-1, 10, 0.5), 0.0)
        self.assertEqual(gates.binom_cdf(10, 10, 0.5), 1.0)

    def test_known_value(self):
        # P(X <= 4 | n=10, p=0.5) = 0.376953125 exactly
        self.assertAlmostEqual(gates.binom_cdf(4, 10, 0.5), 0.376953125)

    def test_monotone_in_k(self):
        vals = [gates.binom_cdf(k, 20, 0.7) for k in range(21)]
        self.assertEqual(vals, sorted(vals))


class TestVerdict(unittest.TestCase):
    def test_turn3_would_have_extended(self):
        """The case the module exists for. 32/36 vs 0.90 is not evidence of
        failure (p = 0.49); it is an under-resolved measurement."""
        v = gates.verdict(32, 36, 0.90)
        self.assertEqual(v.outcome, "EXTEND")
        self.assertGreater(v.p_value, 0.4)

    def test_meeting_threshold_passes(self):
        self.assertEqual(gates.verdict(33, 36, 0.90).outcome, "PASS")
        self.assertEqual(gates.verdict(36, 36, 0.90).outcome, "PASS")

    def test_statistical_failure_fails(self):
        v = gates.verdict(25, 36, 0.90)
        self.assertEqual(v.outcome, "FAIL")
        self.assertLess(v.p_value, 0.05)

    def test_fail_requires_evidence_not_a_near_miss(self):
        """Every k from the FAIL boundary up to threshold*n must be EXTEND --
        there is no near-miss band that hard-fails."""
        for k in range(36 + 1):
            v = gates.verdict(k, 36, 0.90)
            if v.rate < 0.90 and v.p_value >= 0.05:
                self.assertEqual(v.outcome, "EXTEND", f"k={k}")

    def test_pooling_an_extension_can_resolve(self):
        """32/36 EXTEND; a same-rule re-verdict on pooled counts resolves."""
        self.assertEqual(gates.verdict(32, 36, 0.90).outcome, "EXTEND")
        self.assertEqual(gates.verdict(32 + 36, 72, 0.90).outcome, "PASS")
        # 60/72 is p = 0.053 -- still EXTEND, the rule does not round down
        self.assertEqual(gates.verdict(32 + 28, 72, 0.90).outcome, "EXTEND")
        self.assertEqual(gates.verdict(32 + 24, 72, 0.90).outcome, "FAIL")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            gates.verdict(0, 0, 0.9)


class TestPlan(unittest.TestCase):
    def test_turn3_gate_was_undersized(self):
        """Distinguishing 0.90 from 0.80 needs roughly double TURN-3's n=36."""
        d = gates.plan(0.90, 0.80)
        self.assertGreater(d.n, 36)
        self.assertGreaterEqual(d.power, 0.80)

    def test_design_is_self_consistent(self):
        d = gates.plan(0.90, 0.80)
        # the stated critical k must actually FAIL, and k+1 must not
        self.assertEqual(gates.verdict(d.fail_at_or_below, d.n, 0.90).outcome,
                         "FAIL")
        self.assertNotEqual(
            gates.verdict(d.fail_at_or_below + 1, d.n, 0.90).outcome, "FAIL")

    def test_smaller_gap_needs_larger_n(self):
        self.assertGreater(gates.plan(0.90, 0.85).n, gates.plan(0.90, 0.75).n)

    def test_rejects_bad_rates(self):
        with self.assertRaises(ValueError):
            gates.plan(0.80, 0.90)


class TestWilson(unittest.TestCase):
    def test_turn3_interval(self):
        lo, hi = power.wilson(32, 36)
        self.assertAlmostEqual(lo, 0.747, places=3)
        self.assertAlmostEqual(hi, 0.956, places=3)

    def test_degenerate(self):
        self.assertEqual(power.wilson(0, 0), (0.0, 1.0))
        lo, hi = power.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        lo, hi = power.wilson(10, 10)
        self.assertEqual(hi, 1.0)


class TestSeparate(unittest.TestCase):
    def test_normal_quantile(self):
        self.assertAlmostEqual(power._z(0.975), 1.959964, places=5)
        self.assertAlmostEqual(power._z(0.80), 0.841621, places=5)
        self.assertAlmostEqual(power._z(0.5), 0.0, places=9)

    def test_sweep_was_underpowered_and_by_how_much(self):
        """The sweep ran 12 episodes per model. Separating its two central
        rates needed an order of magnitude more."""
        n = power.separate(0.42, 0.58)
        self.assertGreater(n, 100)

    def test_big_gaps_are_cheap(self):
        self.assertLess(power.separate(0.42, 0.83), 30)

    def test_symmetric(self):
        self.assertEqual(power.separate(0.3, 0.6), power.separate(0.6, 0.3))

    def test_rejects_equal_rates(self):
        with self.assertRaises(ValueError):
            power.separate(0.5, 0.5)


if __name__ == "__main__":
    unittest.main()
