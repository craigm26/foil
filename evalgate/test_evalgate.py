"""Tests for evalgate. python3 -m unittest discover -s evalgate -t ."""
import unittest
from . import construct, power, structural


class TestStructural(unittest.TestCase):
    def test_all_hold(self):
        r = structural.verify(lambda s: {"v": s % 5}, lambda c: c["v"] < 5, n=50)
        self.assertTrue(r.ok)
        self.assertEqual(r.hold_rate, 1.0)

    def test_detects_violation(self):
        r = structural.verify(lambda s: {"v": s}, lambda c: c["v"] < 10, n=20)
        self.assertFalse(r.ok)
        self.assertEqual(r.n_holding, 10)
        self.assertTrue(r.failures)

    def test_none_cases_are_not_counted_as_holding(self):
        """A generator that mostly returns None must not look like success."""
        r = structural.verify(lambda s: {"v": s} if s % 4 == 0 else None,
                              lambda c: True, n=20)
        self.assertEqual(r.n_generated, 5)
        self.assertEqual(r.admission_rate, 0.25)
        self.assertTrue(r.ok)

    def test_zero_generated_is_not_ok(self):
        """Nothing generated must not report ok -- vacuous truth is the classic
        way a gate passes while measuring nothing."""
        r = structural.verify(lambda s: None, lambda c: True, n=20)
        self.assertFalse(r.ok)

    def test_predicate_exception_counts_as_failure(self):
        def boom(c):
            raise ValueError("bad")
        r = structural.verify(lambda s: {"v": s}, boom, n=5)
        self.assertFalse(r.ok)
        self.assertEqual(r.n_holding, 0)

    def test_generator_exception_is_reported(self):
        def gen(s):
            if s == 3:
                raise RuntimeError
            return {"v": s}
        r = structural.verify(gen, lambda c: True, n=6)
        self.assertEqual(r.n_generated, 5)
        self.assertTrue(any("generator raised" in m for _, m in r.failures))


class TestPower(unittest.TestCase):
    def test_tie_rate_matches_turn1(self):
        """The exact arithmetic that made TURN-1's margin unsatisfiable."""
        self.assertAlmostEqual(power.tie_rate(0.854, 3), 0.493, places=2)

    def test_more_reps_reduces_ties(self):
        self.assertLess(power.tie_rate(0.7, 7), power.tie_rate(0.7, 3))

    def test_bigger_design_has_more_power(self):
        small = power.paired(effect=0.2, units=12, reps=3, base=0.7, trials=60)
        big = power.paired(effect=0.2, units=40, reps=7, base=0.7, trials=60)
        self.assertLess(small.power, big.power)

    def test_null_respects_alpha(self):
        r = power.paired(effect=0.0, units=24, reps=5, base=0.7,
                         alpha=0.01, trials=120)
        self.assertLessEqual(r.false_positive_rate, 0.06)

    def test_zero_effect_has_no_power(self):
        r = power.paired(effect=0.0, units=24, reps=5, base=0.7, trials=80)
        self.assertLess(r.power, 0.15)


class TestConstruct(unittest.TestCase):
    def test_agreement_passes(self):
        cases = [{"id": i, "want": "B"} for i in range(8)]
        r = construct.verify(lambda c: "B", cases, lambda c: c["want"], reps=2)
        self.assertTrue(r.ok)
        self.assertEqual(r.rate, 1.0)

    def test_disagreement_fails_and_is_reported(self):
        """The TURN-1 failure: the model reliably answers something else."""
        cases = [{"id": i, "want": "B"} for i in range(8)]
        r = construct.verify(lambda c: "C", cases, lambda c: c["want"], reps=2)
        self.assertFalse(r.ok)
        self.assertEqual(r.rate, 0.0)
        self.assertTrue(r.disagreements)
        self.assertEqual(r.disagreements[0][1:], ("B", "C"))

    def test_borderline_respects_min_rate(self):
        cases = [{"want": "B"}] * 10
        seq = iter(["B"] * 17 + ["C"] * 3)
        r = construct.verify(lambda c: next(seq), cases, lambda c: c["want"],
                             reps=2, min_rate=0.90, max_workers=1)
        self.assertAlmostEqual(r.rate, 0.85)
        self.assertFalse(r.ok)

    def test_errors_do_not_count_as_agreement(self):
        def flaky(c):
            raise RuntimeError
        r = construct.verify(flaky, [{"want": "B"}] * 4, lambda c: c["want"], reps=1)
        self.assertEqual(r.errors, 4)
        self.assertFalse(r.ok)


if __name__ == "__main__":
    unittest.main()
