"""Tests for orderprobe. Run: python3 -m unittest discover orderprobe -v"""

import unittest
from collections import Counter

from . import Verdict, probe, probe_call


class TestOrderings(unittest.TestCase):
    def test_canonical_first(self):
        from . import _orderings
        for n in (2, 3, 5, 9):
            self.assertEqual(_orderings(n, 6, 0)[0], tuple(range(n)))

    def test_distinct(self):
        from . import _orderings
        o = _orderings(6, 6, 0)
        self.assertEqual(len(o), len(set(o)), "repeated ordering fakes agreement")

    def test_capped_at_factorial(self):
        """Asking for more orderings than exist must not loop or repeat."""
        from . import _orderings
        self.assertEqual(len(_orderings(3, 20, 0)), 6)   # 3! == 6
        self.assertEqual(len(_orderings(2, 20, 0)), 2)
        self.assertEqual(len(_orderings(1, 20, 0)), 1)
        self.assertEqual(len(_orderings(0, 20, 0)), 1)

    def test_seeded(self):
        from . import _orderings
        self.assertEqual(_orderings(7, 5, 42), _orderings(7, 5, 42))
        self.assertNotEqual(_orderings(7, 5, 1), _orderings(7, 5, 2))

    def test_large_n_samples_without_enumerating(self):
        from . import _orderings
        o = _orderings(50, 4, 0)          # 50! must not be enumerated
        self.assertEqual(len(o), 4)
        self.assertEqual(len(set(o)), 4)


class TestProbe(unittest.TestCase):
    def test_order_independent_is_stable(self):
        r = probe_call(lambda xs: sum(xs), [1, 2, 3, 4], k=6, samples=3)
        self.assertEqual(r.verdict, Verdict.STABLE)
        self.assertEqual(r.dispersion, 0.0)
        self.assertEqual(r.value, 10)

    def test_order_dependent_is_unstable(self):
        r = probe_call(lambda xs: xs[0], [1, 2, 3, 4], k=6, samples=3)
        self.assertEqual(r.verdict, Verdict.UNSTABLE)
        self.assertGreater(r.dispersion, 0.0)

    def test_value_is_the_canonical_answer(self):
        """`value` must be what you'd have shipped -- the given order, not a
        vote across permutations."""
        r = probe_call(lambda xs: xs[0], ["given", "b", "c"], k=6, samples=3)
        self.assertEqual(r.value, "given")

    def test_single_item_is_not_applicable(self):
        """One item cannot be reordered. That is 'not checked', not 'stable'."""
        r = probe_call(lambda xs: xs[0], ["only"], k=6, samples=3)
        self.assertEqual(r.verdict, Verdict.NOT_APPLICABLE)
        self.assertFalse(r.unstable)

    def test_empty_is_not_applicable(self):
        r = probe_call(lambda xs: len(xs), [], k=6, samples=3)
        self.assertEqual(r.verdict, Verdict.NOT_APPLICABLE)

    def test_call_budget_is_respected(self):
        seen = []
        probe_call(lambda xs: seen.append(1), [1, 2, 3, 4], k=5, samples=3)
        self.assertEqual(len(seen), 15)

    def test_call_budget_capped_by_factorial(self):
        seen = []
        r = probe_call(lambda xs: seen.append(1) or 0, [1, 2], k=10, samples=2)
        self.assertEqual(len(seen), 4)      # 2! * 2
        self.assertEqual(r.calls, 4)

    def test_unhashable_return_is_handled(self):
        r = probe_call(lambda xs: {"first": xs[0]}, [1, 2, 3], k=4, samples=2)
        self.assertIn(r.verdict, (Verdict.STABLE, Verdict.UNSTABLE))

    def test_custom_key(self):
        """A key that collapses the difference must report stable."""
        r = probe_call(lambda xs: xs[0], [1, 2, 3], k=6, samples=2,
                       key=lambda v: "same")
        self.assertEqual(r.verdict, Verdict.STABLE)

    def test_threshold_mode(self):
        """threshold is a strict '>' on dispersion, so a threshold equal to the
        observed dispersion reads as stable."""
        r = probe_call(lambda xs: xs[0], [1, 2, 3], k=6, samples=2, threshold=0.99)
        self.assertEqual(r.dispersion, 1.0)
        self.assertEqual(r.verdict, Verdict.UNSTABLE)    # 1.0 > 0.99
        r2 = probe_call(lambda xs: xs[0], [1, 2, 3], k=6, samples=2, threshold=1.0)
        self.assertEqual(r2.verdict, Verdict.STABLE)     # 1.0 is not > 1.0
        r3 = probe_call(lambda xs: xs[0], [1, 2, 3], k=6, samples=2, threshold=0.5)
        self.assertEqual(r3.verdict, Verdict.UNSTABLE)

    def test_errors_counted_not_swallowed(self):
        calls = {"n": 0}

        def flaky(xs):
            calls["n"] += 1
            if calls["n"] % 2:
                raise RuntimeError("boom")
            return "ok"

        r = probe_call(flaky, [1, 2, 3], k=4, samples=2)
        self.assertGreater(r.errors, 0)
        self.assertEqual(r.calls, 8)

    def test_all_errors_is_not_applicable(self):
        def always(xs):
            raise RuntimeError

        r = probe_call(always, [1, 2, 3], k=3, samples=2)
        self.assertEqual(r.verdict, Verdict.NOT_APPLICABLE)
        self.assertEqual(r.errors, 6)

    def test_deterministic(self):
        f = lambda xs: xs[0]
        a = probe_call(f, [1, 2, 3, 4], k=5, samples=3, seed=7)
        b = probe_call(f, [1, 2, 3, 4], k=5, samples=3, seed=7)
        self.assertEqual(a.by_ordering, b.by_ordering)

    def test_concurrent_matches_serial(self):
        f = lambda xs: xs[0]
        a = probe_call(f, [1, 2, 3, 4, 5], k=5, samples=3, seed=3, max_workers=1)
        b = probe_call(f, [1, 2, 3, 4, 5], k=5, samples=3, seed=3, max_workers=4)
        self.assertEqual(a.by_ordering, b.by_ordering)
        self.assertEqual(a.value, b.value)

    def test_decorator(self):
        @probe(k=4, samples=2)
        def decide(items):
            return items[0]

        r = decide(["a", "b", "c"])
        self.assertEqual(r.value, "a")
        self.assertEqual(r.verdict, Verdict.UNSTABLE)
        self.assertEqual(decide.unprobed(["x", "y"]), "x")

    def test_decorator_passes_through_extra_args(self):
        @probe(k=3, samples=1)
        def decide(items, prefix=""):
            return prefix + items[0]

        r = decide(["a", "b", "c"], prefix="p-")
        self.assertTrue(all(a.startswith("p-") for a in r.answers))


class TestStatistic(unittest.TestCase):
    def test_tv_bounds(self):
        from . import _tv
        self.assertEqual(_tv(Counter("aaa"), Counter("aaa")), 0.0)
        self.assertEqual(_tv(Counter("aaa"), Counter("bbb")), 1.0)
        self.assertAlmostEqual(_tv(Counter("aab"), Counter("abb")), 1 / 3)

    def test_dispersion_is_max_pairwise(self):
        """Two orderings agreeing must not mask a third that disagrees."""
        seq = iter(["x", "x", "x", "x", "y", "y"])
        r = probe_call(lambda xs: next(seq), [1, 2, 3], k=3, samples=2)
        self.assertEqual(r.dispersion, 1.0)


if __name__ == "__main__":
    unittest.main()
