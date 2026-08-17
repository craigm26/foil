"""Tests for the canonical statistics in foil.stats."""

import unittest

from .stats import paired_permutation


class TestPairedPermutation(unittest.TestCase):
    def test_strong_effect_is_significant(self):
        self.assertLess(paired_permutation([0.4] * 20, iters=2000), 0.01)

    def test_null_is_not(self):
        deltas = [0.2, -0.2] * 10
        self.assertGreater(paired_permutation(deltas, iters=2000), 0.2)

    def test_matches_published_turn2_p_value(self):
        """TURN-2's per-scenario deltas (last-better coded negative under its
        first-minus-last convention) gave p = 1.0 one-sided. Reproduce the
        direction: an all-negative mean cannot beat sign-flips."""
        deltas = [-0.6] * 2 + [-0.2] * 9 + [0.0] * 21
        self.assertGreater(paired_permutation(deltas, iters=2000), 0.99)

    def test_deterministic(self):
        d = [0.1, -0.3, 0.2, 0.05]
        self.assertEqual(paired_permutation(d, iters=500, seed=7),
                         paired_permutation(d, iters=500, seed=7))


if __name__ == "__main__":
    unittest.main()
