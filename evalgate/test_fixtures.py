"""Tests for evalgate.fixtures.

The load-bearing test is `test_scorer_disagrees_is_structurally_invisible`. If
that ever fails, the fixture has stopped reproducing the failure it exists to
reproduce, and the whole point of the module is gone.

Run: python3 -m unittest discover evalgate -t . -v
"""

import unittest

from . import construct, fixtures as F, variance

SEEDS = range(120)


def _cases(name, seeds=SEEDS):
    f = F.get(name)
    return [f.make_case(s) for s in seeds]


class TestGenerators(unittest.TestCase):
    def test_all_generate_without_raising(self):
        for f in F.FIXTURES:
            with self.subTest(f.name):
                cases = [f.make_case(s) for s in range(60)]
                self.assertEqual(len(cases), 60)

    def test_deterministic(self):
        for f in F.FIXTURES:
            with self.subTest(f.name):
                self.assertEqual(f.make_case(7).claims, f.make_case(7).claims)

    def test_rendered_text_matches_claims(self):
        """A checker must be able to re-derive the property from what the model
        sees, so the text has to carry every claim."""
        for f in F.FIXTURES:
            with self.subTest(f.name):
                c = f.make_case(3)
                for src, reports in c.claims.items():
                    for route in reports:
                        self.assertIn(f"{src} reports {route} is", c.text)

    def test_answer_fields_never_leak_into_text(self):
        """`_model_answer` must be invisible, or scorer_disagrees is detectable
        for the wrong reason."""
        for c in _cases("scorer_disagrees", range(40)):
            self.assertNotEqual(c._model_answer, c.intended)
            # the text must not reveal which of the two the scorer picked
            self.assertNotIn("intended", c.text)
            self.assertNotIn("correct", c.text)


class TestStructuralSeparation(unittest.TestCase):
    def test_structural_failures_are_unsound(self):
        for name in ("under_determined", "over_determined"):
            with self.subTest(name):
                unsound = [c for c in _cases(name) if not F._structurally_sound(c.claims)]
                self.assertGreater(len(unsound), 0.9 * len(SEEDS),
                                   "should be structurally detectable")

    def test_valid_controls_are_sound(self):
        for name in ("decisive", "two_candidate", "calibrated_variance"):
            with self.subTest(name):
                for c in _cases(name):
                    self.assertTrue(F._structurally_sound(c.claims))

    def test_scorer_disagrees_is_structurally_invisible(self):
        """THE point of the module. This fixture must be indistinguishable from
        a valid one to any check that does not call a model."""
        for c in _cases("scorer_disagrees"):
            self.assertTrue(F._structurally_sound(c.claims))
        # and its self-consistency must hold: the scorer agrees with the
        # reference reasoner, so there is no free way to notice the problem
        for c in _cases("scorer_disagrees", range(60)):
            self.assertEqual(F.reference_argmax(c.claims)[0], c.intended)

    def test_over_determined_has_no_decisive_source(self):
        for c in _cases("over_determined", range(40)):
            self.assertEqual(F.decisive_sources(c.claims), [])

    def test_under_determined_leaves_a_route_unattested(self):
        for c in _cases("under_determined", range(40)):
            self.assertNotEqual(F.attested_routes(c.claims), set(F.ROUTES))


class TestAudit(unittest.TestCase):
    def test_reference_check_catches_both_free_failures(self):
        r = F.audit(F.default_structural_check, n=60)
        self.assertEqual(sorted(r.caught), ["over_determined", "under_determined"])
        self.assertEqual(r.false_alarms, [], "valid environments must not be flagged")
        self.assertEqual(r.errors, [])
        self.assertTrue(r.ok)

    def test_construct_only_failures_reported_separately(self):
        r = F.audit(F.default_structural_check, n=40)
        self.assertEqual(sorted(r.unreachable_free),
                         ["no_error_variance", "scorer_disagrees"])
        # they are missed, but that must not count against the verdict
        self.assertEqual(r.reachable_missed, [])
        self.assertTrue(r.ok)

    def test_accept_everything_scores_zero(self):
        r = F.audit(lambda mk: True, n=20)
        self.assertEqual(r.caught, [])
        self.assertFalse(r.ok)

    def test_reject_everything_collects_false_alarms(self):
        r = F.audit(lambda mk: False, n=20)
        self.assertEqual(sorted(r.false_alarms),
                         ["calibrated_variance", "decisive", "two_candidate"])
        self.assertFalse(r.ok)

    def test_raising_check_is_recorded_not_swallowed(self):
        def boom(mk):
            raise RuntimeError("boom")
        r = F.audit(boom, n=10)
        self.assertEqual(len(r.errors), len(F.FIXTURES))
        self.assertFalse(r.ok)


class TestGateMatrix(unittest.TestCase):
    """Each invalid fixture must be caught by its designated gate, and every
    valid one must pass both paid gates."""

    def _reports(self, name):
        cases = _cases(name)
        c = construct.verify(F.simulated_oracle, cases,
                             truth=lambda x: x.intended, min_rate=0.90, reps=1)
        v = variance.verify(F.simulated_experimental_oracle, cases,
                            truth=lambda x: x.intended, min_error_rate=0.05)
        return c, v

    def test_construct_catches_scorer_disagrees_only(self):
        for f in F.FIXTURES:
            c, _ = self._reports(f.name)
            with self.subTest(f.name):
                if f.name == "scorer_disagrees":
                    self.assertFalse(c.ok)
                else:
                    self.assertTrue(c.ok)

    def test_variance_catches_no_error_variance(self):
        _, v = self._reports("no_error_variance")
        self.assertTrue(v.at_ceiling)
        self.assertFalse(v.ok)
        self.assertIn("no errors", v.explain())

    def test_valid_controls_pass_both_paid_gates(self):
        for name in ("decisive", "two_candidate", "calibrated_variance"):
            c, v = self._reports(name)
            with self.subTest(name):
                self.assertTrue(c.ok, "construct should accept a valid environment")
                self.assertTrue(v.ok, "variance should find something to detect")


class TestVarianceGate(unittest.TestCase):
    def test_ceiling_and_floor_are_distinguished(self):
        cases = _cases("decisive", range(20))
        hi = variance.verify(lambda c: c.intended, cases, truth=lambda c: c.intended)
        self.assertTrue(hi.at_ceiling)
        self.assertFalse(hi.at_floor)
        lo = variance.verify(lambda c: "nonsense", cases, truth=lambda c: c.intended)
        self.assertTrue(lo.at_floor)
        self.assertFalse(lo.at_ceiling)
        self.assertFalse(lo.ok, "all-wrong is a broken scorer, not signal")

    def test_explain_names_the_pid1_failure(self):
        cases = _cases("no_error_variance", range(20))
        r = variance.verify(lambda c: c.intended, cases, truth=lambda c: c.intended)
        self.assertIn("PID-1", r.explain())


if __name__ == "__main__":
    unittest.main()
