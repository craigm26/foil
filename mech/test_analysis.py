"""Registered statistics, on synthetic units. numpy only; runs everywhere."""

from __future__ import annotations

import unittest

import numpy as np

from foil.nulls import _verdict
from foil.stats import percentile, total_variation
from mech import analysis as A


def unit(ep, src, full, ko, null_ko, null, dele, ko_sum=None, ko_con=None, carriers=None,
         role="honest", shift=None, on_format=0.99, span_tokens=10, last_slot=False):
    p = {"full": full, "ko": ko, "null_ko": null_ko, "null": null, "del": dele,
         "ko_sum": ko if ko_sum is None else ko_sum,
         "ko_con": full if ko_con is None else ko_con}
    if shift is not None:
        p["shift"] = shift
    return {"episode": ep, "source": src, "role": role, "p": p, "carriers": carriers,
            "on_format": {k: on_format for k in p}, "span_tokens": span_tokens,
            "last_slot": last_slot}


class Decomposition(unittest.TestCase):
    def test_identity_is_exact(self):
        u = unit("e", "A", full=0.9, ko=0.3, null_ko=0.35, null=0.2, dele=0.4)
        c = A.components(u)
        self.assertAlmostEqual(c["info"], -0.6)
        self.assertAlmostEqual(c["arr_null"], 0.05)
        self.assertAlmostEqual(c["text_null"], -0.15)
        self.assertAlmostEqual(c["arr_del"], 0.1)
        self.assertAlmostEqual(c["d_null"], c["info"] + c["arr_null"] + c["text_null"])
        self.assertAlmostEqual(c["d_del"], c["info"] + c["arr_del"])

    def test_cancellation_counts_as_arrangement(self):
        # Information and arrangement cancel: the net effect is small, but the
        # share must still report the arrangement term at full size.
        u = unit("e", "A", full=0.5, ko=0.0, null_ko=0.5, null=0.5, dele=0.5)
        self.assertAlmostEqual(A.arrangement_share([u], "null"), 0.5)

    def test_two_paths_agree_on_the_total(self):
        u = unit("e", "A", full=0.9, ko=0.3, null_ko=0.35, null=0.2, dele=0.4, shift=0.8)
        c = A.components(u)
        self.assertAlmostEqual(c["info"] + c["arr_null"], c["info_rev"] + c["arr_rev"])
        self.assertAlmostEqual(c["arr_rev"], -0.1)

    def test_off_format_unit_is_a_dropout(self):
        self.assertTrue(A.valid_for(unit("e", "A", 0.5, 0.2, 0.2, 0.2, 0.2), "H1"))
        self.assertFalse(A.valid_for(unit("e", "A", 0.5, 0.2, 0.2, 0.2, 0.2, on_format=0.3), "H1"))

    def test_a_failing_carrier_pass_does_not_touch_h1(self):
        u = unit("e", "A", 0.5, 0.2, 0.2, 0.2, 0.2)
        u["on_format"].update({"car_k16": 0.1, "car_matched": 0.9})
        self.assertTrue(A.valid_for(u, "H1_any"))
        self.assertTrue(A.valid_for(u, "H2"))
        self.assertFalse(A.valid_for(u, "H3"))

    def test_activity_uses_every_operator(self):
        self.assertFalse(A.is_active(unit("e", "A", 0.5, 0.52, 0.51, 0.49, 0.5)))
        self.assertTrue(A.is_active(unit("e", "A", 0.5, 0.5, 0.5, 0.5, 0.75)))


def _population(n_eps, arr, rng):
    units = []
    for e in range(n_eps):
        for s in "ABCD":
            full = 0.8
            info = -0.5 + rng.normal(0, 0.05)
            a = arr + rng.normal(0, 0.01)
            ko = full + info
            null_ko = ko + a
            null = null_ko + 0.02
            units.append(unit(f"ep{e}", s, full, ko, null_ko, null, ko + a, shift=full + a))
    return units


class H1(unittest.TestCase):
    def test_minor(self):
        r = A.h1(_population(24, 0.01, np.random.default_rng(0)))
        self.assertEqual(r["outcome"], "ARRANGEMENT MINOR")

    def test_material(self):
        r = A.h1(_population(24, 0.4, np.random.default_rng(1)))
        self.assertEqual(r["outcome"], "ARRANGEMENT MATERIAL")

    def test_degenerate_below_minimums(self):
        r = A.h1(_population(4, 0.01, np.random.default_rng(2)))
        self.assertEqual(r["outcome"], "DEGENERATE")

    def test_strata_reported(self):
        units = _population(24, 0.01, np.random.default_rng(3))
        flags = {f"ep{e}": e % 2 == 0 for e in range(24)}
        r = A.h1(units, flags)
        self.assertEqual(r["regardless"]["by_order_stability"]["bistable"]["n_episodes"], 12)
        self.assertIsNotNone(r["regardless"]["arr_share_reverse_path"]["hi"])

    def test_regardless_survives_degeneracy(self):
        r = A.h1(_population(4, 0.01, np.random.default_rng(5)))
        self.assertEqual(r["outcome"], "DEGENERATE")
        self.assertIn("arr_share_del", r["regardless"])


def _routing(n_eps, sum_frac, con_frac):
    units = []
    for e in range(n_eps):
        for s in "ABCD":
            full, ko = 0.9, 0.2
            info = ko - full
            units.append(unit(f"ep{e}", s, full, ko, ko, ko, ko,
                              ko_sum=full + sum_frac * info, ko_con=full + con_frac * info))
    return units


class H2(unittest.TestCase):
    def test_outcomes(self):
        cases = {
            "SUMMARY-ROUTED": (0.9, 0.1),
            "CONTENT-ROUTED": (0.1, 0.9),
            "REDUNDANT": (0.1, 0.1),
            "JOINT": (0.9, 0.9),
        }
        for want, (fs, fc) in cases.items():
            with self.subTest(want=want):
                self.assertEqual(A.h2(_routing(16, fs, fc))["outcome"], want)

    def test_missing_interval_resolves_nothing(self):
        self.assertEqual(A.route_outcome((0.9, None, 1.0), (0.1, 0.0, 0.2)), "UNRESOLVED")

    def test_motif_reading(self):
        self.assertEqual(A.motif_reading("SUMMARY-ROUTED", (0.7, 0.6, 0.8), 30), "SUPPORTS THE MOTIF")
        self.assertEqual(A.motif_reading("SUMMARY-ROUTED", (0.2, 0.1, 0.3), 30),
                         "CONFOUNDED WITH LINE STRUCTURE")
        self.assertEqual(A.motif_reading("JOINT", (0.7, 0.6, 0.8), 30), "AGAINST THE MOTIF")
        self.assertEqual(A.motif_reading("REDUNDANT", (0.7, 0.6, 0.8), 30), "UNINFORMATIVE ABOUT THE MOTIF")

    def test_prediction_gets_no_easier_rule(self):
        # The point estimates favour SUMMARY-ROUTED, but r_con's interval
        # reaches the line: that is not enough to declare it.
        self.assertEqual(A.route_outcome((0.8, 0.6, 0.9), (0.3, 0.1, 0.55)), "UNRESOLVED")
        self.assertEqual(A.route_outcome((0.8, 0.6, 0.9), (0.3, 0.1, 0.45)), "SUMMARY-ROUTED")


def _carriers(n_eps, frac16, frac_matched, frac_pre=0.0, size_ratio=0.9):
    units = []
    for e in range(n_eps):
        for s in "ABCD":
            full, ko = 0.9, 0.3
            info = ko - full
            car = {"by_k": {k: full + min(1.0, frac16 * k / 16) * info for k in A.K_LADDER},
                   "matched": full + frac_matched * info, "top_abs": full + 0.1 * info,
                   "matched_size_ratio": size_ratio,
                   "shortfall": {}, "top": [], "n_layers": 4,
                   "pre_answer": {"p": full + frac_pre * info, "matched": full, "top_abs": full,
                                  "matched_size_ratio": size_ratio, "top": [], "shortfall": None}}
            u = unit(f"ep{e}", s, full, ko, ko, ko, ko, carriers=car)
            u["on_format"].update({f"car_k{k}": 0.99 for k in A.K_LADDER})
            u["on_format"].update({"car_matched": 0.99, "car_top_abs": 0.99,
                                   "pre": 0.99, "pre_matched": 0.99, "pre_top_abs": 0.99})
            units.append(u)
    return units


class H3(unittest.TestCase):
    def test_few(self):
        self.assertEqual(A.h3(_carriers(16, 0.8, 0.0))["outcome"], "FEW-NEURON MEDIATION")

    def test_not_few(self):
        self.assertEqual(A.h3(_carriers(16, 0.2, 0.0))["outcome"], "NOT FEW")

    def test_an_undersized_control_cannot_confirm_specificity(self):
        # Attribution picks each cell's biggest change, so substitutes run
        # smaller; a win over a much smaller control proves nothing.
        r = A.h3(_carriers(16, 0.8, 0.0, size_ratio=0.3))
        self.assertEqual(r["outcome"], "MEDIATION, SPECIFICITY UNRESOLVED")

    def test_matched_control_must_be_beaten(self):
        # Other units in the same cells, changed as much, doing as well:
        # localized, but not specific to the chosen neurons.
        self.assertEqual(A.h3(_carriers(16, 0.8, 0.8))["outcome"], "MEDIATION, NOT UNIT-SPECIFIC")

    def test_pre_answer_variant_is_separate(self):
        # Mediation at the answer does not imply mediation upstream of it.
        units = _carriers(16, 0.9, 0.0, frac_pre=0.1)
        self.assertEqual(A.h3(units)["outcome"], "FEW-NEURON MEDIATION")
        b = A.h3(units, variant="pre")
        self.assertEqual(b["hypothesis"], "H3b")
        self.assertEqual(b["outcome"], "NOT FEW")


class Integrity(unittest.TestCase):
    def test_void_result_reports_only_the_failure(self):
        rows = [{"episode": "e1", "consistency": {"ok": True}},
                {"episode": "e2", "consistency": {"ok": False}}]
        r = A.void_result(rows)
        self.assertEqual(r["integrity"]["void_episodes"], ["e2"])
        self.assertTrue(all(r[h]["outcome"] == "VOID" for h in ("R0", "H1", "H2", "H3", "H3b")))


class Bootstrap(unittest.TestCase):
    def test_resamples_episodes_not_units(self):
        # One episode with a huge value and 9 with zero: a unit-level bootstrap
        # with 4 units per episode would give a different spread. Check only
        # that the procedure is deterministic and brackets the point.
        units = [{"episode": f"e{i}", "v": (10.0 if i == 0 else 0.0)} for i in range(10) for _ in range(4)]
        stat = lambda us: float(np.mean([u["v"] for u in us]))
        a = A.cluster_bootstrap(units, stat, iters=500)
        b = A.cluster_bootstrap(units, stat, iters=500)
        self.assertEqual(a, b)
        self.assertLessEqual(a[1], a[0])
        self.assertGreaterEqual(a[2], a[0])


class Phase0(unittest.TestCase):
    def test_matches_the_core_rule(self):
        eps = []
        rng = np.random.default_rng(4)
        for e in range(6):
            probs = [rng.dirichlet(np.ones(4)).tolist() for _ in range(6)]
            eps.append({"episode": f"e{e}", "correct_index": 0,
                        "n1": [{"probs": p} for p in probs],
                        "n2": {"probs": rng.dirichlet(np.ones(4)).tolist()},
                        "ref": {s: rng.dirichlet(np.ones(4)).tolist() for s in "ABCD"}})
        got = A.phase0(eps)
        pooled_null, pooled_ref = [], []
        for e in eps:
            d = [np.asarray(x["probs"]) for x in e["n1"]]
            pooled_null += [total_variation(d[i], d[j]) for i in range(6) for j in range(i + 1, 6)]
            pooled_null.append(total_variation(d[0], np.asarray(e["n2"]["probs"])))
            pooled_ref += [total_variation(d[0], np.asarray(v)) for v in e["ref"].values()]
        self.assertEqual(got["T_null_p95"], percentile(pooled_null, 95))
        self.assertEqual(got["T_ablate_median"], percentile(pooled_ref, 50))
        self.assertEqual(got["verdict"], _verdict(got["T_null_p95"], got["T_ablate_median"]))
        self.assertEqual(got["pooled_null_n"], 6 * 16)


if __name__ == "__main__":
    unittest.main()


class RunnerValidation(unittest.TestCase):
    """mech_run.validate_rows: checked before any statistic exists."""

    def _rows(self, n=48, commit="abc", dirty=False):
        from mech.gates import STUDY_SEEDS
        ident = {"model": "m", "revision": "r", "dtype": "float32", "device": "cpu",
                 "hardware": "x", "torch": "t", "transformers": "tf"}
        return [{"episode": f"v3-s{s}", "seed": s, "identity": dict(ident),
                 "provenance": {"git_commit": commit, "git_dirty": dirty}}
                for s in STUDY_SEEDS[:n]]

    def _gates(self, ok=True, **over):
        ident = {"model": "m", "revision": "r", "dtype": "float32", "device": "cpu",
                 "hardware": "x", "torch": "t", "transformers": "tf"}
        ident.update(over)
        return {"all_pass": ok, "identity": ident, "gates": []}

    def test_clean_complete_gated_rows_pass(self):
        import mech_run
        self.assertIsNone(mech_run.validate_rows(self._rows(), self._gates(), exploratory=False))

    def test_refusals(self):
        import mech_run
        rows = self._rows()
        cases = {
            "duplicate": rows + [dict(rows[0])],
            "incomplete": self._rows(47),
            "dirty": self._rows(dirty=True),
        }
        mixed = self._rows()
        mixed[3] = dict(mixed[3], provenance={"git_commit": "other", "git_dirty": False})
        cases["two commits"] = mixed
        other_hw = self._rows()
        other_hw[5] = dict(other_hw[5], identity=dict(other_hw[5]["identity"], hardware="y"))
        cases["two identities"] = other_hw
        for name, rs in cases.items():
            with self.subTest(name):
                self.assertIsNotNone(mech_run.validate_rows(rs, self._gates(), exploratory=False))
        with self.subTest("no gates"):
            self.assertIsNotNone(mech_run.validate_rows(rows, None, exploratory=False))
        with self.subTest("failed gates"):
            self.assertIsNotNone(mech_run.validate_rows(rows, self._gates(ok=False), exploratory=False))
        with self.subTest("gates on other hardware"):
            self.assertIsNotNone(mech_run.validate_rows(rows, self._gates(hardware="gpu"), exploratory=False))
