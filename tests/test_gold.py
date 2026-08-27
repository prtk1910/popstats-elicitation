"""Weighted-quantile and ACS replicate-weight uncertainty tests."""

from __future__ import annotations

import unittest

from popstats.gold import (
    REPLICATE_COUNT,
    normal_ci,
    replicate_weight_names,
    sdr_standard_error,
    weighted_quantile,
)


class TestWeightedQuantile(unittest.TestCase):
    def test_matches_unweighted_median_when_uniform(self):
        vals = [3.0, 1.0, 2.0, 5.0, 4.0]
        w = [1.0] * 5
        self.assertEqual(weighted_quantile(vals, w, 0.5), 3.0)

    def test_weight_dominance(self):
        vals = [1.0, 2.0, 3.0]
        w = [1.0, 100.0, 1.0]
        self.assertEqual(weighted_quantile(vals, w, 0.5), 2.0)
        self.assertEqual(weighted_quantile(vals, w, 0.005), 1.0)
        self.assertEqual(weighted_quantile(vals, w, 0.999), 3.0)

    def test_inverted_cdf_boundary(self):
        vals = [10.0, 20.0]
        w = [1.0, 1.0]
        self.assertEqual(weighted_quantile(vals, w, 0.5), 10.0)
        self.assertEqual(weighted_quantile(vals, w, 0.51), 20.0)

    def test_expansion_equivalence(self):
        vals = [5.0, 7.0, 9.0]
        w = [3.0, 1.0, 2.0]

        expanded = [
            value
            for value, weight in zip(vals, w)
            for _ in range(int(weight))
        ]
        expanded.sort()

        n = len(expanded)
        expected_median = (
            expanded[(n - 1) // 2]
            if n % 2
            else expanded[n // 2 - 1]
        )

        self.assertLessEqual(
            weighted_quantile(vals, w, 0.49),
            expected_median,
        )
        self.assertGreaterEqual(
            weighted_quantile(vals, w, 0.51),
            expected_median,
        )

    def test_errors(self):
        with self.assertRaises(ValueError):
            weighted_quantile([], [], 0.5)

        with self.assertRaises(ValueError):
            weighted_quantile([1.0], [0.0], 0.5)

        with self.assertRaises(ValueError):
            weighted_quantile([1.0], [-1.0], 0.5)

        with self.assertRaises(ValueError):
            weighted_quantile([1.0], [1.0], 1.1)


class TestACSReplicateWeights(unittest.TestCase):
    def test_household_replicate_names(self):
        names = replicate_weight_names("WGTP")

        self.assertEqual(len(names), REPLICATE_COUNT)
        self.assertEqual(names[0], "WGTP1")
        self.assertEqual(names[-1], "WGTP80")

    def test_person_replicate_names(self):
        names = replicate_weight_names("PWGTP")

        self.assertEqual(len(names), REPLICATE_COUNT)
        self.assertEqual(names[0], "PWGTP1")
        self.assertEqual(names[-1], "PWGTP80")

    def test_unknown_weight_rejected(self):
        with self.assertRaises(ValueError):
            replicate_weight_names("WEIGHT")

    def test_sdr_formula(self):
        # 80 replicate estimates, each exactly 10 units away from
        # the full estimate.
        #
        # variance = (4/80) * (80 * 10^2) = 400
        # SE = 20
        reps = [
            90.0 if i % 2 == 0 else 110.0
            for i in range(REPLICATE_COUNT)
        ]

        se = sdr_standard_error(100.0, reps)
        self.assertAlmostEqual(se, 20.0)

    def test_sdr_requires_all_80_replicates(self):
        with self.assertRaises(ValueError):
            sdr_standard_error(100.0, [90.0, 110.0])

    def test_95_percent_normal_ci(self):
        lo, hi = normal_ci(
            estimate=100.0,
            standard_error=10.0,
            confidence=0.95,
        )

        self.assertAlmostEqual(lo, 80.40036015459946)
        self.assertAlmostEqual(hi, 119.59963984540054)

    def test_90_percent_normal_ci(self):
        lo, hi = normal_ci(
            estimate=100.0,
            standard_error=10.0,
            confidence=0.90,
        )

        self.assertAlmostEqual(lo, 83.55146373048528)
        self.assertAlmostEqual(hi, 116.44853626951472)


if __name__ == "__main__":
    unittest.main()
