"""Weighted-quantile math and anchor checks."""

from __future__ import annotations

import unittest

from popstats.gold import weighted_quantile


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
        expanded = [v for v, wt in zip(vals, w) for _ in range(int(wt))]
        expanded.sort()
        n = len(expanded)
        expected_median = expanded[(n - 1) // 2] if n % 2 else expanded[n // 2 - 1]
        self.assertLessEqual(weighted_quantile(vals, w, 0.49), expected_median)
        self.assertGreaterEqual(weighted_quantile(vals, w, 0.51), expected_median)

    def test_errors(self):
        with self.assertRaises(ValueError):
            weighted_quantile([], [], 0.5)
        with self.assertRaises(ValueError):
            weighted_quantile([1.0], [0.0], 0.5)


if __name__ == "__main__":
    unittest.main()
