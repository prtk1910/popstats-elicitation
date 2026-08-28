"""Tests for weighted quantiles and ACS SDR replicate-weight uncertainty."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb

from popstats.gold import (
    REPLICATE_COUNT,
    compute_gold,
    normal_ci,
    replicate_weight_names,
    sdr_standard_error,
    weighted_quantile,
)


class TestWeightedQuantile(unittest.TestCase):
    def test_matches_unweighted_median_when_uniform(self):
        vals = [3.0, 1.0, 2.0, 5.0, 4.0]
        weights = [1.0] * 5

        self.assertEqual(
            weighted_quantile(vals, weights, 0.5),
            3.0,
        )

    def test_weight_dominance(self):
        vals = [1.0, 2.0, 3.0]
        weights = [1.0, 100.0, 1.0]

        self.assertEqual(
            weighted_quantile(vals, weights, 0.5),
            2.0,
        )
        self.assertEqual(
            weighted_quantile(vals, weights, 0.005),
            1.0,
        )
        self.assertEqual(
            weighted_quantile(vals, weights, 0.999),
            3.0,
        )

    def test_inverted_cdf_boundary(self):
        vals = [10.0, 20.0]
        weights = [1.0, 1.0]

        self.assertEqual(
            weighted_quantile(vals, weights, 0.5),
            10.0,
        )
        self.assertEqual(
            weighted_quantile(vals, weights, 0.51),
            20.0,
        )

    def test_expansion_equivalence(self):
        vals = [5.0, 7.0, 9.0]
        weights = [3.0, 1.0, 2.0]

        expanded = [
            value
            for value, weight in zip(vals, weights)
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
            weighted_quantile(vals, weights, 0.49),
            expected_median,
        )
        self.assertGreaterEqual(
            weighted_quantile(vals, weights, 0.51),
            expected_median,
        )

    def test_q_zero_returns_minimum(self):
        vals = [30.0, 10.0, 20.0]
        weights = [1.0, 1.0, 1.0]

        self.assertEqual(
            weighted_quantile(vals, weights, 0.0),
            10.0,
        )

    def test_q_one_returns_maximum(self):
        vals = [30.0, 10.0, 20.0]
        weights = [1.0, 1.0, 1.0]

        self.assertEqual(
            weighted_quantile(vals, weights, 1.0),
            30.0,
        )

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            weighted_quantile([], [], 0.5)

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            weighted_quantile(
                [1.0, 2.0],
                [1.0],
                0.5,
            )

    def test_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            weighted_quantile(
                [1.0],
                [-1.0],
                0.5,
            )

    def test_rejects_zero_total_weight(self):
        with self.assertRaises(ValueError):
            weighted_quantile(
                [1.0],
                [0.0],
                0.5,
            )

    def test_rejects_invalid_quantile(self):
        with self.assertRaises(ValueError):
            weighted_quantile(
                [1.0],
                [1.0],
                1.1,
            )


class TestACSReplicateWeights(unittest.TestCase):
    def test_household_replicate_names(self):
        names = replicate_weight_names("WGTP")

        self.assertEqual(
            len(names),
            REPLICATE_COUNT,
        )
        self.assertEqual(
            names[0],
            "WGTP1",
        )
        self.assertEqual(
            names[-1],
            "WGTP80",
        )

    def test_person_replicate_names(self):
        names = replicate_weight_names("PWGTP")

        self.assertEqual(
            len(names),
            REPLICATE_COUNT,
        )
        self.assertEqual(
            names[0],
            "PWGTP1",
        )
        self.assertEqual(
            names[-1],
            "PWGTP80",
        )

    def test_unknown_weight_rejected(self):
        with self.assertRaises(ValueError):
            replicate_weight_names("WEIGHT")

    def test_sdr_formula(self):
        # Every replicate differs from the full estimate by exactly 10.
        #
        # variance
        #   = (4 / 80) * sum((theta_r - theta)^2)
        #   = (4 / 80) * 80 * 10^2
        #   = 400
        #
        # SE = 20
        reps = [
            90.0 if i % 2 == 0 else 110.0
            for i in range(REPLICATE_COUNT)
        ]

        se = sdr_standard_error(
            100.0,
            reps,
        )

        self.assertAlmostEqual(
            se,
            20.0,
        )

    def test_sdr_zero_when_replicates_equal_full_estimate(self):
        reps = [
            100.0
            for _ in range(REPLICATE_COUNT)
        ]

        se = sdr_standard_error(
            100.0,
            reps,
        )

        self.assertEqual(
            se,
            0.0,
        )

    def test_sdr_requires_all_80_replicates(self):
        with self.assertRaises(ValueError):
            sdr_standard_error(
                100.0,
                [90.0, 110.0],
            )

    def test_95_percent_normal_ci(self):
        lo, hi = normal_ci(
            estimate=100.0,
            standard_error=10.0,
            confidence=0.95,
        )

        self.assertAlmostEqual(
            lo,
            80.40036015459946,
        )
        self.assertAlmostEqual(
            hi,
            119.59963984540054,
        )

    def test_90_percent_normal_ci(self):
        lo, hi = normal_ci(
            estimate=100.0,
            standard_error=10.0,
            confidence=0.90,
        )

        self.assertAlmostEqual(
            lo,
            83.55146373048528,
        )
        self.assertAlmostEqual(
            hi,
            116.44853626951472,
        )

    def test_rejects_unsupported_confidence_level(self):
        with self.assertRaises(ValueError):
            normal_ci(
                estimate=100.0,
                standard_error=10.0,
                confidence=0.99,
            )


class TestComputeGold(unittest.TestCase):
    def _make_household_parquet(
        self,
        directory: Path,
    ) -> Path:
        """Create a tiny ACS-like Parquet file with WGTP1-WGTP80."""
        path = directory / "tiny.parquet"

        columns = [
            "value DOUBLE",
            "WGTP DOUBLE",
        ] + [
            f"WGTP{i} DOUBLE"
            for i in range(1, REPLICATE_COUNT + 1)
        ]

        con = duckdb.connect()

        try:
            con.execute(
                f"""
                CREATE TABLE tiny (
                    {", ".join(columns)}
                )
                """
            )

            # Equal full-sample and replicate weights mean every replicate
            # produces exactly the same quantiles as the full estimate.
            rows = [
                [
                    10.0,
                    1.0,
                    *([1.0] * REPLICATE_COUNT),
                ],
                [
                    20.0,
                    1.0,
                    *([1.0] * REPLICATE_COUNT),
                ],
                [
                    30.0,
                    1.0,
                    *([1.0] * REPLICATE_COUNT),
                ],
                [
                    40.0,
                    1.0,
                    *([1.0] * REPLICATE_COUNT),
                ],
            ]

            placeholders = ", ".join(
                ["?"] * (2 + REPLICATE_COUNT)
            )

            con.executemany(
                f"""
                INSERT INTO tiny
                VALUES ({placeholders})
                """,
                rows,
            )

            con.execute(
                f"""
                COPY tiny
                TO '{path.as_posix()}'
                (
                    FORMAT PARQUET,
                    COMPRESSION ZSTD
                )
                """
            )
        finally:
            con.close()

        return path

    def test_compute_gold_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = self._make_household_parquet(
                Path(tmp)
            )

            gold = compute_gold(
                str(parquet_path),
                value_expr="value",
                weight_col="WGTP",
            )

            self.assertEqual(
                gold.n,
                4,
            )

            # Inverted-CDF definition:
            #
            # q10 threshold = 0.4 -> 10
            # q50 threshold = 2.0 -> 20
            # q90 threshold = 3.6 -> 40
            self.assertEqual(
                gold.q10,
                10.0,
            )
            self.assertEqual(
                gold.q50,
                20.0,
            )
            self.assertEqual(
                gold.q90,
                40.0,
            )

            # Replicate weights are identical to full weights, so all
            # replicate quantiles equal the point estimates.
            self.assertEqual(
                gold.se["q10"],
                0.0,
            )
            self.assertEqual(
                gold.se["q50"],
                0.0,
            )
            self.assertEqual(
                gold.se["q90"],
                0.0,
            )

            self.assertEqual(
                tuple(gold.ci["q50"]),
                (20.0, 20.0),
            )

            self.assertEqual(
                gold.moe90["q50"],
                0.0,
            )

            self.assertEqual(
                gold.uncertainty_method,
                "acs_sdr_replicate_weights",
            )

    def test_compute_gold_respects_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = self._make_household_parquet(
                Path(tmp)
            )

            gold = compute_gold(
                str(parquet_path),
                value_expr="value",
                weight_col="WGTP",
                filter_sql="value >= 30",
            )

            self.assertEqual(
                gold.n,
                2,
            )
            self.assertEqual(
                gold.q50,
                30.0,
            )

    def test_compute_gold_person_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "person.parquet"

            columns = [
                "value DOUBLE",
                "PWGTP DOUBLE",
            ] + [
                f"PWGTP{i} DOUBLE"
                for i in range(1, REPLICATE_COUNT + 1)
            ]

            con = duckdb.connect()

            try:
                con.execute(
                    f"""
                    CREATE TABLE tiny_person (
                        {", ".join(columns)}
                    )
                    """
                )

                rows = [
                    [
                        15.0,
                        1.0,
                        *([1.0] * REPLICATE_COUNT),
                    ],
                    [
                        25.0,
                        1.0,
                        *([1.0] * REPLICATE_COUNT),
                    ],
                    [
                        35.0,
                        1.0,
                        *([1.0] * REPLICATE_COUNT),
                    ],
                ]

                placeholders = ", ".join(
                    ["?"] * (2 + REPLICATE_COUNT)
                )

                con.executemany(
                    f"""
                    INSERT INTO tiny_person
                    VALUES ({placeholders})
                    """,
                    rows,
                )

                con.execute(
                    f"""
                    COPY tiny_person
                    TO '{path.as_posix()}'
                    (
                        FORMAT PARQUET,
                        COMPRESSION ZSTD
                    )
                    """
                )
            finally:
                con.close()

            gold = compute_gold(
                str(path),
                value_expr="value",
                weight_col="PWGTP",
            )

            self.assertEqual(
                gold.q50,
                25.0,
            )
            self.assertEqual(
                gold.se["q50"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
