"""Anchor tests: weighting pipeline must reproduce known population totals.

These guard against column mix-ups and weighting bugs before any gold is generated.
Runs only when extracts are present.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
PQ = ROOT / "artifacts" / "parquet"

ANCHORS = {
    # state: (pop_lo, pop_hi, units_lo, units_hi)
    "ca": (37e6, 41e6, 13e6, 16e6),
    "ny": (18e6, 21e6, 7e6, 10e6),
}


@unittest.skipUnless(PQ.exists(), "parquet extracts not built yet")
class TestPopulationAnchors(unittest.TestCase):
    def test_weight_totals_match_published_population(self):
        con = duckdb.connect()
        try:
            for st, (plo, phi, ulo, uhi) in ANCHORS.items():
                p = PQ / st / "2023_p.parquet"
                h = PQ / st / "2023_h.parquet"
                if not p.exists() or not h.exists():
                    self.skipTest(f"{st} extracts missing")
                pop = con.execute(
                    f"SELECT SUM(PWGTP) FROM read_parquet('{p.as_posix()}')"
                ).fetchone()[0]
                units = con.execute(
                    f"SELECT SUM(WGTP) FROM read_parquet('{h.as_posix()}')"
                ).fetchone()[0]
                self.assertTrue(plo <= pop <= phi,
                                f"{st} population {pop} outside [{plo}, {phi}]")
                self.assertTrue(ulo <= units <= uhi,
                                f"{st} housing units {units} outside [{ulo}, {uhi}]")
        finally:
            con.close()

    def test_income_quantiles_are_sane(self):
        con = duckdb.connect()
        try:
            h = PQ / "ca" / "2023_h.parquet"
            if not h.exists():
                self.skipTest("ca housing extract missing")
            lo, hi = con.execute(
                f"""SELECT quantile_cont(HINCP, 0.1), quantile_cont(HINCP, 0.9)
                    FROM read_parquet('{h.as_posix()}')
                    WHERE HINCP IS NOT NULL AND HINCP >= 0"""
            ).fetchone()
            self.assertGreater(hi, lo)
            self.assertGreater(lo, 0)
            self.assertLess(lo, 60_000)
            self.assertGreater(hi, 150_000)
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
