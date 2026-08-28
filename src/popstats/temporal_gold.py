"""Cross-year ACS gold for the temporal population-knowledge experiment.

This is intentionally separate from the frozen historical 2023 benchmark.
Monetary PUMS outcomes are adjusted by ADJINC within each ACS release and
also expressed in constant 2023 dollars for cross-year comparisons.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

YEARS = (2019, 2021, 2023)

CPI_U = {
    2019: 255.657,
    2021: 270.970,
    2023: 304.702,
}

CPI_2023 = CPI_U[2023]

Z90 = 1.6448536269514722
Z95 = 1.959963984540054


def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights >= 0)
    )

    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        raise ValueError("no finite observations")

    total = float(weights.sum())

    if total <= 0:
        raise ValueError("non-positive total weight")

    order = np.argsort(
        values,
        kind="mergesort",
    )

    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    cutoff = q * total

    idx = int(
        np.searchsorted(
            cumulative,
            cutoff,
            side="left",
        )
    )

    idx = min(
        idx,
        len(values) - 1,
    )

    return float(values[idx])


def quantiles(values, weights):
    return {
        "q10": weighted_quantile(
            values,
            weights,
            0.10,
        ),
        "q50": weighted_quantile(
            values,
            weights,
            0.50,
        ),
        "q90": weighted_quantile(
            values,
            weights,
            0.90,
        ),
    }


def sdr_se(full, replicate_values):
    if len(replicate_values) != 80:
        raise ValueError(
            "ACS SDR requires exactly 80 replicate estimates"
        )

    diffs = np.asarray(
        replicate_values,
        dtype=float,
    ) - float(full)

    return float(
        math.sqrt(
            (4.0 / 80.0)
            * float(np.sum(diffs * diffs))
        )
    )


def scale_dict(values, factor):
    return {
        k: float(v) * factor
        for k, v in values.items()
    }


def uncertainty(
    full_quantiles,
    replicate_quantiles,
):
    se = {}

    for q in (
        "q10",
        "q50",
        "q90",
    ):
        se[q] = sdr_se(
            full_quantiles[q],
            [
                r[q]
                for r in replicate_quantiles
            ],
        )

    moe90 = {
        q: Z90 * se[q]
        for q in se
    }

    ci95 = {
        q: [
            full_quantiles[q]
            - Z95 * se[q],
            full_quantiles[q]
            + Z95 * se[q],
        ]
        for q in se
    }

    return se, moe90, ci95


def outcome_config(outcome):
    if outcome == "income":
        return {
            "file": "h",
            "value_col": "HINCP",
            "weight": "WGTP",
            "rep_prefix": "WGTP",
            "money": True,
        }

    if outcome == "wages":
        return {
            "file": "p",
            "value_col": "WAGP",
            "weight": "PWGTP",
            "rep_prefix": "PWGTP",
            "money": True,
        }

    if outcome == "commute":
        return {
            "file": "p",
            "value_col": "JWMNP",
            "weight": "PWGTP",
            "rep_prefix": "PWGTP",
            "money": False,
        }

    raise ValueError(
        f"unknown outcome: {outcome}"
    )


def compute_cell(
    *,
    year,
    cell_key,
    cell,
):
    cfg = outcome_config(
        cell["outcome"]
    )

    parquet = (
        ROOT
        / "artifacts"
        / "parquet"
        / cell["state"]
        / f"{year}_{cfg['file']}.parquet"
    )

    if not parquet.exists():
        raise FileNotFoundError(
            parquet
        )

    reps = [
        f"{cfg['rep_prefix']}{i}"
        for i in range(1, 81)
    ]

    if cfg["money"]:
        value_expr = (
            f"CAST({cfg['value_col']} AS DOUBLE) "
            "* CAST(ADJINC AS DOUBLE) "
            "/ 1000000.0"
        )

        extra_filter = ""

    else:
        value_expr = (
            f"CAST({cfg['value_col']} AS DOUBLE)"
        )

        # Explicit temporal commute universe.
        # This also excludes any future suppression/sentinel values.
        extra_filter = (
            " AND JWMNP BETWEEN 1 AND 200"
        )

    cols = [
        f"{value_expr} AS V",
        f"CAST({cfg['weight']} AS DOUBLE) AS W",
    ]

    cols.extend(
        f"CAST({r} AS DOUBLE) AS {r}"
        for r in reps
    )

    query = f"""
        SELECT
            {", ".join(cols)}
        FROM read_parquet('{parquet.as_posix()}')
        WHERE ({cell["filter_sql"]})
          AND ({value_expr}) IS NOT NULL
          AND CAST({cfg["weight"]} AS DOUBLE) > 0
          {extra_filter}
    """

    con = duckdb.connect()

    try:
        data = con.execute(
            query
        ).fetchnumpy()

        if cfg["money"]:
            adj_range = con.execute(
                f"""
                SELECT
                    MIN(CAST(ADJINC AS DOUBLE)),
                    MAX(CAST(ADJINC AS DOUBLE))
                FROM read_parquet('{parquet.as_posix()}')
                """
            ).fetchone()
        else:
            adj_range = None

    finally:
        con.close()

    if len(data["V"]) == 0:
        raise RuntimeError(
            f"{year} {cell_key}: empty population"
        )

    values = np.asarray(data["V"], dtype=float)
    full_weights = np.asarray(data["W"], dtype=float)

    full = quantiles(
        values,
        full_weights,
    )

    replicate_quantiles = []

    for rep in reps:
        replicate_quantiles.append(
            quantiles(
                values,
                np.asarray(data[rep], dtype=float),
            )
        )

    se, moe90, ci95 = uncertainty(
        full,
        replicate_quantiles,
    )

    if cfg["money"]:
        real_factor = (
            CPI_2023
            / CPI_U[year]
        )

        gold_2023 = scale_dict(
            full,
            real_factor,
        )

        se_2023 = scale_dict(
            se,
            real_factor,
        )

        moe90_2023 = scale_dict(
            moe90,
            real_factor,
        )

        ci95_2023 = {
            q: [
                bounds[0]
                * real_factor,
                bounds[1]
                * real_factor,
            ]
            for q, bounds
            in ci95.items()
        }

        adjinc_factor = (
            float(adj_range[0])
            / 1_000_000.0
        )

    else:
        real_factor = 1.0
        gold_2023 = dict(full)
        se_2023 = dict(se)
        moe90_2023 = dict(moe90)
        ci95_2023 = dict(ci95)
        adjinc_factor = None

    return {
        "cell_key": cell_key,
        "year": year,
        "state": cell["state"],
        "outcome": cell["outcome"],
        "dims": cell["dims"],
        "population_nl": cell[
            "population_nl"
        ],
        "filter_sql": cell[
            "filter_sql"
        ],
        "n_unweighted": int(
            len(data["V"])
        ),
        "gold_year_dollars": full,
        "se_year_dollars": se,
        "moe90_year_dollars": moe90,
        "ci95_year_dollars": ci95,
        "gold_2023_dollars": gold_2023,
        "se_2023_dollars": se_2023,
        "moe90_2023_dollars":
            moe90_2023,
        "ci95_2023_dollars":
            ci95_2023,
        "adjinc_factor":
            adjinc_factor,
        "cpi_u_annual_average":
            CPI_U[year],
        "cpi_to_2023_factor":
            real_factor,
        "uncertainty_method":
            "acs_sdr_replicate_weights",
        "temporal_money_method": (
            "PUMS value * ADJINC / 1e6; "
            "then CPI-U annual-average ratio to 2023 dollars"
            if cfg["money"]
            else None
        ),
        "commute_valid_range_minutes": (
            [1, 200]
            if not cfg["money"]
            else None
        ),
    }


def build_year(
    year,
):
    if year not in YEARS:
        raise ValueError(
            f"unsupported year: {year}"
        )

    frozen = json.loads(
        (
            ROOT
            / "artifacts"
            / "gold"
            / "cells.json"
        ).read_text()
    )

    output = {}

    for i, (
        cell_key,
        cell,
    ) in enumerate(
        sorted(
            frozen.items()
        ),
        start=1,
    ):
        result = compute_cell(
            year=year,
            cell_key=cell_key,
            cell=cell,
        )

        output[cell_key] = result

        print(
            f"[{i:02d}/40] "
            f"{year} {cell_key} "
            f"q50="
            f"{result['gold_year_dollars']['q50']:.2f}",
            flush=True,
        )

    out_dir = (
        ROOT
        / "artifacts"
        / "gold"
        / "temporal"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out_dir
        / f"cells-{year}.json"
    )

    path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print(
        "temporal gold written:",
        path,
    )

    return path


def main():
    for year in YEARS:
        build_year(
            year
        )


if __name__ == "__main__":
    main()
