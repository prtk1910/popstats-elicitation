"""Weighted population statistics using official ACS replicate weights."""

from __future__ import annotations

from dataclasses import dataclass
import math

import duckdb
import numpy as np


REPLICATE_COUNT = 80
SDR_FACTOR = 4.0 / REPLICATE_COUNT
Z_95 = 1.959963984540054
Z_90 = 1.6448536269514722


def weighted_quantile(values, weights, q: float) -> float:
    """Inverted-CDF weighted quantile."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")

    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    if v.size == 0:
        raise ValueError("empty values")
    if v.size != w.size:
        raise ValueError("length mismatch")
    if np.any(w < 0):
        raise ValueError("negative weight")

    order = np.argsort(v, kind="stable")
    v = v[order]
    w = w[order]

    cdf = np.cumsum(w)
    total = float(cdf[-1])

    if total <= 0:
        raise ValueError("non-positive total weight")

    k = int(np.searchsorted(cdf, q * total, side="left"))
    return float(v[min(k, v.size - 1)])


def replicate_weight_names(weight_col: str) -> list[str]:
    """Return the 80 ACS replicate-weight columns for a full-sample weight."""
    if weight_col == "WGTP":
        prefix = "WGTP"
    elif weight_col == "PWGTP":
        prefix = "PWGTP"
    else:
        raise ValueError(
            f"unsupported ACS weight column {weight_col!r}; "
            "expected WGTP or PWGTP"
        )

    return [f"{prefix}{i}" for i in range(1, REPLICATE_COUNT + 1)]


def sdr_standard_error(
    full_estimate: float,
    replicate_estimates,
) -> float:
    """ACS Successive Difference Replication standard error.

    ACS PUMS supplies 80 replicate weights. For any statistic theta:

        Var(theta) = (4 / 80) * sum((theta_r - theta)^2)

    where theta_r is the statistic recomputed with replicate weight r.
    """
    reps = np.asarray(replicate_estimates, dtype=float)

    if reps.size != REPLICATE_COUNT:
        raise ValueError(
            f"expected {REPLICATE_COUNT} replicate estimates, got {reps.size}"
        )

    variance = SDR_FACTOR * np.sum((reps - float(full_estimate)) ** 2)
    return float(math.sqrt(variance))


def normal_ci(
    estimate: float,
    standard_error: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Normal-approximation confidence interval around a survey estimate."""
    if confidence == 0.95:
        z = Z_95
    elif confidence == 0.90:
        z = Z_90
    else:
        raise ValueError("supported confidence levels are 0.90 and 0.95")

    margin = z * standard_error
    return float(estimate - margin), float(estimate + margin)


@dataclass
class GoldQuantiles:
    q10: float
    q50: float
    q90: float
    n: int
    ci: dict
    se: dict
    moe90: dict
    uncertainty_method: str = "acs_sdr_replicate_weights"


def compute_gold(
    parquet_path: str,
    value_expr: str,
    weight_col: str,
    filter_sql: str = "1=1",
    quantiles=(0.1, 0.5, 0.9),
) -> GoldQuantiles:
    """Compute weighted quantiles and ACS replicate-weight uncertainty.

    Point estimates use WGTP/PWGTP. Standard errors use the corresponding
    80 ACS PUMS replicate weights and the official SDR variance formula.
    """
    replicate_cols = replicate_weight_names(weight_col)

    select_cols = [
        f"CAST({value_expr} AS DOUBLE)",
        f"CAST({weight_col} AS DOUBLE)",
        *[f'CAST("{col}" AS DOUBLE)' for col in replicate_cols],
    ]

    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""
            SELECT {", ".join(select_cols)}
            FROM read_parquet('{parquet_path}')
            WHERE ({filter_sql})
              AND {value_expr} IS NOT NULL
              AND {weight_col} > 0
            """
        ).fetchall()
    finally:
        con.close()

    if not rows:
        raise ValueError("no rows after filtering")

    arr = np.asarray(rows, dtype=float)

    vals = arr[:, 0]
    full_weights = arr[:, 1]
    replicate_weights = arr[:, 2:]

    if replicate_weights.shape[1] != REPLICATE_COUNT:
        raise ValueError(
            f"expected {REPLICATE_COUNT} ACS replicate-weight columns, "
            f"got {replicate_weights.shape[1]}"
        )

    names = {
        0.1: "q10",
        0.5: "q50",
        0.9: "q90",
    }

    point = {}
    se = {}
    ci = {}
    moe90 = {}

    for q in quantiles:
        if q not in names:
            raise ValueError(f"unsupported quantile {q}")

        name = names[q]
        estimate = weighted_quantile(vals, full_weights, q)

        replicate_estimates = np.asarray(
            [
                weighted_quantile(vals, replicate_weights[:, r], q)
                for r in range(REPLICATE_COUNT)
            ],
            dtype=float,
        )

        standard_error = sdr_standard_error(
            estimate,
            replicate_estimates,
        )

        point[name] = estimate
        se[name] = standard_error
        ci[name] = normal_ci(
            estimate,
            standard_error,
            confidence=0.95,
        )
        moe90[name] = Z_90 * standard_error

    return GoldQuantiles(
        q10=point["q10"],
        q50=point["q50"],
        q90=point["q90"],
        n=int(vals.size),
        ci=ci,
        se=se,
        moe90=moe90,
    )


__all__ = [
    "GoldQuantiles",
    "REPLICATE_COUNT",
    "compute_gold",
    "normal_ci",
    "replicate_weight_names",
    "sdr_standard_error",
    "weighted_quantile",
]
