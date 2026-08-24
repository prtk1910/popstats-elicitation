"""Weighted statistics and bootstrap gold computation (numpy-accelerated)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import duckdb
import numpy as np


def weighted_quantile(values, weights, q: float) -> float:
    """Inverted-CDF weighted quantile."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        raise ValueError("empty values")
    if v.size != w.size:
        raise ValueError("length mismatch")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cdf = np.cumsum(w)
    total = cdf[-1]
    if total <= 0:
        raise ValueError("non-positive total weight")
    k = int(np.searchsorted(cdf, q * total, side="left"))
    return float(v[min(k, v.size - 1)])


@dataclass
class GoldQuantiles:
    q10: float
    q50: float
    q90: float
    n: int
    ci: dict


def _boot_quantile(v_sorted_sample, target):
    cdf = np.cumsum(np.ones_like(v_sorted_sample))
    k = int(np.searchsorted(cdf, target, side="left"))
    return float(v_sorted_sample[min(k, v_sorted_sample.size - 1)])


def compute_gold(
    parquet_path: str,
    value_expr: str,
    weight_col: str,
    filter_sql: str = "1=1",
    quantiles=(0.1, 0.5, 0.9),
    n_bootstrap: int = 200,
    seed: int = 13,
) -> GoldQuantiles:
    con = duckdb.connect()
    try:
        arrs = con.execute(
            f"""SELECT CAST({value_expr} AS DOUBLE), CAST({weight_col} AS DOUBLE)
                FROM read_parquet('{parquet_path}')
                WHERE ({filter_sql})
                  AND {value_expr} IS NOT NULL AND {weight_col} > 0"""
        ).fetchall()
    finally:
        con.close()
    vals = np.array([r[0] for r in arrs], dtype=float)
    wts = np.array([r[1] for r in arrs], dtype=float)
    qs = {q: weighted_quantile(vals, wts, q) for q in quantiles}

    rng = np.random.default_rng(seed)
    n = vals.size
    names = {0.1: "q10", 0.5: "q50", 0.9: "q90"}
    ci = {}
    total_w = float(wts.sum())
    for q in quantiles:
        stats = np.empty(n_bootstrap)
        done = 0
        chunk = max(1, min(n_bootstrap, int(4e6 // max(1, n))))
        while done < n_bootstrap:
            m = min(chunk, n_bootstrap - done)
            draws = rng.integers(0, n, size=(m, n))
            samples = vals[draws]
            sw = wts[draws]
            order = np.argsort(samples, axis=1, kind="stable")
            s_sorted = np.take_along_axis(samples, order, axis=1)
            cdf = np.cumsum(sw[np.arange(m)[:, None], order], axis=1)
            ks = (cdf >= (q * total_w)).argmax(axis=1)
            stats[done:done + m] = s_sorted[np.arange(m), ks]
            done += m
        ci[names[q]] = (float(np.quantile(stats, 0.025)),
                        float(np.quantile(stats, 0.975)))

    return GoldQuantiles(q10=qs[0.1], q50=qs[0.5], q90=qs[0.9], n=int(n),
                         ci={names[q]: ci[names[q]] for q in quantiles})


__all__ = ["weighted_quantile", "compute_gold", "GoldQuantiles"]
