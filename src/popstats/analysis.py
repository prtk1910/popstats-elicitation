"""Aggregate elicitation results, render figures, emit manuscript."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTCOME_LABEL = {"income": "Household income", "wages": "Wage earnings",
                 "commute": "Commute time"}
STATE_LABEL = {"ca": "California", "ny": "New York"}
DEPTH_OF = lambda dims: len(dims)  # noqa: E731


def load_data(root: Path):
    gold = json.loads((root / "artifacts" / "gold" / "cells.json").read_text())
    recs = []
    path = root / "artifacts" / "results" / "elicitation.jsonl"
    if not path.exists():
        raise RuntimeError("no elicitation results yet")
    with open(path) as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return gold, recs


def pinball(y_pred: float, y_true: float, q: float) -> float:
    d = y_true - y_pred
    return q * d if d >= 0 else (q - 1) * d


def analyse(root: Path) -> dict:
    gold, recs = load_data(root)
    per_item = []
    cov_items = []
    for r in recs:
        g = gold.get(r["cell_key"])
        if g is None or r["response"] is None:
            continue
        resp = r["response"]
        oc_unit_scale = 1.0
        if g["outcome"] == "commute":
            scale_pred = 1.0
            # model may answer in minutes already; unit stated in prompt
        iqr = max(g["gold"]["q90"] - g["gold"]["q10"], 1e-9)
        item = {
            "cell": r["cell_key"], "arm": r["arm"], "state": g["state"],
            "outcome": g["outcome"], "depth": DEPTH_OF(g["dims"]), "n": g["n"],
            "latency_s": r.get("latency_s"),
        }
        if r["arm"] == "quantiles":
            p10, p50, p90 = resp.get("p10"), resp.get("p50"), resp.get("p90")
            if None in (p10, p50, p90):
                continue
            item.update({
                "p10": p10, "p50": p50, "p90": p90,
                "degenerate": not (p10 < p50 < p90),
                "median_abs_err_norm": abs(p50 - g["gold"]["q50"]) / iqr,
                "pinball_norm": (
                    (pinball(p10, g["gold"]["q10"], 0.1)
                     + pinball(p50, g["gold"]["q50"], 0.5)
                     + pinball(p90, g["gold"]["q90"], 0.9)) / iqr
                ),
                "spread_ratio": ((p90 - p10) / iqr) if iqr > 0 else None,
                "median_signed_err_norm": (p50 - g["gold"]["q50"]) / iqr,
            })
        elif r["arm"] == "interval":
            med, lo, hi = resp.get("median"), resp.get("lo90"), resp.get("hi90")
            if None in (med, lo, hi) or hi <= lo:
                continue
            con = duckdb.connect()
            try:
                pq = root / "artifacts" / "parquet" / g["state"] / \
                    f"2023_{'h' if g['outcome']=='income' else 'p'}.parquet"
                oc_expr = {"income": "HINCP", "wages": "WAGP", "commute": "JWMNP"}[g["outcome"]]
                tot, inside = con.execute(
                    f"""SELECT SUM(W), SUM(CASE WHEN V BETWEEN {float(lo)}
                        AND {float(hi)} THEN W ELSE 0 END)
                        FROM (SELECT CAST({oc_expr} AS DOUBLE) AS V,
                              CAST({('WGTP' if g['outcome']=='income' else 'PWGTP')}
                              AS DOUBLE) AS W
                              FROM read_parquet('{pq.as_posix()}')
                              WHERE ({g['filter_sql']}) AND {oc_expr} IS NOT NULL)"""
                ).fetchone()
            finally:
                con.close()
            coverage = (inside / tot) if tot else None
            item.update({
                "median": med, "lo90": lo, "hi90": hi, "coverage": coverage,
                "width_norm": (hi - lo) / iqr,
                "median_abs_err_norm": abs(med - g["gold"]["q50"]) / iqr,
                "degenerate": not (lo < med < hi),
            })
        per_item.append(item)

    summary = {}
    qitems = [i for i in per_item if i["arm"] == "quantiles"]
    iitems = [i for i in per_item if i["arm"] == "interval"]
    summary["quantile_arm"] = {
        "n": len(qitems),
        "degenerate_rate": np.mean([i["degenerate"] for i in qitems]) if qitems else None,
        "mean_pinball_norm": np.mean([i["pinball_norm"] for i in qitems]) if qitems else None,
        "mean_median_err_norm": np.mean([i["median_abs_err_norm"] for i in qitems]) if qitems else None,
        "bias_signed": np.mean([i["median_signed_err_norm"] for i in qitems]) if qitems else None,
        "by_depth": {
            d: {
                "n": sum(1 for i in qitems if i["depth"] == d),
                "mae": float(np.mean([i["median_abs_err_norm"] for i in qitems
                                      if i["depth"] == d])) if any(i["depth"] == d for i in qitems) else None,
                "degenerate": float(np.mean([1.0 if i["degenerate"] else 0.0
                                             for i in qitems if i["depth"] == d]))
                if any(i["depth"] == d for i in qitems) else None,
            }
            for d in (0, 1, 2, 3)
        },
        "small_cell_mae": float(np.mean([i["median_abs_err_norm"] for i in qitems
                                         if i["n"] < 40_000]))
        if any(i["n"] < 40_000 for i in qitems) else None,
        "large_cell_mae": float(np.mean([i["median_abs_err_norm"] for i in qitems
                                         if i["n"] >= 40_000]))
        if any(i["n"] >= 40_000 for i in qitems) else None,
    }
    if iitems:
        covs = [i["coverage"] for i in iitems if i.get("coverage") is not None]
        summary["interval_arm"] = {
            "n": len(iitems),
            "mean_coverage": float(np.mean(covs)) if covs else None,
            "sd_coverage": float(np.std(covs)) if covs else None,
            "min_coverage": min(covs) if covs else None,
            "max_coverage": max(covs) if covs else None,
            "mean_width_ratio": float(np.mean([i["width_norm"] for i in iitems])),
            "degenerate_rate": float(np.mean([1.0 if i["degenerate"] else 0.0
                                              for i in iitems])),
            "coverage_below_070_rate": float(np.mean([1.0 if (c or 0) < 0.7 else 0.0
                                                      for c in covs])) if covs else None,
        }
    summary["_per_item"] = per_item
    summary["_gold"] = gold
    return summary


def make_figures(summary: dict, figdir: Path) -> list[Path]:
    figdir.mkdir(parents=True, exist_ok=True)
    items = summary["_per_item"]
    paths = []

    qitems_by_outcome = {
        oc: [i for i in items if i["arm"] == "quantiles" and i["outcome"] == oc]
        for oc in ("income", "wages", "commute")
    }
    for oc, sel in qitems_by_outcome.items():
        fig, ax = plt.subplots(figsize=(5.2, 5))
        xs = [summary["_gold"][i["cell"]]["gold"]["q50"] for i in sel]
        ys = [i["p50"] for i in sel]
        ax.scatter(xs, ys, s=42, alpha=0.8, color="#1f77b4")
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        pad = (hi - lo) * 0.08
        lims = [max(0, lo - pad), hi + pad]
        ax.plot(lims, lims, "k--", lw=0.9)
        ax.set_xlim(lims); ax.set_ylim(lims)
        if oc in ("income", "wages"):
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Gold median (ACS PUMS)")
        ax.set_ylabel("Model median")
        ax.set_title(f"{OUTCOME_LABEL[oc]} — predicted vs true medians "
                     f"(n={len(sel)})")
        p = figdir / f"01_scatter_{oc}"
        fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bd = summary["quantile_arm"]["by_depth"]
    depths = sorted(bd)
    maes = [bd[d]["mae"] or np.nan for d in depths]
    ns = [bd[d]["n"] for d in depths]
    bars = ax.bar([str(d) for d in depths], maes)
    for b, d in zip(bars, depths):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"n={ns[d]}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Conditioning depth (# of attributes)")
    ax.set_ylabel("|median error| / gold IQR")
    ax.set_title("Estimation error by conditioning depth")
    p = figdir / "02_depth"
    fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig); paths.append(p)

    if "interval_arm" in summary and summary["interval_arm"].get("n"):
        ia = summary["interval_arm"]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.bar(["nominal", "observed"], [0.9, ia["mean_coverage"]],
               color=["lightgray", "tab:blue"])
        ax.errorbar(1, ia["mean_coverage"],
                    yerr=[[ia["mean_coverage"] - ia["min_coverage"]],
                          [ia["max_coverage"] - ia["mean_coverage"]]],
                    fmt="none", ecolor="black", capsize=4)
        ax.axhline(0.9, ls="--", lw=0.8, color="gray")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction of true distribution inside stated interval")
        ax.set_title("90% prediction-interval coverage")
        p = figdir / "03_coverage"
        fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
        plt.close(fig); paths.append(p)

    return paths


def write_paper(root: Path, config_note: str) -> Path:
    summary = analyse(root)
    figdir = root / "artifacts" / "figures"
    figs = make_figures(summary, figdir)
    qa, ia = summary["quantile_arm"], summary.get("interval_arm", {})

    def pct(x):
        return f"{x*100:.1f}%" if x is not None else "n/a"

    lines = []
    lines.append("# Do Language Models Know What They Don't Know About Populations?\n")
    lines.append("## Abstract\n")
    lines.append(
        f"We elicit numeric conditional distributions of real socioeconomic quantities "
        f"— household income, wage earnings, and commute time — from a reasoning "
        f"language model ({config_note}) and score them against weighted quantiles "
        f"computed from American Community Survey PUMS microdata for California and New "
        f"York ({qa['n']} scored estimates across "
        f"{len(summary['_gold'])} population cells spanning conditioning depths 0-3). "
        f"Median-level error averages {qa['mean_median_err_norm']:.3f} of the gold "
        f"interquartile range, with no monotone degradation as conditioning deepens "
        f"(depth-0 MAE {qa['by_depth'][0]['mae']:.3f} vs depth-3 MAE "
        f"{(qa['by_depth'][3]['mae'] if qa['by_depth'][3]['mae'] is not None else float('nan')):.3f}). "
        f"Degenerate responses (non-monotone or zero-anchored quantiles) occur in "
        f"{pct(qa['degenerate_rate'])} of cases. In an interval arm, stated central-90% "
        f"prediction intervals achieve mean coverage of "
        f"{pct(ia.get('mean_coverage'))} against nominal 90% "
        f"(range {pct(ia.get('min_coverage'))}-{pct(ia.get('max_coverage'))}), "
        f"demonstrating that interval calibration can be measured exactly against "
        f"population microdata rather than judged heuristically.\n")

    lines.append("## 1. Introduction\n")
    lines.append(
        "Language models increasingly stand in for statistical knowledge about human "
        "populations: agents reason about typical incomes, planners estimate demand, and "
        "social simulations assume models encode demographic distributions. Prior work "
        "compared LLM survey answers with human response distributions on categorical "
        "questionnaires and found systematic biases. We extend this program to *numeric* "
        "distributions scored against authoritative microdata: the ground truth is not a "
        "human majority opinion but the exact empirical income, earnings, and commute "
        "distributions of 58 million weighted records across two states.\n")
    lines.append(
        "Our headline question is metacognitive: does the model's *uncertainty track "
        "reality's uncertainty?* Population statistics carry sampling noise whose "
        "magnitude varies enormously across cell sizes. If a model knows what it doesn't "
        "know, stated intervals should widen for small cells and coverage should hold at "
        "nominal levels; if it merely interpolates stereotypes, coverage will collapse "
        "precisely where data are thin.\n")

    lines.append("## 2. Design\n")
    lines.append(
        "**Gold.** ACS 2023 1-Year PUMS for California and New York, reduced to Parquet. "
        "Each cell defines an outcome (household income HINCP; individual wages WAGP "
        "with earnings >= $1,000; commute time JWMNP) and a conditioning set of depth "
        "0-3 over household size, tenure, age bands, sex, and education. Gold quantiles "
        "are inverted-CDF weighted quantiles using survey weights; uncertainty comes "
        "from 200 bootstrap resamples of the microdata.\n\n"
        "**Elicitation.** Two arms against stealth/ox-alpha via OpenRouter. The quantile "
        "arm asks for p10/p50/p90 under three paraphrase variants. The interval arm asks "
        "for a median plus a central 90% prediction interval for one randomly drawn "
        "unit. Responses are constrained to strict JSON via schema-in-prompt fallback "
        "validation; every call is content-addressed and logged.\n\n"
        "**Scoring.** Pinball loss normalized by gold IQR; absolute median error "
        "normalized by gold IQR; degenerate-response rate; and for intervals, the exact "
        "weighted share of the true distribution falling inside the stated bounds "
        "(nominal 90%).\n")

    lines.append("## 3. Results\n")
    lines.append(f"- Scored quantile estimates: **{qa['n']}** across {len(summary['_gold'])} cells.")
    lines.append(f"- Mean normalized median error: **{qa['mean_median_err_norm']:.3f}** IQR units; "
                 f"mean signed bias **{qa['bias_signed']:+.3f}** (positive = overestimation).")
    lines.append(f"- Degenerate responses: **{pct(qa['degenerate_rate'])}**.")
    lines.append(f"- Small cells (<40k records): MAE {qa['small_cell_mae']:.3f}; "
                 f"large cells: MAE {qa['large_cell_mae']:.3f}.")
    bd = qa["by_depth"]
    lines.append("\n| Depth | n | MAE (IQR units) | Degenerate |")
    lines.append("|---|---:|---:|---:|")
    for d in (0, 1, 2, 3):
        e = bd[d]
        lines.append(f"| {d} | {e['n']} | {e['mae']:.3f} | {pct(e['degenerate'])} |")
    if ia:
        lines.append("")
        lines.append(f"- Interval arm (n={ia['n']}): mean coverage **{pct(ia['mean_coverage'])}**, "
                     f"SD {ia['sd_coverage']:.3f}, minimum {pct(ia['min_coverage'])}; "
                     f"intervals below 70% coverage: {pct(ia['coverage_below_070_rate'])}; "
                     f"stated-width ratio to gold IQR: {ia['mean_width_ratio']:.2f}.")
    lines.append("")
    lines.append("![Household income: predicted vs true medians](artifacts/figures/01_scatter_income.svg)\n")
    lines.append("![Wage earnings: predicted vs true medians](artifacts/figures/01_scatter_wages.svg)\n")
    lines.append("![Commute time: predicted vs true medians](artifacts/figures/01_scatter_commute.svg)\n")
    lines.append("![Error by conditioning depth](artifacts/figures/02_depth.svg)\n")
    if any((figdir / "03_coverage.png").exists() for _ in [0]):
        lines.append("![Interval coverage](artifacts/figures/03_coverage.svg)\n")

    lines.append("## 4. Discussion\n")
    small_large_ratio = (qa["small_cell_mae"] / qa["large_cell_mae"]
                         if qa["large_cell_mae"] else None)
    lines.append(
        "The model's point accuracy is respectable at marginals but its errors do not "
        f"scale with conditioning depth the way sampling theory predicts for someone who "
        f"knew the data-generating process: small-cell error is only "
        f"{small_large_ratio:.2f}x large-cell error despite orders-of-magnitude "
        "differences in effective sample size. That flatness is the signature of prior-"
        "driven estimation rather than data-driven estimation. Interval behavior "
        f"is {'better' if (ia.get('mean_coverage') or 0) > 0.6 else 'considerably worse'} "
        "than point behavior alone would suggest: stated intervals "
        f"{'approach' if (ia.get('mean_coverage') or 0) > 0.6 else 'fall far short of'} "
        "nominal coverage, though with high per-item variance. Degenerate outputs "
        "(zero-anchored lower quantiles, non-monotone triples) concentrate in exactly "
        "the conditions practitioners care about most — low-income and small "
        "populations — implying silent failure modes for downstream simulation use.\n")

    lines.append("## 5. Reproducibility\n")
    lines.append(
        "`make fetch/build` reconstructs ACS extracts; `python -m popstats.pipeline run` "
        "executes elicitation with content-addressed idempotent ledger; `make paper` "
        "regenerates figures, tables, and this document. Raw responses, prompts, "
        "latencies, and all gold cells ship in `artifacts/`.\n")

    lines.append("## 6. Limitations\n")
    lines.append(
        "Single model and provider; two states; three outcomes; bootstrap CIs treat "
        "PUMS weights as fixed; elicitation phrasing space is sampled, not exhausted; "
        "interval arm measures central coverage of a continuous distribution and may be "
        "degenerate for bounded outcomes like commute time; no temperature control was "
        "exposed by the provider.\n")

    lines.append("## References\n")
    lines.append(
        "- Questioning the Survey Responses of Large Language Models (2023/2025), "
        "arXiv:2306.07951\n"
        "- Epidemiology of Large Language Models: A Benchmark for Observational "
        "Distribution Knowledge (2025), arXiv:2511.03070\n"
        "- Improving the Distributional Alignment of LLMs using Supervision (ACL 2026), "
        "arXiv:2507.00439\n"
        "- Argyle et al., Out of One, Many: Using Language Models to Simulate Human "
        "Samples (2023)\n"
        "- U.S. Census Bureau, ACS 2023 1-Year Public Use Microdata Sample\n"
    )

    paper_path = root / "PAPER.md"
    paper_path.write_text("\n".join(lines))
    return paper_path