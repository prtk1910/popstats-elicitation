from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]

FIG = ROOT / "artifacts/publication/figures"
TAB = ROOT / "artifacts/publication/tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

YEARS = (2019, 2021, 2023)
SEED = 20260827

OX = {
    "median_mae": 0.0420,
    "bias": 0.0140,
    "depth": {
        0: 0.0390,
        1: 0.0410,
        2: 0.0470,
        3: 0.0250,
    },
    "predictive_coverage": 0.855,
    "predictive_min_coverage": 0.734,
    "predictive_width_iqr": 1.19,
}


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def mean(xs):
    return float(np.mean(xs)) if xs else None


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return

    fields = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def savefig(name):
    plt.tight_layout()

    png = FIG / f"{name}.png"
    pdf = FIG / f"{name}.pdf"

    plt.savefig(png, dpi=240, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

    return png, pdf


gold = read_json(
    ROOT / "artifacts/gold/cells.json"
)

replication = read_jsonl(
    ROOT
    / "artifacts/results"
    / "muse-spark-1.2-original-replication.jsonl"
)

epi_analysis = read_json(
    ROOT
    / "artifacts/analysis"
    / "muse-epistemic-analysis.json"
)

inv_analysis = read_json(
    ROOT
    / "artifacts/analysis"
    / "muse-invariance-analysis.json"
)

temporal_analysis = read_json(
    ROOT
    / "artifacts/analysis"
    / "muse-temporal-analysis.json"
)

temporal_gold = {
    year: read_json(
        ROOT
        / "artifacts/gold/temporal"
        / f"cells-{year}.json"
    )
    for year in YEARS
}

temporal_rows = read_jsonl(
    ROOT
    / "artifacts/results"
    / "muse-spark-1.2-temporal.jsonl"
)


quantile_rows = [
    r
    for r in replication
    if (
        r.get("arm") == "quantiles"
        and not r.get("error")
    )
]

interval_rows = [
    r
    for r in replication
    if (
        r.get("arm") == "interval"
        and not r.get("error")
    )
]


quantile_scored = []

for r in quantile_rows:
    cell = r["cell_key"]
    g = gold[cell]

    q10 = float(g["gold"]["q10"])
    q50 = float(g["gold"]["q50"])
    q90 = float(g["gold"]["q90"])

    iqr = q90 - q10

    x = r["response"]

    quantile_scored.append({
        "cell": cell,
        "state": g["state"],
        "outcome": g["outcome"],
        "depth": len(g["dims"]),
        "variant": int(r["prompt_variant"]),
        "gold_p10": q10,
        "gold_p50": q50,
        "gold_p90": q90,
        "pred_p10": float(x["p10"]),
        "pred_p50": float(x["p50"]),
        "pred_p90": float(x["p90"]),
        "err_p10": abs(float(x["p10"]) - q10) / iqr,
        "err_p50": abs(float(x["p50"]) - q50) / iqr,
        "err_p90": abs(float(x["p90"]) - q90) / iqr,
        "bias_p10": (float(x["p10"]) - q10) / iqr,
        "bias_p50": (float(x["p50"]) - q50) / iqr,
        "bias_p90": (float(x["p90"]) - q90) / iqr,
    })


by_cell = defaultdict(list)

for r in quantile_scored:
    by_cell[r["cell"]].append(r)


cell_summary = []

for cell, rows in sorted(by_cell.items()):
    first = rows[0]

    cell_summary.append({
        "cell": cell,
        "state": first["state"],
        "outcome": first["outcome"],
        "depth": first["depth"],
        "p50_mae": mean([x["err_p50"] for x in rows]),
        "p50_bias": mean([x["bias_p50"] for x in rows]),
    })


overall_muse_mae = mean(
    [x["p50_mae"] for x in cell_summary]
)

overall_muse_bias = mean(
    [x["p50_bias"] for x in cell_summary]
)


# ------------------------------------------------------------------
# TABLE 1 — experiment inventory
# ------------------------------------------------------------------

inventory = [
    {
        "experiment": "Historical Ox Alpha quantiles",
        "model": "stealth/ox-alpha",
        "cells": 40,
        "requests_or_estimates": 120,
        "purpose": "Historical baseline",
    },
    {
        "experiment": "Muse original-prompt replication",
        "model": "muse-spark-1.2",
        "cells": 40,
        "requests_or_estimates": 160,
        "purpose": "Replication + predictive interval",
    },
    {
        "experiment": "Muse epistemic uncertainty",
        "model": "muse-spark-1.2",
        "cells": 40,
        "requests_or_estimates": 120,
        "purpose": "Uncertainty about population median",
    },
    {
        "experiment": "Muse elicitation invariance",
        "model": "muse-spark-1.2",
        "cells": 20,
        "requests_or_estimates": 60,
        "purpose": "Equivalent prompt transformations",
    },
    {
        "experiment": "Muse temporal knowledge",
        "model": "muse-spark-1.2",
        "cells": 40,
        "requests_or_estimates": 120,
        "purpose": "2019 / 2021 / 2023 knowledge",
    },
]

write_csv(
    TAB / "table01_experiment_inventory.csv",
    inventory,
)


# ------------------------------------------------------------------
# TABLE 2 — replication summary
# ------------------------------------------------------------------

replication_summary = [
    {
        "metric": "Normalized median MAE",
        "historical_ox": OX["median_mae"],
        "muse_spark_1_2": overall_muse_mae,
    },
    {
        "metric": "Signed median bias",
        "historical_ox": OX["bias"],
        "muse_spark_1_2": overall_muse_bias,
    },
    {
        "metric": "Degenerate quantile responses",
        "historical_ox": 0.0,
        "muse_spark_1_2": 0.0,
    },
]

write_csv(
    TAB / "table02_replication_summary.csv",
    replication_summary,
)


# ------------------------------------------------------------------
# TABLE 3 — quantile-specific errors
# ------------------------------------------------------------------

quantile_error = []

for q in ("p10", "p50", "p90"):
    quantile_error.append({
        "quantile": q,
        "mae_iqr": mean(
            [x[f"err_{q}"] for x in quantile_scored]
        ),
        "signed_bias_iqr": mean(
            [x[f"bias_{q}"] for x in quantile_scored]
        ),
    })

write_csv(
    TAB / "table03_quantile_error.csv",
    quantile_error,
)


# ------------------------------------------------------------------
# TABLE 4 — depth and outcome structure
# ------------------------------------------------------------------

structure = []

for depth in range(4):
    vals = [
        x
        for x in cell_summary
        if x["depth"] == depth
    ]

    structure.append({
        "group_type": "depth",
        "group": str(depth),
        "n_cells": len(vals),
        "p50_mae_iqr": mean(
            [x["p50_mae"] for x in vals]
        ),
    })

for outcome in ("income", "wages", "commute"):
    vals = [
        x
        for x in cell_summary
        if x["outcome"] == outcome
    ]

    structure.append({
        "group_type": "outcome",
        "group": outcome,
        "n_cells": len(vals),
        "p50_mae_iqr": mean(
            [x["p50_mae"] for x in vals]
        ),
    })

write_csv(
    TAB / "table04_error_structure.csv",
    structure,
)


# ------------------------------------------------------------------
# TABLE 5 — epistemic uncertainty
# ------------------------------------------------------------------

epi_table = [
    {
        "metric": "Nominal coverage",
        "value": 0.90,
    },
    {
        "metric": "Empirical coverage",
        "value": epi_analysis["coverage"]["mean"],
    },
    {
        "metric": "Coverage CI low",
        "value": epi_analysis["coverage"]["ci95"][0],
    },
    {
        "metric": "Coverage CI high",
        "value": epi_analysis["coverage"]["ci95"][1],
    },
    {
        "metric": "Mean normalized median error",
        "value": epi_analysis["error"]["mean"],
    },
    {
        "metric": "Mean interval width / gold IQR",
        "value": epi_analysis["width"]["mean"],
    },
    {
        "metric": "Width vs actual error Pearson",
        "value": epi_analysis["correlations"][
            "width_vs_actual_error_pearson"
        ],
    },
    {
        "metric": "Width vs actual error Spearman",
        "value": epi_analysis["correlations"][
            "width_vs_actual_error_spearman"
        ],
    },
    {
        "metric": "Width vs ACS SE Pearson",
        "value": epi_analysis["correlations"][
            "width_vs_acs_se_pearson"
        ],
    },
    {
        "metric": "Width vs ACS SE Spearman",
        "value": epi_analysis["correlations"][
            "width_vs_acs_se_spearman"
        ],
    },
]

write_csv(
    TAB / "table05_epistemic_summary.csv",
    epi_table,
)


epi_breakdown = []

for variant, vals in epi_analysis["variant_summary"].items():
    epi_breakdown.append({
        "group_type": "prompt_variant",
        "group": variant,
        "coverage": vals["coverage"],
        "median_error_iqr": vals["error"],
        "width_iqr": vals["width"],
    })

for outcome, vals in epi_analysis["outcome_summary"].items():
    epi_breakdown.append({
        "group_type": "outcome",
        "group": outcome,
        "coverage": vals["coverage"],
        "median_error_iqr": vals["error"],
        "width_iqr": vals["width"],
    })

for depth, vals in epi_analysis["depth_summary"].items():
    epi_breakdown.append({
        "group_type": "depth",
        "group": depth,
        "coverage": vals["coverage"],
        "median_error_iqr": vals["error"],
        "width_iqr": vals["width"],
    })

write_csv(
    TAB / "table06_epistemic_breakdown.csv",
    epi_breakdown,
)


# ------------------------------------------------------------------
# TABLE 7 — invariance
# ------------------------------------------------------------------

inv_rows = []

for arm, vals in inv_analysis["summary"].items():
    inv_rows.append({
        "transformation": arm,
        "shift_p10_iqr": vals["shift_p10"]["mean"],
        "shift_p50_iqr": vals["shift_p50"]["mean"],
        "shift_p90_iqr": vals["shift_p90"]["mean"],
        "baseline_p50_mae": vals["baseline_p50_mae"],
        "transformed_p50_mae": vals["transformed_p50_mae"],
        "p50_mae_change": vals["p50_mae_change"],
    })

write_csv(
    TAB / "table07_invariance_summary.csv",
    inv_rows,
)


# ------------------------------------------------------------------
# TABLE 8 — temporal within-year accuracy
# ------------------------------------------------------------------

temporal_accuracy = []

for year in YEARS:
    vals = temporal_analysis["year_summary"][str(year)]

    temporal_accuracy.append({
        "year": year,
        "p10_mae_iqr": vals["p10_mae"],
        "p50_mae_iqr": vals["p50_mae"],
        "p90_mae_iqr": vals["p90_mae"],
        "p50_bias_iqr": vals["p50_bias"],
    })

write_csv(
    TAB / "table08_temporal_accuracy.csv",
    temporal_accuracy,
)


# ------------------------------------------------------------------
# Temporal predictions in constant 2023 units
# ------------------------------------------------------------------

temporal_pred = {}

for r in temporal_rows:
    if r.get("error"):
        continue

    year = int(r["year"])
    cell = r["cell_key"]
    g = temporal_gold[year][cell]

    value = float(r["response"]["p50"])

    if g["outcome"] in {"income", "wages"}:
        value *= float(g["cpi_to_2023_factor"])

    temporal_pred[(cell, year)] = value


def temporal_group_data(start, end, outcomes):
    result = []

    for cell, g23 in temporal_gold[2023].items():
        if g23["outcome"] not in outcomes:
            continue

        q10 = float(g23["gold_2023_dollars"]["q10"])
        q90 = float(g23["gold_2023_dollars"]["q90"])
        iqr = q90 - q10

        gold_delta = (
            float(
                temporal_gold[end][cell]
                ["gold_2023_dollars"]["q50"]
            )
            -
            float(
                temporal_gold[start][cell]
                ["gold_2023_dollars"]["q50"]
            )
        ) / iqr

        model_delta = (
            temporal_pred[(cell, end)]
            -
            temporal_pred[(cell, start)]
        ) / iqr

        result.append(
            (cell, gold_delta, model_delta)
        )

    return result


def slope_origin(g, m):
    g = np.asarray(g, dtype=float)
    m = np.asarray(m, dtype=float)

    denom = np.dot(g, g)

    if denom == 0:
        return math.nan

    return float(
        np.dot(g, m) / denom
    )


rng = np.random.default_rng(SEED)

groups = {
    "All": {"income", "wages", "commute"},
    "Income + wages": {"income", "wages"},
    "Commute": {"commute"},
}

pairs = (
    (2019, 2021),
    (2021, 2023),
    (2019, 2023),
)

temporal_slopes = []

for start, end in pairs:
    for group_name, outcomes in groups.items():
        data = temporal_group_data(
            start,
            end,
            outcomes,
        )

        g = np.asarray(
            [x[1] for x in data],
            dtype=float,
        )

        m = np.asarray(
            [x[2] for x in data],
            dtype=float,
        )

        observed = slope_origin(g, m)

        boot = []

        for _ in range(20000):
            idx = rng.integers(
                0,
                len(data),
                size=len(data),
            )

            s = slope_origin(
                g[idx],
                m[idx],
            )

            if np.isfinite(s):
                boot.append(s)

        lo, hi = np.percentile(
            boot,
            [2.5, 97.5],
        )

        temporal_slopes.append({
            "period": f"{start}-{end}",
            "group": group_name,
            "slope": observed,
            "ci95_low": float(lo),
            "ci95_high": float(hi),
        })


write_csv(
    TAB / "table09_temporal_slopes.csv",
    temporal_slopes,
)


# ------------------------------------------------------------------
# FIGURE 1 — study overview diagram
# ------------------------------------------------------------------

plt.figure(figsize=(12, 6.8))
ax = plt.gca()
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

boxes = [
    (
        0.35, 5.8, 2.2, 1.25,
        "ACS PUMS reference\nCA + NY\n40 population cells"
    ),
    (
        3.15, 5.8, 2.2, 1.25,
        "Replication\nMuse Spark 1.2\n160 unique prompts"
    ),
    (
        5.95, 5.8, 2.2, 1.25,
        "Epistemic uncertainty\n40 cells × 3 variants\n120 prompts"
    ),
    (
        8.75, 5.8, 2.2, 1.25,
        "Elicitation invariance\n20 cells\n60 transformed prompts"
    ),
    (
        3.15, 2.7, 2.2, 1.25,
        "Error structure\nquantile × depth ×\noutcome diagnostics"
    ),
    (
        5.95, 2.7, 2.2, 1.25,
        "Temporal knowledge\n2019 / 2021 / 2023\n120 prompts"
    ),
    (
        8.75, 2.7, 2.2, 1.25,
        "Reliability picture\naccuracy, calibration,\ninvariance, dynamics"
    ),
]

for x, y, w, h, text in boxes:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2,
        fill=False,
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
    )


arrows = [
    ((2.55, 6.4), (3.15, 6.4)),
    ((5.35, 6.4), (5.95, 6.4)),
    ((8.15, 6.4), (8.75, 6.4)),
    ((4.25, 5.8), (4.25, 3.95)),
    ((7.05, 5.8), (7.05, 3.95)),
    ((9.85, 5.8), (9.85, 3.95)),
    ((5.35, 3.3), (5.95, 3.3)),
    ((8.15, 3.3), (8.75, 3.3)),
]

for a, b in arrows:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.1,
        )
    )

ax.text(
    6,
    7.55,
    "Extended evaluation of population knowledge",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
)

savefig("fig01_study_overview")


# ------------------------------------------------------------------
# FIGURES 2–4 — predicted vs ACS median, preserving original-paper view
# ------------------------------------------------------------------

for index, outcome in enumerate(
    ("income", "wages", "commute"),
    start=2,
):
    vals = [
        x
        for x in quantile_scored
        if x["outcome"] == outcome
    ]

    plt.figure(figsize=(6.5, 6))

    gold_vals = np.asarray(
        [x["gold_p50"] for x in vals],
        dtype=float,
    )

    pred_vals = np.asarray(
        [x["pred_p50"] for x in vals],
        dtype=float,
    )

    plt.scatter(
        gold_vals,
        pred_vals,
        alpha=0.75,
    )

    low = min(
        float(np.min(gold_vals)),
        float(np.min(pred_vals)),
    )

    high = max(
        float(np.max(gold_vals)),
        float(np.max(pred_vals)),
    )

    plt.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("ACS weighted median")
    plt.ylabel("Muse predicted median")
    plt.title(
        f"Predicted vs ACS median — {outcome}"
    )

    savefig(
        f"fig0{index}_predicted_vs_acs_{outcome}"
    )


# ------------------------------------------------------------------
# FIGURE 5 — quantile error structure
# ------------------------------------------------------------------

labels = ["p10", "p50", "p90"]

mae = [
    next(
        x["mae_iqr"]
        for x in quantile_error
        if x["quantile"] == q
    )
    for q in labels
]

bias = [
    next(
        x["signed_bias_iqr"]
        for x in quantile_error
        if x["quantile"] == q
    )
    for q in labels
]

x = np.arange(len(labels))
w = 0.36

plt.figure(figsize=(7.2, 5.1))

plt.bar(
    x - w / 2,
    mae,
    w,
    label="MAE",
)

plt.bar(
    x + w / 2,
    bias,
    w,
    label="Signed bias",
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.xticks(
    x,
    labels,
)

plt.ylabel(
    "Normalized error (gold IQR units)"
)

plt.title(
    "Upper-tail estimates are substantially less accurate"
)

plt.legend()

savefig("fig05_quantile_error")


# ------------------------------------------------------------------
# FIGURE 6 — conditioning depth by outcome
# ------------------------------------------------------------------

depth_outcome = defaultdict(dict)

for depth in range(4):
    for outcome in ("income", "wages", "commute"):
        vals = [
            x["p50_mae"]
            for x in cell_summary
            if (
                x["depth"] == depth
                and x["outcome"] == outcome
            )
        ]

        depth_outcome[outcome][depth] = (
            mean(vals)
            if vals
            else None
        )


plt.figure(figsize=(7.8, 5.2))

for outcome in ("income", "wages", "commute"):
    xs = []
    ys = []

    for depth in range(4):
        value = depth_outcome[outcome][depth]

        if value is not None:
            xs.append(depth)
            ys.append(value)

    plt.plot(
        xs,
        ys,
        marker="o",
        label=outcome,
    )

plt.xlabel("Conditioning depth")
plt.ylabel("Normalized median MAE")
plt.xticks([0, 1, 2, 3])

plt.title(
    "Conditioning depth interacts with outcome"
)

plt.legend()

savefig("fig06_depth_outcome_error")


# ------------------------------------------------------------------
# FIGURE 7 — epistemic coverage
# ------------------------------------------------------------------

coverage_categories = [
    ("Nominal", 0.90),
    (
        "Overall",
        epi_analysis["coverage"]["mean"],
    ),
    (
        "Variant 0",
        epi_analysis["variant_summary"]["0"]["coverage"],
    ),
    (
        "Variant 1",
        epi_analysis["variant_summary"]["1"]["coverage"],
    ),
    (
        "Variant 2",
        epi_analysis["variant_summary"]["2"]["coverage"],
    ),
    (
        "Income",
        epi_analysis["outcome_summary"]["income"]["coverage"],
    ),
    (
        "Wages",
        epi_analysis["outcome_summary"]["wages"]["coverage"],
    ),
    (
        "Commute",
        epi_analysis["outcome_summary"]["commute"]["coverage"],
    ),
    (
        "Depth 0",
        epi_analysis["depth_summary"]["0"]["coverage"],
    ),
    (
        "Depth 1",
        epi_analysis["depth_summary"]["1"]["coverage"],
    ),
    (
        "Depth 2",
        epi_analysis["depth_summary"]["2"]["coverage"],
    ),
    (
        "Depth 3",
        epi_analysis["depth_summary"]["3"]["coverage"],
    ),
]

names = [
    x[0]
    for x in coverage_categories
]

values = [
    100 * x[1]
    for x in coverage_categories
]

y = np.arange(len(names))

plt.figure(figsize=(8.5, 6.4))

plt.barh(
    y,
    values,
)

plt.yticks(
    y,
    names,
)

plt.gca().invert_yaxis()

plt.xlim(0, 100)

plt.xlabel("Coverage (%)")

plt.title(
    "Nominal 90% epistemic intervals substantially under-cover"
)

savefig("fig07_epistemic_coverage")


# ------------------------------------------------------------------
# FIGURE 8 — epistemic width vs actual error
# ------------------------------------------------------------------

epi_cells = []

with (
    ROOT
    / "artifacts/analysis"
    / "muse-epistemic-cell-analysis.csv"
).open() as f:
    reader = csv.DictReader(f)

    for row in reader:
        epi_cells.append(row)


plt.figure(figsize=(7, 5.6))

for outcome in ("income", "wages", "commute"):
    vals = [
        r
        for r in epi_cells
        if r["outcome"] == outcome
    ]

    plt.scatter(
        [
            float(r["mean_width_norm"])
            for r in vals
        ],
        [
            float(r["mean_error_norm"])
            for r in vals
        ],
        label=outcome,
        alpha=0.8,
    )


xv = np.asarray(
    [
        float(r["mean_width_norm"])
        for r in epi_cells
    ],
    dtype=float,
)

yv = np.asarray(
    [
        float(r["mean_error_norm"])
        for r in epi_cells
    ],
    dtype=float,
)

if np.std(xv) > 0:
    slope, intercept = np.polyfit(
        xv,
        yv,
        1,
    )

    xx = np.linspace(
        float(np.min(xv)),
        float(np.max(xv)),
        100,
    )

    plt.plot(
        xx,
        intercept + slope * xx,
        linestyle="--",
        linewidth=1,
    )

plt.xlabel(
    "Mean epistemic interval width / gold IQR"
)

plt.ylabel(
    "Mean absolute median error / gold IQR"
)

plt.title(
    "Stated uncertainty weakly predicts actual error"
)

plt.legend()

savefig("fig08_epistemic_width_vs_error")


# ------------------------------------------------------------------
# FIGURE 9 — elicitation invariance
# ------------------------------------------------------------------

transform_order = [
    "reverse_order",
    "verbal_definition",
    "scaled_units",
]

display_names = [
    "Reverse order",
    "Verbal definition",
    "Scaled units",
]

p10 = [
    inv_analysis["summary"][arm]["shift_p10"]["mean"]
    for arm in transform_order
]

p50 = [
    inv_analysis["summary"][arm]["shift_p50"]["mean"]
    for arm in transform_order
]

p90 = [
    inv_analysis["summary"][arm]["shift_p90"]["mean"]
    for arm in transform_order
]

x = np.arange(
    len(transform_order)
)

w = 0.25

plt.figure(figsize=(8.3, 5.2))

plt.bar(
    x - w,
    p10,
    w,
    label="p10",
)

plt.bar(
    x,
    p50,
    w,
    label="p50",
)

plt.bar(
    x + w,
    p90,
    w,
    label="p90",
)

plt.xticks(
    x,
    display_names,
)

plt.ylabel(
    "Mean absolute shift / gold IQR"
)

plt.title(
    "Equivalent prompts perturb the upper tail most"
)

plt.legend()

savefig("fig09_invariance_shifts")


# ------------------------------------------------------------------
# FIGURE 10 — temporal slopes with bootstrap CIs
# ------------------------------------------------------------------

period_order = [
    "2019-2021",
    "2021-2023",
    "2019-2023",
]

plot_groups = [
    "Income + wages",
    "Commute",
]

x = np.arange(
    len(period_order)
)

plt.figure(figsize=(8.3, 5.3))

offsets = {
    "Income + wages": -0.08,
    "Commute": 0.08,
}

for group in plot_groups:
    vals = []

    for period in period_order:
        vals.append(
            next(
                r
                for r in temporal_slopes
                if (
                    r["period"] == period
                    and r["group"] == group
                )
            )
        )

    y = np.asarray(
        [r["slope"] for r in vals]
    )

    lo = np.asarray(
        [r["ci95_low"] for r in vals]
    )

    hi = np.asarray(
        [r["ci95_high"] for r in vals]
    )

    plt.errorbar(
        x + offsets[group],
        y,
        yerr=[
            y - lo,
            hi - y,
        ],
        fmt="o",
        capsize=4,
        label=group,
    )

plt.axhline(
    0,
    linewidth=0.8,
)

plt.axhline(
    1,
    linestyle="--",
    linewidth=0.8,
)

plt.xticks(
    x,
    [
        "2019→2021",
        "2021→2023",
        "2019→2023",
    ],
)

plt.ylabel(
    "Through-origin change slope"
)

plt.title(
    "Temporal tracking is strongly domain dependent"
)

plt.legend()

savefig("fig10_temporal_slopes")


# ------------------------------------------------------------------
# FIGURE 11 — commute temporal trajectories
# ------------------------------------------------------------------

plt.figure(figsize=(8.2, 5.2))

for state, label in (
    ("ca", "California"),
    ("ny", "New York"),
):
    cell = f"{state}:commute:marginal"

    acs = [
        float(
            temporal_gold[year][cell]
            ["gold_2023_dollars"]["q50"]
        )
        for year in YEARS
    ]

    muse = [
        temporal_pred[(cell, year)]
        for year in YEARS
    ]

    plt.plot(
        YEARS,
        acs,
        marker="o",
        label=f"ACS — {label}",
    )

    plt.plot(
        YEARS,
        muse,
        marker="o",
        linestyle="--",
        label=f"Muse — {label}",
    )

plt.xticks(YEARS)

plt.ylabel(
    "Median commute time (minutes)"
)

plt.title(
    "Muse largely misses the pandemic-era commute shift"
)

plt.legend()

savefig("fig11_commute_temporal_trajectory")


# ------------------------------------------------------------------
# Markdown table index
# ------------------------------------------------------------------

table_index = """# Publication tables

Generated deterministically from committed experiment artifacts.

1. `table01_experiment_inventory.csv`
2. `table02_replication_summary.csv`
3. `table03_quantile_error.csv`
4. `table04_error_structure.csv`
5. `table05_epistemic_summary.csv`
6. `table06_epistemic_breakdown.csv`
7. `table07_invariance_summary.csv`
8. `table08_temporal_accuracy.csv`
9. `table09_temporal_slopes.csv`
"""

(TAB / "README.md").write_text(
    table_index
)


# ------------------------------------------------------------------
# Source/output manifest
# ------------------------------------------------------------------

source_paths = [
    ROOT / "artifacts/gold/cells.json",
    ROOT / "artifacts/gold/temporal/cells-2019.json",
    ROOT / "artifacts/gold/temporal/cells-2021.json",
    ROOT / "artifacts/gold/temporal/cells-2023.json",
    ROOT / "artifacts/results/muse-spark-1.2-original-replication.jsonl",
    ROOT / "artifacts/results/muse-spark-1.2-epistemic.jsonl",
    ROOT / "artifacts/results/muse-spark-1.2-invariance.jsonl",
    ROOT / "artifacts/results/muse-spark-1.2-temporal.jsonl",
    ROOT / "artifacts/analysis/muse-epistemic-analysis.json",
    ROOT / "artifacts/analysis/muse-epistemic-cell-analysis.csv",
    ROOT / "artifacts/analysis/muse-invariance-analysis.json",
    ROOT / "artifacts/analysis/muse-temporal-analysis.json",
]

manifest = {
    "sources": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in source_paths
    },
    "generated": {},
}

for path in sorted(
    list(FIG.glob("*"))
    + list(TAB.glob("*"))
):
    if path.is_file():
        manifest["generated"][
            str(path.relative_to(ROOT))
        ] = sha256(path)

manifest_path = (
    ROOT
    / "artifacts/publication"
    / "manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
)


print()
print("=== PUBLICATION ASSET BUILD ===")
print("Figures:", len(list(FIG.glob("*.png"))))
print("Figure PDFs:", len(list(FIG.glob("*.pdf"))))
print("Tables:", len(list(TAB.glob("*.csv"))))
print("Manifest:", manifest_path)

print()
print("=== FIGURES ===")

for path in sorted(FIG.glob("*.png")):
    print(path.name)

print()
print("=== TABLES ===")

for path in sorted(TAB.glob("*.csv")):
    print(path.name)

print()
print("PUBLICATION ASSETS: PASS")
