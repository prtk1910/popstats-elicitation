# popstats-elicitation

**Do language models know what they don't know about populations?**

This repository evaluates language-model knowledge of real population distributions using socioeconomic quantities from the U.S. Census Bureau's American Community Survey (ACS) Public Use Microdata 
Sample.

We study three outcomes:

- household income,
- wage earnings, and
- commute time,

across population cells in California and New York with conditioning depths from 0 to 3.

The original benchmark evaluated **stealth/ox-alpha** through OpenRouter. The `muse-extension` branch replicates and extends the benchmark using **Muse Spark 1.2**.

---

## What we test

The extension separates five questions about population knowledge.

### 1. Population accuracy

Can a language model estimate the 10th percentile, median, and 90th percentile of real population distributions?

### 2. Error structure

Where do those estimates fail?

We examine error across:

- p10, p50, and p90,
- household income, wages, and commute time,
- California and New York,
- conditioning depth, and
- prompt paraphrases.

### 3. Epistemic calibration

Does the model know when its estimate of a **population statistic** is uncertain?

This is different from predicting how variable individuals in the population are.

### 4. Elicitation invariance

Do mathematically equivalent ways of asking the same question recover the same population belief?

We test:

- reversing percentile order,
- replacing percentile notation with verbal definitions, and
- changing to equivalent scaled units.

### 5. Temporal population knowledge

Does the model distinguish population statistics from different years?

We evaluate year-explicit estimates against ACS data from:

- 2019,
- 2021, and
- 2023.

---

## Main findings

### Population estimation replicates

Muse Spark 1.2 approximately replicates the original population-estimation result.

| Metric | Historical Ox Alpha | Muse Spark 1.2 |
| --- | ---: | ---: |
| Normalized median MAE | 0.0420 | **0.0456** |
| Signed median bias | +0.0140 | **+0.0024** |
| Degenerate quantile responses | 0% | **0%** |

The historical Ox comparison is descriptive rather than paired because the original raw Ox responses are not available.

---

### Error is highly structured

Average median accuracy hides large differences across the distribution.

| Quantile | Normalized MAE |
| --- | ---: |
| p10 | **0.0216** |
| p50 | **0.0456** |
| p90 | **0.1618** |

The upper tail is therefore substantially harder to elicit than the lower tail or median.

Median error also differs by outcome:

| Outcome | Normalized median MAE |
| --- | ---: |
| Household income | **0.0275** |
| Wages | **0.0342** |
| Commute | **0.0779** |

Conditioning depth matters as well:

| Conditioning depth | Normalized median MAE |
| --- | ---: |
| 0 | **0.0333** |
| 1 | **0.0364** |
| 2 | **0.0694** |
| 3 | **0.0330** |

The depth-2 degradation persists even when commute cells are excluded, so it is not explained solely by the model's difficulty with commute time.

---

## Predictive vs. epistemic uncertainty

The repository contains two conceptually different uncertainty evaluations.

### Predictive population intervals

The original benchmark asks for a central 90% interval for a **randomly selected individual or household** from the population.

This measures population dispersion, or aleatoric variation.

For Muse Spark 1.2:

- mean predictive coverage: **86.7%**
- minimum cell coverage: **76.1%**
- mean interval width / ACS IQR: **1.253**

The historical Ox Alpha values were:

- mean coverage: 85.5%
- minimum coverage: 73.4%
- width / ACS IQR: 1.190

Muse therefore obtains slightly greater predictive coverage with somewhat wider intervals.

### Epistemic intervals

The extension adds a separate experiment asking for a 90% uncertainty interval for the value of the **population median itself**.

This measures whether the model's expressed uncertainty reflects uncertainty in its estimate.

Results:

- nominal coverage: **90%**
- empirical coverage: **60.8%**
- cell-bootstrap 95% CI: **47.5%-73.3%**
- mean normalized median error: **0.0466**
- mean interval width / ACS IQR: **0.0914**
- mean absolute error / half-width: **1.098**

The model's stated uncertainty also provides only weak information about its actual error:

- interval width vs. actual error, Pearson: **0.235**
- interval width vs. actual error, Spearman: **0.155**
- interval width vs. ACS standard error, Pearson: **0.181**
- interval width vs. ACS standard error, Spearman: **-0.075**

Equivalent epistemic prompt variants produce substantially different coverage:

| Prompt variant | Coverage |
| --- | ---: |
| Variant 0 | **75.0%** |
| Variant 1 | **55.0%** |
| Variant 2 | **52.5%** |

The undercoverage also remains when every cell used during pilot development is excluded.

Across the 28 strictly unseen cells, coverage is **61.9%**.

---

## Elicitation invariance

We test whether equivalent prompt transformations leave the elicited distribution unchanged.

Three transformations are evaluated:

1. reverse the requested percentile order,
2. replace percentile terminology with equivalent verbal definitions,
3. express the same quantities in scaled units.

Mean absolute changes, normalized by the ACS reference IQR:

| Transformation | p10 shift | p50 shift | p90 shift |
| --- | ---: | ---: | ---: |
| Reverse order | 0.0174 | 0.0148 | **0.0595** |
| Verbal definition | 0.0168 | 0.0220 | **0.0485** |
| Scaled units | 0.0169 | 0.0148 | **0.0638** |

The p90 estimate is roughly three to four times more sensitive than p10:

- reverse order: **3.42x**
- verbal definition: **2.89x**
- scaled units: **3.77x**

This suggests that upper-tail population knowledge is not only less accurate but also less stably elicited.

---

## Temporal population knowledge

The temporal experiment evaluates the same population definitions against ACS data from 2019, 2021, and 2023.

Monetary outcomes are converted to constant 2023 dollars before comparing change over time.

### Within-year accuracy

| Year | p10 MAE | p50 MAE | p90 MAE | p50 bias |
| --- | ---: | ---: | ---: | ---: |
| 2019 | 0.0144 | **0.0394** | 0.1450 | -0.0121 |
| 2021 | 0.0272 | **0.0578** | 0.1744 | +0.0126 |
| 2023 | 0.0188 | **0.0424** | 0.1546 | -0.0036 |

Temporal behavior is strongly outcome-dependent.

Income and wage estimates show moderate tracking of real changes across years. Commute estimates show much weaker tracking, particularly around the sharp pandemic-era changes visible in the 2021 ACS 
data.

For income and wages, through-origin slopes relating model-predicted changes to ACS changes are:

| Period | Slope | Bootstrap 95% CI |
| --- | ---: | ---: |
| 2019 -> 2021 | 0.617 | [-0.072, 1.380] |
| 2021 -> 2023 | 0.820 | [0.147, 1.901] |
| 2019 -> 2023 | 0.532 | [0.080, 1.144] |

For commute:

| Period | Slope | Bootstrap 95% CI |
| --- | ---: | ---: |
| 2019 -> 2021 | 0.095 | [-0.119, 0.330] |
| 2021 -> 2023 | -0.016 | [-0.392, 0.333] |
| 2019 -> 2023 | 0.047 | [-0.162, 0.272] |

The evidence therefore supports a **domain-dependent** view of temporal population knowledge rather than a blanket claim that model population priors are either static or fully time-aware.

---

## Experimental inventory

The Muse Spark 1.2 extension contains **460 unique model requests**.

| Experiment | Requests |
| --- | ---: |
| Original-prompt replication | 160 |
| Epistemic uncertainty | 120 |
| Elicitation invariance | 60 |
| Temporal population knowledge | 120 |
| **Total** | **460** |

The replication experiment consists of:

- 120 quantile requests: 40 population cells x 3 prompt variants
- 40 unique predictive-interval requests

All 160 replication requests completed successfully with no malformed or duplicate results.

---

## ACS reference distributions

Population targets are constructed from ACS 1-Year PUMS microdata.

The benchmark uses:

- `WGTP` for household-level outcomes,
- `PWGTP` for person-level outcomes.

### Survey-design uncertainty

The extension uses official ACS replicate weights for uncertainty estimation:

- `WGTP1`-`WGTP80` for household outcomes,
- `PWGTP1`-`PWGTP80` for person outcomes.

Uncertainty is computed using ACS **Successive Difference Replication (SDR)**.

This replaces the row-bootstrap uncertainty calculation used in the original version of the project.

A historical audit verified that this correction changes uncertainty estimates but does **not** change the original 2023 p10, p50, or p90 point estimates:

- 120 quantiles compared
- mismatches: **0**
- maximum absolute difference: **0**

### Temporal monetary adjustment

For the temporal experiment, ACS `ADJINC` is applied to household income and wages.

Historical monetary values are then converted to constant 2023 dollars for cross-year comparisons.

Commute time is left in minutes.

---

## Repository layout

Core source code:

```text
src/popstats/
├── pums.py
├── gold.py
├── grid.py
├── elicit.py
├── api.py
├── analysis.py
├── muse_api.py
├── muse_replication.py
├── muse_epistemic.py
├── muse_invariance.py
├── muse_temporal.py
├── temporal_gold.py
└── publication_assets.py
```

Publication-safe experiment artifacts are version-controlled under:

```text
artifacts/
├── analysis/
├── archive/
│   ├── epistemic-pilot/
│   └── invariance-pilot/
├── gold/
│   └── temporal/
├── results/
└── publication/
    ├── figures/
    ├── tables/
    └── manifest.json
```

Runtime-only files such as API ledgers, raw ACS downloads, reduced local datasets, `.env`, and virtual environments remain ignored.

---

## Experiment artifacts

Exact Muse responses are stored under:

```text
artifacts/results/
├── muse-spark-1.2-original-replication.jsonl
├── muse-spark-1.2-epistemic.jsonl
├── muse-spark-1.2-invariance.jsonl
└── muse-spark-1.2-temporal.jsonl
```

Derived analyses are stored under:

```text
artifacts/analysis/
├── muse-original-replication-score.json
├── muse-diagnostics.json
├── muse-cell-diagnostics.csv
├── muse-epistemic-analysis.json
├── muse-epistemic-cell-analysis.csv
├── muse-invariance-analysis.json
├── muse-invariance-cell-analysis.csv
└── muse-temporal-analysis.json
```

Temporal reference distributions are stored under:

```text
artifacts/gold/temporal/
├── cells-2019.json
├── cells-2021.json
└── cells-2023.json
```

Pilot results used for robustness checks are preserved separately under:

```text
artifacts/archive/
```

---

## Publication figures and tables

Publication assets are generated directly from the committed experiment artifacts.

Run:

```bash
PYTHONPATH=src .venv/bin/python -m popstats.publication_assets
```

This generates:

```text
artifacts/publication/
├── figures/
├── tables/
└── manifest.json
```

The current publication bundle contains:

- **11 figures**
- PNG and PDF versions of every figure
- **9 machine-readable CSV tables**
- a SHA-256 manifest linking generated outputs to their source artifacts

### Figures

```text
fig01_study_overview
fig02_predicted_vs_acs_income
fig03_predicted_vs_acs_wages
fig04_predicted_vs_acs_commute
fig05_quantile_error
fig06_depth_outcome_error
fig07_epistemic_coverage
fig08_epistemic_width_vs_error
fig09_invariance_shifts
fig10_temporal_slopes
fig11_commute_temporal_trajectory
```

### Tables

```text
table01_experiment_inventory.csv
table02_replication_summary.csv
table03_quantile_error.csv
table04_error_structure.csv
table05_epistemic_summary.csv
table06_epistemic_breakdown.csv
table07_invariance_summary.csv
table08_temporal_accuracy.csv
table09_temporal_slopes.csv
```

---

## Reproducing the base data pipeline

Create the environment and run the test suite:

```bash
make setup
make test
```

Fetch ACS PUMS data:

```bash
make fetch
```

Build the reduced analysis dataset:

```bash
make build
```

The API experiments require the corresponding provider credentials when issuing new model calls.

**Do not commit API keys or `.env` files.**

Because exact model responses are committed under `artifacts/results/`, downstream analyses and publication figures can be regenerated without repeating paid model calls.

---

## Reproducibility philosophy

The repository separates four layers of the experiment:

```text
ACS microdata
      |
      v
reference population statistics
      |
      v
raw model responses
      |
      v
analysis outputs
      |
      v
publication tables and figures
```

This allows the reported analyses to be reproduced from preserved model responses without requiring another API run.

The publication manifest additionally records SHA-256 hashes for the inputs and generated publication artifacts.

---

## Takeaway

The experiments support a distinction between **having population knowledge** and **reliably accessing and qualifying that knowledge**.

Muse Spark 1.2 contains useful latent information about real socioeconomic distributions and approximately replicates the original population-estimation result.

At the same time:

- errors are structured rather than uniform,
- upper-tail estimates are substantially weaker,
- epistemic intervals are undercalibrated,
- stated uncertainty only weakly tracks actual error,
- mathematically equivalent prompts can change elicited distributions, and
- temporal population knowledge varies sharply by domain.

Accurate population priors therefore do not by themselves imply reliable metacognitive knowledge of when those priors should be trusted.
