# Manuscript data-validation audit

Date: 2026-05-09

Status note: the small manuscript-level discrepancies flagged by the user
on 2026-05-09 have been applied to `paper/manuscript.tex`; the notes below
preserve the original audit evidence for traceability.

Additional status note: the beta-sensitivity section has also been updated
to use the current filtered 49-run `risk_*` results; see
`results/analysis/beta_sensitivity_current_vs_manuscript.md`.

Scope: compared `paper/manuscript.tex` validation/results claims against current
run-level results in `results/simulation`, figure/table artifacts in
`results/analysis`, and archived analysis tables under
`archive/unused_results/analysis_pruned_20260507_152347`.

## Executive summary

Most Shipai full-scenario results, hazard-location appendix values, SPH density
calibration values, and update-period headline means are consistent with the
underlying analysis data.

The main problem is the simplified-network beta-sensitivity section: the
manuscript beta sweep for beta = 0.1, 0.5, 5, and 10 matches a previously
generated archived table, but that table was produced from a partial
`paired_samples.csv` and interpolated the missing strategies. It does not match
the current actual `results/simulation/risk_*` runs. This affects
Table `tab:beta_sensitivity`, the beta-sweep prose, Figure
`fig:beta_tradeoff_combined`, and the ranking-stability claim.

## High-priority discrepancies

### 1. Beta-sensitivity table and prose do not match actual risk_* runs

Manuscript location: `paper/manuscript.tex:1424-1497`.

The Static, Density-aware, and beta = 1.0 rows match current runs 1--20.
However, beta = 0.1, 0.5, 5.0, and 10.0 do not match current
`results/simulation/risk_01`, `risk_05`, `risk_50`, and `risk_100`.

Current run-level recomputation uses the same definitions as
`strategy_calibration_analysis.py`: 95th percentile across 20 scenario values
for `T_0.95`, `T_0.99`, and soft risk; mean across 20 runs for hard harm.

| beta | Metric | Manuscript | Current actual runs | Status |
|---:|---|---:|---:|---|
| 0.1 | T0.95 | 247.2 | 247.9 | minor diff |
| 0.1 | T0.99 | 311.1 | 307.3 | minor diff |
| 0.1 | E0.95 x10^6 | 1.87 | 1.94 | diff |
| 0.1 | H mean | 931.8 | 678.6 | diff |
| 0.5 | T0.95 | 265.8 | 260.3 | diff |
| 0.5 | T0.99 | 312.2 | 319.8 | diff |
| 0.5 | E0.95 x10^6 | 1.73 | 1.69 | minor diff |
| 0.5 | H mean | 405.4 | 0.0 | major diff |
| 1.0 | all four metrics | matches | matches | ok |
| 5.0 | T0.95 | 290.7 | 353.2 | major diff |
| 5.0 | T0.99 | 323.3 | 431.8 | major diff |
| 5.0 | E0.95 x10^6 | 1.49 | 1.45 | minor diff |
| 5.0 | H mean | 29.4 | 0.0 | major diff |
| 10.0 | T0.95 | 291.8 | 356.8 | major diff |
| 10.0 | T0.99 | 320.9 | 442.8 | major diff |
| 10.0 | E0.95 x10^6 | 1.45 | 1.40 | minor diff |
| 10.0 | H mean | 119.0 | 0.0 | major diff |

Implications:

- The manuscript claim "T0.95 rises by 18.0% (247.2 to 291.8 s)" becomes
  about 43.9% using current actual beta = 0.1 to beta = 10 runs.
- The claim "hard harm attains a unique minimum of 0 at beta = 1.0 and
  re-emerges at higher beta" is not supported by current runs: beta = 0.5,
  1.0, 5.0, and 10.0 all have mean hard harm 0.0.
- Ranking stability for T0.95 is not as written. At beta = 0.1 and 0.5, DH is
  faster than Static in the current actual runs.

Recommended action: regenerate a true seven-strategy paired CSV from current
`static`, `density`, `risk_01`, `risk_05`, `risk_10`, `risk_50`, and
`risk_100` runs, rerun `strategy_calibration_analysis.py`, and update the
table, prose, and figures together.

### 2. Field-update cost reduction factor is inconsistent

Manuscript location: `paper/manuscript.tex:1612-1619`.

The reported means match current runs:

| Period | Total time | Dynamic-field time | T0.95 |
|---|---:|---:|---:|
| 10 s | 100.5 s | 85.6 s | 352.6 s |
| 60 s | 30.0 s | 15.0 s | 367.6 s |

The text says the 60 s setting gives a `4.9x` reduction in field-update cost
relative to 10 s. From the current data:

- dynamic-field-time reduction = 85.6 / 15.0 = 5.7x
- total-time reduction = 100.5 / 30.0 = 3.4x

Recommended correction: use `5.7x` if referring to field-update time, or `3.4x`
if referring to total computation time.

### 3. Flow-width relative-error column is inconsistent with listed empirical ranges

Manuscript location: `paper/manuscript.tex:1250-1267`.

The fitted capacities are consistent with the analysis log:

| Scenario | Analysis Csim | Manuscript Csim | Status |
|---|---:|---:|---|
| Straight | 1.896 | 1.90 | ok |
| L-corner | 1.985 | 1.99 | ok |
| T-junction per arm | 1.148 | 1.14 | ok |

But the manuscript relative-error intervals do not correspond to the empirical
ranges printed in the same table:

- Straight: 1.90 vs 1.6--2.2 implies +18.7% vs lower bound and -13.6% vs upper
  bound, not `-4.5% to +18.7%`.
- L-corner: 1.99 vs 1.5--2.1 implies +32.7% and -5.2%, not `-5.2% to +24.0%`.
- T-junction: 1.14 vs 1.1--1.3 implies +3.6% and -12.3%, close but not exactly
  `-13.6% to +3.6%`.

Recommended action: either revise the empirical ranges or recompute the relative
error column from the ranges currently shown.

### 4. Shipai population and scale wording is not supported by current runs

Manuscript locations: `paper/manuscript.tex:1638-1640` and
`paper/manuscript.tex:1772-1774`.

The manuscript says the Shipai benchmark is populated by 41,463 agents and is
two orders of magnitude larger than the calibration network.

Current run-level data:

- Shipai population varies by scenario: 40,087--49,892 agents.
- Shipai mean population across 50 runs: 45,102 agents.
- Simplified calibration network population across runs 1--20: 2,037--5,766,
  mean 4,109.
- Shipai is about 11x larger in agent count than the simplified network, not
  two orders of magnitude.

Recommended correction: describe Shipai as randomized around roughly
40k--50k agents, or report the mean 45,102. Replace "two orders of magnitude"
with "about one order of magnitude" unless another baseline is intended.

### 5. "Four-order-of-magnitude" hard-harm wording is too strong

Manuscript locations: abstract/discussion/conclusion, especially
`paper/manuscript.tex:104-106`, `paper/manuscript.tex:1736-1740`, and
`paper/manuscript.tex:1816-1820`.

Current Shipai hard-harm means:

- Static: 2,649,532
- DH-60: 2,864
- Ratio: 925x, approximately 3 orders of magnitude.

The 99.9% reduction is correct, but "four-order-of-magnitude" is not. Suggested
wording: "nearly three orders of magnitude" or simply "99.9%".

### 6. Hazard speed-sweep table wording conflicts with parameter-generation code

Manuscript location: `paper/manuscript.tex:2359-2401`.

The manuscript says Group S uses baseline multipliers
`0.01, 0.05, 0.10, 0.50, 1.00`, and gives S01 as
`v_open = 0.001 m/s`, `v_build = 0.0005 m/s`.

`src/python/simulation_io.py` uses direct speed levels:
`v_open = [0.01, 0.05, 0.10, 0.50, 1.00]`, with
`v_build = v_open / 2`.

So S01 should be `v_open = 0.01 m/s`, `v_build = 0.005 m/s` if the
simulation-generation code is the source of truth.

## Values that match

### Shipai full-scenario table

Manuscript location: `paper/manuscript.tex:1671-1706`.

Current recomputation from `results/simulation/sp-static_*`,
`sp-density-60_*`, and `sp-risk-60_*` matches the manuscript table within
rounding.

| Strategy | T0.95 | T0.99 | Sigma_s x10^6 | H x10^3 | Manuscript status |
|---|---:|---:|---:|---:|---|
| Static | 1114 | 1431 | 43.11 | 2649.5 | ok |
| Density-60 | 956 | 1227--1228 | 26.59 | 451.5 | ok |
| DH-60 | 961 | 1191 | 9.39 | 2.86 | ok |

Improvement percentages also match:

- Density-60: T0.95 -14.1%, T0.99 -14.2%, soft risk -38.3%, hard harm -83.0%.
- DH-60: T0.95 -13.7%, T0.99 -16.8%, soft risk -78.2%, hard harm -99.9%.

### Hazard-source location appendix

Manuscript location: `paper/manuscript.tex:2412-2532`.

Current recomputation from `project_random_inputs.csv` and Shipai run-level
metrics matches the manuscript table and narrative values:

| Location | Strategy | n | T0.95 | T0.99 | Sigma_s x10^6 | H x10^3 |
|---|---|---:|---:|---:|---:|---:|
| NE | Static | 18 | 1107 | 1425 | 39.09 | 2509.9 |
| NE | Density-60 | 18 | 969 | 1310 | 27.22 | 604.0 |
| NE | DH-60 | 18 | 963 | 1193 | 9.88 | 4.5 |
| NR | Static | 18 | 1126 | 1456 | 33.53 | 1507.5 |
| NR | Density-60 | 18 | 955 | 1165 | 21.42 | 232.6 |
| NR | DH-60 | 18 | 965 | 1215 | 8.53 | 0.0 |
| FR | Static | 14 | 1107 | 1409 | 60.59 | 4297.3 |
| FR | Density-60 | 14 | 942 | 1201 | 32.43 | 536.7 |
| FR | DH-60 | 14 | 953 | 1157 | 9.86 | 4.4 |

The main reductions in the appendix text also match, including Density-60 soft
risk reductions 30.4%, 36.1%, 46.5%, and DH-60 soft risk reductions 74.7%,
74.6%, 83.7%.

### Update-period calibration headline values

Manuscript location: `paper/manuscript.tex:1541-1620`.

The key means match current runs:

| Period | T0.95 | T0.99 | Congestion intensity | Total time |
|---|---:|---:|---:|---:|
| Static | 422.5 | 539.9 | 1.046 | 18.0 s |
| 10 s | 352.6 | 432.0 | 0.788 | 100.5 s |
| 30 s | 362.6 | 439.8 | 0.841 | 44.2 s |
| 60 s | 367.6 | 447.6 | 0.855 | 30.0 s |
| 120 s | 379.6 | 468.7 | 0.905 | 23.2 s |
| 180 s | 387.6 | 492.4 | 0.940 | 21.2 s |

The 10 s improvement claims also match: T0.95 -16.5%, T0.99 -20.0%, congestion
intensity -24.6%, total cost 5.59x static.

### SPH density calibration appendix

Manuscript location: `paper/manuscript.tex:2043-2067`.

Recomputing the SPH density-calibration script logic gives:

- Random hard-core fit: real density = 1.02024 * SPH density - 0.01857,
  R2 = 0.99994.
- Uniform fit: real density = 1.12949 * SPH density + 0.16945,
  R2 = 0.99472.

The manuscript's random-hard-core relation
`1.020 * rho_SPH - 0.019, R2 = 1.000` is consistent.

### Simplified-network baseline comparison at beta = 1

Manuscript location: `paper/manuscript.tex:1369-1401`.

The three-row beta = 1 baseline table matches current runs 1--20:

| Strategy | T0.95 | T0.99 | E0.95 x10^6 | H |
|---|---:|---:|---:|---:|
| Static | 264.7 | 325.8 | 1.97 | 3185.9 |
| Density-aware | 243.2 | 311.6 | 2.10 | 1219.7 |
| DH beta = 1 | 281.7 | 324.2 | 1.56 | 0.0 |

## Not fully verifiable from current analysis artifacts

### Fundamental-diagram fit and model-level reproduction

Manuscript location: `paper/manuscript.tex:1170-1194`.

The values `v_f = 1.574`, `rho_0 = 1.735`, and `n = 0.852` are repeated
consistently in the manuscript, C++ code, and parameter summary. However, the
current `results/analysis` folder does not contain a machine-readable source
for:

- residual RMSE = 0.037 m/s
- model-level speed-density R2 = 0.943
- model-level flow-density R2 = 0.810

These may be correct, but they cannot be independently validated from the
available analysis outputs without the original empirical/simulation extraction
table or script.

### Paper figures are not byte-identical to current analysis figures

The manuscript figure files under `paper/figures` are not byte-identical to the
same-named artifacts under `results/analysis`. For several PDFs the file sizes
are identical, so this may be metadata-only; for some appendix figures the file
sizes differ. If exact visual provenance matters, recopy figures from the
current analysis outputs after rerunning the corrected analysis pipeline.
