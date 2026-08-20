# Beta-sensitivity current-results check

Date: 2026-05-09

Status note: `paper/manuscript.tex` has been updated to use the filtered
49-run beta-sensitivity results below. The "versus manuscript" differences
refer to the pre-update manuscript values.

Scope: compared the beta-sensitivity table and prose in
`paper/manuscript.tex` against current run-level results in
`results/simulation/{risk_01,risk_05,risk_10,risk_50,risk_100}_*`.

Statistic convention follows `src/python/strategy_calibration_analysis.py`:
`T_q95` is the 95th percentile across run-level `T_0.95`; `T_q99` is the
95th percentile across run-level `T_0.99`; `E_q95` is the 95th percentile
across run-level `soft_risk`; `H_mean` is the mean run-level `hard_harm`.
Bootstrap intervals below use the same seed pattern and `N_BOOTSTRAP=1000`.

## Key finding from the pre-update manuscript

The pre-update manuscript beta-sensitivity section was not supported by the current
`risk_*` results. The old table is close to the first 20 runs for
`beta=1.0`, but `beta=0.1/0.5/5/10` do not match the current direct
simulation results. With all 50 runs, the mismatch is larger because runs
21--50 add more severe tail cases.

The existing strategy-analysis script would remove run 22 under its
IQR-based outlier filter. Even after applying that filter, the manuscript
still differs materially, especially for `beta=5.0` and `beta=10.0`.

## Current direct results: raw 50 runs

| beta | n | T_q95 [CI] | T_q99 [CI] | E_q95 x1e6 [CI] | H_mean [CI] |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 50 | 277.8 [236.6, 424.3] | 333.5 [286.2, 483.8] | 2.076 [1.787, 3.313] | 1879.3 [315.7, 4581.7] |
| 0.5 | 50 | 281.0 [249.7, 486.5] | 331.6 [306.5, 553.3] | 1.701 [1.450, 2.477] | 108.2 [0.0, 324.5] |
| 1.0 | 50 | 286.4 [257.1, 513.0] | 330.1 [312.3, 578.7] | 1.592 [1.379, 2.174] | 10.7 [0.0, 32.1] |
| 5.0 | 50 | 365.1 [278.1, 534.6] | 446.7 [348.5, 606.3] | 1.468 [1.278, 1.859] | 0.0 [0.0, 0.0] |
| 10.0 | 50 | 371.6 [290.3, 542.9] | 457.4 [358.6, 617.8] | 1.432 [1.210, 1.776] | 0.0 [0.0, 0.0] |

## Current results with existing script filter

The paired seven-strategy table has 50 matched run ids. Applying the
current `strategy_calibration_analysis.py` filters removes only run 22.

| beta | n | T_q95 [CI] | T_q99 [CI] | E_q95 x1e6 [CI] | H_mean [CI] |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 49 | 256.4 [234.0, 317.4] | 311.7 [280.1, 364.8] | 1.947 [1.718, 3.244] | 1758.7 [244.3, 4285.1] |
| 0.5 | 49 | 265.3 [249.6, 323.8] | 321.1 [301.6, 379.8] | 1.667 [1.444, 2.305] | 110.4 [0.0, 331.1] |
| 1.0 | 49 | 277.2 [253.9, 331.9] | 327.6 [310.1, 384.7] | 1.552 [1.370, 1.978] | 10.9 [0.0, 32.8] |
| 5.0 | 49 | 345.3 [278.1, 382.8] | 420.3 [338.8, 462.0] | 1.424 [1.200, 1.548] | 0.0 [0.0, 0.0] |
| 10.0 | 49 | 353.9 [283.4, 404.2] | 434.0 [342.7, 488.9] | 1.380 [1.196, 1.463] | 0.0 [0.0, 0.0] |

## Point-estimate differences versus manuscript

Raw 50-run comparison:

| beta | dT_q95 | dT_q99 | dE_q95 x1e6 | dH_mean |
|---:|---:|---:|---:|---:|
| 0.1 | +30.6 s | +22.4 s | +0.206 | +947.5 |
| 0.5 | +15.2 s | +19.4 s | -0.029 | -297.2 |
| 1.0 | +4.7 s | +5.9 s | +0.032 | +10.7 |
| 5.0 | +74.4 s | +123.4 s | -0.022 | -29.4 |
| 10.0 | +79.8 s | +136.5 s | -0.018 | -119.0 |

Filtered 49-run comparison:

| beta | dT_q95 | dT_q99 | dE_q95 x1e6 | dH_mean |
|---:|---:|---:|---:|---:|
| 0.1 | +9.2 s | +0.6 s | +0.077 | +826.9 |
| 0.5 | -0.5 s | +8.9 s | -0.063 | -295.0 |
| 1.0 | -4.5 s | +3.4 s | -0.008 | +10.9 |
| 5.0 | +54.6 s | +97.0 s | -0.066 | -29.4 |
| 10.0 | +62.1 s | +113.1 s | -0.070 | -119.0 |

## Manuscript prose affected

- The table caption says "Values: mean", but the analysis script reports
  cross-run 95th percentiles for `T_0.95`, `T_0.99`, and `soft_risk`; only
  `H` is a mean. The caption should be corrected when the table is updated.
- The claim that `T_0.95` rises by 18.0% from beta 0.1 to 10.0 is wrong.
  It is 33.8% with raw 50 runs, or 38.0% after the script filter.
- The claim that `E_0.95` falls by 22.3% is too small. It is 31.0% with
  raw 50 runs, or 29.1% after the script filter.
- The hard-harm paragraph is directionally wrong. Current results do not
  show a unique hard-harm minimum at beta 1.0. The raw 50-run and filtered
  49-run results both have `H_mean=0` at beta 5.0 and beta 10.0; beta 1.0
  has a small nonzero mean due to run 38.
- The statement that beta 5.0 and beta 10.0 have well-separated nonzero
  hard-harm confidence intervals is not supported. Their current hard-harm
  intervals are both [0.0, 0.0].
- The ranking-stability sentence needs revision. In current results,
  high-beta DH strongly increases tail time while improving exposure and
  eliminating hard harm; the operating-point argument should be reframed
  as a knee/trade-off choice rather than a hard-harm optimum.
