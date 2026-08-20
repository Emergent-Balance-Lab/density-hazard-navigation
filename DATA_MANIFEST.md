# Data manifest and scope

## Run-level analytical results

`results/simulation/` contains 695 run directories:

| Experiment prefix | Runs |
|---|---:|
| `straight` | 25 |
| `l_shape` | 25 |
| `t_shape` | 25 |
| `static` | 50 |
| `density` | 50 |
| `risk_01` | 50 |
| `risk_05` | 50 |
| `risk_10` | 50 |
| `risk_50` | 50 |
| `risk_100` | 50 |
| `static-0` | 20 |
| `density-10` | 20 |
| `density-30` | 20 |
| `density-60` | 20 |
| `density-120` | 20 |
| `density-180` | 20 |
| `sp-static` | 50 |
| `sp-density-60` | 50 |
| `sp-risk-60` | 50 |
| **Total** | **695** |

Every run directory retains:

- `metrics_summary.json`: extracted time, spatial, and risk metrics;
- `alive_series.csv`: evacuation population time series;
- `run_record.csv`: agent-level terminal/run records used by the analysis;
- `bin/meta.txt`: grid and timing metadata required by the plotting scripts.

The 150 Shipai runs additionally retain `bin/layer_7.bin`,
`bin/layer_8.bin`, and `bin/layer_9.bin`, corresponding to cumulative route
use, congestion duration, and crowd-density load. `sp-static_1/bin/layer_2.bin`
is retained as the shared road-network mask.

## Processed analytical results

`results/analysis/` is copied from the source workspace and includes:

- the paired strategy table `uncertainty/paired_samples_all.csv`;
- update-period run/agent-level backtracking tables;
- Shipai global spatial statistics and mean-field NPZ;
- diagnostic tables, manuscript checks, and current generated figures.

The exact submitted figures are kept separately under
`paper_reference/figures/`; current generated figures are not silently
substituted for them.

## Inputs retained

- `project_random_inputs.csv` maps the paired Shipai runs to hazard-source
  classes used by Appendix C.
- `STL_basic/` and `straight_001`--`025`, `l_shape_001`--`025`, and
  `t_shape_001`--`025` support the corridor validation analysis.

## Raw-data boundary

The source workspace's full `results/` tree was about 22 GB. Most of that size
consisted of repeated layer binaries and GeoTIFF conversions for all runs.
This GitHub package retains the **complete analytical records for all 695
runs** and the full Shipai spatial layers needed by Figures 12--13, but it does
not duplicate every intermediate `layer_*.bin` and `*.tif` from every run.

Consequently, this is a complete paper-analysis dataset, not a byte-complete
archive of all simulator intermediate fields. A literal 22 GB raw archive
should be released separately through an archival data repository or a
versioned external dataset, with checksums and a stable DOI/URL.

The 20 `main_disaster_para_*` raw field directories referenced by
`hazard_raster_visualization.py` were not present in the source workspace at
packaging time. The exact submitted Figure 18 is retained, but those missing
inputs cannot be reconstructed from this folder.
