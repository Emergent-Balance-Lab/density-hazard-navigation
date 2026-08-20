# Figure provenance for revision_260718

Status vocabulary:

- **Exact asset**: byte-identical file copied from `revision_260718`.
- **Code + data**: the available plotting code and its analytical inputs are
  included.
- **Source-output hash match**: before the isolated reruns, the submitted asset
  matched the corresponding file under the source workspace's
  `results/analysis/`. Fresh PDFs can differ byte-for-byte because of PDF
  metadata, fonts, and library versions even when the numerical analysis runs.
- **Manual/final-only**: an exact final asset is present, but a complete
  programmatic regeneration path was not found.

Every row below has an exact submitted asset. “Hash match” refers only to the
source-generated analysis copy observed before the isolated reruns; it is not
required for the exact asset itself.

| Paper figure | Submitted file(s) | Plotting source | Input data in package | Regeneration status |
|---|---|---|---|---|
| Fig. 1 | `03_methodology/framework.pdf` | `figure_sources/editable/framework.pptx` | n/a | Manual schematic |
| Fig. 2 | `03_methodology/potential.pdf` | `figure_sources/editable/potential.pptx` | n/a | Manual schematic |
| Fig. 3 | `05_model_calibration_validation/paradigm_obs_mixed_styled.png` | No matching generator found | Original eight empirical trajectory files were not found | Final-only |
| Fig. 4 | `05_model_calibration_validation/corridor_geometries.pdf` | No exact plotting generator identified | Corridor geometry inputs retained | Final/manual asset |
| Fig. 5 | `05_model_calibration_validation/flow_width_validation.pdf` | `src/python/corridor_flow_validation.py` | 75 corridor run summaries and inputs | Code + data; isolated rerun passed; source-output hash match |
| Fig. 6 | `06_strategy_calibration_simplified_network/simplenetwork_2.pdf` | `figure_sources/editable/simplenetwork.pptx` is the available editable predecessor | n/a | Manual network artwork; exact `_2` generator not found |
| Fig. 7 | `06_strategy_calibration_simplified_network/4x2_beta_tradeoff.pdf`; `4x2_beta_tradeoff_harm.pdf` | `src/python/strategy_calibration_analysis.py` | `results/analysis/uncertainty/paired_samples_all.csv` | Code + data; current generated copies differ from submitted assets |
| Fig. 8 | `06_strategy_calibration_simplified_network/fig3_tradeoff.pdf` | `src/python/update_period_analysis.py` | 120 update-period/static run summaries | Code + data; isolated rerun passed; source-output hash match |
| Fig. 9 | `06_strategy_calibration_simplified_network/fig4_marginal_benefit.pdf` | `src/python/update_period_analysis.py` | Same as Fig. 8 | Code + data; isolated rerun passed; source-output hash match |
| Fig. 10 | `06_strategy_calibration_simplified_network/fig5_compute_vs_period.pdf` | `src/python/update_period_analysis.py` | Same as Fig. 8 | Code + data; isolated rerun passed; source generated copy differed from submitted asset |
| Fig. 11 | `07_shipai_case/shipai_layout.png` | No exact programmatic generator identified | n/a | Final/manual layout composite |
| Fig. 12 | `07_shipai_case/fig7_spatial_reduction.pdf` | `src/python/shipai_strategy_analysis.py` | 150 Shipai summaries; layers 8--9; road mask layer 2 | Code + data; isolated rerun passed; source-output hash match |
| Fig. 13 | `07_shipai_case/compare_frequency_density_time_3x3_ppt_2.pdf` | `figure_sources/editable/shipai_spatial_composite.pptx`; raster helpers in `simulation_io.py` | Shipai layers 7--9 retained, but the representative run ID is not declared in the manuscript/code | Manual composite; exact scripted path not established |
| Fig. 14 | `appendix_b/intensity_sweep_tau_rise.png` | `src/python/hazard_parameter_sensitivity.py` | Parameters embedded in script | Code available; isolated rerun passed; output differs from submitted asset |
| Fig. 15 | `appendix_b/intensity_sweep_tau_pre.png` | `src/python/hazard_parameter_sensitivity.py` | Parameters embedded in script | Code available; isolated rerun passed; output differs from submitted asset |
| Fig. 16 | `appendix_b/intensity_sweep_alpha_pre.png` | `src/python/hazard_parameter_sensitivity.py` | Parameters embedded in script | Code available; isolated rerun passed; output differs from submitted asset |
| Fig. 17 | `appendix_b/intensity_sweep_speed_open.png` | `src/python/hazard_parameter_sensitivity.py` | Parameters embedded in script | Code available; isolated rerun passed; output differs from submitted asset |
| Fig. 18 | `appendix_b/disaster_groupS_arrival_intensity_hardmask_stack3x5.pdf` | `src/python/hazard_raster_visualization.py` | Required `main_disaster_para_S01`--`S05` raw field directories absent from source workspace | Final-only with provenance code |
| Fig. 19 | `appendix_c/fig_hz1_grouped_bars.pdf` | `src/python/shipai_hazard_location_analysis.py` | 150 Shipai run records plus `project_random_inputs.csv` | Code + data; isolated rerun passed; output differs from submitted asset |
| Fig. 20 | `appendix_c/fig_hz2_interaction.pdf` | Same as Fig. 19 | Same as Fig. 19 | Code + data; isolated rerun passed; output differs from submitted asset |
| Fig. 21 | `appendix_c/fig_hz3_reduction.pdf` | Same as Fig. 19 | Same as Fig. 19 | Code + data; isolated rerun passed; output differs from submitted asset |
| Fig. 22 | `appendix_c/fig_hz4_violin.pdf` | Same as Fig. 19 | Same as Fig. 19 | Code + data; isolated rerun passed; output differs from submitted asset |
| Fig. 23 | `appendix_c/fig_hz5_delta_bars.pdf` | Same as Fig. 19 | Same as Fig. 19 | Code + data; isolated rerun passed; output differs from submitted asset |

The submitted figure paths above are relative to
`paper_reference/figures/`. Figure numbering was read from
`paper_reference/manuscript.aux`.
