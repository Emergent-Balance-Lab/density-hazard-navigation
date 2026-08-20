# Packaging validation

Validation date: 2026-08-20 (Asia/Shanghai).

## Structural checks

- 24/24 `includegraphics` paths extracted from `paper_reference/manuscript.tex`
  exist in the package.
- 24/24 submitted figure SHA-256 checks passed against
  `SHA256SUMS_PAPER_FIGURES.txt`.
- 695/695 run directories contain `metrics_summary.json`, `alive_series.csv`,
  `run_record.csv`, and `bin/meta.txt`.
- Shipai spatial data include 150 layer-7 files, 150 layer-8 files, 150
  layer-9 files, and one shared layer-2 road mask.
- No C/C++/CUDA source extension or compiled ELF/PE/Mach-O executable was
  found.
- No file exceeds 100 MB.
- No common credential/private-key marker was found in the publishable text,
  code, or top-level input indexes.
- Python bytecode and `__pycache__` created during testing were removed.

## Python validation

All 11 packaged Python files passed syntax parsing. The following scripts were
run from `src/python/` in isolated `uv` environments and completed successfully:

- `corridor_flow_validation.py`: 75 runs loaded; three declared stuck-run
  outliers removed; validation figure generated.
- `strategy_calibration_analysis.py`: 50 paired scenarios loaded; 49 retained
  after the configured filter; figures and LaTeX table generated.
- `update_period_analysis.py`: 120 runs loaded across six configurations;
  figures, table, and summary CSV generated.
- `shipai_strategy_analysis.py`: 150 Shipai runs and spatial layers loaded;
  seven figures and section text generated.
- `shipai_hazard_location_analysis.py`: 150 Shipai runs loaded with no unknown
  hazard class; five Appendix C figures generated.
- `hazard_parameter_sensitivity.py`: four parameter-sensitivity figure sets
  generated.

The initial system-Python attempt lacked optional plotting/geospatial
dependencies. The isolated tests then succeeded after resolving the declared
dependencies; no environment or cache was stored in this package.

`hazard_raster_visualization.py` was not run because its referenced
`main_disaster_para_*` field directories are absent from the source workspace.
This is recorded as a data-provenance gap, not reported as a successful
reproduction.

## GitHub boundary

The assembled folder is approximately 1.7 GB. Git LFS patterns are supplied in
`.gitattributes` for large run-level CSVs, spatial binaries, images, PDFs, NPZ
files, and editable PPTX sources. The release branch preserves the existing EBL
Laboratory repository history and is configured for mirrored publication to
the author repository `Shyr0796/density-hazard-navigation` and the EBL
Laboratory repository `rui-research/density-hazard-navigation`.

The publisher-formatted Elsevier PDF was inspected for bibliographic metadata
but excluded from version control because it states that all rights are
reserved. The README links to the published article through its DOI instead.
