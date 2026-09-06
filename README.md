# Coupled density–hazard navigation for evacuation reliability in dense urban informal settlements

This repository accompanies the published paper **“Coupled density–hazard
navigation for evacuation reliability in dense urban informal settlements.”**

**Published article**

Chun Song, Xuchuan Lin, and Rui Cao. “Coupled density–hazard navigation for
evacuation reliability in dense urban informal settlements.” *Reliability
Engineering & System Safety*, 277 (2027), 113181.
[https://doi.org/10.1016/j.ress.2026.113181](https://doi.org/10.1016/j.ress.2026.113181)

## Author and repository links

- **Author:** Chun Song
- **Personal website:** [https://chun-song.com](https://chun-song.com)
- **Personal GitHub account (shyr):**
  [Shyr0796](https://github.com/Shyr0796)
- **Canonical EBL Laboratory repository:**
  [Emergent-Balance-Lab/density-hazard-navigation](https://github.com/Emergent-Balance-Lab/density-hazard-navigation)

The EBL Laboratory repository is the sole canonical location for the complete
code, data, and figure package. The `Shyr0796` and `rui-research` repositories
contain only informational README files that redirect readers here.

## Reproducibility package

This repository is a GitHub-ready **analysis and figure-generation package**
aligned with the published article. It contains analytical result data, Python
plotting code, and available editable figure sources. It intentionally contains
**no manuscript TeX/BibTeX source, manuscript PDF, submitted-reference figure
bundle, C/C++/CUDA source code, or compiled executables**.

中文说明：这是从 RESS 工作目录独立整理出的论文结果公开包，仅保留分析代码、
基础数据、生成结果和可编辑图源；论文 TeX/BibTeX、论文 PDF 及投稿参考图片均不
在本仓库公开。

## Contents

- `src/python/`: plotting and result-analysis code relevant to the submitted
  figures; simulation launchers and C++ code are not included.
- `results/analysis/`: processed CSV/NPZ results, diagnostics, and generated
  analysis figures available in the source workspace.
- `results/simulation/`: complete per-run analytical records for 695 runs
  (`metrics_summary.json`, `alive_series.csv`, `run_record.csv`, and `meta.txt`),
  plus Shipai layers 7--9 needed for the spatial analyses.
- `data/simulation_inputs/`: the random-input index used for Shipai hazard
  stratification and inputs for the 75 corridor-validation runs.
- `figure_sources/editable/`: available editable PowerPoint sources for manual
  schematics/composites.
- `DATA_MANIFEST.md`: data scope, run counts, and raw-data exclusions.

## Environment

Python 3.10 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cd src/python
```

The scripts retain the original relative-path convention, so run them from
`src/python/`.

Main analysis commands:

```bash
python3 corridor_flow_validation.py
python3 strategy_calibration_analysis.py
python3 update_period_analysis.py
python3 shipai_strategy_analysis.py
python3 shipai_hazard_location_analysis.py
python3 hazard_parameter_sensitivity.py
```

Generated files are written under `results/analysis/`. They are analytical
outputs rather than a bundled copy of the manuscript's submitted-reference
figures.

`hazard_raster_visualization.py` is included for provenance, but its 20
`main_disaster_para_*` field directories were not present in the source
workspace and therefore are not packaged. The corresponding figure cannot be
regenerated from this repository alone.

## Large-file storage

The package is large because it retains run-level CSV records and Shipai
spatial layers. `.gitattributes` routes the large/binary analytical artifacts
through Git LFS. Install Git LFS before cloning or downloading all result data:

```bash
git lfs install
git clone https://github.com/Emergent-Balance-Lab/density-hazard-navigation.git
cd density-hazard-navigation
git lfs pull
```

## Reproducibility boundary

The package supports re-analysis and figure regeneration from already-produced
simulation outputs. It does **not** support rebuilding or rerunning the RESS
simulator because the user-specified public package excludes C++/CUDA source
and executable files. The manuscript source and submitted-reference figures are
also excluded; current generated outputs should not be assumed to reproduce
every publication figure byte-for-byte.

## License

No publication license has been selected. Add an explicit `LICENSE` file before
public release if reuse rights are intended; otherwise normal copyright rules
apply.

The publisher-formatted Elsevier PDF is not distributed in this repository;
use the DOI link above to access the published article.
