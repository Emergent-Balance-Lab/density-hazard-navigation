# Manuscript recheck report, 2026-05-09

Scope: full-text consistency, numerical consistency against current result
artifacts, scientific expression, bibliography/build sanity for
`paper/manuscript.tex`.

## Corrections applied

1. Aligned the main hazard-intensity equation with the implemented model and
   Appendix B:
   - added pre-arrival leakage with `alpha_pre`;
   - replaced the old zero-before-arrival `tau_R` formulation with
     `tau_pre` and `tau_rise`;
   - made the blocking indicator activate only after front arrival.

2. Fixed notation consistency:
   - updated the notation table hazard parameters;
   - changed discussion references from `phi_DH`/`h > h*` to `Phi_DH` and
     hazard intensity `R`;
   - removed ambiguous use of `h` as a hazard variable, because `h` is already
     the SPH smoothing length.

3. Fixed a reliability-metric overclaim:
   - changed the CVaR paragraph from "We report CVaR_0.95" to a
     method-definition statement, since the current results do not report
     formal CVaR estimates.

4. Tightened update-period wording:
   - replaced the ambiguous "3.9% absolute increase" statement with the
     directly verifiable `15.0 s` increase in mean `T_0.95` for 60 s vs 10 s.

5. Fixed bibliography/build hygiene:
   - removed commented DOI field lines that BibTeX was parsing as malformed
     fields;
   - rebuilt with BibTeX and repeated pdflatex passes.

6. Minor expression/layout cleanup:
   - simplified the flow-width sentence;
   - split the acknowledgements paragraph and wrapped it in `sloppypar` to
     remove a severe overfull line.

## Numerical checks against current artifacts

Source artifacts:
- `results/analysis/uncertainty/paired_samples_all.csv`
- `results/simulation/*/metrics_summary.json`
- `results/simulation/*/bin/meta.txt`
- `data/simulation_inputs/project_random_inputs.csv`

Strategy calibration paired samples:
- raw rows: 50; filtered rows: 49; removed run id: 22.
- Static: `T95_p95=263.9`, `T99_p95=330.9`, `E95=2.03e6`, `Hmean=4160.5`.
- Density: `T95_p95=257.3`, `T99_p95=311.3`, `E95=2.10e6`, `Hmean=3670.2`.
- DH beta=1: `T95_p95=277.2`, `T99_p95=327.6`, `E95=1.55e6`, `Hmean=10.9`.
- Beta sweep values in the manuscript match the filtered current data:
  beta 0.1, 0.5, 1, 5, and 10.

Update-period calibration:
- Static: `T95=422.5 s`, total `18.0 s`.
- 10 s: `T95=352.6 s`, `T99=432.0 s`, total `100.5 s`, field `85.6 s`.
- 60 s: `T95=367.6 s`, `T99=447.6 s`, total `30.0 s`, field `15.0 s`.
- 10 s / 60 s total-cost ratio: `3.35x`; field-update ratio: `5.72x`.

Shipai benchmark:
- n = 50 per strategy; population range `40087-49892`, mean `45102`.
- Static: `T95=1114`, `T99=1431`, soft risk `43.11e6`, hard harm `2649.53e3`.
- Density-60: `T95=956`, `T99=1227`, soft risk `26.59e6`, hard harm `451.46e3`.
- DH-60: `T95=961`, `T99=1191`, soft risk `9.39e6`, hard harm `2.86e3`.
- The stated percentage reductions versus Static match the recomputation.

Hazard-source stratification:
- Sample counts are balanced by strategy for each location class:
  near exit `n=18`, near main road `n=18`, far from both `n=14`.
- Recomputed grouped means match the appendix-level trends: DH-60 preserves
  the safety advantage across all three hazard-source classes.

Hazard-threshold sanity check:
- Under the old simple model, `R*=6` would be reached at `110.0 s`.
- Under the implemented `alpha_pre=0.30`, `tau_rise=120 s` model, `R*=6` is
  reached at `67.2 s`, matching the manuscript statement.

## Build status

Fresh output:
- `paper/build/recheck/manuscript.pdf`
- `paper/build/recheck/manuscript.log`
- `paper/build/recheck/manuscript.blg`

Final build status:
- no LaTeX errors;
- no undefined citations;
- no undefined references;
- no BibTeX field/entry errors.

Remaining warnings are layout/toolchain warnings only:
- several overfull/underfull boxes in dense tables and long paragraphs;
- hyperref PDF-string warnings from the author/corresponding-author metadata;
- pdfTeX warnings that included PDF figures are version 1.6/1.7 while the
  output compatibility is 1.5;
- MiKTeX update-check notice.

