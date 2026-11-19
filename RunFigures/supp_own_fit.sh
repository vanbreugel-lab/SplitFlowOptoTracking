#!/usr/bin/env bash
# Supplemental: own-fit R2 summary
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_own_fit).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_own_fit "Supplemental: own-fit R2 summary"
D=FigureGeneration/Supplemental/supp_fig_10_own_fit_r2_pvalues

run_notebook "$D" own_fit_summary.ipynb

end_figure
