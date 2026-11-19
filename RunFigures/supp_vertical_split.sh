#!/usr/bin/env bash
# Supplemental: vertical split
# Runs the plotting notebook for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_vertical_split).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_vertical_split "Supplemental: vertical split"
D=FigureGeneration/Supplemental/supp_fig_03_vertical_split

run_notebook "$D" S1_verticalsplit.ipynb

end_figure
