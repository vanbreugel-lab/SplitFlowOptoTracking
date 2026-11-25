#!/usr/bin/env bash
# Supplemental: vertical split (not in the SI; not run by run_all.sh)
# Runs the plotting notebook for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_vertical_split).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_vertical_split "Supplemental: vertical split"
D=FigureGeneration/Supplemental/unused_vertical_split

run_notebook "$D" S1_verticalsplit.ipynb

end_figure
