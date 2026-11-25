#!/usr/bin/env bash
# Supplemental: all-animal slopes
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_all_animal_slopes).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_all_animal_slopes "Supplemental: all-animal slopes"
D=FigureGeneration/Supplemental/supp_fig_15_all_animal_slopes

# The notebook fills a copy of supp_fig_all_animal_slopes_template.svg and writes
# the finished supp_fig_all_animal_slopes.svg.
run_notebook "$D" HI_slope_vs_speed_over_length.ipynb

# Not included: the per-species notebooks in animal_trajectories/ (data analysis).

end_figure
