#!/usr/bin/env bash
# Supplemental: unifying analysis on shuffled / synthetic data
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_shuffled_synthetic).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_shuffled_synthetic "Supplemental: unifying analysis on shuffled / synthetic data"
D=FigureGeneration/Supplemental/supp_fig_13_shuffled_synthetic_uaa

run_notebook "$D" A_shuffled_synthetic_align_course_directions.ipynb
run_notebook "$D" B_shuffled_trajectories.ipynb
run_notebook "$D" C_shuffled_synthetic_slope_axis_ratio.ipynb
run_notebook "$D" D_MI_plots_shuffled_synthetic.ipynb

end_figure
