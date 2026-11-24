#!/usr/bin/env bash
# Figure 3: trajectory characteristics
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: fig3).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure fig3 "Figure 3: trajectory characteristics"
D=FigureGeneration/Main/fig_3_trajectory_characteristics

# Writes f3_raw_data.svg (from f3_template.svg).
run_notebook "$D" f3_raw_trajectories.ipynb

end_figure
