#!/usr/bin/env bash
# Supplemental: extra real trajectories
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_extra_real_trajectories).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_extra_real_trajectories "Supplemental: extra real trajectories"
D=FigureGeneration/Supplemental/supp_fig_04_real_trajectories

run_notebook "$D" combined_real_trajectories.ipynb

end_figure
