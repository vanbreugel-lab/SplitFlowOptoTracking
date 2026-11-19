#!/usr/bin/env bash
# Supplemental: synthetic trajectories
# Runs the plotting notebook for this figure (no data re-analysis). The notebook
# draws the finished figure (section headers, panel letters, house fonts) and
# writes combined_synthetic_trajectories.svg next to itself.
# Final SVG: see figures.yaml (id: supp_synthetic_trajectories).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_synthetic_trajectories "Supplemental: synthetic trajectories"
D=FigureGeneration/Supplemental/supp_fig_05_synthetic_trajectories

run_notebook "$D" combined_synthetic_trajectories.ipynb

# Not included: plot_trajectories_by_wind_condition.py (writes a standalone PDF, not this figure).

end_figure
