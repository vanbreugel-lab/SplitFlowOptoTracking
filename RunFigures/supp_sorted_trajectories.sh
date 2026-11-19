#!/usr/bin/env bash
# Supplemental: altitude-sorted trajectories (no longer in the SI; not run by run_all.sh)
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_sorted_trajectories).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_sorted_trajectories "Supplemental: altitude-sorted trajectories"
D=FigureGeneration/Supplemental/unused_altitude_sorted_trajectories

run_notebook "$D" supp_All_Trajectories_sorted.ipynb

# Not included: supplemental_autoencoder_altitude_colored.ipynb writes the same output
# file but loads templates that are not in the repo (f4_template.svg,
# r2_supp_all_trajectories_altitude_sorted_template.svg). Add it below if it is
# the version that should produce this figure.

end_figure
