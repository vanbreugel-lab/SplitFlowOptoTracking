#!/usr/bin/env bash
# Supplemental: all animal trajectories and alignment
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_all_animal_trajectories).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_all_animal_trajectories "Supplemental: all animal trajectories and alignment"
D=FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories

run_notebook "$D" animal_trajectories_and_alignment.ipynb

# Not included: the per-species notebooks in animal_data/ (data analysis).

end_figure
