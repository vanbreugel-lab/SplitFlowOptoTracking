#!/usr/bin/env bash
# Supplemental: variable wind, surge
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_variable_wind_surge).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_variable_wind_surge "Supplemental: variable wind, surge"
D=FigureGeneration/Supplemental/supp_fig_09_variable_wind_speed

# One run per column (layers course*_flash / course*_sham).
for fs in flash sham; do
    run_notebook "$D" AB_varwind_figure_course_direction_surge.ipynb FIG_FLASH_OR_SHAM=$fs
done

end_figure
