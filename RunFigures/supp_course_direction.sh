#!/usr/bin/env bash
# Supplemental S2: course direction responses (flash vs. sham, low vs. high flies)
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_course_direction).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_course_direction "Supplemental: course direction responses"
D=FigureGeneration/Supplemental/supp_fig_02_course_direction_response

# Builds supp_course_direction.svg from supp_course_direction_template.svg.
run_notebook "$D" course_direction_responses.ipynb

end_figure
