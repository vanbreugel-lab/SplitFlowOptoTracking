#!/usr/bin/env bash
# Supplemental: altitude responses (flash vs. sham, low vs. high flies)
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_altitude_response).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_altitude_response "Supplemental: altitude responses"
D=FigureGeneration/Supplemental/supp_fig_03_altitude_response

# Builds supp_altitude.svg from supp_altitude_template.svg.
run_notebook "$D" f3_altitude_responses.ipynb

end_figure
