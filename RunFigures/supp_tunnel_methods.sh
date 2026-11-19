#!/usr/bin/env bash
# Supplemental: wind tunnel methods / CFD
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_tunnel_methods).
# Needs Data/wind_tunnel_data/ and Data/2D_PlanarSlices/{XY_steady,XY_unsteady,XZslice}/
# (unzip the three archives inside Data/2D_PlanarSlices/).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_tunnel_methods "Supplemental: wind tunnel methods / CFD"
D=FigureGeneration/Supplemental/supp_fig_01_tunnel_methods

# Order matters: this notebook starts tunnel_methods_supplement.svg from
# tunnel_methods_supplement_template.svg and fills panels A-H ...
run_notebook "$D" wind_tunnel_plot_maker.ipynb
# ... and this one adds the CFD panels L-P to the same file.
run_notebook "$D" CFD_plots.ipynb

end_figure
