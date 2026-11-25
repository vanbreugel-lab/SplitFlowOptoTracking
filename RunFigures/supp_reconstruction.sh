#!/usr/bin/env bash
# Supplemental: autoencoder reconstructions and reconstruction error (real vs synthetic)
# Uses the Fig. 4 model and data in Data/Fig4_vae_data/ (FigureGeneration/VAE_pipeline/vae_utils.py).
# Final SVG: see figures.yaml (id: supp_reconstruction).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_reconstruction "Supplemental: reconstructions, real vs synthetic"
D=FigureGeneration/Supplemental/supp_fig_07_reconstruction

run_python "$D" plot_reconstruction_summary.py

end_figure
