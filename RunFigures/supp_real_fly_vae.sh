#!/usr/bin/env bash
# Supplemental: the Fig. 4 autoencoder + RF trained on real flies only (3 classes)
# Reads the real-fly model, its data and the per-seed classification matrices from Data/final_vae_data/
# (FigureGeneration/VAE_pipeline/vae_utils.py); PDF / PNG rendered with headless Chrome.
# Final SVG: see figures.yaml (id: supp_real_fly_vae).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_real_fly_vae "Supplemental: real-fly autoencoder"
D=FigureGeneration/Supplemental/supp_fig_08_real_fly_vae

run_python "$D" make_supp_real_ae.py

end_figure
