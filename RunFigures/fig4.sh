#!/usr/bin/env bash
# Figure 4: autoencoder
# Draws the figure from the synthetic-trained autoencoder and its data in Data/Fig4_vae_data/ (the Dryad package;
# FigureGeneration/VAE_pipeline/vae_utils.py). Latents and random-forest probabilities are cached in
# FigureGeneration/VAE_pipeline/cache/ after the first run (a few minutes).
# Final SVG: see figures.yaml (id: fig4).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure fig4 "Figure 4: autoencoder"
D=FigureGeneration/Main/fig_4_autoencoder

# Writes output_figs/f4_autoencoder_test.svg (from f4_template.svg); output_figs/f4_autoencoder.svg is the finished figure.
run_python "$D" make_f4.py

end_figure
