#!/usr/bin/env bash
# Supplemental: random forest on the first k principal components of the Fig. 4 latent space (k = 2, 3, 4, 5, 10)
# Fits the PCA / RF per k on the Fig. 4 model's latents (Data/final_vae_data/; about 6 min).
# Final SVG: see figures.yaml (id: supp_pca_rf_classifier).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_pca_rf_classifier "Supplemental: PCA-component RF classifier"
D=FigureGeneration/Supplemental/supp_fig_07_pca_rf_classifier

run_python "$D" plot_pca_rf.py

end_figure
