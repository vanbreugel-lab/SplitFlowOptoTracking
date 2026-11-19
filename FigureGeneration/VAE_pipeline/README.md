# VAE pipeline (Figure 4, SI Figures S6–S8)

This folder has the code for the triplet-VAE analysis:

* the synthetic training data and the CFD test sets;
* training the autoencoders and the random-forest classifier;
* the shared module that the figure scripts import.

All inputs are in `Data/final_vae_data/`, the Dryad package (see its `README.md`). It holds the training and test sets as
parquet files, the two trained models and the classification matrices of the 10 real-fly models of SI Fig. S8E.

## Figure scripts

The figure scripts are in the figure folders. Each runs without arguments, from its own directory.

| Figure | Script | RunFigures | Cold run |
|---|---|---|---|
| Fig. 4 | `Main/fig_4_autoencoder/make_f4.py` | `fig4.sh` | ~3 min (~10 s once cached) |
| S6 reconstructions + reconstruction error | `Supplemental/supp_fig_06_reconstruction/plot_reconstruction_summary.py` | `supp_reconstruction.sh` | ~20 s |
| S7 RF on the first k PCs | `Supplemental/supp_fig_07_pca_rf_classifier/plot_pca_rf.py` | `supp_pca_rf_classifier.sh` | ~6 min |
| S8 real-fly autoencoder | `Supplemental/supp_fig_08_real_fly_vae/make_supp_real_ae.py` | `supp_real_fly_vae.sh` | ~2 min (headless Chrome) |

Each script writes `output_figs/<name>_test.svg` next to the finished, hand-edited figure.

## Files

```
vae_utils.py                 shared by the figure scripts and train_vae.py: data loading from Data/final_vae_data,
                             windowing, encoding, reconstruction, the random forest, the synthetic model's cached results
generate_synthetic_data.py   1. regenerates the synthetic training set (synthetic_training_simple.parquet format); also the
                                generators the trajectory notebooks import (use_clock, use_amp_mod, levy_walk, ...)
generate_cfd_data.py         2. regenerates the CFD casting / circling test sets (needs the CFD velocity fields)
train_vae.py                 3. trains either autoencoder from the parquet files; refits the RF and classifies the test sets
cache/                       (not tracked) synthetic_model.pkl: latents and RF probabilities; rebuilt from the parquet
                             files if missing
output/                      (not tracked) train_vae.py outputs
```

### 1. `generate_synthetic_data.py`

Generates a synthetic training set: six classes (casting, circling, random turns, run and tumble, levy flight and browian motion), 10,000 trajectories of 4 s each (`--n` per class). 

```
python generate_synthetic_data.py [--seed S] [--n N]    
```

* **Output:** `Data/final_vae_data/synthetic_training_regenerated.parquet` (`--out`). 
* **Format:** exactly that of `synthetic_training_simple.parquet`: generic ids `casting_1` … `casting_10000`, `circling_1` …, `class`, `time stamp`, `x`, `y`, `xvel`, `yvel`.
* **Runtime:** about 2 min on 8 workers.

The trajectory notebooks import the generators that are not in `Code/` from this module:

* `TrajectorySimulations/generating_synthetic_test_trajectory_data/*.ipynb`;
* `cfd_casting_circling_sims.py`.

Importing it changes nothing. `opto_setup()` switches `opto_tracking` to a single flash.

### 2. `generate_cfd_data.py`

Simulates the training set's casting and circling agents in the six 3D CFD velocity
fields. The wind lookup is `cfd_casting_circling_sims.FastWindObs`. From these runs it draws the four test sets:

* casting and circling, each in steady and in unsteady flow;
* 300 trajectories per flow scenario

Set `CFD_DIR` to the folder that holds `3D_vel_1.zip` … `3D_vel_6.zip`; the default is `Data/CFD/3D_Velocity_Fields`.

* **Output:** `Data/final_vae_data/cfd_test_sets_regenerated.parquet` (and `..._all.parquet`, every simulated
  trajectory).

### 3. `train_vae.py`

Trains either autoencoder from the parquet files in `Data/final_vae_data/` (through `vae_utils`), refits the calibrated
random forest and writes the test-set classification tables (`output/<kind>_test_classification.parquet`; the shipped
models' tables are `Data/final_vae_data/<kind>_test_classification.parquet`).

```
python train_vae.py train synthetic|real [--seed 42] [--epochs 250] [--threads 5]
python train_vae.py classify synthetic|real [--model PATH.pth]
```

* A retrained model is equivalent to the shipped one, not identical.

## Final models

**Synthetic** (`synthetic_sliding_vae_seed42.pth`; Fig. 4, S6, S7)

* Training set: 6 classes × 10,000 trajectories.
* Every sliding window is a sample: 1.1–3.0, 1.6–3.5 and 2.1–4.0 s

**Real flies** (`real_3class_sliding_vae_seed42.pth`; S8 panels A–D)

* Classes: still air flash, laminar flash, sham; class-balanced 80/20 split + sliding windows
* S8E is the mean matrix of 10 models trained with seeds 0–9: `real_model_seed_matrices.parquet`.

**Classifier**

* Calibrated random forest: 1,000 trees, `random_state` 42, temperature calibration, 5-fold.
* Fitted on the latents of every training window.
* Test trajectories are scored on their 1.1–3.0 s window; a probability below 0.6 counts as OOD.

