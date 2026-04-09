
# Overview

This repository provides all relevant code used in the manuscript titled "Olfactory Search Behavior Across Flow Regimes Supports a Unifying Algorithm", written by authors Jaleesa Houle, Austin Lopez, Gaurav Kumar, David Stupski, Aditya Nair, and Floris van Breugel.

The preprint is available at:

# Downloading the Data

All data necessary to run the analyses and generate the figures can be found on Dryad:

**https://doi.org/10.5061/dryad.x0k6djhwc**

(Note: data will be made public upon publication.)

Download each zip archive from Dryad and unzip into the `Data/` directory. See `Data/download_from_dryad_and_unzip_here.txt` for the full expected directory structure after unzipping.

# Installing the Code

The `Code/` directory is packaged as `splitflow` (configured in `pyproject.toml`) and contains shared utilities used across all analysis and figure notebooks. Install it in editable mode from the repo root:

```bash
pip install -e .
```

Once installed, notebooks can import shared modules as:

```python
import splitflow.kinematics as kinematics
import splitflow.StochasticAlgorithms as algos
import splitflow.opto_tracking as opto
# etc.
```

# Directory structure

```
├── Code/
│   ├── FlyDataProcessingScript.py      # Raw fly data preprocessing pipeline
│   ├── PlotUtilities.py                # Shared plotting helpers
│   ├── WindDataProcessing.py           # Wind tunnel CFD data processing
│   ├── StochasticAlgorithms.py         # Algorithm implementations for Brownian motion, Levy flight, etc.
│   ├── opto_tracking.py                # Algorithm implementations for Casting/Circling in CFD
│   ├── kinematics.py                   # Trajectory kinematics calculations
│   ├── TripletLoss.py                  # Autoencoder training functions
│   ├── vae_optuna.py                   # VAE param search code
│   ├── RFClassifier.py                 # Random forest classifier
│   ├── affine_math_and_plot_helper.py  # Affine transform math and plotting for unifying algorithm
│   ├── staircase_approximation.py      # Staircase function approximation for unifying algorithm
│   ├── BayesFactorFunctions.py         # Bayes factor statistical functions -- used in older version of paper
│   └── FeatureFunctions.py             # Trajectory feature extraction  -- used in older version of paper
│
│
├── Data/
│   ├── download_from_dryad_and_unzip_here.txt   # Download instructions + expected structure
│   │   [download from Dryad and unzip into the directories as shown below]
│   ├── wind_tunnel_data/               # Wind speed measurements (laminar, outdoor, split flow)
│   ├── CFD/
│   │   ├── 2D_PlanarSlices/            # 2D velocity field slices (XY steady/unsteady, XZ)
│   │   ├── 2D_PlumeData/               # 2D odor concentration slices -- used in older version of paper
│   │   └── 3D_Velocity_Fields/         # 3D velocity fields, 20s simulation (6 archives)
│   ├── Experimental_Fly_Data/
│   │   └── preprocessed_hdfs/          # Merged HDF files per split-flow condition
│   ├── Simulated_Trajectory_Data/      # Casting, circling, and unifying algorithm trajectories
│   ├── Animal_Trajectory_Data/         # Cross-species trajectories (15 species/agent types)
│   ├── Unifying_Algo_Results/
│   │   ├── Main/                       # Analysis outputs for main figures (42 parquet files)
│   │   └── Supplemental/               # Analysis outputs for supplemental figures (42 parquet files)
│   ├── Fig1_Input_Data/                # Data specifically used for Figure 1
│   └── Fig5_Input_Data/                # Data specifically used for Figure 5
│
│
├── FigureGeneration/
│   ├── Main/
│   │   ├── fig_1_math_figure/
│   │   │   ├── A_figure_math_affine_explanation.ipynb
│   │   │   ├── B_figure_math_consolidate_params.ipynb
│   │   │   ├── C_f1_example_trajectories.ipynb
│   │   │   └── misc/
│   │   │       └── basic_plotting_examples/
│   │   │           ├── f1_example_trajectories.ipynb
│   │   │           └── slope_vs_speed_over_length.ipynb
│   │   ├── fig_2_wind_tunnel_CFD_and_sup_fig_1/
│   │   │   ├── fig2_CFD.ipynb
│   │   │   ├── fig2_wind_tunnel_plots.ipynb
│   │   │   └── smokepen_images/
│   │   ├── fig_3_trajectory_characteristics/
│   │   │   └── f3_raw_trajectories.ipynb
│   │   ├── fig_4_autoencoder/
│   │   │   ├── f4_autoencoder.ipynb
│   │   │   ├── triplet_vae_6_classes_70_feats.pth   # Trained model weights
│   │   │   └── misc/
│   │   │       └── optuna_hyperparam_search_multiobj_6class.ipynb
│   │   └── fig_5_unifying_algo_analysis/
│   │       ├── analysis_scripts/                    # scripts used to generate the data in Unifying_Algo_Results
│   │       │   ├── analyze_NEW_unifying_sims.py
│   │       │   ├── analyze_cfd_sims.py
│   │       │   ├── analyze_laminar.py
│   │       │   ├── analyze_laminar_wt.py
│   │       │   ├── analyze_still.py
│   │       │   ├── analyze_still_wt.py
│   │       │   ├── analyze_unsteady_exp_bottom_on.py
│   │       │   └── analyze_unsteady_exp_top_on.py
│   │       └── generate_figure_5/
│   │           ├── A_figure_math_explanation.ipynb
│   │           ├── B_figure_example_trajectories.ipynb
│   │           ├── C_figure_align_course_directions.ipynb
│   │           ├── D_figure_unifying_algo_results_slope_axis.ipynb
│   │           ├── E_slope_vs_speed_over_length.ipynb
│   │           └── animal_trajectories/
│   │               ├── albatross/
│   │               ├── moth/
│   │               └── shark/
│   └── Supplemental/
│       ├── sup_fig_all_animal_trajectories/
│       │   ├── animal_trajectories_and_alignment.ipynb
│       │   └── animal_data/                         # Per-species figure notebooks
│       │       ├── albatross/
│       │       ├── cockroach/
│       │       ├── crabs/
│       │       ├── d_melanogaster/
│       │       ├── d_sechelia/
│       │       ├── eels/
│       │       ├── mosquitoes/
│       │       ├── moths/
│       │       ├── mouse/
│       │       ├── nautical_search/
│       │       ├── nautilus/
│       │       ├── RL_agents/
│       │       ├── sharks/
│       │       ├── terns/
│       │       └── walking_drosophila/
│       ├── sup_fig_altitude_response/
│       │   └── f3_altitude_responses.ipynb
│       ├── sup_fig_extra_real_trajectories/
│       │   └── S3_sample_real_trajectories.ipynb
│       ├── sup_fig_sorted_trajectories/
│       │   ├── supp_All_Trajectories_sorted.ipynb
│       │   └── supplemental_autoencoder_altitude_colored.ipynb
│       ├── sup_fig_synthetic_trajectories/
│       │   └── S5_sample_simulated_trajectories.ipynb
│       ├── sup_fig_tunnel_methods/
│       │   ├── CFD_plots.ipynb
│       │   └── wind_tunnel_plot_maker.ipynb
│       ├── sup_fig_unifying_analysis/
│       │   └── generate_supplementary_figure/
│       │       ├── C_figure_align_course_directions.ipynb
│       │       ├── D_figure_unifying_algo_results_slope_axis.ipynb
│       │       ├── make_figure_align_course_directions.py
│       │       ├── make_figure_unifying_algo_results_slope_axis.py
│       │       └── run_all_figures.sh
│       └── sup_fig_vertical_split/
│           └── S1_verticalsplit.ipynb
│
│
├── TrajectorySimulations/
│   ├── generate_unifying_algo_trajectories/
│   │   ├── generate_unifying_trajectories.ipynb                 # Step-by-step walkthrough
│   │   ├── generate_unifying_trajectories.py                    # Script version
│   │   └── plot_trajectories.ipynb
│   └── generating_synthetic_test_trajectory_data/
│       ├── Casting_Circling_CFD_trajectory_generation.ipynb     # Step-by-step walkthrough
│       ├── cfd_casting_circling_sims.py                         # Script version
│       ├── stochastic_trajectories.ipynb                        # Step-by-step walkthrough for all other algorithms
│       └── param_sweep/                            
│
│
├── LICENSE
└── README.md
```

# Running the simulations

Notebooks are provided for a step-by-step walkthrough of trajectory generation:

- **Unifying algorithm:** `TrajectorySimulations/generate_unifying_algo_trajectories/generate_unifying_trajectories.ipynb`
- **Casting/circling (CFD-informed):** `TrajectorySimulations/generating_synthetic_test_trajectory_data/Casting_Circling_CFD_trajectory_generation.ipynb`
- **Stochastic trajectories:** `TrajectorySimulations/generating_synthetic_test_trajectory_data/stochastic_trajectories.ipynb`

Note: simulations using 3D CFD velocity fields can take a while to run. The CFD 3D data archives (`3D_vel_1.zip` – `3D_vel_6.zip`, ~4 GB each) are only required if re-running these simulations from scratch.

# Generating the figures

Each figure is generated by a Jupyter notebook in `FigureGeneration/`. SVG output files are included alongside each notebook.

**Figure 5 / Supplemental unifying analysis** require a preprocessing step before running the figure notebooks:

1. Run each script in `FigureGeneration/Main/fig_5_unifying_algo_analysis/analysis_scripts/` to apply the unifying algorithm to each dataset. Results are saved to `Data/Unifying_Algo_Results/Main/`.
2. For the supplemental version, run `FigureGeneration/Supplemental/sup_fig_unifying_analysis/generate_supplementary_figure/run_all_figures.sh`. Results are saved to `Data/Unifying_Algo_Results/Supplemental/`.
