
# Overview

This repository provides all relevant code used in the manuscript titled "Olfactory Search Behavior Across Flow Regimes Supports a Unifying Algorithm", written by authors Jaleesa Houle, Austin Lopez, Kevin Christie, Gaurav Kumar, David Stupski, Aditya Nair, and Floris van Breugel.

The preprint is available at:

# Downloading the Data

All data necessary to run the analyses and generate the figures can be found on Dryad:

**https://doi.org/10.5061/dryad.x0k6djhwc**

(Note: data will be made public upon publication.)

Download each zip archive from Dryad and unzip into the `Data/` directory. See `Data/download_from_dryad_and_unzip_here.txt` for the full expected directory structure after unzipping.

# Setting up the environment

The figure notebooks were developed and validated on **Python 3.11**. Follow the
steps below to create an environment, install everything, and run the repo.

### 1. Create and activate an environment

Using conda (recommended, matches how the repo was validated):

```bash
conda create -n splitflow python=3.11 -y
conda activate splitflow
```

Or using a plain virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

### 2. Install the Python dependencies and the `splitflow` package

```bash
pip install -r requirements.txt      # all Python dependencies (incl. the private lab repos)
pip install -e .                     # installs the local splitflow package (the Code/ directory)
```

`requirements.txt` pins the dependencies that need care:

- **`pandas < 3.0` (required).** pandas 3.0 makes Copy-on-Write mandatory, which
  returns read-only arrays from `DataFrame.values`; the unifying-algorithm
  alignment code performs in-place array operations and fails with *"output
  array is read-only"* under pandas ≥ 3.0.
- **Private lab repositories** — `align_course_direction_analysis`,
  `braid_tools` (imported as `braid_analysis`), `FlyPlotLib` (`fly_plot_lib`),
  and `FlyStat` (`flystat`) are installed directly from git (pinned URLs in
  `requirements.txt`); you need access to those repositories.

Once installed, notebooks import the shared utilities from the package, e.g.:

```python
import splitflow.kinematics as kinematics
from splitflow.set_zorder_functions import *
from splitflow.affine_math_and_plot_helper import *
```

### 3. Install Gurobi + an academic license (required for the unifying algorithm)

The unifying-algorithm fit is a mixed-integer quadratic program solved with
`cvxpy`'s **GUROBI** backend. Gurobi is commercial but **free for academics**;
the bundled "restricted" license is too small for these problems, so a full
(named-user academic) license is required.

1. `gurobipy` is already installed by `requirements.txt` (or `pip install gurobipy`).
2. Create a free account and request a **Named-User Academic License** at
   https://www.gurobi.com/academia/academic-program-and-licenses/ .
3. From the Gurobi "License Center", copy the `grbgetkey` command for your
   license and run it **while on your university network/VPN**:
   ```bash
   grbgetkey <your-license-key>
   ```
   This writes a `gurobi.lic` file (default `~/gurobi.lic`).
4. Verify it works:
   ```bash
   python -c "import gurobipy; gurobipy.Model(); print('Gurobi OK')"
   ```
   If your license is not in the default location, point Gurobi at it with
   `export GRB_LICENSE_FILE=/path/to/gurobi.lic`.

### 4. Install a LaTeX distribution (for a few notebooks)

A working system LaTeX is required for the notebooks that set matplotlib
`text.usetex=True` (`fig_1/B_figure_math_consolidate_params` and
`Supplemental/unifying_simulation/plot_simulation`). Install **MacTeX** (macOS)
or **TeX Live** (Linux/Windows) system-wide and ensure `latex` is on your `PATH`.

### 5. Get the data and run

Download the data from Dryad into `Data/` (see the "Downloading the Data"
section above), then launch Jupyter from the repo root with the environment
active:

```bash
jupyter lab        # or: jupyter notebook
```

Open any notebook under `FigureGeneration/` and run it top to bottom. Notebooks
read their inputs from `Data/` via relative paths and must be run from their own
directory (Jupyter's default), so no path edits are needed.

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
│   ├── Experimental_Fly_Data/         # full/raw fly datasets, flat (merged raw_data + preprocessed_hdfs):
│   │                                   #   per-condition trimmed HDFs, merged all-nights HDFs,
│   │                                   #   training laminar/stillair sets, variable-wind & CFD parquets
│   ├── Simulated_Trajectory_Data/      # Casting, circling, and unifying algorithm trajectories
│   ├── Animal_Trajectory_Data/         # Cross-species trajectories (15 species/agent types)
│   ├── Unifying_Algo_Results/
│   │   ├── Main/                       # Analysis outputs for main figures (42 parquet files)
│   │   └── Supplemental/               # Supplemental-only outputs (30 parquet files + var_wind_analyzed/):
│   │                                   #   variable-wind, WT-splitflow variants, augmented-sham, df_all_species
│   │                                   #   (results shared with the main figure are read from Main/, not duplicated here)
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
│   │       └── generate_figure_5/       # run A→E in order (they populate one shared SVG)
│   │           ├── A_figure_math_explanation.ipynb
│   │           ├── B_figure_example_trajectories.ipynb
│   │           ├── C_figure_align_course_directions.ipynb
│   │           ├── D_figure_unifying_algo_results_slope_axis.ipynb
│   │           ├── E_variable_wind_slope_axis_ratio.ipynb
│   │           ├── misc/                # alternate/older panel notebooks (not part of the final figure)
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

Each figure is generated by a Jupyter notebook in `FigureGeneration/`. Notebooks
read their inputs from `Data/` via **relative paths** (e.g. `../../../Data/...`),
so they must be run from their own directory — which is Jupyter's default. No
`config.py` or path edits are needed; helper modules import either from the
installed `splitflow` package or from `.py` files that sit alongside the
notebook (e.g. `set_zorder_functions.py`, `unifying_algo_analysis_helper.py`).

Several notebooks build a composite **figurefirst** SVG: they read a layout
template (e.g. `unifying_analysis.svg`) and write the populated figure back. For
`generate_figure_5/`, run `A → B → C → D → E` in order — each populates a
different layer of the same shared SVG.

The pre-existing figure outputs shipped with the repo are kept under an
`output_figs/` subfolder in each figure directory; a fresh run writes its output
alongside with a `_test` suffix so the committed figures are never overwritten.

**Figure 5 / Supplemental unifying analysis** read pre-computed unifying-algorithm
results from `Data/Unifying_Algo_Results/` (`Main/` for the main figure,
`Supplemental/` for the supplemental figures), so the figure notebooks run
without any preprocessing. To *regenerate* those results from the raw trajectory
data, run the scripts in
`FigureGeneration/Main/fig_5_unifying_algo_analysis/analysis_scripts/` (each reads
from `Data/Experimental_Fly_Data/` and writes to `Data/Unifying_Algo_Results/`).

## Known environment requirements / caveats

- **Gurobi academic license.** `fig_5/generate_figure_5/A` and `B` fit the
  unifying-algorithm MIQP live and need a full Gurobi academic license; the
  free "restricted" license is too small and fails with `SolverError`. The
  other Figure 5 notebooks (C, D, E) read pre-computed results and do not need it.
- **LaTeX.** `fig_1/B_figure_math_consolidate_params` and
  `Supplemental/unifying_simulation/plot_simulation` set matplotlib
  `text.usetex=True` and require a working system LaTeX (MacTeX/TeX Live).
- **pandas < 3.0** is required (see the setup section above).
