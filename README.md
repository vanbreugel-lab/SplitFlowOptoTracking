
# Overview

This repository provides all relevant code used in the manuscript titled "Olfactory Search Behavior Across Flow Regimes Supports a Unifying Algorithm", written by authors Jaleesa Houle, Austin Lopez, Kevin Christie, Gaurav Kumar, David Stupski, Aditya Nair, and Floris van Breugel.

The preprint is available on bioRxiv at:

[https://www.biorxiv.org/content/10.64898/2026.07.21.739673v1](https://www.biorxiv.org/content/10.64898/2026.07.21.739673v1)

# Downloading the Data

All data necessary to run the analyses and generate the figures can be found on Dryad:

**https://doi.org/10.5061/dryad.x0k6djhwc**

(Note: data will be made public upon publication.)

Download each zip archive from Dryad and unzip into the `Data/` directory. See `Data/download_from_dryad_and_unzip_here.txt` for the full expected directory structure after unzipping.

## Data locations and checks

Every data file the code uses is listed, with a short key, in **`Data/data_paths.yaml`**. No script or
notebook contains a data path; they ask for a file by key:

```python
from splitflow.data_paths import data_path, data_dir
df = pd.read_hdf(data_path('fly.laminar_c1xwt'))        # Data/Experimental_Fly_Data/flies_laminar_c1xwt_...hdf
df = pd.read_csv(data_path('wind.laminar', speed=40))   # a path with a {placeholder}
folder = data_dir('cfd.xz_slice')
```

To rename or move a data file, edit its line in `Data/data_paths.yaml`; no code changes are needed.
(Unifying-algorithm fit results are looked up by file name with `splitflow.unifying_results.result_path`;
their folder layout is the `unifying_results` section of the same file.) The data directory itself can be
moved with the `SPLITFLOW_DATA_DIR` environment variable.

Two checks, both quick and neither needing Gurobi:

| Script | Question it answers |
|---|---|
| `Data/check_data.sh` | Is every file listed in `data_paths.yaml` in `Data/`? `--unlisted` also lists files in `Data/` that no code refers to; `--deep` also opens every file's header. |
| `RunFigures/check_code_data.sh` | Does every data lookup in every script and notebook (including those the figure scripts never run) use a key that exists and point at a file that exists? Also reports any hard-coded data path. `-v` lists every lookup. |

Run both after downloading the data and after reorganizing `Data/`; `RunFigures/run_all.sh` runs them first.
The section **NOT YET LOCATED** at the bottom of `data_paths.yaml` lists files the code refers to that are
not part of the data on this machine (raw inputs of hand-run notebooks and of the simulation scripts). The
checks print them but do not fail because of them.

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
pip install -r requirements.txt      # all Python dependencies (incl. the lab's GitHub packages)
pip install -e .                     # installs the local splitflow package (the Code/ directory)
```

`requirements.txt` pins the dependencies that need care:

- **`pandas < 3.0` (required).** pandas 3.0 makes Copy-on-Write mandatory, which
  returns read-only arrays from `DataFrame.values`; the unifying-algorithm
  alignment code performs in-place array operations and fails with *"output
  array is read-only"* under pandas ≥ 3.0.
- **Lab packages on GitHub (not on PyPI)** — `align_course_direction_analysis`,
  `braid_tools` (imported as `braid_analysis`), `FlyPlotLib` (`fly_plot_lib`),
  and `FlyStat` (`flystat`) are public repositories that are not published on
  PyPI, so they are installed directly from GitHub (pinned URLs in
  `requirements.txt`).

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
look up their inputs in `Data/data_paths.yaml` (see "Data locations and checks"), so no
path edits are needed. They read figure templates and write their outputs next to
themselves, so run them from their own directory (Jupyter's default).

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
│   ├── wind_tunnel_data/               # Wind speed measurements (laminar_flow, outdoor_data, 10_hz_splitflowdata)
│   ├── 2D_PlanarSlices/                # 2D CFD velocity slices; unzip XY_steady/XY_unsteady/XZslice.zip in place
│   ├── CFD/
│   │   ├── 2D_PlumeData/               # 2D odor concentration slices -- used in older version of paper
│   │   └── 3D_Velocity_Fields/         # 3D velocity fields, 20s simulation (6 archives)
│   ├── Experimental_Fly_Data/         # full/raw fly datasets, flat (merged raw_data + preprocessed_hdfs):
│   │                                   #   per-condition trimmed HDFs, merged all-nights HDFs,
│   │                                   #   training laminar/stillair sets, variable-wind & CFD parquets
│   ├── Fig4_vae_data/                 # Fig. 4 / SI S6-S8: autoencoder training data, test sets and models
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
│   │   │   └── C_f1_example_trajectories.ipynb
│   │   ├── fig_2_wind_tunnel_CFD_and_sup_fig_1/
│   │   │   ├── fig2_CFD.ipynb
│   │   │   ├── fig2_wind_tunnel_plots.ipynb
│   │   │   └── smokepen_images/
│   │   ├── fig_3_trajectory_characteristics/
│   │   │   └── f3_raw_trajectories.ipynb
│   │   ├── fig_4_autoencoder/
│   │   │   ├── make_f4.py                           # Builds Figure 4 (data and model: Data/Fig4_vae_data/)
│   │   │   ├── f4_template.svg                      # figurefirst layout
│   │   │   └── output_figs/f4_autoencoder.svg       # final figure
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
│   │           └── animal_trajectories/
│   │               ├── albatross/
│   │               ├── moth/
│   │               └── shark/
│   ├── Supplemental/                   # one folder per SI figure, numbered in SI order (RunFigures/figures.yaml)
│   │   ├── supp_fig_01_tunnel_methods/          # S1  additional wind tunnel / CFD analysis
│   │   ├── supp_fig_02_course_direction_response/ # S2  course direction responses
│   │   ├── supp_fig_03_altitude_response/       # S3  altitude responses
│   │   ├── supp_fig_04_real_trajectories/       # S4  example real fly trajectories
│   │   ├── supp_fig_05_synthetic_trajectories/  # S5  example synthetic trajectories
│   │   ├── supp_fig_06_pca_rf_classifier/       # S6  random forest on the first k = 2-10 principal components
│   │   ├── supp_fig_07_reconstruction/          # S7  VAE reconstructions + reconstruction error, real vs synthetic
│   │   ├── supp_fig_08_real_fly_vae/            # S8  VAE + random forest trained on real flies only
│   │   ├── supp_fig_09_variable_wind_speed/     # S9  variable wind speed
│   │   ├── supp_fig_10_own_fit_r2_pvalues/      # S10 own-fit R^2 and p-values
│   │   ├── supp_fig_11_unifying_flies_shams/    # S11 unifying framework, WT and sham flies
│   │   ├── supp_fig_12_variable_wind_unifying/  # S12 unifying framework, all variable wind flash flies
│   │   ├── supp_fig_13_shuffled_synthetic_uaa/  # S13 unifying algorithm on shuffled and synthetic trajectories
│   │   ├── supp_fig_14_all_animal_trajectories/ # S14 example trajectories from all species
│   │   ├── supp_fig_15_all_animal_slopes/       # S15 all-animal summary, slope vs movement speed
│   │   └── unifying_simulation/                 # supplementary movies (unifying algorithm animations)
│   │
│   └── VAE_pipeline/                   # Fig. 4 / SI S6-S8 (inputs: Data/Fig4_vae_data/; see VAE_pipeline/README.md)
│       ├── vae_utils.py                # shared by the figure scripts: data loading, encoding, random forest
│       ├── generate_synthetic_data.py  # regenerates the synthetic training set
│       ├── generate_cfd_data.py        # regenerates the CFD casting / circling test sets
│       └── train_vae.py                # trains the autoencoders, refits the random forest
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

`RunFigures/` regenerates every manuscript figure (`RunFigures/run_all.sh`, or one
figure, e.g. `RunFigures/fig4.sh`) and collects the final SVG/PDF/PNGs
(`RunFigures/collect_final_figures.sh`); see `RunFigures/README.md`. The figure ids,
SI numbers and output files are listed in `RunFigures/figures.yaml`. Figure 4 and
SI Figures S6-S8 are built by scripts that use `FigureGeneration/VAE_pipeline/` and read
their data from `Data/Fig4_vae_data/` (see the VAE_pipeline README).

Each figure is generated by a Jupyter notebook in `FigureGeneration/`. Notebooks
look up their inputs by key in `Data/data_paths.yaml` (`splitflow.data_paths`); figure
templates and outputs sit next to the notebook, so run notebooks from their own
directory — which is Jupyter's default. No path edits are needed; helper modules import either from the
installed `splitflow` package or from `.py` files that sit alongside the
notebook (e.g. `set_zorder_functions.py`, `unifying_algo_analysis_helper.py`).

Several notebooks build a composite **figurefirst** SVG: they read a layout
template (e.g. `unifying_analysis.svg`) and write the populated figure back. For
`generate_figure_5/`, run `A → B → C → D → E` in order — each populates a
different layer of the same shared SVG.

Every figure's script writes its finished SVG to the path listed for it in
`RunFigures/figures.yaml`, overwriting the committed copy (`git diff` shows what changed;
`python RunFigures/compare_svg.py` renders a visual comparison).

**Figure 5 / Supplemental unifying analysis** read pre-computed unifying-algorithm
results from `Data/Unifying_Algo_Results/` (one subdirectory per dataset; the
layout is defined under `unifying_results` in `Data/data_paths.yaml`, and code looks a result up by
file name with `splitflow.unifying_results.result_path`), so the figure
notebooks run without any preprocessing. To *regenerate* those results from the raw trajectory
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
