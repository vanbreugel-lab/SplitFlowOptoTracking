# analysis_scripts

These scripts run the unifying-algorithm fit (a mixed-integer quadratic program,
solved with cvxpy's GUROBI solver) on each raw trajectory dataset in
`../raw_data/` and write the fitted results as parquet files to
`../preprocessed_data/`. They are the code that turns the raw data (fetched
from Dryad — see the top-level `README.md`) into the preprocessed data
committed to this repo.

All path handling goes through `../config.py`, so these scripts can be run from
any working directory.

## Setup

1. Install `requirements.txt` (repo root).
2. Point `config.py`'s `GUROBI_LICENSE_FILE` at your Gurobi academic license
   (see the top-level `README.md`).
3. Make sure `../raw_data/` is populated (downloaded from Dryad).

## Running

Each script is standalone:

```bash
python analyze_still.py                    # translation=False (default)
python analyze_still.py --include-translation
```

**These are slow.** Each one fits hundreds of trajectories with bootstrapped
MIQP solves and can take from several minutes to hours depending on the
dataset size and `--include-translation`. Run them individually, not all at
once, unless you have time to wait.

## Scripts

| Script | Reads (`raw_data/`) | Writes (`preprocessed_data/`) |
|---|---|---|
| `analyze_still.py` | `flies_stillair_c1xwt_preprocessed_optotrigger_trimmed.hdf` | `stillair_data_{flash,sham}_translation{Bool}.parquet` |
| `analyze_still_wt.py` | `flies_stillair_wt_preprocessed_optotrigger_trimmed.hdf` | `WT_stillair_data_{flash,sham}_translation{Bool}.parquet` |
| `analyze_laminar.py` | `flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf` | `laminar_data_{flash,sham}_translation{Bool}.parquet` |
| `analyze_laminar_wt.py` | `flies_laminar_wt_preprocessed_optotrigger_trimmed.hdf` | `WT_laminar_data_{flash,sham}_translation{Bool}.parquet` |
| `analyze_unsteady_exp_bottom_on.py` | `flies_splitflow_bottomon_c1xwt_preprocessed_optotrigger_trimmed.hdf` | `steady_bottom_on_data_{flash,sham}_translation{Bool}.parquet` |
| `analyze_unsteady_exp_top_on.py` | `flies_splitflow_topon_c1xwt_preprocessed_optotrigger_trimmed.hdf` | `unsteady_top_on_data_{flash,sham}_translation{Bool}.parquet` |
| `analyze_unsteady_exp_top_on_sham_augmented.py` | `flies_splitflow_topon_c1xwt_preprocessed_optotrigger_trimmed_SHAM_AUGMENTED_merged.parquet` | `unsteady_top_on_data_sham_AUGMENTED_translation{Bool}.parquet` |
| `analyze_unsteady_exp_top_on_wt.py` | `flies_splitflow_topon_wt_preprocessed_optotrigger_trimmed.parquet` | `{unsteady,steady,experiencedunsteady}_top_on_data_WT_{flash,sham}_translation{Bool}.parquet` |
| `analyze_varwind_combined.py` | `flies_varwind_C1WT_combined_optotrigger_fancontrol_trimmed.parquet` | `varwind_N{n_trajecs}_{flash,sham}_{wind_speed}_translation{Bool}.parquet` |
| `analyze_cfd_sims.py` | `flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf`, `cfd_sim_circling_casting_all_wind_conditions_preprocessed.parquet` | `cfd_sim_{condition}_translation{Bool}.parquet` |
| `analyze_NEW_unifying_sims.py` | `new_unifying.parquet` | `new_unifying_{sim_exp}_translation{Bool}.parquet` |
| `preprocess_cfd_data.py` | `casting_sim_constwind_n1000_fixed.parquet` (not currently in `raw_data/` — see note below) | `casting_sim_constwind_n1000_fixed_preprocessed.parquet` (in `raw_data/`, feeds `analyze_cfd_sims.py`'s CFD input after further processing) |

`preprocess_cfd_data.py`'s raw input was not found anywhere in the source
repository during reorganization; place it in `raw_data/` before running that
script.

## find_augmented_shams/

Reference-only provenance notebooks — not runnable outside the original lab.
See `find_augmented_shams/README.md`.
