"""
Where the unifying-algorithm fit results live.

All fit results (one parquet per dataset and flash/sham condition) are stored under
Data/unifying_algo_results/ (Dryad: Data/Unifying_Algo_Results/), one subdirectory per dataset:

    c1xwt_laminar/               laminar_data_{flash,sham}_translationTrue.parquet
    c1xwt_stillair/              stillair_data_{flash,sham}_translationTrue.parquet
    c1xwt_unsteady/              unsteady_top_on_data_flash_..., unsteady_top_on_data_sham_AUGMENTED_...
    c1xwt_variable_wind_series/  varwind_N700_{flash,sham}_{fan setting}_translationTrue.parquet
    wt_laminar/, wt_stillair/, wt_unsteady/
    cfd_sim_<flow>_<behavior>/   casting / circling agents in the CFD flow fields
    unifying_sim_<wind>/         simulated unifying-algorithm trajectories
    synthetic_<algorithm>/       Brownian, Levy flight, random turns, run and tumble
    shuffled_c1xwt_<wind>/       time-shifted fly controls
    deprecated/                  fits that no figure uses, and every fit made without the translation term

File names are unchanged from the old flat layout, so a file name alone identifies a result. Every script and
notebook asks this module for the path (the folder names and the file-name -> folder rules are listed under
`unifying_results` in Data/data_paths.yaml; see splitflow.data_paths):

    from splitflow.unifying_results import result_path
    df = pd.read_parquet(result_path('laminar_data_flash_translationTrue.parquet'))

Only fits that include the translation term are current; files in deprecated/ are never returned.
"""
from .data_paths import DATA_DIR, results_dir, subdir_for, result_path, how_to_make  # noqa: F401
