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
notebook asks this module for the path:

    from splitflow.unifying_results import result_path
    df = pd.read_parquet(result_path('laminar_data_flash_translationTrue.parquet'))

Only fits that include the translation term are current; files in deprecated/ are never returned.
"""
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'Data'
RESULTS_DIR_NAMES = ('unifying_algo_results', 'Unifying_Algo_Results')  # local checkout, Dryad

# (file-name pattern, subdirectory). First match wins.
_LAYOUT = [
    (r'^SHUFFLED_SHUFFLED_flies_laminar_c1xwt', 'shuffled_c1xwt_laminar'),
    (r'^SHUFFLED_SHUFFLED_flies_stillair_c1xwt', 'shuffled_c1xwt_stillair'),
    (r'^synthetic_(brownian|levy_flight|random_turns|run_and_tumble)_', r'synthetic_\1'),
    (r'^new_unifying_(laminar|stillair|unsteady)_', r'unifying_sim_\1'),
    (r'^cfd_sim_still_air_circle_', 'cfd_sim_stillair_circling'),
    (r'^cfd_sim_(laminar|steady|unsteady)_(casting|circling)_', r'cfd_sim_\1_\2'),
    (r'^varwind_N\d+_(flash|sham)_', 'c1xwt_variable_wind_series'),
    (r'^WT_laminar_data_', 'wt_laminar'),
    (r'^WT_stillair_data_', 'wt_stillair'),
    (r'^experiencedunsteady_top_on_data_WT_', 'wt_unsteady'),
    (r'^laminar_data_', 'c1xwt_laminar'),
    (r'^stillair_data_', 'c1xwt_stillair'),
    (r'^unsteady_top_on_data_(flash|sham_AUGMENTED)_', 'c1xwt_unsteady'),
]

# How each result file is made (shown when a file is missing). Paths are relative to the repo root.
_FIT_SCRIPTS = 'FigureGeneration/Main/fig_5_unifying_algo_analysis/analysis_scripts/'
_HOW_TO_MAKE = [
    (r'^SHUFFLED_', 'FigureGeneration/Supplemental/supp_fig_13_shuffled_synthetic_uaa/time_shift_flies/analyze_shuffled.py'),
    (r'^synthetic_', 'FigureGeneration/Supplemental/supp_fig_13_shuffled_synthetic_uaa/uaa_on_synthetic_trajectories/analyze_synthetic.py'),
    (r'^new_unifying_', _FIT_SCRIPTS + 'analyze_NEW_unifying_sims.py --include-translation'),
    (r'^cfd_sim_', _FIT_SCRIPTS + 'analyze_cfd_sims.py --include-translation'),
    (r'^varwind_', _FIT_SCRIPTS + 'analyze_varwind_combined.py --include-translation'),
    (r'^WT_laminar', _FIT_SCRIPTS + 'analyze_laminar_wt.py --include-translation'),
    (r'^WT_stillair', _FIT_SCRIPTS + 'analyze_still_wt.py --include-translation'),
    (r'^experiencedunsteady', _FIT_SCRIPTS + 'analyze_unsteady_exp_top_on_wt.py --include-translation'),
    (r'^laminar_data', _FIT_SCRIPTS + 'analyze_laminar.py --include-translation'),
    (r'^stillair_data', _FIT_SCRIPTS + 'analyze_still.py --include-translation'),
    (r'^unsteady_top_on_data_sham_AUGMENTED', _FIT_SCRIPTS + 'analyze_unsteady_exp_top_on_sham_augmented.py --include-translation'),
    (r'^unsteady_top_on_data_flash', _FIT_SCRIPTS + 'analyze_unsteady_exp_top_on.py --include-translation'),
]


def results_dir():
    """The results directory: Data/unifying_algo_results (local) or Data/Unifying_Algo_Results (Dryad)."""
    for name in RESULTS_DIR_NAMES:
        if (DATA_DIR / name).is_dir():
            return DATA_DIR / name
    return DATA_DIR / RESULTS_DIR_NAMES[0]


def subdir_for(filename):
    """Subdirectory of the results directory that holds `filename`."""
    name = Path(filename).name
    if 'translationFalse' in name:
        return 'deprecated'  # fits without the translation term are no longer used
    for pattern, subdir in _LAYOUT:
        m = re.match(pattern, name)
        if m:
            return m.expand(subdir)
    return 'deprecated'  # datasets no figure uses


def result_path(filename, must_exist=True):
    """
    Full path of a results file, given its file name (any directory part is ignored).

    must_exist=False returns the path a fit script should write to, creating the subdirectory.
    """
    name = Path(filename).name
    subdir = subdir_for(name)
    path = results_dir() / subdir / name
    if not must_exist:
        if subdir == 'deprecated':
            # a fit script was run without --include-translation, or for a dataset no figure uses
            print('NOTE: %s is not used by any figure; writing it to deprecated/' % name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if subdir == 'deprecated':
        raise FileNotFoundError('%s is deprecated (no translation term, or not used by any figure); '
                                'it is kept in %s but is not a current result.' % (name, path.parent))
    if not path.exists():
        how = next((cmd for pattern, cmd in _HOW_TO_MAKE if re.match(pattern, name)), None)
        raise FileNotFoundError('%s not found. Make it with:  python %s' % (path, how))
    return path
