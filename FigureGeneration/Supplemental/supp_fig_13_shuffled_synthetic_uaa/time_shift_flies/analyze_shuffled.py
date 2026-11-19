"""
Run the unifying algorithm analysis on the shuffled trajectories 

Usage:
    python analyze_shuffled.py [--no-include-translation]

Reads Data/Shuffled_Flies/, writes Data/unifying_algo_results/.
"""

import argparse
from pathlib import Path

import pandas as pd

from align_course_direction_analysis import unifying_algo_analysis as uaa

REPO_ROOT = Path(__file__).resolve().parents[4]  # <repo>/FigureGeneration/Supplemental/supp_fig_13_shuffled_synthetic_uaa/<this dir>/<this file>
DATA_DIR = REPO_ROOT / 'Data' / 'Shuffled_Flies'
RESULTS_DIR = REPO_ROOT / 'Data' / 'unifying_algo_results'

SCENARIOS = ['SHUFFLED_flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf',
            'SHUFFLED_flies_stillair_c1xwt_preprocessed_optotrigger_trimmed.hdf',
            ]
N_TRAJECS = 200

# PARAMETERS
PARAMS = dict(
    obj_id_key='obj_id_unique_event',
    abs_min_ix=120,          # template: 120
    abs_max_ix=500,          # template: 500 (clamped to trajectory length anyway)
    min_ix_range=190,
    max_ix_range=210,
    npoints=50,
    n_bootstraps=10,
    use_cvx_affine=True,
    min_trajec_length=321,   # template: function default 321
)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unifying algorithm analysis on SHUFFLED trajectories.')
    parser.add_argument('--include-translation', action=argparse.BooleanOptionalAction, default=True,
                        help='Include translation in the affine transform; '
                             'use --no-include-translation to turn it off (default: True)')
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS, choices=SCENARIOS)
    args = parser.parse_args()

    translation_flag = f'translation{args.include_translation}'
    RESULTS_DIR.mkdir(exist_ok=True)

    for scenario in args.scenarios:
        print(scenario)
        braid_df = pd.read_hdf(DATA_DIR / scenario)
        unifying_algo_data = uaa.run_miop_affine_fit_on_trajectories(
            braid_df, N_TRAJECS,
            include_translation=args.include_translation,
            **PARAMS)
        unifying_algo_data.to_parquet(RESULTS_DIR / f'SHUFFLED_{scenario}_{translation_flag}.parquet')
