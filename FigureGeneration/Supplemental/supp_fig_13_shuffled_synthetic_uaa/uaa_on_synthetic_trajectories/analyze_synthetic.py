"""
Run the unifying algorithm analysis on the synthetic trajectories extracted by
extract_synthetic_trajectories.py. Adapted from analyze_synthetic_template.py.

Differences from the template (needed for the synthetic data):
  * No config.py in this repo -> paths resolved relative to this file; Gurobi license
    taken from GRB_LICENSE_FILE / default location.
  * Synthetic trajectories are ~400 frames (4 s @ 100 Hz, starting at t=0) vs ~521 frames
    for the fly data (starting at t=-0.2 s). The template windows (abs_min_ix=120,
    default min_trajec_length=321) would skip every trajectory / crash in rng.integers.
    abs_min_ix=100 keeps the "start 1.0 s into the trajectory-relative clock" choice
    (fly: 120 frames from start = 1.0 s after flash); fit window sizes (190-210) unchanged.

Reads Data/Synthetic_Flies_for_Unifying/, writes Data/unifying_algo_results/<dataset>/ (see splitflow.unifying_results).

Usage:
    python analyze_synthetic.py [--no-include-translation] [--time-limit S] [--mip-gap G]
                                [--scenarios brownian levy_flight ...]
"""
import argparse
from pathlib import Path

import cvxpy
import pandas as pd

from splitflow.unifying_results import result_path

from align_course_direction_analysis import unifying_algo_analysis as uaa

# Gurobi limits for the MIOP (uaa.miop_optimization hard-codes TimeLimit=20 s, MIPGap=0.01
# with no way to pass them in). When the limit is hit, Gurobi returns its best solution so far.
GUROBI_TIME_LIMIT = 2   # seconds per MIOP solve (one per bootstrap)
GUROBI_MIP_GAP = 0.01    # relative optimality gap at which Gurobi stops


def override_gurobi_limits(time_limit, mip_gap):
    """Patch cvxpy so every GUROBI solve uses these limits instead of uaa's hard-coded ones."""
    orig_solve = cvxpy.Problem.solve

    def solve(self, *args, **kwargs):
        if kwargs.get('solver') == cvxpy.GUROBI:
            kwargs['TimeLimit'] = time_limit
            kwargs['MIPGap'] = mip_gap
        return orig_solve(self, *args, **kwargs)

    cvxpy.Problem.solve = solve


REPO_ROOT = Path(__file__).resolve().parents[4]  # <repo>/FigureGeneration/Supplemental/supp_fig_13_shuffled_synthetic_uaa/<this dir>/<this file>
DATA_DIR = REPO_ROOT / 'Data' / 'Synthetic_Flies_for_Unifying'

SCENARIOS = ['random_turns', 'levy_flight', 'run_and_tumble', 'brownian']
N_TRAJECS = 200

# PARAMETERS
PARAMS = dict(
    obj_id_key='obj_id_unique_event',
    abs_min_ix=10,          # template: 120
    abs_max_ix=400,          # template: 500 (clamped to trajectory length anyway); full 4 s synthetic trajectory
    min_ix_range=190,
    max_ix_range=210,
    npoints=50,
    n_bootstraps=10,
    use_cvx_affine=True,
    min_trajec_length=300,   # template: function default 321
)


def input_path(scenario):
    return DATA_DIR / f'synthetic_{scenario}_n{N_TRAJECS}.parquet'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unifying algorithm analysis on synthetic trajectories.')
    parser.add_argument('--include-translation', action=argparse.BooleanOptionalAction, default=True,
                        help='Include translation in the affine transform; '
                             'use --no-include-translation to turn it off (default: True)')
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument('--time-limit', type=float, default=GUROBI_TIME_LIMIT,
                        help=f'Gurobi TimeLimit (s) per MIOP solve (default: {GUROBI_TIME_LIMIT})')
    parser.add_argument('--mip-gap', type=float, default=GUROBI_MIP_GAP,
                        help=f'Gurobi MIPGap (default: {GUROBI_MIP_GAP})')
    args = parser.parse_args()

    override_gurobi_limits(args.time_limit, args.mip_gap)

    translation_flag = f'translation{args.include_translation}'

    for scenario in args.scenarios:
        print(scenario)
        braid_df = pd.read_parquet(input_path(scenario))
        unifying_algo_data = uaa.run_miop_affine_fit_on_trajectories(
            braid_df, N_TRAJECS,
            include_translation=args.include_translation,
            **PARAMS)
        unifying_algo_data.to_parquet(result_path(f'synthetic_{scenario}_{translation_flag}.parquet', must_exist=False))
