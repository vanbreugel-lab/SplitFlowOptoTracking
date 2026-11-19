"""
Dry-run check that the synthetic parquets will run through
uaa.run_miop_affine_fit_on_trajectories with the parameters in analyze_synthetic.py,
WITHOUT calling the MIQP/Gurobi solver.

Replays the pre-solver logic of run_miop_affine_fit_on_trajectories / bootstrap_miop_affine_fit
(column access, dropna, length filter, bootstrap window draws, point sampling, timestep) for every
trajectory and bootstrap. Also reports what the template's original parameters would do.
"""
import numpy as np
import pandas as pd

from align_course_direction_analysis.util import random_integers
import analyze_synthetic as an

REQUIRED_COLS = [an.PARAMS['obj_id_key'], 'obj_id_unique_event', 'course_smoothish', 'time_relative_to_flash']


def check(braid_df, abs_min_ix, abs_max_ix, min_ix_range, max_ix_range, npoints, n_bootstraps,
          min_trajec_length, seed=0, **_):
    rng = np.random.default_rng(seed)
    n_ok, n_short, errors = 0, 0, []
    for obj_id in braid_df.obj_id_unique_event.unique():
        trajec = braid_df[braid_df.obj_id_unique_event == obj_id].dropna()
        if trajec.shape[0] < min_trajec_length:
            n_short += 1
            continue
        course = trajec.course_smoothish.values
        try:
            assert np.all(np.isfinite(course)) and np.all(np.abs(course) <= np.pi + 1e-9)
            for _ in range(n_bootstraps):
                adj = np.min([abs_max_ix, len(course)])
                min_ix = int(rng.integers(abs_min_ix, adj - min_ix_range - 1))
                adj = np.min([adj, min_ix + max_ix_range])
                max_ix = int(rng.integers(min_ix + min_ix_range, adj))
                ix = np.sort(random_integers(min_ix, max_ix, np.min([npoints, len(course)]), rng=rng))
                assert ix.max() < len(course) and max_ix <= len(course)
            int(np.median(np.diff(trajec.time_relative_to_flash.values)) * 1000) / 1000.
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f'{obj_id}: {type(e).__name__}: {e}')
    return n_ok, n_short, errors


if __name__ == '__main__':
    template = dict(abs_min_ix=120, abs_max_ix=500, min_ix_range=190, max_ix_range=210,
                    npoints=50, n_bootstraps=10, min_trajec_length=321)
    all_ok = True
    for scenario in an.SCENARIOS:
        df = pd.read_parquet(an.input_path(scenario))
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        n_ok, n_short, errs = check(df, **an.PARAMS)
        t_ok, t_short, t_errs = check(df, **template)
        print(f'{scenario:15s} missing cols={missing} | analyze_synthetic params: ok={n_ok} short={n_short} '
              f'errors={len(errs)} | template params: ok={t_ok} skipped_short={t_short} errors={len(t_errs)}')
        for e in errs[:3]:
            print('   ', e)
        all_ok &= (not missing) and n_ok == an.N_TRAJECS and not errs
    print('ALL CHECKS PASSED' if all_ok else 'CHECKS FAILED')
