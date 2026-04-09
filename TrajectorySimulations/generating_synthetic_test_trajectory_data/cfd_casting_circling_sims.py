#!/usr/bin/env python3
"""
Run casting and circling simulations for all four conditions:
  1. Casting  – laminar (constant wind)
  2. Circling – still air (no wind)
  3. Circling – CFD wind  (vel_1 … vel_6)
  4. Casting  – CFD wind  (vel_1 … vel_6)

Then merge, filter, subsample, and save the final parquet.

Key optimizations over the notebook version (CFD sections only)
---------------------------------------------------------------
1. CFD data is stored as a (N_time, N_points, 3) float32 numpy array
   instead of a list of DataFrames (~8.7 GB vs ~26 GB per condition).
2. Wind lookup uses three O(log N) binary searches + a pre-built 3-D
   index array, replacing the O(N_points) DataFrame boolean mask.
3. Trajectories are run in parallel via fork-based multiprocessing;
   the wind array is inherited copy-on-write — never pickled.

Run with:
    conda activate python3.11
    python cfd_casting_circling_sims.py
"""

import datetime
import gc
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opto_tracking
from splitflow import FeatureFunctions
from kinematics import heading as calc_heading

today = str(datetime.date(2026, 1, 15))

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

DIRECTORIES = ['vel_1', 'vel_2', 'vel_3', 'vel_4', 'vel_5', 'vel_6']
OUTPUT_DIR  = 'sim_results'

N_WORKERS  = mp.cpu_count() // 2
NICE_LEVEL = 10

BOUNDS = [(-0.5, 0.5), (-0.25, 0.25), (0, 0.5)]

# Casting – laminar (constant wind)
CASTING_LAMINAR = dict(
    BEHAVIOR      = 'cast_and_surge',
    SEED          = 2177,
    N_TRAJS       = 1000,
    DURATION      = 3.21,
    DT            = 0.01,
    TAU           = 0.25,
    TAU_SURGE     = 0.02,
    NOISE         = 1.5,
    BIAS          = 0.7,
    SURGE_AMP     = 0.42,
    REJECT_THRESH = 0,
    RADIUS        = 0,
    OMEGA         = 0,
)

# Circling – still air (constant wind, zero net flow)
CIRCLING_NOWIND = dict(
    BEHAVIOR      = 'sink_and_circle',
    SEED          = 2177,
    N_TRAJS       = 1000,
    DURATION      = 3.21,
    DT            = 0.01,
    TAU           = 0.25,
    TAU_SURGE     = 0.25,
    NOISE         = 0.7,
    BIAS          = 1.85,
    SURGE_AMP     = 0,
    REJECT_THRESH = 0,
    RADIUS        = 0.3,
    OMEGA         = 4.05,
)

# Circling – CFD wind
CIRCLING_CFD = dict(
    BEHAVIOR      = 'sink_and_circle',
    SEED          = 5613,
    N_TRAJS       = 1000,
    DURATION      = 3.21,
    DT            = 0.01,
    TAU           = 0.25,
    TAU_SURGE     = 0,
    NOISE         = 0.7,
    BIAS          = 1.85,
    SURGE_AMP     = 0,
    REJECT_THRESH = 0.3,
    RADIUS        = 0.3,
    OMEGA         = 4.05,
    BOUNDS        = BOUNDS,
)

# Casting – CFD wind
CASTING_CFD = dict(
    BEHAVIOR      = 'cast_and_surge',
    SEED          = 5612,
    N_TRAJS       = 1000,
    DURATION      = 3.21,
    DT            = 0.01,
    TAU           = 0.25,
    TAU_SURGE     = 0.02,
    NOISE         = 1.5,
    BIAS          = 0.7,
    SURGE_AMP     = 0.42,
    REJECT_THRESH = 0,
    RADIUS        = 0,
    OMEGA         = 0,
    BOUNDS        = BOUNDS,
)

# ── GLOBAL WIND DATA (set before forking, inherited copy-on-write) ─────────────

_g_velocities = None
_g_ux         = None
_g_uy         = None
_g_uz         = None
_g_grid_index = None


# ── RESOURCE HELPERS ──────────────────────────────────────────────────────────

def _available_ram_gb() -> float:
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable'):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float('inf')


# ── CFD PRE-PROCESSING ────────────────────────────────────────────────────────

def _file_num(name: str) -> int:
    return int(name.split('.')[0].split('__')[1])


def preprocess_cfd(directory: str):
    """
    Load all parquet files in *directory* and convert to compact numpy arrays.

    Returns
    -------
    velocities  : float32 array  (N_time, N_points, 3)
    ux, uy, uz  : sorted unique coordinate arrays
    grid_index  : int32 array (Nu, Nv, Nw) — maps (ix, iy, iz) → row index
    """
    files = sorted(
        [f for f in os.listdir(directory) if f.endswith('.parquet')],
        key=_file_num,
    )
    if not files:
        raise FileNotFoundError(f"No .parquet files found in '{directory}'")

    N_t    = len(files)
    est_gb = N_t * 13_657_920 * 3 * 4 / 1e9
    avail  = _available_ram_gb()
    if est_gb > avail:
        raise MemoryError(
            f"Only {avail:.1f} GB available but '{directory}' needs ~{est_gb:.1f} GB."
        )

    print(f"  Loading {N_t} frames from '{directory}' (~{est_gb:.1f} GB) …", flush=True)

    df0 = pd.read_parquet(
        os.path.join(directory, files[0]),
        columns=['x', 'y', 'z', 'xvel', 'yvel', 'zvel'],
    )
    positions  = df0[['x', 'y', 'z']].values.astype(np.float32)
    N_pts      = len(positions)
    velocities = np.empty((N_t, N_pts, 3), dtype=np.float32)
    velocities[0] = df0[['xvel', 'yvel', 'zvel']].values.astype(np.float32)
    del df0

    for i, fname in enumerate(files[1:], 1):
        df = pd.read_parquet(os.path.join(directory, fname), columns=['xvel', 'yvel', 'zvel'])
        velocities[i] = df.values.astype(np.float32)
        del df

    ux = np.sort(np.unique(positions[:, 0]))
    uy = np.sort(np.unique(positions[:, 1]))
    uz = np.sort(np.unique(positions[:, 2]))

    assert len(ux) * len(uy) * len(uz) == N_pts, (
        f"CFD grid is not a complete Cartesian product: "
        f"{len(ux)}×{len(uy)}×{len(uz)} ≠ {N_pts}."
    )

    ix = np.searchsorted(ux, positions[:, 0])
    iy = np.searchsorted(uy, positions[:, 1])
    iz = np.searchsorted(uz, positions[:, 2])
    grid_index = np.empty((len(ux), len(uy), len(uz)), dtype=np.int32)
    grid_index[ix, iy, iz] = np.arange(N_pts, dtype=np.int32)

    print(
        f"  Done. Grid {len(ux)}×{len(uy)}×{len(uz)} = {N_pts:,} pts, "
        f"{N_t} frames — {velocities.nbytes/1e9:.2f} GB as float32",
        flush=True,
    )
    return velocities, ux, uy, uz, grid_index


# ── FAST WIND OBSERVATION ─────────────────────────────────────────────────────

def _nearest_idx(arr: np.ndarray, val: float) -> int:
    i = int(np.searchsorted(arr, val))
    if i == 0:
        return 0
    if i >= len(arr):
        return len(arr) - 1
    return i - 1 if (val - arr[i - 1]) < (arr[i] - val) else i


class FastWindObs:
    """
    Drop-in replacement for opto_tracking.WindObs (wind_data_type='CFD').

    Uses the pre-loaded numpy velocity array and the 3-D grid index for
    O(log Nx + log Ny + log Nz) nearest-grid-point lookup.
    """

    condition    = 'real'
    vector_field = True

    def __init__(self, velocities, ux, uy, uz, grid_index):
        self._vels = velocities
        self._ux, self._uy, self._uz = ux, uy, uz
        self._gidx = grid_index
        self.wind_vector = np.zeros(3, dtype=np.float64)

    def get_correct_wind_vec(self, agent_position, t, ts):
        t_idx = min(int(round(ts * 10)), self._vels.shape[0] - 1)
        xi = _nearest_idx(self._ux, agent_position[0])
        yi = _nearest_idx(self._uy, agent_position[1])
        zi = _nearest_idx(self._uz, agent_position[2])
        pt = self._gidx[xi, yi, zi]
        self.wind_vector = self._vels[t_idx, pt].astype(np.float64)
        return self.wind_vector

    def relative_upwind(self, trajectory_vector, t=None):
        W = self.wind_vector
        T = np.asarray(trajectory_vector, dtype=np.float64)
        U = -W

        tn = np.linalg.norm(T)
        T_hat = T / tn if tn > 1e-12 else T

        U_para = U - np.dot(U, T_hat) * T_hat
        un = np.linalg.norm(U_para)
        U_para_hat = -U_para / un if un > 1e-12 else U_para

        perp = np.cross(W, np.array([1., 1., 1.]))
        pn = np.linalg.norm(perp)
        U_perp_hat = perp / pn if pn > 1e-12 else perp

        return U_perp_hat, U_para_hat, W


# ── PARALLEL TRAJECTORY WORKER ────────────────────────────────────────────────

def _run_one_traj(args):
    """
    Simulate a single trajectory using the module-level wind globals.
    Globals are inherited from the parent process via fork — never pickled.
    """
    traj_idx, base_seed, params = args

    seed = base_seed + traj_idx
    np.random.seed(seed)
    rng  = np.random.RandomState(seed)

    wind = FastWindObs(_g_velocities, _g_ux, _g_uy, _g_uz, _g_grid_index)

    ag = opto_tracking.SurgingAgent(
        seed=seed,
        tau=params['TAU'],
        noise=params['NOISE'],
        bias=params['BIAS'],
        surge_amp=params['SURGE_AMP'],
        tau_surge=params['TAU_SURGE'],
        bounds=params['BOUNDS'],
    )
    pl = opto_tracking.BoolPlume(
        bounds=np.array(params['BOUNDS']),
        lastflash=-4.8,
    )

    start_pos = np.array([
        rng.uniform(-0.35,  0.35),
        rng.uniform(-0.075, 0.075),
        rng.uniform( 0.15,  0.35),
    ])
    start_vel = np.array([
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.2, 0.2),
    ])
    bias_sign = int(rng.choice([-1, 1]))

    traj = ag.track(
        plume=pl,
        wind=wind,
        behavior=params['BEHAVIOR'],
        start_pos=start_pos,
        start_vel=start_vel,
        duration=params['DURATION'],
        dt=params['DT'],
        reject_thresh=params['REJECT_THRESH'],
        constant_noise=True,
        bias_sign=bias_sign,
        radius=params.get('RADIUS'),
        omega=params.get('OMEGA'),
    )
    traj['headings'] = calc_heading(traj['vs'])[:, 2]
    return traj_idx, traj


# ── RESULT → DATAFRAME ────────────────────────────────────────────────────────

def _trajs_to_df(results, params, directory):
    """Convert a list of (traj_idx, traj_dict) pairs into a single DataFrame."""
    n_steps   = int(params['DURATION'] / params['DT'])
    time_axis = np.arange(-200, n_steps * 10 - 200, 10)
    dfs = []
    for traj_idx, traj in results:
        posdf  = pd.DataFrame(traj['xs'],        columns=['x', 'y', 'z'])
        veldf  = pd.DataFrame(traj['vs'],        columns=['xvel', 'yvel', 'zvel'])
        winddf = pd.DataFrame(traj['real_wind'], columns=['U_x', 'U_y', 'U_z'])
        veldf['time stamp']    = time_axis
        veldf['heading']       = np.arctan2(veldf['yvel'], veldf['xvel'])
        veldf['ground speed']  = np.hypot(veldf['xvel'], veldf['yvel'])
        veldf['obj_id_unique'] = f"{today}_{directory}_{traj_idx}"
        veldf['vel_condition'] = directory
        dfs.append(posdf.join([veldf, winddf]))
    return pd.concat(dfs, ignore_index=True)


# ── RUN CFD SIMULATIONS ───────────────────────────────────────────────────────

def run_cfd_sims(params):
    """
    Run simulations for all CFD velocity conditions using fast wind lookup
    and fork-based multiprocessing.  Returns a concatenated DataFrame.
    """
    global _g_velocities, _g_ux, _g_uy, _g_uz, _g_grid_index

    ctx = mp.get_context('fork')
    all_dfs = []

    for directory in DIRECTORIES:
        print(f"{'='*60}")
        print(f"Condition: {directory}")

        (_g_velocities,
         _g_ux, _g_uy, _g_uz,
         _g_grid_index) = preprocess_cfd(directory)

        worker_args = [
            (i, params['SEED'], params)
            for i in range(params['N_TRAJS'])
        ]

        with ctx.Pool(N_WORKERS) as pool:
            results = pool.map(_run_one_traj, worker_args, chunksize=20)

        df = _trajs_to_df(results, params, directory)
        all_dfs.append(df)
        print(f"  {df['obj_id_unique'].nunique()} trajs completed for {directory}")

        del _g_velocities, _g_ux, _g_uy, _g_uz, _g_grid_index
        _g_velocities = _g_ux = _g_uy = _g_uz = _g_grid_index = None
        gc.collect()
        print("  Wind data freed.")

    return pd.concat(all_dfs, ignore_index=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.nice(NICE_LEVEL)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Casting – laminar (constant wind) ──────────────────────────────────
    print("\n" + "="*60)
    print("Casting – laminar (constant wind)")
    p = CASTING_LAMINAR
    sim_results = opto_tracking.run_sims(
        BEHAVIOR=p['BEHAVIOR'],
        wind_condition='constant',
        SEED=p['SEED'], N_TRAJS=p['N_TRAJS'],
        DURATION=p['DURATION'], DT=p['DT'],
        TAU=p['TAU'], NOISE=p['NOISE'], BIAS=p['BIAS'],
        SURGE_AMP=p['SURGE_AMP'], TAU_SURGE=p['TAU_SURGE'],
        REJECT_THRESH=p['REJECT_THRESH'],
        CONSTANT_NOISE=True,
        BOUNDS=BOUNDS, wind_data=None, vector_field=None,
        wind_positions=None, wind_data_type='CFD',
    )
    casting_laminar_df = opto_tracking.convert_and_save_sims(
        sim_results,
        f"sim_results/SimResults_tau_{p['TAU_SURGE']}_noise_{p['NOISE']}_bias_{p['BIAS']}"
        f"_surgeamp_{p['SURGE_AMP']}_tau_{p['TAU']}_tausurge_{p['TAU_SURGE']}"
        f"_n_{p['N_TRAJS']}_seed_{p['SEED']}_",
        f"{today}_{p['BEHAVIOR']}_constantwind",
        remove_wall_hits=False, save=True,
    )
    casting_laminar_df = FeatureFunctions.get_normed_pos_groupby(casting_laminar_df)
    casting_laminar_df.to_hdf('sim_results/casting_sim_constwind_n1000.hdf', key='n')
    print("Saved: sim_results/casting_sim_constwind_n1000.hdf")

    # ── 2. Circling – still air ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("Circling – still air (no wind)")
    p = CIRCLING_NOWIND
    sim_results = opto_tracking.run_sims(
        BEHAVIOR=p['BEHAVIOR'],
        wind_condition='constant',
        SEED=p['SEED'], N_TRAJS=p['N_TRAJS'],
        DURATION=p['DURATION'], DT=p['DT'],
        TAU=p['TAU'], NOISE=p['NOISE'], BIAS=p['BIAS'],
        SURGE_AMP=p['SURGE_AMP'], TAU_SURGE=p['TAU_SURGE'],
        REJECT_THRESH=p['REJECT_THRESH'],
        RADIUS=p['RADIUS'], OMEGA=p['OMEGA'],
        CONSTANT_NOISE=True,
        BOUNDS=BOUNDS, wind_data=None, vector_field=None,
        wind_positions=None, wind_data_type='CFD',
    )
    circling_nowind_df = opto_tracking.convert_and_save_sims(
        sim_results,
        f"sim_results/SimResults_tau_{p['TAU_SURGE']}_noise_{p['NOISE']}_bias_{p['BIAS']}"
        f"_surgeamp_{p['SURGE_AMP']}_tau_{p['TAU']}_tausurge_{p['TAU_SURGE']}"
        f"_rejectthresh_{p['REJECT_THRESH']}_n_{p['N_TRAJS']}_seed_{p['SEED']}_",
        f"{today}_{p['BEHAVIOR']}_constantwind",
        remove_wall_hits=False, save=True,
    )
    circling_nowind_df = FeatureFunctions.get_normed_pos_groupby(circling_nowind_df)
    circling_nowind_df.to_hdf('sim_results/circle_sim_nowind_n1000.hdf', key='n')
    print("Saved: sim_results/circle_sim_nowind_n1000.hdf")

    # ── 3. Circling – CFD wind ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Circling – CFD wind")
    circling_cfd_df = run_cfd_sims(CIRCLING_CFD)
    circling_cfd_df = FeatureFunctions.get_normed_pos_groupby(circling_cfd_df)
    circling_cfd_df.to_hdf(
        f'sim_results/cfd_sim_results_merged_circling_{today}', key='wind'
    )
    print(f"Saved: sim_results/cfd_sim_results_merged_circling_{today}")

    # ── 4. Casting – CFD wind ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Casting – CFD wind")
    casting_cfd_df = run_cfd_sims(CASTING_CFD)
    casting_cfd_df = FeatureFunctions.get_normed_pos_groupby(casting_cfd_df)
    casting_cfd_df.to_hdf(
        f'sim_results/cfd_sim_results_merged_casting_{today}', key='wind'
    )
    print(f"Saved: sim_results/cfd_sim_results_merged_casting_{today}")

    # ── 5. Post-processing: filter, subsample, merge ──────────────────────────
    print("\n" + "="*60)
    print("Post-processing …")

    def _subsample(df, n=200):
        ids = (
            df.drop_duplicates('obj_id_unique')
            .sample(n=n, replace=False)
            .obj_id_unique.values
        )
        return df[df.obj_id_unique.isin(ids)]

    unsteady_casting = casting_cfd_df.groupby('obj_id_unique').filter(
        lambda d: d.z.max() < 0.25
    ).reset_index(drop=True)
    steady_casting = casting_cfd_df.groupby('obj_id_unique').filter(
        lambda d: d.z.min() > 0.25
    ).reset_index(drop=True)

    unsteady_circle = circling_cfd_df.groupby('obj_id_unique').filter(
        lambda d: d.z.max() < 0.25
    ).reset_index(drop=True)
    steady_circle = circling_cfd_df.groupby('obj_id_unique').filter(
        lambda d: d.z.min() > 0.25
    ).reset_index(drop=True)

    unsteady_casting = _subsample(unsteady_casting)
    unsteady_casting['dataset_name'] = 'unsteady casting'

    steady_casting = _subsample(steady_casting)
    steady_casting['dataset_name'] = 'steady casting'

    unsteady_circle = _subsample(unsteady_circle)
    unsteady_circle['dataset_name'] = 'unsteady circling'

    steady_circle = _subsample(steady_circle)
    steady_circle['dataset_name'] = 'steady circling'

    casting_simple = pd.read_hdf('sim_results/casting_sim_constwind_n1000.hdf')
    circle_simple  = pd.read_hdf('sim_results/circle_sim_nowind_n1000.hdf')

    casting_simple = _subsample(casting_simple)
    casting_simple['dataset_name'] = 'laminar casting'

    circle_simple = _subsample(circle_simple)
    circle_simple['dataset_name'] = 'still air circle'

    bigdf = pd.concat([
        unsteady_casting, steady_casting,
        unsteady_circle,  steady_circle,
        casting_simple,   circle_simple,
    ])
    bigdf.to_parquet('example_cast_circle_trajectories_cfd.parquet')
    print("Saved: example_cast_circle_trajectories_cfd.parquet")
    print("Done.")


if __name__ == '__main__':
    main()
