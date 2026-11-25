"""
Generate the CFD test sets of Fig. 4E-F / SI Fig. S6 
"""
import gc
import io
import multiprocessing as mp
import os
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'TrajectorySimulations', 'generating_synthetic_test_trajectory_data'))
import generate_synthetic_data as G                         # (puts Code/ on the path: opto_tracking)
import opto_tracking
from cfd_casting_circling_sims import FastWindObs

CFD_DIR = os.environ.get('CFD_DIR', os.path.join(HERE, '..', '..', 'Data', 'CFD', '3D_Velocity_Fields'))
OUT = os.path.join(HERE, '..', '..', 'Data', 'Fig4_vae_data', 'cfd_test_sets_regenerated.parquet')
CONDITIONS = [f'vel_{i}' for i in range(1, 7)]
N_WORKERS = int(os.environ.get('N_WORKERS', mp.cpu_count() // 2))
DURATION, DT, BOUNDS = 3.21, 0.01, [(-0.5, 0.5), (-0.25, 0.25), (0, 0.5)]
# generator: (behaviour, agent (generate_synthetic_data.use_agent), wind rejection, trajectories per condition, seed)
GENERATORS = {'casting': ('casting', 'casting', 0, 500, 21001),
              'casting_t': ('casting', 'casting_t', 0, 500, 22001),
              'smooth_t4p': ('circling', 'smooth', 0.3, 500, 23001)}
TEST_SETS = {'casting': {'casting': 150, 'casting_t': 150}, 'circling': {'smooth_t4p': 300}}   # per flow
_WIND = None                                                # (velocities, ux, uy, uz, grid index), inherited by the workers


def load_condition(cond):
    # the 33 frames of 3D_<cond>.zip: velocities (frames, points, 3), the grid axes and the grid -> point index
    z = zipfile.ZipFile(os.path.join(CFD_DIR, f'3D_{cond}.zip'))
    names = sorted([n for n in z.namelist() if n.endswith('.parquet')], key=lambda s: int(s.split('__')[1].split('.')[0]))
    df0 = pd.read_parquet(io.BytesIO(z.read(names[0])), columns=['x', 'y', 'z', 'xvel', 'yvel', 'zvel'])
    pos = df0[['x', 'y', 'z']].values.astype(np.float32)
    vel = np.empty((len(names), len(pos), 3), dtype=np.float32)
    vel[0] = df0[['xvel', 'yvel', 'zvel']].values
    del df0
    for i, n in enumerate(names[1:], 1):
        vel[i] = pd.read_parquet(io.BytesIO(z.read(n)), columns=['xvel', 'yvel', 'zvel']).values
    ux, uy, uz = (np.unique(pos[:, k]) for k in range(3))
    assert len(ux) * len(uy) * len(uz) == len(pos), 'CFD grid is not a complete Cartesian product'
    gidx = np.empty((len(ux), len(uy), len(uz)), dtype=np.int32)
    gidx[np.searchsorted(ux, pos[:, 0]), np.searchsorted(uy, pos[:, 1]), np.searchsorted(uz, pos[:, 2])] = np.arange(len(pos))
    return vel, ux, uy, uz, gidx


def run_one(args):
    # one trajectory in the loaded wind field: (index, positions, velocities)
    i, seed, p, reject = args
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    ag = opto_tracking.SurgingAgent(seed=seed, tau=p['TAU'], noise=p['NOISE'], bias=p['BIAS'], surge_amp=p['SURGE_AMP'],
                                    tau_surge=p['TAU_SURGE'], bounds=BOUNDS)
    pl = G.SingleFlashPlume(bounds=np.array(BOUNDS), lastflash=-2.8)          # one flash from 0 ms
    start_pos = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.075, 0.075), rng.uniform(0.15, 0.35)])
    start_vel = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.2, 0.2)])
    tr = ag.track(plume=pl, wind=FastWindObs(*_WIND), behavior=p['BEHAVIOR'], start_pos=start_pos, start_vel=start_vel,
                  duration=DURATION, dt=DT, bias_sign=int(rng.choice([-1, 1])), reject_thresh=reject,
                  constant_noise=True, radius=p.get('RADIUS'), omega=p.get('OMEGA'))
    return i, tr['xs'], tr['vs']


def simulate(scale=1.0, conditions=CONDITIONS):
    # every trajectory of every condition and generator (dataframe with z, generator, behaviour and condition)
    global _WIND
    t_ms = np.arange(int(DURATION / DT)) * 10 - 200
    dfs = []
    for cond in conditions:
        k, t0 = CONDITIONS.index(cond), time.time()
        _WIND = load_condition(cond)
        print(f'{cond}: wind loaded ({_WIND[0].shape[0]} frames) in {time.time() - t0:.0f} s', flush=True)
        for comp, (beh, agent, reject, n, seed) in GENERATORS.items():
            p = G.use_agent(agent)                          # class-level settings, inherited by the fork
            n = max(1, int(round(n * scale)))
            with mp.get_context('fork').Pool(N_WORKERS) as pool:
                res = pool.map(run_one, [(i, seed + 10000 * k + i, p, reject) for i in range(n)], chunksize=10)
            for i, xs, vs in res:
                dfs.append(pd.DataFrame({'time stamp': t_ms, 'x': xs[:, 0], 'y': xs[:, 1], 'z': xs[:, 2], 'xvel': vs[:, 0],
                                         'yvel': vs[:, 1], 'simulation_id': f'cfd2_{cond}_{comp}_{i}', 'generator': comp,
                                         'behavior': beh, 'cfd_velocity_field': cond}))
            print(f'  {comp}: {n} trajectories', flush=True)
        _WIND = None
        gc.collect()
    return pd.concat(dfs, ignore_index=True)


def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]


def test_sets(D, scale=1.0):
    # the steady / unsteady test sets drawn from every trajectory D, renamed as the test-set parquet
    rng = np.random.default_rng(0)
    zr = D.groupby('simulation_id').agg(min=('z', 'min'), max=('z', 'max'), generator=('generator', 'first'),
                                        vel=('cfd_velocity_field', 'first'))
    out = []
    for beh, mix in TEST_SETS.items():
        n0 = 0                                              # ids <behaviour>_<n>_<vel_k>_<flow>: steady first
        for flow, ok in [('steady', zr['min'] > 0.25), ('unsteady', zr['max'] < 0.25)]:
            picked = []
            for comp, k in mix.items():
                ids = zr.index[ok & (zr.generator == comp)].tolist()
                picked += list(rng.choice(ids, min(max(1, int(round(k * scale))), len(ids)), replace=False)) if ids else []
            names = {o: f'{beh}_{n0 + i + 1}_{zr.vel[o]}_{flow}' for i, o in enumerate(sorted(picked, key=natural))}
            n0 += len(picked)
            d = D[D.simulation_id.isin(names)]
            out.append(d.assign(obj_id_unique=d.simulation_id.map(names), test_set=f'CFD {beh}, {flow}'))
            print(f'CFD {beh}, {flow}: {len(picked)}', flush=True)
    T = pd.concat(out, ignore_index=True)
    rank = {o: i for i, o in enumerate(sorted(T.obj_id_unique.unique(), key=natural))}
    T = T.sort_values(['obj_id_unique', 'time stamp'], key=lambda s: s.map(rank) if s.name == 'obj_id_unique' else s)
    return T[['obj_id_unique', 'test_set', 'cfd_velocity_field', 'time stamp', 'x', 'y', 'z', 'xvel', 'yvel']].reset_index(drop=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    out = os.path.abspath(args[args.index('--out') + 1]) if '--out' in args else OUT
    scale = float(args[args.index('--n-scale') + 1]) if '--n-scale' in args else 1.0
    G.opto_setup()
    if '--resample' in args:
        D = pd.read_parquet(out.replace('.parquet', '_all.parquet'))
    else:
        D = simulate(scale)
        D.to_parquet(out.replace('.parquet', '_all.parquet'))
    test_sets(D, scale).to_parquet(out)
    print(out)
