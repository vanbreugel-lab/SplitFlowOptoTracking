"""
Generate the synthetic training set of the autoencoder in Fig 4, in the format of
Data/Fig4_vae_data/synthetic_training_simple.parquet (obj_id_unique casting_1 ..., class, time stamp, x, y, xvel, yvel)

Usage (from this directory): python generate_synthetic_data.py [--seed S] [--n N] [--published] [--out PATH] [--workers N]
Writes: Data/Fig4_vae_data/synthetic_training_regenerated.parquet 
"""
import contextlib
import io
import multiprocessing as mp
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'Code'))
import opto_tracking                                
import StochasticAlgorithms as SA
from splitflow.data_paths import data_path

OUT = str(data_path('vae.synthetic_training_regenerated', must_exist=False))
N_WORKERS = 8
T = 4.0
TT = T + 0.0100001                                          # int(TT / 0.01) = 401 steps (4.01 / 0.01 rounds down to 400)
COLS = ['obj_id_unique', 'time stamp', 'x', 'y', 'xvel', 'yvel']
CLASSES = ['casting', 'circling', 'random_turns', 'brownian', 'levy_flight', 'run_and_tumble']   

RANDOM_SEED_OFFSET = 100000                                 


class SingleFlashPlume(opto_tracking.BoolPlume):
    def get_odor(self, condition, t):
        if getattr(self, 'done', False):
            return 0
        odor = super().get_odor(condition, t)
        if condition != 'initial' and not odor:
            self.done = True
        return odor


def opto_setup():
    opto_tracking.BoolPlume = SingleFlashPlume
    opto_tracking.SurgingAgent.baseline_after = 3.5


ORIG_CAST_AND_SURGE = opto_tracking.SurgingAgent.cast_and_surge


def cast_and_surge_clock(self, plume, wind, start_pos, start_vel, duration, dt, bias_sign, constant_noise, rv):
    n_steps = int(duration / dt)
    ts = np.arange(n_steps) * dt
    self.time = ts
    half = np.clip(np.random.normal(self.half_mean, self.half_sd, 40), 0.2, 1.5)
    switches = np.cumsum(half) - np.random.uniform(0, half[0])
    clock = np.where(np.searchsorted(switches, ts) % 2 == 0, 1.0, -1.0)
    slow_x = np.zeros(n_steps)                             # slow along-wind (x) wander added to the noise
    a = np.exp(-dt / self.slow_x[1])
    for i in range(1, n_steps):
        slow_x[i] = a * slow_x[i - 1] + np.sqrt(1 - a ** 2) * self.slow_x[0] * np.random.normal()
    bs = np.nan * np.zeros((n_steps, 3))
    surges = np.zeros(n_steps)
    b_convo = np.zeros(n_steps)
    vs = np.nan * np.zeros((n_steps, 3))
    xs = np.nan * np.zeros((n_steps, 3))
    odors = np.nan * np.zeros((n_steps,))
    upwind = np.zeros((n_steps, 3))
    crosswind = np.zeros((n_steps, 3))
    real_wind = np.zeros((n_steps, 3))
    for t in range(n_steps):
        if t == 0:
            b = np.array([0, start_pos[1], start_pos[2]])
            b *= (self.bias / np.linalg.norm(b))
            v = start_vel.copy()
            x = start_pos.copy()
            d_upwind = np.array([-.4, 0, 0])
        else:
            eta = np.random.normal(0, self.noise, (3,)) + np.array([slow_x[t], 0, 0])
            d_upwind = (upwind[t] - upwind[t - 1]) / np.linalg.norm(upwind[t] - upwind[t - 1])
            d_cross = (crosswind[t] - crosswind[t - 1]) / np.linalg.norm(crosswind[t] - crosswind[t - 1])
            b = self.bias * b_convo[t] * d_cross * bias_sign
            if np.all(odors[:t] == 0) or self.odor == 1:
                b = 0
            if (len(np.where(odors == 1)[0]) != 0) and (np.abs(ts[t] - np.where(odors == 1)[0][-1] * dt) >
                                                        self.baseline_after):
                b = 0
            v += (dt / self.tau) * (-v + eta + b + surges[t] * d_upwind)
            x += v * dt
        v, x = self.reflect_if_out_of_bounds(v, x)
        if self.odor == False:
            if plume.check_bounds(x):
                self.odor = plume.get_odor('initial', ts[t])
        bs[t] = b
        vs[t] = v
        xs[t] = x
        if wind.vector_field is not None:
            wind.get_correct_wind_vec(agent_position=xs[t], t=t, ts=ts[t])
        crosswind[t], upwind[t], real_wind[t] = wind.relative_upwind(vs[t], t=t)
        odors[t] = self.odor
        if self.odor == True:
            c = 30
            ts_ = ts[t + c:] - ts[t + c]
            surges[t + c:] += (self.surge_amp_ * ts_ * np.exp(-ts_ / self.tau_surge))
            b_convo[:t] = 0
            self.odor = plume.get_odor(self.odor, ts[t])
        if self.odor == False:
            b_convo[t:] = clock[t]
    return {'surges': surges, 'bs': bs, 'vs': vs, 'xs': xs, 'odors': odors, 'ts': ts, 'real_wind': real_wind,
            'b_convo': b_convo}


def use_clock(half_mean, half_sd, slow_x_sd=0.0, slow_x_tau=0.5):
    A = opto_tracking.SurgingAgent
    A.cast_and_surge = cast_and_surge_clock
    A.half_mean, A.half_sd, A.slow_x = half_mean, half_sd, (slow_x_sd, slow_x_tau)


def sink_and_circle_mod(self, plume, wind, start_pos, start_vel, duration, dt, bias_sign, reject_thresh, constant_noise,
                        radius, omega, rv):
    n_steps = int(duration / dt)
    ts = np.arange(n_steps) * dt
    vs = np.nan * np.zeros((n_steps, 3))
    xs = np.nan * np.zeros((n_steps, 3))
    odors = np.nan * np.zeros((n_steps,))
    real_wind = np.zeros((n_steps, 3))
    upwind = np.zeros((n_steps, 3))
    r = np.random.normal(radius, self.radius_sd)
    w = np.random.normal(omega, self.omega_sd)
    mod, mw = np.zeros(n_steps), np.zeros(n_steps)
    a = np.exp(-dt / self.amp_mod[1])
    aw = np.exp(-dt / self.omega_mod[1])
    for i in range(1, n_steps):
        mod[i] = a * mod[i - 1] + np.sqrt(1 - a ** 2) * self.amp_mod[0] * np.random.normal()
        mw[i] = aw * mw[i - 1] + np.sqrt(1 - aw ** 2) * self.omega_mod[0] * np.random.normal()
    phase = np.cumsum(w * np.exp(mw - self.omega_mod[0] ** 2 / 2)) * dt    # mean rate w
    if getattr(self, 'random_phase', False):                            # circle from a random phase, not cos(0)
        phase = phase + np.random.uniform(0, 2 * np.pi)
    mod = mod - self.omega_mod[2] * mw                                     # slower while turning faster
    for t in range(n_steps):
        if t == 0:
            v = start_vel.copy()
            x = start_pos.copy()
        else:
            eta = np.random.normal(0, self.noise, (3,))
            turn = bias_sign if self.both_ways else 1          # both_ways: circle left / right with bias_sign
            b = np.exp(mod[t]) * r * np.array([np.cos(phase[t]), turn * np.sin(phase[t]), 0]) * bias_sign
            reject = 0
            ended = reject_thresh and not np.all(odors[:t] == 0) and self.odor != 1     # after the flash
            if ended and np.linalg.norm(upwind[t - 1]) > 0:
                reject = reject_thresh * -upwind[t - 1] / np.linalg.norm(upwind[t - 1])
            v += (dt / self.tau) * (-v + eta + self.bias * b - reject)
            x += v * dt
        v, x = self.reflect_if_out_of_bounds(v, x)
        if self.odor == False and plume.check_bounds(x):
            self.odor = plume.get_odor('initial', ts[t])
        vs[t] = v
        xs[t] = x
        if wind.vector_field is not None:
            wind.get_correct_wind_vec(agent_position=xs[t], t=t, ts=ts[t])
        _, upwind[t], real_wind[t] = wind.relative_upwind(vs[t], t=t)
        odors[t] = self.odor
        if self.odor == True:
            self.odor = plume.get_odor(self.odor, ts[t])
    return {'surges': np.zeros(n_steps), 'bs': np.zeros((n_steps, 3)), 'vs': vs, 'xs': xs, 'odors': odors, 'ts': ts,
            'real_wind': real_wind}


def use_amp_mod(sd, tau, omega_sd=1.5, w_sd=0.0, w_tau=0.3, coupling=0.0, random_phase=False):
    A = opto_tracking.SurgingAgent
    A.sink_and_circle = sink_and_circle_mod
    A.amp_mod, A.omega_sd, A.omega_mod = (sd, tau), omega_sd, (w_sd, w_tau, coupling)
    A.random_phase = random_phase


def saccadic_circling(rng, T=4.01, dt=0.01, speed=(0.33, 0.06), speed_mod=0.18, speed_tau=0.4, sacc_slow=0.45,
                      straight=(0.21, 0.35), turn_deg=(55, 15), sacc_dur=0.08, drift=1.0, drift_sd=1.5, drift_tau=0.15):
    n = int(T / dt)
    sign = rng.choice([-1, 1])
    m, cv = straight
    w = np.zeros(n)
    t = rng.uniform(0, m)                                  # random phase of the first saccade
    k = max(int(round(sacc_dur / dt)), 1)
    prof = 1 - np.cos(2 * np.pi * (np.arange(k) + 0.5) / k)
    slow = np.zeros(n)
    while t < T:
        i = int(t / dt)
        ang = np.deg2rad(max(rng.normal(*turn_deg), 5))
        seg = slice(i, min(i + k, n))
        w[seg] += sign * ang * prof[:seg.stop - i] / (prof.sum() * dt)
        slow[seg] = np.maximum(slow[seg], prof[:seg.stop - i] / 2)
        t += sacc_dur + (rng.gamma(1 / cv ** 2, m * cv ** 2) if cv > 0 else m)
    ou = np.zeros(n)
    a = np.exp(-dt / drift_tau)
    for i in range(1, n):
        ou[i] = a * ou[i - 1] + np.sqrt(1 - a ** 2) * drift_sd * rng.normal()
    w += sign * drift + ou
    heading = rng.uniform(-np.pi, np.pi) + np.cumsum(w) * dt
    mod = np.zeros(n)
    a = np.exp(-dt / speed_tau)
    for i in range(1, n):
        mod[i] = a * mod[i - 1] + np.sqrt(1 - a ** 2) * speed_mod * rng.normal()
    s = abs(rng.normal(*speed)) * np.exp(mod) * (1 - sacc_slow * slow)
    vx, vy = s * np.cos(heading), s * np.sin(heading)
    return np.cumsum(vx) * dt, np.cumsum(vy) * dt, vx, vy


def levy_walk(rng, speed, T=4.0, dt=0.01, mu=2.0, l_min=0.01, turn_s=0.05, vel_noise=0.02):
    rng.standard_normal(3)                                  

    def segment():                                          
        length = l_min * (1 - rng.uniform()) ** (-1 / (mu - 1))
        rng.standard_normal()
        return length

    n = int(round(T / dt)) + 1
    pos, vel = np.zeros((n, 2)), np.zeros((n, 2))
    heading = rng.uniform(0, 2 * np.pi)
    remaining = segment()
    turn_steps = max(1, int(round(turn_s / dt)))
    turn_left, turn_rate = 0, 0.0
    for i in range(1, n):
        if turn_left == 0 and remaining <= 0:               
            turn_left, turn_rate = turn_steps, rng.uniform(-np.pi, np.pi) / turn_steps
            remaining = segment()
        if turn_left > 0:
            heading += turn_rate
            turn_left -= 1
        v = speed * np.array([np.cos(heading), np.sin(heading)]) + rng.normal(0, vel_noise, 2)
        pos[i], vel[i] = pos[i - 1] + v * dt, v
        if turn_left == 0:                                  
            remaining -= np.hypot(*v) * dt
    return np.arange(n) * dt, pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1]


def off_walls(x, y):
    return np.max(np.abs(x)) < 0.499 and np.max(np.abs(y)) < 0.249


def use_agent(agent):
    A = opto_tracking.SurgingAgent
    if agent == 'casting':                                  
        A.cast_and_surge = ORIG_CAST_AND_SURGE
        return dict(BEHAVIOR='cast_and_surge', TAU=0.25, TAU_SURGE=0.02, NOISE=1, BIAS=0.7, SURGE_AMP=0.42)
    if agent == 'casting_t':                                
        use_clock(half_mean=0.6, half_sd=0.2, slow_x_sd=0.25, slow_x_tau=0.5)
        return dict(BEHAVIOR='cast_and_surge', TAU=0.25, TAU_SURGE=0.02, NOISE=0.5, BIAS=0.7, SURGE_AMP=0.42)
    use_amp_mod(sd=0.3, tau=0.4, random_phase=True)         
    A.both_ways = True
    return dict(BEHAVIOR='sink_and_circle', TAU=0.25, TAU_SURGE=0.25, NOISE=0.3, BIAS=1.8, SURGE_AMP=0, RADIUS=0.3,
                OMEGA=4.05)


def opto_batch(args):
    params, seed, tag = args
    with contextlib.redirect_stdout(io.StringIO()):
        res = opto_tracking.run_sims(SEED=seed, N_TRAJS=500, **params, wind_condition='constant', DURATION=T + 0.21,
                                     DT=0.01, REJECT_THRESH=0, CONSTANT_NOISE=True,
                                     BOUNDS=[(-0.5, 0.5), (-0.25, 0.25), (0, 0.5)], wind_data=None, vector_field=None,
                                     wind_positions=None, wind_data_type='CFD')
        df = opto_tracking.convert_and_save_sims(res, '', f'{tag}_seed{seed}', remove_wall_hits=False, save=False)
    ts = np.arange(len(res[0]['ts'])) * 10 - 200
    keep = [f'{tag}_seed{seed}_{i}' for i, r in enumerate(res) if off_walls(r['xs'][:, 0], r['xs'][:, 1])]
    onsets = [int((np.diff(np.r_[0, np.nan_to_num(r['odors'])]) == 1).sum()) for r in res]
    late_surge = [bool((r['surges'][ts > 1500] > 0.01).any()) for r in res]
    late_bias = [bool((np.abs(np.nan_to_num(r['bs'][ts >= 3500])) > 0).any()) for r in res]
    return df[df.obj_id_unique.isin(keep)][COLS], onsets, late_surge, late_bias


def opto(n, agent, seed0, tag, workers):
    if n == 0:
        return pd.DataFrame(columns=COLS)
    params = use_agent(agent)
    dfs, checks, got, seed = [], [], 0, seed0
    with mp.get_context('fork').Pool(workers) as pool:
        while got < n:
            for df, *c in pool.map(opto_batch, [(params, s, tag) for s in range(seed, seed + 8)]):
                dfs.append(df)
                checks += list(zip(*c))
                got += df.obj_id_unique.nunique()
            seed += 8
    on, late_surge, late_bias = np.array(checks).T
    assert (on == 1).all() and not late_surge.any(), f'{tag}: {np.unique(on)} flash onsets, {late_surge.sum()} late surges'
    if agent.startswith('casting'):
        assert late_bias.all(), f'{tag}: casting bias off before 4 s in {(~late_bias.astype(bool)).sum()} agents'
    df = pd.concat(dfs, ignore_index=True)
    return df[df.obj_id_unique.isin(df.obj_id_unique.drop_duplicates().iloc[:n])]


def frame(oid, x, y, vx, vy, t_ms):
    return pd.DataFrame({'obj_id_unique': oid, 'time stamp': t_ms, 'x': x, 'y': y, 'xvel': vx, 'yvel': vy})


def offset(seed):
    return (0, '') if seed is None else (RANDOM_SEED_OFFSET * (seed + 1), f'_rs{RANDOM_SEED_OFFSET * (seed + 1)}')


def coin(seed, algorithm, n, k=0):
    return np.random.default_rng([seed, CLASSES.index(algorithm), k]).random(n) < 0.5


def casting(n=10000, seed=None, workers=N_WORKERS):
    k = n // 2 if seed is None else int(coin(seed, 'casting', n).sum())
    off, rs = offset(seed)
    out = []
    for constant_period, count in [(True, k), (False, n - k)]:
        if constant_period:
            out.append(('casting_original', opto(count, 'casting', 2177 + off, f'cast_and_surge_constantwind_long{rs}', workers)))
        else:
            out.append(('casting_t', opto(count, 'casting_t', 3001 + off, f'cast_and_surge_tuned_long{rs}', workers)))
    return out


def saccadic(n, variable_straights, drift, seed=None):
    kw = dict(sacc_dur=0.15, sacc_slow=0.25, drift=0.6, drift_sd=1.8, speed_mod=0.12, speed=(0.4, 0.1),
              straight=(0.15, 0), turn_deg=(60, 20))
    if variable_straights:
        kw.update(sacc_dur=0.16, straight=(0.10, 0.8))
    if not drift:
        kw.update(drift=0, drift_sd=0)
    base = 332 if variable_straights else 331
    off, rs = offset(seed)
    name = f'{"saccvar" if variable_straights else "sacc300"}_t7{"" if drift else "_clean"}_long{rs}'
    if seed is None:
        rng, out, i = np.random.default_rng(base), [], 0
        while len(out) < n:
            if (i % 2 == 0) == drift:                       # even: drift, odd: clean
                out.append(frame(f'{name}_{i}', *saccadic_circling(rng, T=TT, **kw), np.arange(int(TT / 0.01)) * 10.))
            else:
                saccadic_circling(rng, T=TT, **kw)          # the other variant's draws (same rng use with or without drift)
            i += 1
    else:
        rng = np.random.default_rng([base, off, int(drift)])
        out = [frame(f'{name}_{i}', *saccadic_circling(rng, T=TT, **kw), np.arange(int(TT / 0.01)) * 10.)
               for i in range(n)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLS)


def circling(n=10000, seed=None, workers=N_WORKERS):
    if seed is None:
        k = {'smooth': n // 2, **{(v, d): n // 8 for v in (False, True) for d in (False, True)}}
    else:
        smooth, var, drift = (coin(seed, 'circling', n, k) for k in range(3))
        k = {'smooth': int(smooth.sum()),
             **{(v, d): int((~smooth & (var == v) & (drift == d)).sum()) for v in (False, True) for d in (False, True)}}
    off, rs = offset(seed)
    out = [('circling_smooth', opto(k['smooth'], 'smooth', 4301 + off, f'sink_and_circle_t4p_long{rs}', workers))]
    for v in (False, True):
        for d in (False, True):
            out.append((f'circling_{"saccvar" if v else "sacc300"}_{"drift" if d else "clean"}', saccadic(k[(v, d)], v, d, seed)))
    return out


def random_turns(n=10000, seed=None, workers=None):
    off, rs = offset(seed)
    np.random.seed(84 + off)
    out = []
    for i in range(n):
        _, x, y, vx, vy, _ = SA.generate_random_turn_trajectory(total_time=TT, straight_duration=0.2, direction=None,
                                                                meanspeed=0.4, random_phase=True)
        out.append(frame(f'random_150ms_to_600ms_constant_turns_phase_nr_long{rs}_{i}', x, y, vx, vy,
                         np.arange(len(x)) * 10.))  
    return [('random_turns', pd.concat(out, ignore_index=True))]


def brownian(n=10000, seed=None, workers=None):
    off, rs = offset(seed)
    np.random.seed(1234 + off)
    return [('brownian', pd.concat([SA.brownian_motion_sim(0.01, T, i).assign(obj_id_unique=f'brownian_nr_long{rs}_{i}')[COLS]
                                    for i in range(n)], ignore_index=True))]


def levy_flight(n=10000, seed=None, workers=None):
    off, rs = offset(seed)
    rng = np.random.default_rng(83 + off)
    out = []
    for i in range(n):
        t, x, y, vx, vy = levy_walk(rng, max(0.01, abs(rng.normal(0.4, 0.15))), T=T)
        out.append(frame(f'levy_optnw415_long{rs}_{i}', x, y, vx, vy, np.round(t * 1000, 6)))
    return [('levy_flight', pd.concat(out, ignore_index=True))]


def run_and_tumble(n=10000, seed=None, workers=None):
    off, rs = offset(seed)
    np.random.seed(42 + off)
    out = []
    for i in range(n):
        speed = abs(np.random.normal(0.4, 0.15))
        time, pos, _ = SA.run_and_tumble(total_time=T, dt=0.01, run_time=.05, tumble_time=.5, speed=speed)
        vel = np.gradient(pos, 0.01, axis=0)
        k = speed / np.hypot(vel[:, 0], vel[:, 1]).mean()
        out.append(frame(f'run_and_tumble_cd415_nr_long{rs}_{i}', *(pos * k).T, *(vel * k).T, time * 1000))
    return [('run_and_tumble', pd.concat(out, ignore_index=True))]


ALGORITHMS = [casting, circling, random_turns, brownian, levy_flight, run_and_tumble]     # class_id order


def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]


def training_set(parts):
    # {generator: (class, trajectories)} (generator order) -> the synthetic_training_simple.parquet table: generic ids
    # <class>_1 ... <class>_n (numbered in generator, then simulation id order), class, time stamp, x, y, xvel, yvel
    df = pd.concat([d.assign(generator=g) for g, (_, d) in parts.items() if len(d)], ignore_index=True)
    df['time stamp'] = df['time stamp'].round(3)
    info = df.groupby('obj_id_unique', sort=True).generator.first().rename_axis('simulation_id').reset_index()
    info['class'] = info.generator.map({g: c for g, (c, _) in parts.items()})
    gen = list(parts)
    info = info.iloc[sorted(range(len(info)), key=lambda i: (CLASSES.index(info['class'][i]), gen.index(info.generator[i]),
                                                              natural(info.simulation_id[i])))]
    info['obj_id_unique'] = [f'{c}_{k + 1}' for c, k in zip(info['class'], info.groupby('class').cumcount())]
    out = df.rename(columns={'obj_id_unique': 'simulation_id'}).merge(info[['simulation_id', 'obj_id_unique', 'class']],
                                                                       on='simulation_id')
    out = out[['obj_id_unique', 'class', 'time stamp', 'x', 'y', 'xvel', 'yvel']]
    rank = {o: i for i, o in enumerate(sorted(info.obj_id_unique, key=natural))}   # casting_2 before casting_10
    out = out.sort_values(['obj_id_unique', 'time stamp'], key=lambda s: s.map(rank) if s.name == 'obj_id_unique' else s)
    out['class'] = out['class'].astype('category')
    return out.reset_index(drop=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
    out = os.path.abspath(opt('--out', OUT))
    assert os.path.basename(out) != 'synthetic_training_simple.parquet', 'would overwrite the shipped training set'
    workers, n, seed = opt('--workers', N_WORKERS), opt('--n', 10000), opt('--seed', 0)
    seed = None if '--published' in args else seed
    opto_setup()
    parts = {}
    for algorithm in ALGORITHMS:
        for generator, d in algorithm(n, seed, workers):
            parts[generator] = (algorithm.__name__, d)
            print(f'{generator}: {d.obj_id_unique.nunique()} trajectories', flush=True)
    df = training_set(parts)
    span = df.groupby('obj_id_unique', observed=True)['time stamp'].agg(['min', 'max'])
    assert (span['max'] >= 4000).all() and (span['min'] <= 1100).all(), 'trajectories shorter than 4 s'
    df.to_parquet(out)
    print(df.groupby('class', observed=True).obj_id_unique.nunique().to_string())
    print(out)
