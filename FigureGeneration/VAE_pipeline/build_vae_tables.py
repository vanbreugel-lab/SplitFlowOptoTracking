"""Build the VAE data tables of Data/Fig4_vae_data/ from the source data in Data/.

Usage (from this directory): python build_vae_tables.py
Writes (Data/Fig4_vae_data/): real_training_rebuilt.parquet, real_model_test_sets_rebuilt.parquet,
synthetic_model_test_sets_rebuilt.parquet
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from vae_utils import (data_path, FEATS, WINDOWS, N_T, REAL_SOURCES, REAL_CLASSES, SYN_CLASSES, VARWIND_SPEED, fill_gaps,
                       windows, natural, balanced_split)

HERE = os.path.dirname(os.path.abspath(__file__))
TOP_ON = 'fly.all_nights_top_laminar'                      # keys in Data/data_paths.yaml
BOTTOM_ON = 'fly.all_nights_bottom_laminar'
COLS = ['obj_id_unique', 'time stamp', 'x', 'y', 'z', 'xvel', 'yvel']
CFD = {'CFD casting, steady': 'steady casting', 'CFD casting, unsteady': 'unsteady casting',
       'CFD circling, steady': 'steady circling', 'CFD circling, unsteady': 'unsteady circling'}
UNIFYING = {'Unifying, laminar': 'laminar', 'Unifying, still air': 'stillair', 'Unifying, unsteady': 'unsteady'}
OUT_COLS = {
    'real_training': ['obj_id_unique', 'class', 'source', 'split', 'time stamp', 'x', 'y', 'z', 'xvel', 'yvel'],
    'real_model_test_sets': ['obj_id_unique', 'section', 'test_set', 'time stamp', 'x', 'y', 'z', 'xvel', 'yvel', 'class',
                             'cfd_velocity_field'],
    'synthetic_model_test_sets': ['obj_id_unique', 'section', 'test_set', 'time stamp', 'x', 'y', 'xvel', 'yvel', 'z',
                                  'cfd_velocity_field']}


# ---------------------------------------------------------------- source data
def read(key):
    path = str(data_path(key))
    return pd.read_parquet(path) if path.endswith('.parquet') else pd.read_hdf(path)


def events(df):
    # one trajectory per flash event, time stamp in ms from flash onset; mean ground speed > 0.15 m/s
    df = df.assign(**{'time stamp': (df['time_relative_to_flash'] * 1000).round(), 'obj_id_unique': df.obj_id_unique_event})
    speed = np.hypot(df.xvel, df.yvel).groupby(df.obj_id_unique).transform('mean')
    return df[speed > 0.15]


def is_sham(d):
    # sham: light intensity 0 (duration 0 in the WT split-flow file)
    return (d.duration == 0) | (d.intensity == 0)


def event_file(kind, genotype):
    # kind: stillair, laminar or splitflow_topon; genotype: c1xwt (CsChrimson) or wt (wild type)
    return events(read(f'fly.{kind}_{genotype}'))


def varwind():
    # (fan %, 'flash' | 'sham') -> variable-wind CsChrimson events
    df = events(read('fly.varwind'))
    fan = df.groupby('obj_id_unique')['fan_speed_percent'].transform('first')
    kind = np.where(df.intensity > 0, 'flash', 'sham')
    return {(int(f), k): d for (f, k), d in df.groupby([fan, kind])}


def split_flow(key):
    # flash / sham trajectories tracked >= 3.0 s after the flash with mean ground speed > 0.15 m/s; lower region = whole
    # trajectory below z = 0.25 m, upper region = whole trajectory above
    df = pd.read_hdf(data_path(key))
    ok = lambda d: d.groupby('obj_id_unique').filter(
        lambda g: np.hypot(g.xvel, g.yvel).mean() > 0.15 and g['time stamp'].max() >= 3000)
    low = lambda d: d.groupby('obj_id_unique').filter(lambda g: g.z.max() < 0.25)
    up = lambda d: d.groupby('obj_id_unique').filter(lambda g: g.z.min() > 0.25)
    flash, sham = ok(df[df.duration > 0]), ok(df[df.duration == 0])
    return low(flash), low(sham), up(sham)


def split_flow_groups():
    # unsteady flow: lower region of the top-on file; steady flow: lower region of the bottom-on file and upper region of
    # the top-on file. Flash groups: lower region only
    flash_top, sham_low_top, sham_up_top = split_flow(TOP_ON)
    flash_bottom, sham_low_bottom, sham_up_bottom = split_flow(BOTTOM_ON)
    return {'Unsteady, flash': flash_top, 'Steady, flash': flash_bottom,
            'Unsteady, sham': pd.concat([sham_low_top, sham_up_bottom]),
            'Steady, sham': pd.concat([sham_low_bottom, sham_up_top]),
            'Unsteady sham': sham_low_top, 'Steady sham': sham_low_bottom}      # real model: same file and height as flash


def scored(df):
    # ids with all 191 samples of the 1.1-3.0 s window
    t = df['time stamp'].round(3)
    n = df[t.between(*WINDOWS[0])].groupby('obj_id_unique').size()
    return n.index[n == N_T].to_numpy()


def complete(df):
    # real-fly model: ids whose 1.1-3.0 s window is complete after interpolating gaps < 15 frames
    ids = np.array(sorted(df.obj_id_unique.unique()))
    return ids[windows(fill_gaps(df[['obj_id_unique', 'time stamp'] + FEATS]), ids, WINDOWS[:1])[1][:, 0]]


def part(df, ids, cols=COLS, **labels):
    d = df[df.obj_id_unique.isin(set(ids))]
    return d[[c for c in cols if c in d]].assign(**labels)


def renamed(df, prefix):
    # simulated trajectories renamed <prefix>_1 ... in natural order of their ids
    names = {o: f'{prefix}_{i + 1}' for i, o in enumerate(sorted(df.obj_id_unique.unique(), key=natural))}
    return df.assign(obj_id_unique=df.obj_id_unique.map(names))


def table(sets):
    return pd.concat([d.assign(section=s, test_set=n) for s, n, d in sets], ignore_index=True)


def save(df, name):
    path = str(data_path('vae.rebuilt', name=name, must_exist=False))
    for c in ['class', 'split', 'section', 'test_set', 'source']:
        if c in df:
            df[c] = df[c].astype('category')
    rank = {o: i for i, o in enumerate(sorted(df.obj_id_unique.unique(), key=natural))}
    df = df.sort_values(['obj_id_unique', 'time stamp'], key=lambda s: s.map(rank) if s.name == 'obj_id_unique' else s)
    df[[c for c in OUT_COLS[name] if c in df]].reset_index(drop=True).to_parquet(path)
    print(f'{os.path.basename(path)}: {df.obj_id_unique.nunique()} trajectories, {len(df)} rows', flush=True)


# ---------------------------------------------------------------- simulations
def cfd_and_unifying():
    # CFD casting / circling test sets (300 each, ids <behaviour>_<n>_<vel_k>_<flow>; generate_cfd_data.py) and the
    # unifying-algorithm simulations
    cfd = pd.read_parquet(data_path('vae.cfd_test_simple')).astype({'class': str})
    cfd['dataset_name'] = cfd.flow.astype(str) + ' ' + cfd['class']
    cfd['cfd_velocity_field'] = cfd.obj_id_unique.str.extract(r'(vel_\d)', expand=False)
    uni = pd.read_parquet(data_path('sim.unifying'))
    uni['time stamp'] = (uni['time_relative_to_flash'] * 1000).round()
    uni['obj_id_unique'] = uni.obj_id_unique.map(lambda s: s.decode() if isinstance(s, bytes) else s)
    sets = []
    for name, src in CFD.items():
        d = cfd[cfd.dataset_name == src]
        sets.append(('CFD simulations', name, part(d, scored(d), COLS + ['cfd_velocity_field'])))
    for name, wt in UNIFYING.items():
        d = uni[uni.windtype == wt]
        sets.append(('unifying algorithm', name, renamed(part(d, scored(d)), f'unifying_{wt}')))
    return sets


def synthetic_held_out():
    # held-out 20% of the synthetic training set: train_test_split (stratified by class, random_state 42) over the
    # trajectories in sorted order of their simulation ids (generate_synthetic_data.py --published simulations), renamed
    # to the generic ids as generate_synthetic_data.training_set. Returns the held-out generic ids
    import generate_synthetic_data as GS
    GS.opto_setup()
    sims = pd.concat([d[['obj_id_unique']].drop_duplicates().assign(generator=g, cls=a.__name__)
                      for a in GS.ALGORITHMS for g, d in a(10000, None, GS.N_WORKERS)], ignore_index=True)
    gen = list(dict.fromkeys(sims.generator))
    sims = sims.iloc[sorted(range(len(sims)), key=lambda i: (SYN_CLASSES.index(sims.cls[i]), gen.index(sims.generator[i]),
                                                              natural(sims.obj_id_unique[i])))]
    sims['generic'] = sims.cls + '_' + (sims.groupby('cls').cumcount() + 1).astype(str)
    sims = sims.sort_values('obj_id_unique')
    _, held = train_test_split(np.arange(len(sims)), test_size=0.2, random_state=42,
                               stratify=sims.cls.map(SYN_CLASSES.index))
    return sims.iloc[held].generic.values


# ---------------------------------------------------------------- tables
def real_flies():
    # training flies of the real-fly model (every fly with a complete 1.1-3.0 s window) and the seed-42 split
    still, lam = event_file('stillair', 'c1xwt'), event_file('laminar', 'c1xwt')
    raw = {'Event still air flash': still[~is_sham(still)], 'Event laminar flash': lam[~is_sham(lam)],
           'Event still air sham': still[is_sham(still)], 'Event laminar sham': lam[is_sham(lam)],
           **{f'Varwind {f}% {k}': d for (f, k), d in varwind().items()}}
    info = pd.concat([pd.DataFrame({'obj_id_unique': complete(raw[s]), 'class': c, 'source': s})
                      for c in REAL_CLASSES for s in REAL_SOURCES[c]], ignore_index=True)
    tr, va = balanced_split(info['class'].to_numpy(), 42)
    info['split'] = 'unused (class balancing)'
    info.loc[tr, 'split'], info.loc[va, 'split'] = 'train', 'held-out test'
    return info, raw


def real_training(info, raw):
    return pd.concat([part(raw[s], g.obj_id_unique).merge(g, on='obj_id_unique') for s, g in info.groupby('source')])


def real_tests(info, raw, sf, sims):
    sets = []
    for c in REAL_CLASSES:
        g = info[(info.split == 'held-out test') & (info['class'] == c)]
        sets.append(('validation set', c, pd.concat([part(raw[s], gg.obj_id_unique, **{'class': c})
                                                     for s, gg in g.groupby('source')])))
    for name, g in [('Unsteady flash', 'Unsteady, flash'), ('Steady flash', 'Steady, flash'),
                    ('Unsteady sham', 'Unsteady sham'), ('Steady sham', 'Steady sham')]:
        d = sf[g]
        sets.append(('split flow', name, part(d, complete(d))))
    train = set(info.obj_id_unique[info.split == 'train'])
    for kind in ['flash', 'sham']:
        for fan, speed in VARWIND_SPEED.items():
            d = raw[f'Varwind {fan}% {kind}']
            ids = [i for i in complete(d) if i not in train]       # 0, 0.3, 0.6 m/s: only flies outside the training 80%
            sets.append((f'variable wind speed, {kind}', f'{speed:g} m/s {kind}', part(d, ids)))
    held = synthetic_held_out()
    syn = pd.read_parquet(data_path('vae.synthetic_training'))
    syn = syn[syn.obj_id_unique.isin(held)].astype({'class': str})
    sets += [('synthetic algorithms', c, syn[syn['class'] == c]) for c in SYN_CLASSES]
    return table(sets + sims)


def synthetic_tests(sf, sims):
    still, lam = event_file('stillair', 'c1xwt'), event_file('laminar', 'c1xwt')
    wt_still, wt_lam, wt_split = (event_file(k, 'wt') for k in ('stillair', 'laminar', 'splitflow_topon'))
    vw = varwind()
    unsteady = scored(sf['Unsteady, flash'])          # Fig. 4 unsteady flash: 99% part of a 99/1 split
    unsteady = unsteady[train_test_split(np.arange(len(unsteady)), test_size=0.01, random_state=44,
                                         stratify=np.zeros(len(unsteady)))[0]]
    sets = [('flash', 'Still air, flash', read('fly.training_stillair')[COLS]),   # validation flies
            ('flash', 'Unsteady, flash', part(sf['Unsteady, flash'], unsteady)),
            ('flash', 'Steady, flash', part(sf['Steady, flash'], scored(sf['Steady, flash']))),
            ('flash', 'Laminar, flash', read('fly.training_laminar')[COLS])]
    for name, d in [('Still air, sham', still[is_sham(still)]), ('Unsteady, sham', sf['Unsteady, sham']),
                    ('Steady, sham', sf['Steady, sham']), ('Laminar, sham', lam[is_sham(lam)])]:
        sets.append(('sham', name, part(d, scored(d))))
    for kind in ['flash', 'sham']:
        for fan, speed in VARWIND_SPEED.items():
            d = vw[(fan, kind)]
            sets.append((f'variable wind, {kind}', f'Var. wind {speed:g} m/s, {kind}', part(d, scored(d))))
    # wild type: flash and sham pooled; unsteady = trajectory enters z < 0.25 m, steady = whole trajectory above
    zmin = wt_split.groupby('obj_id_unique').z.transform('min')
    for name, d in [('WT still air', wt_still), ('WT unsteady', wt_split[zmin < 0.25]),
                    ('WT steady', wt_split[zmin > 0.25]), ('WT laminar', wt_lam)]:
        sets.append(('wild type', name, part(d, scored(d))))
    return table(sets + sims)


if __name__ == '__main__':
    sf, sims = split_flow_groups(), cfd_and_unifying()
    info, raw = real_flies()
    print(info.groupby(['class', 'split']).size().to_string(), flush=True)
    save(real_training(info, raw), 'real_training')
    save(real_tests(info, raw, sf, sims), 'real_model_test_sets')
    save(synthetic_tests(sf, sims), 'synthetic_model_test_sets')
