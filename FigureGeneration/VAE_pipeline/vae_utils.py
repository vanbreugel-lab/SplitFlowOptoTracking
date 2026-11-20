"""
Shared code of the autoencoder figures (Fig. 4, SI Figs. S6-S8) and of train_vae.py.
"""
import os
import re
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import splitflow.TripletLoss as CCL

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'Data', 'final_vae_data')
CACHE = os.path.join(HERE, 'cache')
MODELS = {'synthetic': 'synthetic_sliding_vae_seed42.pth', 'real': 'real_3class_sliding_vae_seed42.pth'}

FEATS = ['x', 'y', 'xvel', 'yvel']
N_T = 191
WINDOWS = [(1100, 3000), (1600, 3500), (2100, 4000)]           
MAX_GAP = 15                                                   
THRESHOLD = 0.6


SYN_CLASSES = ['casting', 'circling', 'random_turns', 'brownian', 'levy_flight', 'run_and_tumble']
LABELS = {0: 'Casting', 1: 'Circling', 2: 'Random turns', 3: 'Brownian', 4: 'Levy flight', 5: 'Run & tumble'}
CLASS_COLORS = ['orangered', 'darkcyan', 'orchid', '#dfa45cff', '#a6cd4eff', 'darkviolet']
OOD = 'OOD/Rejected'
ROWS = {'Casting': 'orangered', 'Circling': 'darkcyan', 'Random turns': 'orchid', 'Levy flight': '#a6cd4eff',
        'Run & tumble': 'darkviolet', 'Brownian': '#dfa45cff', OOD: 'k'}   

REAL_SOURCES = {'Still air flash': ['Event still air flash', 'Varwind 12% flash'],
                'Laminar flash': ['Event laminar flash', 'Varwind 40% flash', 'Varwind 90% flash'],
                'Sham': ['Event still air sham', 'Varwind 12% sham', 'Event laminar sham', 'Varwind 40% sham',
                         'Varwind 90% sham']}
REAL_CLASSES = list(REAL_SOURCES)
VARWIND_SPEED = {12: 0.0, 20: 0.03, 23: 0.1, 28: 0.2, 40: 0.3, 90: 0.6}     # variable-wind fan setting (%) -> m/s


def fill_gaps(df):
    df = df.sort_values(['obj_id_unique', 'time stamp']).reset_index(drop=True)
    for c in FEATS:
        na = df[c].isna()
        if not na.any():
            continue
        run = ((na != na.shift()) | (df.obj_id_unique != df.obj_id_unique.shift())).cumsum()
        short = na & (na.groupby(run).transform('sum') < MAX_GAP)
        interp = df.groupby('obj_id_unique')[c].transform(lambda s: s.interpolate(limit_area='inside'))
        df.loc[short, c] = interp[short]
    return df


def windows(df, ids, wins=WINDOWS):
    pos = pd.Series(np.arange(len(ids)), index=pd.Index(ids))
    Xs, ok = [], []
    for t0, t1 in wins:
        d = df[df['time stamp'].round(3).between(t0, t1) & df.obj_id_unique.isin(pos.index)]
        r = pos[d.obj_id_unique].values
        k = np.round((d['time stamp'].values - t0) / 10).astype(int)
        X = np.full((len(ids), N_T, len(FEATS)), np.nan, np.float32)
        X[r, k] = d[FEATS].values
        Xs.append(X)
        ok.append((np.bincount(r, minlength=len(ids)) == N_T) & np.isfinite(X).all((1, 2)))
    return Xs, np.stack(ok, 1)


def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]


def order_ids(ids):
    # casting_2 before casting_10
    return sorted(ids, key=natural)


def synthetic_training(wins=WINDOWS):
    # synthetic_training_simple.parquet: windows [3 x (60000, 191, 4)], class ids, ids (casting_1 ...), train / held-out
    # indices. The held-out 20% (the VAE's validation set) are the synthetic test set of real_model_test_sets.parquet
    df = pd.read_parquet(os.path.join(DATA, 'synthetic_training_simple.parquet'),
                         columns=['obj_id_unique', 'class', 'time stamp'] + FEATS,
                         filters=[('time stamp', '>=', wins[0][0] - 1)])
    info = df.groupby('obj_id_unique', sort=False)['class'].first().astype(str)
    ids = np.array(order_ids(info.index))
    y = info.loc[ids].map(SYN_CLASSES.index).to_numpy()
    real = pd.read_parquet(os.path.join(DATA, 'real_model_test_sets.parquet'), columns=['obj_id_unique', 'section'])
    held = np.isin(ids, real.obj_id_unique[real.section == 'synthetic algorithms'].unique())
    assert held.sum() == len(ids) // 5, 'held-out set: 20% of the synthetic training set'
    Xs, ok = windows(df, ids, wins)
    assert ok.all(), 'a synthetic trajectory without all its windows'
    return Xs, y, ids, np.flatnonzero(~held), np.flatnonzero(held)


def test_sets(kind, wins=WINDOWS[:1]):
    df = pd.read_parquet(os.path.join(DATA, f'{kind}_model_test_sets.parquet'),
                         columns=['obj_id_unique', 'section', 'test_set', 'time stamp'] + FEATS,
                         filters=[('time stamp', '>=', wins[0][0] - 1), ('time stamp', '<=', wins[-1][1] + 1)])
    out = {}
    for (sec, name), d in df.groupby(['section', 'test_set'], observed=True, sort=False):
        d = fill_gaps(d) if kind == 'real' else d
        ids = np.array(order_ids(d.obj_id_unique.unique()))
        Xs, ok = windows(d, ids, wins)
        assert ok[:, 0].all(), f'{name}: trajectories without the scored window'
        out[name] = (Xs if len(wins) > 1 else Xs[0], ids, sec)
    return out


def real_training(wins=WINDOWS):
    df = fill_gaps(pd.read_parquet(os.path.join(DATA, 'real_training.parquet'),
                                   columns=['obj_id_unique', 'class', 'source', 'split', 'time stamp'] + FEATS))
    info = df.groupby('obj_id_unique')[['class', 'source', 'split']].first()
    rank = {s: i for i, s in enumerate(s for c in REAL_CLASSES for s in REAL_SOURCES[c])}
    info = info.assign(r=info.source.astype(str).map(rank)).sort_values('r', kind='stable').drop(columns='r')
    Xs, ok = windows(df, info.index.to_numpy(), wins)
    assert ok[:, 0].all(), 'a training fly without the 1.1-3.0 s window'
    return Xs, ok, info['class'].to_numpy(), info.source.to_numpy(), info.split.to_numpy(), info.index.to_numpy()


def balanced_split(y, seed=42):
    # classes balanced by drawing flies, then a stratified 80/20 split; both with the seed
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    n = min((y == c).sum() for c in REAL_CLASSES)
    flies = np.sort(np.concatenate([rng.choice(idx[y == c], n, replace=False) for c in REAL_CLASSES]))
    return train_test_split(flies, test_size=0.2, random_state=seed, stratify=[REAL_CLASSES.index(c) for c in y[flies]])


def real_split(y, split):
    tr, va = balanced_split(y, 42)
    assert (split[tr] == 'train').all() and (split[va] == 'held-out test').all() and \
        (split == 'train').sum() == len(tr), 'the split of real_training.parquet is not reproduced'
    return tr, va


def new_model():
    return CCL.TripletLossVAE(input_dim=N_T * len(FEATS), hidden_dim=540, latent_dim=70, margin=1.5)


def load_model(kind):
    model = new_model()
    model.load_state_dict(torch.load(os.path.join(DATA, MODELS[kind])))
    model.eval()
    return model


def normalize(X):
    flat = X.reshape(len(X), -1).astype(np.float64)
    mean, sd = flat.mean(1, keepdims=True), flat.std(1, keepdims=True)
    sd[sd == 0] = 1.0
    return ((flat - mean) / sd).astype(np.float32), mean, sd


def encode(model, X, batch=4096):
    x = normalize(X)[0]
    with torch.no_grad():
        return np.concatenate([model.encode(torch.from_numpy(x[i:i + batch]))[0].numpy()
                               for i in range(0, len(x), batch)])


def reconstruct(model, X, batch=4096):
    x, mean, sd = normalize(X)
    with torch.no_grad():
        xhat = np.concatenate([model.decode(model.encode(torch.from_numpy(x[i:i + batch]))[0]).numpy()
                               for i in range(0, len(x), batch)])
    return x, xhat, X.astype(np.float64), (xhat.astype(np.float64) * sd + mean).reshape(X.shape)


def fit_rf(Z, y, trees=1000):
    return CalibratedClassifierCV(RandomForestClassifier(n_estimators=trees, n_jobs=-1, random_state=42),
                                  method='temperature', cv=5).fit(Z, y)


def label(P, names):
    return np.where(P.max(1) >= THRESHOLD, np.asarray(names)[P.argmax(1)], OOD)


def count_matrix(labels, rows=ROWS):
    return pd.DataFrame({g: pd.Series(l).value_counts() for g, l in labels.items()}).reindex(list(rows)).fillna(0) \
        .astype(int)

def synthetic_results():
    path = os.path.join(CACHE, 'synthetic_model.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    model = load_model('synthetic')
    Xs, y, ids, i_tr, i_te = synthetic_training()
    Z_train = np.concatenate([encode(model, X[i_tr]) for X in Xs])
    y_train = np.tile(y[i_tr], len(Xs))
    out = dict(Z_train=Z_train, y_train=y_train, X_held=Xs[0][i_te], y_held=y[i_te], id_held=ids[i_te])
    del Xs
    out['Z_held'] = encode(model, out['X_held'])
    rf = fit_rf(Z_train, y_train)
    out['P_held'] = rf.predict_proba(out['Z_held'])
    sets = test_sets('synthetic')
    out.update(X_sets={g: X for g, (X, _, _) in sets.items()}, ids_sets={g: i for g, (_, i, _) in sets.items()},
               sections={g: s for g, (_, _, s) in sets.items()}, Z_sets={g: encode(model, X) for g, (X, _, _) in sets.items()})
    out['P_sets'] = {g: rf.predict_proba(Z) for g, Z in out['Z_sets'].items()}
    os.makedirs(CACHE, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    return out


def synthetic_counts(r, cols=None):
    names = [LABELS[c] for c in range(len(LABELS))]
    return count_matrix({g: label(r['P_sets'][g], names) for g in (cols or r['P_sets'])})


def held_out_accuracy(r):
    return 100 * (r['P_held'].argmax(1) == r['y_held']).mean()
