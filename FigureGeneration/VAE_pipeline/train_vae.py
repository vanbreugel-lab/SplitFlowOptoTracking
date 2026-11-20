"""Train the final sliding-window triplet VAE, fit the calibrated random forest and classify the test sets.


Usage (from this directory):
    python train_vae.py train synthetic|real [--seed 42] [--epochs 250] [--threads 5]
    python train_vae.py classify synthetic|real [--model PATH.pth]   (default: the shipped model in Data/final_vae_data)
    python train_vae.py seeds real [--seeds 0-9] [--epochs 250] [--threads 5]   (real_model_seed_matrices.parquet format)

    """
import os
import sys

import numpy as np
import pandas as pd
import torch

from vae_utils import (synthetic_training, real_training, real_split, balanced_split, test_sets, new_model, encode, fit_rf,
                       label, CCL, DATA, MODELS, WINDOWS, SYN_CLASSES, REAL_CLASSES, VARWIND_SPEED, OOD)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
SECTIONS = ['validation set', 'split flow', 'variable wind speed, flash', 'variable wind speed, sham', 'synthetic algorithms',
            'CFD simulations', 'unifying algorithm']
SPLIT_FLOW = ['Unsteady flash', 'Steady flash', 'Unsteady sham', 'Steady sham']


def samples(Xs, ok, y, tr, va):
    # every available window of the training / validation flies is a sample
    return {name: (np.concatenate([X[ii][ok[ii, k]] for k, X in enumerate(Xs)]),
                   np.concatenate([y[ii][ok[ii, k]] for k in range(len(WINDOWS))])) for name, ii in [('train', tr), ('val', va)]}


def training_data(kind):

    if kind == 'synthetic':
        Xs, y, _, tr, va = synthetic_training()
        ok = np.ones((len(y), len(WINDOWS)), bool)
    else:
        Xs, ok, cls, _, split, _ = real_training()
        y = np.array([REAL_CLASSES.index(c) for c in cls])
        tr, va = real_split(cls, split)
    return samples(Xs, ok, y, tr, va)


def trim(X, y):
    if len(y) % 128 == 1:                                   # a last batch of one sample breaks BatchNorm in training mode
        return X[:-1], y[:-1]
    return X, y


def fast_triplet_loss(model, z, labels):

    n = len(labels)
    same = labels[:, None] == labels[None, :]
    pos_m, neg_m = same & ~torch.eye(n, dtype=torch.bool), ~same
    ok = pos_m.any(1) & neg_m.any(1)
    if not ok.any():
        return torch.tensor(0.0)
    pos = torch.where(pos_m, torch.rand(n, n), -1.).argmax(1)
    neg = torch.where(neg_m, torch.rand(n, n), -1.).argmax(1)
    return model.triplet_loss(z[ok], z[pos[ok]], z[neg[ok]])


def fit_vae(kind, sets, seed, epochs):
    np.random.seed(seed)
    torch.manual_seed(seed)
    (Xtr, ytr), (Xva, yva) = trim(*sets['train']), sets['val']
    print(f'{kind}: {len(ytr)} training windows, {len(yva)} validation windows, {epochs} epochs', flush=True)
    loader = lambda X, y, shuffle: CCL.DataLoader(
        CCL.TrajectoryDataset(X, y, np.arange(len(y)), normalization_type='zero_mean'), batch_size=128, shuffle=shuffle)
    model = new_model()
    if kind == 'synthetic':
        model.compute_triplet_loss = lambda z, labels: fast_triplet_loss(model, z, labels)
    hist = CCL.train_vae_with_triplet(model, loader(Xtr, ytr, True), loader(Xva, yva, False), epochs=epochs,
                                      triplet_weight=0.2)
    model.eval()
    return model, hist


def train(kind, seed, epochs, threads):
    torch.set_num_threads(threads)
    model, hist = fit_vae(kind, training_data(kind), seed, epochs)
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, f'{kind}_retrained_vae_seed{seed}.pth'))
    pd.DataFrame(dict(zip(['train_total', 'val_total', 'train_triplet', 'val_triplet', 'train_kl', 'val_kl'], hist))
                 ).rename_axis('epoch').reset_index().to_parquet(os.path.join(OUT, f'{kind}_history_seed{seed}.parquet'))
    print(f'saved output/{kind}_retrained_vae_seed{seed}.pth', flush=True)


def classify(kind, path):
    model = new_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    sets = training_data(kind)
    Xtr, ytr = sets['train']
    if kind == 'real':                                      # as the real model's training samples
        Xtr, ytr = trim(Xtr, ytr)
    rf = fit_rf(encode(model, Xtr), ytr)
    Xva, yva = sets['val']
    print(f'{kind}: validation-window accuracy {100 * (rf.predict(encode(model, Xva)) == yva).mean():.1f}%', flush=True)
    classes = SYN_CLASSES if kind == 'synthetic' else REAL_CLASSES
    rows = []
    for name, (X, ids, sec) in test_sets(kind).items():
        lab = pd.Series(label(rf.predict_proba(encode(model, X)), classes)).replace({OOD: 'OOD / uncertain'})
        rows.append(pd.Series({**(lab.value_counts(normalize=True) * 100).to_dict(), 'n': len(ids)}, name=(sec, name)))
    out = pd.DataFrame(rows).reindex(columns=classes + ['OOD / uncertain', 'n']).fillna(0)
    out.index = pd.MultiIndex.from_tuples(out.index, names=['section', 'test_set'])
    os.makedirs(OUT, exist_ok=True)
    out.round(1).reset_index().to_parquet(os.path.join(OUT, f'{kind}_test_classification.parquet'))
    print(out.round(1).to_string())


def seed_matrices(seeds, epochs, threads):
    # real-fly model per seed: the seed's class balancing and 80/20 split, VAE, RF, test sets on their 1.1-3.0 s window.
    # The validation set and the 0, 0.3 and 0.6 m/s sets (training sources) are the seed's: flies outside its training 80%
    torch.set_num_threads(threads)
    Xs, ok, cls, src, _, _ = real_training()
    y = np.array([REAL_CLASSES.index(c) for c in cls])
    shared = {(sec, name): X for name, (X, _, sec) in test_sets('real').items() if sec != 'validation set'}
    own = {(f'variable wind speed, {k}', f'{speed:g} m/s {k}'): f'Varwind {fan}% {k}'
           for fan, speed in VARWIND_SPEED.items() for k in ['flash', 'sham'] if f'Varwind {fan}% {k}' in set(src)}
    rows = []
    for seed in seeds:
        tr, va = balanced_split(cls, seed)
        sets = samples(Xs, ok, y, tr, va)
        model, _ = fit_vae('real', sets, seed, epochs)
        Xtr, ytr = trim(*sets['train'])
        rf = fit_rf(encode(model, Xtr), ytr)
        unseen = np.setdiff1d(np.arange(len(y)), tr)
        X_sets = {**{('validation set', c): Xs[0][va[cls[va] == c]] for c in REAL_CLASSES}, **shared,
                  **{key: Xs[0][unseen[src[unseen] == s]] for key, s in own.items()}}
        for (sec, name), X in X_sets.items():
            lab = pd.Series(label(rf.predict_proba(encode(model, X)), REAL_CLASSES)).replace({OOD: 'Uncertain'})
            n = lab.value_counts().reindex(REAL_CLASSES + ['Uncertain'], fill_value=0)
            rows += [dict(seed=seed, section=sec, test_set=name, label=l, count=int(c), percent=100 * c / len(X))
                     for l, c in n.items()]
        print(f'seed {seed}: done', flush=True)
    out = pd.DataFrame(rows)
    known = {'validation set': REAL_CLASSES, 'split flow': SPLIT_FLOW, 'synthetic algorithms': SYN_CLASSES}
    order = lambda r: (r.seed, SECTIONS.index(r.section), known[r.section].index(r.test_set) if r.section in known else 0,
                       r.test_set, (REAL_CLASSES + ['Uncertain']).index(r.label))
    out = out.iloc[sorted(range(len(out)), key=lambda i: order(out.iloc[i]))].reset_index(drop=True)
    for c in ['section', 'test_set', 'label']:
        out[c] = pd.Categorical(out[c], categories=list(dict.fromkeys(out[c])))
    os.makedirs(OUT, exist_ok=True)
    out.to_parquet(os.path.join(OUT, 'real_model_seed_matrices.parquet'))
    print('saved output/real_model_seed_matrices.parquet', flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
    action, kind = args[0], args[1]
    assert kind in MODELS, 'kind: synthetic or real'
    if action == 'train':
        train(kind, opt('--seed', 42), opt('--epochs', 250), opt('--threads', 5))
    elif action == 'seeds':
        assert kind == 'real', 'seeds: real-fly model only'
        a, _, b = opt('--seeds', '0-9').partition('-')
        seed_matrices(range(int(a), int(b or a) + 1), opt('--epochs', 250), opt('--threads', 5))
    else:
        classify(kind, opt('--model', os.path.join(DATA, MODELS[kind])))
