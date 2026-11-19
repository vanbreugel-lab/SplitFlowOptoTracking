"""Train the final sliding-window triplet VAE, fit the calibrated random forest and classify the test sets.


Usage (from this directory):
    python train_vae.py train synthetic|real [--seed 42] [--epochs 250] [--threads 5]
    python train_vae.py classify synthetic|real [--model PATH.pth]   (default: the shipped model in Data/final_vae_data)

    """
import os
import sys

import numpy as np
import pandas as pd
import torch

from vae_utils import (synthetic_training, real_training, real_split, test_sets, new_model, encode, fit_rf, label, CCL,
                       DATA, MODELS, WINDOWS, SYN_CLASSES, REAL_CLASSES, OOD)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')


def training_data(kind):

    if kind == 'synthetic':
        Xs, y, _, tr, va = synthetic_training()
        ok = np.ones((len(y), len(WINDOWS)), bool)
    else:
        Xs, ok, cls, _, split, _ = real_training()
        y = np.array([REAL_CLASSES.index(c) for c in cls])
        tr, va = real_split(cls, split)
    return {name: (np.concatenate([X[ii][ok[ii, k]] for k, X in enumerate(Xs)]),
                   np.concatenate([y[ii][ok[ii, k]] for k in range(len(WINDOWS))])) for name, ii in [('train', tr), ('val', va)]}


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


def train(kind, seed, epochs, threads):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    sets = training_data(kind)
    (Xtr, ytr), (Xva, yva) = sets['train'], sets['val']
    if len(ytr) % 128 == 1:                                 # a last batch of one sample breaks BatchNorm in training mode
        Xtr, ytr = Xtr[:-1], ytr[:-1]
    print(f'{kind}: {len(ytr)} training windows, {len(yva)} validation windows, {epochs} epochs', flush=True)
    loader = lambda X, y, shuffle: CCL.DataLoader(
        CCL.TrajectoryDataset(X, y, np.arange(len(y)), normalization_type='zero_mean'), batch_size=128, shuffle=shuffle)
    model = new_model()
    if kind == 'synthetic':
        model.compute_triplet_loss = lambda z, labels: fast_triplet_loss(model, z, labels)
    hist = CCL.train_vae_with_triplet(model, loader(Xtr, ytr, True), loader(Xva, yva, False), epochs=epochs,
                                      triplet_weight=0.2)
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
    if kind == 'real' and len(ytr) % 128 == 1:              # as the real model's training samples
        Xtr, ytr = Xtr[:-1], ytr[:-1]
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


if __name__ == '__main__':
    args = sys.argv[1:]
    opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
    action, kind = args[0], args[1]
    assert kind in MODELS, 'kind: synthetic or real'
    if action == 'train':
        train(kind, opt('--seed', 42), opt('--epochs', 250), opt('--threads', 5))
    else:
        classify(kind, opt('--model', os.path.join(DATA, MODELS[kind])))
