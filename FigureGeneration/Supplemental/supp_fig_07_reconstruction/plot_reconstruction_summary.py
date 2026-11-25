"""
SI Fig. S7: reconstructions and reconstruction error of the synthetic-trained autoencoder (Fig. 4;
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d
import figurefirst as fifi
import fly_plot_lib.plot as fpl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'VAE_pipeline'))
import vae_utils as V
from vae_utils import reconstruct, LABELS, CLASS_COLORS
V.reproducible_svg()

OUT = os.path.join(HERE, 'output_figs')
# real-fly groups: name in the figure -> test set; colours as Fig. 4
FIG4 = {'Still air': 'Still air, flash', 'Unsteady, flash': 'Unsteady, flash', 'Steady, flash': 'Steady, flash',
        'Laminar': 'Laminar, flash'}
SHAMS = {'Steady, sham': 'Steady, sham', 'Unsteady, sham': 'Unsteady, sham'}
REAL_COLORS = {'Still air': '#063161d2', 'Laminar': '#991128d2', 'Steady, flash': '#da834dff',
               'Unsteady, flash': '#2b75b3ff', 'Steady, sham': '#e8b48fff', 'Unsteady, sham': '#8db6daff'}
N_EXAMPLES, EXAMPLE_SEED = 5, 1
CAST_LEGS, CAST_MIN_LEG, CAST_RATIO = 3, 0.04, 1.5
LOG_TICKS = {'pos_norm': [0.01, 0.03, 0.1, 0.3, 1], 'heading_deg': [1, 3, 10, 30, 100]}
C_YLABELS = {'pos_norm': 'Position error\n(normalized)', 'heading_deg': 'Heading error (deg)'}
METRICS = {'pos_norm': 'Position error (normalized units)', 'vel_norm': 'Velocity error (normalized units)',
           'heading_deg': 'Heading error (deg)', 'pos_cm': 'Position error (cm)',
           'shape_cm': 'Shape error, offset removed (cm)'}
PAPER_RC = {'font.sans-serif': ['Helvetica'], 'font.family': 'sans-serif', 'font.size': 6,
            'font.weight': 'normal', 'mathtext.fontset': 'cm', 'text.usetex': False, 'svg.fonttype': 'none',
            'axes.spines.top': True, 'axes.spines.right': True}


def heading_error_deg(V, Vh):
    dth = np.angle(np.exp(1j * (np.arctan2(Vh[..., 1], Vh[..., 0]) - np.arctan2(V[..., 1], V[..., 0]))))
    w = np.linalg.norm(V, axis=2)
    return np.degrees((np.abs(dth) * w).sum(axis=1) / w.sum(axis=1))


def score(model, arr):
    x, xhat, orig, recon = reconstruct(model, arr)
    xn, xhn = x.reshape(arr.shape), xhat.reshape(arr.shape)
    d = recon[..., :2] - orig[..., :2]
    m = pd.DataFrame({
        'pos_norm': np.linalg.norm(xhn[..., :2] - xn[..., :2], axis=2).mean(1),
        'vel_norm': np.linalg.norm(xhn[..., 2:] - xn[..., 2:], axis=2).mean(1),
        'heading_deg': heading_error_deg(orig[..., 2:], recon[..., 2:]),
        'pos_cm': 100 * np.linalg.norm(d, axis=2).mean(1),
        'shape_cm': 100 * np.linalg.norm(d - d.mean(1, keepdims=True), axis=2).mean(1)})
    return m, orig, recon


def stereotypical_casting(P):
    out = []
    for p in P:
        cut = np.where(np.diff(np.sign(uniform_filter1d(p[:, 3], 10))) != 0)[0] + 1
        legs = sum(np.ptp(p[seg, 1]) >= CAST_MIN_LEG for seg in np.split(np.arange(len(p)), cut))
        out.append(legs >= CAST_LEGS and np.ptp(p[:, 1]) >= CAST_RATIO * np.ptp(p[:, 0]))
    return np.array(out)


def nice_bar(span):

    return max([b for b in [0.01, 0.02, 0.05, 0.1, 0.2] if b <= 0.4 * span] or [0.01])


def example_block(fig, name, ex, picks, color, x0, y0, s, gap, W, H):

    orig, recon = ex
    for i, k in enumerate(picks):
        ax = fig.add_axes([(x0 + i * (s + gap)) / W, y0 / H, s / W, s / H])
        both = np.concatenate([orig[k, :, :2], recon[k, :, :2]])
        cx, cy = (both.min(0) + both.max(0)) / 2
        half = 0.55 * np.ptp(both, axis=0).max()
        ax.plot(orig[k, :, 0], orig[k, :, 1], color=color, lw=0.75)
        ax.plot(recon[k, :, 0], recon[k, :, 1], color='k', lw=0.5, ls=(0, (2, 1)))
        ax.plot(orig[k, 0, 0], orig[k, 0, 1], 'o', markersize=2, markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=0.5, zorder=3)
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect('equal')
        fifi.mpl_functions.adjust_spines(ax, [])
        bar = nice_bar(2 * half)
        xl, yl = cx - half, cy - half - 0.1 * half
        ax.plot([xl, xl + bar], [yl, yl], color='k', lw=0.75, clip_on=False, solid_capstyle='butt')
        ax.annotate(f'{bar * 100:g} cm', xy=(xl + bar / 2, yl), xytext=(0, -1.5), textcoords='offset points',
                    ha='center', va='top', fontsize=6, annotation_clip=False)
    fig.text(x0 / W, (y0 + s + 0.03) / H, name.replace('Levy', 'Lévy'), ha='left', va='bottom', fontsize=6)


def log_violin(ax, pos, v, color, alpha, side):
    v = np.log10(v[v > 0])
    parts = ax.violinplot([v], positions=[pos], widths=0.85, side=side, showextrema=False)
    parts['bodies'][0].set_facecolor(color); parts['bodies'][0].set_alpha(alpha); parts['bodies'][0].set_edgecolor('none')
    lo, hi = {'low': (pos - 0.32, pos), 'high': (pos, pos + 0.32), 'both': (pos - 0.3, pos + 0.3)}[side]
    ax.plot([lo, hi], [np.median(v)] * 2, color='k', lw=1, solid_capstyle='butt')


def error_distribution(ax, E, syn, real, colors, metric):
    # panel C: synthetic split violins (train | held-out), real flies one dot per trajectory
    np.random.seed(0)                                         # jitter in fpl.scatter_box_violin_distribution
    syn_pos = list(range(len(syn)))
    real_pos = [len(syn) + 1 + i for i in range(len(real))]
    ax.axhline(np.log10(E.loc[E.split == 'held-out', metric].median()), color='k', ls='--', lw=0.5, zorder=0)
    for pos, g in zip(syn_pos, syn):
        log_violin(ax, pos, E.loc[(E.group == g) & (E.split == 'train'), metric].values, colors[g], 0.35, 'low')
        log_violin(ax, pos, E.loc[(E.group == g) & (E.split == 'held-out'), metric].values, colors[g], 0.85, 'high')
    for pos, g in zip(real_pos, real):
        v = np.log10(E.loc[E.group == g, metric].values)
        counts, bins = np.histogram(v, bins=20)
        fpl.scatter_box_violin_distribution(ax, pos, v.copy(), counts, bins.copy(), xwidth=0.35, color=colors[g],
                                            markersize=0.75)
    ticks = np.log10(LOG_TICKS[metric])
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_xlim(-0.6, real_pos[-1] + 0.6)
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'], yticks=ticks, xticks=syn_pos + real_pos, tick_length=2.5,
                                     spine_locations={'left': 5, 'bottom': 5}, linewidth=0.5)
    ax.set_yticklabels([f'{t:g}' for t in LOG_TICKS[metric]])
    ax.set_xticklabels([g.replace('Levy', 'Lévy') for g in syn + real], rotation=45, ha='right')
    ax.set_ylabel(C_YLABELS[metric], labelpad=1)
    ax.tick_params(axis='y', pad=2); ax.tick_params(axis='x', pad=2)
    for x, lab in [(np.mean(syn_pos), 'Synthetic (train | held-out)'), (np.mean(real_pos), 'Real flies')]:
        ax.text(x, 1.02, lab, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6)
    fifi.mpl_functions.set_fontsize(ax, 6)


def summary_figure(E, ex, syn, real, colors, path):
    rng = np.random.default_rng(EXAMPLE_SEED)
    pool = {g: np.where(stereotypical_casting(ex[g][0]))[0] if g == 'Casting' else np.arange(len(ex[g][0]))
            for g in syn + real}
    picks = {g: rng.choice(pool[g], N_EXAMPLES, replace=False) for g in syn + real}
    with plt.rc_context(PAPER_RC):
        # one page within 1 in margins on 8.5 x 11 in: 6.5 in wide, 7 in tall; fonts >= 6 pt, panel letters 10 pt
        W, H, left, right, gap = 6.5, 7.0, 0.25, 0.05, 0.04
        title_h, bar_h, row_gap, block_gap, sec_gap = 0.13, 0.17, 0.02, 0.25, 0.08
        s = (W - left - right - block_gap - 2 * 4 * gap) / 10     # 2 blocks of 5 panels per row
        row_h = title_h + s + bar_h + row_gap
        n_a, n_b = int(np.ceil(len(syn) / 2)), int(np.ceil(len(real) / 2))
        hC_below, hC_above = 0.62, 0.2
        hC_axes = H - 0.02 - (n_a + n_b) * row_h - 2 * sec_gap - hC_above - hC_below
        fig = plt.figure(figsize=(W, H))

        def rows(groups, top):
            for r in range(int(np.ceil(len(groups) / 2))):
                y = top - r * row_h - title_h - s
                for c, g in enumerate(groups[2 * r:2 * r + 2]):
                    example_block(fig, g, ex[g], picks[g], colors[g], left + c * (5 * s + 4 * gap + block_gap), y, s,
                                  gap, W, H)

        top = H - 0.02
        fig.text(0, top / H, 'A', fontsize=10, fontweight='bold', va='top')
        rows(syn, top)
        top -= n_a * row_h + sec_gap
        fig.text(0, top / H, 'B', fontsize=10, fontweight='bold', va='top')
        rows(real, top)
        top -= n_b * row_h + sec_gap
        fig.text(0, top / H, 'C', fontsize=10, fontweight='bold', va='top')
        cw = (W - 0.6 - 0.65 - right) / 2
        for i, metric in enumerate(['pos_norm', 'heading_deg']):
            ax = fig.add_axes([(0.6 + i * (cw + 0.65)) / W, hC_below / H, cw / W, hC_axes / H])
            error_distribution(ax, E, syn, real, colors, metric)
        for e in ['pdf', 'svg']:                                  # exact page size, no tight crop
            fig.savefig(f'{path}.{e}', transparent=True)
        plt.close(fig)


def iqr_str(s):
    q1, q2, q3 = s.quantile([.25, .5, .75])
    return f'{q2:.3f} [{q1:.3f}, {q3:.3f}]'


def rank_biserial(a, b):
    return 2 * stats.mannwhitneyu(a, b, alternative='two-sided')[0] / (len(a) * len(b)) - 1


if __name__ == '__main__':
    model = V.load_model('synthetic')

    # synthetic: the windows the model was trained / validated on
    X, y, _, i_tr, i_te = V.synthetic_training()
    classes = list(np.unique(y))
    syn = [LABELS[c] for c in classes]
    colors = {LABELS[c]: CLASS_COLORS[c] for c in classes} | REAL_COLORS
    rows, ex = [], {}
    for split, idx in [('train', i_tr), ('held-out', i_te)]:
        for w, x in enumerate(X):
            m, orig, recon = score(model, x[idx])
            rows.append(m.assign(group=pd.Series(y[idx]).map(LABELS).values, split=split, window=w, source='Synthetic'))
            if split == 'held-out' and w == 0:               # examples: held-out 1.1-3.0 s windows
                ex |= {LABELS[c]: (orig[y[idx] == c], recon[y[idx] == c]) for c in classes}
    del X

    # real flies: 1.1-3.0 s window of the Fig. 4 test sets
    X_sets = V.synthetic_results()['X_sets']
    for g, t in {**FIG4, **SHAMS}.items():
        m, orig, recon = score(model, X_sets[t])
        rows.append(m.assign(group=g, split='real', window=0, source='Real'))
        ex[g] = (orig, recon)
    E = pd.concat(rows, ignore_index=True)
    real, shams = list(FIG4), list(SHAMS)

    path = os.path.join(OUT, 'supplemental_reconstruction_losses')
    summary_figure(E, ex, syn, real, colors, path)

    # table: median [IQR] per group, effect size vs all held-out synthetic windows
    held = E[E.split == 'held-out']
    groups = {'All synthetic (train)': E[E.split == 'train'], 'All synthetic (held-out)': held,
              **{f'{g} (train)': E[(E.group == g) & (E.split == 'train')] for g in syn},
              **{f'{g} (held-out)': held[held.group == g] for g in syn},
              'Fig. 4 flies': E[E.group.isin(real)], 'Shams': E[E.group.isin(shams)],
              **{g: E[E.group == g] for g in real + shams}}
    T = pd.DataFrame({g: {'n': len(d), **{m: iqr_str(d[m]) for m in METRICS},
                          **{f'r {m} vs held-out': round(rank_biserial(d[m].values, held[m].values), 3)
                             for m in ['pos_norm', 'vel_norm', 'heading_deg']}}
                      for g, d in groups.items()}).T
    pd.set_option('display.width', 250)
    print(T[['n', 'pos_norm', 'heading_deg', 'r pos_norm vs held-out', 'r heading_deg vs held-out']].to_string())
    print(path + '.svg')
