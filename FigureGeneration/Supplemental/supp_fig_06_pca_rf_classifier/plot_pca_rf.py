"""SI Fig. S6: random-forest classification in reduced latent spaces."""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'VAE_pipeline'))
import vae_utils as V
import splitflow.RFClassifier as RF

plt.rcParams.update({'font.size': 7, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
V.reproducible_svg()
OUT = os.path.join(HERE, 'output_figs')
STACKED = (2, 3, 4, 5, 10)                       # numbers of components shown (with the full latent)
SPEEDS = ['0', '0.03', '0.1', '0.2', '0.3', '0.6']
# matrix sections (test sets of synthetic_model_test_sets.parquet) and column colours
GROUPS = [('Fig. 4 flies', ['Still air, flash', 'Unsteady, flash', 'Steady, flash', 'Laminar, flash']),
          ('shams', ['Still air, sham', 'Unsteady, sham', 'Steady, sham', 'Laminar, sham']),
          ('speed series, flash', [f'Var. wind {s} m/s, flash' for s in SPEEDS]),   # laminar speed series; set names as stored
          ('speed series, sham', [f'Var. wind {s} m/s, sham' for s in SPEEDS]),
          ('wild type', ['WT still air', 'WT unsteady', 'WT steady', 'WT laminar']),
          ('CFD', ['CFD casting, steady', 'CFD casting, unsteady', 'CFD circling, steady', 'CFD circling, unsteady']),
          ('unifying', ['Unifying, laminar', 'Unifying, still air', 'Unifying, unsteady'])]
COLORS = {'Still air, flash': '#063161d2', 'Unsteady, flash': '#2b75b3ff', 'Steady, flash': '#da834dff',
          'Laminar, flash': '#991128d2', 'Still air, sham': '#7f9cc4', 'Unsteady, sham': '#8db6daff',
          'Steady, sham': '#e8b48fff', 'Laminar, sham': '#d9909d',
          **dict(zip(GROUPS[2][1], plt.cm.Purples(np.linspace(0.35, 1, len(SPEEDS))))),
          **dict(zip(GROUPS[3][1], plt.cm.Greys(np.linspace(0.3, 0.9, len(SPEEDS))))),
          **dict(zip(GROUPS[4][1], ['#5b7fa8', '#8fc0e3', '#ecb995', '#c3707f'])),
          **dict(zip(GROUPS[5][1], ['#e8590c', '#ff922b', '#0b7285', '#3bc9db'])),
          **dict(zip(GROUPS[6][1], ['#991128d2', '#063161d2', '#2b75b3ff']))}


def helvetica_bold(ttc='/System/Library/Fonts/Helvetica.ttc'):
    # matplotlib reads only the first (regular) face of a .ttc: save the Bold face (index 1) as a .ttf and register it
    from fontTools.ttLib import TTCollection
    from matplotlib import font_manager
    path = os.path.join(V.CACHE, 'Helvetica-Bold.ttf')
    if not os.path.exists(path):
        os.makedirs(V.CACHE, exist_ok=True)
        TTCollection(ttc).fonts[1].save(path)
    font_manager.fontManager.addfont(path)


# layout on the letter page, pt from the top left (the published, formerly hand-finished, layout): heatmaps, the
# class / test-set colour strips left of and below them, the colour bar right of the last; titles and letters above
PAGE = (612, 792)
HM_X, HM_W, HM_H = 118.13, 376.10, 69.81
HM_TOP = [83.10, 167.02, 252.48, 337.95, 423.42, 511.98]
STRIP_W, STRIP_H = 2.72, 2.60
CB_X, CB_W = 498.89, 3.10
TITLE_X, TITLE_BASE = 293.6, [80.54, 164.46, 249.74, 335.96, 419.68, 507.53]
LETTER_X, LETTER_BASE = 75.0, [81.09, 162.45, 244.89, 328.72, 409.89, 495.06]
INK = (0, 0, 0, 0.886)                                     # titles and letters


def matrix_figure(counts, path):
    # the matrices stacked on a letter page: row labels, column labels and the percentage colour bar on the last panel
    # only; each panel titled with its name and lettered A, B, ...; sections separated by lines; Helvetica 6/8/10 pt
    rows = list(next(iter(counts.values())).index)
    cols = [c for _, g in GROUPS for c in g]
    assert len(counts) == len(HM_TOP)
    fs = dict(cell=6, label=6, name=8, letter=10)
    helvetica_bold()
    plt.rcParams.update({'font.family': 'Helvetica', 'pdf.fonttype': 42})
    W, H = PAGE
    box = lambda x, top, w, h: [x / W, 1 - (top + h) / H, w / W, h / H]          # page pt (y down) -> figure fraction
    fig = plt.figure(figsize=(W / 72, H / 72))
    for i, (name, cnt) in enumerate(counts.items()):
        last = i == len(counts) - 1
        top = HM_TOP[i]
        ax_hm = fig.add_axes(box(HM_X, top, HM_W, HM_H))
        ax_y = fig.add_axes(box(HM_X - STRIP_W, top, STRIP_W, HM_H))
        ax_x = fig.add_axes(box(HM_X, top + HM_H, HM_W, STRIP_H))
        ax_cb = fig.add_axes(box(CB_X, top, CB_W, HM_H)) if last else None
        cnt = cnt[cols]
        named = {k: f"{k.replace('Var. wind', 'Speed series')} ({cnt[k].sum()})" for k in cols}   # displayed column labels
        cnt = cnt.rename(index={'Levy flight': 'Lévy flight'})
        RF.plot_classification_matrix_with_colorbars(
            percentage_matrix=(cnt / cnt.sum() * 100).rename(columns=named), count_matrix=cnt.rename(columns=named),
            ax1=ax_hm, ax2=ax_y, ax3=ax_x, ax4=ax_cb, y_colors=[V.ROWS[r] for r in rows],
            x_colors=[COLORS[c] for c in cols], show_y_labels=last, show_x_labels=last, colorbar_width=0.005,
            heatmap_cmap='gist_gray_r', colorbar_border=False, show_percentage_cbar=last)
        for tx in ax_hm.texts:
            tx.set_fontsize(fs['cell'])
        if last:
            for ax in (ax_x, ax_y, ax_cb):
                ax.tick_params(labelsize=fs['label'], length=1.5, pad=1, width=0.4)
            ax_cb.yaxis.label.set_size(fs['label'])
            ax_cb.yaxis.label.set_rotation(270)                   # reads top to bottom
            ax_cb.yaxis.label.set_va('bottom')
        for x0 in np.cumsum([len(g) for _, g in GROUPS])[:-1]:
            ax_hm.axvline(x0, color='k', lw=0.6)
        fig.text(TITLE_X / W, 1 - TITLE_BASE[i] / H, name, fontsize=fs['name'], color=INK, ha='center',
                 va='baseline')
        fig.text(LETTER_X / W, 1 - LETTER_BASE[i] / H, 'ABCDEFGHIJ'[i], fontsize=fs['letter'], fontweight='bold',
                 color=INK, ha='center', va='baseline')
    for e in ['pdf', 'svg']:                                  # the whole page
        fig.savefig(f'{path}.{e}', bbox_inches=None, transparent=True)
    plt.close(fig)


def main():
    r = V.synthetic_results()
    Z_tr, y_tr, Z_te, y_te = r['Z_train'], r['y_train'], r['Z_held'], r['y_held']
    names = [V.LABELS[c] for c in range(len(V.LABELS))]
    pca = PCA(random_state=0).fit(Z_tr)
    ev_cum = np.cumsum(pca.explained_variance_ratio_)
    counts = {}
    for k in list(STACKED) + [None]:
        proj = (lambda Z: Z) if k is None else (lambda Z, k=k: pca.transform(Z)[:, :k])
        name = (f'Full {Z_tr.shape[1]}-dim latent space (100% explained variance)' if k is None else
                f'{k} PCs ({100 * ev_cum[k - 1]:.0f}% explained variance)')
        rf = V.fit_rf(proj(Z_tr), y_tr)
        acc = 100 * (rf.classes_[rf.predict_proba(proj(Z_te)).argmax(1)] == y_te).mean()
        c = V.count_matrix({g: V.label(rf.predict_proba(proj(Z)), names) for g, Z in r['Z_sets'].items()})
        pct = c / c.sum() * 100
        print(f'{name}: held-out {acc:.2f}% | still air -> circling {pct.loc["Circling", "Still air, flash"]:.1f}% | '
              f'laminar -> casting {pct.loc["Casting", "Laminar, flash"]:.1f}% | CFD casting steady / unsteady '
              f'{pct.loc["Casting", "CFD casting, steady"]:.1f} / {pct.loc["Casting", "CFD casting, unsteady"]:.1f}% | '
              f'CFD circling steady / unsteady {pct.loc["Circling", "CFD circling, steady"]:.1f} / '
              f'{pct.loc["Circling", "CFD circling, unsteady"]:.1f}%', flush=True)
        counts[name] = c
    matrix_figure(counts, os.path.join(OUT, 'supplemental_pca_2_to_10_rf'))
    print(os.path.join(OUT, 'supplemental_pca_2_to_10_rf.svg'))


if __name__ == '__main__':
    main()
