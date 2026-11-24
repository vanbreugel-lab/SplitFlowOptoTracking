"""SI Fig. S8: the Fig. 4 triplet VAE + calibrated RF trained from scratch on real CsChrimson flies, 3 classes (still air
flash, laminar flash, sham; sliding-window training, 80/20 split of class-balanced flies). 
"""
import os
import re
import sys
import subprocess
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'VAE_pipeline'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'Main', 'fig_4_autoencoder'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import figurefirst as fifi
from matplotlib.colors import to_rgb
from sklearn.decomposition import PCA

import vae_utils as V
from vae_utils import encode, CLASS_COLORS
import make_f4 as F4                                        # (resets the rcParams: ours below)

MIN_PT = 6                        # smallest text, pt
plt.rcParams.update({'font.size': MIN_PT, 'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica'],
                     'axes.linewidth': 0.5, 'legend.fontsize': MIN_PT, 'axes.titlesize': MIN_PT,
                     'axes.labelsize': MIN_PT, 'xtick.labelsize': MIN_PT, 'ytick.labelsize': MIN_PT})
V.reproducible_svg()
OUT = os.path.join(HERE, 'output_figs')
MAX_PTS = 400                     # points drawn per group (random subsample; n in the legend is the whole group)
PAD = 0.12                        # PCA limits: 0.5-99.5th percentile of the training flies + this fraction per side
NAMES = {'Still air flash': 'Still air', 'Laminar flash': 'Laminar', 'Sham': 'Sham'}
COLORS = {'Still air flash': '#063161d2', 'Laminar flash': '#991128d2', 'Sham': '0.45'}   # training classes
HELD_SHAM = '0.2'                 # held-out shams (the Sham contour is 0.45 grey)
# test sets (real_model_test_sets.parquet) of B-D: legend name, colour
SPLIT_FLASH = {'Steady flash': ('Steady', '#da834dff'), 'Unsteady flash': ('Unsteady', '#2b75b3ff')}
SPLIT_SHAM = {'Steady sham': ('Steady', '#c0662b'), 'Unsteady sham': ('Unsteady', '#2b6cb0')}
SPEEDS = ['0', '0.03', '0.1', '0.2', '0.3', '0.6']
SPEED_COLORS = ['#2d4f86', '#6a3d9a', '#8e5fb8', '#b088d0', '#b33a50', '#7d1a2c']
SYNTHETIC = {'casting': ('Casting', CLASS_COLORS[0]), 'circling': ('Circling', 'darkcyan'),
             'random_turns': ('Random turns', CLASS_COLORS[2]), 'brownian': ('Brownian', CLASS_COLORS[3]),
             'levy_flight': ('Lévy flight', CLASS_COLORS[4]), 'run_and_tumble': ('Run & tumble', CLASS_COLORS[5])}
CFD = {'CFD casting, steady': ('Casting, steady', '#e8590c'), 'CFD casting, unsteady': ('Casting, unsteady', '#ff922b'),
       'CFD circling, steady': ('Circling, steady', '#0b7285'),
       'CFD circling, unsteady': ('Circling, unsteady', '#3bc9db')}
UNIFYING = {'Unifying, laminar': ('Laminar', COLORS['Laminar flash']),
            'Unifying, still air': ('Still air', COLORS['Still air flash']),
            'Unifying, unsteady': ('Unsteady', '#2b75b3ff')}
# matrix (E) columns: test set -> (label, colour strip); sections relabelled
SPLIT_COLUMNS = {'Unsteady flash': '#2b75b3ff', 'Steady flash': '#da834dff', 'Unsteady sham': '#8db6daff',
                 'Steady sham': '#e8b48fff'}
SECTIONS = {'validation set': 'validation', 'variable wind speed, flash': 'var. wind, flash',
            'variable wind speed, sham': 'var. wind, sham', 'synthetic algorithms': 'synthetic',
            'CFD simulations': 'CFD', 'unifying algorithm': 'unifying'}
LETTER_PT = 8                     # panel letters (bold)

# panel A: Fig. 4A's region of the template (mm: x, y, width, height), its scale here, the template text removed (class
# labels and numerals, A and B; the tops of Fig. 4's C / D labels and of D's colour strip, inside the crop)
A_CROP, A_SCALE = (14.88, 24.5, 135.0, 50.5), 1.191
A_AT = (0.0, 0.0219)              # its top left on the page, inches
# panel A's own layout, page pt (x right, y down from the top left): the example flies (inputs ai, reconstructions ao;
# centres, drawn at their template size, i.e. not enlarged with A), class labels (right-aligned, baseline) with their
# numerals below, and the numerals on the inset's contours
A_TRAJ = {'ai1': (17.4, 47.7), 'ai2': (57.05, 48.4), 'ai3': (17.9, 89.7), 'ai4': (50.25, 89.85), 'ai5': (51.2, 125.45),
          'ai6': (22.6, 134.95), 'ao1': (407.0, 44.8), 'ao2': (443.5, 46.4), 'ao3': (400.3, 85.0), 'ao4': (435.6, 84.0),
          'ao5': (435.3, 117.5), 'ao6': (408.2, 127.6)}
A_LABELS = [(25.1, 30.1), (27.6, 71.8), (21.3, 115.6)]                 # per class (V.REAL_CLASSES order)
A_NUMERALS = [(8.5, 35.4), (7.5, 77.2), (7.4, 121.8)]
INSET_NUMERALS = [(219.7, 111.4), (258.4, 112.3), (245.6, 159.7)]
ROMAN = ['i', 'ii', 'iii']
INK = (0, 0, 0, 0.886)
A_DROP = ['text2145', 'text2145-0', 'text2145-0-9', 'text2145-0-9-8', 'text2145-0-9-8-2', 'text2145-0-9-8-2-0',
          'text2145-7', 'text2145-7-0', 'text2145-7-4', 'text2145-7-01', 'text2145-7-5', 'text2145-7-5-2', 'text2199',
          'text2199-4', 'text2199-5', 'text12636', 'text2199-5-4', 'text2145-7-6', 'g329893']
A_PICKS = (0.4, 0.6)              # reconstruction-error percentiles of the two example flies per class
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
NS = {'': 'http://www.w3.org/2000/svg', 'xlink': 'http://www.w3.org/1999/xlink',
      'inkscape': 'http://www.inkscape.org/namespaces/inkscape', 'figurefirst': 'http://flyranch.github.io/figurefirst/',
      'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'}

# layout (inches): content W x at most MAX_H, top left of a letter page
W, MAX_H, PAGE = 6.5, 9.0, (8.5, 11)
A_W, A_H = A_CROP[2] / 25.4 * A_SCALE, A_CROP[3] / 25.4 * A_SCALE
A_GAP = 0.0956                    # A to the B-D titles
COL_W = W / 3                     # B, C, D: one column each, two PCA panels per column
PCA_LEFT, PCA_W, PCA_H = 0.17, 1.22, 1.12     # PCA axes in its column (PC2 label left)
PCA_ROW = 0.15 + PCA_H + 0.12 + 0.1           # title, axes, PC1 label, gap
# legends: offset of each panel's legend (B top, B bottom, C top, ...), pt (right, down), from just right of the axes
LEGEND_SHIFT = [(-29.444, 58.888), (-27.777, 61.666), (-16.046, 34.464), (-12.912, 33.211), (-10.026, 33.211),
                (-7.520, 53.890)]
DIVIDERS = [(134.2, 187.2, 393.2), (297.9, 186.6, 392.6)]             # between B, C and D: x, y top, y bottom (pt)
CELL_W, CELL_H, STRIP = 0.158, 0.13, 0.05     # matrix cells, colour strips
MAT_LEFT, MAT_HEAD, MAT_BELOW = 0.5578, 0.3, 1.0   # matrix: row labels, section labels above, column labels below


def contours(ax, P, tr, y, classes, lw=0.5):
    # training flies per class: filled density contours with outlines (B-D and the inset in A)
    for c in classes:
        m = tr[y[tr] == c]
        sns.kdeplot(x=P[m, 0], y=P[m, 1], levels=4, thresh=0.1, fill=True, alpha=0.25, color=COLORS[c], ax=ax)
        sns.kdeplot(x=P[m, 0], y=P[m, 1], levels=4, thresh=0.1, color=COLORS[c], linewidths=lw, ax=ax)


def pca_panel(ax, title, groups, P, tr, y, classes, pca, lim, rng, legend_shift):
    contours(ax, P, tr, y, classes)
    for name, (Q, col) in groups.items():
        show = Q if len(Q) <= MAX_PTS else Q[rng.choice(len(Q), MAX_PTS, replace=False)]
        ax.scatter(*show.T, s=1.5, lw=0, color=col, alpha=0.75, label=f'{name} ({len(Q)})', rasterized=True)
    dx, dy = legend_shift
    ax.legend(frameon=False, markerscale=2.5, loc='upper left',
              bbox_to_anchor=(1.02 + dx / 72 / PCA_W, 1 - dy / 72 / PCA_H), handlelength=0.8, handletextpad=0.3,
              borderaxespad=0, labelspacing=0.3)
    ax.set_title(title, pad=2, fontweight='bold')
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel(f'PC1 ({100 * pca.explained_variance_ratio_[0]:.0f}%)', labelpad=1)
    ax.set_ylabel(f'PC2 ({100 * pca.explained_variance_ratio_[1]:.0f}%)', labelpad=1)


def model_data():
    # the seed-42 model, its training flies (1.1-3.0 s windows, class, training / held-out indices in the original
    # order), the PCA of the training flies' latents, their projections, and the B-D panels:
    # [(title, {name: (points, colour)})]
    model = V.load_model('real')
    (X,), _, y, _, split, _ = V.real_training(V.WINDOWS[:1])
    tr, te = V.real_split(y, split)
    Z = encode(model, X).astype(np.float64)
    pca = PCA(n_components=2, random_state=0).fit(Z[tr])
    P = pca.transform(Z)
    tests = V.test_sets('real')
    proj = lambda g: pca.transform(encode(model, tests[g][0]).astype(np.float64))    # real flies
    one = lambda g: pca.transform(encode(model, tests[g][0]))                        # simulated trajectories
    panels = [                                                                 # by column: B, C, D
        ('split flow, flash', {n: (proj(g), c) for g, (n, c) in SPLIT_FLASH.items()}),
        ('split flow, sham', {n: (proj(g), c) for g, (n, c) in SPLIT_SHAM.items()}),
        ('variable wind, flash', {f'{s} m/s': (proj(f'{s} m/s flash'), c) for s, c in zip(SPEEDS, SPEED_COLORS)}),
        ('variable wind, sham', {f'{s} m/s': (proj(f'{s} m/s sham'), c) for s, c in zip(SPEEDS, SPEED_COLORS)}),
        ('synthetic algorithms', {n: (one(g), c) for g, (n, c) in SYNTHETIC.items()}),
        ('unifying algorithm', {n: (one(g), c) for g, (n, c) in UNIFYING.items()}),
    ]
    return model, X, y, tr, te, P, pca, panels


def mean_counts():
    # E: mean counts over the seeds (rows: classes + Uncertain; columns (section, test set) in file order), and the
    # columns' labels and colours
    M = pd.read_parquet(os.path.join(V.DATA, 'real_model_seed_matrices.parquet'))
    counts = M.pivot_table(index='label', columns=['section', 'test_set'], values='count', aggfunc='mean',
                           observed=True, sort=False).reindex(V.REAL_CLASSES + ['Uncertain'])
    counts = counts[list(dict.fromkeys(zip(M.section, M.test_set)))]
    names, colors = [], []
    for sec, n in counts.columns:
        if sec == 'validation set':
            name, col = n, COLORS[n]
        elif sec == 'split flow':
            name, col = n, SPLIT_COLUMNS[n]
        elif sec.startswith('variable wind'):
            speed = n.split(' m/s')[0]
            name, col = f'{speed} m/s', SPEED_COLORS[SPEEDS.index(speed)]
        else:
            name, col = {**SYNTHETIC, **CFD, **UNIFYING}[n]
        names.append(name)
        colors.append(col)
    return counts, names, colors


def matrix(fig, FIG_H, x0, y0, counts, names, x_colors, classes):
    # E: % of each column per predicted class (mean counts), heat map with row / column colour strips, section labels
    pct = (counts / counts.sum() * 100).values
    secs = [SECTIONS.get(s, s) for s, _ in counts.columns]
    nr, nc = pct.shape
    ax = fig.add_axes([x0 / W, y0 / FIG_H, nc * CELL_W / W, nr * CELL_H / FIG_H])
    ax.imshow(pct, cmap='gist_gray_r', vmin=0, vmax=100, aspect='auto', interpolation='nearest')
    for i in range(nr):
        for j in range(nc):
            ax.text(j, i, f'{pct[i, j]:.0f}', ha='center', va='center', color='w' if pct[i, j] > 55 else 'k')
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for j, (col, name) in enumerate(zip(x_colors, names)):                     # column strip and labels
        fig.patches.append(plt.Rectangle(((x0 + j * CELL_W) / W, (y0 - STRIP) / FIG_H), CELL_W / W, STRIP / FIG_H,
                                         fc=col, ec='none', lw=0, transform=fig.transFigure))
        fig.text((x0 + (j + 0.5) * CELL_W) / W, (y0 - STRIP - 0.03) / FIG_H, f'{name} ({counts.iloc[:, j].sum():.0f})',
                 rotation=90, ha='center', va='top')
    rows = [NAMES[c] for c in classes] + ['Uncertain']
    for i, (col, name) in enumerate(zip([COLORS[c] for c in classes] + ['k'], rows)):   # row strip and labels
        yi = y0 + (nr - 1 - i) * CELL_H
        fig.patches.append(plt.Rectangle(((x0 - STRIP) / W, yi / FIG_H), STRIP / W, CELL_H / FIG_H, fc=col, ec='none',
                                         lw=0, transform=fig.transFigure))
        fig.text((x0 - STRIP - 0.04) / W, (yi + CELL_H / 2) / FIG_H, name, ha='right', va='center')
    start = 0
    for s in dict.fromkeys(secs):                                              # section labels and dividers
        n = secs.count(s)
        if start:
            ax.axvline(start - 0.5, color='k', lw=0.8)
        top = y0 + nr * CELL_H
        fig.text((x0 + (start + n / 2) * CELL_W) / W, (top + 0.06) / FIG_H, s, ha='center', va='bottom')
        fig.lines.append(plt.Line2D([(x0 + (start + 0.1) * CELL_W) / W, (x0 + (start + n - 0.1) * CELL_W) / W],
                                    [(top + 0.035) / FIG_H] * 2, color='k', lw=0.5, transform=fig.transFigure))
        start += n
    cax = fig.add_axes([(x0 + nc * CELL_W + 0.1) / W, y0 / FIG_H, 0.06 / W, nr * CELL_H / FIG_H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap='gist_gray_r', norm=plt.Normalize(0, 100)), cax=cax, ticks=[0, 50, 100])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, pad=1, width=0.5)
    cb.set_label('% classified', labelpad=2)


def panel_a(model, X, y, te, P, tr, path):
    # Fig. 4A filled with this model (figurefirst on make_f4's flattened template); written to path (full page)
    layout = fifi.svg_to_axes.FigureLayout(F4.flat_template(), autogenlayers=True, make_mplfigures=True, hide_layers=[])
    fig = layout.figures['A1']
    k_pt = A_SCALE * 72 / 25.4                                                 # template mm -> page pt
    tw, th = fig.get_size_inches() * 25.4                                      # the template page, mm

    def frac(x_pt, y_pt):                                                      # page pt -> template figure fraction
        return ((x_pt - A_AT[0] * 72) / k_pt + A_CROP[0]) / tw, 1 - ((y_pt - A_AT[1] * 72) / k_pt + A_CROP[1]) / th

    for name, (cx, cy) in A_TRAJ.items():                                      # A_TRAJ centres, template size
        ax = layout.axes[('A1' if name.startswith('ai') else 'A2', name)]
        p = ax.get_position()
        fx, fy = frac(cx, cy)
        ax.set_position([fx - p.width / A_SCALE / 2, fy - p.height / A_SCALE / 2, p.width / A_SCALE,
                         p.height / A_SCALE])
    for k, c in enumerate(V.REAL_CLASSES):                                     # rows 1-2, 3-4, 5-6
        Xc = X[te[y[te] == c]]
        order = np.argsort(F4.reconstruct(model, Xc)[2])
        color = COLORS[c] if c != 'Sham' else HELD_SHAM
        for j, q in enumerate(A_PICKS):
            F4.draw_reconstruction(layout.axes[('A1', f'ai{2 * k + j + 1}')], layout.axes[('A2', f'ao{2 * k + j + 1}')],
                                   model, Xc[order[int(round(q * (len(order) - 1)))]], color, lw=F4.A_LINE_WIDTH)
        fig.text(*frac(*A_LABELS[k]), NAMES[c], ha='right', va='baseline', fontsize=MIN_PT / A_SCALE, color=color,
                 family='Helvetica')
        for f, at in ((fig, A_NUMERALS[k]), (layout.figures['Z'], INSET_NUMERALS[k])):   # (the inset is on top)
            f.text(*frac(*at), ROMAN[k], ha='center', va='baseline', fontsize=MIN_PT / A_SCALE, color=INK,
                   family='Helvetica')
    ax = layout.axes[('Z', 'z1')]                                               # inset: as B-D's contours
    contours(ax, P, tr, y, V.REAL_CLASSES, lw=0.4)
    ax.axis('off')
    for layer in ['A1', 'A2', 'Z']:
        layout.append_figure_to_layer(layout.figures[layer], layer, cleartarget=True)
    layout.write_svg(path)


def template_text(A):
    # the template's text in A at least MIN_PT once placed (A_SCALE); its bullet glyphs (the dotted node columns) as dots
    parent = {c: p for p in A.iter() for c in p}

    def scale(e):                                                              # linear scale of e's and its ancestors'
        k = 1.                                                                 # transforms (mm user units)
        while e is not None:
            for m in re.finditer(r'(matrix|scale)\(([^)]+)\)', e.get('transform') or ''):
                v = [float(x) for x in re.split(r'[\s,]+', m.group(2).strip())]
                k *= abs(v[0] * v[3] - v[1] * v[2]) ** 0.5 if m.group(1) == 'matrix' else abs(v[0])
            e = parent.get(e)
        return k
    smallest = np.inf
    for e in list(A.iter()):
        tag = e.tag.split('}')[1]
        m = re.search(r'font-size:([\d.]+)px', e.get('style') or '')
        if tag != 'text' or not m:
            if tag == 'tspan' and m:
                pt = float(m.group(1)) * scale(e) * A_SCALE * 72 / 25.4
                if pt < MIN_PT:
                    e.set('style', e.get('style').replace(m.group(0), f'font-size:{float(m.group(1)) * MIN_PT / pt:.5f}px'))
            continue
        if ''.join(e.itertext()).strip() == '•':                               # Helvetica's bullet: centre (0.175,
            f, x, y = float(m.group(1)), float(e.get('x')), float(e.get('y'))   # 0.315) em, radius 0.125 em
            dot = ET.Element(f'{{{NS[""]}}}circle', cx=f'{x + 0.175 * f:.5f}', cy=f'{y - 0.315 * f:.5f}',
                             r=f'{0.125 * f:.5f}', style='fill:#000000;stroke:none')
            if e.get('transform'):
                dot.set('transform', e.get('transform'))
            p = parent[e]
            p.insert(list(p).index(e), dot)
            p.remove(e)
            continue
        pt = float(m.group(1)) * scale(e) * A_SCALE * 72 / 25.4
        if pt < MIN_PT:
            e.set('style', e.get('style').replace(m.group(0), f'font-size:{float(m.group(1)) * MIN_PT / pt:.5f}px'))
        smallest = min(smallest, max(pt, MIN_PT))
    return smallest


def compose(mpl_svg, a_svg, out, x_in, y_in):
    # panel A's crop of a_svg (its template ids prefixed A_) into the matplotlib SVG, top left at (x_in, y_in) inches
    for k, v in NS.items():
        ET.register_namespace(k, v)
    A = ET.parse(a_svg).getroot()
    for parent in list(A.iter()):
        for e in list(parent):
            if e.get('id') in A_DROP or e.tag.split('}')[1] in ('namedview', 'metadata'):
                parent.remove(e)
    template_text(A)
    for e in A.iter():
        for k, v in e.attrib.items():
            v = re.sub(r'url\(#([^)]+)\)', r'url(#A_\1)', v)
            if k == 'id' or (k.endswith('href') and v.startswith('#')):
                v = re.sub(r'^(#?)', r'\1A_', v)
            e.set(k, v)
    root = ET.parse(mpl_svg).getroot()
    x0, y0, w, h = A_CROP
    k = A_SCALE * 72 / 25.4                                                    # mm -> pt
    g = ET.SubElement(root, f'{{{NS[""]}}}g', transform=f'translate({x_in * 72:.3f},{y_in * 72:.3f}) scale({k:.6f}) '
                                                        f'translate({-x0},{-y0})', id='panel_A')
    clip = ET.SubElement(ET.SubElement(g, f'{{{NS[""]}}}defs'), f'{{{NS[""]}}}clipPath', id='panel_A_crop')
    ET.SubElement(clip, f'{{{NS[""]}}}rect', x=str(x0), y=str(y0), width=str(w), height=str(h))
    inner = ET.SubElement(g, f'{{{NS[""]}}}g', {'clip-path': 'url(#panel_A_crop)'})
    inner.extend(list(A))
    root.remove(g)                                                             # beneath the matplotlib
    root.insert(1, g)                                                          # content (panel letter A)
    root.set('width', f'{PAGE[0]}in'), root.set('height', f'{PAGE[1]}in')      # the content at the top left of a
    root.set('viewBox', f'0 0 {PAGE[0] * 72:g} {PAGE[1] * 72:g}')              # letter page
    ET.ElementTree(root).write(out, xml_declaration=True, encoding='utf-8')


def render(svg, w_in, h_in):
    # PDF (vector) and PNG (300 dpi) of the composed SVG, headless Chrome (it can hang after writing: killed)
    base = svg[:-4]
    html = f'{base}_print.html'
    with open(html, 'w') as f:
        f.write(f'<html><head><style>@page {{size: {w_in}in {h_in}in; margin: 0}} html, body {{margin: 0}} '
                f'img {{width: {w_in}in; height: {h_in}in; display: block}}</style></head>'
                f'<body><img src="{os.path.basename(svg)}"></body></html>')
    prof = os.path.join(V.CACHE, 'chrome_profile')
    for flags, target in [(['--no-pdf-header-footer', f'--print-to-pdf={base}.pdf'], f'{base}.pdf'),
                          ([f'--window-size={round(w_in * 96)},{round(h_in * 96)}', '--force-device-scale-factor=3.125',
                            f'--screenshot={base}.png'], f'{base}.png')]:
        if os.path.exists(target):
            os.remove(target)
        p = subprocess.Popen([CHROME, '--headless=new', '--disable-gpu', f'--user-data-dir={prof}', *flags,
                              f'file://{html}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            pass
        p.kill()
        assert os.path.exists(target), f'Chrome did not write {target}'
    os.remove(html)


def label(fig, FIG_H, x, y, s):
    fig.text(x / W, y / FIG_H, s, fontsize=LETTER_PT, fontweight='bold', ha='left', va='top')


if __name__ == '__main__':
    model, X, y, tr, te, P, pca, panels = model_data()
    classes = V.REAL_CLASSES
    counts, col_names, col_colors = mean_counts()

    HM_H = CELL_H * len(counts)
    FIG_H = A_AT[1] + A_H + A_GAP + 2 * PCA_ROW + MAT_HEAD + HM_H + STRIP + MAT_BELOW
    assert FIG_H <= MAX_H + 1e-9, f'figure {FIG_H:.2f} in tall, more than {MAX_H}'
    fig = plt.figure(figsize=(W, FIG_H))
    lo, hi = np.percentile(P[tr], 0.5, axis=0), np.percentile(P[tr], 99.5, axis=0)
    pad = PAD * (hi - lo)
    lim = [(lo[d] - pad[d], hi[d] + pad[d]) for d in range(2)]
    rng = np.random.default_rng(0)

    # A (composed in below)
    label(fig, FIG_H, 0, FIG_H, 'A')

    # B-D: one column each, two panels per column, divided by lines
    for k, (title, groups) in enumerate(panels):
        c, r = divmod(k, 2)
        top = FIG_H - A_AT[1] - A_H - A_GAP - r * PCA_ROW
        x = c * COL_W + PCA_LEFT
        ax = fig.add_axes([x / W, (top - 0.15 - PCA_H) / FIG_H, PCA_W / W, PCA_H / FIG_H])
        pca_panel(ax, title, groups, P, tr, y, classes, pca, lim, rng, LEGEND_SHIFT[k])
        if r == 0:
            label(fig, FIG_H, c * COL_W, top + 0.02, 'BCD'[c])
    for x, y0, y1 in DIVIDERS:
        fig.lines.append(plt.Line2D([x / 72 / W] * 2, [1 - y0 / 72 / FIG_H, 1 - y1 / 72 / FIG_H], color='k', lw=0.5,
                                    transform=fig.transFigure))

    # E: classification matrix, mean of 10 seeds
    m_y0 = MAT_BELOW + STRIP
    matrix(fig, FIG_H, MAT_LEFT, m_y0, counts, col_names, col_colors, classes)
    label(fig, FIG_H, 0, m_y0 + HM_H + MAT_HEAD, 'E')

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(V.CACHE, exist_ok=True)
    path = os.path.join(OUT, 'supplemental_realdata_VAE')
    fig.savefig(f'{path}_mpl.svg', transparent=True, dpi=300)       # no tight bbox: panel A is placed in inches
    plt.close(fig)
    panel_a(model, X, y, te, P, tr, f'{path}_A.svg')
    compose(f'{path}_mpl.svg', f'{path}_A.svg', f'{path}.svg', *A_AT)
    os.remove(f'{path}_mpl.svg'), os.remove(f'{path}_A.svg')
    render(f'{path}.svg', *PAGE)
    print(f'{path}: {W} x {FIG_H:.2f} in content on a {PAGE[0]} x {PAGE[1]} in page')
