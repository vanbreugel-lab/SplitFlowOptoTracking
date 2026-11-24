"""Fig. 4 (layout: f4_template.svg): example trajectories and their reconstructions, the latent space in 3D PCA and the
random-forest classification matrices of the synthetic-trained autoencoder.


Usage (from this directory): python make_f4.py

"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'VAE_pipeline'))
import vae_utils as V

import matplotlib.pyplot as plt
import figurefirst as fifi
from sklearn.decomposition import PCA
import splitflow.RFClassifier as RF

from scipy.stats import gaussian_kde
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D, Bbox, ScaledTranslation
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from skimage.measure import marching_cubes

plt.rcdefaults()
plt.rcParams.update({'font.size': 6, 'text.usetex': False, 'ps.usedistiller': 'xpdf', 'font.family': 'sans-serif',
                     'font.sans-serif': ['Helvetica'], 'font.weight': 'normal', 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Helvetica', 'mathtext.it': 'Helvetica:italic'})
V.reproducible_svg()

TEMPLATE = os.path.join(HERE, 'f4_template.svg')

ROWS = [0, 1, 2, 4, 5, 3]
ROW_COLORS = {0: 'orangered', 1: 'darkcyan', 2: 'orchid', 3: '#dfa45cff', 4: '#a5cc4cff', 5: '#661ac6ff'}

EXAMPLES = {
    'synthetic': {0: 'casting_5681', 1: 'circling_7411', 2: 'random_turns_9746', 3: 'brownian_2816',
                  4: 'levy_flight_7289', 5: 'run_and_tumble_9020'},
    'CFD circling, unsteady': ['circling_540_vel_5_unsteady', 'circling_574_vel_6_unsteady', 'circling_322_vel_1_unsteady'],
    'CFD casting, unsteady': ['casting_326_vel_1_unsteady', 'casting_349_vel_2_unsteady', 'casting_359_vel_2_unsteady'],
    'Unifying, still air': ['unifying_stillair_62', 'unifying_stillair_151', 'unifying_stillair_96'],
    'Unifying, unsteady': ['unifying_unsteady_41', 'unifying_unsteady_162', 'unifying_unsteady_148'],
    'Unifying, laminar': ['unifying_laminar_5', 'unifying_laminar_183', 'unifying_laminar_3'],
    'Steady, flash': ['11319_20240516_185822'],
    'Unsteady, flash': ['13155_20240111_160822'],
    'Laminar, flash': ['1306_20220926_173927'],
    'Still air, flash': ['20220817_173045_2473_72'],
}

SET_COLORS = {'Steady, flash': '#da834d', 'Unsteady, flash': '#2b75b3', 'Laminar, flash': '#991128',
              'Still air, flash': '#07325f', 'CFD casting, unsteady': '#11736f', 'CFD circling, unsteady': '#235c94',
              'Unifying, laminar': '#ff735b', 'Unifying, unsteady': '#83bbf5', 'Unifying, still air': '#06315f'}

EXAMPLE_PANELS = [('C2', ['c_t1', 'c_t2', 'c_t3'], 'CFD circling, unsteady'),
                  ('C2', ['c_t4', 'c_t5', 'c_t6'], 'CFD casting, unsteady'),
                  ('D2', ['d1a', 'd1b', 'd1c'], 'Unifying, still air'),
                  ('D2', ['d2a', 'd2b', 'd2c'], 'Unifying, unsteady'),
                  ('D2', ['d3a', 'd3b', 'd3c'], 'Unifying, laminar'),
                  ('E2', ['e1a'], 'Steady, flash'), ('E2', ['e1b'], 'Unsteady, flash'), ('E2', ['e1c'], 'Laminar, flash'),
                  ('E2', ['e1d'], 'Still air, flash')]

PCA_PANELS = [('Z', 'z1', [], 'scatter'),
              ('B', 'b1', [('Laminar, flash', '#991128'), ('Still air, flash', '#053062')], 'scatter'),
              ('B', 'b2', [('Steady, flash', 'darkorange'), ('Unsteady, flash', plt.get_cmap('RdBu', 16)(13))], 'scatter'),
              ('C', 'c1', [('CFD circling, unsteady', SET_COLORS['CFD circling, unsteady'])], 'shells'),
              ('C', 'c2', [('CFD casting, unsteady', SET_COLORS['CFD casting, unsteady'])], 'shells'),
              ('D1', 'd1', [('Unifying, still air', SET_COLORS['Unifying, still air'])], 'shells'),
              ('D1', 'd2', [('Unifying, unsteady', SET_COLORS['Unifying, unsteady'])], 'shells'),
              ('D1', 'd3', [('Unifying, laminar', SET_COLORS['Unifying, laminar'])], 'shells')]

MATRICES = [('RF1', 'rf1', ['Still air, flash', 'Unsteady, flash', 'Steady, flash', 'Laminar, flash'],
             ['#063161d2', '#2b75b3ff', '#da834dff', '#991128d2'], 6),
            ('RF2', 'rf2', ['CFD circling, unsteady', 'CFD casting, unsteady'], ['#0e4075ca', '#11736ff2'], 6),
            ('RF3', 'rf3', ['Unifying, still air', 'Unifying, unsteady', 'Unifying, laminar'],
             ['#06315fca', '#83bbf5cb', '#ff735bca'], 5)]
KDE_PER_CLASS = 8000                                       
NUMERALS = {0: 'i', 1: 'ii', 2: 'iii', 4: 'iv', 5: 'v', 3: 'vi'}

SHELLS = [(0.2, 0.1), (0.5, 0.18), (0.75, 0.3)]
TEST_SHELLS = [(0.2, 0.53), (0.5, 0.67), (0.75, 0.81)]
N_KDE = 3000                                               
CAMERA_3D = dict(x=-1.34, y=-0.92, z=0.28)                
                                                         
N_MESH = 32                                         
                                                        
FOCAL_3D = 2.0                                              
ZOOM_3D = 1.35                                             
ZOOM_INSET_3D = 0.85                                    
SHIFT_3D = (-0.1, 0.0)                                     
PANEL_SHIFT_3D = {'b2': (-0.12, 0.0)}
# finishing of the published figure (formerly done by hand in Inkscape), as page transforms in pt (x right, y down):
# X' = s X + (tx, ty) on the 3D drawing and its clip area. Panel C left, E left, G and the inset B (enlarged)
FINISH_3D = {'b1': (1, -10.612715, 0.910405), 'c1': (1, -9, 0), 'd1': (1, -9.378866, -19.820324),
             'd2': (1, -8.854473, -21.160162), 'd3': (1, -9, -21), 'z1': (1.1839877, -38.720401, -37.288418)}
# 2D panels: ('move', dx, dy) or ('rotate', degrees clockwise, cx, cy), page pt. A: reconstructions nudged, the
# Brownian example turned 90 deg (input and reconstruction); E, G: example trajectories; D, F, H: the matrices
FINISH_2D = {('A1', 'ai6'): ('rotate', 90, 100.45742, 202.45504), ('A2', 'ao6'): ('rotate', 90, 364.23521, 194.58719),
             ('A2', 'ao1'): ('move', 4.5, 7.5), ('A2', 'ao2'): ('move', 6, 6), ('A2', 'ao3'): ('move', 3, 4.5),
             ('A2', 'ao4'): ('move', 3, 3), ('A2', 'ao5'): ('move', 6, 0),
             **{('C2', n): ('move', -9, 0) for n in ['c_t1', 'c_t2', 'c_t3']},
             **{('D2', n): ('move', -10.5, 0) for n in ['d1a', 'd1b', 'd1c']},
             **{('D2', n): ('move', -6, 0) for n in ['d2a', 'd2b', 'd2c']},
             **{('D2', n): ('move', -9, 0) for n in ['d3a', 'd3b', 'd3c']},
             ('E2', 'e1c'): ('move', -7.5, 1.5), ('E2', 'e1a'): ('move', -1.5, 1.5),
             ('RF1', 'rf1'): ('move', 0, -3), ('RF2', 'rf2'): ('move', 0, -4.5), ('RF3', 'rf3'): ('move', 0, -1.5)}
PAGE_PT = (612, 792)                                       # every figurefirst figure is the whole letter page
GRID_3D = 8                                               
LIGHT_3D = dict(ambient=0.62, diffuse=0.45, specular=0.12, shininess=20, up=0.6, left=0.4)  
AXIS_KEY = dict(origin=(-0.172827, 0.113207), arrows=[(0.104, -0.055), (0.095, 0.046), (0.0, 0.1375)],
                labels=[('left', 'center', (0.004, -0.012)), ('left', 'center', (0.004, 0.012)),
                        ('center', 'bottom', (0.042781, 0.017459))])
KEY_PANELS = ('b1',)                                       
KEY_FONT = 6                                   

ARROWS = {'path44700': ('b1', 'Laminar, flash'), 'path44700-2': ('b1', 'Still air, flash'),
          'path44700-8': ('b2', 'Steady, flash'), 'path44700-2-7': ('b2', 'Unsteady, flash'),
          'path44700-82': ('c1', 'CFD circling, unsteady'), 'path44700-8-9': ('c2', 'CFD casting, unsteady')}
ARROW_GAP_MM = 2.5
GRID_PAD_3D = 0.3                                         
                                                          
                                               
POINT_3D = 1                                               
A_LINE_WIDTH = 0.6                                         
RECON_COLOR = '#b3b3b3'                                 


def reconstruct(model, X):
    x, xhat = V.reconstruct(model, X)[:2]
    return x.reshape(X.shape), xhat.reshape(X.shape), np.sqrt(((xhat - x) ** 2).mean(1))


def draw_reconstruction(ax_in, ax_out, model, X, color, lw=1., recon_color=None):
    x, xhat, _ = reconstruct(model, X[None])
    ax_out.plot(xhat[0, :, 0], xhat[0, :, 1], '--', lw=lw, color=recon_color or color)
    ax_in.plot(x[0, :, 0], x[0, :, 1], lw=lw, color=color)
    for ax in [ax_in] if ax_in is ax_out else [ax_in, ax_out]:
        ax.axis('off')
        ax.set_aspect('equal', 'box' if ax is ax_out else None)


def flat_template():
    s = open(TEMPLATE).read()
    g = re.search(r'<g\b[^>]*\bid="g93854"[^>]*>', s)
    mat = g and re.search(r'matrix\(([^)]+)\)', g.group(0))
    assert mat, f'{TEMPLATE}: A2 group g93854 changed; update flat_template'
    a, _, _, d, e, f = map(float, mat.group(1).split(','))
    j = s.index('</g>', g.end())

    def bake(m):
        r = m.group(0)
        t = re.search(r'\s*transform="translate\(([-\d.e]+)\)"', r)
        tx = float(t.group(1)) if t else 0.
        r = r.replace(t.group(0), '') if t else r
        for attr, fn in [('x', lambda v: a * (v + tx) + e), ('y', lambda v: d * v + f), ('width', lambda v: a * v),
                         ('height', lambda v: d * v)]:
            r = re.sub(rf'(\s{attr}=")([-\d.]+)"', lambda q: f'{q.group(1)}{fn(float(q.group(2))):.6f}"', r, count=1)
        return r
    s = s[:g.start()] + '<g id="g93854">' + re.sub(r'<rect[^>]*>', bake, s[g.end():j]) + s[j:]
    s = re.sub(r'translate\(([-\d.e]+)\)', r'translate(\1,0)', s)
    s = re.sub(r'(<rect\b[^>]*?style=")[^"]*("[^>]*>\s*<figurefirst:axis)', r'\1fill:none;stroke:none\2', s)
    os.makedirs(V.CACHE, exist_ok=True)
    path = os.path.join(V.CACHE, 'f4_template_flat.svg')
    with open(path, 'w') as fh:
        fh.write(s)
    return path


def axis_key(ax, size_in):

    w, h = size_in
    o = np.array(AXIS_KEY['origin'])
    to_axes = lambda p: (p[0] / w, p[1] / h)        
    for d, (v, (ha, va, off)) in enumerate(zip(AXIS_KEY['arrows'], AXIS_KEY['labels'])):
        tip = o + np.array(v)
        ax.annotate('', xy=to_axes(tip), xytext=to_axes(o), xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', lw=0.5, color='#333', mutation_scale=4, shrinkA=0, shrinkB=0))
        ax.text(*to_axes(tip + np.array(off)), f'PC{d + 1}', transform=ax.transAxes, fontsize=KEY_FONT, color='#333',
                ha=ha, va=va)


def shell_meshes(P, levels, glo, ghi, seed=0):

    Q = P[np.random.default_rng(seed).choice(len(P), min(len(P), N_KDE), replace=False)]
    kde = gaussian_kde(Q.T)
    at_pts = kde(Q.T)
    grid = np.stack(np.meshgrid(*[np.linspace(glo[d], ghi[d], N_MESH) for d in range(3)], indexing='ij'))
    D = kde(grid.reshape(3, -1)).reshape((N_MESH,) * 3)
    out = []
    for q, op in levels:
        lv = np.quantile(at_pts, q)
        if not D.min() < lv < D.max():
            continue
        v, f, n, _ = marching_cubes(D, lv, spacing=(1 / (N_MESH - 1),) * 3)   # unit-cube coordinates
        fn = n[f].mean(1)
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        out.append((glo + v[f] * (ghi - glo), fn, op))
    return out


def view_vectors():

    forward = -np.array([CAMERA_3D[k] for k in 'xyz'], float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    return forward, right, np.cross(right, forward)


def lit(rgb, normals):

    forward, right, up = view_vectors()
    light = -forward + LIGHT_3D['up'] * up - LIGHT_3D['left'] * right
    light /= np.linalg.norm(light)
    half = light - forward
    half /= np.linalg.norm(half)
    diffuse = np.abs(normals @ light)
    spec = np.abs(normals @ half) ** LIGHT_3D['shininess']
    c = np.asarray(rgb)[None] * (LIGHT_3D['ambient'] + LIGHT_3D['diffuse'] * diffuse)[:, None] + \
        LIGHT_3D['specular'] * spec[:, None]
    return np.clip(c, 0, 1)


def draw_shells(ax, meshes, rgb, name, a, b, z0):

    forward = view_vectors()[0]
    halves = []
    for k, (tris, normals, op) in enumerate(meshes):
        centre = tris.reshape(-1, 3).mean(0)
        out = np.sign(((tris.mean(1) - centre) / (b - a) * normals).sum(1, keepdims=True))   # normals pointing out
        facing = (normals * out) @ -forward > 0
        halves.append((k, tris, normals, op, facing))
    order = [(k, ~f, 'back') for k, _, _, _, f in halves] + [(k, f, 'front') for k, _, _, _, f in halves[::-1]]
    for z, (k, sel, side) in enumerate(order):
        tris, normals, op = halves[k][1], halves[k][2], halves[k][3]
        if not sel.any():
            continue
        pc = Poly3DCollection(tris[sel], facecolors=lit(rgb, normals[sel]), alpha=op, edgecolors='none', linewidths=0)
        pc.set_gid(f'shell_{name}_{k}_{side}'.replace(' ', '_').replace(',', '').replace('&', 'and'))
        pc.set_zorder(z0 + z)
        ax.add_collection3d(pc)
    return z0 + len(order)


def back_grid(ax, a, b):

    eye = np.array([CAMERA_3D[k] for k in 'xyz'])
    wall = np.where(eye < 0, b, a)                         
    ticks = [np.linspace(a[d], b[d], GRID_3D) for d in range(3)]
    lines = []
    for d in range(3):                                     
        u, v = [e for e in range(3) if e != d]
        for t in ticks[u]:
            p, q = np.zeros(3), np.zeros(3)
            p[d] = q[d] = wall[d]
            p[u] = q[u] = t
            p[v], q[v] = a[v], b[v]
            lines.append([p, q])
        for t in ticks[v]:
            p, q = np.zeros(3), np.zeros(3)
            p[d] = q[d] = wall[d]
            p[v] = q[v] = t
            p[u], q[u] = a[u], b[u]
            lines.append([p, q])
    lc = Line3DCollection(lines, colors='#dddddd', linewidths=0.4, zorder=0)
    lc.set_gid('grid')
    ax.add_collection3d(lc)


def axes_3d(fig, pos, full, shift, a, b, zoom):

    fw, fh = fig.get_size_inches()
    cx, cy = pos.x0 + pos.width / 2 + shift[0] / fw, pos.y0 + pos.height / 2 + shift[1] / fh
    ax = fig.add_axes([cx - full[0] / fw / 2, cy - full[1] / fh / 2, full[0] / fw, full[1] / fh], projection='3d')
    ax.set_proj_type('persp', focal_length=FOCAL_3D)
    ax.computed_zorder = False
    ax.set_xlim(a[0], b[0])
    ax.set_ylim(a[1], b[1])
    ax.set_zlim(a[2], b[2])
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=np.degrees(np.arctan2(CAMERA_3D['z'], np.hypot(CAMERA_3D['x'], CAMERA_3D['y']))),
                 azim=np.degrees(np.arctan2(CAMERA_3D['y'], CAMERA_3D['x'])))
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    back_grid(ax, a, b)
    return ax


def page_transform(bbox, s, tx, ty):
    # an axes position (figure fraction, y up) under the page transform X' = s X + (tx, ty) (pt, y down)
    W, H = PAGE_PT
    x0, y0 = s * bbox.x0 * W + tx, s * (1 - bbox.y1) * H + ty
    return Bbox.from_bounds(x0 / W, 1 - (y0 + s * bbox.height * H) / H, s * bbox.width, s * bbox.height)


def finish_2d(layout):
    # FINISH_2D: moves (the axes) and quarter turns (its lines, about a page point; unclipped)
    W, H = PAGE_PT
    for (layer, name), (kind, *p) in FINISH_2D.items():
        ax = layout.axes[(layer, name)]
        if kind == 'move':
            ax.set_position(page_transform(ax.get_position(), 1, *p))
        else:
            deg, cx, cy = p                                 # (display units follow the dpi the SVG is drawn at)
            fig = ax.figure
            c = np.array([cx / W, 1 - cy / H]) * fig.get_size_inches()
            turn = (ScaledTranslation(*-c, fig.dpi_scale_trans) + Affine2D().rotate_deg(-deg) +
                    ScaledTranslation(*c, fig.dpi_scale_trans))
            for line in ax.lines:
                line.set_transform(line.get_transform() + turn)
                line.set_clip_on(False)


def projected_mm(ax, xyz, page_in):
   
    x2, y2, _ = proj3d.proj_transform(*xyz, ax.get_proj())
    fx, fy = ax.figure.transFigure.inverted().transform(ax.transData.transform((x2, y2)))
    return fx * page_in[0] * 25.4, (1 - fy) * page_in[1] * 25.4


def aim_arrows(svg, targets):
    s = open(svg).read()
    for pid, tgt in targets.items():
        m = re.search(rf'<path\b[^>]*\bid="{re.escape(pid)}"[^>]*>', s)
        d = re.search(r'\sd="([mM])\s*([-\d.e]+),([-\d.e]+)\s+([-\d.e]+),([-\d.e]+)"', m.group(0))
        start = np.array([float(d.group(2)), float(d.group(3))])
        end = np.array(tgt)
        end = end + (start - end) / np.linalg.norm(start - end) * ARROW_GAP_MM
        tag = m.group(0).replace(d.group(0), f' d="M {start[0]:.4f},{start[1]:.4f} {end[0]:.4f},{end[1]:.4f}"')
        s = s.replace(m.group(0), tag)
    with open(svg, 'w') as fh:
        fh.write(s)


def panel_inches(layout, layer, name):
    ax = layout.axes[(layer, name)]
    return ax.get_position().width * ax.figure.get_figwidth(), ax.get_position().height * ax.figure.get_figheight()


def pca_panels_3d(layout, pca, P_tr, y_sub, Z_sets):

    a, b = np.percentile(P_tr, 0.5, axis=0), np.percentile(P_tr, 99.5, axis=0)
    glo, ghi = a - GRID_PAD_3D * (b - a), b + GRID_PAD_3D * (b - a)
    classes = np.unique(y_sub)
    train_meshes = {c: shell_meshes(P_tr[y_sub == c], SHELLS, glo, ghi) for c in classes}
    forward = view_vectors()[0]
    depth = lambda X: ((np.median(X, 0) - a) / (b - a)) @ forward              # larger = further from the camera
    canvas = np.max([panel_inches(layout, l, n) for l, n, _, _ in PCA_PANELS if n != 'z1'], 0)
    axes3d = {}
    for layer, name, groups, style in PCA_PANELS:
        ax2 = layout.axes[(layer, name)]
        fig, pos = ax2.figure, ax2.get_position()
        ax2.axis('off')
        full = np.array(panel_inches(layout, layer, name)) if name == 'z1' else canvas
        shift = np.array(SHIFT_3D) + np.array(PANEL_SHIFT_3D.get(name, (0, 0)))
        ax = axes_3d(fig, pos, full, shift, a, b, ZOOM_3D * (1 if name != 'z1' else ZOOM_INSET_3D))
        fs = FINISH_3D.get(name, (1, 0, 0))[0]
        if name in FINISH_3D:
            ax.set_position(page_transform(ax.get_position(), *FINISH_3D[name]))
            pos = page_transform(pos, *FINISH_3D[name])
        objects = [(depth(P_tr[y_sub == c]), train_meshes[c], to_rgb(V.CLASS_COLORS[c]), V.LABELS[c]) for c in classes]
        points, tests = [], []
        for g, color in groups:
            Q = pca.transform(Z_sets[g])
            if style == 'scatter':
                points.append((Q, color))
            else:
                tests.append((depth(Q), shell_meshes(Q, TEST_SHELLS, glo, ghi), to_rgb(color), g))
        z = 1                                               # training shells far to near, then the test sets on top
        for _, meshes, rgb, label in sorted(objects, key=lambda o: -o[0]) + tests:
            z = draw_shells(ax, meshes, rgb, label, a, b, z)
        for Q, color in points:
            sc = ax.scatter(*Q.T, s=(POINT_3D * 0.75) ** 2, color=color, alpha=0.85, linewidths=0, depthshade=False,
                            zorder=z)
            sc.set_gid(f'points_{color}')
        if name == 'z1':                                    # class numerals just above each class's median
            for c in classes:
                m = np.median(P_tr[y_sub == c], 0)
                ax.text(m[0], m[1], m[2] + 0.06 * (b - a)[2], NUMERALS[c], fontsize=6 * fs, ha='center', va='bottom',
                        zorder=z + 1)
        clip = Rectangle((pos.x0, pos.y0), pos.width, pos.height, transform=fig.transFigure, facecolor='none')
        for art in ax.collections + ax.texts:
            art.set_clip_path(clip)
        if name in KEY_PANELS:
            axis_key(ax2, panel_inches(layout, layer, name))
            ax2.set_zorder(ax.get_zorder() + 1)             # the key over the 3D drawing
        axes3d[name] = ax
    targets = {}                                            # ARROWS: test-set medians as drawn
    for layer in {l for l, n, _, _ in PCA_PANELS if n in {p for p, _ in ARROWS.values()}}:
        layout.figures[layer].canvas.draw()
    for pid, (name, g) in ARROWS.items():
        m = np.median(pca.transform(Z_sets[g]), 0)
        targets[pid] = projected_mm(axes3d[name], m, axes3d[name].figure.get_size_inches())
    print(f'  3D PCs: {(100 * pca.explained_variance_ratio_).round(1)} % of latent variance')
    return targets


def main(out):
    r = V.synthetic_results()
    model = V.load_model('synthetic')
    print(f'synthetic model (seed 42): held-out accuracy {V.held_out_accuracy(r):.1f}%', flush=True)
    layout = fifi.svg_to_axes.FigureLayout(flat_template(), autogenlayers=True, make_mplfigures=True, hide_layers=[])


    for i, c in enumerate(ROWS):
        j = np.where(r['id_held'] == EXAMPLES['synthetic'][c])[0][0]
        print(f'  row {i + 1} ({V.LABELS[c]}): {r["id_held"][j]}')
        draw_reconstruction(layout.axes[('A1', f'ai{i + 1}')], layout.axes[('A2', f'ao{i + 1}')], model, r['X_held'][j],
                            ROW_COLORS[c], lw=A_LINE_WIDTH)


    for layer, names, g in EXAMPLE_PANELS:
        ids = list(r['ids_sets'][g])
        print(f'  {g}: {", ".join(EXAMPLES[g])}')
        for name, oid in zip(names, EXAMPLES[g]):
            ax = layout.axes[(layer, name)]
            draw_reconstruction(ax, ax, model, r['X_sets'][g][ids.index(oid)], SET_COLORS[g], recon_color=RECON_COLOR)

    pca = PCA(n_components=3, random_state=0).fit(r['Z_train'])
    print(f'  PCA explained variance: {pca.explained_variance_ratio_.round(3)}')
    sub = np.concatenate([np.where(r['y_train'] == c)[0][:KDE_PER_CLASS] for c in np.unique(r['y_train'])])
    targets = pca_panels_3d(layout, pca, pca.transform(r['Z_train'][sub]), r['y_train'][sub], r['Z_sets'])


    counts = V.synthetic_counts(r)
    for layer, name, cols, x_colors, size in MATRICES:
        plt.rcParams['font.size'] = size
        cnt = counts[cols]
        bars = ([layout.axes[(b, b.lower())] for b in ['CBTRAIN', 'CBTEST', 'CBHEATMAP']] if layer == 'RF1' else
                [plt.subplots(figsize=(1, 2))[1] for _ in range(3)])
        RF.plot_classification_matrix_with_colorbars(
            percentage_matrix=cnt / cnt.sum() * 100, count_matrix=cnt, ax1=layout.axes[(layer, name)], ax2=bars[0],
            ax3=bars[1], ax4=bars[2], y_colors=[V.ROWS[l] for l in cnt.index], x_colors=x_colors,
            show_y_labels=False, show_x_labels=False, colorbar_width=0.005, heatmap_cmap='gist_gray_r',
            colorbar_border=False, show_percentage_cbar=True)
    plt.rcParams['font.size'] = 6
    pct = counts / counts.sum() * 100
    print(pct[[c for _, _, cols, _, _ in MATRICES for c in cols]].round(1).to_string())
    finish_2d(layout)
    for layer in ['A1', 'A2', 'B', 'C', 'C2', 'D1', 'D2', 'E2', 'Z', 'RF1', 'RF2', 'RF3']:   # (not D's colour bar)
        layout.append_figure_to_layer(layout.figures[layer], layer, cleartarget=True)
    layout.write_svg(out)
    aim_arrows(out, targets)
    print(out)


if __name__ == '__main__':
    main(os.path.join(HERE, 'output_figs', 'f4_autoencoder.svg'))
