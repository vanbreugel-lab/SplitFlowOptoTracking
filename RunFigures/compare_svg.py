#!/usr/bin/env python
"""Compare a generated figure SVG with a reference (e.g. a former hand-finished copy from git history).

    python RunFigures/compare_svg.py <generated.svg> <reference.svg> [--dpi 100] [--out <dir>]

Prints a structural diff: page size, element ids in only one file, transforms that differ on shared ids, and text
present in only one file. Then renders both files with Inkscape (the same converter as collect_final_figures.sh).
It writes <out>/<name>_diff.png (red: reference only, blue: generated only) and <out>/<name>_sidebyside.png, and
prints the % of changed pixels. <out> defaults to a compare/ folder next to the generated SVG.
"""
import argparse
import collections
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

INKSCAPE = os.environ.get('INKSCAPE') or shutil.which('inkscape') or '/Applications/Inkscape.app/Contents/MacOS/inkscape'
SKIP = {'metadata', 'RDF', 'Work', 'type', 'date', 'format', 'creator', 'Agent', 'title', 'namedview', 'style'}


def local(tag):
    return tag.rsplit('}', 1)[-1]


def summary(path):
    root = ET.parse(path).getroot()
    ids, texts = {}, collections.Counter()
    for e in root.iter():
        if local(e.tag) in SKIP:
            continue
        if e.get('id'):
            ids[e.get('id')] = e.get('transform')
        if local(e.tag) == 'text':
            t = ' '.join(s.strip() for s in e.itertext() if s.strip())
            if t:
                texts[t] += 1
    return root, ids, texts


def structural(gen, ref):
    rg, ig, tg = summary(gen)
    rr, ir, tr = summary(ref)
    size = lambda r: (r.get('width'), r.get('height'), r.get('viewBox'))
    if size(rg) != size(rr):
        print(f'page size differs: generated {size(rg)} vs reference {size(rr)}')
    only_g, only_r = sorted(set(ig) - set(ir)), sorted(set(ir) - set(ig))
    print(f'ids: {len(ig)} generated, {len(ir)} reference, {len(only_g)} only in generated, {len(only_r)} only in reference')
    for name, lst in [('only in reference', only_r), ('only in generated', only_g)]:
        if lst:
            print(f'  {name}: {", ".join(lst[:40])}{" ..." if len(lst) > 40 else ""}')
    moved = [(k, ig[k], ir[k]) for k in set(ig) & set(ir) if ig[k] != ir[k]]
    if moved:
        print(f'transforms differing on {len(moved)} shared ids:')
        for k, a, b in sorted(moved)[:30]:
            print(f'  {k}: generated {a} | reference {b}')
    for name, d in [('text only in reference', tr - tg), ('text only in generated', tg - tr)]:
        if d:
            print(f'{name}: {sorted(d.elements())[:60]}')


def render(svg, png, dpi):
    subprocess.run([INKSCAPE, svg, '--export-type=png', f'--export-filename={png}', f'--export-dpi={dpi}',
                    '--export-background=white', '--export-background-opacity=1'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.asarray(Image.open(png).convert('L'), dtype=float) / 255


def pixels(gen, ref, dpi, out):
    with tempfile.TemporaryDirectory() as tmp:
        g = render(gen, os.path.join(tmp, 'g.png'), dpi)
        r = render(ref, os.path.join(tmp, 'r.png'), dpi)
    if g.shape != r.shape:
        print(f'rendered sizes differ: generated {g.shape} vs reference {r.shape}; padding to compare')
        h, w = max(g.shape[0], r.shape[0]), max(g.shape[1], r.shape[1])
        g, r = (np.pad(a, ((0, h - a.shape[0]), (0, w - a.shape[1])), constant_values=1) for a in (g, r))
    ink = (g < 0.9) | (r < 0.9)
    changed = (np.abs(g - r) > 0.25) & ink
    pct = 100 * changed.sum() / max(ink.sum(), 1)
    print(f'pixels: {pct:.2f}% of inked pixels differ ({changed.sum()} px at {dpi} dpi)')
    rgb = np.ones(g.shape + (3,))
    rgb[..., 0] = np.minimum(1, 1 - (1 - g) * 0.15)     # pale grey where both agree
    rgb[..., 1] = rgb[..., 0]
    rgb[..., 2] = rgb[..., 0]
    rgb[(r < g - 0.25)] = (0.85, 0.1, 0.1)              # ink only in the reference
    rgb[(g < r - 0.25)] = (0.1, 0.3, 0.9)               # ink only in the generated file
    name = os.path.splitext(os.path.basename(gen))[0]
    os.makedirs(out, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(out, f'{name}_diff.png'))
    side = np.concatenate([g, np.zeros((g.shape[0], 4)), r], axis=1)
    Image.fromarray((side * 255).astype(np.uint8)).save(os.path.join(out, f'{name}_sidebyside.png'))
    print(f'wrote {os.path.join(out, name)}_diff.png and _sidebyside.png (left: generated, right: reference)')
    return pct


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('generated')
    ap.add_argument('reference')
    ap.add_argument('--dpi', type=float, default=100)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if not os.path.exists(INKSCAPE):
        sys.exit('Inkscape not found (set INKSCAPE=/path/to/inkscape)')
    structural(a.generated, a.reference)
    pixels(a.generated, a.reference, a.dpi, a.out or os.path.join(os.path.dirname(os.path.abspath(a.generated)), 'compare'))
