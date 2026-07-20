"""
animate_fifi.py
---------------
Frame-by-frame animation using figurefirst + ghost_tail helpers.

Pipeline per frame
------------------
  1. Clear & redraw all matplotlib artists on the axes
  2. Embed the updated mpl figure into the SVG layout via figurefirst
  3. Write the composite SVG to disk
  4. Rasterise that SVG to a PNG via cairosvg

Why a plain for-loop instead of FuncAnimation?
  figurefirst's write cycle (append_figure_to_layer → write_svg) is
  file-based and has no concept of a live display loop.  FuncAnimation
  is designed for interactive/video rendering — it calls blit or draw
  but never writes files.  A for-loop gives you full control and makes
  the SVG→PNG step trivial to slot in.

Dependencies
------------
  pip install figurefirst cairosvg matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import figurefirst as fifi
import cairosvg
from pathlib import Path

from splitflow.ghost_tail import (
    setup_xy_ghost_plot,
    setup_time_ghost_plot,
    update_ghost_tail,
    update_current_point,
)

# ── Output directory ──────────────────────────────────────────────────────────
OUT = Path("frames")
OUT.mkdir(exist_ok=True)

SVG_TEMPLATE = "animation_figure.svg"   # your figurefirst layout file

# ── Synthetic data ────────────────────────────────────────────────────────────
N    = 400
t    = np.linspace(0, 4 * np.pi, N)
x    = np.cos(t) * (1 + 0.3 * np.sin(5 * t))
y    = np.sin(t) * (1 + 0.3 * np.cos(5 * t))
v    = np.sin(t) + 0.3 * np.random.default_rng(0).standard_normal(N)
TAIL = 80


def build_layout():
    """
    (Re-)construct the figurefirst layout and return axes handles.

    figurefirst embeds mpl figures directly into the SVG document tree,
    so the layout object must be re-created each frame — it is not
    cheaply reusable across write cycles.
    """
    layout = fifi.svg_to_axes.FigureLayout(
        SVG_TEMPLATE,
        autogenlayers=True,
        make_mplfigures=True,
        hide_layers=[],
    )
    plt.close("all")   # figurefirst opens figures internally; close stale ones

    ax_xy = layout.axes[("animation", "xy")]
    ax_ts = layout.axes[("animation", "time")]
    return layout, ax_xy, ax_ts


def render_frame(frame_idx: int, stride: int = 1):
    """
    Render a single frame.

    Parameters
    ----------
    frame_idx : 0-based index into the data arrays
    stride    : only write every nth frame (default = every frame)

    Returns
    -------
    svg_path : Path to the written SVG
    png_path : Path to the written PNG
    """
    if frame_idx % stride != 0:
        return None, None

    # ── 1. Build fresh layout + axes ─────────────────────────────────────────
    layout, ax_xy, ax_ts = build_layout()

    # ── 2. Draw all three layers on each axis ─────────────────────────────────
    # Phase plot
    lc_xy, pt_xy = setup_xy_ghost_plot(
        ax_xy, x, y,
        tail_length=TAIL,
        tail_kw=dict(color="royalblue", linewidth=2, alpha_max=0.85),
        point_kw=dict(color="navy", markersize=7),
    )
    ax_xy.set_xlim(-1.6, 1.6)
    ax_xy.set_ylim(-1.6, 1.6)
    ax_xy.set_aspect("equal")

    update_ghost_tail(lc_xy, x, y, frame_idx)
    update_current_point(pt_xy, x, y, frame_idx)

    # Time-series plot
    lc_ts, pt_ts = setup_time_ghost_plot(
        ax_ts, t, v,
        tail_length=TAIL,
        tail_kw=dict(color="tomato", linewidth=2, alpha_max=0.85),
        point_kw=dict(color="darkred", markersize=7),
    )
    ax_ts.set_xlim(t[0], t[-1])
    ax_ts.set_ylim(-2, 2)

    update_ghost_tail(lc_ts, t, v, frame_idx)
    update_current_point(pt_ts, t, v, frame_idx)

    # ── 3. Embed mpl figure into the SVG layout ───────────────────────────────
    layout.append_figure_to_layer(
        layout.figures["animation"], "animation", cleartarget=True
    )

    svg_path = OUT / f"frame_{frame_idx:04d}.svg"
    layout.write_svg(str(svg_path))

    # ── 4. Rasterise SVG → PNG ────────────────────────────────────────────────
    png_path = svg_path.with_suffix(".png")
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=150)

    plt.close("all")   # prevent figure accumulation
    return svg_path, png_path


# ── Main loop ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    STRIDE = 4   # render every 4th frame; set to 1 for every frame

    for i in range(N):
        svg_path, png_path = render_frame(i, stride=STRIDE)
        if png_path:
            print(f"  frame {i:04d} → {png_path}")

    print(f"\nDone. Frames written to {OUT.resolve()}/")
    print("Combine into a video with e.g.:")
    print("  ffmpeg -r 30 -pattern_type glob -i 'frames/*.png' -c:v libx264 out.mp4")
