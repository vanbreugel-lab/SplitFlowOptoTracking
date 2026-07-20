"""
ghost_tail.py
-------------
Helper functions for matplotlib animations that render a "ghost tail" —
a fading trace of the last N frames — using LineCollection for efficiency.

Two flavors:
  - plot_xy_ghost:   2-D phase-space plots  (x vs y)
  - plot_time_ghost: time-series plots       (time vs value)

Both follow the same three-layer convention:
  1. Full trace    — light gray, plotted once outside the animation loop
  2. Ghost tail    — LineCollection with per-segment alpha, updated each frame
  3. Current point — single dark marker, updated each frame
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_segments(x, y):
    """
    Turn x/y arrays into an (N-1, 2, 2) array of line segments suitable
    for LineCollection:  [ [[x0,y0],[x1,y1]], [[x1,y1],[x2,y2]], ... ]
    """
    points = np.stack([x, y], axis=1)                       # (N, 2)
    return np.stack([points[:-1], points[1:]], axis=1)      # (N-1, 2, 2)


def _alpha_array(n_segments, alpha_max=0.8, alpha_min=0.0):
    """
    Return a linearly-spaced alpha array of length n_segments,
    oldest segment → alpha_min, newest segment → alpha_max.
    """
    return np.linspace(alpha_min, alpha_max, n_segments)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def add_full_trace(ax, x, y, **kwargs):
    """
    Layer 1 — draw the complete trace in light gray.
    Call once before the animation loop.

    Parameters
    ----------
    ax      : matplotlib Axes
    x, y    : full data arrays (for time-series pass time as x, values as y)
    **kwargs: forwarded to ax.plot(); sensible defaults applied

    Returns
    -------
    line : the Line2D artist (rarely needed, but handy)
    """
    kw = dict(color="lightgray", linewidth=1, zorder=1)
    kw.update(kwargs)
    (line,) = ax.plot(x, y, **kw)
    return line


def add_ghost_tail(ax, tail_length=100, color="steelblue",
                   linewidth=2, alpha_max=0.8, zorder=2):
    """
    Layer 2 — create an *empty* LineCollection for the ghost tail.
    Call once before the animation loop, then update with update_ghost_tail().

    Parameters
    ----------
    ax          : matplotlib Axes
    tail_length : max number of past *frames* to show
    color       : tail color (any matplotlib color spec)
    linewidth   : line width
    alpha_max   : opacity of the most-recent segment (oldest → 0)
    zorder      : drawing order

    Returns
    -------
    lc : LineCollection — pass this to update_ghost_tail() each frame
    """
    lc = LineCollection([], linewidth=linewidth, color=color, zorder=zorder)
    ax.add_collection(lc)
    lc._ghost_tail_length = tail_length
    lc._ghost_alpha_max = alpha_max
    return lc


def update_ghost_tail(lc, x, y, frame_idx):
    """
    Layer 2 — update the ghost tail each animation frame.

    Parameters
    ----------
    lc        : LineCollection returned by add_ghost_tail()
    x, y      : full data arrays for the entire animation
    frame_idx : index of the *current* frame (0-based)
    """
    tail_length = lc._ghost_tail_length
    alpha_max   = lc._ghost_alpha_max

    start = max(0, frame_idx - tail_length)
    end   = frame_idx + 1          # include the current point

    xs = x[start:end]
    ys = y[start:end]

    if len(xs) < 2:
        lc.set_segments([])
        return

    segments = _make_segments(xs, ys)
    alphas   = _alpha_array(len(segments), alpha_max=alpha_max)

    lc.set_segments(segments)

    # Build per-segment RGBA colors
    base_color = plt.matplotlib.colors.to_rgb(lc.get_color()[0])
    rgba = np.array([[*base_color, a] for a in alphas])
    lc.set_color(rgba)


def add_current_point(ax, color="black", marker="o",
                      markersize=8, zorder=3, **kwargs):
    """
    Layer 3 — create an *empty* artist for the current-frame point.
    Call once before the animation loop, then update with update_current_point().

    Parameters
    ----------
    ax         : matplotlib Axes
    color      : marker color
    marker     : marker style
    markersize : marker size
    zorder     : drawing order
    **kwargs   : forwarded to ax.plot()

    Returns
    -------
    point : Line2D — pass this to update_current_point() each frame
    """
    kw = dict(color=color, marker=marker, markersize=markersize,
              linestyle="none", zorder=zorder)
    kw.update(kwargs)
    (point,) = ax.plot([], [], **kw)
    return point


def update_current_point(point, x, y, frame_idx):
    """
    Layer 3 — move the current-point marker each animation frame.

    Parameters
    ----------
    point     : Line2D returned by add_current_point()
    x, y      : full data arrays
    frame_idx : index of the current frame (0-based)
    """
    point.set_data([x[frame_idx]], [y[frame_idx]])


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────

def setup_xy_ghost_plot(ax, x, y, tail_length=100,
                        trace_kw=None, tail_kw=None, point_kw=None):
    """
    One-shot setup for a 2-D x-y phase plot.

    Creates all three layers and returns the artists you need to update
    inside the animation loop.

    Parameters
    ----------
    ax          : matplotlib Axes
    x, y        : full data arrays
    tail_length : frames of history to show
    trace_kw    : dict of kwargs for add_full_trace()
    tail_kw     : dict of kwargs for add_ghost_tail()
    point_kw    : dict of kwargs for add_current_point()

    Returns
    -------
    lc    : LineCollection  →  update_ghost_tail(lc, x, y, frame_idx)
    point : Line2D          →  update_current_point(point, x, y, frame_idx)
    """
    add_full_trace(ax, x, y, **(trace_kw or {}))
    lc    = add_ghost_tail(ax, tail_length=tail_length, **(tail_kw or {}))
    point = add_current_point(ax, **(point_kw or {}))
    return lc, point


def setup_time_ghost_plot(ax, t, v, tail_length=100,
                          trace_kw=None, tail_kw=None, point_kw=None):
    """
    One-shot setup for a time-series plot (time on x-axis, value on y-axis).

    Identical interface to setup_xy_ghost_plot(); just aliases t→x, v→y
    for readability.

    Returns
    -------
    lc    : LineCollection  →  update_ghost_tail(lc, t, v, frame_idx)
    point : Line2D          →  update_current_point(point, t, v, frame_idx)
    """
    return setup_xy_ghost_plot(ax, t, v, tail_length=tail_length,
                               trace_kw=trace_kw, tail_kw=tail_kw,
                               point_kw=point_kw)
