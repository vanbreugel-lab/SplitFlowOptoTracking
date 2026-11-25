"""
Plot sample flash-trial trajectories from the laminar and still-air datasets,
original vs. SHUFFLED_, in the style of
FigureGeneration/Supplemental/supp_fig_04_real_trajectories/combined_real_trajectories.svg

Four panels: rows = laminar / still air, columns = original / shuffled. Each
row shows the same trajectory IDs in both columns.

Output: shuffled_trajectories.svg (8.5 x 11 in) in this directory.

Run after make_shuffled_datasets.py:
    python plot_shuffled_trajectories.py [--seed 42]

Unlike the real-trajectory figure, each trajectory is centered on its own
bounding box: shuffled trajectories are not constrained by the arena walls and
drift far outside the arena coordinates.

Each PANEL has its own scale (shown by its own scale bar): the window is
auto-sized so every trajectory in the panel fits, but never smaller than the
real-trajectory figure's +/-0.5 x +/-0.25 m. Shuffled flies travel much farther
than real ones, so shuffled panels are drawn zoomed out; compare distances via
the scale bars, not by eye.

Options:
    --half-width W   use one fixed +/-W x +/-W/2 m window for all panels
                     (trajectories larger than this get clipped)
    --arena-coords   absolute arena coordinates, fixed +/-0.5 m window
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from splitflow.data_paths import data_path

HERE = Path(__file__).resolve().parent

# ── Layout / style (matches combined_real_trajectories) ──────────────────────
N_TRAJ_COLS = 3
N_TRAJ_ROWS = 6
PANEL_SPAN = 2                 # outer grid rows per panel (2x the 3-row panels of the real figure)
N_TRAJS = N_TRAJ_COLS * N_TRAJ_ROWS
MAX_TIME_SEC = 4
FLASH_END_SEC = 0.675
MIN_GS = 0.15                  # min mean ground speed (m/s) for selection
LW = 0.5
ARROW_LENGTH = 0.04
MIN_HALF_WIDTH = 0.5           # m; smallest panel window = real-figure scale
WINDOW_PAD = 1.05              # margin around the largest trajectory in a panel
SCALE_BAR_SIZES = [0.1, 0.2, 0.5, 1.0]   # m

CONDITIONS = [
    ("Laminar flow (0.4 m/s)", "laminar"),   # wind: fly.<wind>_c1xwt and shuffled.flies in Data/data_paths.yaml
    ("Still air", "stillair"),
]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica"] + plt.rcParams["font.sans-serif"]
plt.rcParams["text.usetex"] = False
plt.rcParams["svg.fonttype"] = "none"


def _mean_angle(angles):
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def _plot_arrowhead_trajectory(x, y, ax, color="black", arrow_length=0.04,
                               arrow_angle=28, linewidth=0.5):
    if len(x) < 2:
        return
    heading = np.arctan2(np.gradient(y), np.gradient(x))
    last_deg = np.rad2deg(_mean_angle(heading[-3:]))
    ax.plot(x, y, color=color, linewidth=linewidth)
    cx = x[-1] + np.cos(np.deg2rad(last_deg)) * arrow_length * 0.25
    cy = y[-1] + np.sin(np.deg2rad(last_deg)) * arrow_length * 0.25
    wedge = mpatches.Wedge((cx, cy), arrow_length,
                           last_deg + 180 - arrow_angle / 2,
                           last_deg + 180 + arrow_angle / 2)
    ax.add_collection(PatchCollection([wedge], facecolors=color, edgecolors="none"))


def get_trajectory(df, traj_id, center=True):
    """One trajectory, chronological, clipped to MAX_TIME_SEC, optionally bbox-centered."""
    t = (df[df["obj_id_unique_event"] == traj_id]
         .sort_values("frame")
         .dropna(subset=["x", "y"])
         .loc[lambda d: d["t_sec"] <= MAX_TIME_SEC]
         .copy())
    if center:
        t["x"] -= (t["x"].max() + t["x"].min()) / 2
        t["y"] -= (t["y"].max() + t["y"].min()) / 2
    return t


def fit_half_width(trajs):
    """x half-width of a 2:1 window that fits every (centered) trajectory."""
    need = max(max((t["x"].max() - t["x"].min()) / 2, t["y"].max() - t["y"].min())
               for t in trajs)
    return max(MIN_HALF_WIDTH, need * WINDOW_PAD)


def plot_trajectory(t, ax, hw):
    xlim, ylim = (-hw, hw), (-hw / 2, hw / 2)
    arrow_length = ARROW_LENGTH * hw / MIN_HALF_WIDTH   # same size on paper in every panel

    pre = t[t["t_sec"] < 0]
    dur = t[(t["t_sec"] >= 0) & (t["t_sec"] <= FLASH_END_SEC)]
    post = t[t["t_sec"] > FLASH_END_SEC]

    if len(post) >= 2:
        _plot_arrowhead_trajectory(post["x"].values, post["y"].values, ax,
                                   linewidth=LW, arrow_length=arrow_length)
    if len(pre) >= 2:
        ax.plot(pre["x"], pre["y"], color="grey", alpha=0.8, linewidth=LW)
    if len(dur) >= 2:
        ax.plot(dur["x"], dur["y"], color="r", linewidth=LW)

    ax.axis("off")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")


def add_scale_bar(ax, hw, fontsize=5.5):
    """Scale bar just below the bottom-right trajectory axes of a panel (same data scale)."""
    x0, x1 = -hw, hw
    y0, y1 = -hw / 2, hw / 2
    bar = max(b for b in SCALE_BAR_SIZES if b <= 0.3 * (x1 - x0))
    xe = x1 - 0.05 * (x1 - x0)
    yb = y0 - 0.08 * (y1 - y0)
    ax.plot([xe - bar, xe], [yb, yb], color="k", linewidth=1.5,
            solid_capstyle="butt", clip_on=False)
    ax.text(xe - bar / 2, yb - 0.05 * (y1 - y0), f"{int(round(bar * 100))} cm",
            ha="center", va="top", fontsize=fontsize, clip_on=False)


def load(path):
    df = pd.read_hdf(path)
    df = df[df["intensity"] == 100]          # flash trials only (sham: intensity == 0)
    df = df.copy()
    df["t_sec"] = df["time_relative_to_flash"]
    return df


def qualified_ids(df):
    """Spans flash onset, >= 3 s of valid post-flash data, mean ground speed >= MIN_GS."""
    v = df.dropna(subset=["x", "y"])
    g = v.groupby("obj_id_unique_event")
    stats = g.agg(tmin=("t_sec", "min"), tmax=("t_sec", "max"), gs=("speed_xy", "mean"))
    ok = stats[(stats.tmin <= 0) & (stats.tmax >= 3.0) & (stats.gs >= MIN_GS)]
    return np.sort(ok.index.values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arena-coords", action="store_true")
    parser.add_argument("--half-width", type=float, default=None,
                        help="x half-width of each trajectory window in m (y is half of this)")
    args = parser.parse_args()
    fixed_hw = MIN_HALF_WIDTH if args.arena_coords else args.half_width
    rng = np.random.default_rng(args.seed)

    # Panels in row-major order: one row per condition, [original, shuffled],
    # both showing the same trajectory IDs. (title, df, traj_ids)
    panels = []
    for label, wind in CONDITIONS:
        orig = load(data_path(f"fly.{wind}_c1xwt"))
        shuf = load(data_path("shuffled.flies", wind=wind))
        ids = qualified_ids(orig)   # speed/NaN pattern identical in shuffled copy
        n = min(N_TRAJS, len(ids))
        chosen = rng.choice(ids, size=n, replace=False)
        print(f"{label:25s} qualified={len(ids):4d}  plotted={n}")
        panels.append((f"{label}: original", orig, chosen))
        panels.append((f"{label}: SHUFFLED", shuf, chosen))

    n_cols = 2
    n_rows = (len(panels) + 1) // n_cols
    fig = plt.figure(figsize=(8.5, 11))
    outer = GridSpec(max(n_rows * PANEL_SPAN, 5), n_cols, figure=fig,
                     left=1 / 8.5, right=7.5 / 8.5, bottom=1 / 11, top=10 / 11,
                     wspace=0.15, hspace=0.32)

    for i, (title, df, ids) in enumerate(panels):
        pr, pc = divmod(i, n_cols)
        # last (empty) row leaves room for the panel's scale bar
        inner = GridSpecFromSubplotSpec(N_TRAJ_ROWS + 2, N_TRAJ_COLS,
                                        subplot_spec=outer[pr * PANEL_SPAN:(pr + 1) * PANEL_SPAN, pc],
                                        wspace=0.02, hspace=0.02,
                                        height_ratios=[0.2] + [1] * N_TRAJ_ROWS + [0.3])
        ax_t = fig.add_subplot(inner[0, :])
        ax_t.text(0.5, 0.15, title, transform=ax_t.transAxes,
                  ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax_t.axis("off")
        trajs = [get_trajectory(df, tid, center=not args.arena_coords) for tid in ids]
        hw = fixed_hw if fixed_hw is not None else fit_half_width(trajs)
        print(f"  {title:35s} window = +/-{hw:.2f} x +/-{hw / 2:.2f} m")
        for j, t in enumerate(trajs):
            ax = fig.add_subplot(inner[j // N_TRAJ_COLS + 1, j % N_TRAJ_COLS])
            plot_trajectory(t, ax, hw)
        add_scale_bar(ax, hw)   # bottom-right trajectory of the panel

    suffix = "_arena" if args.arena_coords else ""
    out = HERE / f"shuffled_trajectories{suffix}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
