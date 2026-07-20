"""
Sample real fly trajectories from flies_varwind_C1WT_combined_optotrigger_fancontrol_trimmed.parquet.
One panel per fan speed condition (6 conditions total), all on a single page.
Output: variablewind_sample_trajectories.svg  (8.5 × 11 in, 1 in margins)

Color code:
  grey  = pre-stimulus  (time_relative_to_flash < 0)
  red   = during lights-on (lights_on == 255)
  black = post-stimulus with arrowhead (time_relative_to_flash > 0.5 s)

Trajectory inclusion criteria (all must pass):
  1. Confirmed flash event: duration_of_flash > 0 AND lights_on reaches 255
  2. Trajectory spans t=0 (i.e. includes the flash onset)
  3. At least 3 s of data after t=0 (time_relative_to_flash.max() >= 3.0)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pynumdiff

# ---------------------------------------------------------------------------
# Helpers (arrowhead style matches PlotUtilities.py, scaled for thinner lines)
# ---------------------------------------------------------------------------

def _mean_angle(angle):
    return np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle)))


def _plot_arrowhead_trajectory(x, y, color="black", arrow_length=0.04,
                               arrow_angle=28, ax=None, linewidth=0.5):
    if len(x) < 2:
        return
    xvel = pynumdiff.finite_difference.second_order(x, 1)[1]
    yvel = pynumdiff.finite_difference.second_order(y, 1)[1]
    orientations = np.arctan2(yvel, xvel)
    last_deg = np.rad2deg(_mean_angle(orientations[-3:]))

    ax.plot(x, y, color=color, linewidth=linewidth)

    # Tip placed slightly ahead of last point; wedge opens backward so tip points forward
    cx = x[-1] + np.cos(np.deg2rad(last_deg)) * arrow_length * 0.25
    cy = y[-1] + np.sin(np.deg2rad(last_deg)) * arrow_length * 0.25
    theta1 = last_deg + 180 - arrow_angle / 2
    theta2 = last_deg + 180 + arrow_angle / 2
    wedge = mpatches.Wedge((cx, cy), arrow_length, theta1, theta2)
    pc = PatchCollection([wedge], facecolors=color, edgecolors="none")
    ax.add_collection(pc)


def _plot_trajectory(df, traj_id, ax, xlim, ylim, lw=0.5, arrow_length=0.04,
                     scale_bar=False):
    # Sort by raw frame number — the authoritative chronological order
    t = df[df.obj_id_unique == traj_id].sort_values("frame")

    pre  = t[t["time_relative_to_flash"] < 0]
    dur  = t[t["lights_on"] == 255.0]
    post = t[t["time_relative_to_flash"] > 0.5]   # after flash ends (~0.49 s)

    if len(post) >= 2:
        _plot_arrowhead_trajectory(post["x"].values, post["y"].values,
                                   color="black", ax=ax, linewidth=lw,
                                   arrow_length=arrow_length)
    if len(pre) >= 2:
        ax.plot(pre["x"], pre["y"], color="grey", alpha=0.8, linewidth=lw)
    if len(dur) >= 2:
        ax.plot(dur["x"], dur["y"], color="r", linewidth=lw)

    ax.axis("off")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")

    if scale_bar:
        x0, x1 = xlim
        y0, y1 = ylim
        # Pick a nice bar size: largest of [0.05, 0.1, 0.2] m that is ≤ 25% of x range
        x_range = x1 - x0
        bar_size = 0.05
        for candidate in [0.1, 0.2]:
            if candidate <= 0.25 * x_range:
                bar_size = candidate
        bar_x_end   = x1 - 0.05 * x_range
        bar_x_start = bar_x_end - bar_size
        bar_y       = y0 + 0.08 * (y1 - y0)
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y],
                color="k", linewidth=1.5, solid_capstyle="butt")
        label = f"{int(bar_size * 100)} cm"
        ax.text((bar_x_start + bar_x_end) / 2,
                bar_y - 0.03 * (y1 - y0),
                label, ha="center", va="top", fontsize=5.5, color="k")


# ---------------------------------------------------------------------------
# Style — Helvetica
# ---------------------------------------------------------------------------

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = (
    ["Helvetica", "Helvetica Neue", "Arial"] + plt.rcParams["font.sans-serif"]
)
plt.rcParams["text.usetex"] = False
plt.rcParams["font.weight"] = "normal"

# ---------------------------------------------------------------------------
# Load & filter data
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../Data/Experimental_Fly_Data/"
    "flies_varwind_C1WT_combined_optotrigger_fancontrol_trimmed.parquet",
)
df = pd.read_parquet(DATA_PATH)

# Ensure all trajectories are in strict chronological order
df = df.sort_values(["obj_id_unique", "frame"]).reset_index(drop=True)

# Drop rows where fan speed is unknown
df = df.dropna(subset=["fan_speed_percent"])

# Confirmed flash events only:
#   duration_of_flash > 0  → experiment log says a real flash was delivered
#   lights_on reaches 255  → sensor confirms lights actually turned on
df = df.groupby("obj_id_unique").filter(
    lambda d: (d["duration_of_flash"] > 0).any() and (d["lights_on"] == 255.0).any()
)

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

FAN_SPEED_LABELS = {
    12.0: "0 m/s",
    20.0: "0.03 m/s",
    23.0: "0.1 m/s",
    28.0: "0.2 m/s",
    40.0: "0.3 m/s",
    90.0: "0.6 m/s",
}
FAN_SPEEDS = sorted(FAN_SPEED_LABELS.keys())

# Grid of trajectories within each condition panel
N_TRAJ_COLS = 3
N_TRAJ_ROWS = 4
N_TRAJS = N_TRAJ_COLS * N_TRAJ_ROWS   # 12 per condition
SEED = 42

# ---------------------------------------------------------------------------
# Sample trajectories; compute shared axis limits
# ---------------------------------------------------------------------------

sampled = {}
for fs in FAN_SPEEDS:
    subset = df[df["fan_speed_percent"] == fs]
    # Keep only trajectories that:
    #   • span the flash onset (min time ≤ 0)
    #   • have ≥ 3 s of data after t=0
    qualified = subset.groupby("obj_id_unique").filter(
        lambda d: (
            d["time_relative_to_flash"].min() <= 0
            and d["time_relative_to_flash"].max() >= 3.0
        )
    )
    rng = np.random.default_rng(SEED)
    ids = rng.choice(qualified["obj_id_unique"].unique(), size=N_TRAJS, replace=False)
    sampled[fs] = (qualified, ids)

all_ids   = np.concatenate([ids for _, ids in sampled.values()])
all_data  = pd.concat([sub for sub, _ in sampled.values()])
sel       = all_data[all_data["obj_id_unique"].isin(all_ids)]
pad_x     = (sel["x"].max() - sel["x"].min()) * 0.05
pad_y     = (sel["y"].max() - sel["y"].min()) * 0.05
XLIM      = (sel["x"].min() - pad_x, sel["x"].max() + pad_x)
YLIM      = (sel["y"].min() - pad_y, sel["y"].max() + pad_y)

# ---------------------------------------------------------------------------
# Figure: 8.5 × 11 in, 1 in margins → 6.5 × 9 in content
# Condition panels: 3 rows × 2 cols
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(8.5, 11))

L = 1 / 8.5    # left margin fraction
R = 7.5 / 8.5  # right  (8.5 - 1 in) / 8.5
B = 1 / 11     # bottom
T = 10 / 11    # top    (11 - 1 in) / 11

outer_gs = GridSpec(
    3, 2, figure=fig,
    left=L, right=R, bottom=B, top=T,
    wspace=0.18, hspace=0.30,
)

for i, fs in enumerate(FAN_SPEEDS):
    row, col = divmod(i, 2)
    subset, traj_ids = sampled[fs]

    # Inner gridspec: title row + N_TRAJ_ROWS × N_TRAJ_COLS trajectory grid
    inner_gs = GridSpecFromSubplotSpec(
        N_TRAJ_ROWS + 1, N_TRAJ_COLS,
        subplot_spec=outer_gs[row, col],
        wspace=0.02, hspace=0.02,
        height_ratios=[0.2] + [1] * N_TRAJ_ROWS,
    )

    # Condition label spanning all columns
    ax_title = fig.add_subplot(inner_gs[0, :])
    ax_title.text(
        0.5, 0.15,
        FAN_SPEED_LABELS[fs],
        transform=ax_title.transAxes,
        ha="center", va="bottom",
        fontsize=9, fontweight="bold",
        fontfamily="Helvetica",
    )
    ax_title.axis("off")

    # Trajectory panels; scale bar on the bottom-right subplot of the last panel
    last_condition = (i == len(FAN_SPEEDS) - 1)
    last_slot      = N_TRAJS - 1
    for j, traj_id in enumerate(traj_ids):
        r = j // N_TRAJ_COLS + 1   # +1 to skip title row
        c = j %  N_TRAJ_COLS
        ax = fig.add_subplot(inner_gs[r, c])
        _plot_trajectory(subset, traj_id, ax, xlim=XLIM, ylim=YLIM,
                         lw=0.5, arrow_length=0.04,
                         scale_bar=(last_condition and j == last_slot))

# Legend centered near bottom of content area
legend_handles = [
    mpatches.Patch(color="grey",  alpha=0.8, label="Pre-stimulus"),
    mpatches.Patch(color="red",             label="Lights on"),
    mpatches.Patch(color="black",           label="Post-stimulus"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center", ncol=3,
    fontsize=7, frameon=False,
    bbox_to_anchor=(0.5, 0.02),
)

# ---------------------------------------------------------------------------
# Save as SVG
# ---------------------------------------------------------------------------

OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "variablewind_sample_trajectories.svg",
)
fig.savefig(OUT_PATH, format="svg", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
