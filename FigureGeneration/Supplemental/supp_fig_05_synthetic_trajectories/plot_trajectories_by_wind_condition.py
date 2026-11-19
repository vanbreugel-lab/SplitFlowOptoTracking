"""
Plot 3x3 sample trajectories for each wind condition from the unifying model parquet.
All three conditions appear on a single page PDF.

Color code (matching S5 notebook style):
  grey  = pre-stimulus  (time_relative_to_flash < 0)
  red   = during lights-on (lights_on == 1, ~0-0.66 s)
  black = post-stimulus with arrowhead (time_relative_to_flash > 0.66 s)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import pynumdiff

# ---------------------------------------------------------------------------
# Inline helpers (from SplitFlowOptoTracking_new/Code/PlotUtilities.py)
# ---------------------------------------------------------------------------

def _mean_angle(angle):
    return np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle)))


def _plot_arrowhead_trajectory(x, y, color="black", arrow_length=0.06,
                               arrow_angle=30, ax=None, linewidth=1):
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


def _plot_trajectory(df, traj_id, ax, xlim, ylim):
    lw = 0.85
    t = df[df.obj_id_unique == traj_id].sort_values("time_relative_to_flash")

    pre  = t[t["time_relative_to_flash"] < 0]
    dur  = t[t["lights_on"] == 1.0]
    post = t[t["time_relative_to_flash"] > 0.66]

    if len(post) >= 2:
        _plot_arrowhead_trajectory(post["x"].values, post["y"].values,
                                   color="black", ax=ax, linewidth=lw)
    if len(pre) >= 2:
        ax.plot(pre["x"], pre["y"], color="grey", alpha=0.8, linewidth=lw)
    if len(dur) >= 2:
        ax.plot(dur["x"], dur["y"], color="r", linewidth=lw)

    ax.axis("off")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Matplotlib style (matching notebook)
# ---------------------------------------------------------------------------

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"] + plt.rcParams["font.serif"]
plt.rcParams["text.usetex"] = False
plt.rcParams["font.weight"] = "normal"
plt.rcParams["mathtext.fontset"] = "cm"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../Data/Simulated_Trajectory_Data/new_unifying.parquet",
)
df = pd.read_parquet(DATA_PATH)

WIND_CONDITIONS = ["laminar", "unsteady", "stillair"]
WIND_LABELS = {"laminar": "Laminar", "unsteady": "Unsteady", "stillair": "Still air"}
N_TRAJS = 9
SEED = 42

# ---------------------------------------------------------------------------
# Sample trajectories for all conditions, then compute shared axis limits
# ---------------------------------------------------------------------------

sampled = {}
for condition in WIND_CONDITIONS:
    subset = df[df["windtype"] == condition]
    rng = np.random.default_rng(SEED)
    traj_ids = rng.choice(subset["obj_id_unique"].unique(), size=N_TRAJS, replace=False)
    sampled[condition] = (subset, traj_ids)

# Global limits across every sampled trajectory, with 5 % padding
all_ids = np.concatenate([traj_ids for _, traj_ids in sampled.values()])
all_subsets = pd.concat([subset for subset, _ in sampled.values()])
sel = all_subsets[all_subsets["obj_id_unique"].isin(all_ids)]
pad_x = (sel["x"].max() - sel["x"].min()) * 0.05
pad_y = (sel["y"].max() - sel["y"].min()) * 0.05
XLIM = (sel["x"].min() - pad_x, sel["x"].max() + pad_x)
YLIM = (sel["y"].min() - pad_y, sel["y"].max() + pad_y)

# ---------------------------------------------------------------------------
# Build single-page figure with 3 subfigures (one per condition)
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(10, 14))
subfigs = fig.subfigures(3, 1, hspace=0.03)

for subfig, condition in zip(subfigs, WIND_CONDITIONS):
    subset, traj_ids = sampled[condition]

    subfig.suptitle(WIND_LABELS[condition], fontsize=10, fontweight="bold", x=0.02,
                    ha="left")
    axes = subfig.subplots(3, 3, gridspec_kw={"wspace": 0.01, "hspace": 0.01})

    for i, ax in enumerate(axes.flat):
        _plot_trajectory(subset, traj_ids[i], ax, xlim=XLIM, ylim=YLIM)

# Legend at bottom
legend_handles = [
    mpatches.Patch(color="grey",  alpha=0.8, label="Pre-stimulus"),
    mpatches.Patch(color="red",             label="Lights on"),
    mpatches.Patch(color="black",           label="Post-stimulus"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.005))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "trajectories_by_wind_condition.pdf",
)
fig.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: {OUT_PATH}")
