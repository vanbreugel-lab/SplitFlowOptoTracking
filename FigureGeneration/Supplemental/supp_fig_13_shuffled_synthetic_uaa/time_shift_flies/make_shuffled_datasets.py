"""
Create time-shuffled ("SHUFFLED_") copies of the laminar and still-air C1xWT
optotrigger datasets.

For every trajectory (obj_id_unique_event):
  1. Take the angular velocity time series (ang_vel_smoothish) and randomly
     permute it in time.
  2. Integrate the shuffled angular velocity, starting from the trajectory's
     original initial course, and wrap to [-pi, pi] -> new course direction.
  3. Rebuild xvel / yvel from the new course and the ORIGINAL ground speed
     (speed_xy), then integrate those from the original initial (x, y) to get
     new positions.

Columns overwritten: x, y, xvel, yvel, course, course_smoothish,
ang_vel_smoothish. Everything else (z, zvel, speed_xy, timing, flash/sham
metadata, ...) is copied unchanged. Trailing NaN padding rows stay NaN.

Reads Data/Experimental_Fly_Data/, writes Data/Shuffled_Flies/.

Run from anywhere:
    python make_shuffled_datasets.py [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]  # <repo>/FigureGeneration/Supplemental/supp_fig_13_shuffled_synthetic_uaa/<this dir>/<this file>
DATA_DIR = REPO_ROOT / "Data" / "Experimental_Fly_Data"
OUT_DIR = REPO_ROOT / "Data" / "Shuffled_Flies"

FILENAMES = [
    "flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf",
    "flies_stillair_c1xwt_preprocessed_optotrigger_trimmed.hdf",
]

OBJ_ID_KEY = "obj_id_unique_event"
DT = 0.01  # s; 100 Hz, matches diff of time_relative_to_flash


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def shuffle_trajectory(g, rng, dt=DT):
    """Return x, y, xvel, yvel, course, angvel arrays for one event (sorted by frame)."""
    n = len(g)
    out = {k: np.full(n, np.nan) for k in ["x", "y", "xvel", "yvel", "course", "angvel"]}

    valid = g[["x", "y", "speed_xy", "course_smoothish", "ang_vel_smoothish"]].notna().all(axis=1).values
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return out
    # Only trailing NaN padding is expected; use the contiguous valid block.
    assert np.all(np.diff(idx) == 1), "non-contiguous valid samples within an event"

    angvel = rng.permutation(g["ang_vel_smoothish"].values[idx])
    speed = g["speed_xy"].values[idx]
    course0 = g["course_smoothish"].values[idx[0]]
    x0, y0 = g["x"].values[idx[0]], g["y"].values[idx[0]]

    # Forward-Euler integration; sample 0 keeps the original initial state.
    course_unwrapped = course0 + np.concatenate([[0.0], np.cumsum(angvel[:-1] * dt)])
    xvel = speed * np.cos(course_unwrapped)
    yvel = speed * np.sin(course_unwrapped)
    x = x0 + np.concatenate([[0.0], np.cumsum(xvel[:-1] * dt)])
    y = y0 + np.concatenate([[0.0], np.cumsum(yvel[:-1] * dt)])

    for k, v in zip(["x", "y", "xvel", "yvel", "course", "angvel"],
                    [x, y, xvel, yvel, wrap(course_unwrapped), angvel]):
        out[k][idx] = v
    return out


def shuffle_dataset(df, rng):
    """Return a copy of df with shuffled kinematics; original row order/index kept."""
    df = df.copy()
    # Positional sort by (event, frame); results are written back to the same positions.
    order = np.lexsort((df["frame"].values, df[OBJ_ID_KEY].values))
    sorted_df = df.iloc[order]
    cols = {k: np.full(len(df), np.nan) for k in ["x", "y", "xvel", "yvel", "course", "angvel"]}

    pos = 0
    for _, g in sorted_df.groupby(OBJ_ID_KEY, sort=False):
        res = shuffle_trajectory(g, rng)
        for k in cols:
            cols[k][order[pos:pos + len(g)]] = res[k]
        pos += len(g)

    df["x"] = cols["x"]
    df["y"] = cols["y"]
    df["xvel"] = cols["xvel"]
    df["yvel"] = cols["yvel"]
    df["course"] = cols["course"]
    df["course_smoothish"] = cols["course"]
    df["ang_vel_smoothish"] = cols["angvel"]
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname in FILENAMES:
        src = DATA_DIR / fname
        dst = OUT_DIR / f"SHUFFLED_{fname}"
        with pd.HDFStore(src, "r") as store:
            key = store.keys()[0]
        df = pd.read_hdf(src, key=key)
        shuffled = shuffle_dataset(df, rng)

        # Sanity checks: same shape/NaN pattern, ground speed preserved.
        assert shuffled.shape == df.shape
        assert (shuffled["x"].isna().values == df["x"].isna().values).all()
        assert np.allclose(np.hypot(shuffled.xvel.values, shuffled.yvel.values),
                           df["speed_xy"].values, equal_nan=True)

        shuffled.to_hdf(dst, key=key, mode="w")
        print(f"{fname}: {df[OBJ_ID_KEY].nunique()} trajectories -> {dst.name}")


if __name__ == "__main__":
    main()
