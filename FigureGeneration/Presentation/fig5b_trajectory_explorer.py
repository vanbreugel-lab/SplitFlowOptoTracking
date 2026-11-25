import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import functools
    import re
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import imageio_ffmpeg
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import animation
    from matplotlib.collections import LineCollection

    from align_course_direction_analysis import unifying_algo_analysis as uaa

    # Use the ffmpeg binary bundled with imageio-ffmpeg so .mp4 export works without a system ffmpeg.
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["svg.fonttype"] = "none"
    return (
        LineCollection,
        Path,
        animation,
        functools,
        mo,
        np,
        pd,
        plt,
        re,
        uaa,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Figure 5B trajectory explorer

    Pick a dataset and a trajectory to plot its **xy path** (top) and **course direction** (bottom) in the style of Figure 5B.
    Choosing a dataset loads the trajectory shown in Figure 5B for that dataset (marked ★ in the trajectory list).

    The unifying-algorithm fit is recomputed with Gurobi using the same settings and seed as Figure 5B,
    so the ★ trajectories reproduce the published panels. Set `GRB_LICENSE_FILE` before launching
    marimo, or switch the fit source to the saved bootstrap fits.
    """)
    return


@app.cell
def _(Path, mo):
    NB_DIR = Path(mo.notebook_dir() or Path(__file__).parent)
    DATA_DIR = (NB_DIR / "../../Data").resolve()
    OUT_DIR = NB_DIR / "output_figs"

    # Figure 5B settings, copied from
    # FigureGeneration/Main/fig_5_unifying_algo_analysis/generate_figure_5/B_figure_example_trajectories.ipynb
    # To add a dataset, add an entry here (fig5_obj_id=None if it is not in Figure 5B).
    DATASETS = {
        "Laminar": dict(
            file="Experimental_Fly_Data/flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf",
            flash_filter=lambda df: df.intensity == 100,
            fig5_obj_id="20220927_170437_3415_122",  # unique()[40] in notebook B
            seed=1,
            flash_frame_end=88,
            saved_fits="laminar_data_flash",
        ),
        "Still air": dict(
            file="Experimental_Fly_Data/flies_stillair_c1xwt_preprocessed_optotrigger_trimmed.hdf",
            flash_filter=lambda df: df.intensity == 100,
            fig5_obj_id="20220815_162329_1835_55",
            seed=2,
            flash_frame_end=88,
            saved_fits="stillair_data_flash",
        ),
        "Unsteady (split flow, top on)": dict(
            file="Experimental_Fly_Data/flies_splitflow_topon_c1xwt_preprocessed_optotrigger_trimmed.hdf",
            flash_filter=lambda df: df.intensity > 0,
            fig5_obj_id="20240110_164820_3734_82",
            seed=3,
            flash_frame_end=88,
            saved_fits="unsteady_top_on_data_flash",
        ),
        "Low wind (variable wind)": dict(
            file="Experimental_Fly_Data/flies_varwind_C1WT_combined_optotrigger_fancontrol_trimmed.parquet",
            flash_filter=lambda df: df.intensity > 0,
            fig5_obj_id="20260406_185025_103927_683",
            seed=4,
            flash_frame_end=70,
            saved_fits="varwind_N700_flash_28.0",
        ),
    }

    # Fit settings used for every Figure 5B panel
    FIT_PARAMS = dict(abs_min_ix=120, abs_max_ix=500, min_ix_range=190, max_ix_range=210,
                      npoints=50, n_bootstraps=10, use_cvx_affine=True, include_translation=True)
    FLASH_FRAME_START = 20  # index of time 0 (flash onset) in every trajectory
    OBJ_ID_KEY = "obj_id_unique_event"
    return (
        DATASETS,
        DATA_DIR,
        FIT_PARAMS,
        FLASH_FRAME_START,
        OBJ_ID_KEY,
        OUT_DIR,
    )


@app.cell
def _(DATASETS, DATA_DIR, OBJ_ID_KEY, functools, pd):
    COLUMNS = [OBJ_ID_KEY, "x", "y", "time_relative_to_flash", "lights_on", "course_smoothish"]

    @functools.cache
    def load_dataset(name):
        """Flash trajectories of one dataset, NaN rows dropped (as in notebook B)."""
        cfg = DATASETS[name]
        path = DATA_DIR / cfg["file"]
        df = pd.read_hdf(path) if path.suffix == ".hdf" else pd.read_parquet(path)
        df = df[cfg["flash_filter"](df)]
        return df[COLUMNS].dropna()
    return (load_dataset,)


@app.cell
def _(DATASETS, mo):
    dataset_select = mo.ui.dropdown(options=list(DATASETS), value="Laminar", label="Dataset")
    return (dataset_select,)


@app.cell
def _(DATASETS, OBJ_ID_KEY, dataset_select, load_dataset, mo):
    dataset_cfg = DATASETS[dataset_select.value]
    with mo.status.spinner(f"Loading {dataset_select.value}..."):
        dataset_df = load_dataset(dataset_select.value)

    _ids = list(dataset_df[OBJ_ID_KEY].unique())
    _fig5 = dataset_cfg["fig5_obj_id"]
    _options = {}
    if _fig5 in _ids:
        _options[f"★ Figure 5B example: {_fig5}"] = _fig5
    _options.update({_id: _id for _id in _ids if _id != _fig5})

    trajectory_select = mo.ui.dropdown(
        options=_options, value=next(iter(_options)), searchable=True,
        label=f"Trajectory ({len(_ids)} in dataset)", full_width=True,
    )
    return dataset_cfg, dataset_df, trajectory_select


@app.cell
def _(dataset_cfg, mo):
    fit_source = mo.ui.radio(
        options=["Re-fit with Gurobi (as in Figure 5B)", "Saved bootstrap fits (no Gurobi)"],
        value="Re-fit with Gurobi (as in Figure 5B)", label="Fit source",
    )
    seed_input = mo.ui.number(start=0, stop=10_000, step=1, value=dataset_cfg["seed"],
                              label="Fit seed (Figure 5B value by default)")

    show_flash = mo.ui.checkbox(value=True, label="Flash (red shading)")
    show_window = mo.ui.checkbox(value=True, label="Fit window (yellow shading)")
    show_linear = mo.ui.checkbox(value=True, label="Linear fit (blue)")
    show_affine = mo.ui.checkbox(value=True, label="Affine fit (magenta)")

    fig_width = mo.ui.number(start=1, stop=20, step=0.25, value=4, label="Width (in)")
    fig_height = mo.ui.number(start=1, stop=20, step=0.25, value=5.5, label="Height (in)")
    font_size = mo.ui.number(start=4, stop=30, step=1, value=12, label="Font size")
    export_dpi = mo.ui.number(start=50, stop=600, step=10, value=200, label="Export dpi (PNG and animation)")
    arrow_size = mo.ui.slider(1, 10, step=0.5, value=4, show_value=True,
                              label="Arrowhead size (% of trajectory)")
    return (
        arrow_size,
        fig_height,
        fig_width,
        fit_source,
        font_size,
        export_dpi,
        seed_input,
        show_affine,
        show_flash,
        show_linear,
        show_window,
    )


@app.cell
def _(
    arrow_size,
    dataset_cfg,
    dataset_select,
    fig_height,
    fig_width,
    fit_source,
    font_size,
    mo,
    export_dpi,
    seed_input,
    show_affine,
    show_flash,
    show_linear,
    show_window,
    trajectory_select,
):
    _is_fig5 = trajectory_select.value == dataset_cfg["fig5_obj_id"]
    _banner = (
        mo.callout(mo.md(f"Showing the **Figure 5B example** for {dataset_select.value}."), kind="success")
        if _is_fig5 else
        mo.callout(mo.md("Not the Figure 5B example. Pick the ★ entry in the trajectory list to return to it."), kind="neutral")
    )
    mo.vstack([
        mo.hstack([dataset_select, trajectory_select], widths=[1, 3]),
        _banner,
        mo.hstack([fit_source, seed_input], justify="start", gap=2),
        mo.md("**Show on course plot**"),
        mo.hstack([show_flash, show_window, show_linear, show_affine], justify="start", gap=2),
        mo.md("**Figure size**"),
        mo.hstack([fig_width, fig_height, font_size, export_dpi], justify="start", gap=2),
        arrow_size,
    ])
    return


@app.cell
def _(
    FIT_PARAMS,
    DATA_DIR,
    OBJ_ID_KEY,
    functools,
    np,
    pd,
    uaa,
):
    @functools.cache
    def _saved_fits_table(name):
        from splitflow.unifying_results import result_path
        try:
            return pd.read_parquet(result_path(f"{name}_translationTrue.parquet"))
        except FileNotFoundError:
            return None

    def saved_fits(name, obj_id):
        table = _saved_fits_table(name)
        if table is None:
            return pd.DataFrame()
        return table[table[OBJ_ID_KEY] == obj_id].reset_index(drop=True)

    @functools.cache
    def refit(course_bytes, seed):
        course = np.frombuffer(course_bytes)
        p = FIT_PARAMS
        return uaa.bootstrap_miop_affine_fit(
            course.copy(), p["abs_min_ix"], p["abs_max_ix"], p["min_ix_range"], p["max_ix_range"],
            npoints=p["npoints"], n_bootstraps=p["n_bootstraps"], use_cvx_affine=p["use_cvx_affine"],
            include_translation=p["include_translation"], seed=seed)

    def get_best_fit(course, obj_id, saved_name, use_saved, seed):
        """Best (lowest rmse_affine) bootstrap fit, plus a note on where it came from."""
        note = ""
        fits = pd.DataFrame()
        if not use_saved:
            try:
                fits = refit(np.ascontiguousarray(course, dtype=float).tobytes(), int(seed))
                note = f"Re-fit with Gurobi, seed {int(seed)}."
            except Exception as e:  # e.g. cvxpy SolverError from a missing/restricted license
                note = f"Gurobi fit failed ({type(e).__name__}: {e}). "
        if len(fits) == 0:
            fits = saved_fits(saved_name, obj_id)
            note += "Using saved bootstrap fits." if len(fits) else "No saved fits for this trajectory."
        if len(fits) == 0:
            return None, note
        return fits.sort_values("rmse_affine").iloc[0], note
    return (get_best_fit,)


@app.cell
def _(
    FLASH_FRAME_START,
    OBJ_ID_KEY,
    dataset_cfg,
    dataset_df,
    fit_source,
    get_best_fit,
    mo,
    np,
    seed_input,
    trajectory_select,
    uaa,
):
    _trajec = dataset_df[dataset_df[OBJ_ID_KEY] == trajectory_select.value]
    _dt = float(np.median(np.diff(_trajec.time_relative_to_flash.values)))
    _course = _trajec.course_smoothish.values.astype(float)

    with mo.status.spinner("Fitting the unifying algorithm..."):
        best_fit, fit_note = get_best_fit(
            _course, trajectory_select.value, dataset_cfg["saved_fits"],
            use_saved=fit_source.value.startswith("Saved"), seed=seed_input.value)

    # Everything the plotting functions need, with time in seconds relative to flash onset
    traj = dict(
        obj_id=trajectory_select.value,
        x=_trajec.x.values, y=_trajec.y.values,
        t=(np.arange(len(_course)) - FLASH_FRAME_START) * _dt,
        dt=_dt,
        lights_on=_trajec.lights_on.values,
        course=_course,
        flash_end_s=(dataset_cfg["flash_frame_end"] - FLASH_FRAME_START) * _dt,
    )
    if best_fit is not None:
        _ix, _, _lin, _aff, _ = uaa.build_miop_and_affine_predictions_from_model(_course.copy(), best_fit)
        _start = int(best_fit.abs_min_ix)
        traj.update(
            fit_t=traj["t"][_start:], linear=_lin[_start:], affine=_aff[_start:],
            window_s=((best_fit.min_ix - FLASH_FRAME_START) * _dt, (best_fit.max_ix - FLASH_FRAME_START) * _dt),
            axis_ratio=float(best_fit.axis_ratio), slope=float(best_fit.slope),
        )
    return fit_note, traj


@app.cell
def _(LineCollection, np, plt):
    from matplotlib.patches import Polygon

    LIGHT ="#cfcfcf"   # unplayed part of the path in animations
    FLASH_RED = "red"
    PRE_GRAY = "gray"

    def nice_scale_bar_length(extent):
        """Largest 1/2/5 x 10^n value that is at most 30% of the extent (as in notebook B)."""
        max_len = 0.3 * extent
        mag = 10 ** np.floor(np.log10(max_len))
        return max(s * mag for s in (1, 2, 5) if s * mag <= max_len)

    ARROW_FRAC = 0.04  # default arrowhead length as a fraction of the trajectory's larger xy extent

    def point_back_along_path(x, y, dist):
        """The point `dist` back from the end of the path, measured along the path (arc length).

        Returns (i, px, py): the point lies between samples i and i+1. Points the arrowhead along
        the stretch of path it actually covers, so small wiggles in the last few samples
        don't swing its direction.
        """
        seg = np.hypot(np.diff(x), np.diff(y))
        back = np.cumsum(seg[::-1])  # back[k]: arc length of the last k+1 segments
        k = int(np.searchsorted(back, dist))
        if k >= len(back):  # path shorter than the arrowhead
            return 0, x[0], y[0]
        i = len(x) - 2 - k
        f = (dist - (back[k - 1] if k > 0 else 0.0)) / seg[i]
        return i, x[i + 1] + f * (x[i] - x[i + 1]), y[i + 1] + f * (y[i] - y[i + 1])

    def arrowhead(base, tip, width, color):
        """Filled triangle with its tip at `tip` and the middle of its base at `base` (data units)."""
        base, tip = np.asarray(base, float), np.asarray(tip, float)
        d = tip - base
        perp = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1.0) * width / 2
        return Polygon([tip, base + perp, base - perp], closed=True, facecolor=color,
                       edgecolor="none", zorder=3)

    def export_pixel_size(opts):
        """Pixel size shared by the PNG and the animation: width/height x dpi, rounded to even numbers."""
        return tuple(2 * max(1, round(v * opts["dpi"] / 2)) for v in (opts["width"], opts["height"]))

    def fit_inside_canvas(fig, pad_in=0.05, n_iter=4):
        """Move the subplot margins so everything drawn (incl. labels) sits inside the canvas.

        Works with the equal-aspect trajectory axis, which tight_layout cannot handle.
        """
        w, h = fig.get_size_inches()
        for _ in range(n_iter):
            fig.canvas.draw()
            bb = fig.get_tightbbox()  # inches
            sp = fig.subplotpars
            fig.subplots_adjust(
                left=sp.left + (pad_in - bb.x0) / w, right=sp.right - (bb.x1 - (w - pad_in)) / w,
                bottom=sp.bottom + (pad_in - bb.y0) / h, top=sp.top - (bb.y1 - (h - pad_in)) / h)

    def style_course_axis(ax, traj, fs):
        ax.set_xlim(traj["t"][0], traj["t"][-1])
        ax.set_ylim(-np.pi, np.pi)
        _fe = traj["flash_end_s"]
        _ticks = [traj["t"][0], 0, _fe, 1, 2, 3, 4, 5]
        _labels = ["", "0", f"{_fe:.2g}".lstrip("0"), "", "2", "3", "4", "5"]
        ax.set_xticks(_ticks, _labels)
        ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], [r"$-\pi$", "", "$0$", "", r"$\pi$"])
        ax.set_xlabel("Time relative to flash (s)", fontsize=fs)
        ax.set_ylabel("Course direction", fontsize=fs, labelpad=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 0.6 * fs))
            ax.spines[side].set_linewidth(0.08 * fs)
        ax.tick_params(labelsize=fs, length=0.35 * fs, width=0.08 * fs, pad=0.25 * fs)

    def draw_figure(traj, opts, animated=False):
        """Two stacked panels: xy trajectory (top) and course direction (bottom).

        animated=False: Figure 5B colouring (gray before flash, red during, black after) with an arrowhead.
        animated=True: the full path and course are drawn light, and the returned `update(i)`
        function colours them in up to frame i (path: red during the flash, black otherwise;
        course: black) and moves the current-frame markers.
        """
        fs = opts["font_size"]
        lw = 0.15 * fs
        x, y, t, course = traj["x"], traj["y"], traj["t"], traj["course"]
        colors = np.where(t < 0, PRE_GRAY, "black").astype(object)
        colors[traj["lights_on"] != 0] = FLASH_RED

        # PNG and animation share this exact canvas: even pixel dimensions (libx264 needs them, so the
        # MP4 is never padded) at the export dpi. The +0.25 px guards against float truncation to px-1.
        dpi = opts["dpi"]
        w_px, h_px = export_pixel_size(opts)
        fig, (ax_xy, ax_c) = plt.subplots(
            2, 1, figsize=((w_px + 0.25) / dpi, (h_px + 0.25) / dpi), dpi=dpi,
            gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.25))

        # --- xy trajectory ------------------------------------------------------
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.axis("off")

        # scale bar under the trajectory
        xr, yr = np.ptp(x), np.ptp(y)
        bar = nice_scale_bar_length(xr)
        bar_y = y.min() - 0.12 * max(xr, yr)
        ax_xy.plot([x.min(), x.min() + bar], [bar_y, bar_y], color="black", linewidth=0.08 * fs * 1.5,
                   solid_capstyle="butt")
        ax_xy.text(x.min() + bar / 2, bar_y - 0.03 * max(xr, yr), f"{bar:g} m",
                   ha="center", va="top", fontsize=fs)
        ax_xy.set_xlim(x.min() - 0.05 * xr, x.max() + 0.05 * xr)
        ax_xy.set_ylim(bar_y - 0.15 * max(xr, yr), y.max() + 0.05 * yr)

        # --- course direction ---------------------------------------------------
        if opts["show_flash"]:
            ax_c.axvspan(0, traj["flash_end_s"], facecolor="red", alpha=0.2, linewidth=0, zorder=-1000)
        if opts["show_window"] and "window_s" in traj:
            ax_c.axvspan(*traj["window_s"], facecolor="yellow", alpha=0.2, linewidth=0, zorder=-1000)

        ms = 0.3 * fs
        ax_c.plot(t, course, ".", color=LIGHT if animated else "black", markersize=ms,
                  markeredgecolor="none", zorder=1)
        if opts["show_linear"] and "linear" in traj:
            ax_c.plot(traj["fit_t"], traj["linear"], ".", color="blue", markersize=0.8 * ms,
                      markeredgecolor="none", zorder=3)
        if opts["show_affine"] and "affine" in traj:
            ax_c.plot(traj["fit_t"], traj["affine"], ".", color="magenta", markersize=0.8 * ms,
                      markeredgecolor="none", zorder=4)
        style_course_axis(ax_c, traj, fs)
        # Lay out before drawing anything that differs between the static and animated versions,
        # so both get identical axes positions and overlay exactly. Labels must fit inside the
        # canvas because exports are the full canvas (no bbox_inches="tight").
        fit_inside_canvas(fig, pad_in=0.04 * fs / 10)

        if not animated:
            # The line stops at the arrowhead's base, and the head points along the stretch of path it covers.
            head_len = opts.get("arrow_frac", ARROW_FRAC) * max(np.ptp(x), np.ptp(y))
            i, bx, by = point_back_along_path(x, y, head_len)
            xs, ys = np.append(x[: i + 1], bx), np.append(y[: i + 1], by)
            segs = np.stack([np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])], axis=1)
            ax_xy.add_collection(LineCollection(segs, colors=colors[1 : i + 2], linewidths=lw, capstyle="round"))
            ax_xy.add_patch(arrowhead((bx, by), (x[-1], y[-1]), width=0.6 * head_len, color=colors[-1]))
            return fig, None

        ax_xy.plot(x, y, color=LIGHT, linewidth=lw, solid_capstyle="round", zorder=1)

        # Played path: one segment per sample step, red during the flash and black otherwise
        # (the pre-flash stretch is black rather than the static plot's gray, so it stands out from the light path).
        all_segs = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
        seg_colors = np.where(colors[1:] == FLASH_RED, FLASH_RED, "black").tolist()
        played_xy = LineCollection([], linewidths=lw, capstyle="round", joinstyle="round", zorder=2)
        ax_xy.add_collection(played_xy)
        head_xy, = ax_xy.plot([], [], "o", color="black", markersize=0.8 * fs, zorder=5)
        played_c, = ax_c.plot([], [], ".", color="black", markersize=ms, markeredgecolor="none", zorder=2)
        head_c, = ax_c.plot([], [], "o", color="black", markersize=0.8 * fs, zorder=5, clip_on=False)

        def update(i):
            played_xy.set_segments(all_segs[:i])
            played_xy.set_color(seg_colors[:i])
            head_xy.set_data([x[i]], [y[i]])
            played_c.set_data(t[: i + 1], course[: i + 1])
            head_c.set_data([t[i]], [course[i]])
            return played_xy, head_xy, played_c, head_c

        return fig, update
    return draw_figure, export_pixel_size


@app.cell
def _(
    arrow_size,
    export_dpi,
    fig_height,
    fig_width,
    font_size,
    show_affine,
    show_flash,
    show_linear,
    show_window,
):
    plot_opts = dict(
        width=fig_width.value, height=fig_height.value, font_size=font_size.value,
        arrow_frac=arrow_size.value / 100, dpi=export_dpi.value,
        show_flash=show_flash.value, show_window=show_window.value,
        show_linear=show_linear.value, show_affine=show_affine.value,
    )
    return (plot_opts,)


@app.cell
def _(dataset_select, plot_opts, re, traj):
    def output_stem():
        """File name stem: dataset, trajectory, and which course-plot layers are shown."""
        layers = [name for key, name in [("show_flash", "flash"), ("show_window", "window"),
                                         ("show_linear", "linear"), ("show_affine", "affine")]
                  if plot_opts[key]]
        ds = re.sub(r"[^A-Za-z0-9]+", "_", dataset_select.value).strip("_").lower()
        return f"{ds}_{traj['obj_id']}_{'-'.join(layers) or 'nolayers'}"
    return (output_stem,)


@app.cell
def _(draw_figure, export_pixel_size, fit_note, mo, plot_opts, plt, traj):
    static_fig, _ = draw_figure(traj, plot_opts)
    plt.close(static_fig)
    _stats = (f"axis ratio {traj['axis_ratio']:.2f}, slope {traj['slope']:.3f} rad/frame"
              if "axis_ratio" in traj else "no fit available")
    _w, _h = export_pixel_size(plot_opts)
    mo.vstack([mo.md(f"<small>{fit_note} Best bootstrap fit: {_stats}. "
                     f"PNG and animation export at exactly {_w} × {_h} px.</small>"), static_fig])
    return (static_fig,)


@app.cell
def _(mo):
    png_background = mo.ui.radio(options=["Transparent", "White"], value="Transparent",
                                 label="PNG background", inline=True)
    save_png_button = mo.ui.run_button(label="Save PNG")
    mo.hstack([png_background, save_png_button], justify="start", gap=2)
    return png_background, save_png_button


@app.cell
def _(OUT_DIR, mo, output_stem, plot_opts, png_background, save_png_button, static_fig):
    mo.stop(not save_png_button.value)
    OUT_DIR.mkdir(exist_ok=True)
    _white = png_background.value == "White"
    # "_white" suffix so the transparent and white versions don't overwrite each other
    _path = OUT_DIR / f"{output_stem()}{'_white' if _white else ''}.png"
    # Full canvas (no tight crop) at the export dpi, so it has the same pixel size as the animation
    static_fig.savefig(_path, dpi=plot_opts["dpi"], transparent=not _white,
                       facecolor="white" if _white else "none")
    mo.vstack([
        mo.md(f"Saved `{_path}`"),
        mo.download(data=_path.read_bytes(), filename=_path.name, mimetype="image/png", label="Download PNG"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Animation

    The full path and course are drawn light gray and turn black as time advances; a large dot marks the current frame.
    Fit lines and shading follow the checkboxes above. The animation is saved to `output_figs/` and shown below.
    """)
    return


@app.cell
def _(mo):
    frame_step = mo.ui.slider(1, 10, value=2, label="Use every Nth frame", show_value=True)
    speed = mo.ui.slider(0.1, 2.0, step=0.05, value=0.5, label="Playback speed (× real time)", show_value=True)
    anim_format = mo.ui.radio(options=["mp4", "gif"], value="mp4", label="Format", inline=True)
    render_button = mo.ui.run_button(label="Render animation", kind="success")
    mo.vstack([
        mo.hstack([frame_step, speed], justify="start", gap=2),
        mo.hstack([anim_format, render_button], justify="start", gap=2),
    ])
    return anim_format, frame_step, render_button, speed


@app.cell
def _(
    OUT_DIR,
    anim_format,
    animation,
    draw_figure,
    frame_step,
    mo,
    output_stem,
    plot_opts,
    plt,
    render_button,
    speed,
    traj,
):
    mo.stop(not render_button.value, mo.md("<small>Click **Render animation** to make the video.</small>"))

    _step = frame_step.value
    _fps = speed.value / (traj["dt"] * _step)
    if anim_format.value == "gif":
        # Browsers slow down GIFs with frame delays under 20 ms, so cap at 50 fps by skipping frames.
        while _fps > 50:
            _step += 1
            _fps = speed.value / (traj["dt"] * _step)
    _frames = list(range(0, len(traj["x"]), _step))
    if _frames[-1] != len(traj["x"]) - 1:
        _frames.append(len(traj["x"]) - 1)

    _fig, _update = draw_figure(traj, plot_opts, animated=True)
    _fig.patch.set_facecolor("white")
    _anim = animation.FuncAnimation(_fig, _update, frames=_frames, blit=True)

    OUT_DIR.mkdir(exist_ok=True)
    _path = OUT_DIR / f"{output_stem()}_animation.{anim_format.value}"
    if anim_format.value == "mp4":
        _writer = animation.FFMpegWriter(
            fps=_fps, codec="libx264", bitrate=-1,
            # yuv420p for QuickTime/PowerPoint. The canvas already has even pixel dimensions (see
            # draw_figure), so no padding is needed and frames match the PNG pixel for pixel.
            extra_args=["-pix_fmt", "yuv420p"])
    else:
        _writer = animation.PillowWriter(fps=_fps)

    with mo.status.progress_bar(total=len(_frames), title="Rendering animation") as _bar:
        _anim.save(_path, writer=_writer, dpi=plot_opts["dpi"],
                   progress_callback=lambda i, n: _bar.update())
    plt.close(_fig)

    _media = (mo.video(src=_path.read_bytes(), controls=True, loop=True, width=int(plot_opts["width"] * 120))
              if anim_format.value == "mp4" else mo.image(src=_path.read_bytes()))
    mo.vstack([
        mo.md(f"Saved `{_path}` ({len(_frames)} frames at {_fps:.1f} fps, every {_step} frame(s))"),
        _media,
        mo.download(data=_path.read_bytes(), filename=_path.name, label=f"Download {anim_format.value.upper()}"),
    ])
    return


if __name__ == "__main__":
    app.run()
