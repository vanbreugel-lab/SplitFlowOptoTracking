"""
Shared code (copied from FigureGeneration/Main/fig_5_unifying_algo_analysis/generate_figure_5/slope_r2_helper.py;
used here by own_fit_helper.py for file lookup, data loading, plotting constants and SVG output).

Computation and plotting follow (and are duplicated from)
    FigureGeneration/Supplemental/supp_fig_11_unifying_flies_shams/D_slope_vs_r2_flies_allwtsplitflow.ipynb
    FigureGeneration/Supplemental/supp_fig_12_variable_wind_unifying/E_slope_vs_rmse_var_wind.ipynb
    FigureGeneration/Supplemental/supp_fig_13_shuffled_synthetic_uaa/E_slope_vs_r2_shuffled_synthetic.ipynb
(PERIOD_SHIFTS = [0] by default; a per-dataset list can be passed to compute_slope_r2,
as notebook E does for variable-wind flash at 60 cm/s).

All paths are relative to this folder; run the notebook from its own directory.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from matplotlib.patches import Ellipse

import figurefirst as fifi
from align_course_direction_analysis import unifying_algo_analysis as uaa

FIGURE_NAME = 'unifying_analysis_translationTrue.svg'
TRANSLATION = True

DATA_DIR = Path('../../../Data')
TRAJEC_DIR = DATA_DIR / 'Experimental_Fly_Data'
SHUFFLED_SYNTHETIC_DIR = Path('../supp_fig_13_shuffled_synthetic_uaa')

# Average model window: columns of the aligned arrays (-300..300 samples at 0.01 s), 200:400 = aligned time -1..1 s
ROI = slice(200, 400)
MIN_SAMPLES = 160  # trajectories with data for less than 80% of the ROI (1.6 of 2 s) are dropped
PERIOD_SHIFTS = [0]  # default: score each trajectory at its default alignment only
# Where R2 is scored. 'aligned': aligned time -1..1 s (ROI above) for every trajectory.
# 'fit_roi' (option D in surge_r2_window/): each trajectory's own fit ROI (min_ix:max_ix of its best fit), so
# pre-fit surge flight that the -1..1 s window can reach back into is excluded. Period shifts don't apply.
R2_WINDOW = 'aligned'

MARKERSIZE = 0.75
ALPHA = 0.5

# Confidence ellipse around the (median x, median slope)
ELLIPSE_CONFIDENCE = 0.95
N_BOOTSTRAP = 2000
ELLIPSE_ALPHA = 0.3
RNG_SEED = 0

R2_POS_SCALE = 3.0  # stretch factor for R2 >= 0 on the plotted axis
XLIM = (-5.5, R2_POS_SCALE)  # transformed units; -5.5 corresponds to R2 of about -244
R2_TICKS = [-100, -10, -1, 0, 0.5, 1]
R2_TICKLABELS = ['-100', '-10', '', '0', '', '1']
X_LABEL = 'Circular $R^2$ to mean model'

YLIM = (0, 8)  # weighted |slope| (rad/s)
YTICKS = {(0, 8): ([0, 2, 4, 6, 8], ['0', '', '', '', '8']),
          (0, 30): ([0, 10, 20, 30], ['0', '', '', '30'])}


# --------------------------------------------------------------------------------------------------
# File lookup
# --------------------------------------------------------------------------------------------------
def unifying_results_path(filename):
    """
    Locate a unifying-algo results parquet. The local checkout has a flat Data/unifying_algo_results/;
    the Dryad layout splits it into Data/Unifying_Algo_Results/{Main,Supplemental}/.
    """
    candidates = [DATA_DIR / 'unifying_algo_results' / filename,
                  DATA_DIR / 'Unifying_Algo_Results' / 'Main' / filename,
                  DATA_DIR / 'Unifying_Algo_Results' / 'Supplemental' / filename]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(filename)


def fly_trajec_filename(unifying_filename):
    """Trajectory file matching a fly unifying-results file (laminar / unsteady / still air datasets)."""
    name = Path(unifying_filename).name
    if 'WT_laminar' in name:
        return TRAJEC_DIR / 'flies_laminar_wt_preprocessed_optotrigger_trimmed.hdf'
    if 'unsteady_top_on_data_WT' in name:  # includes experiencedunsteady_
        return TRAJEC_DIR / 'flies_splitflow_topon_wt_preprocessed_optotrigger_trimmed.parquet'
    if 'WT_stillair' in name:
        return TRAJEC_DIR / 'flies_stillair_wt_preprocessed_optotrigger_trimmed.hdf'
    if name.startswith('laminar_data'):
        return TRAJEC_DIR / 'flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf'
    if name.startswith('stillair_data'):
        return TRAJEC_DIR / 'flies_stillair_c1xwt_preprocessed_optotrigger_trimmed.hdf'
    if 'unsteady' in name and 'flash' in name:
        return TRAJEC_DIR / 'flies_splitflow_topon_c1xwt_preprocessed_optotrigger_trimmed.hdf'
    if 'unsteady' in name and 'sham' in name:
        return TRAJEC_DIR / 'flies_splitflow_topon_c1xwt_preprocessed_optotrigger_trimmed_SHAM_AUGMENTED_merged.parquet'
    raise ValueError('No trajectory file known for ' + name)


# Figure 5 colors for the variable wind data: all speeds light purple, 20 cm/s (fan setting 28) highlighted.
# Gray is reserved for sham data.
VARWIND_COLOR = '#c4aee0'
VARWIND_HIGHLIGHT_COLOR = '#a245ff'
VARWIND_HIGHLIGHT_SPEED = '28'  # 20 cm/s

VARWIND_TRAJEC_FILENAME = TRAJEC_DIR / 'flies_varwind_C1WT_combined_optotrigger_fancontrol_trimmed.parquet'
VARWIND_SPEEDS = ['12', '20', '23', '28', '40', '90']
VARWIND_SPEED_LABELS = {'12': 0, '20': 3, '23': 10, '28': 20, '40': 30, '90': 60}  # cm/s
# Sham uses Kevin's analyzed files, flash uses the N700 files (as in the variable-wind supplemental figure)
VARWIND_SHAM_FILENAMES = {'12': 'var-wind-12-fan_0-cms_all-traj_align_SHAM.parquet',
                          '20': 'var-wind-20-fan_0-03-cms_all-traj_align_SHAM.parquet',
                          '23': 'var-wind-23-fan_0-1-cms_all-traj_align_SHAM.parquet',
                          '28': 'var-wind-28-fan_0-2-cms_all-traj_align_SHAM.parquet',
                          '40': 'var-wind-40-fan_0-3-cms_all-traj_align_SHAM.parquet',
                          '90': 'var-wind-90-fan_0-6-cms_all-traj_align_SHAM.parquet'}


def varwind_unifying_filename(condition, speed):
    if condition == 'sham':
        return unifying_results_path(Path('var_wind_kevin_analyzed') / VARWIND_SHAM_FILENAMES[speed])
    return unifying_results_path('varwind_N700_%s_%s.0_translation%s.parquet' % (condition, speed, TRANSLATION))


def shuffled_synthetic_scenarios():
    """Import scenarios.py from the shuffled/synthetic supplemental folder; returns the module."""
    folder = str(SHUFFLED_SYNTHETIC_DIR.resolve())
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import scenarios
    return scenarios


def shuffled_synthetic_filenames(key):
    """(unifying results, trajectories) for a shuffled/synthetic scenario, relative to this folder."""
    scenarios = shuffled_synthetic_scenarios()
    return (SHUFFLED_SYNTHETIC_DIR / scenarios.get_unifying_filename(key),
            SHUFFLED_SYNTHETIC_DIR / scenarios.get_trajec_filename(key))


_trajec_cache = {}
def load_trajectories(filename):
    filename = str(filename)
    if filename not in _trajec_cache:
        if filename.endswith('.hdf'):
            _trajec_cache[filename] = pd.read_hdf(filename)
        else:
            _trajec_cache[filename] = pd.read_parquet(filename)
    return _trajec_cache[filename]


# --------------------------------------------------------------------------------------------------
# Computation
# --------------------------------------------------------------------------------------------------
def mean_angle(angle):
    """Circular mean ignoring NaNs."""
    return np.arctan2(np.nanmean(np.sin(angle)), np.nanmean(np.cos(angle)))


def r2_axis_transform(r2, pos_scale=R2_POS_SCALE):
    """pos_scale*R2 for R2 >= 0, -ln(1 - R2) for R2 < 0 (continuous at 0)."""
    r2 = np.asarray(r2, dtype=float)
    return np.where(r2 >= 0, pos_scale * r2, -np.log(1 - np.minimum(r2, 0)))


def get_weighted_slopes(unifying_algo_data):
    """Weighted |slope| (rad/s) per trajectory, keyed by obj_id."""
    timestep = unifying_algo_data.timestep_sec.iloc[0]  # converts rad/sample -> rad/s
    slopes = {}
    for obj_id in unifying_algo_data.obj_id_unique_event.unique():
        data = unifying_algo_data[unifying_algo_data.obj_id_unique_event == obj_id]
        # get_weighted_value takes abs() of col, weights = 1/rmse_affine**2
        slopes[obj_id] = uaa.get_weighted_value(data, col='slope', use='rmse_affine') / timestep
    return slopes


def get_r2_to_mean_model(unifying_algo_data, trajec_df, period_shifts=PERIOD_SHIFTS, return_aligned=False,
                         r2_window=R2_WINDOW):
    """
    Circular R2 between each aligned trajectory and the mean of the aligned affine fits over ROI
    (r2_window='aligned'), or over the trajectory's own fit ROI (r2_window='fit_roi'; period_shifts must be [0]).

    period_shifts: shifts (in periods of each trajectory's own best fit, shifted left) to try; the best R2 is kept.
    [0] = default alignment only. Same logic as the variable-wind supplemental notebook E.
    Returns (r2, shift_used), both keyed by obj_id. With return_aligned=True, also returns a dict with the
    aligned arrays (rows follow obj_ids), the mean model and obj_ids.
    """
    obj_ids = unifying_algo_data.obj_id_unique_event.unique()
    trajec_df = trajec_df[trajec_df.obj_id_unique_event.isin(obj_ids)]

    results = uaa.get_aligned_course_array(unifying_algo_data, trajec_df, return_linear_and_affine_fits=True)
    ix_master, aligned_time_arr, aligned_course_arr, aligned_roi_arr, aligned_linear_fits, aligned_affine_fits = results

    mean_affine_fit = np.array([mean_angle(aligned_affine_fits[:, i]) for i in range(aligned_affine_fits.shape[1])])
    mean_roi = mean_affine_fit[ROI]
    n_cols = aligned_course_arr.shape[1]

    if r2_window not in ('aligned', 'fit_roi'):
        raise ValueError("r2_window must be 'aligned' or 'fit_roi'")
    if r2_window == 'fit_roi' and list(period_shifts) != [0]:
        raise ValueError("period shifts only apply to r2_window='aligned'")

    r2 = {}
    shift_used = {}
    for i, obj_id in enumerate(obj_ids):  # rows follow obj_ids order
        if r2_window == 'fit_roi':
            ix = (aligned_roi_arr[i] == 1) & ~np.isnan(aligned_course_arr[i]) & ~np.isnan(mean_affine_fit)
            r2[obj_id] = uaa.circular_r2(aligned_course_arr[i, ix], mean_affine_fit[ix]) if ix.sum() >= MIN_SAMPLES else np.nan
            shift_used[obj_id] = 0
            continue
        # period (in samples) of this trajectory's best fit, the same fit used for its alignment
        fits = unifying_algo_data[unifying_algo_data.obj_id_unique_event == obj_id].sort_values('rmse_affine')
        period = int(np.round(2 * np.pi / np.abs(fits.slope.iloc[0])))

        best = (np.nan, np.nan)  # (r2, n_periods)
        for n_periods in period_shifts:
            # shifting the trajectory left by n periods = comparing its course n periods later to the model
            start = ROI.start + n_periods * period
            src = np.arange(start, start + (ROI.stop - ROI.start))
            course_win = np.full(len(src), np.nan)
            ok_src = src < n_cols
            course_win[ok_src] = aligned_course_arr[i, src[ok_src]]

            ix = ~np.isnan(course_win) & ~np.isnan(mean_roi)
            if ix.sum() < MIN_SAMPLES:
                continue
            r2_val = uaa.circular_r2(course_win[ix], mean_roi[ix])
            if np.isnan(best[0]) or r2_val > best[0]:
                best = (r2_val, n_periods)
        r2[obj_id], shift_used[obj_id] = best
    if return_aligned:
        aligned = {'obj_ids': obj_ids, 'ix_master': ix_master, 'course': aligned_course_arr, 'roi': aligned_roi_arr,
                   'linear_fits': aligned_linear_fits, 'affine_fits': aligned_affine_fits, 'mean_model': mean_affine_fit}
        return r2, shift_used, aligned
    return r2, shift_used


def compute_slope_r2(unifying_filename, trajec_filename, label='', ylim=YLIM, period_shifts=PERIOD_SHIFTS,
                     r2_window=R2_WINDOW):
    """
    Returns {'x': transformed R2, 'x_raw': R2, 'slopes': weighted |slope| (rad/s)} for trajectories with a valid R2.
    """
    unifying_algo_data = pd.read_parquet(unifying_filename)
    trajec_df = load_trajectories(trajec_filename)

    slopes = get_weighted_slopes(unifying_algo_data)
    r2, shift_used = get_r2_to_mean_model(unifying_algo_data, trajec_df, period_shifts, r2_window=r2_window)

    obj_ids = list(slopes.keys())
    x = np.array([r2[o] for o in obj_ids])
    y = np.array([slopes[o] for o in obj_ids])
    ok = ~np.isnan(x)
    r = {'slopes': y[ok], 'x': r2_axis_transform(x[ok]), 'x_raw': x[ok]}

    n_out = np.sum((r['x'] < XLIM[0]) | (r['x'] > XLIM[1]) | (r['slopes'] > ylim[1]))
    print('%-28s n = %4d (dropped %d)  median slope = %.2f  median R2 = %.2f  outside limits: %d'
          % (label, ok.sum(), (~ok).sum(), np.median(r['slopes']), np.median(r['x_raw']), n_out))
    if list(period_shifts) != [0]:
        s = np.array([shift_used[o] for o in obj_ids])[ok]
        print('      fraction using each period shift:', {p: round(float(np.mean(s == p)), 2) for p in period_shifts})
    return r


# --------------------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------------------
def plot_scatter(ax, r, color):
    ax.plot(r['x'], r['slopes'], 'o', markersize=MARKERSIZE, color=color, markeredgecolor='none', alpha=ALPHA)


def plot_median_confidence_ellipse(ax, x, y, color, rng, ylim=YLIM,
                                   confidence=ELLIPSE_CONFIDENCE, n_bootstrap=N_BOOTSTRAP, alpha=ELLIPSE_ALPHA):
    """
    Confidence ellipse for the joint (median x, median y): bootstrap trajectories, take both medians
    per resample, then draw the chi2(2) ellipse of the bootstrap covariance, centered on the observed medians.
    Also draws dashed guide lines from the median to both axes. x should already be in plotted coordinates.
    """
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    center = np.array([np.median(x), np.median(y)])
    width = height = np.nan

    if n >= 2:
        ix = rng.integers(0, n, size=(n_bootstrap, n))
        boot = np.column_stack([np.median(x[ix], axis=1), np.median(y[ix], axis=1)])
        cov = np.cov(boot, rowvar=False)
        if np.all(np.isfinite(cov)):
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 0, None)
            scale = np.sqrt(scipy.stats.chi2.ppf(confidence, df=2))
            width, height = 2 * scale * np.sqrt(eigvals[::-1])
            angle = np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
            ax.add_patch(Ellipse(center, width, height, angle=angle,
                                 facecolor=color, alpha=alpha, edgecolor='none', zorder=10))
            ax.add_patch(Ellipse(center, width, height, angle=angle,
                                 facecolor='none', edgecolor=color, linewidth=0.5, zorder=11))
    ax.plot(*center, 'o', markersize=1.5, color=color, markeredgecolor='none', zorder=12)

    ax.plot([XLIM[0], center[0]], [center[1], center[1]], linestyle=(0, (2, 1.5)),
            color=color, linewidth=0.5, zorder=9)
    ax.plot([center[0], center[0]], [ylim[0], center[1]], linestyle=(0, (2, 1.5)),
            color=color, linewidth=0.5, zorder=9)
    return center, width, height


def plot_groups(ax, groups, rng, ylim=YLIM):
    """groups: list of (result dict, color). All scatter first, then all ellipses on top."""
    for r, color in groups:
        plot_scatter(ax, r, color)
    for r, color in groups:
        plot_median_confidence_ellipse(ax, r['x'], r['slopes'], color, rng, ylim=ylim)


def clean_labels(ax, ylim=YLIM, show_xlabel=True, show_ylabel=True):
    yticks, yticklabels = YTICKS[tuple(ylim)]
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.set_xticks(r2_axis_transform(R2_TICKS))
    ax.set_xticklabels(R2_TICKLABELS)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(X_LABEL if show_xlabel else '', labelpad=1)
    ax.set_ylabel('Slope (rad/s)' if show_ylabel else '', labelpad=-2)
    ax.tick_params(axis='y', pad=2)
    ax.tick_params(axis='x', pad=2)
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5)
    fifi.mpl_functions.set_fontsize(ax, 6)


# --------------------------------------------------------------------------------------------------
# SVG output
# --------------------------------------------------------------------------------------------------
def uniquify_glyph_ids(svg_filename, layer_names):
    """
    Prefix matplotlib glyph ids (e.g. Cmr10-31) inside the given layers with the layer name, so layers
    written in separate runs / matplotlib versions can't pick up each other's glyph definitions
    (otherwise e.g. the 2 in R^2 can render as a 1). Copied from supp_fig_11_unifying_flies_shams notebook D.
    """
    s = open(svg_filename).read()
    for layer in layer_names:
        start = re.search(r'<g [^>]*id="%s"[^>]*inkscape:groupmode="layer"' % layer, s).start()
        depth = 0
        for m in re.finditer(r'<g[ >]|</g>', s[start:]):
            depth += 1 if m.group(0) != '</g>' else -1
            if depth == 0:
                end = start + m.end()
                break
        block = s[start:end]
        glyph_ids = set(re.findall(r'<path id="([^"]+)" d=', block)) & set(re.findall(r'xlink:href="#([^"]+)"', block))
        for gid in glyph_ids:
            if gid.startswith(layer + '_'):
                continue  # already prefixed in an earlier run
            block = block.replace('id="%s"' % gid, 'id="%s_%s"' % (layer, gid))
            block = block.replace('xlink:href="#%s"' % gid, 'xlink:href="#%s_%s"' % (layer, gid))
        s = s[:start] + block + s[end:]
    open(svg_filename, 'w').write(s)


def load_layout(figure_name=FIGURE_NAME):
    import matplotlib.pyplot as plt
    layout = fifi.svg_to_axes.FigureLayout(figure_name, autogenlayers=True, make_mplfigures=True,
                                           hide_layers=[], dpi=600)
    plt.close('all')
    return layout


def save_figure(layout, fifi_figure_label, figure_name=FIGURE_NAME):
    """Write only this figurefirst figure into its own layer of the shared SVG."""
    layout.append_figure_to_layer(layout.figures[fifi_figure_label], fifi_figure_label, cleartarget=True)
    layout.write_svg(figure_name)
    uniquify_glyph_ids(figure_name, [fifi_figure_label])


# --------------------------------------------------------------------------------------------------
# R2-only distribution plots (style of Figure 5 panel D: fpl.scatter_box_violin_distribution)
# --------------------------------------------------------------------------------------------------
R2_FIGURE_NAME = 'unifying_analysis_translationTrue.svg'  # Figure 5
R2_LABEL = 'Circular $R^2$\nto mean model'
# Symmetric R2 axis for the distribution plots: R2 in [0, 1] gets the same height as R2 in [R2_FLOOR, 0],
# so 0 sits in the middle of the axis. R2_FLOOR = 1 - e^5.5 (about -244), the same lower limit as the scatter plots.
R2_DIST_POS_SCALE = 5.5
R2_DIST_LIM = (-R2_DIST_POS_SCALE, R2_DIST_POS_SCALE)
R2_DIST_ZERO_LINE = True  # thin gray reference line at R2 = 0
DIST_XWIDTH = 0.3  # half-width of each violin column, in category units (fpl default, as in Figure 5 D)


def plot_r2_distribution(ax, x, r, color, xwidth=DIST_XWIDTH, n_bins=20):
    """One column: jittered points + median line + bootstrapped 95% CI of the median, on the plotted R2 axis."""
    import copy
    import fly_plot_lib.plot as fpl
    y = r2_axis_transform(r['x_raw'], pos_scale=R2_DIST_POS_SCALE)
    hist_counts, hist_bins = np.histogram(y, bins=n_bins)
    fpl.scatter_box_violin_distribution(ax, x, copy.copy(y), hist_counts, hist_bins, xwidth=xwidth,
                                        color=color, markersize=MARKERSIZE)


def clean_r2_labels(ax, xticks, xticklabels, xlim, show_ylabel=True, rotation=0, ticklabel_colors=None, xlabel=''):
    ax.set_ylim(*R2_DIST_LIM)  # the R2 axis is now vertical
    ax.set_yticks(r2_axis_transform(R2_TICKS, pos_scale=R2_DIST_POS_SCALE))
    if R2_DIST_ZERO_LINE:
        ax.axhline(0, color='0.75', linewidth=0.5, zorder=0)
    ax.set_yticklabels(R2_TICKLABELS)
    ax.set_ylabel(R2_LABEL if show_ylabel else '', labelpad=1)

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, labelpad=1)

    ax.tick_params(axis='y', pad=2)
    ax.tick_params(axis='x', pad=2)
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5)
    fifi.mpl_functions.set_fontsize(ax, 6)

    # tick labels last: adjust_spines resets their rotation and color
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=rotation, ha='right' if rotation else 'center',
                       rotation_mode='anchor' if rotation else 'default', fontsize=6)
    if ticklabel_colors is not None:
        for t, c in zip(ax.get_xticklabels(), ticklabel_colors):
            t.set_color(c)
