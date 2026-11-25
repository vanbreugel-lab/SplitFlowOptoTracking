"""
Shared code for r2_distribution_summary.ipynb (Figure 5 panel E: circular R2 to the population model, which is
the median-parameter model drawn as the pink lines in Figure 5 C-D; see POPULATION_MODEL).
Copied from Figure_5_option/ (which also has the slope-vs-R2 scatter version, slope_vs_r2_summary.ipynb).

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

from splitflow.data_paths import data_path, trajectory_for_result
from splitflow.unifying_results import result_path

import numpy as np
import pandas as pd
import scipy.stats
from matplotlib.patches import Ellipse

import figurefirst as fifi
from align_course_direction_analysis import unifying_algo_analysis as uaa

FIGURE_NAME = 'unifying_analysis_translationTrue.svg'
TRANSLATION = True

SHUFFLED_SYNTHETIC_DIR = Path('../../../Supplemental/supp_fig_13_shuffled_synthetic_uaa')

# Average model window: columns of the aligned arrays (-300..300 samples at 0.01 s), 200:400 = aligned time -1..1 s
ROI = slice(200, 400)
MIN_SAMPLES = 160  # trajectories with data for less than 80% of the ROI (1.6 of 2 s) are dropped
PERIOD_SHIFTS = [0]  # default: score each trajectory at its default alignment only
# Where R2 is scored. 'aligned': aligned time -1..1 s (ROI above) for every trajectory.
# 'fit_roi' (option D in surge_r2_window/): each trajectory's own fit ROI (min_ix:max_ix of its best fit), so
# pre-fit surge flight that the -1..1 s window can reach back into is excluded. Period shifts don't apply.
R2_WINDOW = 'aligned'

# The single model each trajectory of a regime is scored against.
# 'median_parameters': the model built from the regime's median fit parameters (median |slope|, element-wise median
#     affine matrix, median translation). This is the pink line of Figure 5 C-D (median_parameter_model below;
#     C_figure_align_course_directions.ipynb draws the line with the same function).
# 'mean_of_fits': the circular mean, at each aligned time point, of the trajectories' own aligned affine fits
#     (used until 2026-11-24; it is not the line drawn in Figure 5 C-D).
POPULATION_MODEL = 'median_parameters'

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
X_LABEL = 'Circular $R^2$ to median model'

YLIM = (0, 8)  # weighted |slope| (rad/s)
YTICKS = {(0, 8): ([0, 2, 4, 6, 8], ['0', '', '', '', '8']),
          (0, 30): ([0, 10, 20, 30], ['0', '', '', '30'])}


# --------------------------------------------------------------------------------------------------
# File lookup
# --------------------------------------------------------------------------------------------------
def unifying_results_path(filename):
    """Locate a unifying-algo results parquet by file name (layout: splitflow.unifying_results)."""
    return result_path(filename)


def fly_trajec_filename(unifying_filename):
    """Trajectory file matching a fly unifying-results file (laminar / unsteady / still air datasets)."""
    return trajectory_for_result(unifying_filename)  # mapping: unifying_results.trajectories in Data/data_paths.yaml


# Figure 5 colors for the variable wind data: all speeds light purple, 20 cm/s (fan setting 28) highlighted.
# Gray is reserved for sham data.
VARWIND_COLOR = '#c4aee0'
VARWIND_HIGHLIGHT_COLOR = '#a245ff'
VARWIND_HIGHLIGHT_SPEED = '28'  # 20 cm/s

VARWIND_TRAJEC_FILENAME = data_path('fly.varwind')
VARWIND_SPEEDS = ['12', '20', '23', '28', '40', '90']
VARWIND_SPEED_LABELS = {'12': 0, '20': 3, '23': 10, '28': 20, '40': 30, '90': 60}  # cm/s


def varwind_unifying_filename(condition, speed):
    """Speed-series fits for 'flash' or 'sham' at one fan setting ('12' ... '90')."""
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


def _best_fit_parameters(unifying_algo_data):
    """
    Parameters of each trajectory's best fit (lowest rmse_affine), normalized to a positive slope as in
    uaa.get_aligned_course_array (mirror: flip the sign of the off-diagonal AM terms and of tx).
    Returns (obj_ids, |slopes| (n,), AMs (n, 2, 2), translations (n, 2)).
    """
    best = (unifying_algo_data.sort_values('rmse_affine', kind='stable')
            .groupby('obj_id_unique_event', sort=False).head(1))
    flip = np.diag([-1.0, 1.0])
    obj_ids, slopes, AMs, ts = [], [], [], []
    for _, row in best.iterrows():
        AM = np.array([[row.AM_0_0, row.AM_0_1], [row.AM_1_0, row.AM_1_1]])
        t = np.nan_to_num(np.array([row.tx, row.ty], dtype=float))  # fits without translation store None
        if row.slope < 0:
            AM, t = flip @ AM @ flip, flip @ t
        obj_ids.append(row.obj_id_unique_event); slopes.append(abs(row.slope)); AMs.append(AM); ts.append(t)
    return np.array(obj_ids), np.array(slopes), np.array(AMs), np.array(ts)


def _model_from_parameters(slopes, AMs, ts, ix_master):
    phi = np.median(slopes) * np.asarray(ix_master, dtype=float)
    w = np.median(AMs, axis=0) @ np.vstack((np.sin(phi), np.cos(phi))) + np.median(ts, axis=0)[:, None]
    return np.arctan2(w[0], w[1])


def median_parameter_model(unifying_algo_data, ix_master):
    """
    Model course vs. aligned time built from a regime's median fit parameters: the pink line of Figure 5 C-D and
    the model that panel E scores every trajectory against.
    Model: phi = median|slope| * aligned_ix (phase 0 at aligned t=0, a 0-crossing of the linear fit),
           course = atan2(w0, w1), w = median(AM) @ [sin(phi), cos(phi)] + median([tx, ty])   ((sin, cos) convention)
    """
    _, slopes, AMs, ts = _best_fit_parameters(unifying_algo_data)
    return _model_from_parameters(slopes, AMs, ts, ix_master)


def _leave_one_out_median_parameter_models(unifying_algo_data, ix_master, obj_ids):
    """median_parameter_model built without each trajectory in turn; one row per obj_id (order of obj_ids)."""
    ids, slopes, AMs, ts = _best_fit_parameters(unifying_algo_data)
    models = []
    for obj_id in obj_ids:
        keep = ids != obj_id
        models.append(_model_from_parameters(slopes[keep], AMs[keep], ts[keep], ix_master))
    return np.array(models)


def get_r2_to_mean_model(unifying_algo_data, trajec_df, period_shifts=PERIOD_SHIFTS, return_aligned=False,
                         r2_window=R2_WINDOW):
    """
    Circular R2 between each aligned trajectory and the regime's population model (POPULATION_MODEL: the
    median-parameter model, or the mean of the aligned affine fits) over ROI
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

    if POPULATION_MODEL == 'median_parameters':
        mean_affine_fit = median_parameter_model(unifying_algo_data, ix_master)
    elif POPULATION_MODEL == 'mean_of_fits':
        mean_affine_fit = np.array([mean_angle(aligned_affine_fits[:, i]) for i in range(aligned_affine_fits.shape[1])])
    else:
        raise ValueError("POPULATION_MODEL must be 'median_parameters' or 'mean_of_fits'")
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
                   'linear_fits': aligned_linear_fits, 'affine_fits': aligned_affine_fits, 'mean_model': mean_affine_fit,
                   'unifying_algo_data': unifying_algo_data}
        return r2, shift_used, aligned
    return r2, shift_used


# --------------------------------------------------------------------------------------------------
# Held-out (cross-validated) R2 to the mean model, against overfitting
# --------------------------------------------------------------------------------------------------
HELDOUT_LAGS = np.arange(-100, 101, 2)  # time-alignment lags (samples at 100 Hz) searched on the training half
HELDOUT_MIN_COVERAGE = 0.9  # a lag is allowed only if the model is defined for >= 90% of the training samples


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _leave_one_out_mean_models(affine_fits):
    """Circular mean of the aligned affine fits of all OTHER trajectories, per row. Returns an array like affine_fits."""
    s, c = np.sin(affine_fits), np.cos(affine_fits)
    S, C = np.nansum(s, axis=0), np.nansum(c, axis=0)
    N = np.sum(~np.isnan(affine_fits), axis=0)
    S_loo = S[None, :] - np.nan_to_num(s)
    C_loo = C[None, :] - np.nan_to_num(c)
    N_loo = N[None, :] - (~np.isnan(affine_fits))
    loo = np.arctan2(S_loo, C_loo)
    loo[N_loo == 0] = np.nan
    return loo


def _fit_sign_and_lag(course_train, pos_train, model, lags=HELDOUT_LAGS):
    """
    Choose the mirror sign (+1/-1) and time lag of the model that minimize the mean squared angular error on the
    training samples. Returns (sign, lag) or None if no lag has enough model coverage.
    """
    n_cols = len(model)
    pos = pos_train[None, :] + lags[:, None]  # (lags, samples)
    valid = (pos >= 0) & (pos < n_cols)
    m = np.full(pos.shape, np.nan)
    m[valid] = model[pos[valid]]
    coverage = np.mean(~np.isnan(m), axis=1)
    best = None
    for sign in (1, -1):
        err = np.nanmean(_wrap(course_train[None, :] - sign * m) ** 2, axis=1)
        err[coverage < HELDOUT_MIN_COVERAGE] = np.inf
        k = int(np.argmin(err))
        if np.isfinite(err[k]) and (best is None or err[k] < best[0]):
            best = (err[k], sign, int(lags[k]))
    return None if best is None else best[1:]


def get_heldout_r2_to_mean_model(aligned):
    """
    Cross-validated circular R2 of each trajectory to the mean model, with nothing about the model's prediction
    estimated from the samples it is scored on:

      - Population model: built from the OTHER trajectories only (leave-one-out median parameters, or the
        leave-one-out circular mean of the aligned affine fits, per POPULATION_MODEL).
      - Two-fold split in time of the trajectory's own fit ROI (first / second half of its valid samples). On the
        training half, the model's mirror sign and time alignment (lag) are re-estimated by least squares, replacing
        the sign and alignment shift taken from the trajectory's own fit. The model with those settings is then
        scored on the test half.
      - Baseline: constant course at the circular mean of the TEST half itself (fitted in-sample, which favors the
        baseline).

    R2_cv = 1 - (SS_model,1 + SS_model,2) / (SS_const,1 + SS_const,2), summed over the two folds.
    R2_cv > 0: the out-of-sample model predicts the course better than an in-sample constant course.
    Returns a dict keyed by obj_id (NaN if fewer than MIN_SAMPLES valid samples or no admissible lag).
    """
    if POPULATION_MODEL == 'median_parameters':
        loo_models = _leave_one_out_median_parameter_models(aligned['unifying_algo_data'], aligned['ix_master'],
                                                            aligned['obj_ids'])
    else:
        loo_models = _leave_one_out_mean_models(aligned['affine_fits'])
    r2_cv = {}
    for i, obj_id in enumerate(aligned['obj_ids']):
        course = aligned['course'][i]
        pos = np.where((aligned['roi'][i] == 1) & ~np.isnan(course))[0]
        if len(pos) < MIN_SAMPLES:
            r2_cv[obj_id] = np.nan
            continue
        half = len(pos) // 2
        folds = [(pos[:half], pos[half:]), (pos[half:], pos[:half])]
        ss_model = ss_const = 0.0
        ok = True
        for train, test in folds:
            fit = _fit_sign_and_lag(course[train], train, loo_models[i])
            if fit is None:
                ok = False
                break
            sign, lag = fit
            test_pos = test + lag
            in_range = (test_pos >= 0) & (test_pos < len(course))
            pred = np.full(len(test), np.nan)
            pred[in_range] = sign * loo_models[i][test_pos[in_range]]
            keep = ~np.isnan(pred)
            if keep.sum() < 0.9 * len(test):
                ok = False
                break
            c = course[test][keep]
            ss_model += np.sum(_wrap(c - pred[keep]) ** 2)
            ss_const += np.sum(_wrap(c - mean_angle(c)) ** 2)
        r2_cv[obj_id] = 1 - ss_model / ss_const if ok and ss_const > 0 else np.nan
    return r2_cv


def r2_vs_constant_course_log10p(r2):
    """
    One p-value per regime: is the mean model a better fit than a constant-course model?

    Circular R2 uses a constant-course model as its baseline: R2 = 1 - SS_model / SS_constant, where
    SS_constant is the squared angular deviation of the course from its own circular mean over the same samples
    (the trajectory flying at a constant heading). So R2 > 0 <=> the mean model beats the constant course for that
    trajectory. Across trajectories (independent), one-sided exact sign test that more than half have R2 > 0, i.e.
    that the median R2 is above 0. A sign test because R2 is bounded above by 1 but unbounded below, so
    magnitude-based tests would be dominated by the negative tail. Use with the held-out R2
    (get_heldout_r2_to_mean_model) to test against overfitting.
    Returns log10 p (binomial log survival function; p underflows float64 for large regimes).
    """
    r2 = np.asarray(r2, dtype=float)
    r2 = r2[r2 != 0]  # ties with the constant model carry no sign
    k = int(np.sum(r2 > 0))
    return scipy.stats.binom.logsf(k - 1, len(r2), 0.5) / np.log(10)  # log P(K >= k)


def compute_slope_r2(unifying_filename, trajec_filename, label='', ylim=YLIM, period_shifts=PERIOD_SHIFTS,
                     r2_window=R2_WINDOW):
    """
    Returns {'x': transformed R2, 'x_raw': R2, 'slopes': weighted |slope| (rad/s)} for trajectories with a valid R2.
    """
    unifying_algo_data = pd.read_parquet(unifying_filename)
    trajec_df = load_trajectories(trajec_filename)

    slopes = get_weighted_slopes(unifying_algo_data)
    r2, shift_used, aligned = get_r2_to_mean_model(unifying_algo_data, trajec_df, period_shifts, return_aligned=True,
                                                   r2_window=r2_window)

    obj_ids = list(slopes.keys())
    x = np.array([r2[o] for o in obj_ids])
    y = np.array([slopes[o] for o in obj_ids])
    ok = ~np.isnan(x)
    r = {'slopes': y[ok], 'x': r2_axis_transform(x[ok]), 'x_raw': x[ok]}
    if r2_window == 'fit_roi':
        r2_cv = get_heldout_r2_to_mean_model(aligned)
        r['r2_heldout'] = np.array([r2_cv[o] for o in obj_ids])[ok]  # may contain NaN

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
R2_LABEL = 'Circular R²\nto median model'  # plain text so it renders in Helvetica like the rest of Figure 5
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
