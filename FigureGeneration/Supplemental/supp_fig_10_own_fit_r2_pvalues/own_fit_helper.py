"""
Shared code for own_fit_summary.ipynb: per-trajectory goodness of fit of each trajectory's OWN best unifying-algorithm
fit (not the mean / parametric model used in r2_summary.svg).

For every trajectory the best fit row (lowest rmse_affine) is rebuilt from the stored parameters (slope, intercept,
AM, tx/ty) with uaa.build_miop_and_affine_predictions_from_model and scored over that fit's own ROI (min_ix:max_ix),
exactly as uaa.evaluate_miop_affine_fit does when the fit is made. Two numbers per trajectory:

  - circular R^2 (uaa.circular_r2) of the course vs. its own affine fit
  - p-value of the Jammalamadaka-SenGupta circular correlation between the course and its own fit. The correlation
    coefficient is the 'rho' column stored in the results parquet (recomputed here; it matches to ~1e-14). The test
    statistic is rho * sqrt(n * l20 * l02 / l22), two-sided normal p (as in pycircstat.corrcc / astropy circcorrcoef).
    p is computed in log space (scipy.stats.norm.logsf) because it underflows float64 for most trajectories.

Caveats (see own_fit_summary.ipynb): the test assumes n independent samples, but n is ~200 samples at 100 Hz of a
smoothed course, and the fit was optimized on these same samples. So p is far too small in an absolute sense; the
controls show what it looks like without flash-locked structure.

Supplemental figure (own_fit_summary.svg). All paths are relative to this folder; run the notebook from its own directory.
"""
import copy

import numpy as np
import pandas as pd
import scipy.stats

import figurefirst as fifi
from align_course_direction_analysis import unifying_algo_analysis as uaa

import slope_r2_helper as h

FIGURE_NAME = 'own_fit_summary.svg'

# p-value axis: log10(p)
LOG10P_LIM = (-50, 0)
LOG10P_TICKS = [-50, -40, -30, -20, -10, 0]
LOG10P_TICKLABELS = ['$10^{-50}$', '', '', '', '', '$1$']
P_REFERENCE = 0.05  # thin gray reference line
P_LABEL = '$p$, circular correlation\nto own fit'

# R2 axis: linear
R2_LIM = (-0.5, 1)
R2_TICKS = [-0.5, 0, 0.5, 1]
R2_TICKLABELS = ['-0.5', '0', '', '1']
R2_LABEL = 'Circular $R^2$\nto own fit'

MARKERSIZE = h.MARKERSIZE
DIST_XWIDTH = h.DIST_XWIDTH
MIN_SAMPLES = h.MIN_SAMPLES  # trajectories whose fit ROI has fewer valid samples are dropped


def _row_with_none_translation(row):
    """build_miop_and_affine_predictions_from_model tests 'tx is None'; parquet may hold NaN instead."""
    row = row.copy()
    if row.tx is None or (isinstance(row.tx, float) and np.isnan(row.tx)):
        row['tx'] = None
        row['ty'] = None
    return row


def circular_corr_log10p(true, pred):
    """rho (as uaa.get_circular_rho) and two-sided log10 p of the Jammalamadaka-SenGupta test."""
    rho = uaa.get_circular_rho(true, pred)
    sa = np.sin(true - uaa.mean_angle(true))
    sb = np.sin(pred - uaa.mean_angle(pred))
    l20, l02, l22 = np.mean(sa ** 2), np.mean(sb ** 2), np.mean(sa ** 2 * sb ** 2)
    ts = rho * np.sqrt(len(true) * l20 * l02 / l22)
    log10p = (np.log(2) + scipy.stats.norm.logsf(np.abs(ts))) / np.log(10)
    return rho, log10p


def compute_own_fit_stats(unifying_filename, trajec_filename, label=''):
    """
    Returns {'r2': circular R2 to own fit, 'log10p': log10 p, 'rho': rho, 'rho_stored': rho column, 'n': samples}
    for trajectories with >= MIN_SAMPLES valid samples in their fit ROI.
    """
    d = pd.read_parquet(unifying_filename)
    tdf = h.load_trajectories(trajec_filename)
    best = d.sort_values('rmse_affine', kind='stable').groupby('obj_id_unique_event', sort=False).head(1)

    rows = []
    for _, row in best.iterrows():
        row = _row_with_none_translation(row)
        course = tdf[tdf.obj_id_unique_event == row.obj_id_unique_event].course_smoothish.values.astype(float).copy()
        _, course, _, affine_fit, _ = uaa.build_miop_and_affine_predictions_from_model(course, row)
        roi = slice(int(row.min_ix), int(row.max_ix))
        true, fit = course[roi], affine_fit[roi]
        ok = ~np.isnan(true) & ~np.isnan(fit)
        if ok.sum() < MIN_SAMPLES:
            rows.append((np.nan, np.nan, np.nan, row.rho, ok.sum()))
            continue
        true, fit = true[ok], fit[ok]
        pred = true - uaa.angle_distance(true, fit)  # as in uaa.evaluate_miop_affine_fit
        rho, log10p = circular_corr_log10p(true, pred)
        rows.append((uaa.circular_r2(true, fit), log10p, rho, row.rho, ok.sum()))

    a = np.array(rows, dtype=float)
    ok = ~np.isnan(a[:, 0])
    r = {'r2': a[ok, 0], 'log10p': a[ok, 1], 'rho': a[ok, 2], 'rho_stored': a[ok, 3], 'n': a[ok, 4]}
    n_out = np.sum((r['r2'] < R2_LIM[0]) | (r['log10p'] < LOG10P_LIM[0]))
    print('%-28s n = %4d (dropped %d)  median R2 = %.2f  median p = 10^%.1f  frac p<0.05 = %.2f  '
          'outside limits: %d  max|rho - stored| = %.1e'
          % (label, ok.sum(), (~ok).sum(), np.median(r['r2']), np.median(r['log10p']),
             np.mean(r['log10p'] < np.log10(P_REFERENCE)), n_out, np.max(np.abs(r['rho'] - r['rho_stored']))))
    return r


# --------------------------------------------------------------------------------------------------
# Plotting (style of r2_summary.svg / Figure 5 panel D)
# --------------------------------------------------------------------------------------------------
def plot_distribution(ax, x, y, color, ylim, xwidth=DIST_XWIDTH, n_bins=20):
    """One column: jittered points + median line + bootstrapped 95% CI of the median. Values are clipped to ylim."""
    import fly_plot_lib.plot as fpl
    y = np.clip(np.asarray(y, dtype=float), *ylim)
    hist_counts, hist_bins = np.histogram(y, bins=n_bins)
    fpl.scatter_box_violin_distribution(ax, x, copy.copy(y), hist_counts, hist_bins, xwidth=xwidth,
                                        color=color, markersize=MARKERSIZE)


def plot_column(ax_p, ax_r2, x, r, color):
    plot_distribution(ax_p, x, r['log10p'], color, LOG10P_LIM)
    plot_distribution(ax_r2, x, r['r2'], color, R2_LIM)


def _clean(ax, ylim, yticks, yticklabels, ylabel, xticks, xticklabels, xlim, rotation, ticklabel_colors, xlabel,
           reference=None):
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    if reference is not None:
        ax.axhline(reference, color='0.75', linewidth=0.5, zorder=0)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, labelpad=1)
    ax.tick_params(axis='both', pad=2)
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'], tick_length=2.5,
                                     spine_locations={'left': 5, 'bottom': 5}, linewidth=0.5)
    fifi.mpl_functions.set_fontsize(ax, 6)
    # tick labels last: adjust_spines resets their rotation and color
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=rotation, ha='right' if rotation else 'center',
                       rotation_mode='anchor' if rotation else 'default', fontsize=6)
    if ticklabel_colors is not None:
        for t, c in zip(ax.get_xticklabels(), ticklabel_colors):
            t.set_color(c)


def clean_column_labels(ax_p, ax_r2, xticks, xticklabels, xlim, show_ylabel=True, rotation=0, ticklabel_colors=None,
                        xlabel=''):
    """p-value row: no x tick labels; R2 row: tick labels (and optional xlabel) as in r2_summary.svg."""
    _clean(ax_p, LOG10P_LIM, LOG10P_TICKS, LOG10P_TICKLABELS, P_LABEL if show_ylabel else '',
           xticks, [''] * len(xticks), xlim, 0, None, '', reference=np.log10(P_REFERENCE))
    _clean(ax_r2, R2_LIM, R2_TICKS, R2_TICKLABELS, R2_LABEL if show_ylabel else '',
           xticks, xticklabels, xlim, rotation, ticklabel_colors, xlabel, reference=0)
