import argparse
import matplotlib
matplotlib.rcParams['font.serif'] = ['Times'] + matplotlib.rcParams['font.serif']
matplotlib.rcParams['font.size'] = 6
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams["ps.usedistiller"] = 'xpdf'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.weight'] = 'normal'
matplotlib.rcParams["mathtext.fontset"] = 'cm'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import math

import pandas as pd

import copy

import cvxpy
cp = cvxpy

import figurefirst as fifi

from braid_analysis import braid_analysis_plots

from align_course_direction_analysis import unifying_algo_analysis as uaa
from align_course_direction_analysis import unifying_algo_plots as uap

from set_zorder_functions import *

FIGURE_NAME = 'supplemental_unifying_analysis.svg'
TRANSLATION = True
COURSE_MARKER_SIZE = 2
COURSE_ALPHA_MULTIPLIER = 3


def clean_labels(ax, show_labels, spines=['left', 'bottom']):
    ax.set_rasterization_zorder(0)

    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlim(-2, 2)
    ax.set_xticks([-200, -100, 0, 100, 200])

    if show_labels:
        ax.set_xticklabels(['$-2$', '$-1$', '$0$', '$1$', '$2$'])
    else:
        ax.set_xticklabels([])

    if show_labels:
        ax.set_yticklabels(['$-\\pi$', '','$0$','','$\\pi$',])
    else:
        ax.set_yticklabels([])

    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'])

    if show_labels:
        ax.set_ylabel('Course direction', labelpad=-2)
        ax.set_xlabel('Aligned time (s)', labelpad=-1)
    else:
        ax.set_ylabel('')
        ax.set_xlabel('')

    ax.tick_params(axis='y', pad=2)
    ax.tick_params(axis='x', pad=2)

    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5)
    fifi.mpl_functions.set_fontsize(ax, 6)


class LabelToMetadata:
    def __init__(self):
        self.flash = metadata = {'../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/laminar_data_flash_translation' + str(TRANSLATION) + '.parquet': [1, '#991128ff', 'laminar'],
                                 '../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/unsteady_top_on_data_flash_translation' + str(TRANSLATION) + '.parquet': [2, '#2b75b3b3', 'unsteady'],
                                 '../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/stillair_data_flash_translation' + str(TRANSLATION) + '.parquet': [3, '#084a72ff', 'stillair'],
                               }

        self.sham = metadata = {'../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/laminar_data_sham_translation' + str(TRANSLATION) + '.parquet': [1, '#991128ff', 'laminar'],
                                 '../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/unsteady_top_on_data_sham_translation' + str(TRANSLATION) + '.parquet': [2, '#2b75b3b3', 'unsteady'],
                                 '../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/stillair_data_sham_translation' + str(TRANSLATION) + '.parquet': [3, '#084a72ff', 'stillair'],
                               }

        self.WT_flash = metadata = {'../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/WT_laminar_data_flash_translation' + str(TRANSLATION) + '.parquet': [1, '#991128ff', 'laminar'],
                                 'None': [2, '#2b75b3b3', 'unsteady'],
                                 '../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/WT_stillair_data_flash_translation' + str(TRANSLATION) + '.parquet': [3, '#084a72ff', 'stillair'],
                               }

        self.WT_sham = metadata = {'../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/WT_laminar_data_sham_translation' + str(TRANSLATION) + '.parquet': [1, '#991128ff', 'laminar'],
                                 'None': [2, '#2b75b3b3', 'unsteady'],
                                 '../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/WT_stillair_data_sham_translation' + str(TRANSLATION) + '.parquet': [3, '#084a72ff', 'stillair'],
                               }

        self.CFD_casting = metadata = {'../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/cfd_sim_steady_casting_translation' + str(TRANSLATION) + '.parquet': [1, '#991128ff', 'laminar'],
                                 '../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/cfd_sim_unsteady_casting_translation' + str(TRANSLATION) + '.parquet': [2, '#2b75b3b3', 'unsteady'],
                                 'None': [3, '#084a72ff', 'stillair'],
                               }

        self.CFD_circling = metadata = {'../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/cfd_sim_steady_circling_translation' + str(TRANSLATION) + '.parquet': [1, '#991128ff', 'laminar'],
                                 '../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/cfd_sim_unsteady_circling_translation' + str(TRANSLATION) + '.parquet': [2, '#2b75b3b3', 'unsteady'], 
                                 '../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/cfd_sim_still_air_circle_translation' + str(TRANSLATION) + '.parquet': [3, '#084a72ff', 'stillair'],
                               }

        self.unifying = metadata = {'../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/new_unifying_laminar_translation' + str(TRANSLATION) + '.parquet': [1, '#991128ff', 'laminar'],
                         '../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/new_unifying_unsteady_translation' + str(TRANSLATION) + '.parquet': [2, '#2b75b3b3', 'unsteady'],
                         '../../../../Data/Unifying_Algo_Results/Supplemental/generate_supplementary_figure/unifying_algo_results/new_unifying_stillair_translation' + str(TRANSLATION) + '.parquet': [3, '#084a72ff', 'stillair'],
                       }


def get_filename_for_wind_type(metadata, windtype):
    filename = None
    for key, val in metadata.items():
        if windtype in val:
            filename = key
    return filename


def get_trajec_filename_from_unifying_filename(unifying_filename):
    if 'cfd_sim' in unifying_filename:
        return '../../../../Data/Experimental_Fly_Data/cfd_sim_circling_casting_all_wind_conditions_preprocessed.parquet'
    elif 'new_unifying' in unifying_filename:
        return '../../../../Data/Experimental_Fly_Data/new_unifying.parquet'
    elif 'WT_laminar' in unifying_filename:
        return '../../../../Data/Experimental_Fly_Data/flies_laminar_wt_preprocessed_optotrigger_trimmed.hdf'
    elif 'WT_stillair' in unifying_filename:
        return '../../../../Data/Experimental_Fly_Data/flies_stillair_wt_preprocessed_optotrigger_trimmed.hdf'
    elif 'laminar_data' in unifying_filename:
        return '../../../../Data/Experimental_Fly_Data/flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf'
    elif 'stillair_data' in unifying_filename:
        return '../../../../Data/Experimental_Fly_Data/flies_stillair_c1xwt_preprocessed_optotrigger_trimmed.hdf'
    elif 'unsteady_top_on' in unifying_filename:
        return '../../../../Data/Experimental_Fly_Data/flies_splitflow_topon_c1xwt_preprocessed_optotrigger_trimmed.hdf'
    return None


def get_filenames_for_metadata_windtype(metadata, windtype):
    print(windtype)
    unifying_filename = get_filename_for_wind_type(metadata, windtype)
    print(unifying_filename)
    trajectory_filename = None
    df = None
    unifying_algo_data = None

    if unifying_filename != 'None' and unifying_filename is not None:
        unifying_algo_data = pd.read_parquet(unifying_filename)

        trajectory_filename = get_trajec_filename_from_unifying_filename(unifying_filename)
        if '.hdf' in trajectory_filename:
            df = pd.read_hdf(trajectory_filename)
        else:
            df = pd.read_parquet(trajectory_filename)

    else:
        unifying_algo_data = None

    
    print(trajectory_filename)
    return unifying_algo_data, df


def __main__(label):


    label_to_metadata = LabelToMetadata()
    fifi_label_course = label + '_' + 'course'
    metadata = label_to_metadata.__getattribute__(label)

    layout = fifi.svg_to_axes.FigureLayout(FIGURE_NAME, autogenlayers=True, make_mplfigures=True, hide_layers=[], dpi=600)
    plt.close('all')

    # With real laminar data
    show_labels = True
    windtype = 'laminar'
    unifying_algo_data, df = get_filenames_for_metadata_windtype(metadata, windtype)
    if unifying_algo_data is not None:
        ax = layout.axes[(fifi_label_course, windtype)]

        # build aligned arrays
        results = uaa.get_aligned_course_array(unifying_algo_data, df, return_linear_and_affine_fits=True)
        ix_master, aligned_time_since_flash_arr, aligned_course_arr, aligned_roi_arr, aligned_linear_fits, aligned_affine_fits = results

        # build flash arrays from aligned time
        flash_length = 675 # ms
        aligned_flash_arr = np.zeros_like(aligned_time_since_flash_arr)
        ix = np.where( (aligned_time_since_flash_arr>0) * (aligned_time_since_flash_arr<flash_length/1000.) )
        aligned_flash_arr[ix] = 1

        # make the plot
        uap.plot_aligned_course_from_array(ix_master,
                                           aligned_time_since_flash_arr,
                                           aligned_course_arr,
                                           aligned_roi_arr,
                                           aligned_flash_arr=None,
                                           aligned_linear_fits=None, # if None, skips plot
                                           aligned_affine_fits=None, # if None, skips plot
                                           xlim_start=-300,
                                           xlim_end=300,
                                           ax=ax,
                                           clean_spines=True,
                                           course_marker_size=COURSE_MARKER_SIZE,
                                           course_alpha_multiplier=COURSE_ALPHA_MULTIPLIER,
                                           )

        clean_labels(ax, show_labels)
    else:
        ax = layout.axes[(fifi_label_course, windtype)]
        fifi.mpl_functions.adjust_spines(ax, [])

    # With real unsteady data
    show_labels = False
    windtype = 'unsteady'
    unifying_algo_data, df = get_filenames_for_metadata_windtype(metadata, windtype)
    if unifying_algo_data is not None:
        ax = layout.axes[(fifi_label_course, windtype)]

        # build aligned arrays
        results = uaa.get_aligned_course_array(unifying_algo_data, df, return_linear_and_affine_fits=True)
        ix_master, aligned_time_since_flash_arr, aligned_course_arr, aligned_roi_arr, aligned_linear_fits, aligned_affine_fits = results

        # build flash arrays from aligned time
        flash_length = 675 # ms
        aligned_flash_arr = np.zeros_like(aligned_time_since_flash_arr)
        ix = np.where( (aligned_time_since_flash_arr>0) * (aligned_time_since_flash_arr<flash_length/1000.) )
        aligned_flash_arr[ix] = 1

        # make the plot
        uap.plot_aligned_course_from_array(ix_master,
                                           aligned_time_since_flash_arr,
                                           aligned_course_arr,
                                           aligned_roi_arr,
                                           aligned_flash_arr=None,
                                           aligned_linear_fits=None, # if None, skips plot
                                           aligned_affine_fits=None, # if None, skips plot
                                           xlim_start=-300,
                                           xlim_end=300,
                                           ax=ax,
                                           clean_spines=True,
                                           course_marker_size=COURSE_MARKER_SIZE,
                                           course_alpha_multiplier=COURSE_ALPHA_MULTIPLIER,
                                           )
        clean_labels(ax, show_labels)
    else:
        ax = layout.axes[(fifi_label_course, windtype)]
        fifi.mpl_functions.adjust_spines(ax, [])

    # With real stillair data
    show_labels = False
    windtype = 'stillair'
    unifying_algo_data, df = get_filenames_for_metadata_windtype(metadata, windtype)
    if unifying_algo_data is not None:
        ax = layout.axes[(fifi_label_course, windtype)]

        # build aligned arrays
        results = uaa.get_aligned_course_array(unifying_algo_data, df, return_linear_and_affine_fits=True)
        ix_master, aligned_time_since_flash_arr, aligned_course_arr, aligned_roi_arr, aligned_linear_fits, aligned_affine_fits = results

        # build flash arrays from aligned time
        flash_length = 675 # ms
        aligned_flash_arr = np.zeros_like(aligned_time_since_flash_arr)
        ix = np.where( (aligned_time_since_flash_arr>0) * (aligned_time_since_flash_arr<flash_length/1000.) )
        aligned_flash_arr[ix] = 1

        # make the plot
        uap.plot_aligned_course_from_array(ix_master,
                                           aligned_time_since_flash_arr,
                                           aligned_course_arr,
                                           aligned_roi_arr,
                                           aligned_flash_arr=None,
                                           aligned_linear_fits=None, # if None, skips plot
                                           aligned_affine_fits=None, # if None, skips plot
                                           xlim_start=-300,
                                           xlim_end=300,
                                           ax=ax,
                                           clean_spines=True,
                                           course_marker_size=COURSE_MARKER_SIZE,
                                           course_alpha_multiplier=COURSE_ALPHA_MULTIPLIER,
                                           )
        clean_labels(ax, show_labels)
    else:
        ax = layout.axes[(fifi_label_course, windtype)]
        fifi.mpl_functions.adjust_spines(ax, [])

    layout.append_figure_to_layer(layout.figures[fifi_label_course], fifi_label_course, cleartarget=True)
    layout.write_svg(FIGURE_NAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate aligned course direction figure.')
    parser.add_argument('--label', type=str, default='CFD_casting',
                        help='Label specifying the dataset to use (e.g. flash, sham, WT_flash, WT_sham, CFD_casting, CFD_circling, unifying)')
    args = parser.parse_args()
    __main__(args.label)
