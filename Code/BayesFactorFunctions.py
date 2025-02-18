from __future__ import print_function

import os, sys
import matplotlib
import matplotlib.gridspec as gridspec
import random
import json
# import seaborn as sns
#import cv2
from matplotlib.path import Path
import matplotlib.patches as patches
import scipy
import statsmodels as sm
from decimal import Decimal
import numpy as np

'''
This code was originally written by Kate Leitch and can be found here: https://github.com/kateleitch/drosophila_wind_assisted_dispersal/tree/master

'''

def calculate_levels_for_kde_plot(list_of_confidence_intervals, normalized_pdf):
    #https://stackoverflow.com/questions/35225307/set-confidence-levels-in-seaborn-kdeplot

    # Take histogram bin membership as proportional to Likelihood
    # This is true when data comes from a Markovian process # <----- need to evaluate this; ask Kellan
    def objective(limit, target):
        w = np.where(normalized_pdf > limit)
        count = normalized_pdf[w]
        return count.sum() - target

    # Find levels by summing histogram to objective
    list_of_levels = []
    for target in list_of_confidence_intervals:
        lev = scipy.optimize.bisect(objective, normalized_pdf.min(), normalized_pdf.max(), args=(target,))
        list_of_levels.append(lev)

    highest_level=normalized_pdf.max()
    list_of_levels.append(highest_level)
    return list_of_levels

def hex2rgb(hexval):
    r = int(hexval[1:3], 16)/255.
    g = int(hexval[3:5], 16)/255.
    b = int(hexval[5:], 16)/255.
    return [r,g,b]

def color_fade(c, n):
    c_list = [c]
    for i in range(n-1):
        c_list.append([1-(1-x)*3/4 for x in c_list[-1]])
    return c_list[::-1]

def find_nearest_index(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def normal(x, mean, sigma):
    return (sigma*np.sqrt(2*np.pi))**-1 * np.exp(-(x-mean)**2/(2*sigma**2))

def normal_kernel_2D (x_array,y_array, mean, sigma_x, sigma_y):
    return np.array(normal(x_array, mean[0], sigma_x)*normal(y_array,mean[1],sigma_y))

def adjust_spines(ax_handle, spines):
    for loc, spine in ax_handle.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            #spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax_handle.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax_handle.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax_handle.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax_handle.xaxis.set_ticks([])
