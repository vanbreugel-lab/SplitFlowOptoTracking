import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import random
import math

import pandas as pd
import pandas

import copy

import cvxpy
cp = cvxpy

import figurefirst as fifi

from braid_analysis import braid_analysis_plots
from braid_analysis import flymath
import fly_plot_lib.plot as fpl

import pynumdiff

import scipy.stats

###
from affine_math_and_plot_helper import *
###


# PARAMETERS

RANDOMNESS = True

SPEED = 0.3
SPEED_STD = 0.1

UPWIND_TRANSLATION_MULTIPLIER = 0.2
UPWIND_TRANSLATION_MULTIPLIER_STD = 0.1

UPWIND_GAMMA_MULTIPLIER = 0.08
UPWIND_GAMMA_MULTIPLIER_STD = 0.02

SACCADE_THRESHOLD_PROPORTIONAL = 8
SACCADE_THRESHOLD_PROPORTIONAL_STD = 3
SACCADE_THRESHOLD_DERIVATIVE = 3
SACCADE_THRESHOLD_DERIVATIVE_STD = 1

SMOOTHING_WINDOW = 11
SMOOTHING_WINDOW_STD = 4

SLOPE = 3.95
SLOPE_STD = 0.3

SIGMOID_MIDPOINT = 0.1
SIGMOID_MIDPOINT_STD = 0.1

SIGMOID_STEEPNESS = 8
SIGMOID_STEEPNESS_STD = 2

COURSE_ANGLE_STD = 0.2

dt = 0.01
t, wind_direction, wind_speed = get_wind('laminar', RANDOMNESS, 11, dt)
LENGTH = len(t)

# Get random parameter values

def get_smoothing_window():
    v = int(np.random.normal(SMOOTHING_WINDOW, SMOOTHING_WINDOW_STD))
    if int((v-1)/2) == (v-1)/2: # odd
        pass
    else:
        v += 1
    if v < 3:
        v = 3
    return v

def get_slope():
    s = np.random.normal(SLOPE, SLOPE_STD)
    noise = np.random.normal(0, 1, LENGTH)
    w = 10*get_smoothing_window()
    noise = pynumdiff.savgoldiff(noise, dt, [3,w,w])[0]
    
    return s + noise

def get_sigmoid_midpoint():
    m = np.random.normal(SIGMOID_MIDPOINT, SIGMOID_MIDPOINT_STD)
    if m<0.02:
        m = 0.02
    return m

def get_sigmoid_steepness():
    s = np.random.normal(SIGMOID_STEEPNESS, SIGMOID_STEEPNESS_STD)
    if s<2:
        s = 2
    return s

def get_speed():
    s = np.random.normal(SPEED, SPEED_STD, LENGTH)
    w = get_smoothing_window()
    s = pynumdiff.savgoldiff(s, dt, [3,w,w])[0]
    return s

def get_noisy_course(course_angle):
    noise = np.random.normal(0, COURSE_ANGLE_STD, LENGTH)
    w = get_smoothing_window()
    noise = pynumdiff.savgoldiff(noise, dt, [3,w,w])[0]
    course_angle = wrap_angle(course_angle + noise)
    return course_angle

def get_upwind_translation_multiplier():
    s = np.abs(np.random.normal(UPWIND_TRANSLATION_MULTIPLIER, UPWIND_TRANSLATION_MULTIPLIER_STD))
    sign = np.random.choice([1,1,1,-1])
    return s*sign

def get_pre_search_course(wind_direction, wind_magnitude, 
                          before_flash_frames=20, 
                          after_flash_delay_frames=10,
                          anemometric_turn_frames=10,
                          total_after_flash_frames=100,
                          dt=0.01,
                         ):
    initial_heading = np.random.uniform(-np.pi/2, np.pi/2)
    pre_trajec = [initial_heading]*(before_flash_frames+after_flash_delay_frames)
    anemometric_turn = np.random.uniform(-np.pi, np.pi)
    pre_trajec.extend([initial_heading+anemometric_turn]*10)

    num_surge_frames = total_after_flash_frames - anemometric_turn_frames - after_flash_delay_frames
    surge = wind_direction + np.pi

    # don't surge if wind magnitude is really low
    ix = np.where(wind_magnitude<0.03)[0]
    surge[ix] = pre_trajec[-1]

    surge = surge[0:num_surge_frames]
    
    pre_trajec = np.hstack((pre_trajec, surge))

    pre_trajec = wrap_angle(pre_trajec)

    t = np.arange(-before_flash_frames, total_after_flash_frames, 1)
    
    return t*dt, pre_trajec

def generate_trajectory(scenario, length, add_pre_search=True, max_length=500):

    t, wind_direction, wind_speed = get_wind(scenario, RANDOMNESS, get_smoothing_window(), dt, length)
    raw_circle_course = get_circle_basis(t, RANDOMNESS, get_slope())
    axis_ratio = get_axis_ratio(wind_direction, wind_speed, get_sigmoid_midpoint(), get_sigmoid_steepness())
    color = get_color(scenario)

    raw_circle_xy = get_circle_xy(raw_circle_course)

    # affine trajec
    affine_trans_warped_course_angle, affine_trans_warped_circle_xy, mean_affine_trans_warped_circle_xy = get_affine_warped_course_angle(axis_ratio, 
                                                                                                                   wind_direction, 
                                                                                                                   raw_circle_xy,
                                                                                                                   translation_magnitude=get_upwind_translation_multiplier()
                                                                                                                   )
    if add_pre_search:
        pre_search_t, angle = get_pre_search_course(wind_direction, wind_speed)
        course = np.hstack((angle, affine_trans_warped_course_angle))[0:max_length]
        t = np.hstack((pre_search_t, t+1))[0:max_length]
        lights_on = np.zeros_like(t)
        lights_on[20:87] = 1

    x, y, xvel, yvel = get_trajec(get_noisy_course(course), dt, get_speed(), return_speed=True)

    # saccadic trajec
    staircase_warped_course_angle = get_upwind_biased_trajec_saccade(wind_direction, axis_ratio, affine_trans_warped_circle_xy, 
                                     dt,
                                     0,
                                     SMOOTHING_WINDOW, 
                                     SACCADE_THRESHOLD_PROPORTIONAL, SACCADE_THRESHOLD_DERIVATIVE,
                                     RANDOMNESS=False)
    if add_pre_search:
        pre_search_t, angle = get_pre_search_course(wind_direction, wind_speed)
        course = np.hstack((angle, staircase_warped_course_angle))[0:max_length]
        t = np.hstack((pre_search_t, t+1))[0:max_length]
        lights_on = np.zeros_like(t)
        lights_on[20:87] = 1

    sac_x, sac_y, sac_xvel, sac_yvel = get_trajec(get_noisy_course(course), dt, get_speed(), return_speed=True)

    df_affine = pandas.DataFrame({'x': x,
                                  'y': y,
                                  'xvel': xvel,
                                  'yvel': yvel,
                                  'time_relative_to_flash': t,
                                  'lights_on': lights_on})

    df_saccadic = pandas.DataFrame({'x': x,
                                    'y': y,
                                    'xvel': sac_xvel,
                                    'yvel': sac_yvel,
                                    'time_relative_to_flash': t,
                                    'lights_on': lights_on})

    return df_affine, df_saccadic

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)


    num_trajecs = 200

    obj_id = 0
    full_df = None
    for scenario in ['laminar', 'unsteady', 'stillair']:
        
        for i in range(num_trajecs):
            obj_id += 1

            df_affine, df_sac = generate_trajectory(scenario, 5)
            
            df = df_sac

            df['windtype'] = [scenario]*len(df_sac)
            df['obj_id'] = [obj_id]*len(df_sac)
            df['obj_id_unique'] = df['obj_id']
            df['obj_id_unique_event'] = df['obj_id']

            # calculate some useful things:
            print('Calculating speed (xy ground speed)')
            df['speed_xy'] = np.sqrt(df.xvel**2 + df.yvel**2)
            print('Calculating angular velocities -- this can take a really long time')
            df = flymath.assign_course_and_ang_vel_to_dataframe(df)

            if full_df is None:
                full_df = df    
            else:
                full_df = pandas.concat((full_df, df), ignore_index=True)

            

    full_df.to_parquet('new_unifying.parquet')
