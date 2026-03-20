import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import random
import math

import pandas as pd

import copy

import cvxpy
cp = cvxpy

import figurefirst as fifi

from braid_analysis import braid_analysis_plots
from braid_analysis import flymath
import fly_plot_lib.plot as fpl

import pynumdiff

import scipy.stats

# Plotting parameters
course_markersize = 2
linear_model_markersize = 3
affine_model_markersize = 3

def construct_affine_transform(rotation_angle, major_axis, minor_axis, translation_direction=None, translation_magnitude=None):
    """
    rotation_angle: in radians
    major_axis, minor_axis: scaling factors
    """
    # Rotation matrix
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array([[c, -s], [s, c]])
    
    # Scaling matrix
    S = np.array([[major_axis, 0], [0, minor_axis]])
    
    # Linear transformation
    A = R @ S @ R.T
    
    # Translation (adjust as needed)
    if translation_direction is not None:
        b = np.array([[translation_magnitude*np.cos(translation_direction)], [translation_magnitude*np.sin(translation_direction)]])  # or compute based on desired center
    
    if translation_direction is not None:
        return A, b
    else:
        return A

def stretch_sigmoid(x, midpoint=0.1, steepness=8):
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

def wrap_angle(angle, midangle=0):
    """
    midangle = 0: Wrap angle to range [-π, π]
    midangle = 'pi': Wrap angle to range [, 2π]
    """
    if midangle == 0:
        return ((angle + np.pi) % (2*np.pi)) - np.pi
    elif midangle == 'pi':
        return ((angle) % (2*np.pi)) - np.pi

def mean_angle(angle):
    mean = np.arctan2( np.mean(np.sin(angle)), np.mean(np.cos(angle)) )
    return mean

def get_wind(scenario, RANDOMNESS=False, SMOOTHING_WINDOW=11, dt=0.01, length=5):
    tstart = 5
    if RANDOMNESS:
        tstart = np.random.uniform(2, 14, 1)
    interp_t = np.arange(tstart, tstart+length, dt) 
    if len(interp_t) > int(length/dt):
        interp_t = interp_t[0:int(length/dt)]

    t = interp_t - interp_t[0]
    
    # load unsteady wind
    unsteady_wind = pd.read_hdf('unsteady_timeseries.hdf')
    cfd_t = unsteady_wind.Time.values
    cfd_xvel = unsteady_wind.xvel.values
    cfd_yvel = unsteady_wind.yvel.values
    interp_cfd_xvel = np.interp(interp_t, cfd_t, cfd_xvel)
    interp_cfd_yvel = np.interp(interp_t, cfd_t, cfd_yvel)
    interp_cfd_xvel_smooth, _ = pynumdiff.savgoldiff(interp_cfd_xvel, dt, [3, SMOOTHING_WINDOW, SMOOTHING_WINDOW])
    interp_cfd_yvel_smooth, _ = pynumdiff.savgoldiff(interp_cfd_yvel, dt, [3, SMOOTHING_WINDOW, SMOOTHING_WINDOW])
    cfd_wind_direction = np.arctan2(interp_cfd_yvel_smooth, interp_cfd_xvel_smooth)
    cfd_wind_speed = (interp_cfd_xvel_smooth**2 + interp_cfd_yvel_smooth**2)**0.5
    
    # generate laminar wind
    if RANDOMNESS:
        std = 0.05
    else:
        std = 0
    lam_xvel = -0.4 + np.random.normal(0, std, len(t))
    lam_yvel = 0 + np.random.normal(0, std, len(t))
    lam_xvel_smooth, _ = pynumdiff.savgoldiff(lam_xvel, dt, [3, SMOOTHING_WINDOW, SMOOTHING_WINDOW])
    lam_yvel_smooth, _ = pynumdiff.savgoldiff(lam_yvel, dt, [3, SMOOTHING_WINDOW, SMOOTHING_WINDOW])
    lam_wind_direction = np.arctan2(lam_yvel_smooth, lam_xvel_smooth)
    lam_wind_speed = (lam_xvel_smooth**2 + lam_yvel_smooth**2)**0.5

    
    
    # generate still air
    if RANDOMNESS:
        std = 0.02
    else:
        std = 0
    still_xvel = -0 + np.random.normal(0, std, len(t))
    still_yvel = 0 + np.random.normal(0, std, len(t))
    still_xvel_smooth, _ = pynumdiff.savgoldiff(still_xvel, dt, [3, SMOOTHING_WINDOW, SMOOTHING_WINDOW])
    still_yvel_smooth, _ = pynumdiff.savgoldiff(still_yvel, dt, [3, SMOOTHING_WINDOW, SMOOTHING_WINDOW])
    still_wind_direction = np.arctan2(still_yvel_smooth, still_xvel_smooth)
    still_wind_speed = (still_xvel_smooth**2 + still_yvel_smooth**2)**0.5

    if scenario == 'laminar':
        return t, lam_wind_direction, lam_wind_speed

    if scenario == 'stillair':
        return t, still_wind_direction, still_wind_speed

    if scenario == 'unsteady':
        return t, cfd_wind_direction, cfd_wind_speed

    if scenario == 'angled_laminar':
        # generate angled laminar wind
        if RANDOMNESS:
            std = 0.05
        else:
            std = 0
        lam_xvel = -0.3*0.8 + np.random.normal(0, std, len(t))
        lam_yvel = -0.1*0.8 + np.random.normal(0, std, len(t))
        lam_xvel_smooth, _ = pynumdiff.savgoldiff(lam_xvel, dt, [3, SMOOTHING_WINDOW, SMOOTHING_WINDOW])
        lam_yvel_smooth, _ = pynumdiff.savgoldiff(lam_yvel, dt, [3, SMOOTHING_WINDOW, SMOOTHING_WINDOW])
        lam_wind_direction = np.arctan2(lam_yvel_smooth, lam_xvel_smooth)
        lam_wind_speed = (lam_xvel_smooth**2 + lam_yvel_smooth**2)**0.5
        return t, lam_wind_direction, lam_wind_speed


def get_circle_basis(t, RANDOMNESS, slope=3.95):
    intercept = 0
    if RANDOMNESS:
        intercept = np.random.uniform(-np.pi, np.pi, 1)
        sign = np.random.choice([1, -1])
        slope *= sign
    smooth_circle_course = slope*t + intercept
    raw_circle_course = wrap_angle(smooth_circle_course)
    return raw_circle_course

def get_axis_ratio(wind_direction, wind_speed, midpoint=0.1, steepness=8):
    rotation = wrap_angle( wind_direction + np.pi/2. )
    wind_x = wind_speed*np.cos(wind_direction)
    wind_y = wind_speed*np.sin(wind_direction)
    wind_xy = np.vstack((wind_x, wind_y))
    vecstr = np.linalg.norm(np.sum(wind_xy, axis=1)) / wind_xy.shape[1]
    axis_ratio = 1-stretch_sigmoid(vecstr, midpoint=midpoint, steepness=steepness)
    return axis_ratio

def get_color(scenario):
    if scenario == 'laminar':
        return '#991128ff'

    if scenario == 'angled_laminar':
        return '#991128ff'

    if scenario == 'unsteady':
        return '#2b75b3b3'

    if scenario == 'stillair':
        return '#084a72ff'

def get_circle_xy(raw_circle_course):
    # raw circle
    raw_circle_x = np.cos(raw_circle_course)
    raw_circle_y = np.sin(raw_circle_course)
    raw_circle_xy = np.vstack([raw_circle_x, raw_circle_y])
    return raw_circle_xy
    
def get_affine_warped_course_angle(axis_ratio, wind_direction, raw_circle_xy, 
                                   translation_magnitude=0):
    # rotation
    rotation = wind_direction + np.pi/2

    # major and minor axes
    major_axis = 1
    minor_axis = major_axis*axis_ratio

    # affine warping
    if translation_magnitude != 0:
        AMs, Bs = [], []
        for i in range(len(rotation)):
            upwind_direction = wrap_angle(wind_direction[i]+np.pi)
            AM, B = construct_affine_transform(rotation[i], major_axis, minor_axis, 
                                               translation_direction=upwind_direction, translation_magnitude=translation_magnitude)
            AMs.append(AM)
            Bs.append(B)
        affine_warped_circle_xy = np.hstack([AMs[i]@raw_circle_xy[:,i:i+1]+Bs[i] for i in range(len(AMs))])
        affine_warped_course_angle = np.arctan2(affine_warped_circle_xy[1,:], affine_warped_circle_xy[0,:])
    else:
        AMs = [construct_affine_transform(r, major_axis, minor_axis) for r in rotation]
        affine_warped_circle_xy = np.hstack([AMs[i]@raw_circle_xy[:,i:i+1] for i in range(len(AMs))])
        affine_warped_course_angle = np.arctan2(affine_warped_circle_xy[1,:], affine_warped_circle_xy[0,:])

    # mean
    if translation_magnitude != 0:
        mean_rotation = mean_angle(rotation)
        mean_upwind_direction = mean_angle(upwind_direction)
        AM, B = construct_affine_transform(mean_rotation, major_axis, minor_axis, 
                                               translation_direction=upwind_direction, translation_magnitude=translation_magnitude)
        mean_affine_warped_circle_xy = AM@raw_circle_xy + B
    else:
        mean_rotation = mean_angle(rotation)
        AM = construct_affine_transform(mean_rotation, major_axis, minor_axis)
        mean_affine_warped_circle_xy = AM@raw_circle_xy
    
    return affine_warped_course_angle, affine_warped_circle_xy, mean_affine_warped_circle_xy

def get_mean_affine_matrix(axis_ratio, wind_direction, raw_circle_xy):
    # rotation
    rotation = wind_direction + np.pi/2

    # major and minor axes
    major_axis = 1
    minor_axis = major_axis*axis_ratio
    
    # mean
    mean_rotation = mean_angle(rotation)
    AM = construct_affine_transform(mean_rotation, major_axis, minor_axis)
    
    return AM

def get_trajec(course_angle, dt, SPEED, return_speed=False):
        
    speed = SPEED
    velx = speed*np.cos(course_angle)
    vely = speed*np.sin(course_angle)
    
    x = np.cumsum(velx)*dt
    y = np.cumsum(vely)*dt
    
    if not return_speed:
        return x, y
    else:
        return x, y, velx, vely

def get_upwind_biased_trajec_cvx(wind_direction, axis_ratio, affine_warped_course_angle, 
                                 dt,
                                 UPWIND_GAMMA_MULTIPLIER, SMOOTHING_WINDOW, SPEED, RANDOMNESS=False):
    if RANDOMNESS:
        print('Not implemented yet! could implement random gamma mulktiplier')
    #######################
    upwind_direction = wrap_angle(wind_direction+np.pi)
    
    # option A: penalize turning through downwind if there is wind
    gamma = UPWIND_GAMMA_MULTIPLIER*(1-axis_ratio)
    course_cvx = cvxpy.Variable(len(affine_warped_course_angle))
    pis = cvxpy.Variable(len(affine_warped_course_angle), integer=True)
    
    eq = course_cvx - affine_warped_course_angle + pis*2*np.pi
    L1 = cvxpy.sum(cvxpy.huber(eq, M=0.1))
    L2 = cvxpy.sum_squares(course_cvx-upwind_direction)
    
    objective = cvxpy.Minimize(L1 + gamma*L2
                               )
    constraints = [
      pis >= -10, pis <= 10,  # Adjust based on your data
    ]
    
    problem = cvxpy.Problem(objective, constraints)
    
    problem.solve(solver=cvxpy.GUROBI,
                Threads=8,              # Use multiple cores
                Method=2,               # Try different algorithms: -1=auto, 0=primal, 1=dual, 2=barrier
                Presolve=2,             # Aggressive presolve
                FeasibilityTol=1e-5,    # Relax from 1e-6
                OptimalityTol=1e-5,     # Relax from 1e-6
                TimeLimit=20,          # Stop after 5 minutes
                MIPGap=0.01,            # For MIP: accept 1% gap (if applicable)
                ) 
    
    unwrapped_course_cvx = flymath.unwrap_angle(course_cvx.value, correction_window_for_2pi=40)
    cvx_warped_course_angle = wrap_angle(pynumdiff.savgoldiff(unwrapped_course_cvx, dt, [3,SMOOTHING_WINDOW,SMOOTHING_WINDOW])[0])
    cvx_warped_circle_xy = np.vstack([np.cos(cvx_warped_course_angle), np.sin(cvx_warped_course_angle)])

    return cvx_warped_course_angle, cvx_warped_circle_xy

def plot_course(ax, t, raw_circle_course, affine_warped_course_angle, cvx_warped_course_angle, 
                show_labels=False, 
                colors=['blue', 'magenta', 'black'],
                markersizes=[3,3,2],
                spines=['left','bottom']):

    # Course
    if raw_circle_course is not None:
        ax.plot(t, raw_circle_course, '.', label='line fit', color=colors[0], markersize=markersizes[0], markeredgecolor='none')
    if affine_warped_course_angle is not None:
        ax.plot(t, affine_warped_course_angle, '.', label='line fit', color=colors[1], markersize=markersizes[1], markeredgecolor='none')
    if cvx_warped_course_angle is not None:
        ax.plot(t, cvx_warped_course_angle, '.', label='raw data', color=colors[2], markersize=markersizes[2], markeredgecolor='none')
    
    xticks = [0, 1, 2, 3, 4, 5]
    xticklabels = np.array(xticks)
    
    ax.set_xlim(0, 5)
    ax.set_xticks(xticks)
    
    if show_labels:
        ax.set_xticklabels(['$0$', '$1$', '$2$', '$3$', '$4$', '$5$'])
    else:
        ax.set_xticklabels([])
        
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_ylim(-np.pi, np.pi)
    
    if show_labels:
        ax.set_yticklabels(['$-\pi$', '','$0$','','$\pi$',])
    else:
        ax.set_yticklabels([])
        
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'])
    
    if show_labels:
        ax.set_ylabel('Course direction', labelpad=-2)
        ax.set_xlabel('Time (s)')
    
    ax.tick_params(axis='y', pad=2)
    ax.tick_params(axis='x', pad=2)
    
    fifi.mpl_functions.adjust_spines(ax, spines,
                                     tick_length=2.5,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5)
    fifi.mpl_functions.set_fontsize(ax, 6)

def get_upwind_biased_trajec_saccade(wind_direction, axis_ratio, affine_warped_circle_xy, 
                                     dt,
                                     UPWIND_GAMMA_MULTIPLIER,
                                     SMOOTHING_WINDOW, 
                                     SACCADE_THRESHOLD_PROPORTIONAL, SACCADE_THRESHOLD_DERIVATIVE,
                                     RANDOMNESS=False):
    # upwind
    U = np.vstack((np.cos(wind_direction+np.pi), np.sin(wind_direction+np.pi)))
    
    gamma = UPWIND_GAMMA_MULTIPLIER*(1-axis_ratio)#**2
    upwind_vector = U
    weighted_affine_warped_circle_xy = affine_warped_circle_xy*(1-gamma) + upwind_vector*gamma

    warped_circle_x = weighted_affine_warped_circle_xy[0,:]
    warped_circle_y = weighted_affine_warped_circle_xy[1,:]
    course_angle = np.arctan2(warped_circle_y, warped_circle_x)
    
    unwrapped_course_angle = flymath.unwrap_angle(course_angle, correction_window_for_2pi=40)
    
    _, unwrapped_course_angle_diff = pynumdiff.savgoldiff(unwrapped_course_angle, dt, [3, SMOOTHING_WINDOW, SMOOTHING_WINDOW])


    threshold = SACCADE_THRESHOLD_PROPORTIONAL
    staircase_approx = [unwrapped_course_angle[0]]
    error = 0
    for i in range(1, len(unwrapped_course_angle)):
        error += unwrapped_course_angle[i] - staircase_approx[-1]
        if np.abs(error) > threshold:
            staircase_approx.append(unwrapped_course_angle[i] + unwrapped_course_angle_diff[i]*SACCADE_THRESHOLD_DERIVATIVE*dt)
            error = 0
        else:
            staircase_approx.append(staircase_approx[-1])
    
    staircase_warped_course_angle = np.array(staircase_approx)
    
    staircase_warped_course_x = np.cos(staircase_warped_course_angle)
    staircase_warped_course_y = np.sin(staircase_warped_course_angle)
    
    staircase_warped_course_x, _ = pynumdiff.savgoldiff(staircase_warped_course_x, dt, [3, SMOOTHING_WINDOW, SMOOTHING_WINDOW])
    staircase_warped_course_y, _ = pynumdiff.savgoldiff(staircase_warped_course_y, dt, [3, SMOOTHING_WINDOW, SMOOTHING_WINDOW])
    staircase_warped_course_angle = np.arctan2(staircase_warped_course_y, staircase_warped_course_x)

    return staircase_warped_course_angle

###################

import matplotlib.collections as mcollections
import matplotlib.lines as mlines

import matplotlib.collections as mcollections
import matplotlib.lines as mlines

def set_selective_rasterization(ax, 
                                rasterize_markers=None, 
                                rasterize_collections=None,
                                rasterize_linestyles=None,
                                raster_zorder=-1,
                                threshold=0,
                                preserve_other_zorder=True):
    """
    Selectively rasterize plot elements in a matplotlib axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to modify
    rasterize_markers : list of str, optional
        List of marker types to rasterize (e.g., ['.', 'o', '*'])
        If None, marker type is not used as a criterion
    rasterize_collections : list of type, optional
        List of collection types to rasterize 
        (e.g., [mcollections.PathCollection, mcollections.LineCollection])
        If None, collection type is not used as a criterion
    rasterize_linestyles : list of str, optional
        List of line styles to rasterize (e.g., ['-', '--'])
        If None, linestyle is not used as a criterion
    raster_zorder : float, default=-1
        zorder value for rasterized elements (must be < threshold)
    threshold : float, default=0
        Rasterization threshold. Elements with zorder < threshold are rasterized
    preserve_other_zorder : bool, default=True
        If True, only modify zorder of elements being rasterized.
        If False, set non-rasterized elements to their original zorder (no change).
    
    Notes
    -----
    When both rasterize_markers and rasterize_linestyles are specified,
    a line will be rasterized if it matches EITHER criterion (OR logic).
    Elements not selected for rasterization are left unchanged by default.
    """
    
    # Handle collections
    for collection in ax.collections:
        should_rasterize = False
        
        if rasterize_collections is not None:
            # Check if collection type is in the rasterize list
            should_rasterize = any(isinstance(collection, ctype) 
                                  for ctype in rasterize_collections)
        
        if should_rasterize:
            collection.set_zorder(raster_zorder)
        # If preserve_other_zorder is True, don't touch non-rasterized elements
    
    # Handle line plots
    for line in ax.lines:
        should_rasterize = False
        
        marker = line.get_marker()
        linestyle = line.get_linestyle()
        
        # Check marker criteria
        marker_match = False
        if rasterize_markers is not None:
            if marker != 'None' and marker in rasterize_markers:
                marker_match = True
        
        # Check linestyle criteria  
        linestyle_match = False
        if rasterize_linestyles is not None:
            if linestyle != 'None' and linestyle in rasterize_linestyles:
                linestyle_match = True
        
        # OR logic: rasterize if EITHER condition is met
        if rasterize_markers is not None or rasterize_linestyles is not None:
            should_rasterize = marker_match or linestyle_match
        
        if should_rasterize:
            line.set_zorder(raster_zorder)
        # If preserve_other_zorder is True, don't touch non-rasterized elements
    
    # Set the rasterization threshold
    ax.set_rasterization_zorder(threshold)

def diagnose_axis_elements(ax, verbose=True):
    """
    Diagnose all plot elements in an axis that could be selectively rasterized.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to diagnose
    verbose : bool, default=True
        If True, print detailed information. If False, return data structure only.
    
    Returns
    -------
    dict
        Dictionary containing information about all rasterizable elements:
        {
            'collections': list of dicts with collection info,
            'lines': list of dicts with line info,
            'summary': dict with counts by type
        }
    """
    
    diagnosis = {
        'collections': [],
        'lines': [],
        'summary': {
            'total_collections': 0,
            'total_lines': 0,
            'collection_types': {},
            'marker_types': {},
            'linestyle_types': {}
        }
    }
    
    # Analyze collections
    if verbose:
        print("=" * 70)
        print("COLLECTIONS (scatter, fill_between, etc.)")
        print("=" * 70)
    
    for i, collection in enumerate(ax.collections):
        ctype = type(collection)
        ctype_name = ctype.__name__
        
        # Get collection properties
        info = {
            'index': i,
            'type': ctype,
            'type_name': ctype_name,
            'zorder': collection.get_zorder(),
            'label': collection.get_label() if hasattr(collection, 'get_label') else None
        }
        
        # Try to get additional properties
        try:
            info['facecolor'] = collection.get_facecolor()
        except:
            pass
        
        try:
            info['edgecolor'] = collection.get_edgecolor()
        except:
            pass
        
        diagnosis['collections'].append(info)
        
        # Update summary
        diagnosis['summary']['total_collections'] += 1
        diagnosis['summary']['collection_types'][ctype_name] = \
            diagnosis['summary']['collection_types'].get(ctype_name, 0) + 1
        
        if verbose:
            print(f"\nCollection {i}:")
            print(f"  Type: {ctype_name}")
            print(f"  Full type: {ctype}")
            print(f"  Label: {info['label']}")
            print(f"  Current zorder: {info['zorder']}")
    
    # Analyze lines
    if verbose:
        print("\n" + "=" * 70)
        print("LINES (plot, with markers and linestyles)")
        print("=" * 70)
    
    for i, line in enumerate(ax.lines):
        marker = line.get_marker()
        linestyle = line.get_linestyle()
        
        info = {
            'index': i,
            'marker': marker,
            'linestyle': linestyle,
            'color': line.get_color(),
            'linewidth': line.get_linewidth(),
            'label': line.get_label(),
            'zorder': line.get_zorder()
        }
        
        diagnosis['lines'].append(info)
        
        # Update summary
        diagnosis['summary']['total_lines'] += 1
        
        if marker != 'None':
            diagnosis['summary']['marker_types'][marker] = \
                diagnosis['summary']['marker_types'].get(marker, 0) + 1
        
        if linestyle != 'None':
            diagnosis['summary']['linestyle_types'][linestyle] = \
                diagnosis['summary']['linestyle_types'].get(linestyle, 0) + 1
        
        if verbose:
            print(f"\nLine {i}:")
            print(f"  Label: {info['label']}")
            print(f"  Marker: '{marker}'")
            print(f"  Linestyle: '{linestyle}'")
            print(f"  Color: {info['color']}")
            print(f"  Linewidth: {info['linewidth']}")
            print(f"  Current zorder: {info['zorder']}")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nTotal collections: {diagnosis['summary']['total_collections']}")
        if diagnosis['summary']['collection_types']:
            print("\nCollection types found:")
            for ctype, count in diagnosis['summary']['collection_types'].items():
                print(f"  {ctype}: {count}")
        
        print(f"\nTotal lines: {diagnosis['summary']['total_lines']}")
        if diagnosis['summary']['marker_types']:
            print("\nMarker types found:")
            for marker, count in diagnosis['summary']['marker_types'].items():
                print(f"  '{marker}': {count}")
        
        if diagnosis['summary']['linestyle_types']:
            print("\nLinestyle types found:")
            for linestyle, count in diagnosis['summary']['linestyle_types'].items():
                print(f"  '{linestyle}': {count}")
        
        # Provide suggestions
        print("\n" + "=" * 70)
        print("RASTERIZATION SUGGESTIONS")
        print("=" * 70)
        
        if diagnosis['summary']['collection_types']:
            print("\nTo rasterize collections, use:")
            print("  rasterize_collections=[")
            for ctype in diagnosis['summary']['collection_types'].keys():
                print(f"      mcollections.{ctype},")
            print("  ]")
        
        if diagnosis['summary']['marker_types']:
            print("\nTo rasterize by marker type, use:")
            markers = list(diagnosis['summary']['marker_types'].keys())
            print(f"  rasterize_markers={markers}")
        
        if diagnosis['summary']['linestyle_types']:
            print("\nTo rasterize by linestyle, use:")
            linestyles = list(diagnosis['summary']['linestyle_types'].keys())
            print(f"  rasterize_linestyles={linestyles}")
        
        print("\n" + "=" * 70)
    
    return diagnosis
