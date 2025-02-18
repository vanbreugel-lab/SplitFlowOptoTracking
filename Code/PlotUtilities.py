import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fly_plot_lib.plot as fpl
import figurefirst as fifi
from matplotlib import patches
from matplotlib.collections import PatchCollection
import pynumdiff

def mean_angle(angle):
    return np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle)))
#############################################################################################################
# Arrow head trajectories

def get_wedges_for_heading_plot(x, y, color, orientation, size_radius=0.1, size_angle=20, colormap='jet', colornorm=None, size_radius_range=(0.01,.1), size_radius_norm=None, edgecolor='none', alpha=1, flip=True, deg=True, nskip=0, center_offset_fraction=0.75):
    '''
    Returns a Patch Collection of Wedges, with arbitrary color and orientation
    
    Outputs:
    Patch Collection
    
    Inputs:
    x, y        - x and y positions (np.array or list, each of length N)
    color       - values to color wedges by (np.array or list, length N), OR color string. 
       colormap - specifies colormap to use (string, eg. 'jet')
       norm     - specifies range you'd like to normalize to, 
                  if none, scales to min/max of color array (2-tuple, eg. (0,1) )
    orientation - angles are in degrees, use deg=False to convert radians to degrees
    size_radius - radius of wedge, in same units as x, y. Can be list or np.array, length N, for changing sizes
       size_radius_norm - specifies range you'd like to normalize size_radius to, if size_radius is a list/array
                  should be tuple, eg. (0.01, .1)
    size_angle  - angular extent of wedge, degrees. Can be list or np.array, length N, for changing sizes
    edgecolor   - color for lineedges, string or np.array of length N
    alpha       - transparency (single value, between 0 and 1)
    flip        - flip orientations by 180 degrees, default = True
    nskip       - allows you to skip between points to make the points clearer, nskip=1 skips every other point
    center_offset_fraction  - (float in range (0,1) ) - 0 means (x,y) is at the tip, 1 means (x,y) is at the edge
    '''
    cmap = plt.get_cmap(colormap)
    
    # norms
    if colornorm is None and type(color) is not str:
        colornorm = plt.Normalize(np.min(color), np.max(color))
    elif type(color) is not str:
        colornorm = plt.Normalize(colornorm[0], colornorm[1])
    if size_radius_norm is None:
        size_radius_norm = plt.Normalize(np.min(size_radius), np.max(size_radius), clip=True)
    else:
        size_radius_norm = plt.Normalize(size_radius_norm[0], size_radius_norm[1], clip=True)
        
    indices_to_plot = np.arange(0, len(x), nskip+1)
        
    # fix orientations
    if type(orientation) is list:
        orientation = np.array(orientation)
    if deg is False:
        orientation = orientation*180./np.pi
    if flip:
        orientation += 180
    
    flycons = []
    n = 0
    for i in indices_to_plot:
        # wedge parameters
        if type(size_radius) is list or type(size_radius) is np.array or type(size_radius) is np.ndarray: 
            r = size_radius_norm(size_radius[i])*(size_radius_range[1]-size_radius_range[0]) + size_radius_range[0] 
        else: r = size_radius
        
        if type(size_angle) is list or type(size_angle) is np.array or type(size_angle) is np.ndarray: 
            angle_swept = size_radius[i]
        else: angle_swept = size_radius
        theta1 = orientation[i] - size_angle/2.
        theta2 = orientation[i] + size_angle/2.
        
        center = [x[i], y[i]]
        center[0] -= np.cos(orientation[i]*np.pi/180.)*r*center_offset_fraction
        center[1] -= np.sin(orientation[i]*np.pi/180.)*r*center_offset_fraction
        
        wedge = patches.Wedge(center, r, theta1, theta2)
        flycons.append(wedge)
        
    # add collection and color it
    pc = PatchCollection(flycons, cmap=cmap, norm=colornorm)
    
    # set properties for collection
    pc.set_edgecolors(edgecolor)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        if type(color) is list:
            color = np.asarray(color)
        pc.set_array(color[indices_to_plot])
    else:
        pc.set_facecolors(color)
    pc.set_alpha(alpha)
    
    return pc

def plot_arrowhead_trajectory(x, y, color='black', arrow_length=0.06, arrow_angle=30, ax=None, linewidth=1):
    xvel = pynumdiff.finite_difference.second_order(x, 1)[1]
    yvel = pynumdiff.finite_difference.second_order(y, 1)[1]
    orientations = np.arctan2(yvel, xvel)

    last_orientation = mean_angle(orientations[-3:])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        
    ax.plot(x,y, color=color, linewidth=linewidth)
    wedge = get_wedges_for_heading_plot([x[-1],],[y[-1],], color, [last_orientation*180/np.pi,], 
                                        size_radius=arrow_length, size_angle=arrow_angle)
    ax.add_collection(wedge)
    

def plot_feature(dfs, feature_name, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for i, df in enumerate(dfs):
        feature_val = df.groupby('obj_id_unique')[feature_name].mean().values 

        b, h = np.histogram(feature_val)
        fpl.scatter_box_violin_distribution(ax, i, feature_val, b, h, color=df_colors[i])

    ax.set_xticks(np.arange(len(dfs)))
    ax.set_xticklabels(df_labels)
    ax.set_ylabel(feature_name)
    


def get_subplot(df, traj_id,ax):
    ax = ax
    lw=1
    ms=2
    t1 = df[df.obj_id_unique==traj_id]
    pre=t1[t1['time stamp'].between(-500, 0)]
    dur=t1[t1['time stamp'].between(0, 675)]
    post = t1[t1['time stamp'].between(675,4500)]
    
    #ax.plot(pre['x'], pre['y'], color ='k',alpha= 0.8, linewidth =lw)
    plot_arrowhead_trajectory(t1[t1['time stamp'].between(675,4500)].x.values, t1[t1['time stamp'].between(675,4500)].y.values,ax=ax)
    ax.plot(pre['x'], pre['y'], color ='grey',alpha= 0.8, linewidth =lw)
    ax.plot(dur['x'], dur['y'], color ='r', alpha =1, linewidth =lw)
    
    #ax.plot(post['x'], post['y'], color ='k', alpha = .8, linewidth = lw)
    #ax.plot(pre['x'].iloc[0], pre['y'].iloc[0], '>', color='k', markersize=ms)
    ax.axis('off')
    ax.set_xlim(-.5,.5)
    ax.set_ylim(-.25,.25)
    ax.set_aspect('equal')
    return ax
        
    
def plot_several_features(df, labels, cmap='YlOrBr', n=8,  ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    feature_cmap = plt.get_cmap(cmap, n) 
    for i, feature_name in enumerate(df.columns):
        feature_val = df[feature_name].values

        b, h = np.histogram(feature_val)
        fpl.scatter_box_violin_distribution(ax, i, feature_val, b, h, markersize=1, color=feature_cmap(i+2))
 

    fifi.mpl_functions.adjust_spines(ax, ['bottom',], xticks =[0,1,2,3],  spine_locations={'bottom':5, }, tick_length=3, linewidth=.5)

    fifi.mpl_functions.set_fontsize(ax, 6)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(labels)
    ax.set_aspect(2)

    
def plot_feature(dfs, feature_name,df_colors, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for i, df in enumerate(dfs):
        feature_val = df.groupby('obj_id_unique')[feature_name].mean().values 

        b, h = np.histogram(feature_val)
        fpl.scatter_box_violin_distribution(ax, i, feature_val, b, h, markersize=1, color=df_colors[i])
    
    fifi.mpl_functions.adjust_spines(ax, ['left','bottom'], xticks =[0,1],  spine_locations={'left':5, 'bottom':5, }, tick_length=3, linewidth=.5)
    fifi.mpl_functions.set_fontsize(ax, 6)
    ax.set_xticks(np.arange(len(dfs)))
    ax.set_xticklabels(['',''])
    #ax.set_ylabel(feature_name)