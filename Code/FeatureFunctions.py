import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import scipy
import pynumdiff
import sklearn

import fly_plot_lib.plot as fpl

def cull_slow_flies(df, tref=1100):
    cull_objids = []
    for objid in df.obj_id_unique.unique():
        trajec = df[df.obj_id_unique==objid]
        snip = trajec[(trajec['time stamp']>tref)]
        speed = np.sqrt(snip.xvel**2 + snip.yvel**2)
        if np.median(speed) < 0.15:
            cull_objids.append(objid)
    print('Culled ' + str(len(cull_objids)) + ' trajectories')
    df = df[~df['obj_id_unique'].isin(cull_objids)]
    return df

# calculate circlyness 

def fit_circle(trajec, start_time, window_time, plot=False):
    '''
    fit a circle
    start_time, window_time in ms
    '''
    snip = trajec[(trajec['time stamp']>start_time)*(trajec['time stamp']<(start_time+window_time))]
    x = snip.x.values
    y = snip.y.values

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = snip.x.mean(), snip.y.mean()
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(snip.x, snip.y)
        circle1 = plt.Circle((xc_2, yc_2), R_2, color='r')
        ax.add_patch(circle1)

    return xc_2, yc_2, R_2

def circle_cross(trajec, start_time, window_time):
    '''
    cross product between xy vel and a circular trajectory
    '''
    xc, yc, R = fit_circle(trajec, start_time, window_time)
    snip = trajec[(trajec['time stamp']>start_time)*(trajec['time stamp']<(start_time+window_time))]
    x = snip.x - xc
    y = snip.y - yc
    x_y = np.vstack([x,y]).T
    x_y = x_y / np.atleast_2d(np.linalg.norm(x_y, axis=1)).T
    xvel_yvel = np.vstack([snip.xvel.values, snip.yvel.values]).T
    xvel_yvel = xvel_yvel / np.atleast_2d(np.linalg.norm(xvel_yvel, axis=1)).T
    cross = np.sum(np.cross(x_y, xvel_yvel, axis=1)) / x_y.shape[0]
    return cross, R, xc, yc

def calc_circlyness(df, start_time=900, window_time=1500):
    df['circle_R'] = np.nan
    df['circle_x'] = np.nan
    df['circle_y'] = np.nan
    df['circle_cross'] = np.nan
    
    for objid in df.obj_id_unique.unique():
        trajec = df[df.obj_id_unique==objid]
        
        snip = trajec[(trajec['time stamp']>start_time)*(trajec['time stamp']<(start_time+window_time))]
        if len(snip)<10:
            continue
        cross, R, xc, yc = circle_cross(trajec, start_time, window_time)
        #if R < 0.5:
        #    df_cross.append(cross)
        #    df_R.append(R)
            
        df.loc[trajec.index, 'circle_R'] = R
        df.loc[trajec.index, 'circle_x'] = xc
        df.loc[trajec.index, 'circle_y'] = yc
        df.loc[trajec.index, 'circle_cross'] = cross
            
    return df

# area function 
def calc_hull_area(df, start_time=900, window_time=1500):
    df['hull_area'] = np.nan
    df['hull_roundness'] = np.nan
    
    for objid in df.obj_id_unique.unique():
        trajec = df[df.obj_id_unique==objid]
        snip = trajec[(trajec['time stamp']>start_time)*(trajec['time stamp']<(start_time+window_time))]
        if len(snip)<10:
            continue
        hull = scipy.spatial.ConvexHull( np.vstack((snip.x.values, snip.y.values)).T )

        x = hull.points[:,0]
        xm = np.mean(x)
        y = hull.points[:,1]
        ym = np.mean(y)

        rmin = np.min( np.sqrt( (x-xm)**2 + (y-ym)**2 ) )
        #rmax = np.max( np.sqrt( (x-xm)**2 + (y-ym)**2 ) )
        rmax = np.sqrt(hull.area/np.pi)
        roundness = rmin/rmax
        
        df.loc[trajec.index, 'hull_area'] = hull.area
        df.loc[trajec.index, 'hull_roundness'] = roundness

    return df



def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

def mean_angle(angle):
    return np.arctan2(np.nanmean(np.sin(angle)), np.nanmean(np.cos(angle)))
    
def std_angle(angle):
    return np.sqrt( np.sum(wrap_angle(angle - mean_angle(angle))**2) / len(angle) )

def unwrap_angle(z, correction_window_for_2pi=10, n_range=2, plot=False):
    smooth_zs = np.array(z[0:2])
    for i in range(2, len(z)):
        first_ix = np.max([0, i-correction_window_for_2pi])
        last_ix = i

        nbase = np.round( (smooth_zs[-1] - z[i])/(2*np.pi) )

        candidates = []
        for n in range(-1*n_range, n_range):
            candidates.append(n*2*np.pi+nbase*2*np.pi+z[i])
        error = np.abs(candidates - np.mean(smooth_zs[first_ix:last_ix])) 
        smooth_zs = np.hstack(( smooth_zs, [candidates[np.argmin(error)]] ))
    if plot:
        plt.plot(smooth_zs, '.', color='black', markersize=1)
    return smooth_zs

def calc_heading_derivative(df):
    df['heading_unwrapped'] = np.nan
    df['heading_derivative'] = np.nan
    df['groundspeed'] = np.nan
    
    for objid in df.obj_id_unique.unique():
        trajec = df[df.obj_id_unique==objid]
        heading_unwrapped = unwrap_angle(trajec['heading'].values)
        _, heading_derivative = pynumdiff.linear_model.savgoldiff(heading_unwrapped, 0.01, [2,20,20])
        df.loc[trajec.index, 'heading_unwrapped'] = pd.Series(heading_unwrapped, index=trajec.index)
        df.loc[trajec.index, 'heading_derivative'] = pd.Series(heading_derivative, index=trajec.index)
        
        groundspeed = np.sqrt(trajec.xvel**2 + trajec.yvel**2)
        df.loc[trajec.index, 'groundspeed'] = pd.Series(groundspeed,index=trajec.index)
        
    return df

def calc_heading_derivative_mean(df, start_time=900, window_time=1500):
    df['heading_derivative_absmean'] = np.nan
    
    for objid in df.obj_id_unique.unique():
        trajec = df[df.obj_id_unique==objid]
        snip = trajec[(trajec['time stamp']>start_time)*(trajec['time stamp']<(start_time+window_time))]
        if len(snip)<10:
            continue
        df.loc[trajec.index, 'heading_derivative_absmean'] = pd.Series(np.abs(snip.heading_derivative.mean()), index=trajec.index)

    return df

def calc_heading_difference_sliding_window(df, sliding_window_size=15, start_time=900, window_time=1500):
    '''
    sliding_window_size in indices
    start_time and window_time in ms
    '''
    
    df['heading_difference_sliding'] = np.nan
    
    for objid in df.obj_id_unique.unique():
        trajec = df[df.obj_id_unique==objid]
        h = trajec.heading_unwrapped.values
        hd = np.hstack((np.nan*np.zeros(15), h[15:] - h[0:-15]))
        df.loc[trajec.index, 'heading_difference_sliding'] = pd.Series(hd, index=trajec.index)
        
        
    df['heading_difference_sliding_absmean'] = np.nan
    for objid in df.obj_id_unique.unique():
        trajec = df[df.obj_id_unique==objid]
        snip = trajec[(trajec['time stamp']>start_time)*(trajec['time stamp']<(start_time+window_time))]
        if len(snip)<10:
            continue
        df.loc[trajec.index, 'heading_difference_sliding_absmean'] = pd.Series(np.abs(snip.heading_difference_sliding.mean()), index=trajec.index)
        
    return df



# vertical area function 
def calc_vertical_hull_area(df, start_time=900, window_time=1500):
    df['vert_hull_area'] = np.nan
    df['vert_hull_roundness'] = np.nan
    
    for objid in df.obj_id_unique.unique():
        trajec = df[df.obj_id_unique==objid]
        snip = trajec[(trajec['time stamp']>start_time)*(trajec['time stamp']<(start_time+window_time))]
        if len(snip)<10:
            continue
        xy = np.sqrt(snip.x.values**2 + snip.y.values**2)
        hull = scipy.spatial.ConvexHull( np.vstack((snip.z.values, xy)).T )

        x = hull.points[:,0]
        xm = np.mean(x)
        y = hull.points[:,1]
        ym = np.mean(y)

        rmin = np.min( np.sqrt( (x-xm)**2 + (y-ym)**2 ) )
        #rmax = np.max( np.sqrt( (x-xm)**2 + (y-ym)**2 ) )
        rmax = np.sqrt(hull.area/np.pi)
        roundness = rmin/rmax
        
        df.loc[trajec.index, 'vert_hull_area'] =pd.Series( hull.area, index=trajec.index)
        df.loc[trajec.index, 'vert_hull_roundness'] = pd.Series(roundness, index=trajec.index)

    return df

# position at odor off

def calc_position_at_odor_off(df):
    df['x_odor_off'] = np.nan
    df['y_odor_off'] = np.nan
    df['z_odor_off'] = np.nan
    
    df['speed_off'] = np.nan
    df['speed_during'] = np.nan
    
    df['x_odor_during'] = np.nan
    df['y_odor_during'] = np.nan
    df['z_odor_during'] = np.nan
    
    for objid in df.obj_id_unique.unique():
        trajec = df[df.obj_id_unique==objid]
        snip = trajec[(trajec['time stamp']>700)*(trajec['time stamp']<900)]
        df.loc[trajec.index, 'x_odor_off'] =pd.Series( snip.x.mean(), index=trajec.index)
        df.loc[trajec.index, 'y_odor_off'] = pd.Series(snip.y.mean(), index=trajec.index)
        df.loc[trajec.index, 'z_odor_off'] = pd.Series(snip.z.mean(), index=trajec.index)
        df.loc[trajec.index, 'speed_off'] = pd.Series(snip['ground speed'].mean(), index=trajec.index)
        
        #snip2 = trajec[trajec['time stamp'].between(0,0)]
        snip1 = trajec[trajec['time stamp'].between(300,680)]
        df.loc[trajec.index, 'x_odor_during'] =pd.Series( snip1.x.mean(), index=trajec.index)
        df.loc[trajec.index, 'y_odor_during'] = pd.Series(snip1.y.mean(), index=trajec.index)
        df.loc[trajec.index, 'z_odor_during'] = pd.Series(snip1.z.mean(), index=trajec.index)
        df.loc[trajec.index, 'speed_during'] = pd.Series(snip1['ground speed'].mean(), index=trajec.index)

    return df

def calculate_features(dfs, start_time=900):

    for i, df in enumerate(dfs):
        print('df #', i)
        dfs[i]= df.groupby('obj_id_unique',).filter(lambda d: d['time stamp'].max()>=1700)
        dfs[i] = dfs[i][dfs[i]['time stamp'].between(start_time, 3000)]
        dfs[i]['ground speed'] = np.sqrt(dfs[i].xvel **2 + dfs[i].yvel **2)
        dfs[i] = cull_slow_flies(dfs[i])
        dfs[i] = calc_circlyness(dfs[i])
        dfs[i] = calc_heading_derivative(dfs[i])
        dfs[i] = calc_heading_difference_sliding_window(dfs[i])
        dfs[i] = calc_hull_area(dfs[i])
        dfs[i] = calc_heading_derivative_mean(dfs[i])

        print('n trajs:', len(df.obj_id_unique.unique()))
    return dfs


###function that calculates the heading at any given time, and the angular velocity for a trajectory
def get_angular_velocity(df, time_step):
        d_head_dt_vec =[np.nan]
        df['heading']=np.arctan2(df['yvel'], df['xvel'])
        for i in range(len(df)):
            if i !=0:
                vec_1 =[df['xvel'].iloc[i-1], df['yvel'].iloc[i-1]]
                vec_2 = [df['xvel'].iloc[i], df['yvel'].iloc[i]]
                norm_1 = vec_1/np.linalg.norm(vec_1)
                norm_2 =vec_2/np.linalg.norm(vec_2)
                dot_product = np.dot(norm_1, norm_2)
                d_angle_dt = np.arccos(dot_product)/time_step
                d_head_dt_vec.append(d_angle_dt)
        df['ang vel']= d_head_dt_vec
        return df

###Function that applies it to an entire data set
def get_angular_full_dataset(data_set, time_step):
    df_obj_vec =[]
    for i in data_set['obj_id_unique'].unique():
        d=data_set[data_set['obj_id_unique']==i]
        d_ = get_angular_velocity(d, time_step)
        df_obj_vec.append(d_)
    fdf = pd.concat(df_obj_vec)
    return fdf


def get_normed_pos_groupby(df):
    df = df.groupby('obj_id_unique', as_index=False).apply(lambda df: df.assign(normed_x_680 = lambda d: d.x - d[d['time stamp']==680].x.iloc[0] ))
    df = df.groupby('obj_id_unique', as_index=False).apply(lambda df: df.assign(normed_y_680 = lambda d: d.y - d[d['time stamp']==680].y.iloc[0] ))
    df = df.groupby('obj_id_unique', as_index=False).apply(lambda df: df.assign(normed_z_680 = lambda d: d.z - d[d['time stamp']==680].z.iloc[0] ))
    return df

def get_normed_positions(df, time_stamp=680):

    D = []

    for i in df.obj_id_unique.unique():
        dum = df[df.obj_id_unique==i].copy()
        if dum['time stamp'].max() < 1680:
            continue
       # if len(dum)<75:
       #     continue    
        else: 
            base_y= np.array(dum.y[dum['time stamp']==time_stamp])[0]
            base_x= np.array(dum.x[dum['time stamp']==time_stamp])[0]
            base_z= np.array(dum.z[dum['time stamp']==time_stamp])[0]

            dum.loc[:,'normed_y_'+ str(time_stamp)]= dum.y.copy() - base_y
            dum.loc[:,'normed_x_'+ str(time_stamp)]= dum.x.copy() - base_x
            dum.loc[:,'normed_z_'+ str(time_stamp)]= dum.z.copy() - base_z

            D.append(dum)
        
    newdf = pd.concat(D)

    return newdf


#helper function for slicing means with 95% CI's
def slice_means_gs(df, index):
    time_vec=[]
    mean_vec=[]
    upper_vec =[]
    lower_vec =[]
    for i in np.sort(df['time stamp'].unique()):
        if i >=-500 and i <3000:
            dummy_df =df[df['time stamp']==i]
            dummy_df=dummy_df[dummy_df['ground speed'].between(0, 2.0)]
            time_vec.append(i)
            mean = dummy_df[index].mean()
            upper= dummy_df[index].mean() + dummy_df[index].sem()*1.96
            lower= dummy_df[index].mean() - dummy_df[index].sem()*1.96
            mean_vec.append(mean)
            upper_vec.append(upper)
            lower_vec.append(lower)
    return time_vec, mean_vec, upper_vec, lower_vec  
