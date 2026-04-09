
import sys
sys.path.append('/home/flybot/.local/lib/python2.7/site-packages/')


import pandas as pd 
import numpy as np 
import argparse
import h5py
import os
import pynumdiff
import cvxpy
from braid_analysis import braid_filemanager
from braid_analysis import braid_slicing
from braid_analysis import braid_analysis_plots
import zipfile
import pickle
import scipy
import matplotlib.pyplot as plt
import figurefirst as fifi


def get_pandas_dataframe_from_uncooperative_hdf5(filename, key='first_key'):
    f = h5py.File(filename,'r')
    all_keys = list(f.keys())
    if key == 'first_key':
#         print('Dataset contains these keys: ')
        print(all_keys)
        key = all_keys[0]
#         print('Using only the first key: ', key)
    data = f[key][()]
    dic = {}
    for column_label in data.dtype.fields.keys():
        dic.setdefault(column_label, data[column_label])
    df = pd.DataFrame(dic)
    df.rename(columns={'data_1':'obj_id', 'data_2':'frame', 'data_4':'duration' }, inplace=True)
    df['obj_id'] =df['obj_id'].astype(int) 
    df.drop(columns=['data_0','data_3','t_secs', 't_nsecs','t'], inplace=True)
    return df



def get_braid_file(braid_handle, trigger_df):
    braid_df = braid_filemanager.load_filename_as_dataframe_3d(braid_handle)
    braid_df['millis']=(braid_df.timestamp-braid_df.timestamp.iloc[0])*1000 
    braid_df.drop(columns=['P00','P01','P02','P11','P12','P22','P33','P44','P55'], inplace=True)
    return braid_df #, braid_df_noevent


def assign_unique_id(braid_df, handle_):
	"""This Function adds a unique ID column to the braid data frame"""
	braid_df['obj_id_unique']=braid_df['obj_id'].apply(lambda x: str(x)+ '_' + handle_)
	return braid_df



def identify_duplicate_ids(trigger_df):
    trigger_df['duplicate_obj_id_bool'] = trigger_df.duplicated(subset='obj_id', keep=False)
    return trigger_df

def merge_braid_with_triggers(braid_df, trigger_df):
    braid_df.sort_values(by=['obj_id','frame'], ascending=True, inplace=True)
    braid_df.reset_index(inplace=True, drop=True)
    braid_df = braid_df.merge(trigger_df, how='outer', on=['frame', 'obj_id'])
    return braid_df


def get_last_flash_info(braid_df, trigger_df):
    braid_df['Trigger_bool'] = braid_df['frame'].isin(trigger_df['frame'].to_list())
    braid_df['Flash_bool'] = braid_df['duration'].isin([100])
    smalldf = braid_df[braid_df['Flash_bool']==True].copy()
    smalldf['last_flash_event_start_millis'] = smalldf['millis'].shift(1)
    braid_df = braid_df.combine_first(smalldf)
    braid_df.dropna(subset='x', inplace=True)
    braid_df.sort_values(by=['frame'], ascending=True, inplace=True)
    braid_df['last_flash_event_start_millis']=braid_df['last_flash_event_start_millis'].ffill()
    braid_df['time_since_last_event_millis'] = braid_df['millis']-( braid_df['last_flash_event_start_millis'] + 680) 
    
    #add duration to all frames
    small2 = braid_df[(braid_df.obj_id.isin(trigger_df.obj_id))]
    small2.duration=small2.groupby(['obj_id'])['duration'].ffill()
    small2.duration=small2.groupby(['obj_id'])['duration'].bfill()
    braid_df = braid_df.combine_first(small2)
    braid_df.duration = braid_df.duration.fillna(0)
    
    return braid_df



def time_stamp(braid_df):
    """Create empty list to catch all of the processed individual trajectories"""
    df_catcher=[]
    """Loop through all of the trajectories one by one"""
    for i in braid_df['obj_id'].unique():
        d_ = braid_df[braid_df['obj_id']==i]

        try:
            """find the point in the trajectory where the trigger event happened"""
            ind = np.where(d_['Flash_bool']==True)[0][0]
            #print("can i print this")
            pre_df=d_.iloc[0:ind]
            post_df=d_.iloc[ind:]
            if d_['xvel'].iloc[ind]>0:
                d_['orientation']=['u']*len(d_)
            else:
                d_['orientation']=['d']*len(d_)
            pre_len = len(pre_df)
            #print(' pre length is '+str(pre_len))
            post_len = len(post_df)
            d_['time stamp']=np.linspace(-10*(pre_len), 10*(post_len-1), len(d_))
            df_catcher.append(d_)
        
        except:
            if d_['time_since_last_event_millis'].min()<10000:
                continue
            else:
                ind = 50
                pre_df=d_.iloc[0:ind]
                post_df=d_.iloc[ind:]
                if d_['xvel'].iloc[ind]>0:
                    d_['orientation']=['u']*len(d_)
                else:
                    d_['orientation']=['d']*len(d_)
                pre_len = len(pre_df)
                #print(' pre length is '+str(pre_len))
                post_len = len(post_df)
                d_['time stamp']=np.linspace(-10*(pre_len), 10*(post_len-1), len(d_))
                #print(len(d_))
                df_catcher.append(d_)
            
    stamped_df=pd.concat(df_catcher)
    return stamped_df

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


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

def unwrap_angle(z, correction_window_for_2pi=100, n_range=2, plot=False):
    if 0: # option one
        zs = []
        for n in range(-1*n_range, n_range):
            zs.append(z+n*np.pi*2)
        zs = np.vstack(zs)

        smooth_zs = np.array(z[0:2])

        for i in range(2, len(z)):
            first_ix = np.max([0, i-correction_window_for_2pi])
            last_ix = i
            error = np.abs(zs[:,i] - np.mean(smooth_zs[first_ix:last_ix])) 
            smooth_zs = np.hstack(( smooth_zs, [zs[:,i][np.argmin(error)]] ))

        if plot:
            for r in range(zs.shape[0]):
                plt.plot(zs[r,:], '.', markersize=1)
            plt.plot(smooth_zs, '.', color='black', markersize=1)
        
    else: # option two, automatically scales n_range to most recent value, and maybe faster
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



#Smoothing function for mean angular velocity and ground speed plots
def filter_testing(traj, head_key='heading', filter_params =[1, 2, 8], dt=.01 ):
    head_vec = traj[head_key].to_numpy()
    head_vec = np.unwrap(head_vec)
    traj=traj[traj['time stamp'].between(-500, 10000)]
    traj['head unwrap']=head_vec
    traj['raw angvel']=traj['head unwrap'].diff()/dt
    traj['raw angvel']=traj['raw angvel']
    raw_angvel= np.abs(np.diff(head_vec)/dt)
    smootha, da =pynumdiff.linear_model.savgoldiff(head_vec, dt, filter_params)
    smooth_xvel, _ =pynumdiff.linear_model.savgoldiff(traj.xvel.to_numpy(), dt, filter_params)
    smooth_yvel, _ =pynumdiff.linear_model.savgoldiff(traj.yvel.to_numpy(), dt, filter_params)
    np.insert(da, [0], np.nan)
    filter_param_tag =""
    for i in filter_params:
        filter_param_tag+=str(i)   
    traj['xvel']=smooth_xvel
    traj['yvel']=smooth_yvel
    traj['heading']=wrap_angle(smootha)  
    traj['smoothed_angvel_params_'+filter_param_tag]=da
    return traj


#Applies smoothing to every unique trajectory in the data set
def filter_angvel_whole_data_set(df, filter_params=[1, 2, 8], dt=.01, id_key='obj_id_unique', head_key='heading'):
    collector =[]
    for i in df[id_key].unique():
        d =df[df[id_key]==i]
        d=d[d['time stamp'].between(-500, 10000)]
        d= filter_testing(d, head_key=head_key, filter_params=filter_params, dt=dt)
        collector.append(d)
    fin=pd.concat(collector)
    return fin

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def do_filtering(df1):
    long_obj_ids1 = braid_slicing.get_long_obj_ids_fast_pandas(df1, length=150)
    df1= df1[df1.obj_id.isin(long_obj_ids1)]
    far1 = braid_slicing.get_trajectories_that_travel_far(df1, xdist_travelled=0.1)
    df1 =df1[df1.obj_id.isin(far1)]
    return df1

def filter_bad_trajectories(df):
    df_catcher = []
    badones = []
    for trajectory_id in df['obj_id'].unique():
        # Subslice the larger dataset to just the parts that are that trajectory
        d = df[df['obj_id'] == trajectory_id]
        d = d[(d['time stamp'] >= -500) & (d['time stamp'] <= 10000)]
        d_trigger = d[(d['time stamp'] >= 0) & (d['time stamp'] <= 675)]
        try:
            if d.time_since_last_event_millis.iloc[0]/1000 <10:
                print('trigger too recent')
                continue
        except:
            pass
        if np.mean(d_trigger['z']) > 0.5 or np.mean(d_trigger['z']) < 0:
            print('out of z range')
            continue
        if np.max(d_trigger['z']) > 0.5 or np.min(d_trigger['z']) < -0.1:
            print('out of z range, %s' % d.obj_id.unique())
            badones.append(d)
            continue
        if (d['time stamp'].max() < 1680):
            print('traj < 1680ms, %s' % d.obj_id.unique())
            badones.append(d)
            continue
        if d['xvel'].max() >= 2 or d['yvel'].max() >= 2 or d['zvel'].max() >= 2: 
            print('invalid ground speed, %s' % d.obj_id.unique())
            badones.append(d)
            continue
        if (scipy.spatial.distance.cdist(np.atleast_2d(d.x).T,np.atleast_2d(d.x).T).max()) < 0.02: 
            print('didnt go far in x, %s' % d.obj_id.unique())
            badones.append(d)
            continue 
        if (scipy.spatial.distance.cdist(np.atleast_2d(d.y).T,np.atleast_2d(d.y).T).max()) < 0.02: 
            print('didnt go far in y, %s' % d.obj_id.unique())
            badones.append(d)
            continue         
        if (scipy.spatial.distance.cdist(np.atleast_2d(d.z).T,np.atleast_2d(d.z).T).max()) < 0.02: 
            print('didnt go far in z, %s' % d.obj_id.unique())
            badones.append(d)
            continue        
            
        else:    
            df_catcher.append(d)
            print('good traj:', d.obj_id.unique())

    return pd.concat(df_catcher).sort_values(by='time stamp')#, pd.concat(badones).sort_values(by='time stamp')

def run_all_processing(braid_handle, trigger_handle):
    trigger_df = get_pandas_dataframe_from_uncooperative_hdf5(trigger_handle)
    braid_df = get_braid_file(braid_handle, trigger_df)
    trigger_df= identify_duplicate_ids(trigger_df)
    braid_df = merge_braid_with_triggers(braid_df, trigger_df)
    braid_df = get_last_flash_info(braid_df, trigger_df)
    braid_df = do_filtering(braid_df)
    filtered_df = time_stamp(braid_df)
    filtered_df, __ = filter_bad_trajectories(filtered_df)
    uniq_id_prefix= braid_handle.split('/')[-1].split('.')[0] 
    filtered_df = assign_unique_id(filtered_df, uniq_id_prefix)
    filtered_df= get_angular_full_dataset(filtered_df, .01)
    filtered_df=filter_angvel_whole_data_set(filtered_df)
    filtered_df = get_normed_positions(filtered_df)
    return filtered_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('braid_filename', type=str, help="full path to the braid file of interest")
    parser.add_argument('trigger_filename', type=str, help= 'full path the the trigger bag hdf5')
    args = parser.parse_args()
    if not os.path.exists(args.braid_filename):
        print('No file %s' % args.braid_filename, file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.trigger_filename):
        print('No file %s' % args.trigger_filename, file=sys.stderr)
        sys.exit(1)
    
    braid_handle = str(args.braid_filename)
    trigger_handle = str(args.trigger_filename)
    

    run_all_processing(braid_handle, trigger_handle)