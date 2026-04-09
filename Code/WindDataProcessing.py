import pandas as pd 
import numpy as np


from scipy.spatial import KDTree

import os

import scipy
from scipy.interpolate import griddata
from scipy import signal
from scipy import fft
#from scipy.stats import circstd
import math

from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
                   arange, sort, amin, amax, atleast_1d, sqrt, array,
                   compress, pi, exp, ravel, count_nonzero, sin, cos,
                   arctan2, hypot)


########################################################################################################################

def organize_directory(n):
    return int(n.split('.')[0].split('__')[1])

def load_wind_data(directory, filetype,start_id=None, level=None, datatype='CFD', coord_system='fly_experiments'):
    data =[]
    if datatype=='real': #for data collected with anemometer
        for i in os.listdir(directory):
            if i.startswith(start_id):
                if i.endswith(level + '.' + filetype):
                    print (i)
                    if filetype=='csv':
                        data.append(pd.read_csv(directory + '/' + i))
                    elif filetype=='hdf':
                        data.append(pd.read_hdf(directory + '/' + i))
    if datatype=='CFD':
        if coord_system == 'fly_experiments':
            for i in sorted(os.listdir(directory), key=organize_directory):
                    if i.endswith(filetype):
                        print(i)
                        # CSV = Raw files from CFD experiments, renaming columns and changing coord frame to match fly data
                        if filetype=='csv': 
                            df = pd.read_csv(directory + '/' + i, dtype=np.float32)
                            df.rename(columns={"Points:0":'x',"Points:1":'z',"Points:2":'y', "U:0":'xvel', "U:1":'zvel', "U:2":'yvel'}, inplace=True)

                            file_name = i.split('.')[0]
                            data =  df[(df['x'].between(.29,1.19)) & (df['y'].between(0.001,0.4825)) & (df['z'].between(0.001,0.4825))].copy()

                            data['x'] =  data['x'] - .74
                            data['y'] = data['y'] - 0.2418

                            data.xvel = -1*data.xvel
                            data= data.astype(np.float32,copy=False)
                            data.index=data.index.astype(int, copy=False)
                            data.to_parquet(directory + '/' + file_name + '.parquet')
                    #the parquet files are the preprocessed output from above^^       
                    if filetype=='parquet':
                        df = pd.read_parquet(directory + '/' + i, columns=['x','y','z','xvel','yvel','zvel'])
                        data.append(df) 

        elif coord_system == 'CFD': 
            for i in sorted(os.listdir(directory), key=organize_directory):
                if i.endswith(filetype):
                    print(i)
                    if filetype=='csv':
                        df = pd.read_csv(directory + '/' + i, dtype=np.float32) 
                        df.rename(columns={"Points:0":'x',"Points:1":'z',"Points:2":'y', "U:0":'xvel', "U:1":'zvel', "U:2":'yvel', 'vorticity:0':'vorticity_x', 'vorticity:1':'vorticity_z', 'vorticity:2':'vorticity_y'}, inplace=True)
                        df['unit_xvel'] = df.xvel/np.linalg.norm(df[['xvel','yvel','zvel']], axis=1)
                        df['unit_yvel'] = df.yvel/np.linalg.norm(df[['xvel','yvel','zvel']], axis=1)
                        df['unit_zvel'] = df.zvel/np.linalg.norm(df[['xvel','yvel','zvel']], axis=1)
                        file_name = i.split('.')[0]
                        df= df.astype(np.float32,copy=False)
                        df.index=df.index.astype(int, copy=False)
                        data.append(df) 
            data = pd.concat(data)

    return data       





########################################################################################################################

def get_fft(data,time):
    # Number of sample points
    N = len(time)
    # sample spacing
    time=np.array(time)
    T = 1.0 /(len(time)/(time[-1]-time[0]))
    # print(T)
    x = np.linspace(0.0, N*T, N)
    y = np.array(data)
    yf = scipy.fft.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
#     p=np.multiply(yf,time[-1])
    X=np.log10(xf[1:])
    Y=np.log10(2.0/N * np.abs(yf[1:(N//2)]))
#     P=np.log10(2.0/N * np.abs(p[1:(N//2)]))
    return(X,Y)
########################################################################################################################
def FFT(h, dt=.1):
    """
    basic spectral estimation
    Returns frequencies, power spectrum, and
    power spectral density.
    Only positive frequencies between (and not including)
    zero and the Nyquist are output.
    """
    nt = len(h)
    npositive = nt//2
    pslice = slice(1, npositive)
    freqs = np.fft.fftfreq(nt, d=dt)[pslice] 
    ft = np.fft.fft(h)[pslice]
    psraw = np.abs(ft) ** 2
    # Double to account for the energy in the negative frequencies.
    psraw *= 2
    # Normalization for Power Spectrum
    psraw /= nt**2
    # Convert PS to Power Spectral Density
    psdraw = psraw * dt * nt  # nt * dt is record length
    return freqs, psraw, psdraw

########################################################################################################################
def _circfuncs_common(samples, high, low, nan_policy='propagate'):
    # Ensure samples are array-like and size is not zero
    samples = np.asarray(samples)
    if samples.size == 0:
        return np.nan, np.asarray(np.nan), np.asarray(np.nan), None

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = sin((samples - low)*2.*pi / (high - low))
    cos_samp = cos((samples - low)*2.*pi / (high - low))
    #sin_samp=sin(samples)
    #cos_samp=cos(samples)
    mask = None
    return samples, sin_samp, cos_samp, mask

def circstd(samples, high=360, low=0, axis=None, nan_policy='omit'):

    samples, sin_samp, cos_samp, mask = _circfuncs_common(samples, high, low,
                                                          nan_policy=nan_policy)
    if mask is None:
        sin_mean = sin_samp.mean(axis=axis)  # [1] (2.2.3)
        cos_mean = cos_samp.mean(axis=axis)  # [1] (2.2.3)
    else:
        nsum = np.asarray(np.sum(~mask, axis=axis).astype(float))
        nsum[nsum == 0] = np.nan
        sin_mean = sin_samp.sum(axis=axis) / nsum
        cos_mean = cos_samp.sum(axis=axis) / nsum
    # hypot can go slightly above 1 due to rounding errors
    with np.errstate(invalid='ignore'):
        R = np.minimum(1, hypot(sin_mean, cos_mean))  # [1] (2.2.4)
    res = np.sqrt(2 * (1 -R))
    
    return res*180/np.pi

