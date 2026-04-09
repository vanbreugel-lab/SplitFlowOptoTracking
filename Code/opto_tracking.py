from __future__ import division, print_function
import numpy as np


import numpy as np
import random
from random import choice
from pprint import pprint
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ks_2samp, ttest_rel
from scipy.signal import resample
from sklearn import linear_model
from scipy.spatial import KDTree
import time
from splitflow.kinematics import heading as calc_heading
from splitflow.kinematics import norm
import pynumdiff

import pandas as pd

########################################################################################################################
class BoolPlume(object):
    '''simulate an odor experience the same way as our optogenetic experiments:

    if a fly enters the predetermined trigger bounds it will receive a flash for a specified duration

    bounds: [xmin, xmax], [ymin, ymax], [zmin, zmax]
    lastflash: when the flash turned off
    refractory: minimum time before agent can trigger flash again
    duration: length of flash'''
        
        
    def __init__(self, bounds, lastflash=0, refractory=3, duration=0.675):

        self.trigger_bounds = bounds
        #self.flashintensity = intensity
        self.triggerduration = duration
        self.lastflashtime = lastflash
        self.refractory = refractory
    
    def check_bounds(self, x):
        temp_bool=[0,0,0]
        for i in range(3):
            #print('position',x[i])
            #print('min:',self.trigger_bounds[i][0])
            #print('max:',self.trigger_bounds[i][1])
            if (x[i] >= self.trigger_bounds[i][0]) & (x[i] <= self.trigger_bounds[i][1]):
                temp_bool[i]=1
      #  print('temp_bool:',temp_bool)
        if temp_bool == [1,1,1]:
            return 1
        else:
            return 0
        
        
    def get_odor(self, condition, t):
        #print('current t:', t)
        if t - self.refractory <= self.lastflashtime:
            #print('too soon to trigger: current time is', t, 'last flash is', self.lastflashtime)
            return 0 
        
        if t - self.refractory >= self.lastflashtime: 
            if condition == 'initial':
                self.start_time = t
                #print('last plume started at', self.start_time)
            if t - self.start_time <= self.triggerduration:
                #print('odor on')
                return 1

            elif t - self.start_time > self.triggerduration:  
                #print('duration ended')
                self.lastflashtime= t
                return 0

            

    # for each time step, check if initial odor encounter time < duration.        
########################################################################################################################
class SurgingAgent(object):
    """
    centerline:
    Simple tracking agent that tracks a plume based on inferring its centerline
    and is biased both toward the centerline and upwind, with the strength of
    the upwind bias increasing with the certainty around the centerline location.

    The equation for the agent's 3D motion is:

    $\tau \dot{v} = -v + \eta + b$

    where v is the velocity, \tau is a time constant, \eta is i.i.d. 3D Gaussian
    noise with identity covariance, and b is the bias term.
    
    
    Tracking agent model with same base dynamics as Centerline-inferring model
    except that plume-crossing triggers a brief upwind surge, and there is
    no memory across multiple crossings.
    """

    def __init__(self, seed, tau, noise, bias,
            surge_amp, tau_surge, bounds):

        self.tau = tau
        self.noise = noise
        self.bias = bias
        self.seed=seed

        self.odor= False
        self.surge_amp = surge_amp
        self.tau_surge = tau_surge

        self.surge_amp_ = surge_amp / (tau_surge * np.exp(-1))

        self.bounds = bounds

    def reflect_if_out_of_bounds(self, v, x):
        """
        Check if a position is within the bounds
        :param v: current velocity
        :param x: current position
        :return: v, x corrected if x was out of bounds
        """

        if self.bounds is None:

            return v, x

        else:

            v_new = v.copy()
            x_new = x.copy()

            for dim in range(3):

                if x[dim] < self.bounds[dim][0]:

                    v_new[dim] *= -1
                    x_new[dim] = 2 * self.bounds[dim][0] - x[dim]

                elif x[dim] > self.bounds[dim][1]:

                    v_new[dim] *= -1
                    x_new[dim] = 2 * self.bounds[dim][1] - x[dim]

            return v_new, x_new


    def track(self, behavior, plume, wind, start_pos, start_vel, duration, dt, bias_sign,reject_thresh,constant_noise=True, radius=None, omega=None,):
   
    #    ag.track(plume=pl, wind=wind1, behavior=BEHAVIOR, start_pos=start_pos, start_vel=start_vel, duration=DURATION, dt=DT, reject_thresh=REJECT_THRESH, constant_noise=CONSTANT_NOISE,bias_sign=np.random.choice([-1,1]), )
        
        if behavior=='cast_and_surge':
                return_dict = self.cast_and_surge(plume=plume, wind=wind, start_pos=start_pos, start_vel=start_vel, duration=duration, dt=dt, bias_sign=bias_sign,constant_noise=constant_noise, rv= np.random.normal(2,0.35))  #np.random.normal(2,0.5))

        elif behavior=='sink_and_circle':
            return_dict = self.sink_and_circle(plume, wind, start_pos, start_vel, duration, dt, bias_sign, reject_thresh, constant_noise, radius, omega,rv=np.random.normal(2,1)) #
            
# two alternative behaviors I looked at but that are excluded from paper/analysis...            
        elif behavior=='saccade_circle':
            return_dict = self.sink_and_circle_saccade(plume, wind, start_pos, start_vel, duration, dt, bias_sign, reject_thresh, rv=np.random.normal(2.75,0.25))
        elif behavior=='surge_and_circle':
                return_dict = self.surge_and_circle(plume, wind, start_pos, start_vel, duration, dt, bias_sign, reject_thresh, rv= np.random.normal(2,1))  #np.random.normal(2,0.5))

        return return_dict 

########################################################################################################################    
    def cast_and_surge(self, plume, wind, start_pos, start_vel, duration, dt, bias_sign,constant_noise,rv,):
        """
        Track a plume using the basic algorithm.

        :param plume: plume object with get_odor method
        :param threshold: odor threshold for detection
        :param start_pos: starting position
        :param duration: duration of simulation
        :param dt: numerical integration time step
        :return:
        """
        
        n_steps = int(duration / dt)
        ts = np.arange(n_steps) * dt
        self.time = ts
        bs = np.nan * np.zeros((n_steps, 3))
        surges = np.zeros(n_steps)
        casting = np.zeros(n_steps)
        b_convo = np.zeros(n_steps)
        vs = np.nan * np.zeros((n_steps, 3))
        xs = np.nan * np.zeros((n_steps, 3))
        odors = np.nan * np.zeros((n_steps,))
        upwind = np.zeros((n_steps, 3))
        crosswind = np.zeros((n_steps, 3))
        real_wind = np.zeros((n_steps, 3))
        surge_vec = np.zeros((n_steps, 3))
        noise_convo = np.zeros(n_steps)


        
        for t in range(n_steps):
           # print(t)
            if t == 0:
                b = np.array([0, start_pos[1], start_pos[2]])
                b *= (self.bias/ np.linalg.norm(b))
                v = start_vel.copy()
                x = start_pos.copy()
                d_upwind = np.array([-.4,0,0])
                new_noise= self.noise
            else:

                # get driving terms
                eta = np.random.normal(0, new_noise, (3,))

                d_upwind = (upwind[t] - upwind[t-1])  / np.linalg.norm(upwind[t] - upwind[t-1]) 
                d_cross = (crosswind[t]-crosswind[t-1]) / np.linalg.norm(crosswind[t]-crosswind[t-1])
                
                b=self.bias*b_convo[t]*d_cross * bias_sign
                
                # before odor, no casting or upwind bias 
                if np.all(odors[:t]==0):
                    b=0
                    new_noise=self.noise
                    eta = np.random.normal(0, new_noise, (3,))

                
                #during surge, noise is reduced (if not constant):
                if self.odor==1: 
                    b=0    
                    new_noise = self.noise*2/3
                    eta = np.random.normal(0, new_noise, (3,))
                
                #return to baseline behavior after 2 seconds
                if (len(np.where(odors==1)[0]) != 0) and (np.abs(ts[t] - np.where(odors==1)[0][-1]*dt) > 2.5):
                    b=0
                    
                if constant_noise==True: 
                    new_noise=self.noise
                    eta = np.random.normal(0, new_noise, (3,))
              
                # update velocity and position
                v += (dt / self.tau) * (-v + eta + b + surges[t]*d_upwind)  
                x += v * dt

                    
            #check if agent is in bounds, make it change direction if hitting bound
            v, x = self.reflect_if_out_of_bounds(v, x)
            
            
            #check if eligible to get odor
            if self.odor == False:
                #check if agent moved into trigger zone
                inTriggerZone = plume.check_bounds(x)
                if inTriggerZone:
                    #if agent in triggerzone, check if it can get odor based on last trigger time
                    self.odor = plume.get_odor('initial', ts[t])
            
            #save values to dict
            bs[t] = b
            vs[t] = v
            xs[t] = x
            noise_convo[t]=new_noise
            
            #get the instantaneous wind from CFD or real data
            if wind.vector_field is not None:
                wind.get_correct_wind_vec(agent_position=xs[t], t=t, ts=ts[t]) 

            
            #calculate crosswind/upwind for fly
            crosswind[t], upwind[t], real_wind[t] = wind.relative_upwind(vs[t], t=t)
            #print('cross:', crosswind[t], 'up:', upwind[t])
            
            #save values to dict
            surge_vec[t] = surges[t]*d_upwind
            odors[t] = self.odor

        
            # add new surge force if hit has occurred
            if self.odor==True:
                c=30 #arbitrary surge start point based on empirical data
                ts_ = ts[t+c:] - ts[t+c]
                surges[t+c:] += (self.surge_amp_*ts_*np.exp(-ts_/self.tau_surge))
                #update if flash is still on
                b_convo[:t] = 0
                self.odor = plume.get_odor(self.odor, ts[t])


                
            if self.odor==False:
                b_convo[t:] = np.sign(np.sin(rv*np.pi*ts[t]))
                # set noise value if constant noise = False
                new_noise = self.noise*ts[t]*1/3 
                if new_noise > self.noise:
                    new_noise= self.noise
        
        

            
        return_dict = {
            'surges': surges,
            'surge vec': surge_vec,
            'bs': bs,
            'vs': vs,
            'xs': xs,
            'odors': odors,
            'ts': ts,
            'real_wind': real_wind,
            'casting':casting,
            'b_convo': b_convo,
            'noise_convo': noise_convo,
        }

        return return_dict
########################################################################################################################    
    def check_last_odor(self,odors,t):
        if (odors[t]==0) and (odors[t-1]==1):
            #print('odor ended')
            return 1
        else:
            return 0

########################################################################################################################

    def sink_and_circle(self, plume, wind, start_pos, start_vel, duration, dt, bias_sign, reject_thresh, constant_noise, radius, omega,rv):
        """
        Track a plume using the basic algorithm.

        :param plume: plume object with get_odor method
        :param threshold: odor threshold for detection
        :param start_pos: starting position
        :param duration: duration of simulation
        :param dt: numerical integration time step
        :return:
        """

        n_steps = int(duration / dt)
        ts = np.arange(n_steps) * dt

        bs = np.nan * np.zeros((n_steps, 3))
        surges = np.zeros(n_steps)
        casting = np.zeros(n_steps)
        b_convo = np.zeros(n_steps)
        vs = np.nan * np.zeros((n_steps, 3))
        xs = np.nan * np.zeros((n_steps, 3))
        odors = np.nan * np.zeros((n_steps,))
        upwind = np.zeros((n_steps, 3))
        crosswind = np.zeros((n_steps, 3))
        surge_vec = np.zeros((n_steps, 3))
        real_wind = np.zeros((n_steps, 3))

        r= np.random.normal(radius,0.05)
        w=np.random.normal(omega,1.5)
        bool_checks = np.zeros(n_steps)
        noise_convo = np.zeros(n_steps)


        for t in range(n_steps):
            
            if t == 0:
                b = np.array([0, -start_pos[1], -start_pos[2]])
                b *= (self.bias/ np.linalg.norm(b))
                v = start_vel.copy()
                x = start_pos.copy()
                new_noise= self.noise
            else:
                # get driving terms
                eta = np.random.normal(0, new_noise, (3,))
                
                d_cross = (crosswind[t] - crosswind[t-1]) / np.linalg.norm(crosswind[t] - crosswind[t-1])
                d_upwind = (upwind[t] - upwind[t-1]) 
                d_upwind = d_upwind / np.linalg.norm(d_upwind)
                x1= r*np.cos(w*ts[t])*bias_sign
                y1= r*np.sin(w*ts[t])*bias_sign 
                z=-d_upwind[2]*dt 
                b= np.array([x1,y1,0])
                # before odor, no casting or upwind bias 
                if np.all(odors[:t]==0):
                    new_noise=self.noise
                    eta = np.random.normal(0, new_noise, (3,))
                
                #during odor, noise is reduced
                if self.odor==1 and (np.sum(odors[:t])<=20):
                    b=0
                    new_noise = self.noise*2/3
                    eta = np.random.normal(0, new_noise, (3,))
                
                if np.all(bool_checks[:t]==0):
                #    b=0
                    d_upwind=0
                    
                if constant_noise==True: 
                    new_noise=self.noise
                    eta = np.random.normal(0, new_noise, (3,))
                    
                v += (dt / self.tau) * (-v + eta + self.bias*b - reject_thresh*d_upwind)
                #v += (dt / self.tau) * ( eta)
                
                x += v * dt

            
            v, x = self.reflect_if_out_of_bounds(v, x)

            
            if self.odor == False:
                #check if agent moved into trigger zone
                inTriggerZone = plume.check_bounds(x)
                if inTriggerZone:
                    #if agent in triggerzone, check if it can get odor
                    self.odor = plume.get_odor('initial', ts[t])
                       

            # store data for this time step
            bs[t] = b
            vs[t] = v
            xs[t] = x
            noise_convo[t]=new_noise


            if wind.vector_field is not None:
                wind.get_correct_wind_vec(agent_position=xs[t], t=t, ts=ts[t])

            crosswind[t], upwind[t], real_wind[t] = wind.relative_upwind(vs[t], t=t)

            odors[t] = self.odor
            
            bool_checks[t] = self.check_last_odor(odors,t)
            if bool_checks[t]==1:
                circle_center = xs[t]
                
            if self.odor==True:    
                self.odor = plume.get_odor(self.odor, ts[t])
            
            # if constant noise = False
            if np.sum(odors[:t])>20:    
                new_noise = self.noise*ts[t]*2/3
                if new_noise > self.noise:
                    new_noise= self.noise      
                
        return_dict = {
            'surges': surges,
            'surge vec': surge_vec,
            'bs': bs,
            'vs': vs,
            'xs': xs,
            'odors': odors,
            'ts': ts,
            'real_wind': real_wind, 
            'upwind': upwind,
            'crosswind': crosswind,
            'casting':casting,
            'b_convo': b_convo,
            'noise_convo': noise_convo,
        }

        return return_dict
   
        
    
########################################################################################################################
    def sink_and_circle_saccade(self, plume, wind, start_pos, start_vel, duration, dt, bias_sign, reject_thresh, rv):
        """
        Track a plume using the basic algorithm.

        :param plume: plume object with get_odor method
        :param threshold: odor threshold for detection
        :param start_pos: starting position
        :param duration: duration of simulation
        :param dt: numerical integration time step
        :return:
        """

        n_steps = int(duration / dt)
        ts = np.arange(n_steps) * dt

        bs = np.nan * np.zeros((n_steps, 3))
        surges = np.zeros(n_steps)
        casting = np.zeros(n_steps)
        b_convo = np.zeros(n_steps)
        vs = np.nan * np.zeros((n_steps, 3))
        xs = np.nan * np.zeros((n_steps, 3))
        odors = np.nan * np.zeros((n_steps,))
        upwind = np.zeros((n_steps, 3))
        crosswind = np.zeros((n_steps, 3))
        surge_vec = np.zeros((n_steps, 3))
        h=0
        k=0
        r= np.random.normal(.1,0.015)
        w=np.random.normal(6,1.5)
        bool_checks = np.zeros(n_steps)
        noise_convo = np.zeros(n_steps)
        #wind_start_time =np.round(np.random.uniform(0,6.8),1)
        extra_noise= np.random.uniform(-0.05,0.05,1)
       # n1 = np.random.normal(0.1,0.01)*np.random.choice([-1,1])
   
 
        # Calculate the corresponding y values using the stair step function
        phase = np.floor(ts*rv) *  np.random.normal(np.pi/2,0.3) # + extra_noise
        n1 = np.random.normal(0.05,0.01)*np.random.choice([-1,1])
        
        for t in range(n_steps):
            
            if t == 0:
                b = np.array([0, -start_pos[1], -start_pos[2]])
                #b = np.array([-start_pos[0], -start_pos[1], -start_pos[2]])/dt
                b *= (self.bias/ np.linalg.norm(b))
                v = start_vel.copy()
                #v = np.array([.30,.20,-0.1]).astype('float64')
                x = start_pos.copy()
                new_noise= self.noise
            else:
                # get driving terms
                #eta = np.random.normal(0, self.noise, (3,))
                eta = np.random.normal(0, new_noise, (3,))
                
                #d_cross = (crosswind[t] - crosswind[t-1]) / np.linalg.norm(crosswind[t] - crosswind[t-1])

                
                d_upwind = (upwind[t] - upwind[t-1]) 
                d_upwind = d_upwind / np.linalg.norm(d_upwind)
                
                
                #if h==0:
                #    w=5

                x1 = r*np.sin(phase[t])*bias_sign #+ np.random.uniform(-0.05,0.05,len(t))
                y1 = r*np.cos(phase[t]) *bias_sign
                
                #x1= r*np.sin(w*ts[t]+phase[t])*bias_sign
                #y1= r*np.cos(w*ts[t]+phase[t])*bias_sign
                z=-d_upwind[2]*dt
                b= np.array([x1,y1,z])
                # before odor, no casting or upwind bias 
                if np.all(odors[:t]==0):
                    new_noise=self.noise
                    eta = np.random.normal(0, new_noise, (3,))
                    #print('no odor yet, noise =', new_noise)
                
                #during odor, noise is reduced
                if self.odor==1 and (np.sum(odors[:t])<=20):
                    b=0
                    #new_noise = np.sqrt(self.noise)
                    #eta = np.random.normal(0, new_noise, (3,))
                
                if np.all(bool_checks[:t]==0):
                #    b=0
                    d_upwind=0
                #eta = np.random.normal(0, self.noise, (3,))   
                v += (dt / self.tau) * (-v +eta + self.bias*b - reject_thresh*d_upwind)
                #v += (dt / self.tau) * ( eta)
                x += v * dt

                #if h !=0 :    
 
            
            v, x = self.reflect_if_out_of_bounds(v, x)

            
            if self.odor == False:
                #check if agent moved into trigger zone
                inTriggerZone = plume.check_bounds(x)
                if inTriggerZone:
                    #if agent in triggerzone, check if it can get odor
                    self.odor = plume.get_odor('initial', ts[t])
                       

            # store data for this time step
            bs[t] = b
            vs[t] = v
            xs[t] = x
            noise_convo[t]=new_noise

            #start_time = time.time()
            if wind.vector_field is not None:
                wind.get_correct_wind_vec(agent_position=xs[t], t=t, ts=ts[t])
                #print('out_array:', self.wind_vector)
            #end_time = time.time()
            #print('time to get correct wind main loop:', start_time - end_time)
            
            
            #start_time = time.time()
            crosswind[t], upwind[t] = wind.relative_upwind(vs[t], t=t)
            
            #end_time = time.time()
            #print('total time to get cross + up:', start_time - end_time)
            
            
            #print('calculated rel up')
            odors[t] = self.odor
            
            
            bool_checks[t] = self.check_last_odor(odors,t)
                                                           
            if self.odor==True:    
                #print('updated odor:', self.odor)
                self.odor = plume.get_odor(self.odor, ts[t])
            
            #if self.odor==False:
            if np.sum(odors[:t])>20:    
                new_noise = self.noise*ts[t]# +.25
                if new_noise > self.noise:
                    new_noise= self.noise      
                
        return_dict = {
            'surges': surges,
            'surge vec': surge_vec,
            'bs': bs,
            'vs': vs,
            'xs': xs,
            'odors': odors,
            'ts': ts,
            'upwind': upwind,
            'crosswind': crosswind,
            'casting':casting,
            'b_convo': b_convo,
            'noise_convo': noise_convo,
        }

        return return_dict

########################################################################################################################
    def surge_and_circle(self, plume, wind, start_pos, start_vel, duration, dt, bias_sign, reject_thresh, rv):
        """
        Track a plume using the basic algorithm.

        :param plume: plume object with get_odor method
        :param threshold: odor threshold for detection
        :param start_pos: starting position
        :param duration: duration of simulation
        :param dt: numerical integration time step
        :return:
        """

        n_steps = int(duration / dt)
        ts = np.arange(n_steps) * dt
        self.time = ts
        bs = np.nan * np.zeros((n_steps, 3))
        surges = np.zeros(n_steps)
        casting = np.zeros(n_steps)
        b_convo = np.zeros(n_steps)
        vs = np.nan * np.zeros((n_steps, 3))
        xs = np.nan * np.zeros((n_steps, 3))
        odors = np.nan * np.zeros((n_steps,))
        upwind = np.zeros((n_steps, 3))
        crosswind = np.zeros((n_steps, 3))
        surge_vec = np.zeros((n_steps, 3))
        noise_convo = np.zeros(n_steps)
        wind_start_time =np.round(np.random.uniform(0,6.8),1)
        #j=0
        r= np.random.normal(.3,0.05)
        w=np.random.normal(6,1.5)
        bool_checks = np.zeros(n_steps)
        self.reject_thresh=reject_thresh
        
        for t in range(n_steps):

            if t == 0:
                b = np.array([0, start_pos[1], start_pos[2]])
                b *= (self.bias/ np.linalg.norm(b))
                v = start_vel.copy()
                x = start_pos.copy()
                d_upwind = np.array([-.4,0,0])
                new_noise= self.noise
            else:

                # get driving terms
                eta = np.random.normal(0, new_noise, (3,))
                d_upwind = (upwind[t] - upwind[t-1])  / np.linalg.norm(upwind[t] - upwind[t-1]) 
                d_cross = (crosswind[t]-crosswind[t-1]) / np.linalg.norm(crosswind[t]-crosswind[t-1])
                #b=self.bias*b_convo[t]*d_cross *bias_sign
                x1= r*np.sin(w*ts[t])*bias_sign
                y1= r*np.cos(w*ts[t])*bias_sign

                z=-d_upwind[2]*dt
                b= np.array([x1,y1,z])
                # before odor, no casting or upwind bias 
                if np.all(odors[:t]==0):
                    b*=0
                    new_noise=self.noise
                    eta = np.random.normal(0, new_noise, (3,))
                    #print('no odor yet, noise =', new_noise)
                
                #during surge, noise is reduced:
                if self.odor==1: # and (np.sum(odors[:t])<=25):
                    b*=0    
                    new_noise = .66*self.noise
                    #new_noise = np.sqrt(self.noise)
                    eta = np.random.normal(0, new_noise, (3,))
                
                #return to baseline behavior after 2 seconds of casting
                if (len(np.where(odors==1)[0]) != 0) and (np.abs(ts[t] - np.where(odors==1)[0][-1]*dt) > 2.2):
                    b*=0
                
                if np.all(bool_checks[:t]==0):
                #    b=0
                    reject_thresh=0  
                elif np.any(bool_checks[:t]!=0): 
                    reject_thresh=self.reject_thresh
                #eta = np.random.normal(0, self.noise, (3,))
                    #print(reject_thresh)
                # update velocity and position
                v += (dt / self.tau) * (-v + eta + self.bias*b + surges[t]*d_upwind - reject_thresh*d_upwind )  
                x += v * dt

                    
            #check if agent is in bounds, make it change direction if hitting bound
            v, x = self.reflect_if_out_of_bounds(v, x)
            
            
            #check if eligible to get odor
            if self.odor == False:
                #check if agent moved into trigger zone
                inTriggerZone = plume.check_bounds(x)

                if inTriggerZone:
                    #if agent in triggerzone, check if it can get odor
                    self.odor = plume.get_odor('initial', ts[t])
                    
            #save values to dict
            bs[t] = b
            vs[t] = v
            xs[t] = x
            noise_convo[t]=new_noise
            
            #get the instantaneous wind from CFD or real data
            if wind.vector_field is not None:
                wind.get_correct_wind_vec(agent_position=xs[t], t=t, ts=ts[t]) # +wind_start_time)
 
            
            #calculate crosswind/upwind for fly
            crosswind[t], upwind[t] = wind.relative_upwind(vs[t], t=t)
            #print('cross:', crosswind[t], 'up:', upwind[t])

            #save values to dict
            surge_vec[t] = surges[t]*d_upwind
            odors[t] = self.odor

            bool_checks[t] = self.check_last_odor(odors,t)
            #print('bool check:', bool_checks[t])
            # add new surge force if hit has occurred
            if self.odor==True:
                # compute alpha function starting at current time point
                # and add it to surges array
                #c=45
                c=25
                ts_ = ts[t+c:] - ts[t+c]
                surges[t+c:] += (self.surge_amp_*ts_*np.exp(-ts_/self.tau_surge))
                #update if flash is still on
                b_convo[:t] = 0
                self.odor = plume.get_odor(self.odor, ts[t])
                #print('updated odor:', self.odor)

                
            if self.odor==False:
                b_convo[t:] = np.sign(np.sin(rv*np.pi*ts[t]))
                #new_noise = new_noise + ts[t]*.001
           # if (np.sum(odors[:t])>25):     
                new_noise = self.noise*ts[t]*.33# +.25
                #j+=1
                if new_noise > self.noise:
                    new_noise= self.noise
        
        

            
        return_dict = {
            'surges': surges,
            'surge vec': surge_vec,
            'bs': bs,
            'vs': vs,
            'xs': xs,
            'odors': odors,
            'ts': ts,
            'upwind': upwind,
            'crosswind': crosswind,
            'casting':casting,
            'b_convo': b_convo,
            'noise_convo': noise_convo,
            'bool_check': bool_checks
        }

        return return_dict
    

    
 ########################################################################################################################   
class WindObs(object):
    
    def __init__(self, condition, speed=0.4, vector=None, vector_field=None, wind_positions=None, wind_data_type='real'):
        
        self.condition = condition
        self.vector_field=vector_field
        self.wind_positions=wind_positions
        self.wind_data_type= wind_data_type
        
        if condition=='constant':
            self.wind_vector = [-speed,0,0]
        if condition=='real':
            self.wind_vector = vector

            
    def get_correct_wind_vec(self, agent_position, t, ts):
        
        def closest(lst, K):
            lst = np.asarray(lst)
            idx = (np.abs(lst - K)).argmin()
            return lst[idx]
        
        if self.wind_data_type=='real':
            kdtree=KDTree(self.wind_positions)
            dist, idx = kdtree.query(agent_position)
            closest_wind = self.vector_field[idx][['xz_X', 'xy_Y', 'xz_Z']]
            ## make the x direction consistent 
            self.wind_vector = np.array(closest_wind.iloc[t])    
        
        
        elif self.wind_data_type=='CFD':
            #kdtree=KDTree(self.wind_positions)
            #print(agent_position[:2])
            #dist, idx = kdtree.query(agent_position)

            x_list = self.wind_positions[0]
            y_list = self.wind_positions[1]
            z_list = self.wind_positions[2]
            closest_x = closest(x_list, agent_position[0])
            closest_y = closest(y_list, agent_position[1])
            closest_z = closest(z_list, agent_position[2])
            
           # print('nearest positions:', closest_x, closest_y,closest_z)
            closest_wind= self.vector_field[int(np.round(ts, 1)*10)]

            

            point_closest = {'x': closest_x, 'y': closest_y,'z':  closest_z}
            point_logical = (closest_wind['x'] == point_closest['x']) & (closest_wind['y'] == point_closest['y']) & (closest_wind['z'] == point_closest['z'])
            


            point_index = int(np.squeeze(np.where(point_logical)[0]))
            closest_wind = closest_wind.iloc[[point_index], 3:]

            #closest_wind = closest_wind[['xvel', 'yvel', 'zvel']][(closest_wind.x==closest_x) & (closest_wind.y==closest_y) & (closest_wind.z==closest_z)].to_numpy()
                        
            #closest_wind = closest_wind[(closest_wind.x==closest_x) & (closest_wind.y==closest_y) & (closest_wind.z==closest_z)].to_numpy()
            #closest_wind=closest_wind[:,3:]
            
            #et = time.time()
            #delta = et - st
            #print('elapsed time for spite test:', delta)
            #print(closest_wind)
            
           # if len(closest_wind==0):
           #     self.wind_vector 
            #else:    
            #print('almost')
 
            self.wind_vector = closest_wind
                #print('wind vec:', self.wind_vector, 'ts:', ts)
        return self.wind_vector
    
    

    def relative_upwind(self, trajectory_vector, t=None):
        """Calculate the relative upwind direction given the wind vector and the trajectory vector.

        Parameters:
        wind_vector (array-like): The wind vector [Wx, Wy, Wz].
        trajectory_vector (array-like): The trajectory vector [Tx, Ty, Tz].

        Returns:
        array: The unit vector representing the relative upwind direction.
        """
        
        #if (self.condition=='real') and ((np.any(self.vector_field) == None)):
        #    wind_vec = self.wind_vector[t]

        #else:
        wind_vec = self.wind_vector
     
        
        # Convert input to numpy arrays
        W = np.array(wind_vec)
        T = np.array(trajectory_vector)


        # Calculate the upwind vector (opposite to wind vector)
        U = -W

        # Normalize the trajectory vector
        T_hat = T / np.linalg.norm(T)

        # Project U onto T
        U_parallel = np.dot(U, T_hat) * T_hat
        U_para = U - U_parallel
        
        U_para_hat = -1* U_para / np.linalg.norm(U_para)

        perpendicular_vector = np.cross(wind_vec, np.array([1, 1,1]))
        U_perp_hat = perpendicular_vector / np.linalg.norm(perpendicular_vector)
        
            
        return U_perp_hat, U_para_hat , wind_vec

    
########################################################################################################################

def run_sims(SEED, BEHAVIOR,
        N_TRAJS, DURATION, DT, BOUNDS,
        TAU, NOISE, BIAS,
        SURGE_AMP, TAU_SURGE, 
        REJECT_THRESH, 
        RADIUS=0, OMEGA=0, CONSTANT_NOISE=False,
        wind_condition='constant',
        wind_data=None, vector_field=None, wind_positions=None, wind_data_type=None):

    # build wind profile
    if wind_condition=='constant':
        wind1 = WindObs(condition='constant')
    elif wind_condition=='data':
        wind1 = WindObs(condition='real', vector=wind_data, vector_field=vector_field, wind_positions=wind_positions, wind_data_type=wind_data_type)
    
    # build agent
    ag = SurgingAgent(seed=SEED,
        tau=TAU, noise=NOISE, bias=BIAS,
         surge_amp=SURGE_AMP, tau_surge=TAU_SURGE,
        bounds=BOUNDS)
    


    # GENERATE TRAJECTORIES
    np.random.seed(SEED)

    trajs = []

    for _ in range(N_TRAJS):
        print('traj #', _)
        # choose random start position
        start_pos = np.array([
            np.random.uniform(*(-0.35, .35)),
            np.random.uniform(*(-0.075, 0.075)),
            np.random.uniform(*(0.15, 0.35)),
        ])
        
        start_vel = np.array([
            np.random.uniform(*(-.5, 0.5)),
            np.random.uniform(*(-.5, 0.5)),
            np.random.uniform(*(-0.2, 0.2)),
        ])

        # build plume
        pl = BoolPlume(bounds=np.array([(-0.5, 0.5), (-0.25, 0.25), (0, 0.5)]), lastflash=-2.8, )
       
        # make trajectory
        if BEHAVIOR=='cast_and_surge':
            traj = ag.track(plume=pl, wind=wind1, behavior=BEHAVIOR, start_pos=start_pos, start_vel=start_vel, duration=DURATION, dt=DT, reject_thresh=REJECT_THRESH, constant_noise=CONSTANT_NOISE,bias_sign=np.random.choice([-1,1]), )
        if BEHAVIOR=='sink_and_circle':
            traj = ag.track(plume=pl, wind=wind1, behavior=BEHAVIOR, start_pos=start_pos, start_vel=start_vel, duration=DURATION, dt=DT, reject_thresh=REJECT_THRESH, radius=RADIUS, omega=OMEGA, constant_noise=CONSTANT_NOISE, bias_sign=np.random.choice([-1,1]),)
            
            
        traj['headings'] = calc_heading(traj['vs'])[:, 2]
        trajs.append(traj)
        
    return trajs    

########################################################################################################################

def convert_and_save_sims(simlist, filename, date, remove_wall_hits=True, save=False):
    time=len(simlist[0]['ts'])*10-200
    dfs=[]
    for i, traj in enumerate(simlist):
        #if simulation hit the "wall", exclude from analysis
        if remove_wall_hits==True:
            if (np.max(traj['xs'][:,0]) >= 0.499) or (np.min(traj['xs'][:,0]) <= -0.499) or (np.max(traj['xs'][:,1]) >= 0.249) or (np.max(traj['xs'][:,0]) <= -0.249):
                #print('hit wall')
                continue        
        veldf = pd.DataFrame(traj['vs']).rename(columns={0:'xvel',1:'yvel',2:'zvel'})
        posdf = pd.DataFrame(traj['xs']).rename(columns={0:'x',1:'y',2:'z'})
        winddf = pd.DataFrame(traj['real_wind']).rename(columns={0:'U_x',1:'U_y',2:'U_z'})
        veldf['time stamp'] = np.arange(-200,time,10)
        veldf['heading'] = np.arctan2(veldf.yvel, veldf.xvel)
        veldf['ground speed'] = np.sqrt(veldf.xvel**2 + veldf.yvel**2)
        veldf['obj_id_unique']=date + '_' + str(i)    
        
        dfs.append(posdf.join([veldf, winddf]))   

    df = pd.concat(dfs)
    df.reset_index(drop=True)
    if save==True:
        filename = filename + date + '.h5'
        df.to_hdf(filename, key='sims')
    
    return df