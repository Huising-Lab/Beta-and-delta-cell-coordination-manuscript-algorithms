#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:27:00 2024

@author: Mohammad Pourhosseinzadeh
"""

from scipy import signal
import numpy as np
import pandas as pd
import math
import os
from scipy.signal import find_peaks

from matplotlib import pyplot as plt


def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def Enquiry(x):
    if not x:
        return 1
    else:
        return 0
    
def norm(x):
    n_x = (x - x.min()) / (x - x.min()).max()
    return n_x

def data_anotation(data_array, z_score, min_duration, z_threshold, sample_rate):
    data={}
    z_score=np.nan_to_num(z_score, nan=100) #Replace nans with some large number like 100
    for m in range(data_array.shape[1]):
        data[f'{key[m]}'] = {'Cell_type': [], 'Start_hill': [], 'End_hill': [], 'Start_valley': [], 'End_valley': [], 'On_rate': [], 'Off_rate': [], 'Duration': []}
        data[f'{key[m]}']['Cell_type']=raw_data.iloc[0,m+1]
        fx=z_score[:,m]
        peak=find_peaks(-(np.gradient(np.gradient(fx))/((1+(np.gradient(fx))**2)**(3/2))),height=0.005)[0]
        peaks_above_baseline=peak[np.where(z_score[peak,m]>z_threshold)]
        peaks_above_baseline=np.insert(peaks_above_baseline,[0,np.shape(peaks_above_baseline)[0]],[0,np.shape(data_array)[0]-1])
        cross_threshold=(z_score[:,m]-4)*np.roll(z_score[:,m]-4,1)
        valleys=np.where(cross_threshold<0)[0]
        #-----------------------------------------------------------
        for n in range(len(valleys)):
            vly=valleys[n]
            if z_score[vly,m]>4 and (vly==0 or vly==(data_array.shape[0]-1)):
                continue
            elif z_score[vly,m]>4 and z_score[valleys[(n-1)%len(valleys)],m]<z_score[valleys[n],m] and z_score[valleys[(n+1)%len(valleys)],m]<z_score[valleys[n],m]:
                valleys[n]=vly-1
            elif z_score[vly,m]>4 and np.gradient(z_score[:,m])[vly]!=0:
                valleys[n]=vly-(np.gradient(z_score[:,m])[vly]/(abs(np.gradient(z_score[:,m])[vly])))
        
        # Address the case where there is no pulse in a trace
        if np.shape(valleys)[0]==0:
            continue
        
        # Get rid of valleys occuring at the begining and end of a trace as they are not really imformative
        g = 0
        while g <= np.shape(valleys)[0] - 1:
            if valleys[g]==0:
                valleys=np.delete(valleys,g,axis=0)
            elif valleys[g]==np.shape(data_array)[0]-1:
                valleys=np.delete(valleys,g,axis=0)
            else:
                g+=1
            
        # Classify valleys as either the begining or end of a valley by looking at the points flanking it to see if there is a positive trend, negative trend, or saddle point
        g=0
        while g <= np.shape(valleys)[0]-1:
            if z_score[valleys[g]-1,m] < z_score[valleys[g],m] and z_score[valleys[g],m] < z_score[valleys[g]+1,m]:
                data[f'{key[m]}']['End_valley'].append(valleys[g])
                g+=1
            elif z_score[valleys[g]-1,m] > z_score[valleys[g],m] and z_score[valleys[g],m] > z_score[valleys[g]+1,m]:
                data[f'{key[m]}']['Start_valley'].append(valleys[g])
                g+=1
            elif z_score[valleys[g]-1,m] > z_score[valleys[g],m] and z_score[valleys[g],m] < z_score[valleys[g]+1,m]:
                if np.sum(valleys[g-1]==data[f'{key[m]}']['End_valley'])==1:
                    data[f'{key[m]}']['Start_valley'].append(valleys[g])
                elif np.sum(valleys[g-1]==data[f'{key[m]}']['Start_valley'])==1:
                    data[f'{key[m]}']['End_valley'].append(valleys[g])
                elif np.mean([z_score[valleys[g],m],z_score[valleys[g]-1,m]]) < np.mean([z_score[valleys[g],m],z_score[valleys[g]+1,m]]):
                    data[f'{key[m]}']['End_valley'].append(valleys[g])
                elif np.mean([z_score[valleys[g],m],z_score[valleys[g]-1,m]]) > np.mean([z_score[valleys[g],m],z_score[valleys[g]+1,m]]):
                    data[f'{key[m]}']['Start_valley'].append(valleys[g])
                g+=1
            else:
                g+=1
        
        # This ensures that every trace begins with an End_valley and ends with a Start_valley
        if np.shape(data[f'{key[m]}']['End_valley'])[0]==0 and np.shape(data[f'{key[m]}']['Start_valley'])[0]!=0:
            del data[f'{key[m]}']['Start_valley'][:]
        if np.shape(data[f'{key[m]}']['Start_valley'])[0]==0 and np.shape(data[f'{key[m]}']['End_valley'])[0]!=0: 
            data[f'{key[m]}']['Start_valley'].append(np.shape(data_array[:,m])[0]-1)
            
        if np.shape(data[f'{key[m]}']['End_valley'])[0]==0 and np.shape(data[f'{key[m]}']['Start_valley'])[0]==0:
            data[f'{key[m]}']['End_valley']=[]
            data[f'{key[m]}']['Start_valley']=[]
        
        if np.shape(data[f'{key[m]}']['End_valley'])[0]!=0 and np.shape(data[f'{key[m]}']['Start_valley'])[0]!=0:
            if data[f'{key[m]}']['Start_valley'][0] <= data[f'{key[m]}']['End_valley'][0]:
               bad_starts=np.where(data[f'{key[m]}']['Start_valley'] <= data[f'{key[m]}']['End_valley'][0])[0]
               r=0
               for n in bad_starts:
                   del data[f'{key[m]}']['Start_valley'][n-r]
                   r+=1
        
            if data[f'{key[m]}']['End_valley'][-1] >= data[f'{key[m]}']['Start_valley'][-1]:
                data[f'{key[m]}']['Start_valley'].append(np.shape(data_array[:,m])[0]-1)
        
        # This is to take care of mutliple End_valleys or Start_valleys next to eachother
        len_ev=len(data[f'{key[m]}']['End_valley'])
        len_sv=len(data[f'{key[m]}']['Start_valley'])
        if len_ev != len_sv:
            n=1
            while n < (len_ev*(len_ev<len_sv)+len_sv*(len_sv<len_ev)+len_ev*(len_ev==len_sv))-1:
                if data[f'{key[m]}']['End_valley'][n]<data[f'{key[m]}']['Start_valley'][n-1]:
                    del data[f'{key[m]}']['End_valley'][n]
                if data[f'{key[m]}']['Start_valley'][n]<data[f'{key[m]}']['End_valley'][n]:
                    del data[f'{key[m]}']['End_valley'][n]
                else:
                    n+=1
        
        # Find the maximum between the end valley and start valley to ensure that end hill and start hill are not mislabeled near the bottom of the pulse 
        g=0
        temp_max=list([])
        while g <= np.shape(data[f'{key[m]}']['End_valley'])[0]-1:
            temp_max.append(z_score[data[f'{key[m]}']['End_valley'][g]:data[f'{key[m]}']['Start_valley'][g],m].max())
            g+=1

        # Calculate the begining and end of a hill along with the duration  
        g=0      
        while g <= np.shape(data[f'{key[m]}']['End_valley'])[0]-1:
            temp_peaks_above_baseline=peaks_above_baseline[np.where(z_score[peaks_above_baseline,m]>=(0.5*temp_max[g]))[0]]
            start=data[f'{key[m]}']['End_valley'][g]
            end=data[f'{key[m]}']['Start_valley'][g]
            #start_hill=peaks_above_baseline[np.where(peaks_above_baseline>data[f'{key[m]}']['End_valley'][g])[0]]
            if np.shape(temp_peaks_above_baseline[np.where(temp_peaks_above_baseline>data[f'{key[m]}']['End_valley'][g])[0]])[0]==0 and np.shape(temp_peaks_above_baseline[np.where(temp_peaks_above_baseline>data[f'{key[m]}']['Start_valley'][g])[0]])[0]==0:
                data[f'{key[m]}']['Start_hill'].append(np.where(z_score[start:end,m]==z_score[start:end,m].max())[0][0] + start)
                data[f'{key[m]}']['End_hill'].append(np.where(z_score[start:end,m]==z_score[start:end,m].max())[0][0] + start)
                g+=1
            else:
                sh=temp_peaks_above_baseline[np.where(temp_peaks_above_baseline>data[f'{key[m]}']['End_valley'][g])[0]]
                eh=temp_peaks_above_baseline[np.where(temp_peaks_above_baseline<data[f'{key[m]}']['Start_valley'][g])[0]]
                if start<sh[0]<end and start<eh[-1]<end:
                    data[f'{key[m]}']['Start_hill'].append(sh[0])
                    data[f'{key[m]}']['End_hill'].append(eh[-1])
                    g+=1
                else:
                    data[f'{key[m]}']['Start_hill'].append(np.where(z_score[start:end,m]==z_score[start:end,m].max())[0][0] + start)
                    data[f'{key[m]}']['End_hill'].append(np.where(z_score[start:end,m]==z_score[start:end,m].max())[0][0] + start)
                    g+=1
        
        # I'm adding one to everything since it seems that all of the indicies are behind by one. I need to figure out why this is, but for now it is a temporary solution
        data[f'{key[m]}']['End_valley']=list((np.array(data[f'{key[m]}']['End_valley'])+np.ones(len(data[f'{key[m]}']['End_valley']))).astype(int))
        data[f'{key[m]}']['Start_hill']=list((np.array(data[f'{key[m]}']['Start_hill'])+np.ones(len(data[f'{key[m]}']['Start_hill']))).astype(int))
        data[f'{key[m]}']['End_hill']=list((np.array(data[f'{key[m]}']['End_hill'])+np.ones(len(data[f'{key[m]}']['End_hill']))).astype(int))
        data[f'{key[m]}']['Start_valley']=list((np.array(data[f'{key[m]}']['Start_valley'])+np.ones(len(data[f'{key[m]}']['Start_valley']))).astype(int))
                    
        data[f'{key[m]}']['Duration']=list(np.array(data[f'{key[m]}']['Start_valley'])-np.array(data[f'{key[m]}']['End_valley']))
        
        # Get rid of any peaks that are less than a minute long as I use 1 min for the fast component filter
        short_pulse=np.where(np.array(data[f'{key[m]}']['Duration'])<=min_duration)[0]
        r=0
        for n in short_pulse:
            del data[f'{key[m]}']['Start_valley'][n-r]
            del data[f'{key[m]}']['End_valley'][n-r]
            del data[f'{key[m]}']['Start_hill'][n-r]
            del data[f'{key[m]}']['End_hill'][n-r]
            del data[f'{key[m]}']['Duration'][n-r]
            r+=1
        
        # Calculate off and on rates
        for l in range(0,len(data[f'{key[m]}']['End_valley'])):
            if data[f'{key[m]}']['Start_hill'][l] - data[f'{key[m]}']['End_valley'][l] == 1:
                data[f'{key[m]}']['On_rate'].append(sample_rate)
            else:
                data[f'{key[m]}']['On_rate'].append(abs(np.gradient(norm(z_score[data[f'{key[m]}']['End_valley'][l]:data[f'{key[m]}']['Start_hill'][l],m]))).max() * sample_rate)
                
            if data[f'{key[m]}']['Start_valley'][l] - data[f'{key[m]}']['End_hill'][l] == 1:
                data[f'{key[m]}']['Off_rate'].append(sample_rate)
            else:
                data[f'{key[m]}']['Off_rate'].append(abs(np.gradient(norm(z_score[data[f'{key[m]}']['End_hill'][l]:data[f'{key[m]}']['Start_valley'][l],m]))).max() * sample_rate)
    return data

period_low = 20 #Choose cutoff frequency of slow component
cutoff_frequency_low = 1/(60*period_low)
period_high=1 #Choose cutoff frequency of fast component
cutoff_frequency_high=1/(60*period_high)
z_threshold=4

t1='start'
t2='end'

avg = 16  # choose the amount of averaging applied during imaging

directory='directory'
filename='filename.csv'

raw_data = pd.read_csv(f'{directory}/photon_counts/{filename}')
time = raw_data['Time [s]'][1:].astype(int)

# import data here, x-axis=time and y-axis=data
if t1=='start' and t2=='end':
    data = raw_data.iloc[1:, 1:]
else:
    idx_1=np.where(time>t1*60)[0][0]+1 # This is to correct for the additional string, labeling the cell id in the first row
    idx_2=np.where(time>t2*60)[0][0]+1
    data=raw_data.iloc[idx_1:idx_2,1:]
    time=time[idx_1:idx_2]

data_array = np.array(data.values.tolist()).astype(int)
data_array = data_array-(np.ones(np.shape(data_array))*((np.min(data_array,axis=0)<0)*np.min(data_array,axis=0)))
key = list(data.keys())

sample_rate = 1/np.diff(time).mean()
signal_length = len(time)
nyq_freq = sample_rate/2
z_scores = {'Time [s]': time}

# implement a loop so that we can run through all cells in a csv file
baseline_low_array=np.zeros(np.shape(data_array))
baseline_high_array=np.zeros(np.shape(data_array))
for m in range(data_array.shape[1]):
    # Filter signal x, result stored to y
    n = 1
    # need to make a copy of y_noise, saving it as temp links the two variables temp and y_noise meaning that changes to temp also change y_noise, thats why I used np.copy()
    temp = np.copy(data_array[:, m])
    while n <= 10:
        baseline_low = butter_lowpass_filter(temp, cutoff_frequency_low, nyq_freq)
        baseline_low[np.where(baseline_low < temp.min())] = temp.min()
        # grad_baseline_low = np.gradient(np.gradient(baseline_low))
        z = ((data_array[:, m]-baseline_low) /
             (np.sqrt(baseline_low)/((np.log(avg)/np.log(10))+1)))
        # Works well as a baseline when z is set low, when set high the baseline better approximates the low pass filtered signal
        flag = np.where(z >= 4)[0]
        temp[flag] = baseline_low[flag]
        n += 1
    baseline_low_array[:,m]=baseline_low
    
for m in range(data_array.shape[1]):
    # Filter signal x, result stored to y
    n = 1
    # need to make a copy of y_noise, saving it as temp links the two variables temp and y_noise meaning that changes to temp also change y_noise, thats why I used np.copy()
    temp = np.copy(data_array[:, m])
    while n <= 10:
        baseline_high = butter_lowpass_filter(temp, cutoff_frequency_high, nyq_freq)
        baseline_high[np.where(baseline_high < temp.min())] = temp.min()
        # grad_baseline_low = np.gradient(np.gradient(baseline_low))
        z = ((data_array[:, m]-baseline_high) /
             (np.sqrt(baseline_high)/((np.log(avg)/np.log(10))+1)))
        # Works well as a baseline when z is set low, when set high the baseline better approximates the low pass filtered signal
        flag = np.where(z >= 4)[0]
        temp[flag] = baseline_high[flag]
        n += 1
    baseline_high_array[:,m]=baseline_high
    
slow_component=baseline_high_array-baseline_low_array
slow_component_z_score= (slow_component / (np.sqrt(baseline_low_array)/((np.log(avg)/np.log(10))+1)))
slow_component_z_score=np.nan_to_num(slow_component_z_score, nan=100)
slow_component_z_score_dictionary={key[m]: slow_component_z_score[:,m] for m in range(np.shape(data_array)[1])}
slow_component_z_score_dictionary.update({'Time_[s]': time})

fast_component=data_array-baseline_high_array
fast_component_z_score=(fast_component / (np.sqrt(baseline_high_array)/((np.log(avg)/np.log(10))+1)))
fast_component_z_score=np.nan_to_num(fast_component_z_score, nan=100)
fast_component_z_score_dictionary={key[m]: fast_component_z_score[:,m] for m in range(np.shape(data_array)[1])}
fast_component_z_score_dictionary.update({'Time_[s]': time})

data_slow=data_anotation(data_array, slow_component_z_score, 0*sample_rate, z_threshold, sample_rate)
data_fast=data_anotation(data_array, fast_component_z_score, 0*sample_rate, z_threshold, sample_rate)

slow_component_z_score_df=pd.DataFrame.from_dict(slow_component_z_score_dictionary)    
fast_component_z_score_df=pd.DataFrame.from_dict(fast_component_z_score_dictionary)      
baseline_low_df = pd.DataFrame.from_dict(baseline_low_array)  # save low and high freqency filtered baselines
baseline_high_df = pd.DataFrame.from_dict(baseline_high_array)
data_slow_df=pd.DataFrame.from_dict(data_slow)
data_fast_df=pd.DataFrame.from_dict(data_fast)


if not os.path.exists(f'{directory}/time_series_filter/'):
    os.makedirs(f'{directory}/time_series_filter/')
    os.makedirs(f'{directory}/time_series_filter/slow_component_z_score/')
    os.makedirs(f'{directory}/time_series_filter/fast_component_z_score/')
    os.makedirs(f'{directory}/time_series_filter/baseline_low/')
    os.makedirs(f'{directory}/time_series_filter/baseline_high/')
    os.makedirs(f'{directory}/time_series_filter/data_slow/')
    os.makedirs(f'{directory}/time_series_filter/data_fast/')

if t1=='start' and t2=='end':
    slow_component_z_score_df.to_csv(f'{directory}/time_series_filter/slow_component_z_score/{filename[:-4]}_period_{period_low}_z_score.csv', index=False) # save to a csv file
    fast_component_z_score_df.to_csv(f'{directory}/time_series_filter/fast_component_z_score/{filename[:-4]}_period_{period_high}_z_score.csv', index=False) # save to a csv file
    baseline_low_df.to_csv(f'{directory}/time_series_filter/baseline_low/{filename[:-4]}_period_{period_low}_array.csv', index=False) # save to a csv file
    baseline_high_df.to_csv(f'{directory}/time_series_filter/baseline_high/{filename[:-4]}_period_{period_high}_array.csv', index=False) # save to a csv file
    data_slow_df.to_pickle(f'{directory}/time_series_filter/data_slow/{filename[:-4]}_period_{period_low}_{period_high}_slow_component_analyzed_data.pkl')
    data_fast_df.to_pickle(f'{directory}/time_series_filter/data_fast/{filename[:-4]}_period_{period_low}_{period_high}_fast_component_analyzed_data.pkl')
else:
    slow_component_z_score_df.to_csv(f'{directory}/time_series_filter/slow_component_z_score/{filename[:-4]}_time{t1}-{t2}s_period_{period_low}_z_score.csv', index=False) # save to a csv file
    fast_component_z_score_df.to_csv(f'{directory}/time_series_filter/fast_component_z_score/{filename[:-4]}_time{t1}-{t2}s_period_{period_high}_z_score.csv', index=False) # save to a csv file
    baseline_low_df.to_csv(f'{directory}/time_series_filter/baseline_low/{filename[:-4]}_time{t1}-{t2}s_period_{period_low}_array.csv', index=False) # save to a csv file
    baseline_high_df.to_csv(f'{directory}/time_series_filter/baseline_high/{filename[:-4]}_time{t1}-{t2}s_period_{period_high}_array.csv', index=False) # save to a csv file
    data_slow_df.to_pickle(f'{directory}/time_series_filter/data_slow/{filename[:-4]}_time{t1}-{t2}s_period_{period_low}_{period_high}_slow_component_analyzed_data.pkl')
    data_fast_df.to_pickle(f'{directory}/time_series_filter/data_fast/{filename[:-4]}_time{t1}-{t2}s_period_{period_low}_{period_high}_fast_component_analyzed_data.pkl')
    
    
    
    
    
