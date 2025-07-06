#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:55:24 2024

@author: Mohammad Pourhosseinzadeh
"""

import numpy as np
import pandas as pd
import statistics
import os

def norm(x):
    if len(np.where(x>=1000)[0])>0:
        x[np.where(x>=1000)[0]]=1000
    n_x=(x-x.min())/((x-x.min()).max())
    return n_x

def dataframe_concat(a,b,maximum):
    com=a.copy()
    for n in a.keys():
        for m in range(1,len(a[n].keys())):
            if m<5:
                com[n][m]=(list(a[n][m])+list(np.add(b[n][m],maximum)))
            else:
                com[n][m]=(list(a[n][m])+list(b[n][m]))
    return com

directory='directory'
filename='filename_photon_counts'
output='features/'
ts_filter = 'time_series_filter'

num_data_sets=1

trace_design=pd.read_csv(directory + f'trace_design/{filename}.csv')
trace_key=trace_design.keys()

photon_counts=pd.read_csv(directory + f'photon_counts/{filename}.csv')
raw_data=photon_counts.iloc[1:,1:]
time=photon_counts.iloc[1:,0]

if num_data_sets==2:
    peaks_data_slow_1=pd.read_pickle(directory + f'{ts_filter}/data_slow/{filename}_1_period_20_1_slow_component_analyzed_data.pkl')
    peaks_data_slow_2=pd.read_pickle(directory + f'{ts_filter}/data_slow/{filename}_2_period_20_1_slow_component_analyzed_data.pkl')
    
    slow_z_score_1=pd.read_csv(directory + f'{ts_filter}/slow_component_z_score/{filename}_1_period_20_z_score.csv')
    slow_z_score_2=pd.read_csv(directory + f'{ts_filter}/slow_component_z_score/{filename}_2_period_20_z_score.csv')
    slow_z_score=pd.concat([slow_z_score_1,slow_z_score_2],ignore_index=True)
    slow_z_score_avg=norm(np.mean(slow_z_score[slow_z_score.keys()[0:-1]],axis=1))
    peaks_data_slow=dataframe_concat(peaks_data_slow_1,peaks_data_slow_2,len(slow_z_score_1))
    
    peaks_data_fast_1=pd.read_pickle(directory + f'{ts_filter}/data_fast/{filename}_1_period_20_1_fast_component_analyzed_data.pkl')
    peaks_data_fast_2=pd.read_pickle(directory + f'{ts_filter}/data_fast/{filename}_2_period_20_1_fast_component_analyzed_data.pkl')

    fast_z_score_1=pd.read_csv(directory + f'{ts_filter}/fast_component_z_score/{filename}_1_period_1_z_score.csv')
    fast_z_score_2=pd.read_csv(directory + f'{ts_filter}/fast_component_z_score/{filename}_2_period_1_z_score.csv')
    fast_z_score=pd.concat([fast_z_score_1,fast_z_score_2],ignore_index=True)
    fast_z_score_avg=norm(np.mean(fast_z_score[fast_z_score.keys()[0:-1]],axis=1))
    peaks_data_fast=dataframe_concat(peaks_data_fast_1,peaks_data_fast_2,len(fast_z_score_1))
                          
else:  
    peaks_data_slow=pd.read_pickle(directory + f'{ts_filter}/data_slow/{filename}_period_20_1_slow_component_analyzed_data.pkl')
    peaks_data_fast=pd.read_pickle(directory + f'{ts_filter}/data_fast/{filename}_period_20_1_fast_component_analyzed_data.pkl')
    slow_z_score=pd.read_csv(directory + f'{ts_filter}/slow_component_z_score/{filename}_period_20_z_score.csv')
    slow_z_score_avg=norm(np.mean(slow_z_score[slow_z_score.keys()[0:-1]],axis=1))
    fast_z_score=pd.read_csv(directory + f'{ts_filter}/fast_component_z_score/{filename}_period_1_z_score.csv')
    fast_z_score_avg=norm(np.mean(fast_z_score[fast_z_score.keys()[0:-1]],axis=1))
    
cell_key=peaks_data_slow.keys()

features={}

for n in trace_key:

    r=0
    t1=[]
    t2=[]
    t=[]
    slow_z_score_avg_temp=[]
    fast_z_score_avg_temp=[]
    while r<sum(~np.isnan(trace_design[n]))-1:
       start=trace_design[n][r]*60
       t1_temp=np.where(abs(time-start)==abs(time-start).min())[0][0]
        
       end=trace_design[n][r+1]*60
       t2_temp=np.where(abs(time-end)==abs(time-end).min())[0][0]
       t=list(t)+list([t1_temp])+list([t2_temp])
       slow_z_score_avg_temp=slow_z_score_avg_temp + list(slow_z_score_avg[t1_temp:t2_temp])
       fast_z_score_avg_temp=fast_z_score_avg_temp + list(fast_z_score_avg[t1_temp:t2_temp])
       r+=1
    
    dt=np.diff(time[t[0]:t[1]]).mean()
    
    features[n]={"cell_id":[], "cell_type":[], "frequency_slow":[],
              "duty_cycle_slow":[], "on_rate_slow_mean":[], "on_rate_slow_median":[], 
              "on_rate_slow_stdev":[], "off_rate_slow_mean":[], "off_rate_slow_median":[], 
              "off_rate_slow_stdev":[], "max_amplitude_slow":[],"min_amplitude_slow":[],
              "mean_amplitude_slow":[], "median_amplitude_slow":[], "stdev_amplitude_slow":[],
              "cross_correlation_slow":[],"no_peaks_slow":[], "one_peak_slow":[], "frequency_fast":[],
              "duty_cycle_fast":[], "on_rate_fast_mean":[], "on_rate_fast_median":[],
              "on_rate_fast_stdev":[], "off_rate_fast_mean":[], "off_rate_fast_median":[],
              "off_rate_fast_stdev":[], "max_amplitude_fast":[], "min_amplitude_fast":[],
              "mean_amplitude_fast":[], "median_amplitude_fast":[], "stdev_amplitude_fast":[],
              "cross_correlation_fast":[],"no_peaks_fast":[], "one_peak_fast":[]}
    
    num_peaks_slow=[]
    frequency_slow=[]
    duty_cycle_slow=[]
    num_peaks_fast=[]
    frequency_fast=[]
    duty_cycle_fast=[]
    
    for g in cell_key:
        
        features[n]["cell_id"].append(g)
        features[n]["cell_type"].append(peaks_data_slow[g]["Cell_type"])
        
        # slow component
        r=0
        norm_slow_total=list(norm(slow_z_score[g]))
        norm_slow=[]
        end_valley_temp=[]
        start_valley_temp=[]
        start_hill_temp=[]
        on_rate=[]
        off_rate=[]
        total_time=0
        while r<=len(t)-2:
            norm_slow=np.array(list(norm_slow) + list(norm_slow_total[t[r]:t[r+1]]))
            above_ti=np.where(peaks_data_slow[g]['End_valley']>t[r])[0]
            below_tf=np.where(peaks_data_slow[g]['End_valley']<t[r+1])[0]
            if len(set(above_ti).intersection(set(below_tf)))!=0:
                wndw=list(set(above_ti).intersection(set(below_tf)))
                end_valley_temp=list(end_valley_temp) + list(np.array(peaks_data_slow[g]['End_valley'])[wndw])
                start_valley_temp=list(start_valley_temp) + list(np.array(peaks_data_slow[g]['Start_valley'])[wndw])
                start_hill_temp=list(start_hill_temp) + list(np.array(peaks_data_slow[g]['Start_hill'])[wndw])
                on_rate=list(on_rate) + list(np.array(peaks_data_slow[g]['On_rate'])[wndw])
                off_rate=list(off_rate) + list(np.array(peaks_data_slow[g]['Off_rate'])[wndw])
                total_time=total_time + (t[r+1]-t[r])
                r+=1
            else:
                total_time=total_time + (t[r+1]-t[r])
                r+=1
                
        num_peaks_slow.append(len(end_valley_temp))
        frequency_slow.append(num_peaks_slow[-1]/(total_time*dt))
        duty_cycle_slow.append(np.sum(np.divide(np.subtract(start_valley_temp,end_valley_temp),total_time)))
        
        features[n]["frequency_slow"].append(frequency_slow[-1])
        features[n]["duty_cycle_slow"].append(duty_cycle_slow[-1])
        
        if len(end_valley_temp)==0:
            features[n]["on_rate_slow_mean"].append(np.nan) # instead of putting nan values impute these missing values with the mean and then create a new binary column dictating whether or not a there is a nan value her representing the presence of absence of a response
            features[n]["on_rate_slow_median"].append(np.nan)
            features[n]["off_rate_slow_mean"].append(np.nan)
            features[n]["off_rate_slow_median"].append(np.nan)
            features[n]["max_amplitude_slow"].append(0)
            features[n]["min_amplitude_slow"].append(0)
            features[n]["mean_amplitude_slow"].append(0)
            features[n]["median_amplitude_slow"].append(0)
            features[n]["no_peaks_slow"].append(1)
        else:
            features[n]["on_rate_slow_mean"].append(statistics.mean(np.multiply(on_rate,dt)))
            features[n]["on_rate_slow_median"].append(statistics.median(np.multiply(on_rate,dt)))
            features[n]["off_rate_slow_mean"].append(statistics.mean(np.multiply(off_rate,dt)))
            features[n]["off_rate_slow_median"].append(statistics.median(np.multiply(off_rate,dt)))
            features[n]["max_amplitude_slow"].append(np.array(norm_slow_total)[start_hill_temp].max())
            features[n]["min_amplitude_slow"].append(np.array(norm_slow_total)[start_hill_temp].min())
            features[n]["mean_amplitude_slow"].append(statistics.mean(np.array(norm_slow_total)[start_hill_temp]))
            features[n]["median_amplitude_slow"].append(statistics.median(np.array(norm_slow_total)[start_hill_temp]))
            features[n]["no_peaks_slow"].append(0)
        
        if len(off_rate)<=1:
            features[n]["off_rate_slow_stdev"].append(np.nan)
            features[n]["on_rate_slow_stdev"].append(np.nan)
        else:
            features[n]["off_rate_slow_stdev"].append(statistics.stdev(np.multiply(off_rate,dt)))
            features[n]["on_rate_slow_stdev"].append(statistics.stdev(np.multiply(on_rate,dt)))
    
        if len(np.array(norm_slow_total)[start_hill_temp])<=1:
            features[n]["stdev_amplitude_slow"].append(np.nan)
            features[n]["one_peak_slow"].append(1)
        else:
            features[n]["stdev_amplitude_slow"].append(statistics.stdev(np.array(norm_slow_total)[start_hill_temp]))
            features[n]["one_peak_slow"].append(0)
        
        features[n]["cross_correlation_slow"].append(np.corrcoef(norm_slow,slow_z_score_avg_temp)[0,1])
        
        # fast component
        r=0
        norm_fast_total=list(norm(fast_z_score[g]))
        norm_fast=[]
        end_valley_temp=[]
        start_valley_temp=[]
        start_hill_temp=[]
        on_rate=[]
        off_rate=[]
        while r<=len(t)-2:
            norm_fast=np.array(list(norm_fast) + list(norm_fast_total[t[r]:t[r+1]]))
            above_ti=np.where(peaks_data_fast[g]['End_valley']>t[r])[0]
            below_tf=np.where(peaks_data_fast[g]['End_valley']<t[r+1])[0]
            if len(set(above_ti).intersection(set(below_tf)))!=0:
                wndw=list(set(above_ti).intersection(set(below_tf)))
                end_valley_temp=list(end_valley_temp) + list(np.array(peaks_data_fast[g]['End_valley'])[wndw])
                start_valley_temp=list(start_valley_temp) + list(np.array(peaks_data_fast[g]['Start_valley'])[wndw])
                start_hill_temp=list(start_hill_temp) + list(np.array(peaks_data_fast[g]['Start_hill'])[wndw])
                on_rate=list(on_rate) + list(np.array(peaks_data_fast[g]['On_rate'])[wndw])
                off_rate=list(off_rate) + list(np.array(peaks_data_fast[g]['Off_rate'])[wndw])
                total_time=total_time + (t[r+1]-t[r])
                r+=1
            else:
                total_time=total_time + (t[r+1]-t[r])
                r+=1
        
        num_peaks_fast.append(len(end_valley_temp))
        frequency_fast.append(num_peaks_fast[-1]/(total_time*dt))
        duty_cycle_fast.append(np.sum(np.divide(np.subtract(start_valley_temp,end_valley_temp),total_time)))
        
        features[n]["frequency_fast"].append(frequency_fast[-1])
        features[n]["duty_cycle_fast"].append(duty_cycle_fast[-1])
        
        if len(end_valley_temp)==0:
            features[n]["on_rate_fast_mean"].append(np.nan)
            features[n]["on_rate_fast_median"].append(np.nan)
            features[n]["off_rate_fast_mean"].append(np.nan)
            features[n]["off_rate_fast_median"].append(np.nan)
            features[n]["max_amplitude_fast"].append(0)
            features[n]["min_amplitude_fast"].append(0)
            features[n]["mean_amplitude_fast"].append(0)
            features[n]["median_amplitude_fast"].append(0)
            features[n]["no_peaks_fast"].append(1)
        else:
            features[n]["on_rate_fast_mean"].append(statistics.mean(np.multiply(on_rate,dt)))
            features[n]["on_rate_fast_median"].append(statistics.median(np.multiply(on_rate,dt)))
            features[n]["off_rate_fast_mean"].append(statistics.mean(np.multiply(off_rate,dt)))
            features[n]["off_rate_fast_median"].append(statistics.median(np.multiply(off_rate,dt)))
            features[n]["max_amplitude_fast"].append(np.array(norm_fast_total)[start_hill_temp].max())
            features[n]["min_amplitude_fast"].append(np.array(norm_fast_total)[start_hill_temp].min())
            features[n]["mean_amplitude_fast"].append(statistics.mean(np.array(norm_fast_total)[start_hill_temp]))
            features[n]["median_amplitude_fast"].append(statistics.median(np.array(norm_fast_total)[start_hill_temp]))
            features[n]["no_peaks_fast"].append(0)
        
        if len(off_rate)<=1:
            features[n]["off_rate_fast_stdev"].append(np.nan)
            features[n]["on_rate_fast_stdev"].append(np.nan)
        else:
            features[n]["off_rate_fast_stdev"].append(statistics.stdev(np.multiply(off_rate,dt)))
            features[n]["on_rate_fast_stdev"].append(statistics.stdev(np.multiply(on_rate,dt)))
        
        if len(np.array(norm_fast_total)[start_hill_temp])<=1:
            features[n]["stdev_amplitude_fast"].append(np.nan)
            features[n]["one_peak_fast"].append(1)
        else:
            features[n]["stdev_amplitude_fast"].append(statistics.stdev(np.array(norm_fast_total)[start_hill_temp]))
            features[n]["one_peak_fast"].append(0)
        
        features[n]["cross_correlation_fast"].append(np.corrcoef(norm_fast,fast_z_score_avg_temp)[0,1])
    
    num_peaks_slow_avg=np.mean(num_peaks_slow)
    frequency_slow_avg=np.mean(frequency_slow)
    duty_cycle_slow_avg=np.mean(duty_cycle_slow)
    num_peaks_fast_avg=np.mean(num_peaks_fast)
    frequency_fast_avg=np.mean(frequency_fast)
    duty_cycle_fast_avg=np.mean(duty_cycle_fast)
    
    features[n]["frequency_slow"]=list(np.divide(features[n]["frequency_slow"],frequency_slow_avg))
    features[n]["duty_cycle_slow"]=list(np.divide(features[n]["duty_cycle_slow"],duty_cycle_slow_avg))
    features[n]["frequency_fast"]=list(np.divide(features[n]["frequency_fast"],frequency_fast_avg))
    features[n]["duty_cycle_fast"]=list(np.divide(features[n]["duty_cycle_fast"],duty_cycle_fast_avg))

if not os.path.exists(directory + output):
    os.makedirs(directory + output)

features_df=pd.DataFrame.from_dict(features)

features_df.to_pickle(directory + output + f'{filename}_features.pkl')
    
