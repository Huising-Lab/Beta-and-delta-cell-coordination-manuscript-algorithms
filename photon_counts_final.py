#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 18:05:44 2024

@author: Mohammad Pourhosseinzadeh

The script below was used to convert the raw data, in arbitraty units, collected from a laser scanning confocal microscope
into phton counts. 

The input data represented by the variable "data_filename" is structured such that the first column contains the time stamp for each frame and each
preceeding column contains the raw data from each region of interest (ROI) drawn. The first column contains headers starting
with the time column, labeled "Time [s]", and followed by headers for each ROI, generally labeled "Mean (#)".

The script requires an additional file represented as the variable "ROI_area_filenmae" below. This file contains, as a row vector,
the area, in um**2 for each ROI presented in the raw data. This was obtained by exporting the "area" mesurement in Fiji.
"""

import numpy as np
import pandas as pd
import os

def intensity_to_photons(raw_data,ROI_area,um_px,Gain,CF,offset):
    raw_data_key = raw_data.keys()
    ROI_area_key = ROI_area.keys()
    photon_count = {raw_data_key[0] : raw_data[raw_data_key[0]]}
    for n in range(np.shape(raw_data_key)[0] - 1):
        ROI_px_area=ROI_area[ROI_area_key[1]][n] / (um_px**2)
        photons = list((((raw_data[raw_data_key[n+1]][1:].astype(float) - offset) * ROI_px_area) / CF).astype(int))
        photons.insert(0,raw_data[raw_data_key[n+1]][0])
        photon_count.update({raw_data_key[n+1] : photons})
    return photon_count

dir_name = 'directory' #Input directory where the raw data and ROI_area files are stored, this is the same directory that will be used to save the output csv files
data_filename = 'filename'
ROI_area_filename = 'filename_ROI_area'

um_px = 0.14 #Input the pixel size, gain, and offset used during imaging
Gain = 65
offset = -10

raw_data = pd.read_csv(f'{dir_name}/{data_filename}.csv')
ROI_area = pd.read_csv(f'{dir_name}/ROI_area/{ROI_area_filename}.csv')

''' Below are the best fit curve and lines used to determine the appropriate correction factor and offset to use given the gain and offset used during imaging'''
confocal_CF_GFP = 11.052 * np.e ** (0.0362 * Gain)
confocal_offset_GFP = (0.206 * (offset ** 2)) + (9.515 * offset) + 113.25

photon_count = intensity_to_photons(raw_data, ROI_area, um_px, Gain, confocal_CF_GFP, confocal_offset_GFP)

if not os.path.exists(f'{dir_name}/photon_counts/'):
    os.makedirs(f'{dir_name}/photon_counts/')

df_photon_count = pd.DataFrame.from_dict(photon_count)
df_photon_count.to_csv(f'{dir_name}/photon_counts/{data_filename}_photon_counts.csv',index=False)
    
    
