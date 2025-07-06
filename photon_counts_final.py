#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 18:05:44 2024

@author: Mohammad Pourhosseinzadeh
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

dir_name = 'directory'
data_filename = 'filename'
ROI_area_filename = 'filename_ROI_area'

um_px = 0.14
Gain = 65
offset = -10

raw_data = pd.read_csv(f'{dir_name}/{data_filename}.csv')
ROI_area = pd.read_csv(f'{dir_name}/ROI_area/{ROI_area_filename}.csv')

old_confocal_CF_GFP = 11.052 * np.e ** (0.0362 * Gain)
old_confocal_offset_GFP = (0.206 * (offset ** 2)) + (9.515 * offset) + 113.25

light_sheet_CF_GFP = 25.49 * np.e ** (0.0491 * Gain)
light_sheet_offset_GFP = (0.0833 * (offset ** 2)) + (3.7258 * offset) + 47.049

light_sheet_CF_RFP = 35.843 * np.e ** (0.0553 * Gain)
light_sheet_offset_RFP = (0.0841 * (offset **2)) + (3.685 * offset) + 44.922

photon_count = intensity_to_photons(raw_data, ROI_area, um_px, Gain, old_confocal_CF_GFP, old_confocal_offset_GFP)

if not os.path.exists(f'{dir_name}/photon_counts/'):
    os.makedirs(f'{dir_name}/photon_counts/')

df_photon_count = pd.DataFrame.from_dict(photon_count)
df_photon_count.to_csv(f'{dir_name}/photon_counts/{data_filename}_photon_counts.csv',index=False)
    
    
