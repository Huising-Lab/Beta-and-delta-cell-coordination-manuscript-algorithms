#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:36:07 2024

@author: Mohammad Pourhosseinzadeh
"""

#11/3/24
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

from skimage.segmentation import expand_labels
from scipy.signal import find_peaks
from collections import Counter
from numpy.fft import fft2,ifft2
from skimage.filters import threshold_mean

import os

from tifffile.tifffile import astype
from tifffile import imsave
from skimage.filters import sobel
from skimage.measure import label,regionprops, regionprops_table
from skimage.segmentation import watershed, expand_labels
from matplotlib import pyplot as plt
from statistics import median
from PIL import Image
from scipy import ndimage
from scipy.signal import argrelextrema

def norm(x):
  n_x=(x-x.min())/((x-x.min()).max())
  return n_x

def triangle(image,n,offset):
  # Is an implementation of the triangle threshold protocol from imagej
  counts,bins=np.histogram(image,n)
  n_counts=norm(counts[1:])
  n_bins=norm(bins[1:]) # normalize both counts and bins so x and y axis are scaled 0 to 1
  pks=find_peaks(n_counts) # use find peaks to locate the first peak in the histogram of pixel intensities, this is meant to filter out the large peak near a pixel value of zero that some images pocess
  hist_max=pks[0][0]
  x_1=n_bins[hist_max]
  y_2=n_counts[hist_max]
  index=np.linspace(0,len(n_counts)-1,len(n_counts))
  non_zero_idx=index*(n_counts!=0)
  hist_min=non_zero_idx[non_zero_idx!=0].max().astype(int) # find the index of the last bin with a non-zero frequency or count
  x_2=n_bins[hist_min]
  y_1=n_counts[hist_min]
  #f_x=(1/1-x_1)*(-np.array(n_bins)+1) # function for a line representing the hypotenus of a triangle drawn from the maximum to minimum point in the histogram
  f_x=((y_1 - y_2) / (x_2 - x_1)) * (np.array(n_bins) - x_1) + y_2
  g_x=f_x[hist_max:hist_min]-n_counts[hist_max:hist_min] # Calculates the distance between each point in the histogram and the hypotenus described above, this distance is proportional to the distance of a line drawn perpendicular to the hypotenus that touches each point on the histogram since d=arcsin(pi/4)*h
  g_idx=np.linspace(0,len(g_x)-1,len(g_x)).astype(int)
  g_max=g_idx[g_idx*(g_x==g_x.max())!=0][0] # find the index with maximum distance between the histogram and they hypotenus
  thresh=bins[g_max+hist_max+np.round(offset*n).astype(int)] # adds a small user defined offset
  binary=image>=thresh
  return binary,thresh

def borders(cell_mask,distance):
  kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]]) # Kernel used in nd.image.laplace
  dim_col=np.shape(cell_mask)[1]+distance
  dim_row=np.shape(cell_mask)[0]+distance
  F_kernel=fft2(kernel,(dim_row,dim_col))
  F_cell_mask=fft2(cell_mask,(dim_row,dim_col)) # need to exapnd kernel by the desired distance of border thickness to account for phase shift in ifft image
  membrane=ifft2((F_kernel**(distance))*F_cell_mask)[distance:,distance:] #get rid of zero padded section of image to get back to the original image dimensions
  border=(np.round(np.abs(membrane))>0)*cell_mask
  return border

directory = r'directory'
output = r'output_folder'

stain = os.listdir(directory + r'/2D')
number = os.listdir(directory + r'/2D/' + stain[0])
for n in number:
  for roots, dirs, files in os.walk(directory + fr'/2D/{stain[0]}/{n}'):
    for g in files:
      cell_mask = np.array(Image.open(directory + fr'/2D/membrane/{n}/{g[:-7]}_membrane_cp_masks.png'))
      gap_junc = np.array(Image.open(directory + fr'/2D/Cx36/{n}/{g[:-7]}_C1.tif'))
      ins = np.array(Image.open(directory + fr'/2D/insulin/{n}/{g[:-7]}_C0.tif'))
      sst = np.array(Image.open(directory + fr'/2D/Sst/{n}/{g[:-7]}_C3.tif'))

      id_area = cell_mask - borders(cell_mask, 5)

      membrane = cell_mask * (np.abs(ndimage.laplace(cell_mask)) > 0)
      expand_membrane = expand_labels(membrane, distance = 2)
      im_binary, im_thresh = triangle(gap_junc, 256, 0)
      image = im_binary * gap_junc

      px = 5
      LPF1 = cv2.GaussianBlur(image, (px, px), 0)
      LPF2 = cv2.GaussianBlur(image, (px * 2 + 1,px * 2 + 1), 0)
      BPF = LPF1.astype(int) - LPF2.astype(int)
      BPF_image = (BPF > 0) * BPF
      thresh, binary = cv2.threshold(BPF_image.astype('uint16'), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

      neighborhood_size = 5

      data_max = filters.maximum_filter(BPF, neighborhood_size)
      maxima = (BPF == data_max)
      data_min = filters.minimum_filter(BPF, neighborhood_size)
      diff = ((data_max - data_min) > 5) # For real images this number needs to be a small positive number like 5
      maxima[diff == 0] = 0
      maxima = maxima * ((BPF_image > thresh) * 1)

      labeled, num_objects = ndimage.label(maxima)
      slices = ndimage.find_objects(labeled)
      x, y = [], []
      for dy,dx in slices:
          x_center = (dx.start + dx.stop - 1)/2
          x.append(x_center)
          y_center = (dy.start + dy.stop - 1)/2
          y.append(y_center)

      m = (BPF > thresh) * BPF #for real images, this parameter needs to be a small positive number like 10
      mask = np.zeros([m.shape[0], m.shape[1]])
      x = np.array(x).astype(int)
      y = np.array(y).astype(int)

      for row in range(0, m.shape[0]-1):
        for column in range(0, m.shape[1]-1):
          if m[row,column] > 0:
            distance = (((row - y.astype(int)) ** 2) + ((column - x.astype(int)) ** 2)) ** (1 / 2)
            mask[row, column]=pd.Series(distance).idxmin() + 1

      mask = mask.astype(int)

      #Cell identification

      bcell_count=[0]
      dcell_count=[0]
      nacell_count=[0]

      #####This is the code needed to automatically print the ROIS####
      #pulls properties from image overlapping with mask
      labels=np.int64(id_area)
      props = regionprops(labels, gap_junc)
      props1 = regionprops(labels,ins)
      props2 = regionprops(labels,sst)
      props_mask=regionprops(cell_mask)

      lbl=np.zeros(labels.max())
      mean_GFP=np.zeros(labels.max())
      std_GFP=np.zeros(labels.max())
      med_GFP=np.zeros(labels.max())

      mean_RFP=np.zeros(labels.max())
      std_RFP=np.zeros(labels.max())
      med_RFP=np.zeros(labels.max())

      mean_Cy5=np.zeros(labels.max())
      std_Cy5=np.zeros(labels.max())
      med_Cy5=np.zeros(labels.max())

      x0=np.zeros(labels.max())
      y0=np.zeros(labels.max())

      #####creates a data frame of label number, x and y position, as well as mean, median, standard deviation of pixel intensity with in each ROI#####
      # This extracts the data listed above from the image
      for index in range(0, len(props)):
          lbl[index] = props[index].label
          intensity_GFP=getattr(props[index],'intensity_image').flatten()
          intensity_GFP=intensity_GFP[np.where(intensity_GFP>0)]
          if len(intensity_GFP) == 0:
              mean_GFP[index] = 0
              std_GFP[index] = 10000
              med_GFP[index] = 0
          else:
              mean_GFP[index]=getattr(props[index],'mean_intensity')
              std_GFP[index]=intensity_GFP.std()
              med_GFP[index]=median(intensity_GFP)

          intensity_RFP=getattr(props1[index],'intensity_image').flatten()
          intensity_RFP=intensity_RFP[np.where(intensity_RFP>0)]
          if len(intensity_RFP) == 0:
              mean_RFP[index] = 0
              std_RFP[index] = 10000
              med_RFP[index] = 0
          else:
              mean_RFP[index]=getattr(props1[index],'mean_intensity')
              std_RFP[index]=intensity_RFP.std()
              med_RFP[index]=median(intensity_RFP)

          intensity_Cy5=getattr(props2[index],'intensity_image').flatten()
          intensity_Cy5=intensity_Cy5[np.where(intensity_Cy5>0)]
          if len(intensity_Cy5) == 0:
              mean_Cy5[index] = 0
              std_Cy5[index] = 10000
              med_Cy5[index] = 0
          else:
              mean_Cy5[index]=getattr(props2[index],'mean_intensity')
              std_Cy5[index]=intensity_Cy5.std()
              med_Cy5[index]=median(intensity_Cy5)
              
          x0[index], y0[index] = getattr(props_mask[index],'centroid')

      # Here we place all of that data into a data frame
      dict={'labels':lbl,'mean_intensity_RFP':mean_RFP,'Std_intensity_RFP':std_RFP,'median_intensity_RFP':med_RFP,'mean_intensity_Cy5':mean_Cy5,'Std_intensity_Cy5':std_Cy5,'median_intensity_Cy5':med_Cy5,'mean_intensity_GFP':mean_GFP,'Std_intensity_GFP':std_GFP,'median_intensity_GFP':med_GFP,'x_center':x0, 'y_center':y0}
      df=pd.DataFrame(dict)
      df['labels']=df['labels'].astype(float)
      df['mean_intensity_RFP']=df['mean_intensity_RFP'].astype(float)
      df['mean_intensity_Cy5']=df['mean_intensity_Cy5'].astype(float)
      df['mean_intensity_GFP']=df['mean_intensity_GFP'].astype(float)
      df['Std_intensity_RFP']=df['Std_intensity_RFP'].astype(float)
      df['Std_intensity_Cy5']=df['Std_intensity_Cy5'].astype(float)
      df['Std_intensity_GFP']=df['Std_intensity_GFP'].astype(float)
      df['median_intensity_RFP']=df['median_intensity_RFP'].astype(float)
      df['median_intensity_Cy5']=df['median_intensity_Cy5'].astype(float)
      df['median_intensity_GFP']=df['median_intensity_GFP'].astype(float)
      df['x_center']=df['x_center'].astype(float)
      df['y_center']=df['y_center'].astype(float)

      ##### Catagorize data into cell types####
      #Calculate the covariance within each roi (I use this later on as a means of comparing the uniformity of intensity within the ROI, real staining should be quite uniform)
      cov_GFP=df['Std_intensity_GFP']/df['mean_intensity_GFP']
      cov_RFP=df['Std_intensity_RFP']/df['mean_intensity_RFP']
      cov_Cy5=df['Std_intensity_Cy5']/df['mean_intensity_Cy5']

      # Create binary mask of image based on the threshold calculated above
      blur_RFP=cv2.GaussianBlur(ins,(5,5),0)
      blur_Cy5=cv2.GaussianBlur(sst,(5,5),0)
      thresh1 = threshold_mean(blur_RFP)
      thresh2 = threshold_mean(blur_Cy5)
      thresh_RFP,binary_RFP=cv2.threshold(blur_RFP,thresh1,1,cv2.THRESH_BINARY)
      thresh_Cy5,binary_Cy5=cv2.threshold(blur_Cy5,thresh2,1,cv2.THRESH_BINARY)

      # post-processing, erosion by one pixel just to clean up the masks
      kernel=np.ones((5,5),np.uint8)
      binary_RFP=cv2.erode(binary_RFP,kernel,iterations=1)
      binary_Cy5=cv2.erode(binary_Cy5,kernel,iterations=1)

      # Use the mask to calculate the area of RFP and Cy5 covering each id_area, used later on as a threshold for potential beta and alpha cells
      props1 = regionprops(labels,binary_RFP.astype(float))
      props2 = regionprops(labels,binary_Cy5.astype(float))

      area_RFP=np.zeros(labels.max())
      area_Cy5=np.zeros(labels.max())

      for index in range(0, len(props)):
          area_RFP[index]=getattr(props1[index],'intensity_mean')
          area_Cy5[index]=getattr(props2[index],'intensity_mean')

      df['area_RFP']=area_RFP
      df['area_Cy5']=area_Cy5
      cell_id=['na']*(len(df['labels']))
      df['cell_type']=cell_id

      # We use these beta and alpha cells to define the distribution of expected RFP and Cy5 signals in real beta and alpha cells and use this to calculate a z-score for all cells in the image
      bcell=(ins*binary_RFP).flatten()
      bcell_med=median(bcell[bcell!=0])
      bcell_MAD=median(np.abs(bcell[bcell!=0]-bcell_med))
      MAD_score_RFP=(df['median_intensity_RFP']-bcell_med)/bcell_MAD

      dcell=(sst*binary_Cy5).flatten()
      dcell_med=median(dcell[dcell!=0])
      dcell_MAD=median(np.abs(dcell[dcell!=0]-dcell_med))
      MAD_score_Cy5=(df['median_intensity_Cy5']-dcell_med)/dcell_MAD

      # First we identify beta and alpha cells that we are certain of
      df.loc[(df['median_intensity_RFP']>=thresh_RFP) & (df['median_intensity_Cy5']<thresh_Cy5) &
                        (area_RFP>=0.8) & (cov_RFP<=1) & ((MAD_score_RFP*1.4826)>=-2.5),'cell_type']='beta'
      df.loc[(df['median_intensity_Cy5']>=thresh_Cy5) & (df['median_intensity_RFP']<thresh_RFP) &
                        (area_Cy5>=0.8) & (cov_Cy5<=1) & ((MAD_score_Cy5*1.4826)>=-2.5),'cell_type']='delta'

      # Now we identify beta and alpha cells that have overlap of Cy5 and RFP, these are cells that are more difficult to calssify
      df.loc[(df['median_intensity_Cy5']>=thresh_Cy5) & (df['median_intensity_RFP']>=thresh_RFP) & (MAD_score_RFP>MAD_score_Cy5) &
                        (area_RFP>=0.8) & ((MAD_score_RFP*1.4826)>=-2.5) & ((MAD_score_Cy5*1.4826)<=-2.5),'cell_type']='beta'
      df.loc[(df['median_intensity_Cy5']>=thresh_Cy5) & (df['median_intensity_RFP']>=thresh_RFP) & (MAD_score_Cy5>MAD_score_RFP) &
                        (area_Cy5>=0.8) & ((MAD_score_Cy5*1.4826)>=-2.5) & ((MAD_score_RFP*1.4826)<=-2.5),'cell_type']='delta'

      #Create mask for gap junctions
      data = {}
      perimeter = Counter(membrane.flatten()[membrane.flatten() != 0])
      for x in np.unique(df['labels'])[np.unique(df['labels']) != 0]:
        data[f'idx_{x.astype(int)}'] = {'Cell_type': df.loc[df['labels']==x]['cell_type'].tolist()[0], 'x_center': df.loc[df['labels']==x]['x_center'].tolist()[0], 'y_center': df.loc[df['labels']==x]['y_center'].tolist()[0], 'membrane_perimeter': perimeter[x], 'GJ_punctae': []}

      for x in range(0, np.shape(mask)[1] - 1):
        for y in range(0, np.shape(mask)[0] - 1):
          if mask[y, x]!=0 and expand_membrane[y, x]!=0 and expand_membrane[y, x] in set(df['labels']):
            data[f'idx_{expand_membrane[y, x]}']['GJ_punctae'].append(mask[y, x]) # If a gap junction sits on a cells membrane then add its index to that cells GJ_punctae nested dictionary
      gj=[]
      for m in np.unique(df['labels'])[np.unique(df['labels']) != 0]:
        data[f'idx_{m.astype(int)}']['GJ_punctae'] = np.unique(data[f'idx_{m.astype(int)}']['GJ_punctae'])
        gj.extend(data[f'idx_{m.astype(int)}']['GJ_punctae'])

      unique, counts = np.unique(gj, return_counts = 'TRUE')
      real_gj = unique[counts>1]
      #np.unique(np.where(counts > 1)[0])

      for m in np.unique(df['labels'])[np.unique(df['labels']) != 0]:
        data[f'idx_{m.astype(int)}']['GJ_punctae'] = list(set(data[f'idx_{m.astype(int)}']['GJ_punctae']) & set(real_gj))


      if not os.path.exists(directory + '/{output}/' + n):
          os.makedirs(directory + r'/{output}/' + n)
          os.makedirs(directory + fr'/{output}/{n}/gap_junction_data')
          os.makedirs(directory + fr'/{output}/{n}/gj_mask')
          os.makedirs(directory + fr'/{output}/{n}/raw_data')
          
      df_data = pd.DataFrame.from_dict(data)
      imsave(directory + fr'/{output}/{n}/gj_mask/' + g[:-7] + r'_gj_mask.tif', mask.astype('uint16'))
      df.to_csv(directory + fr'/{output}/{n}/raw_data/' + g[:-7] + r'.csv')
      df_data.to_csv(directory + fr'/{output}/{n}/gap_junction_data/' + g[:-7] + r'_gap_junction_data.csv')
