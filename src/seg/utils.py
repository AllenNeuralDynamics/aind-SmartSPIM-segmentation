#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:23:13 2022

@author: nicholas.lusk
"""

import importlib

import numpy as np

from .pymusica import musica
from scipy import ndimage as ndi
from astropy.stats import SigmaClip
from photutils.background import Background2D


# preprocessing function to standardize the image stack for training networks
# mainly taken from https://photutils.readthedocs.io/en/stable/background.html
# modified to deal with 3D signal and more complex background
def astro_preprocess(img, estimator, box = (50, 50), filt = (3, 3), sig_clip = 3, pad = 0, smooth = False):
    
    bkg_sub_array = np.zeros(img.shape)
    est = getattr(importlib.import_module('photutils.background'), estimator)
    
    L = 7
    params_m = {'M': 1023.0, 'a': np.full(L, 11), 'p': 0.7}
    
    for depth in range(img.shape[0]):
        
        curr_img = img[depth, :, :]
        sigma_clip = SigmaClip(sigma = sig_clip, maxiters = 10)
        # check if padding has been added and mask regions accordingly
        if pad > 0:
            if depth >= pad or depth < (img.shape[0] - pad):

                mask = np.full(curr_img.shape, True)
                mask[pad:-pad, pad:-pad] = False
                
                curr_img[pad:-pad, pad:-pad] = musica(curr_img[pad:-pad, pad:-pad], L, params_m)
                bkg = Background2D(curr_img, box_size = box, filter_size = filt, 
                                   bkg_estimator = est(), fill_value=0.0, 
                                   sigma_clip=sigma_clip, coverage_mask = mask, 
                                   exclude_percentile = 50)
                     
        else:
            curr_img = musica(curr_img, L, params_m)
            bkg = Background2D(curr_img, box_size = box, filter_size = filt, 
                               bkg_estimator = est(), fill_value=0.0)
         
        curr_img = curr_img - bkg.background
        curr_img = np.where(curr_img < 0, 0, curr_img)    
            
        bkg_sub_array[depth, :, :] = curr_img
            
    
    if smooth:
        bkg_sub_array = ndi.gaussian_filter(bkg_sub_array, sigma = 1.0, mode = 'constant', cval = 0, truncate = 2)
        
    return bkg_sub_array



