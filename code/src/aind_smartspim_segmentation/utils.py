#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:23:13 2022

@author: nicholas.lusk
"""

import importlib

import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D
from scipy import ndimage as ndi


# preprocessing function to standardize the image stack for training networks
# mainly taken from https://photutils.readthedocs.io/en/stable/background.html
# modified to deal with 3D signal and more complex background
def bkg_transforms(img, bkg, sigma):

    curr_norm = img - bkg.background
    curr_norm = ndi.gaussian_filter(
        curr_norm, sigma=1.0, mode="constant", cval=0, truncate=2
    )
    thresh = np.mean(curr_norm[curr_norm > 0]) + sigma * np.std(
        curr_norm[curr_norm > 0]
    )
    curr_thresh = np.where(curr_norm > thresh, curr_norm, 0)
    curr_scaled = np.where(curr_thresh > 0, img, 0)
    curr_scaled = ndi.gaussian_filter(
        curr_scaled, sigma=1.0, mode="constant", cval=0, truncate=2
    )
    return curr_scaled


def astro_preprocess(
    img,
    estimator,
    box=(50, 50),
    filt=(3, 3),
    sigma=3,
    sig_clip=3,
    pad=0,
    smooth=True,
):

    bkg_sub_array = np.zeros(img.shape)
    est = getattr(importlib.import_module("photutils.background"), estimator)

    for depth in range(img.shape[0]):

        curr_img = img[depth, :, :]
        sigma_clip = SigmaClip(sigma=sig_clip, maxiters=10)
        # check if padding has been added and mask regions accordingly
        if pad > 0:
            if depth >= pad or depth < (img.shape[0] - pad):

                mask = np.full(curr_img.shape, True)
                mask[pad:-pad, pad:-pad] = False

                bkg = Background2D(
                    curr_img,
                    box_size=box,
                    filter_size=filt,
                    bkg_estimator=est(),
                    fill_value=0.0,
                    sigma_clip=sigma_clip,
                    coverage_mask=mask,
                    exclude_percentile=50,
                )

                bkg_sub_array[depth, :, :] = bkg_transforms(
                    curr_img, bkg, sigma
                )
        else:

            bkg = Background2D(
                curr_img,
                box_size=box,
                filter_size=filt,
                bkg_estimator=est(),
                fill_value=0.0,
            )

            bkg_sub_array[depth, :, :] = bkg_transforms(curr_img, bkg, sigma)

    if smooth:
        bkg_sub_array = ndi.gaussian_filter(
            bkg_sub_array, sigma=1.0, mode="constant", cval=0, truncate=2
        )

    return bkg_sub_array
