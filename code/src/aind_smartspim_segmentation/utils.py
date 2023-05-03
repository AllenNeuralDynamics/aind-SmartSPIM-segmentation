#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:23:13 2022

@author: nicholas.lusk
"""
import importlib
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from aind_data_schema import Processing
from astropy.stats import SigmaClip
from photutils.background import Background2D
from scipy import ndimage as ndi

from .pymusica import musica

PathLike = Union[str, Path]


def astro_preprocess(img, estimator, box=(50, 50), filt=(3, 3), sig_clip=3, pad=0, smooth=False):
    """
    Preprocessing function to standardize the image stack for training networks. Combines a Laplacian Pyramid
    for contrast enhancement and statistical background subtraction

    Parameters
    ----------
    img : array
        Dask array of lightsheet data. the first dimension should be the Z plane
    estimator : str
        one of the background options provided by https://photutils.readthedocs.io/en/stable/background.html
    box : tuple, optional
        dimensions in pixels for each tile for background subtraction. The default is (50, 50).
    filt : tuple, optional
        filter size for estimator within each tile. The default is (3, 3).
    sig_clip : int, optional
        standard deviation above the mean for masking pixel for background subtraction. The default is 3.
    pad : int, optional
        number of pixels of padding applied to image. usually only apllies if running on subsection. The default is 0.
    smooth : boolean, optional
        whether to apply 3D gaussian smoothing over array. The default is False.

    Returns
    -------
    bkg_sub_array : array
        dask array with preprocessed images

    """

    # get estimator function for background subtraction
    est = getattr(importlib.import_module("photutils.background"), estimator)

    # set parameters for Laplacian pyramid
    L = 7
    params_m = {"M": 1023.0, "a": np.full(L, 11), "p": 0.7}

    # loop through z plane and run preprocessing
    for depth in range(img.shape[0]):
        curr_img = np.array(img[depth, :, :])
        sigma_clip = SigmaClip(sigma=sig_clip, maxiters=10)

        # check if padding has been added and mask regions accordingly
        if pad > 0:
            if depth >= pad or depth < (img.shape[0] - pad):
                # create boolean mask over padded region
                mask = np.full(curr_img.shape, True)
                mask[pad:-pad, pad:-pad] = False

                # calculate Laplacian Pyramid
                curr_img[pad:-pad, pad:-pad] = musica(curr_img[pad:-pad, pad:-pad], L, params_m)

                # get background statistics
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

        else:
            # calculate Laplacian Pyramid
            curr_img = musica(curr_img, L, params_m)

            # get background statistics
            bkg = Background2D(
                curr_img,
                box_size=box,
                filter_size=filt,
                bkg_estimator=est(),
                fill_value=0.0,
                sigma_clip=sigma_clip,
                exclude_percentile=50,
            )

        # subtract background and clip min to 0
        curr_img = curr_img - bkg.background
        curr_img = np.where(curr_img < 0, 0, curr_img)

        img[depth, :, :] = curr_img.astype("uint16")

    # smooth over Z plan with gaussian filter
    if smooth:
        img = ndi.gaussian_filter(img, sigma=1.0, mode="constant", cval=0, truncate=2)

    return img


def generate_processing(
    data_processes: List[dict],
    dest_processing: PathLike,
    pipeline_version: str,
) -> None:
    """
    Generates data description for the output folder.
    Parameters
    ------------------------
    data_processes: List[dict]
        List with the processes aplied in the pipeline.
    dest_processing: PathLike
        Path where the processing file will be placed.
    pipeline_version: str
        Terastitcher pipeline version
    """

    # flake8: noqa: E501
    processing = Processing(
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-SmartSPIM-segmentation",
        pipeline_version=pipeline_version,
        data_processes=data_processes,
    )

    with open(dest_processing, "w") as f:
        f.write(processing.json(indent=3))


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.
    Parameters
    ------------------------
    dest_dir: PathLike
        Path where the folder will be created if it does not exist.
    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.
    Raises
    ------------------------
    OSError:
        if the folder exists.
    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
