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

import dask
import dask.array as da
import numpy as np
import skimage.io
from aind_data_schema import Processing
from astropy.stats import SigmaClip
from cellfinder_core.detect import detect
from photutils.background import Background2D
from scipy import ndimage as ndi
from scipy.signal import medfilt2d

from .pymusica import musica

PathLike = Union[str, Path]


@dask.delayed
def astro_preprocess(
    img,
    estimator="MedianBackground",
    box=(50, 50),
    filt=(3, 3),
    sig_clip=3,
    pad=0,
    reflect=0,
    amplify=False,
    smooth=False,
):
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

    # convert dask array array
    img = np.array(img).astype("uint16")

    if reflect != 0:
        img = np.pad(img, reflect, mode="reflect")

    if pad != 0:
        img = np.pad(img, pad, mode="constant", constant_values=0)

    print(img.shape)

    # get estimator function for background subtraction
    est = getattr(importlib.import_module("photutils.background"), estimator)

    # set parameters for Laplacian pyramid
    L = 5
    params_m = {"M": 1023.0, "a": np.full(L, 11), "p": 0.7}
    backgrounds = []
    for depth in range(img.shape[0]):
        curr_img = img[depth, :, :].copy()

        sigma_clip = SigmaClip(sigma=sig_clip, maxiters=10)

        # check if padding has been added and mask regions accordingly
        if pad > 0:
            if depth >= pad or depth < (img.shape[0] - pad):
                # create boolean mask over padded region
                mask = np.full(curr_img.shape, True)
                mask[pad:-pad, pad:-pad] = False

                # calculate Laplacian Pyramid
                if amplify:
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
            if amplify:
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
        backgrounds.append([bkg.background.mean(), bkg.background.std()])
        del curr_img, bkg

    # smooth over Z plan with gaussian filter
    if smooth:
        img = ndi.gaussian_filter(img, sigma=1.0, mode="constant", cval=0, truncate=2)

    return img


@dask.delayed
def delay_preprocess(img, reflect, pad, subtract, bkg=None):
    if subtract:
        img2 = img - bkg
        img2 = np.clip(img2, a_min=0, a_max=65535)
    else:
        img2 = img

    img2 = np.pad(img2, reflect, mode="reflect")
    img2 = np.pad(img2, pad, mode="constant", constant_values=0)

    return img2


@dask.delayed
def delay_detect(img, save_path, block, offset, padding, process_by, stat, smartspim_config):
    cell_count = detect.main(
        signal_array=np.asarray(img),
        save_path=save_path,
        block=block,
        offset=offset,
        padding=padding,
        process_by=process_by,
        stats=stat,
        **smartspim_config,
    )

    return cell_count


@dask.delayed
def delayed_rechunk(array, by, size=1024):
    if by == "plane":
        rechunk_size = tuple([1, array.shape[1], array.shape[2]])
    elif by == "cube":
        rechunk_size = [axis * (self.args["chunk_size"] // axis) for axis in signal_array.chunksize]
        rechunk_size = tuple(rechunk_size)

    return array.rechunk(rechunk_size)


@dask.delayed
def delayed_plane_stats(plane, log_sigma_size, soma_diameter, count):
    plane = np.asarray(plane).squeeze()
    gaussian_sigma = log_sigma_size * soma_diameter
    filtered_img = medfilt2d(plane.astype(np.float64))
    filtered_img = ndi.gaussian_filter(filtered_img, gaussian_sigma)
    filtered_img = ndi.laplace(filtered_img)
    filtered_img = filtered_img * -1

    filtered_img = filtered_img - filtered_img.min()
    filtered_img = np.nan_to_num(filtered_img)

    if filtered_img.max() != 0:
        filtered_img = filtered_img / filtered_img.max()

    # To leave room to label in the 3d detection.
    out_img = filtered_img * (2**16 - 3)

    return np.array(
        [count, filtered_img.ravel().max(), out_img.ravel().mean(), out_img.ravel().std()]
    )


def reg_detect(img, save_path, block, offset, padding, smartspim_config):
    cell_count = detect.main(
        signal_array=img,
        save_path=save_path,
        block=block,
        offset=offset,
        padding=padding,
        **smartspim_config,
    )

    return cell_count


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
