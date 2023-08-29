#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:23:13 2022

@author: nicholas.lusk
"""

import os
import dask
import importlib
import skimage.io

import numpy as np
import dask.array as da

from pathlib import Path
from typing import List, Optional, Union

from scipy import ndimage as ndi
from scipy.signal import medfilt2d
from astropy.stats import SigmaClip
from aind_data_schema import Processing
from cellfinder_core.detect import detect
from photutils.background import Background2D
from imlib.IO.cells import get_cells, save_cells

from .pymusica import musica

PathLike = Union[str, Path]

@dask.delayed
def delay_astro(img, estimator="MedianBackground", box=(50, 50), filt=(3, 3), sig_clip=3, pad=0, reflect = 0, amplify = False, smooth=False):
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

    #convert dask array array
    img = np.array(img).astype("uint16")

    if reflect != 0:
        img = np.pad(img, reflect, mode = "reflect")
        
    if pad != 0:
        img = np.pad(img, pad, mode = "constant", constant_values = 0)

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
                    exclude_percentile=100,
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
def delay_preprocess(img, reflect, pad, subtract = False, bkg = None):

    if subtract:
        img = img - bkg
        img = np.clip(img, a_min = 0, a_max = 65535)

    img = np.pad(img, reflect, mode = "reflect")
    img = np.pad(img, pad, mode = "constant", constant_values = 0)
    
    return img

@dask.delayed
def delay_detect(img, save_path, count, offset, padding, process_by, smartspim_config):

    cells = detect.main(
        signal_array=np.asarray(img),
        save_path=save_path,
        block=count,
        offset=offset,
        padding=padding,
        process_by=process_by,
        **smartspim_config
    )

    return cells

@dask.delayed
def delay_postprocess(count, save_path, cells, offset, padding, dims):
    
    bad_cells = []
    for c, cell in enumerate(cells):
        loc = [
            cell.x - padding,
            cell.y - padding,
            cell.z - padding
            ]
        
        if min(loc) < 0 or max([l - (s - 2 * padding) for l, s in zip(loc, dims)]) > 0:
            bad_cells.append(c)
        else:
            cell.x = loc[0] + offset[0]
            cell.y = loc[1] + offset[1]
            cell.z = loc[2] + offset[2]
                    
            if cell.type == -1:
                cell.type = 1
                    
            cells[c] = cell
            
    for c in bad_cells[::-1]:
        cells.pop(c)
    
    
    # save the blocks 
    fname = 'cells_block_' + str(count) + '.xml'
    print(f"Saving cells to path: {fname}")
    save_cells(cells, os.path.join(save_path, fname))
    
    return len(cells)

@dask.delayed
def delay_plane_stats(plane, log_sigma_size, soma_diameter, count):
    plane = plane.squeeze()
    gaussian_sigma = log_sigma_size * soma_diameter
    plane = medfilt2d(plane.astype(np.float64))
    plane = ndi.gaussian_filter(plane, gaussian_sigma)
    plane = ndi.laplace(plane)
    plane = plane * -1
    
    plane = plane - plane.min()
    plane = np.nan_to_num(plane)
    
    if plane.max() != 0:
        maxima = plane.max()
        plane = plane / maxima

    # To leave room to label in the 3d detection.
    plane = plane * (2**16 - 3)

    return np.array([count, maxima, plane.ravel().mean(), plane.ravel().std()])

@dask.delayed
def delay_all(img, reflect, pad, save_path, process_by, stat, offset, dims, count, smartspim_config):
    
    img = np.asarray(img)
    img = np.pad(img, reflect, mode = "reflect")
    img = np.pad(img, pad, mode = "constant", constant_values = 0)

    cells = detect.main(
        signal_array=img,
        save_path=save_path,
        process_by=process_by,
        stats=stat,
        **smartspim_config
    )

    bad_cells = []
    padding = pad + reflect
    for c, cell in enumerate(cells):
        loc = [
            cell.x - padding,
            cell.y - padding,
            cell.z - padding
            ]
        
        if min(loc) < 0 or max([l - (s - 2 * padding) for l, s in zip(loc, dims)]) > 0:
            bad_cells.append(c)
        else:
            cell.x = loc[0] + offset[0]
            cell.y = loc[1] + offset[1]
            cell.z = loc[2] + offset[2]
                    
            if cell.type == -1:
                cell.type = 1
                    
            cells[c] = cell
            
    for c in bad_cells[::-1]:
        cells.pop(c)
    
    
    # save the blocks 
    fname = 'cells_block_' + str(count) + '.xml'
    print(f"Saving cells to path: {fname}")
    save_cells(cells, os.path.join(save_path, fname))
    
    return len(cells)

def generate_processing(
    data_processes: List[dict], dest_processing: PathLike, pipeline_version: str,
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
