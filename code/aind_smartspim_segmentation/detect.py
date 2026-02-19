#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 00:33:10 2025

@author: nicholas.lusk
"""

import logging
import multiprocessing
import os
import warnings

from time import time
from typing import Dict, Optional, Tuple, Set, Any

import cupy
import cupyx.scipy.ndimage as ndi
import dask.array as da
import numpy as np
import pandas as pd
import psutil
import torch
from aind_data_schema.core.processing import DataProcess, ProcessName
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    concatenate_lazy_data,
    recover_global_position,
    unpad_global_coords,
)
from aind_large_scale_prediction.io import ImageReaderFactory
from aind_smartspim_segmentation._shared.types import ArrayLike, PathLike
from pathos.pools import _ProcessPool
from scipy.ndimage import (
    gaussian_filter, 
    binary_opening, 
    binary_fill_holes, 
    binary_closing, 
    label,
)


from .__init__ import __maintainers__, __pipeline_version__, __version__
from .traditional_detection.puncta_detection import prune_blobs, traditional_3D_spot_detection
from .utils import utils


def calculate_mask_cpu(image: ArrayLike) -> Tuple[np.ndarray, float]:
    """
    Creates a binary mask using max entropy (triangle) method on CPU.
    CPU version to avoid GPU memory issues during mask creation.
    
    Parameters
    ----------
    image: ArrayLike
        Input volume
    
    Returns
    -------
    np.ndarray
        Binary mask (uint8)
    float
        Percent of volume covered by mask
    """
    
    vol = np.asarray(image.squeeze().astype(np.float32))
    smoothed = gaussian_filter(vol, sigma=1.0)
    
    non_zero = smoothed[smoothed > 0]
    
    # Maximum entropy method
    hist, bin_edges = np.histogram(non_zero, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peak of histogram (background peak)
    peak_idx = int(np.argmax(hist))
    
    # Find last significant bin (tissue tail)
    cumsum = np.cumsum(hist[::-1])[::-1]
    last_idx = int(np.where(cumsum > cumsum[0] * 0.01)[0][-1])
    
    # Triangle method: draw line from peak to last bin, find max distance
    if last_idx > peak_idx:
        x1, y1 = peak_idx, float(hist[peak_idx])
        x2, y2 = last_idx, float(hist[last_idx])
        
        # Distance from line to histogram (vectorized)
        indices = np.arange(peak_idx, last_idx)
        hist_segment = hist[peak_idx:last_idx]
        
        # Vectorized distance calculation
        numerator = np.abs((y2-y1)*(indices-x1) - (x2-x1)*(hist_segment-y1))
        denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        distances = numerator / denominator
        
        threshold_idx = peak_idx + int(np.argmax(distances))
        threshold = float(bin_centers[threshold_idx])
    else:
        # Fallback to 70th percentile if distribution is unusual
        threshold = float(np.percentile(non_zero, 70))
    
    binary = vol > threshold
    
    struct_3d = np.ones((7, 7, 7), dtype=bool)
    opened = binary_opening(binary, structure=struct_3d, iterations=3)
    filled = binary_fill_holes(opened)
    
    # Largest component
    labeled, num_features = label(filled)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0
        largest_idx = np.argmax(component_sizes)
        mask = (labeled == largest_idx)
    else:
        mask = filled
    
    # Smooth
    mask = binary_closing(mask, structure=np.ones((3, 3, 3), dtype=bool))
    mask_cpu = mask.astype(np.uint8)
    
    coverage = np.sum(mask_cpu > 0) / binary.size * 100
   
    return mask_cpu, coverage


def create_downsampled_mask(
    zarr_path: PathLike,
    downsample_factor: int,
    logger: logging.Logger = None,
) -> np.ndarray:
    """
    Creates a downsampled segmentation mask from the full volume.
    Uses CPU to avoid GPU memory issues.
    
    Parameters
    ----------
    zarr_path: PathLike
        Path to the zarr dataset
    downsample_factor: int
        Zarr level to use for mask creation
    logger: logging.Logger
        Logger object
    
    Returns
    -------
    np.ndarray
        Downsampled binary mask (uint8)
    """
    if logger:
        logger.info(f"Creating downsampled mask with Zarr level {downsample_factor}")
    
    downsampled_volume = da.from_zarr(zarr_path, str(downsample_factor)).squeeze()
    
    if isinstance(downsampled_volume, da.Array):
        downsampled_volume = downsampled_volume.compute()
    
    if logger:
        logger.info(f"Loaded downsampled volume: {downsampled_volume.shape}")
    
    mask, mask_coverage = calculate_mask_cpu(downsampled_volume)
    
    if logger:
        logger.info(f"Downsampled mask shape: {mask.shape}, "
                   f"coverage: {mask_coverage:.2f}%")
    
    return mask


def compute_block_tissue_map(
    downsampled_mask: np.ndarray,
    original_volume_shape: Tuple[int, ...],
    super_chunk_size: Tuple[int, ...],
    prediction_chunksize: Tuple[int, ...],
    overlap_chunksize: Tuple[int, ...],
    min_coverage_percent: float = 0.0,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """
    Pre-compute which batches have tissue by scanning the mask once.
    
    Returns a lookup table for fast filtering during iteration.
    
    Parameters
    ----------
    downsampled_mask : np.ndarray
        Binary mask from create_downsampled_mask
    original_volume_shape : Tuple[int, ...]
        Original volume shape (Z, Y, X)
    super_chunk_size : Tuple[int, ...]
        Super chunk dimensions
    prediction_chunksize : Tuple[int, ...]
        Batch prediction chunk size
    overlap_chunksize : Tuple[int, ...]
        Overlap padding
    min_coverage_percent : float
        Minimum tissue coverage to process block
    logger : logging.Logger
        For progress logging
        
    Returns
    -------
    Dict with tissue block information
    """
    
    if logger:
        logger.info("Pre-computing tissue block map...")
        logger.info(f"  Mask shape: {downsampled_mask.shape}")
        logger.info(f"  Volume shape: {original_volume_shape}")
        logger.info(f"  Super chunk size: {super_chunk_size}")
        logger.info(f"  Prediction chunk size: {prediction_chunksize}")
        logger.info(f"  Min coverage: {min_coverage_percent}%")
    
    # Calculate downsample factors
    downsample_factors = np.array(original_volume_shape) / np.array(downsampled_mask.shape)
    
    # Effective batch size (prediction + overlap)
    batch_size = tuple(np.array(prediction_chunksize) + np.array(overlap_chunksize))
    
    # Set to store batch identifiers with tissue
    batches_with_tissue = set()
    
    # Count totals
    total_batches_scanned = 0
    
    # Iterate through all possible batch positions
    for sz in range(0, original_volume_shape[0], prediction_chunksize[0]):
        for sy in range(0, original_volume_shape[1], prediction_chunksize[1]):
            for sx in range(0, original_volume_shape[2], prediction_chunksize[2]):
                total_batches_scanned += 1
                
                # Batch bounds in global coordinates
                batch_start = (sz, sy, sx)
                batch_end = (
                    min(sz + batch_size[0], original_volume_shape[0]),
                    min(sy + batch_size[1], original_volume_shape[1]),
                    min(sx + batch_size[2], original_volume_shape[2])
                )
                
                # Skip if entirely outside volume
                if (batch_start[0] >= original_volume_shape[0] or
                    batch_start[1] >= original_volume_shape[1] or
                    batch_start[2] >= original_volume_shape[2]):
                    continue
                
                # Map to downsampled coordinates
                ds_start = tuple(
                    int(np.floor(batch_start[i] / downsample_factors[i]))
                    for i in range(3)
                )
                ds_end = tuple(
                    int(np.ceil(batch_end[i] / downsample_factors[i]))
                    for i in range(3)
                )
                
                # Clamp to mask bounds
                ds_start = tuple(
                    max(0, min(ds_start[i], downsampled_mask.shape[i]))
                    for i in range(3)
                )
                ds_end = tuple(
                    max(0, min(ds_end[i], downsampled_mask.shape[i]))
                    for i in range(3)
                )
                
                # Extract mask region
                mask_region = downsampled_mask[
                    ds_start[0]:ds_end[0],
                    ds_start[1]:ds_end[1],
                    ds_start[2]:ds_end[2]
                ]
                
                if mask_region.size == 0:
                    continue
                
                tissue_voxels = np.sum(mask_region > 0)
                
                if min_coverage_percent == 0.0:
                    # Special case: skip only completely empty blocks
                    has_tissue = (tissue_voxels > 0)
                else:
                    # Use percentage threshold
                    coverage_percent = (tissue_voxels / mask_region.size) * 100
                    has_tissue = (coverage_percent >= min_coverage_percent)
                
                if has_tissue:
                    batches_with_tissue.add(batch_start)
                
                # Log progress every 10000 batches
                if total_batches_scanned % 10000 == 0 and logger:
                    logger.info(f"  Scanned {total_batches_scanned} batches, "
                              f"found {len(batches_with_tissue)} with tissue")
    
    tissue_map = {
        'batches_with_tissue': batches_with_tissue,
        'total_batches_scanned': total_batches_scanned,
        'tissue_batches': len(batches_with_tissue),
    }
    
    if logger:
        skip_rate = (1 - len(batches_with_tissue) / max(1, total_batches_scanned)) * 100
        speedup = total_batches_scanned / max(1, len(batches_with_tissue))
        
        logger.info("="*70)
        logger.info("Tissue block map computed:")
        logger.info(f"  Total batches scanned: {total_batches_scanned}")
        logger.info(f"  Batches with tissue: {len(batches_with_tissue)}")
        logger.info(f"  Batches to skip: {total_batches_scanned - len(batches_with_tissue)}")
        logger.info(f"  Skip rate: {skip_rate:.1f}%")
        logger.info(f"  Expected speedup: {speedup:.2f}x")
        logger.info("="*70)
    
    return tissue_map


def batch_identifier_from_sample(
    batch_super_chunk: Tuple[slice, ...],
    batch_internal_slice: Tuple[slice, ...]
) -> Tuple[int, int, int]:
    """
    Extract batch global position as identifier for lookup.
    
    Returns
    -------
    Tuple[int, int, int]
        (z_start, y_start, x_start) in global coordinates
    """
    # Get super chunk start
    if isinstance(batch_super_chunk[0], slice):
        super_start_z = batch_super_chunk[0].start
        super_start_y = batch_super_chunk[1].start
        super_start_x = batch_super_chunk[2].start
    else:
        super_start_z, super_start_y, super_start_x = batch_super_chunk[:3]
    
    # Get internal start
    if isinstance(batch_internal_slice[0], slice):
        internal_start_z = batch_internal_slice[0].start
        internal_start_y = batch_internal_slice[1].start
        internal_start_x = batch_internal_slice[2].start
    else:
        raise ValueError("batch_internal_slice should contain slices")
    
    # Global position
    return (
        super_start_z + internal_start_z,
        super_start_y + internal_start_y,
        super_start_x + internal_start_x
    )

def apply_mask(data: ArrayLike, mask: ArrayLike = None) -> ArrayLike:
    """
    Applies the mask to the current data.

    Parameters
    ----------
    data: ArrayLike
        Data to mask.
    mask: ArrayLike
        Segmentation mask.

    Returns
    -------
    ArrayLike
        Data after applying masking
    """

    if mask is None:
        return data

    orig_dtype = data.dtype

    mask[mask > 0] = 1
    if isinstance(mask, torch.Tensor):
        mask = mask.to(torch.uint8)
    else:
        mask = mask.astype(np.uint8)

    data = data * mask

    if isinstance(data, torch.Tensor):
        data = data.to(orig_dtype)
    else:
        data = data.astype(orig_dtype)

    return data


def remove_points_in_pad_area(points: ArrayLike, unpadded_slices: Tuple[slice]) -> ArrayLike:
    """
    Removes points in padding area.

    Parameters
    ----------
    points: ArrayLike
        3D points in the chunk of data.
    unpadded_slices: Tuple[slice]
        Slices that point to the non-overlapping area between chunks.

    Returns
    -------
    ArrayLike
        Points within the non-overlapping area.
    """

    unpadded_points = points[
        (points[:, 0] >= unpadded_slices[0].start)
        & (points[:, 0] <= unpadded_slices[0].stop)
        & (points[:, 1] >= unpadded_slices[1].start)
        & (points[:, 1] <= unpadded_slices[1].stop)
        & (points[:, 2] >= unpadded_slices[2].start)
        & (points[:, 2] <= unpadded_slices[2].stop)
    ]

    return unpadded_points


def execute_worker(
    data: ArrayLike,
    batch_super_chunk: Tuple[slice],
    batch_internal_slice: Tuple[slice],
    spot_parameters: Dict,
    overlap_prediction_chunksize: Tuple[int],
    dataset_shape: Tuple[int],
    logger: logging.Logger,
) -> np.array:
    """
    Function that executes each worker for spot detection.

    Parameters
    ----------
    data: ArrayLike
        Data to process.
    batch_super_chunk: Tuple[slice]
        Slices of the super chunk loaded in shared memory.
    batch_internal_slice: Tuple[slice]
        Internal slice of the current chunk of data.
    spot_parameters: Dict
        Spot detection parameters.
    logger: logging.Logger
        Logging object

    Returns
    -------
    np.array
        Array with the global location of the identified points.
    """
    global_worker_spots = None
    curr_pid = os.getpid()

    mask = None
    # (Batch, channels, Z, Y, X)
    if len(data.shape) == 5 and data.shape[-4] == 2:
        mask = data[:, 1, ...]
        data = data[:, 0, ...]
        data = apply_mask(data=data, mask=mask.detach().clone())

    data_block_cupy = cupy.asarray(data)

    # Processing batch
    for batch_idx in range(0, data.shape[0]):
        curr_block = cupy.squeeze(data_block_cupy[batch_idx, ...])
        message = (
            f"Worker [{curr_pid}] Processing inner batch {batch_idx} out of {data.shape[0]}"
            f"- Data shape: {data.shape} - Current block: {curr_block.shape}"
        )
        logger.info(message)

        # Spot detection
        spots = traditional_3D_spot_detection(
            data_block=curr_block,
            background_percentage=spot_parameters["background_percentage"],
            sigma_zyx=spot_parameters["sigma_zyx"],
            min_zyx=spot_parameters["min_zyx"],
            filt_thresh=spot_parameters["filt_thresh"],
            raw_thresh=spot_parameters["raw_thresh"],
            context_radius=spot_parameters["context_radius"],
            radius_confidence=spot_parameters["radius_confidence"],
            logger=logger,
        )

        # Adding spots to current batch list
        curr_spots = None
        if spots is None:
            logger.info(f"Worker [{curr_pid}] - No spots found in inner batch {batch_idx}")


        if spots is not None and len(spots) > 0:
            # Recover global position of internal chunk
            (
                global_coord_pos,
                global_coord_positions_start,
                global_coord_positions_end,
            ) = recover_global_position(
                super_chunk_slice=batch_super_chunk,
                internal_slices=batch_internal_slice,
            )

            unpadded_global_slice, unpadded_local_slice = unpad_global_coords(
                global_coord_pos=global_coord_pos[-3:],
                block_shape=curr_block.shape[-3:],
                overlap_prediction_chunksize=overlap_prediction_chunksize[-3:],
                dataset_shape=dataset_shape[-3:],  # zarr_dataset.lazy_data.shape,
            )
            
            if mask is not None:
                # Getting spots IDs, adding mask ID to the spot as extra value at the end
                mask = torch.squeeze(mask)
                mask_ids = np.expand_dims(mask[spots[:, 0], spots[:, 1], spots[:, 2]], axis=0)
                spots = np.append(spots.T, mask_ids, axis=0).T
                
            curr_spots = spots.copy().astype(np.float32)
            # Converting to global coordinates, only to ZYX position, leaving mask ID if exists
            curr_spots[:, :3] = np.array(global_coord_positions_start)[:, -3:] + np.array(
                spots[:, :3]
            )

            # Removing points within pad area
            curr_spots = remove_points_in_pad_area(
                points=curr_spots, unpadded_slices=unpadded_global_slice
            )

            message = (
                f"Worker {curr_pid}: Found {len(curr_spots)} spots for in inner batch {batch_idx}"
                f"- Internal pos: {batch_internal_slice} - Global coords: {global_coord_pos}"
                f"- upadded global coords: {unpadded_global_slice}"
            )
            logger.info(message)

            # Adding spots to the worker batch
            if global_worker_spots is None:
                global_worker_spots = curr_spots.copy()
                
            else:
                global_worker_spots = np.append(
                    global_worker_spots,
                    curr_spots,
                    axis=0,
                )

    return global_worker_spots


def _execute_worker(params: Dict):
    """
    Worker interface to provide parameters

    Parameters
    ----------
    params: Dict
        Dictionary with the parameters to provide
        to the execution function.
    """
    return execute_worker(**params)


def has_enough_gpu_memory(
    cupy_device,
    num_blocks: int,
    block_shape: tuple,
    dtype: np.dtype,
    usage_fraction: float = 0.8,
) -> Tuple[bool, float]:
    """
    Check if the GPU has enough memory to hold `num_blocks` of a given block shape and dtype.

    Parameters:
        num_blocks (int): Number of blocks.
        block_shape (tuple): Shape of a single block (e.g., (64, 128, 128)).
        dtype (np.dtype): Numpy dtype (e.g., np.float32).
        usage_fraction (float): Fraction of total memory allowed to be used.

    Returns:
        bool: True if enough memory is available, False otherwise.
        float: memory info
    """
    try:
        _, total_memory = cupy_device.mem_info
        target_memory = int(float(total_memory) * usage_fraction)

        block_size_bytes = np.prod(block_shape) * np.dtype(dtype).itemsize
        total_required_memory = num_blocks * block_size_bytes

    except cupy.cuda.runtime.CUDARuntimeError as e:
        print(f"[GPU ERROR] CuPy could not access the device: {e}")
        return False, 0.0

    return total_required_memory <= target_memory, float(total_memory)

def smartspim_cell_detection(
    dataset_path: PathLike,
    name: str,
    multiscale: str,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    axis_pad: int,
    batch_size: int,
    output_folder: str,
    metadata_path: str,
    spot_parameters: Dict,
    logger: logging.Logger,
    super_chunksize: Optional[Tuple[int, ...]] = None,
    segmentation_mask_path: Optional[PathLike] = None,
    create_mask_from_downsampled: bool = False,
    downsample_factor: Optional[int] = None,
    min_tissue_coverage: float = 5.0,
) -> pd.DataFrame:
    """
    Chunked puncta detection

    Parameters
    ----------
    dataset_path: PathLike
        Path where the zarr dataset is stored. It could
        be a local path or in a S3 path.

    name: str
        name of the dataset formated as SmartSPIM_***_stitched_***

    multiscale: str
        Multiscale to process

    prediction_chunksize: Tuple[int, ...]
        Prediction chunksize the model will pull from
        the raw data

    target_size_mb: int
        Target size in megabytes the data loader will
        load in memory at a time

    n_workers: int
        Number of workers that will concurrently pull
        data from the shared super chunk in memory

    batch_size: int
        Batch size processed each time

    output_folder: str
        Output folder for the detected spots

    logger: logging.Logger
        Logging object

    super_chunksize: Optional[Tuple[int, ...]]
        Super chunk size that will be in memory at a
        time from the raw data. If provided, then
        target_size_mb is ignored. Default: None

    segmentation_mask_path: Optional[PathLike]
        Path where the segmentation mask is stored. It could
        be a local path or in a S3 path.
        Default None
        
    create_mask_from_downsampled : bool
        If True, create a mask from downsampled data for filtering
    
    downsample_factor : Optional[int]
        Zarr level to use for mask creation (e.g., 3)
        
    min_tissue_coverage : float
        Minimum % of tissue in a block to process it (default 5%)
        

    Returns
    -------
    str, List[float]
        Path where the CSV with the idenfied proposals is stored.
        List with the voxel size in ZYX order
    """
    available_cpus = int(utils.get_cpu_limit())
    data_processes = []

    if n_workers > available_cpus:
        raise ValueError(f"Provided workers {n_workers} > current workers {available_cpus}")

    logger.info(f"{20*'='} Running cell proposal detection {20*'='} New workers pull")
    logger.info(f"Output folder: {output_folder}")

    utils.print_system_information(logger)

    logger.info(f"Processing dataset {dataset_path} with mulsticale {multiscale}")
    logger.info(f"Using {available_cpus} workers...")

    # Tracking compute resources
    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    logger.info("Creating chunked data loader")
    shm_memory = psutil.virtual_memory()
    logger.info(f"Shared memory information: {shm_memory}")

    device = None

    pin_memory = True
    if device is not None:
        pin_memory = False
        multiprocessing.set_start_method("spawn", force=True)

    start_date_time = time()

    overlap_prediction_chunksize = (axis_pad, axis_pad, axis_pad)
    if segmentation_mask_path:
        logger.info(f"Using segmentation mask in {segmentation_mask_path}")
        lazy_data = concatenate_lazy_data(
            dataset_paths=[dataset_path, segmentation_mask_path],
            multiscales=[multiscale, "0"],
            concat_axis=-4,
        )
        overlap_prediction_chunksize = (0, axis_pad, axis_pad, axis_pad)
        prediction_chunksize = (lazy_data.shape[-4],) + prediction_chunksize

        message = (
            f"Segmentation mask provided! New prediction chunksize: {prediction_chunksize}"
            f" - New overlap: {overlap_prediction_chunksize}"
        )
        logger.info(message)

    else:
        # No segmentation mask
        lazy_data = (
            ImageReaderFactory()
            .create(data_path=dataset_path, parse_path=False, multiscale=multiscale)
            .as_dask_array()
        )
        
    
    # Store original shape
    original_lazy_data_shape = lazy_data.shape
    if len(original_lazy_data_shape) == 5:
        original_volume_shape = original_lazy_data_shape[-3:]
    elif len(original_lazy_data_shape) == 3:
        original_volume_shape = original_lazy_data_shape
    else:
        raise ValueError(f"Unexpected shape: {original_lazy_data_shape}")

    image_metadata = (
        ImageReaderFactory()
        .create(data_path=dataset_path, parse_path=False, multiscale=multiscale)
        .metadata()
    )

    logger.info(f"Full image metadata: {image_metadata}")
    
    image_metadata = utils.parse_zarr_metadata(metadata=image_metadata, multiscale=multiscale)
    
    logger.info(f"Filtered Image metadata: {image_metadata}")
    end_date_time = time()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_IMPORTING,
            software_version=__version__,
            start_date_time=start_date_time,
            end_date_time=end_date_time,
            input_location=str(dataset_path),
            output_location=str(dataset_path),
            outputs={},
            code_url="https://github.com/AllenNeuralDynamics/aind-SmartSPIM-segmentation",
            code_version=__version__,
            parameters={},
            notes="Importing fused data for cell proposal detection.",
        )
    )

    zarr_data_loader, zarr_dataset = create_data_loader(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        prediction_chunksize=tuple(prediction_chunksize),
        overlap_prediction_chunksize=tuple([axis_pad] * 3),
        n_workers=n_workers,
        batch_size=batch_size,
        dtype=np.float32,  # Allowed data type to process with pytorch cuda
        super_chunksize=super_chunksize,
        lazy_callback_fn=None, # partial_lazy_deskewing,
        logger=logger,
        device=device,
        pin_memory=pin_memory,
        override_suggested_cpus=False,
        drop_last=True,
        locked_array=False,
    )

    overlap_prediction_chunksize = tuple([axis_pad] * 3)
    
    # Get actual super chunk size
    if hasattr(zarr_dataset, 'super_chunk_size'):
        actual_super_chunk_size = zarr_dataset.super_chunk_size
    else:
        actual_super_chunk_size = super_chunksize or (1024, 1152, 512)
    
    logger.info(f"Super chunk size: {actual_super_chunk_size}")

    tissue_map = None
    
    if create_mask_from_downsampled:
        if downsample_factor is None:
            raise ValueError(
                "downsample_factor must be provided when create_mask_from_downsampled=True"
            )
        
        logger.info("="*70)
        logger.info("Creating tissue filtering mask...")
        logger.info("="*70)
        
        # Create downsampled mask (CPU-based to avoid GPU memory issues)
        downsampled_mask = create_downsampled_mask(
            zarr_path=dataset_path,
            downsample_factor=downsample_factor,
            logger=logger,
        )
        
        # Pre-compute which blocks have tissue
        tissue_map = compute_block_tissue_map(
            downsampled_mask=downsampled_mask,
            original_volume_shape=original_volume_shape,
            super_chunk_size=actual_super_chunk_size,
            prediction_chunksize=prediction_chunksize,
            overlap_chunksize=overlap_prediction_chunksize,
            min_coverage_percent=min_tissue_coverage,
            logger=logger,
        )
    
    message = (
        f"Running puncta detection in chunked data. Prediction chunksize: {prediction_chunksize}"
        f"- Overlap chunksize: {overlap_prediction_chunksize}"
    )
    logger.info(message)

    start_time = time()

    total_batches = sum(zarr_dataset.internal_slice_sum) / batch_size
    
    samples_per_iter = n_workers * batch_size
    logger.info(f"Number of batches: {total_batches}")
    
    # Tracking statistics
    batches_scanned = 0
    batches_processed = 0
    batches_skipped = 0
    
    if tissue_map:
        logger.info(f"Will process {tissue_map['tissue_batches']} batches with tissue")
        logger.info(f"Will skip ~{tissue_map['total_batches_scanned'] - tissue_map['tissue_batches']} empty batches")
    
    spots_global_coordinate = None

    # Setting exec workers to available CPUs
    exec_n_workers = available_cpus

    # Create a pool of processes
    pool = _ProcessPool(exec_n_workers)

    # Variables for multiprocessing
    picked_blocks = []
    curr_picked_blocks = 0

    logger.info(f"Number of workers processing data: {exec_n_workers}")

    with cupy.cuda.Device(device=device) as cupy_device:
        workers_gpus_valid, gpu_mem_info = has_enough_gpu_memory(
            num_blocks=available_cpus,
            block_shape=tuple(np.array(prediction_chunksize) + np.array(overlap_prediction_chunksize)),
            dtype=np.float32,
            usage_fraction=0.8,
            cupy_device=cupy_device,
        )

        logger.info(f"GPU available information: {gpu_mem_info}")

        if not workers_gpus_valid:
            raise ValueError(
                f"Not enough memory for {available_cpus} workers with 80% usage factor!"
            )

        with cupy.cuda.Stream.null:
            for i, sample in enumerate(zarr_data_loader):
                batches_scanned += 1
                
                if tissue_map is not None:
                    batch_id = batch_identifier_from_sample(
                        sample.batch_super_chunk[0],
                        sample.batch_internal_slice
                    )
                    
                    if batch_id not in tissue_map['batches_with_tissue']:
                        batches_skipped += 1                        
                        continue
                
                batches_processed += 1
                
                message = (
                    f"Batch {i}: {sample.batch_tensor.shape} - "
                    f"Pinned?: {sample.batch_tensor.is_pinned()} - "
                    f"dtype: {sample.batch_tensor.dtype} - device: {sample.batch_tensor.device}"
                )
                logger.info(message)

                picked_blocks.append({
                    "data": sample.batch_tensor,
                    "batch_super_chunk": sample.batch_super_chunk[0],
                    "batch_internal_slice": sample.batch_internal_slice,
                    "overlap_prediction_chunksize": overlap_prediction_chunksize,
                    "dataset_shape": zarr_dataset.lazy_data.shape,
                    "spot_parameters": spot_parameters,
                    "logger": logger,
                })
                curr_picked_blocks += 1

                if curr_picked_blocks == exec_n_workers:
                    # Assigning blocks to execution workers
                    jobs = [
                        pool.apply_async(_execute_worker, args=(picked_block,))
                        for picked_block in picked_blocks
                    ]

                    logger.info(f"Dispatcher PID {os.getpid()} dispatching {len(jobs)} jobs")

                    global_workers_spots = []

                    # Wait for all processes to finish
                    for job in jobs:
                        worker_response = job.get()

                        if worker_response is not None:
                            # Global coordinate points
                            global_workers_spots.append(worker_response.astype(np.float32))

                    # Setting variables back to init
                    curr_picked_blocks = 0
                    picked_blocks = []

                    # Concatenate worker spots
                    if len(global_workers_spots):
                        global_workers_spots = np.concatenate(
                            global_workers_spots, axis=0, dtype=np.float32
                        )

                        # Adding picked spots to global list of spots
                        if spots_global_coordinate is None:
                            spots_global_coordinate = global_workers_spots.copy()
                            
                        else:
                            spots_global_coordinate = np.append(
                                spots_global_coordinate,
                                global_workers_spots,
                                axis=0,
                            )

                if i + samples_per_iter > total_batches:
                    message = (
                        f"Not enough samples to retrieve from workers, remaining"
                        f": {i + samples_per_iter - total_batches}"
                    )
                    logger.info(message)
                    break

    if curr_picked_blocks != 0:
        logger.info(f"Blocks not processed inside of loop: {curr_picked_blocks}")
        # Assigning blocks to execution workers
        jobs = [
            pool.apply_async(_execute_worker, args=(picked_block,)) for picked_block in picked_blocks
        ]

        logger.info(f"Dispatcher PID {os.getpid()} dispatching {len(jobs)} jobs")

        global_workers_spots = []

        # Wait for all processes to finish
        for job in jobs:
            worker_response = job.get()

            if worker_response is not None:
                # Global coordinate points
                global_workers_spots.append(worker_response.astype(np.float32))
        
        # Setting variables back to init
        curr_picked_blocks = 0
        picked_blocks = []
        
        # Concatenate worker spots
        if len(global_workers_spots):
            global_workers_spots = np.concatenate(global_workers_spots, axis=0, dtype=np.float32)
            
            # Adding picked spots to global list of spots
            if spots_global_coordinate is None:
                spots_global_coordinate = global_workers_spots.copy()
                
            else:
                spots_global_coordinate = np.append(
                    spots_global_coordinate,
                    global_workers_spots,
                    axis=0,
                )

    end_time = time()

    if tissue_map:
        logger.info("="*70)
        logger.info("TISSUE FILTERING STATISTICS:")
        logger.info(f"  Total batches scanned: {batches_scanned}")
        logger.info(f"  Batches processed: {batches_processed}")
        logger.info(f"  Batches skipped: {batches_skipped}")
        if batches_scanned > 0:
            skip_rate = (batches_skipped / batches_scanned) * 100
            speedup = batches_scanned / max(1, batches_processed)
            logger.info(f"  Skip rate: {skip_rate:.1f}%")
            logger.info(f"  Actual speedup: {speedup:.2f}x")
        logger.info("="*70)

    if spots_global_coordinate is None:
        logger.info("No spots found!")
        spots_df = pd.DataFrame()

    else:
        spots_global_coordinate = spots_global_coordinate.astype(np.float32)
        # Final prunning, might be spots in boundaries where spots where splitted
        start_final_prunning_time = time()
        spots_global_coordinate_prunned, removed_pos = prune_blobs(
            # Prunning only ZYX locations, careful with Masks IDs
            blobs_array=spots_global_coordinate.copy(),
            distance=spot_parameters["min_zyx"][-1] + spot_parameters["radius_confidence"],
        )
        end_final_prunning_time = time()

        message = (
            f"Time taken for final pruning {end_final_prunning_time - start_final_prunning_time} "
            f"before: {len(spots_global_coordinate)} After: {len(spots_global_coordinate_prunned)}"
        )
        logger.info(message)

        logger.info(f"Processing time: {end_time - start_time} seconds")
        
        # Saving spots as numpy and csv
        # np.save(f"{output_folder}/spots.npy", spots_global_coordinate_prunned)

        columns = ["Z", "Y", "X", "Z_center", "Y_center", "X_center", "dist", "r", "fg", "bg"]
        sort_column = "Z"
        int_columns = ["Z", "Y", "X"]

        # If segmentation mask is provided, let's sort by ID
        if segmentation_mask_path:
            columns.append("SEG_ID")
            sort_column = "SEG_ID"
            int_columns.append("SEG_ID")

        spots_df = pd.DataFrame(spots_global_coordinate_prunned, columns=columns)

        spots_df[int_columns] = spots_df[int_columns].astype("int")
        spots_df = spots_df.sort_values(by=sort_column)

        output_csv = f"{output_folder}/cell_likelihoods.csv"
        # Saving spots
        spots_df.to_csv(
            output_csv,
            index=False
        )
        
        # Saving spots
        # proposal_df = pd.DataFrame(spots_global_coordinate_prunned[:, :3], columns = ['x', 'y', 'z'])
        # proposal_df.to_csv(
        #    f"{output_folder}/detected_cells.csv",
        #    index=False
        # ).astype("int")

        data_processes.append(
            DataProcess(
                name=ProcessName.IMAGE_SPOT_DETECTION,
                software_version=__version__,
                start_date_time=start_time,
                end_date_time=end_time,
                input_location=str(dataset_path),
                output_location=str(output_folder),
                outputs={},
                code_url="https://github.com/AllenNeuralDynamics/aind-SmartSPIM-segmentation",
                code_version=__version__,
                parameters={
                    "multiscale": multiscale,
                    "spot_parameters": spot_parameters,
                    "segmentation_mask_path": segmentation_mask_path,
                    "create_mask_from_downsampled": create_mask_from_downsampled,
                    "downsample_factor": downsample_factor,
                    "min_tissue_coverage": min_tissue_coverage,
                    "scheduler_params": {
                        "prediction_chunksize": prediction_chunksize,
                        "target_size_mb": target_size_mb,
                        "n_workers": n_workers,
                        "axis_pad": axis_pad,
                        "batch_size": batch_size,
                    },
                    "output_folder": output_folder,
                },
                notes=f"Detecting cell proposals in path: {dataset_path}",
            )
        )

        utils.generate_processing(
            data_processes=data_processes,
            dest_processing=str(metadata_path),
            processor_full_name=__maintainers__[0],
            pipeline_version=__pipeline_version__,
        )

    # Stop resource tracking
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            metadata_path,
            "smartspim_cell_proposals",
        )

    return spots_df


if __name__ == "__main__":
    smartspim_cell_detection()