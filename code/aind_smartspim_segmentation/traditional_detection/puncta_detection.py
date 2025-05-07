"""
Tim's Wang puncta detection algorithm.
Initially modified by: Camilo Laiton
"""

import logging
import math
from copy import copy
from time import time
from typing import List, Optional

import cupy
import numpy as np
from cupyx.scipy.ndimage import gaussian_laplace
from cupyx.scipy.ndimage import maximum_filter as mf
from scipy import spatial
from scipy.special import erf

from .._shared.types import ArrayLike


def get_centers_in_slice(slice_pos: int, centers: List[int]):
    """
    Returns two list with the Y and X positions
    of the detected puncta

    Parameters
    ----------
    slice_pos: int
        Slice position in the array

    centers: List[int]
        List with the location of the puncta in
        order YX

    Returns
    -------
    Tuple[ List[int, int], List[int], List[int] ]
        Returns a tuple with the identified positions
        for that slide with the Y and X locations
        in separate arrays
    """
    if isinstance(centers, cupy.ndarray):
        centers = centers.get()

    identified_spots = []
    xs = []
    ys = []

    for idx in range(len(centers)):
        if centers[idx][0] == slice_pos:
            center = centers[idx]
            identified_spots.append(center)
            xs.append(center[2])
            ys.append(center[1])

    return identified_spots, xs, ys


def prune_blobs(blobs_array: ArrayLike, distance: int):
    """
    Prune blobs based on a radius distance.

    Parameters
    ----------
    blobs_array: ArrayLike
        Array that contains the YX position of
        the identified blobs.

    distance: int
        Minimum distance to prune blobs

    Returns
    -------
    cupy.ndarray
        Cupy array with the pruned blobs
    """

    tree = spatial.cKDTree(blobs_array[:, :3])
    pairs = np.array(list(tree.query_pairs(distance)))
    for i, j in pairs:
        blob1, blob2 = blobs_array[i], blobs_array[j]
        if blob1[-1] > blob2[-1]:
            blob2[-1] = 0
        else:
            blob1[-1] = 0
    return cupy.array([b for b in blobs_array if b[-1] > 0])


def identify_initial_spots(
    data_block: ArrayLike,
    background_percentage: int,
    sigma_zyx: List[int],
    pad_size: int,
    min_zyx: List[int],
    filt_thresh: int,
    raw_thresh: int,
    pad_mode: Optional[str] = "reflect",
):
    """
    Identifies the initial spots using Laplacian of
    Gaussian, filtering and maximum filter.

    Parameters
    ----------
    data_block: ArrayLike
        Block of data to process

    background_percentage: int
        Background percentage based on image data

    sigma_zyx: List[int]
        Sigmas to be applied over each axis in
        the Laplacian of Gaussian.

    pad_size: int
        Padding size in the borders of the image

    min_zyx: List[int]
        Shape of the subarray taken by the maximum
        filter over each axis.

    filt_thresh: int
        Threshold used for the image after is filtered
        with the Laplacian of Gaussian.

    raw_thresh: int
        Raw threshold used in the raw data. Helps to
        decrease the computation.

    pad_mode: str
        Padding applied to the image, this helps in the
        non-linear filtering.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Tuple with the identified spots and the
        Laplace of Gaussian image.
    """

    spots = None
    LoG_image = None

    data_block = cupy.array(data_block, np.float32)
    non_zero_indices = data_block[data_block > 0]

    if len(non_zero_indices):
        background_image = cupy.percentile(non_zero_indices, background_percentage)

        data_block[data_block < background_image] = background_image
        data_block = cupy.pad(data_block, pad_size, mode=pad_mode)

        LoG_image = -gaussian_laplace(data_block, sigma_zyx)

        thresholded_img = cupy.logical_and(  # Logical and door to get truth values
            cupy.logical_and(
                LoG_image
                <= mf(  # Maximum filter (non-linear filter) to find local maxima
                    LoG_image,
                    min_zyx,  # shape that is taken from the input array, at every element position, to define the input to the filter function
                ),
                LoG_image > filt_thresh,  # array with the values higher than the filter threshold
            ),
            data_block > raw_thresh,  # Array with the values higher than the raw threshold
        )

        spots = cupy.column_stack(  # Creates a vertical stack using the second axis as guide
            cupy.nonzero(thresholded_img)  # Returns the indices that are not zero
        )

    return spots, LoG_image


def scan(img: ArrayLike, spots: ArrayLike, radius: int):
    """
    Scans the spots to get image data
    for each of them. Additionally, we prune
    spots around borders of the image since
    the algorithm is sensitive to these areas
    and we also have padding.

    Parameters
    ----------
    img: ArrayLike
        Image where the spots are located

    spots: ArrayLike
        Identified spots

    radius: int
        Search radius around each spot

    Returns
    -------
    Tuple
        Tuple with the spot location and
        image data of shape radius in each axis.
    """
    output = []

    spots = spots.astype(int)

    for spot in spots:
        list_spot = tuple(slice(int(x - radius), int(x + radius + 1)) for x in spot[:3])
        w = img[list_spot]
        if np.product(w.shape) == (2 * radius + 1) ** 3:  # just ignore near edge
            output.append([spot, w])

    return tuple(output)


def fit_gaussian(S, fit_sigmas, r):
    """
    Gaussian fitting to prune false positives

    Parameters
    ----------
    S: ArrayLike

    fit_sigmas: List[int]
        Sigmas to be applied

    r: int
        Radius size
    """
    maxIter = 50
    minChange = 0.06
    # initialize
    iteration = 0
    change = 1
    guess = np.array([r] * 3, dtype=np.float32)
    last = copy(guess)
    S = np.maximum(S, 0)
    limit = r * 2 + 1
    mesh = np.mgrid[:(limit), :(limit), :(limit)]

    while change > minChange:
        N = intensity_integrated_gaussian3D(guess, fit_sigmas, 2 * r + 1)
        # print(f"S: {S.shape} and N: {N.shape}")
        int_sum = S * N
        int_sum_sum = int_sum.sum()
        if (
            iteration >= maxIter
            or int_sum_sum == 0
            or np.logical_and(guess >= 0, guess < (2 * r + 1)).sum() != 3
        ):
            return False
        for i, m in enumerate(mesh):
            guess[i] = (m * int_sum).sum() / int_sum_sum  # estimate for each dimension
        change = math.sqrt(sum((last - guess) ** 2))
        iteration += 1
        last = copy(guess)
    intensity = int(int_sum_sum / (N * N).sum())
    if intensity > 0:
        return [
            guess,
            int(int_sum_sum / (N * N).sum()),
            np.corrcoef(N.flatten(), S.flatten())[0, 1],
        ]
    else:
        return False


def intensity_integrated_gaussian3D(center, sigmas, limit):
    diff = np.empty((2, 3), object)  # +/- and per dim
    len_sigmas = len(sigmas)
    for i in range(len_sigmas):
        for j, d in enumerate([-0.5, 0.5]):
            res = (np.arange(limit) + d - center[i]) / math.sqrt(2) / sigmas[i]
            res_erf = erf(res)
            if res_erf is not None:
                diff[j, i] = res_erf

    diff = abs(diff[0] - diff[1])
    return np.prod(np.meshgrid(*diff), 0)


def traditional_3D_spot_detection(
    data_block: ArrayLike,
    background_percentage: int,
    sigma_zyx: List[int],
    pad_size: int,
    min_zyx: List[int],
    filt_thresh: int,
    raw_thresh: int,
    logger: logging.Logger,
    context_radius: Optional[int] = 3,
    radius_confidence: Optional[float] = 0.05,
    verbose: Optional[bool] = False,
):
    """
    Runs the spot detection algorithm.
    1. Identify initial spots using:
        1. A. Laplacian of Gaussian to enhance regions
        where the intensity changes dramatically (higher gradient).
        1. B. Percentile to get estimated background image.
        1. C. Combination of logical ANDs to filter the LoG image
        using threshold values and non-linear maximum filter.
    """
    puncta = None

    initial_spots_start_time = time()
    initial_spots, gaussian_laplaced_img = identify_initial_spots(
        data_block=data_block,
        background_percentage=background_percentage,
        sigma_zyx=sigma_zyx,
        pad_size=pad_size,
        min_zyx=min_zyx,
        filt_thresh=filt_thresh,
        raw_thresh=raw_thresh,
    )
    initial_spots_end_time = time()

    if verbose:
        logger.info(f"Initial spots time: {initial_spots_end_time - initial_spots_start_time}")
    len_spots = len(initial_spots) if initial_spots is not None else None

    if initial_spots is not None and len(initial_spots) and gaussian_laplaced_img is not None:

        minYX = min_zyx[-1]

        prunning_start_time = time()
        pruned_spots = prune_blobs(initial_spots.get(), minYX + radius_confidence)
        prunning_end_time = time()

        if verbose:
            logger.info(
                f"Prunning {len(initial_spots)} spots time: {prunning_end_time - prunning_start_time}"
            )
        guassian_laplaced_img_memory = gaussian_laplaced_img.get()

        scanning_start_time = time()
        scanned_spots = scan(guassian_laplaced_img_memory, pruned_spots, context_radius)
        scanning_end_time = time()
        if verbose:
            logger.info(f"Prunning spots time: {scanning_end_time - scanning_start_time}")

        data_block_shape = cupy.asarray(data_block.shape)
        results = []

        fit_gau_spots_start_time = time()
        for coord, context in scanned_spots:

            out = fit_gaussian(context, sigma_zyx, context_radius)
            if not out:
                continue

            center, N, r = out
            center -= [context_radius] * 3
            unpadded_coord = coord[:3] - pad_size
            if (
                True in cupy.less(data_block_shape, unpadded_coord)
                or np.where(unpadded_coord < 0)[0].shape[0]
            ):
                continue

            results.append(unpadded_coord.tolist() + center.tolist() + [np.linalg.norm(center), r])

        fit_gau_spots_end_time = time()

        if verbose:
            logger.info(
                f"Fitting gaussian to {len(scanned_spots)} spots time: {fit_gau_spots_end_time - fit_gau_spots_start_time}"
            )

        results_np = np.array(results).astype(int)

        if not len(results_np):
            return None

        puncta = results_np[:, :3]

    return puncta
