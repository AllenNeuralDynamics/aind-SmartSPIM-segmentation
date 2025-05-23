"""
Traditional puncta detection algorithm.
"""

import logging
from copy import copy
from time import time
from typing import Iterable, List, Optional, Tuple

import cupy
import numpy as np
from cupyx.scipy.ndimage import gaussian_laplace
from cupyx.scipy.ndimage import maximum_filter as cupy_maximum_filter
from cupyx.scipy.ndimage import minimum_filter as cupy_minimum_filter
from scipy import spatial
from scipy.special import erf

from .._shared.types import ArrayLike


def prune_blobs(blobs_array: ArrayLike, distance: int, eps=0) -> Tuple[ArrayLike, ArrayLike]:
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
    Tuple[ArrayLike, ArrayLike]
    cupy.ndarray
        Cupy array with the pruned blobs
    np.array
        Removed spots positions
    """
    tree = spatial.cKDTree(blobs_array[:, :3])
    pairs = np.array(list(tree.query_pairs(distance, eps=eps)))
    removed_positions = []
    for i, j in pairs:
        blob1, blob2 = blobs_array[i], blobs_array[j]
        if blob1[-1] > blob2[-1]:
            removed_positions.append(j)
            blob2[-1] = 0
        else:
            removed_positions.append(i)
            blob1[-1] = 0

    return blobs_array[blobs_array[:, -1] > 0], np.array(removed_positions)


def prune_blobs_optimized(blobs_array, distance: int, eps=0) -> cupy.ndarray:
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
    tree = spatial.cKDTree(blobs_array)
    pairs = np.array(list(tree.query_pairs(distance, eps=eps)))

    i_indices, j_indices = zip(*pairs)
    i_indices = np.array(i_indices)
    j_indices = np.array(j_indices)

    # print(i_indices, j_indices)
    # Extract values from blobs_array using the indices
    blob1_values = blobs_array[i_indices, -1]
    blob2_values = blobs_array[j_indices, -1]
    # print(blob1_values[0], blob2_values[0])

    # Find the indices where blob1_values are greater than blob2_values
    greater_indices = np.where(blob1_values > blob2_values)[0]

    # Set values to 0 based on the comparison
    blobs_array[i_indices[greater_indices], -1] = 0
    blobs_array[j_indices[~greater_indices], -1] = 0

    return blobs_array[blobs_array[:, -1] > 0]


def calculate_threshold(block_data):
    block_mean = cupy.mean(block_data)
    block_std = cupy.std(block_data)

    filt_thresh = block_mean + block_std * 2.5

    # empericallly derived threshold range
    if filt_thresh < 10:
        filt_thresh = 10
    elif filt_thresh > 25:
        filt_thresh = 25

    return filt_thresh


def identify_initial_spots(
    data_block: ArrayLike,
    background_percentage: int,
    sigma_zyx: List[int],
    min_zyx: List[int],
    filt_thresh: int,
    raw_thresh: int,
) -> Tuple[ArrayLike, ArrayLike]:
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
        data_block = cupy.maximum(background_image, data_block)

        if raw_thresh < 0:
            block_min = cupy_minimum_filter(data_block, min_zyx)
            raw_thresh = cupy.median(data_block - block_min)

        # data_block[data_block < background_image] = background_image
        # Taking pad from original data, do not reflect if possible
        # data_block = cupy.pad(data_block, pad_size, mode=pad_mode)

        LoG_image = -gaussian_laplace(data_block, sigma_zyx)

        if filt_thresh < 0:
            filt_thresh = calculate_threshold(LoG_image)

        thresholded_img = cupy.logical_and(  # Logical and door to get truth values
            cupy.logical_and(
                LoG_image
                == cupy_maximum_filter(  # Maximum filter (non-linear filter) to find local maxima
                    LoG_image,
                    min_zyx,  # shape that is taken from the input array
                ),  # at every element position, to define the input to the filter function
                LoG_image > filt_thresh,  # array with the values higher than the filter threshold
            ),
            data_block > raw_thresh,  # Array with the values higher than the raw threshold
        )

        spots = cupy.column_stack(  # Creates a vertical stack using the second axis as guide
            cupy.nonzero(thresholded_img)  # Returns the indices that are not zero
        )

    return spots, LoG_image


def scan(img: ArrayLike, spots: ArrayLike, radius: int) -> Tuple[ArrayLike, ArrayLike]:
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
    val = img.shape
    prunned_indices = np.all((spots - radius >= 0) & (spots + radius + 1 <= val), axis=1)
    spots_prunned = spots[prunned_indices]

    if not spots_prunned.size:
        return np.array([]), np.array([])

    weights = np.stack(
        [
            img[
                sp[0] - radius : sp[0] + radius + 1,  # noqa: E203
                sp[1] - radius : sp[1] + radius + 1,  # noqa: E203
                sp[2] - radius : sp[2] + radius + 1,  # noqa: E203
            ]
            for sp in spots_prunned
        ]
    )
    return spots_prunned, weights


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
        change = np.sqrt(sum((last - guess) ** 2))
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


def intensity_integrated_gaussian3D(center, sigmas, limit) -> ArrayLike:
    """
    Calculates the integrated intensity of a 3D Gaussian distribution over a voxel grid.

    This function computes the volume integral of a 3D Gaussian function by using
    the error function (erf) to analytically solve the integral over each voxel.
    The calculation is done independently for each dimension and then combined
    through multiplication.

    Parameters
    ----------
    center : ArrayLike
        The coordinates (x, y, z) of the Gaussian center. Should be a sequence
        of 3 floating-point numbers.
    sigmas : ArrayLike
        The standard deviations (σx, σy, σz) of the Gaussian in each dimension.
        Should be a sequence of 3 positive floating-point numbers.
    limit : int
        The size of the calculation grid in each dimension.

    Returns
    -------
    ArrayLike
        A 3D array of shape (limit, limit, limit) containing the integrated
        Gaussian intensities for each voxel.

    Notes
    -----
    The function works by:
    1. Creating integration bounds at ±0.5 around each voxel center
    2. Computing the error function for these bounds
    3. Taking the difference to get the integrated intensity
    4. Multiplying the results from each dimension

    The error function integration accounts for the total probability contained
    within each voxel boundary.
    """
    # Create an array of shape (2, len(sigmas), limit) with values -0.5 and 0.5
    d_values = np.array([[-0.5, 0.5]], dtype=np.float32).reshape(2, 1, 1)

    # Create an array of shape (2, len(sigmas), limit) with values center[i]
    center_values = np.array(center, dtype=cupy.float32).reshape(1, -1, 1)

    # Create an array of shape (2, len(sigmas), limit) with values sigmas[i]
    sigma_values = np.array(sigmas, dtype=np.float32).reshape(1, -1, 1)

    # Compute the argument of the error function
    res = (np.arange(limit) + d_values - center_values) / np.sqrt(2) / sigma_values

    # Compute the error function
    res_erf = erf(res)

    # Compute the absolute differences between the error function values
    # for the two sides of the center
    diff = np.abs(res_erf[0] - res_erf[1])
    # Compute the product along the first axis
    return np.prod(np.meshgrid(*diff), axis=0)


def scan_bbox(img: ArrayLike, spots: ArrayLike, radius: int) -> Iterable[Tuple[List, ArrayLike]]:
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
        Iterable with array with image data of shape radius in each axis.
    """
    depth, height, width = img.shape

    for p in spots:
        z_min = int(max(0, p[0] - radius))
        z_max = int(min(depth - 1, p[0] + radius))

        y_min = int(max(0, p[1] - radius))
        y_max = int(min(height - 1, p[1] + radius))

        x_min = int(max(0, p[2] - radius))
        x_max = int(min(width - 1, p[2] + radius))

        yield p, img[
            z_min:z_max,  # noqa: E203
            y_min:y_max,  # noqa: E203
            x_min:x_max,  # noqa: E203
        ]


def estimate_background_foreground(
    buffer_context: ArrayLike,
    context_radius: int,
    background_percentile: Optional[float] = 1.0,
) -> Tuple[float, float]:
    """
    Estimates the background foreground and background
    of a given spot. The spot must be in the center of
    the buffer context image.

    Parameters
    ----------
    buffer_context: ArrayLike
        Spot data around a buffer radius
        which should be bigger or equal to
        context radius. The spot must be in
        the center of this block of data.

    context_radius: int
        Radius used when fitting the gaussian.

    background_percentile: Optional[float]
        Background percentile to use.
        Default: 1.0

    Returns
    -------
    Tuple[float, float]
        Foreground and background spot intensities.
    """
    spot_center = np.array(buffer_context.shape) // 2

    # Grid for the spherical mask
    z, y, x = np.ogrid[
        : buffer_context.shape[0],
        : buffer_context.shape[1],
        : buffer_context.shape[2],
    ]

    # Condition to create a circular mask
    fg_condition = (
        (z - spot_center[0]) ** 2 + (y - spot_center[1]) ** 2 + (x - spot_center[2]) ** 2
    ) <= context_radius**2

    bg_condition = np.bitwise_not(fg_condition)

    # Background intensities
    bg_intensities = buffer_context * bg_condition
    bg_intensities = bg_intensities[bg_intensities != 0]

    # Foreground intensities
    fg_intensities = buffer_context * fg_condition
    fg_intensities = fg_intensities[fg_intensities != 0]

    # Manual inspection of spot
    spot_bg = -1
    spot_fg = -1

    if not bg_intensities.shape[0]:
        print(f"Problem in spot {spot_center}, non-zero background is: {bg_intensities}")

    else:
        # Getting background percentile
        spot_bg = np.percentile(bg_intensities.flatten(), background_percentile)

    if not fg_intensities.shape[0]:
        print(f"Problem in spot {spot_center}, non-zero foreground is: {fg_intensities}")

    else:
        # Getting foreground mean
        spot_fg = np.mean(fg_intensities)

    return spot_fg, spot_bg


def traditional_3D_spot_detection(
    data_block: ArrayLike,
    background_percentage: int,
    sigma_zyx: List[int],
    min_zyx: List[int],
    filt_thresh: int,
    raw_thresh: int,
    logger: logging.Logger,
    context_radius: Optional[int] = 3,
    radius_confidence: Optional[float] = 0.05,
    eps: Optional[int] = 0,
    run_context_estimates: Optional[bool] = True,
    buffer_radius: Optional[int] = None,
    background_percentile: Optional[float] = 1.0,
    verbose: Optional[bool] = False,
) -> ArrayLike:
    """
    Runs the spot detection algorithm.

    1. Identify initial spots using:
        1. A. Laplacian of Gaussian to enhance regions
        where the intensity changes dramatically (higher gradient).
        1. B. Percentile to get estimated background image.
        1. C. Combination of logical ANDs to filter the LoG image
        using threshold values and non-linear maximum filter.
    2. Prune identified spots within a certain radius.
    3. Get a small image (context) within a radius for each spot.
    4. Fit a 3D gaussian using the context of each spot and leave the
        ones that are able to converge.

    Parameters
    ----------
    data_block: ArrayLike
        Block of data where we want to run spot detection.

    background_percentage: int
        Background percentage we want to not have in the
        prediction.

    sigma_zyx: List[int]
        Sigma values for the laplacian of gaussians.

    min_zyx: List[int]
        Min zyx for the spot.

    filt_thresh: int
        Threshold for the filtered image.

    raw_thresh: int
        Threshold for the raw data.

    logger: logging.Logger
        Logging object

    context_radius: Optional[int] = 3
        Radius that will be used to pull data
        to be able to fit the 3D gaussian.

    radius_confidence: Optional[float] = 0.05
        Radius confidence. This is used along
        with min_zyx parameter to prune spots
        with the kdTree.

    eps: Optional[int] = 0
        Parameter for the approximate seach.
        Plese check: scipy.spatial.KDTree.query_pairs

    run_context_estimates: Optional[bool]
        Runs estimation of foreground and background
        per spot.

    buffer_radius: Optional[int]
        Buffer radius. This must be bigger than
        the context radius.

    background_percentile: Optional[float]
        Background percentile used to compute
        the spot background.
        Default: 1.0

    verbose: Optional[bool] = False,
        Verbose to show logs.

    Returns
    -------
    ArrayLike
        3D ZYX points in the center of the bloby objects.
        If a segmentation mask is provided, the points will have
        the form of Z,Y,X,MASK_ID where MASK_ID represents the ID
        of the segmentation mask in the area where the spot is located.
    """
    puncta = None

    initial_spots_start_time = time()
    initial_spots, gaussian_laplaced_img = identify_initial_spots(
        data_block=data_block,
        background_percentage=background_percentage,
        sigma_zyx=sigma_zyx,
        min_zyx=min_zyx,
        filt_thresh=filt_thresh,
        raw_thresh=raw_thresh,
    )
    initial_spots_end_time = time()

    if verbose:
        logger.info(f"Initial spots time: {initial_spots_end_time - initial_spots_start_time}")

    if initial_spots is not None and len(initial_spots) and gaussian_laplaced_img is not None:
        minYX = min_zyx[-1]

        prunning_start_time = time()
        pruned_spots, _ = prune_blobs(initial_spots.get(), minYX + radius_confidence, eps=eps)
        prunning_end_time = time()
        if verbose:
            logger.info(f"Prunning spots time: {prunning_end_time - prunning_start_time}")

        guassian_laplaced_img_memory = gaussian_laplaced_img.get()

        scanning_start_time = time()
        scanned_spots, scanned_contexts = scan(
            guassian_laplaced_img_memory, pruned_spots, context_radius
        )
        scanning_end_time = time()
        if verbose:
            logger.info(f"Scanning spots time: {scanning_end_time - scanning_start_time}")

        results = []

        fit_gau_spots_start_time = time()
        for idx in range(0, scanned_spots.shape[0]):
            coord = scanned_spots[idx]
            context = scanned_contexts[idx]

            out = fit_gaussian(context, sigma_zyx, context_radius)
            if not out:
                continue

            center, N, r = out
            center -= [context_radius] * 3
            unpadded_coord = coord[:3]

            results.append(unpadded_coord.tolist() + center.tolist() + [np.linalg.norm(center), r])

        fit_gau_spots_end_time = time()

        if verbose:
            logger.info(
                f"Fitting gaussian to {len(scanned_spots)} spots time: {fit_gau_spots_end_time - fit_gau_spots_start_time}"  # noqa: E501
            )

        puncta = np.array(results).astype(np.float32)

        if not len(puncta):
            return None

        if run_context_estimates:
            data_block = data_block.get()

            # Making sure buffer radius is correct
            if buffer_radius is None or buffer_radius < context_radius:
                buffer_radius = context_radius * 2

            # Estimating spots foreground - background
            spots_fg_bg = np.array(
                [
                    estimate_background_foreground(
                        buffer_context=buffer_context,
                        background_percentile=background_percentile,
                        context_radius=context_radius,
                    )
                    for _, buffer_context in scan_bbox(data_block, puncta, buffer_radius)
                ]
            )

            # horizontal stacking
            puncta = np.append(puncta.T, spots_fg_bg.T, axis=0).T

    return puncta
