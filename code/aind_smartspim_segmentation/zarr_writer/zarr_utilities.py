"""
Module to the zarr utilities
"""

import multiprocessing
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import dask
import numpy as np
import pims
from dask.array import concatenate, pad
from dask.array.core import Array
from dask.base import tokenize
from natsort import natsorted
from numcodecs import blosc
from skimage.io import imread as sk_imread

PathLike = Union[str, Path]
ArrayLike = Union[dask.array.core.Array, np.ndarray]
blosc.use_threads = False


def add_leading_dim(data: ArrayLike) -> ArrayLike:
    """
    Adds a leading dimension

    Parameters
    ------------------------

    data: ArrayLike
        Input array that will have the
        leading dimension

    Returns
    ------------------------

    ArrayLike:
        Array with the new dimension in front.
    """
    return data[None, ...]


def pad_array_n_d(arr: ArrayLike, dim: int = 5) -> ArrayLike:
    """
    Pads a daks array to be in a 5D shape.

    Parameters
    ------------------------

    arr: ArrayLike
        Dask/numpy array that contains image data.
    dim: int
        Number of dimensions that the array will be padded

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """
    if dim > 5:
        raise ValueError("Padding more than 5 dimensions is not supported.")

    while arr.ndim < dim:
        arr = arr[np.newaxis, ...]
    return arr


def read_image_directory_structure(folder_dir: PathLike) -> dict:
    """
    Creates a dictionary representation of all the images
    saved by folder/col_N/row_N/images_N.[file_extention]

    Parameters
    ------------------------
    folder_dir:PathLike
        Path to the folder where the images are stored

    Returns
    ------------------------
    dict:
        Dictionary with the image representation where:
        {channel_1: ... {channel_n: {col_1: ... col_n: {row_1: ... row_n: [image_0, ..., image_n]} } } }
    """

    directory_structure = {}
    folder_dir = Path(folder_dir)

    channel_paths = natsorted(
        [
            folder_dir.joinpath(folder)
            for folder in os.listdir(folder_dir)
            if os.path.isdir(folder_dir.joinpath(folder))
        ]
    )

    for channel_idx in range(len(channel_paths)):
        directory_structure[channel_paths[channel_idx]] = {}

        cols = natsorted(os.listdir(channel_paths[channel_idx]))

        for col in cols:
            possible_col = channel_paths[channel_idx].joinpath(col)

            if os.path.isdir(possible_col):
                directory_structure[channel_paths[channel_idx]][col] = {}

                rows = natsorted(os.listdir(possible_col))

                for row in rows:
                    possible_row = (
                        channel_paths[channel_idx].joinpath(col).joinpath(row)
                    )

                    if os.path.isdir(possible_row):
                        directory_structure[channel_paths[channel_idx]][col][
                            row
                        ] = natsorted(os.listdir(possible_row))

    return directory_structure


def lazy_tiff_reader(
    filename: PathLike,
    shape: Optional[Tuple[int]] = None,
    dtype: Optional[type] = None,
):
    """
    Creates a dask array to read an image located in a specific path.

    Parameters
    ------------------------

    filename: PathLike
        Path to the image

    Returns
    ------------------------

    dask.array.core.Array
        Array representing the image data
    """
    name = "imread-%s" % tokenize(filename, map(os.path.getmtime, filename))

    if dtype is None or shape is None:
        with pims.open(filename) as imgs:
            dtype = np.dtype(imgs.pixel_type)
            shape = (1,) + (len(imgs),) + imgs.frame_shape

    key = [(name,) + (0,) * len(shape)]
    value = [(add_leading_dim, (sk_imread, filename))]
    dask_arr = dict(zip(key, value))
    chunks = tuple((d,) for d in shape)

    return Array(dask_arr, name, chunks, dtype)


def fix_image_diff_dims(
    new_arr: ArrayLike, chunksize: Tuple[int], len_chunks: int, work_axis: int
) -> ArrayLike:
    """
    Fixes the array dimension to match the shape of
    the chunksize.

    Parameters
    ------------------------

    new_arr: ArrayLike
        Array to be fixed

    chunksize: Tuple[int]
        Chunksize of the original array

    len_chunks: int
        Length of the chunksize. Used as a
        parameter to avoid computing it
        multiple times

    work_axis: int
        Axis to concatenate. If the different
        axis matches this one, there is no need
        to fix the array dimension

    Returns
    ------------------------

    ArrayLike
        Array with the new dimensions
    """

    zeros_dim = []
    diff_dim = -1
    c = 0

    for chunk_idx in range(len_chunks):
        new_chunk_dim = new_arr.chunksize[chunk_idx]

        if new_chunk_dim != chunksize[chunk_idx]:
            c += 1
            diff_dim = chunk_idx

        zeros_dim.append(abs(chunksize[chunk_idx] - new_chunk_dim))

    if c > 1:
        raise ValueError("Block has two different dimensions")
    else:
        if (diff_dim - len_chunks) == work_axis:
            return new_arr

        n_pad = tuple(tuple((0, dim)) for dim in zeros_dim)
        new_arr = pad(
            new_arr, pad_width=n_pad, mode="constant", constant_values=0
        ).rechunk(chunksize)

    return new_arr


def concatenate_dask_arrays(arr_1: ArrayLike, arr_2: ArrayLike, axis: int) -> ArrayLike:
    """
    Concatenates two arrays in a given
    dimension

    Parameters
    ------------------------

    arr_1: ArrayLike
        Array 1 that will be concatenated

    arr_2: ArrayLike
        Array 2 that will be concatenated

    axis: int
        Concatenation axis

    Returns
    ------------------------

    ArrayLike
        Concatenated array that contains
        arr_1 and arr_2
    """

    shape_arr_1 = arr_1.shape
    shape_arr_2 = arr_2.shape

    if shape_arr_1 != shape_arr_2:
        slices = []
        dims = len(shape_arr_1)

        for shape_dim_idx in range(dims):
            if shape_arr_1[shape_dim_idx] > shape_arr_2[shape_dim_idx] and (
                shape_dim_idx - dims != axis
            ):
                raise ValueError(
                    f"""
                    Array 1 {shape_arr_1} must have
                     a smaller shape than array 2 {shape_arr_2}
                     except for the axis dimension {shape_dim_idx}
                     {dims} {shape_dim_idx - dims} {axis}
                    """
                )

            if shape_arr_1[shape_dim_idx] != shape_arr_2[shape_dim_idx]:
                slices.append(slice(0, shape_arr_1[shape_dim_idx]))

            else:
                slices.append(slice(None))

        slices = tuple(slices)
        arr_2 = arr_2[slices]

    try:
        res = concatenate([arr_1, arr_2], axis=axis)
    except ValueError:
        raise ValueError(
            f"""
            Unable to cancat arrays - Shape 1:
             {shape_arr_1} shape 2: {shape_arr_2}
            """
        )

    return res


def read_chunked_stitched_image_per_channel(
    directory_structure: dict,
    channel_name: str,
    start_slice: int,
    end_slice: int,
) -> ArrayLike:
    """
    Creates a dask array of the whole image volume
    based on image chunks preserving the chunksize.

    Parameters
    ------------------

    directory_structure:dict
        dictionary to store paths of images with the following structure:
        {channel_1: ... {channel_n: {col_1: ... col_n: {row_1: ... row_n: [image_0, ..., image_n]} } } }

    channel_name : str
        Channel name to reconstruct the image volume

    start_slice: int
        When using multiprocessing, this is
        the start slice the worker will use for
        the array concatenation

    end_slice: int
        When using multiprocessing, this is
        the final slice the worker will use for
        the array concatenation

    Returns
    ------------------------

    ArrayLike
        Array with the image volume
    """
    concat_z_3d_blocks = concat_horizontals = horizontal = None

    # Getting col structure
    rows = list(directory_structure.values())[0]
    rows_paths = list(rows.keys())
    first = True

    for slice_pos in range(start_slice, end_slice):
        idx_col = 0
        idx_row = 0

        concat_horizontals = None

        for row_name in rows_paths:
            idx_row = 0
            horizontal = []
            shape = None
            dtype = None
            column_names = list(directory_structure[channel_name][row_name].keys())
            n_cols = len(column_names)

            for column_name_idx in range(n_cols):
                valid_image = True
                column_name = column_names[column_name_idx]
                last_col = column_name_idx == n_cols - 1

                if last_col:
                    shape = None
                    dtype = None

                try:
                    slice_name = directory_structure[channel_name][row_name][
                        column_name
                    ][slice_pos]

                    filepath = str(
                        channel_name.joinpath(row_name)
                        .joinpath(column_name)
                        .joinpath(slice_name)
                    )

                    new_arr = lazy_tiff_reader(filepath, dtype=dtype, shape=shape)

                    if shape is None or dtype is None:
                        shape = new_arr.shape
                        dtype = new_arr.dtype
                        last_col = False

                except ValueError:
                    print("No valid image in ", slice_pos)
                    valid_image = False

                if valid_image:
                    horizontal.append(new_arr)

                idx_row += 1

            # Concatenating horizontally lazy images
            horizontal_concat = concatenate(horizontal, axis=-1)

            if not idx_col:
                concat_horizontals = horizontal_concat
            else:
                concat_horizontals = concatenate_dask_arrays(
                    arr_1=concat_horizontals, arr_2=horizontal_concat, axis=-2
                )

            idx_col += 1

        if first:
            concat_z_3d_blocks = concat_horizontals
            first = False

        else:
            concat_z_3d_blocks = concatenate_dask_arrays(
                arr_1=concat_z_3d_blocks, arr_2=concat_horizontals, axis=-3
            )

    return concat_z_3d_blocks, [start_slice, end_slice]


def _read_chunked_stitched_image_per_channel(args_dict: dict):
    """
    Function used to be dispatched to workers
    by using multiprocessing
    """
    return read_chunked_stitched_image_per_channel(**args_dict)


def channel_parallel_reading(
    directory_structure: dict,
    channel_idx: int,
    workers: Optional[int] = 0,
    chunks: Optional[int] = 1,
    ensure_parallel: Optional[bool] = True,
) -> ArrayLike:
    """
    Creates a dask array of the whole image channel volume based
    on image chunks preserving the chunksize and using
    multiprocessing.

    Parameters
    ------------------------

    directory_structure: dict
        dictionary to store paths of images with the following structure:
        {channel_1: ... {channel_n: {col_1: ... col_n: {row_1: ... row_n: [image_0, ..., image_n]} } } }

    channel_name : str
        Channel name to reconstruct the image volume

    sample_img: ArrayLike
        Image used as guide for the chunksize

    workers: Optional[int]
        Number of workers that will be used
        for reading the chunked image.
        Default value 0, it means that the
        available number of cores will be used.

    chunks: Optional[int]
        Chunksize of the 3D chunked images.

    ensure_parallel: Optional[bool]
        True if we want to read the images in
        parallel. False, otherwise.

    Returns
    ------------------------

    ArrayLike
        Array with the image channel volume
    """
    if workers == 0:
        workers = multiprocessing.cpu_count()

    cols = list(directory_structure.values())[0]
    n_images = len(list(list(cols.values())[0].values())[0])
    #     print(f"n_images: {n_images}")

    channel_paths = list(directory_structure.keys())
    dask_array = None

    if n_images < workers and ensure_parallel:
        workers = n_images

    if n_images < workers or not ensure_parallel:
        dask_array = read_chunked_stitched_image_per_channel(
            directory_structure=directory_structure,
            channel_name=channel_paths[channel_idx],
            start_slice=0,
            end_slice=n_images,
        )[0]
        print(f"No need for parallel reading... {dask_array}")

    else:
        images_per_worker = n_images // workers
        print(
            f"Setting workers to {workers} - {images_per_worker} - total images: {n_images}"
        )

        # Getting 5 dim image TCZYX
        args = []
        start_slice = 0
        end_slice = images_per_worker

        for idx_worker in range(workers):
            arg_dict = {
                "directory_structure": directory_structure,
                "channel_name": channel_paths[channel_idx],
                "start_slice": start_slice,
                "end_slice": end_slice,
            }

            args.append(arg_dict)

            if idx_worker + 1 == workers - 1:
                start_slice = end_slice
                end_slice = n_images
            else:
                start_slice = end_slice
                end_slice += images_per_worker

        res = []
        with multiprocessing.Pool(workers) as pool:
            results = pool.imap(
                _read_chunked_stitched_image_per_channel,
                args,
                chunksize=chunks,
            )

            for pos in results:
                res.append(pos)

        for res_idx in range(len(res)):
            if not res_idx:
                dask_array = res[res_idx][0]
            else:
                dask_array = concatenate([dask_array, res[res_idx][0]], axis=-3)

            print(f"Slides: {res[res_idx][1]}")

    return dask_array


def parallel_read_chunked_stitched_multichannel_image(
    directory_structure: dict,
    workers: Optional[int] = 0,
    ensure_parallel: Optional[bool] = True,
    divide_channels: Optional[bool] = True,
) -> ArrayLike:
    """
    Creates a dask array of the whole image volume based
    on image chunks preserving the chunksize and using
    multiprocessing.

    Parameters
    ------------------------

    directory_structure: dict
        dictionary to store paths of images with the following structure:
        {channel_1: ... {channel_n: {col_1: ... col_n: {row_1: ... row_n: [image_0, ..., image_n]} } } }

    sample_img: ArrayLike
        Image used as guide for the chunksize

    workers: Optional[int]
        Number of workers that will be used
        for reading the chunked image.
        Default value 0, it means that the
        available number of cores will be used.

    ensure_parallel: Optional[bool]
        True if we want to read the images in
        parallel. False, otherwise.

    Returns
    ------------------------

    ArrayLike
        Array with the image channel volume
    """

    multichannel_image = None

    channel_paths = list(directory_structure.keys())

    multichannels = []
    read_channels = {}
    print(f"Channel in directory structure: {channel_paths}")

    for channel_idx in range(len(channel_paths)):
        print(f"Reading images from {channel_paths[channel_idx]}")
        start_time = time.time()
        read_chunked_channel = channel_parallel_reading(
            directory_structure,
            channel_idx,
            workers=workers,
            ensure_parallel=ensure_parallel,
        )
        end_time = time.time()

        print(f"Time reading single channel image: {end_time - start_time}")

        # Padding to 4D if necessary
        ch_name = Path(channel_paths[channel_idx]).name

        read_chunked_channel = pad_array_n_d(read_chunked_channel, 4)
        multichannels.append(read_chunked_channel)
        read_channels[ch_name] = read_chunked_channel

    if divide_channels:
        return read_channels

    if len(multichannels) > 1:
        multichannel_image = concatenate(multichannels, axis=0)
    else:
        multichannel_image = multichannels[0]

    return multichannel_image
