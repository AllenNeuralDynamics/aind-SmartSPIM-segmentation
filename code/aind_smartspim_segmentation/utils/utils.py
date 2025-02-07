#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:23:13 2022

@author: nicholas.lusk
@Modified by: camilo.laiton
"""

import importlib
import json
import logging
import multiprocessing
import os
import platform
import time
from datetime import datetime
from typing import List, Optional, Tuple

from pathlib import Path
import dask
import matplotlib.pyplot as plt
import numpy as np
import psutil
from aind_data_schema.core.processing import DataProcess, PipelineProcess, Processing
from astropy.stats import SigmaClip
from cellfinder_core.detect import detect
from photutils.background import Background2D
from scipy import ndimage as ndi
from scipy.signal import argrelmin

from .._shared.types import ArrayLike, PathLike
from .pymusica import musica


@dask.delayed
def delay_astro(
    img: ArrayLike,
    estimator: str = "MedianBackground",
    box: Tuple[int] = (50, 50),
    filt: Tuple[int] = (3, 3),
    sig_clip: int = 3,
    pad: int = 0,
    reflect: int = 0,
    amplify: bool = False,
    smooth: bool = False,
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

    if reflect != 0:
        img = np.pad(img, reflect, mode="reflect")

    if pad != 0:
        img = np.pad(img, pad, mode="constant", constant_values=0)

    # print(img.shape)

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
                exclude_percentile=100,
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
def delay_detect(
    img: ArrayLike,
    save_path: PathLike,
    count: int,
    offset: int,
    padding: int,
    process_by,
    smartspim_config: dict,
):
    """
    Delayed function to run cellfinder
    main script.
    """
    cells = detect.main(
        signal_array=np.asarray(img),
        save_path=save_path,
        block=count,
        offset=offset,
        padding=padding,
        process_by=process_by,
        **smartspim_config,
    )

    return cells


def find_good_blocks(img, counts, chunk, ds=3):
    """
    Function to Identify good blocks to process
    using downsampled zarr

    Parameters
    ----------
    img : da.array
        dask array of low resolution image
    counts : list[int]
        chunks per dimension of level 0 array.
    chunk : int
        chunk size used for cell detection
    ds : int
        the factor by which the array is downsampled

    Returns
    -------
    block_dict : dict
        dictionary with information on which blocks to
        process. key = block number and value = bool

    """
    if isinstance(img, dask.array.core.Array):
        img = np.asarray(img)

    img = ndi.gaussian_filter(img, sigma=5.0, mode="constant", cval=0)
    count, bin_count = np.histogram(img.astype("uint16"), bins=2**16, range=(0, 2**16), density=True)

    try:
        thresh = argrelmin(count, order=10)[0][0]
    except IndexError:
        thresh = 100

    img_binary = np.where(img >= thresh, 1, 0)

    cz = int(chunk / 2**ds)
    dims = list(img_binary.shape)

    b = 0
    block_dict = {}

    """ there are rare occasions where the level 0 and
    level 3 array disagree on chuncks so have some
    catches to account for that """
    for z in range(counts[0]):
        z_l, z_u = z * cz, (z + 1) * cz
        if z_l > dims[0] - 1:
            z_l = dims[0] - 2
        if z_u > dims[0] - 1:
            z_u = dims[0] - 1
        for y in range(counts[1]):
            y_l, y_u = y * cz, (y + 1) * cz
            if y_l > dims[1] - 1:
                y_l = dims[1] - 2
            if y_u > dims[1] - 1:
                y_u = dims[1] - 1
            for x in range(counts[2]):
                x_l, x_u = x * cz, (x + 1) * cz
                if x_l > dims[2] - 1:
                    x_l = dims[2] - 2
                if x_u > dims[2] - 1:
                    x_u = dims[2] - 1

                block = img_binary[
                    z_l:z_u,
                    y_l:y_u,
                    x_l:x_u,
                ]

                if np.sum(block) > 0:
                    block_dict[b] = True
                else:
                    block_dict[b] = False

                b += 1

    return block_dict


def create_logger(output_log_path: PathLike):
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """

    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/fusion_log_{CURR_DATE_TIME}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


def read_json_as_dict(filepath: str):
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


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


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: PathLike,
    processor_full_name: str,
    pipeline_version: str,
):
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    data_processes: List[dict]
        List with the processes aplied in the pipeline.

    dest_processing: PathLike
        Path where the processing file will be placed.

    processor_full_name: str
        Person in charged of running the pipeline
        for this data asset

    pipeline_version: str
        Terastitcher pipeline version

    """
    # flake8: noqa: E501
    processing_pipeline = PipelineProcess(
        data_processes=data_processes,
        processor_full_name=processor_full_name,
        pipeline_version=pipeline_version,
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        note="Metadata for segmentation step",
    )

    processing = Processing(
        processing_pipeline=processing_pipeline,
        notes="This processing only contains metadata of cell segmentation \
            and needs to be compiled with other steps at the end",
    )

    print(f"Generating detection processing in {dest_processing}: Processing: {processing}")

    processing.write_standard_file(output_directory=dest_processing)


def profile_resources(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    monitoring_interval: int,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    monitoring_interval: int
        Monitoring interval in seconds
    """
    start_time = time.time()

    while True:
        current_time = time.time() - start_time
        time_points.append(current_time)

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=monitoring_interval)
        cpu_percentages.append(cpu_percent)

        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usages.append(memory_info.percent)

        time.sleep(monitoring_interval)


def generate_resources_graphs(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    output_path: str,
    prefix: str,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    output_path: str
        Path where the image will be saved

    prefix: str
        Prefix name for the image
    """
    time_len = len(time_points)
    memory_len = len(memory_usages)
    cpu_len = len(cpu_percentages)

    min_len = min([time_len, memory_len, cpu_len])
    if not min_len:
        return

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_points[:min_len], cpu_percentages[:min_len], label="CPU Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_points[:min_len], memory_usages[:min_len], label="Memory Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/{prefix}_compute_resources.png", bbox_inches="tight")


def stop_child_process(process: multiprocessing.Process):
    """
    Stops a process

    Parameters
    ----------
    process: multiprocessing.Process
        Process to stop
    """
    process.terminate()
    process.join()


def get_size(bytes, suffix: str = "B") -> str:
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'

    Parameters
    ----------
    bytes: bytes
        Bytes to scale

    suffix: str
        Suffix used for the conversion
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_code_ocean_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1

    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())

        container_cpus = cfs_quota_us // cfs_period_us

    except FileNotFoundError as e:
        container_cpus = 0

    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus


def print_system_information(logger: logging.Logger):
    """
    Prints system information

    Parameters
    ----------
    logger: logging.Logger
        Logger object
    """
    co_memory = os.environ.get("CO_MEMORY")
    co_memory = int(co_memory) if co_memory else None

    # System info
    sep = "=" * 40
    logger.info(f"{sep} Code Ocean Information {sep}")
    logger.info(f"Code Ocean assigned cores: {get_code_ocean_cpu_limit()}")

    if co_memory:
        logger.info(f"Code Ocean assigned memory: {get_size(co_memory)}")

    logger.info(f"Computation ID: {os.environ.get('CO_COMPUTATION_ID')}")
    logger.info(f"Capsule ID: {os.environ.get('CO_CAPSULE_ID')}")
    logger.info(f"Is pipeline execution?: {bool(os.environ.get('AWS_BATCH_JOB_ID'))}")

    logger.info(f"{sep} System Information {sep}")
    uname = platform.uname()
    logger.info(f"System: {uname.system}")
    logger.info(f"Node Name: {uname.node}")
    logger.info(f"Release: {uname.release}")
    logger.info(f"Version: {uname.version}")
    logger.info(f"Machine: {uname.machine}")
    logger.info(f"Processor: {uname.processor}")

    # Boot info
    logger.info(f"{sep} Boot Time {sep}")
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    logger.info(f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")

    # CPU info
    logger.info(f"{sep} CPU Info {sep}")
    # number of cores
    logger.info(f"Physical node cores: {psutil.cpu_count(logical=False)}")
    logger.info(f"Total node cores: {psutil.cpu_count(logical=True)}")

    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    logger.info(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    logger.info(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    logger.info(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    # CPU usage
    logger.info("CPU Usage Per Core before processing:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        logger.info(f"Core {i}: {percentage}%")
    logger.info(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # Memory info
    logger.info(f"{sep} Memory Information {sep}")
    # get the memory details
    svmem = psutil.virtual_memory()
    logger.info(f"Total: {get_size(svmem.total)}")
    logger.info(f"Available: {get_size(svmem.available)}")
    logger.info(f"Used: {get_size(svmem.used)}")
    logger.info(f"Percentage: {svmem.percent}%")
    logger.info(f"{sep} Memory - SWAP {sep}")
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    logger.info(f"Total: {get_size(swap.total)}")
    logger.info(f"Free: {get_size(swap.free)}")
    logger.info(f"Used: {get_size(swap.used)}")
    logger.info(f"Percentage: {swap.percent}%")

    # Network information
    logger.info(f"{sep} Network Information {sep}")
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            logger.info(f"=== Interface: {interface_name} ===")
            if str(address.family) == "AddressFamily.AF_INET":
                logger.info(f"  IP Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == "AddressFamily.AF_PACKET":
                logger.info(f"  MAC Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    logger.info(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    logger.info(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")

def check_path_instance(obj: object) -> bool:
    """
    Checks if an objects belongs to pathlib.Path subclasses.

    Parameters
    ------------------------

    obj: object
        Object that wants to be validated.

    Returns
    ------------------------

    bool:
        True if the object is an instance of Path subclass, False otherwise.
    """

    for childclass in Path.__subclasses__():
        if isinstance(obj, childclass):
            return True

    return False

def save_dict_as_json(
    filename: str, dictionary: dict, verbose: Optional[bool] = False
) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------

    filename: str
        Name of the json file.

    dictionary: dict
        Dictionary that will be saved as json.

    verbose: Optional[bool]
        True if you want to print the path where the file was saved.

    """

    if dictionary is None:
        dictionary = {}

    else:
        for key, value in dictionary.items():
            # Converting path to str to dump dictionary into json
            if check_path_instance(value):
                # TODO fix the \\ encode problem in dump
                dictionary[key] = str(value)

    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)

    if verbose:
        print(f"- Json file saved: {filename}")
