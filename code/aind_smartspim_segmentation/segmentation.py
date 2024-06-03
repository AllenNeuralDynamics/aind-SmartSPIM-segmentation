"""
Created on Thu Dec  8 15:06:14 2022

@author: nicholas.lusk

Modified by camilo.laiton on Tue Jan 10 12:19:00 2022

Module for the segmentation of smartspim datasets
"""

import json
import logging
import multiprocessing
import os
from datetime import datetime
from glob import glob
from pathlib import Path

import dask
import dask.array as da
import numpy as np
from aind_data_schema.core.processing import DataProcess, ProcessName
from dask.distributed import Client, LocalCluster, performance_report
from imlib.IO.cells import get_cells, save_cells
from natsort import natsorted
from ng_link import NgState
from ng_link.ng_state import get_points_from_xml

from .__init__ import __version__
from ._shared.types import PathLike
from .utils import utils


def set_up_dask_config(tmp_folder: PathLike):
    """
    Sets up dask configuration

    Parameters
    ----------
    tmp_folder: PathLike
        Path to temporary folder

    """
    dask.config.set(
        {
            "temporary-directory": tmp_folder,
            "local_directory": tmp_folder,
            "tcp-timeout": "300s",
            "array.chunk-size": "384MiB",
            "distributed.comm.timeouts": {
                "connect": "300s",
                "tcp": "300s",
            },
            "distributed.scheduler.bandwidth": 100000000,
            "distributed.worker.memory.rebalance.measure": "optimistic",
            "distributed.worker.memory.target": 0.90,  # 0.85,
            "distributed.worker.memory.spill": 0.92,  # False,#
            "distributed.worker.memory.pause": 0.95,  # False,#
            "distributed.worker.memory.terminate": 0.98,  # False, #
        }
    )


def __read_zarr_image(image_path: PathLike):
    """
    Reads a zarr image

    Parameters
    -------------
    image_path: PathLike
        Path where the zarr image is located

    Returns
    -------------
    da.core.Array
        Dask array with the zarr image
    """

    image_path = str(image_path)
    signal_array = da.from_zarr(image_path)

    return signal_array


def calculate_offsets(blocks, chunk_size):
    """
    creates list of offsets for each chunk based on its location
    in the dask array

    Parameters
    ----------
    blocks : tuple
        The number of blocks in each direction (z, col, row)

    chunk_size : tuple
        The number of values along each dimention of a chunk (z, col, row)

    Return
    ------
    offests: list
        The offsets of each block in "C order
    """
    offsets = []
    for dv in range(blocks[0]):
        for ap in range(blocks[1]):
            for ml in range(blocks[2]):
                offsets.append(
                    [
                        chunk_size[2] * ml,
                        chunk_size[1] * ap,
                        chunk_size[0] * dv,
                    ]
                )
    return offsets

# Function missed during formatting check
def cell_detection(smartspim_config: dict, logger: logging.Logger):
    image_path = Path(smartspim_config["input_data"]).joinpath(
        f"{smartspim_config['input_channel']}/{smartspim_config['input_scale']}"
    )

    mask_path = Path(smartspim_config["input_data"]).joinpath(
        f"{smartspim_config['input_channel']}/{smartspim_config['mask_scale']}"
    )

    data_processes = []
    logger.info(f"Image to process: {image_path}")

    # load image data
    start_date_time = datetime.now()
    signal_array = __read_zarr_image(image_path)
    mask_array = __read_zarr_image(mask_path)
    end_date_time = datetime.now()

    # remove extra dimensions
    signal_array = signal_array[0, 0, :, :, :]
    mask_array = mask_array[0, 0, :, :, :]

    if smartspim_config["cellfinder_params"]["end_plane"] == -1:
        smartspim_config["cellfinder_params"]["end_plane"] = signal_array.shape[-3]

    logger.info(
        f"Starting detection with array {signal_array} with"
        f" start in {smartspim_config['cellfinder_params']['start_plane']}"
        f" and end in {smartspim_config['cellfinder_params']['end_plane']}"
    )

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_IMPORTING,
            software_version=__version__,
            start_date_time=start_date_time,
            end_date_time=end_date_time,
            input_location=str(image_path),
            output_location=str(image_path),
            outputs={},
            code_url="https://github.com/AllenNeuralDynamics/aind-SmartSPIM-segmentation",
            code_version=__version__,
            parameters={},
            notes="Importing fused data for cell segmentation",
        )
    )

    start_date_time = datetime.now()

    # get into roughly 500px chunks
    if smartspim_config["chunk_size"] % 64 == 0:
        chunk_step = 512
    elif smartspim_config["chunk_size"] == 1 or smartspim_config["chunk_size"] % 250 == 0:
        chunk_step = 500

    logger.info(
        f"z-plane chunk size: {smartspim_config['chunk_size']}. Processing with chunk size: {chunk_step}."
    )

    # get quality blocks using mask
    chunks = [int(np.ceil(x / chunk_step)) for x in signal_array.shape]
    good_blocks = utils.find_good_blocks(
        mask_array, chunks, chunk_step, smartspim_config["mask_scale"]
    )

    rechunk_size = [axis * (chunk_step // axis) for axis in signal_array.chunksize]
    signal_array = signal_array.rechunk(tuple(rechunk_size))
    logger.info(f"Rechunk dask array to {signal_array.chunksize}.")

    all_blocks = signal_array.to_delayed().ravel()
    all_offsets = calculate_offsets(signal_array.numblocks, signal_array.chunksize)

    blocks, offsets, counts = [], [], []

    for c, gb in good_blocks.items():
        if gb:
            blocks.append(all_blocks[c])
            offsets.append(all_offsets[c])
            counts.append(c)

    logger.info(f"There are {len(blocks)} delayed blocks to process.")

    logger.info(f"Running background subtraction and segmentation with array {signal_array}")

    # start client
    cluster = LocalCluster(
        n_workers=smartspim_config["workers"],
        processes=True,
        threads_per_worker=1,
    )

    logger.info(f"running with {smartspim_config['workers']} workers")

    client = Client(cluster)
    dask_report_file = f"{smartspim_config['metadata_path']}/dask_profile.html"
    with performance_report(filename=dask_report_file):
        count = 0
        offload = len(blocks) // 2

        # breacking up loop to avoid dask hanging
        loop_chunks = [
            (blocks[:offload], offsets[:offload], counts[:offload]),
            (blocks[offload:], offsets[offload:], counts[offload:]),
        ]

        for lc in loop_chunks:
            results = []
            for block, offset, count in zip(*lc):
                if smartspim_config["bkg_subtract"]:
                    bkg_sub = utils.delay_astro(
                        block,
                        pad=smartspim_config["padding"],
                        reflect=smartspim_config["cellfinder_params"]["soma_diameter"],
                    )
                else:
                    bkg_sub = utils.delay_preprocess(
                        img=block,
                        reflect=smartspim_config["cellfinder_params"]["soma_diameter"],
                        pad=smartspim_config["padding"],
                    )

                cell_count = utils.delay_detect(
                    bkg_sub,
                    smartspim_config["metadata_path"],
                    count,
                    offset,
                    smartspim_config["padding"]
                    + smartspim_config["cellfinder_params"]["soma_diameter"],
                    "block",
                    smartspim_config["cellfinder_params"],
                )

                results.append(da.from_delayed(cell_count[1], shape=(1,), dtype=int))

            arr = da.concatenate(results, axis=0, allow_unknown_chunksizes=True)
            _ = arr.compute()
            logger.info("Reseting client to try to avoid memory issues.")
            client.restart()

    end_date_time = datetime.now()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_CELL_SEGMENTATION,
            software_version=__version__,
            start_date_time=start_date_time,
            end_date_time=end_date_time,
            input_location=str(image_path),
            output_location=str(smartspim_config["metadata_path"]),
            outputs={},
            code_url="https://github.com/AllenNeuralDynamics/aind-SmartSPIM-segmentation",
            code_version=__version__,
            parameters={
                "chunk_step": chunk_step,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "smartspim_cell_config": smartspim_config,
            },
            notes=f"Segmenting channel in path: {image_path}",
        )
    )

    return str(image_path), data_processes


def merge(metadata_path: PathLike, save_path: PathLike, logger: logging.Logger):
    """
    Saves list of all cells
    """

    # load temporary files and save to a single list
    logger.info(f"Reading XMLS from cells path: {metadata_path}")
    cells = []
    tmp_files = glob(metadata_path + "/*.xml")

    for f in natsorted(tmp_files):
        try:
            cells.extend(get_cells(f))
        except:
            pass

    # save list of all cells
    save_cells(
        cells=cells,
        xml_file_path=os.path.join(save_path, "detected_cells.xml"),
    )


def generate_neuroglancer_link(
    image_path: str,
    dataset_name: str,
    channel_name: str,
    detected_cells_path: str,
    output: str,
    voxel_sizes: list,
    logger: logging.Logger,
):
    """
    Generates neuroglancer link with the cell location
    for a specific dataset

    Parameters
    -----------
    image_path: str
        Path to the zarr file

    dataset_name: str
        Dataset name where the data will be stored
        in the cloud. Follows SmartSPIM_***_stitched_***

    channel_name: str
        Channel name that was processed

    detected_cells_path: str
        Path to the detected cells

    output: str
        Output path of the neuroglancer
        config and precomputed format

    voxel_sizes: list
        list of um per voxel along each dimension
        ordered [z, y, x]
    """

    logger.info(f"Reading cells from {detected_cells_path}")
    cells = get_points_from_xml(detected_cells_path)

    output_precomputed = os.path.join(output, "visualization/precomputed")
    json_name = os.path.join(output, "visualization/neuroglancer_config.json")
    utils.create_folder(output_precomputed)
    print(f"Output cells precomputed: {output_precomputed}")

    logger.info(f"Image path in {image_path}")
    example_data = {
        "dimensions": {
            # check the order
            "z": {"voxel_size": voxel_sizes[0], "unit": "microns"},
            "y": {"voxel_size": voxel_sizes[1], "unit": "microns"},
            "x": {"voxel_size": voxel_sizes[2], "unit": "microns"},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": [
            {
                "source": image_path,
                "type": "image",
                "channel": 0,
                "shader": {"color": "gray", "emitter": "RGB", "vec": "vec3"},
                "shaderControls": {"normalized": {"range": [0, 500]}},  # Optional
            },
            {
                "type": "annotation",
                "source": f"precomputed://{output_precomputed}",
                "tool": "annotatePoint",
                "name": "annotation_name_layer",
                "annotations": cells,
            },
        ],
    }
    bucket_path = "aind-open-data"
    neuroglancer_link = NgState(
        input_config=example_data,
        base_url="https://aind-neuroglancer-sauujisjxq-uw.a.run.app",
        mount_service="s3",
        bucket_path=bucket_path,
        output_json=os.path.join(output, "visualization"),
        json_name=json_name,
    )

    json_state = neuroglancer_link.state
    json_state[
        "ng_link"
    ] = f"https://aind-neuroglancer-sauujisjxq-uw.a.run.app#!s3://{bucket_path}/{dataset_name}/image_cell_segmentation/{channel_name}/visualization/neuroglancer_config.json"

    json_state["layers"][0][
        "source"
    ] = f"zarr://s3://{bucket_path}/{dataset_name}/image_tile_fusing/OMEZarr/{channel_name}.zarr"

    json_state["layers"][1][
        "source"
    ] = f"precomputed://s3://{bucket_path}/{dataset_name}/image_cell_segmentation/{channel_name}/visualization/precomputed"

    logger.info(f"Visualization link: {json_state['ng_link']}")
    output_path = os.path.join(output, json_name)

    with open(output_path, "w") as outfile:
        json.dump(json_state, outfile, indent=2)


def main(
    data_folder: PathLike,
    output_segmented_folder: PathLike,
    intermediate_segmented_folder: PathLike,
    smartspim_config: dict,
):
    """
    This function detects cells

    Parameters
    -----------
    data_folder: PathLike
        Path where the image data is located

    output_segmented_folder: PathLike
        Path where the OMEZarr and metadata will
        live after fusion

    intermediate_segmented_folder: PathLike
        Path where the intermediate files
        will live. These will not be in the final
        folder structure. e.g., 3D fused chunks
        from TeraStitcher

    smartspim_config: dict
        Dictionary with the smartspim configuration
        for that dataset

    """

    set_up_dask_config(intermediate_segmented_folder)
    utils.create_folder(smartspim_config["metadata_path"])

    # Logger pointing everything to the metadata path
    logger = utils.create_logger(output_log_path=smartspim_config["metadata_path"])
    utils.print_system_information(logger)

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

    # run cell detection
    image_path, data_processes = cell_detection(smartspim_config=smartspim_config, logger=logger)

    # merge block .xmls into single file
    merge(smartspim_config["metadata_path"], smartspim_config["save_path"], logger)

    # Generating neuroglancer precomputed format
    detected_cells_path = os.path.join(smartspim_config["save_path"], "detected_cells.xml")
    image_path = os.path.abspath(
        f"{smartspim_config['input_data']}/{smartspim_config['input_channel']}"
    )

    # create neuroglancer link
    generate_neuroglancer_link(
        image_path,
        smartspim_config["name"],
        smartspim_config["channel"],
        detected_cells_path,
        smartspim_config["save_path"],
        smartspim_config["cellfinder_params"]["voxel_sizes"],
        logger,
    )

    utils.generate_processing(
        data_processes=data_processes,
        dest_processing=str(smartspim_config["metadata_path"]),
        processor_full_name="Nicholas Lusk",
        pipeline_version="1.5.0",
    )

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            smartspim_config["metadata_path"],
            "smartspim_detection",
        )


if __name__ == "__main__":
    main()
