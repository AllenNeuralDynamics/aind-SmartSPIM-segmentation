"""
Created on Thu Dec  8 15:06:14 2022

@author: nicholas.lusk

Modified by camilo.laiton on Tue Jan 10 12:19:00 2022

Module for the segmentation of smartspim datasets
"""
import json
import logging
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Union

import dask
import dask.array as da
import numpy as np
import yaml
from aind_data_schema.processing import DataProcess
from argschema import ArgSchema, ArgSchemaParser, InputFile
from argschema.fields import Boolean, Int, List, Str
from cellfinder_core.detect import detect
from dask.distributed import Client, LocalCluster, performance_report
from imlib.IO.cells import get_cells, save_cells
from natsort import natsorted
from ng_link import NgState
from ng_link.ng_state import get_points_from_xml

from .__init__ import __version__
from .utils import astro_preprocess, create_folder, generate_processing

PathLike = Union[str, Path]

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_smartspim_default_config() -> dict:
    """
    Effective parameters found to work with SmartSPIM nuclei signals

    Returns
    ------------
    dict
        Parameters to work with SmartSPIM nuclei signals
    """
    return {
        "start_plane": 0,
        "end_plane": 0,
        "n_free_cpus": 2,
        "voxel_sizes": [4, 1.8, 1.8],  # in microns
        "soma_diameter": 9,  # in microns
        "ball_xy_size": 8,
        "ball_overlap_fraction": 0.6,
        "log_sigma_size": 0.1,
        "n_sds_above_mean_thresh": 3,
        "soma_spread_factor": 1.4,
        "max_cluster_size": 100000,
    }


def get_yaml_config(filename: str) -> dict:
    """
    Get default configuration from a YAML file.
    Parameters
    ------------------------
    filename: str
        String where the YAML file is located.
    Returns
    ------------------------
    Dict
        Dictionary with the configuration
    """

    filename = Path(os.path.dirname(__file__)).joinpath(filename)

    config = None
    try:
        with open(filename, "r") as stream:
            config = yaml.safe_load(stream)
    except Exception as error:
        logger.error(error)

    return config


class SegSchema(ArgSchema):
    """
    Schema format for Segmentation
    """

    config_file = InputFile(
        required=True,
        metadata={"description": "Path to the YAML config file."},
        dump_default="smartspim_config.yaml",
    )

    input_data = Str(
        metadata={
            "required": True,
            "description": "Dataset path where the OMEZarr is located",
        }
    )

    input_channel = Str(metadata={"required": True, "description": "Channel to segment"})

    input_scale = Int(metadata={"required": True, "description": "Zarr scale to start with"})

    chunk_size = Int(
        metadata={
            "required": True,
            "description": """
            Number of planes per chunk
            (needed to prevent memory crashes)
            """,
        },
        dump_default=250,
    )

    bkg_subtract = Boolean(
        metadata={
            "required": True,
            "description": "Whether to run background subtraction",
        },
        dump_default=False,
    )

    subsample = List(
        Int(),
        metadata={
            "required": True,
            "description": "Whether to downsample along a particular dimention",
        },
        cli_as_single_argument=True,
        dump_default=[1, 1, 1],
    )

    save_path = Str(
        metadata={
            "required": True,
            "description": "Location to save segmentation .xml file",
        }
    )

    metadata_path = Str(
        metadata={
            "required": True,
            "description": "Location to save metadata files",
        }
    )

    signal_start = Int(
        metadata={
            "required": True,
            "description": "Z index (slice) where we want to start running segmentation on",
        },
        dump_default=0,
    )

    signal_end = Int(
        metadata={
            "required": True,
            "description": "Z index (slice) where we want to end running segmentation on",
        },
        dump_default=-1,
    )

    bucket_path = Str(
        required=True,
        metadata={"description": "Amazon Bucket or Google Bucket name"},
    )


def set_up_dask_config(tmp_folder: PathLike):
    """
    Sets up dask configuration

    Parameters
    ----------
    tmp_folder: PathLike
        Path to temporary folder

    """
    dask.config.set(
        {"temporary_directory": tmp_folder},
        {
            "tick": {
                "interval": "20ms",
                "limit": "30s",
                "cycle": "1s",
            },
        },
    )


class Segment(ArgSchemaParser):

    """
    Class for segmenting lightsheet data
    """

    default_schema = SegSchema

    def __read_zarr_image(self, image_path: PathLike) -> da.core.Array:
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

    def run(self, results_path_co: PathLike) -> str:
        """
        Runs SmartSPIM Segmentation

        Parameters
        ----------
        results_path_co: PathLike
            Results path in code ocean

        Returns
        ----------
        str
            Image path that was used for cell segmentation
        """

        smartspim_config = get_yaml_config(self.args["config_file"])

        if smartspim_config is None:
            smartspim_config = get_smartspim_default_config()
            logger.info(
                f"""
                Error while reading YAML.
                Using default config {smartspim_config}
                """
            )

        # create metadata folder
        create_folder(self.args["metadata_path"])

        image_path = Path(self.args["input_data"]).joinpath(
            f"{self.args['input_channel']}/{self.args['input_scale']}"
        )

        if not os.path.isdir(str(image_path)):
            root_path = Path(self.args["input_data"])
            channels = [folder for folder in os.listdir(root_path) if folder != ".zgroup"]

            selected_channel = channels[0]

            logger.info(
                f"""Directory {image_path} does not exist!
                Setting segmentation to the first
                available channel: {selected_channel}"""
            )
            image_path = root_path.joinpath(f"{selected_channel}/{self.args['input_scale']}")

        data_processes = []

        # load signal data
        start_date_time = datetime.now()
        signal_array = self.__read_zarr_image(image_path)
        end_date_time = datetime.now()

        # Loading only 3D data
        signal_start = self.args["signal_start"]
        signal_end = self.args["signal_end"]
        if signal_end == -1:
            signal_end = signal_array.shape[2]

        signal_array = signal_array[0, 0, :, :, :]
        logger.info(
            f"Starting detection with array {signal_array} with start in {signal_start} and end in {signal_end}"
        )

        # Setting up configuration
        smartspim_config["start_plane"] = signal_start
        smartspim_config["end_plane"] = signal_end

        data_processes.append(
            DataProcess(
                name="Image importing",
                version=__version__,
                start_date_time=start_date_time,
                end_date_time=end_date_time,
                input_location=str(image_path),
                output_location=str(image_path),
                code_url="https://github.com/AllenNeuralDynamics/aind-SmartSPIM-segmentation",
                parameters={},
                notes="Importing fused data for cell segmentation",
            )
        )

        # setup step range for segmentation based on zarr chunking
        steps_z = np.append(
            np.arange(
                0,
                signal_array.shape[0],
                self.args["chunk_size"],
            ),
            signal_array.shape[0],
        )

        holdover = {}

        # check if background subtraction will be run
        if self.args["bkg_subtract"]:
            logger.info(f"Running background subtraction and segmentation with array {signal_array}")

            # start client
            cluster = LocalCluster(
                n_workers=16,
                processes=True,
                threads_per_worker=1,
            )

            client = Client(cluster)

            for z in range(len(steps_z) - 1):
                dask_report_file = f"{results_path_co}/dask_profile_loop_{z}.html"

                with performance_report(filename=dask_report_file):
                    bkg_start_date_time = datetime.now()

                    bkg_array = da.map_blocks(
                        astro_preprocess,
                        signal_array[steps_z[z] : steps_z[z + 1], :, :],
                        "MMMBackground",
                        dtype=signal_array.dtype,
                        chunks=(
                            signal_array.chunks[0][z],
                            signal_array.chunks[1],
                            signal_array.chunks[2],
                        ),
                    ).compute()

                    bkg_array = np.array(bkg_array)
                    bkg_array = bkg_array[
                        :: self.args["subsample"][0],
                        :: self.args["subsample"][1],
                        :: self.args["subsample"][2],
                    ]

                    bkg_end_date_time = datetime.now()

                    holdover = detect.main(
                        signal_array=bkg_array,
                        save_path=self.args["metadata_path"],
                        chunk_size=self.args["chunk_size"],
                        block=z,
                        holdover=holdover,
                        **smartspim_config,
                    )

                    cell_end_date_time = datetime.now()

                    del bkg_array

            logger.info("Background subtraction completed. Closing dask client.")
            client.close()

            data_processes.append(
                DataProcess(
                    name="Image background subtraction",
                    version=__version__,
                    start_date_time=bkg_start_date_time,
                    end_date_time=bkg_end_date_time,
                    input_location=str(image_path),
                    output_location="In memory array",
                    code_url="https://github.com/astropy/astropy.github.com",
                    parameters=smartspim_config,
                    notes="Background subtraction",
                )
            )

            data_processes.append(
                DataProcess(
                    name="Image cell segmentation",
                    version=__version__,
                    start_date_time=bkg_end_date_time,
                    end_date_time=cell_end_date_time,
                    input_location=str(image_path),
                    output_location="In memory array",
                    code_url="https://github.com/camilolaiton/cellfinder-core/tree/feature/bkg_sub",
                    parameters=smartspim_config,
                    notes="Calculated using background subtraction",
                )
            )
        else:
            logger.info(f"Running segmentation with array {signal_array}")
            start_date_time = datetime.now()

            for z in range(len(steps_z) - 1):
                bkg_array = np.array(signal_array[steps_z[z] : steps_z[z + 1], :, :])
                bkg_array = bkg_array[
                    :: self.args["subsample"][0],
                    :: self.args["subsample"][1],
                    :: self.args["subsample"][2],
                ]

                holdover = detect.main(
                    signal_array=bkg_array,
                    save_path=self.args["metadata_path"],
                    chunk_size=self.args["chunk_size"],
                    block=z,
                    holdover=holdover,
                    **smartspim_config,
                )

                del bkg_array

            end_date_time = datetime.now()

            data_processes.append(
                DataProcess(
                    name="Image cell segmentation",
                    version=__version__,
                    start_date_time=start_date_time,
                    end_date_time=end_date_time,
                    input_location=str(image_path),
                    output_location=str(self.args["metadata_path"]),
                    code_url="https://github.com/camilolaiton/cellfinder-core/tree/feature/lazy_reader",
                    parameters=smartspim_config,
                    notes="Cell segmentation with XML outputs",
                )
            )

        processing_path = Path(self.args["metadata_path"]).joinpath("processing.json")

        generate_processing(
            data_processes=data_processes,
            dest_processing=processing_path,
            pipeline_version=__version__,
        )

        return str(image_path)

    def merge(self):
        """
        Saves list of all cells
        """

        # load temporary files and save to a single list
        logger.info(f"Reading XMLS from cells path: {self.args['metadata_path']}")
        cells = []
        tmp_files = glob(self.args["metadata_path"] + "/*.xml")

        for f in natsorted(tmp_files):
            cells.extend(get_cells(f))

        # save list of all cells
        save_cells(
            cells=cells,
            xml_file_path=os.path.join(self.args["save_path"], "detected_cells.xml"),
        )


def generate_neuroglancer_link(image_path: str, detected_cells_path: str, output: str):
    """
    Generates neuroglancer link with the cell location
    for a specific dataset

    Parameters
    -----------
    image_path: str
        Path to the zarr file

    detected_cells_path: str
        Path to the detected cells

    output: str
        Output path of the neuroglancer
        config and precomputed format

    """

    logger.info(f"Reading cells from {detected_cells_path}")
    cells = get_points_from_xml(detected_cells_path)
    smartspim_config_path = os.path.abspath(
        "/code/src/aind_smartspim_segmentation/smartspim_config.yml"
    )
    smartspim_config = get_yaml_config(smartspim_config_path)

    # Getting path
    dataset_name = []
    include = False
    # Excluding multiscale
    for folder in image_path.split("/")[:-1]:
        if "SmartSPIM" in folder:
            include = True

        if include:
            dataset_name.append(folder)

    image_path = "/".join(dataset_name)

    output_precomputed = os.path.join(output, "visualization/precomputed")
    json_name = os.path.join(output, "visualization/neuroglancer_config.json")
    create_folder(output_precomputed)

    if smartspim_config is None:
        smartspim_config = get_smartspim_default_config()
        logger.info(
            f"""
            Error while reading YAML.
            Using default config {smartspim_config}
            """
        )
    else:
        logger.info(f"Image path in {image_path}")
        example_data = {
            "dimensions": {
                # check the order
                "z": {"voxel_size": smartspim_config["voxel_sizes"][0], "unit": "microns"},
                "y": {"voxel_size": smartspim_config["voxel_sizes"][1], "unit": "microns"},
                "x": {"voxel_size": smartspim_config["voxel_sizes"][2], "unit": "microns"},
                "t": {"voxel_size": 0.001, "unit": "seconds"},
            },
            "layers": [
                {
                    "source": image_path,
                    "type": "image",
                    "channel": 0,
                    # 'name': 'image_name_0',
                    "shader": {"color": "gray", "emitter": "RGB", "vec": "vec3"},
                    "shaderControls": {"normalized": {"range": [0, 500]}},  # Optional
                },
                {
                    "type": "annotation",
                    "source": f"precomputed://{output_precomputed}",
                    "tool": "annotatePoint",
                    "name": "annotation_name_layer",
                    "annotations": cells,
                    # Pass None or delete limits if
                    # you want to include all the points
                    # "limits": [100, 200],  # None # erase line
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
        channel_name = dataset_name[4].replace(".zarr", "")
        json_state[
            "ng_link"
        ] = f"https://aind-neuroglancer-sauujisjxq-uw.a.run.app#!s3://{bucket_path}/{dataset_name[0]}/processed/Cell_Segmentation/{channel_name}/visualization/neuroglancer_config.json"

        json_state["layers"][1][
            "source"
        ] = f"precomputed://s3://{bucket_path}/{dataset_name[0]}/processed/Cell_Segmentation/{channel_name}/visualization/precomputed"

        logger.info(f"Visualization link: {json_state['ng_link']}")
        output_path = os.path.join(output, json_name)

        with open(output_path, "w") as outfile:
            json.dump(json_state, outfile, indent=2)


def main():
    """
    Main function
    """
    results_path = os.path.abspath("../results/")
    default_params = {
        "bkg_subtract": True,
        "subsample": [1, 1, 1],
        "save_path": results_path,
        "metadata_path": f"{results_path}/metadata",
    }

    set_up_dask_config(os.path.abspath("../scratch/"))

    seg = Segment(default_params)
    image_path = seg.run(results_path)
    seg.merge()

    # Generating neuroglancer precomputed format
    detected_cells_path = os.path.join(default_params["save_path"], "detected_cells.xml")
    generate_neuroglancer_link(image_path, detected_cells_path, results_path)

    return image_path


if __name__ == "__main__":
    main()
