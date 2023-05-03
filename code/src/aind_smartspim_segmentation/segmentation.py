"""
Created on Thu Dec  8 15:06:14 2022

@author: nicholas.lusk

Modified by camilo.laiton on Tue Jan 10 12:19:00 2022

Module for the segmentation of smartspim datasets
"""
import logging
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional, Union

import dask.array as da
import yaml
from aind_data_schema.processing import DataProcess
from argschema import ArgSchema, ArgSchemaParser, InputFile
from argschema.fields import Boolean, Int, Str
from cellfinder_core.detect import detect
from imlib.IO.cells import get_cells, save_cells
from natsort import natsorted

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
        "voxel_sizes": [2, 1.8, 1.8],  # in microns
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
        dump_default=500,
    )

    bkg_subtract = Boolean(
        metadata={
            "required": True,
            "description": "Whether to run background subtraction",
        },
        dump_default=False,
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

    def run(self) -> str:
        """
        Runs SmartSPIM Segmentation

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
                notes="Importing stitched data for cell segmentation",
            )
        )

        # check if background sublations will be run
        if self.args["bkg_subtract"]:
            logger.info("Starting background substraction")
            start_date_time = datetime.now()
            signal_array = astro_preprocess(signal_array, "MMMBackground")
            end_date_time = datetime.now()

            data_processes.append(
                DataProcess(
                    name="Image background subtraction",  # Cell segmentation
                    version=__version__,
                    start_date_time=start_date_time,
                    end_date_time=end_date_time,
                    input_location=str(image_path),
                    output_location="In memory array",
                    code_url="https://github.com/astropy/astropy.github.com",
                    parameters=smartspim_config,
                    notes="Background subtraction",
                )
            )

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

        start_date_time = datetime.now()
        detect.main(
            signal_array=signal_array,
            save_path=self.args["metadata_path"],
            chunk_size=self.args["chunk_size"],
            **smartspim_config,
        )
        end_date_time = datetime.now()

        data_processes.append(
            DataProcess(
                name="Image cell segmentation",  # Cell segmentation
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

        # delete tmp folder
        # try:
        #     shutil.rmtree(self.args["metadata_path"])
        # except OSError as e:
        #     logger.error(
        #         f"Error removing temp file {self.args["metadata_path"]} : {e.strerror}"
        #     )


def main():
    """
    Main function
    """
    default_params = {
        "bkg_subtract": False,
        "save_path": "/results/",
        "metadata_path": "/results/metadata",
    }

    seg = Segment(default_params)
    image_path = seg.run()
    seg.merge()

    return image_path


if __name__ == "__main__":
    main()
