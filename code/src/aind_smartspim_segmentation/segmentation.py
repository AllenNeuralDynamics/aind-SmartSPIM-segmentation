"""
Created on Thu Dec  8 15:06:14 2022

@author: nicholas.lusk

Modified by camilo.laiton on Tue Jan 10 12:19:00 2022

Module for the segmentation of smartspim datasets
"""
import logging
import os
import shutil
from glob import glob
from pathlib import Path
from typing import Union

import dask.array as da
import s3fs
import yaml
from argschema import ArgSchema, ArgSchemaParser, InputFile
from argschema.fields import Boolean, Int, Str
from imlib.IO.cells import get_cells, save_cells
from natsort import natsorted

from .cellfinder_core.detect import detect

PathLike = Union[str, Path]

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    input_channel = Str(
        metadata={"required": True, "description": "Channel to segment"}
    )

    input_scale = Int(
        metadata={"required": True, "description": "Zarr scale to start with"}
    )

    chunk_size = Int(
        metadata={
            "required": True,
            "description": "Number of planes per chunk (needed to prevent memory crashes)",
        }
    )

    bkg_subtract = Boolean(
        metadata={
            "required": True,
            "description": "Whether to run background subtraction",
        }
    )

    signal_data = Str(
        metadata={
            "required": True,
            "description": "Location to save segmentation .xml file",
        }
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

    def run(self) -> None:
        """
        Runs SmartSPIM Segmentation
        """

        smartspim_config = get_smartspim_default_config(
            self.args["config_file"]
        )

        # create temporary folder for storing chunked data
        self.tmp_path = os.path.join(os.getcwd(), "tmp")
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)

        image_path = Path(self.args["input_data"]).joinpath(
            f"{self.args['input_channel']}/{self.args['input_scale']}"
        )

        # load signal data
        signal_array = self.__read_zarr_image(image_path)

        # check if background sublations will be run - ASK NICK
        # if self.args['bkg_subtract']:
        #     signal_array

        detect.main(
            signal_array=signal_array,
            save_path=self.tmp_path,
            chunk_size=self.args["chunk_size"],
            **smartspim_config,
        )

    def merge(self):
        """
        Saves list of all cells
        """

        # load temporary files and save to a single list
        cells = []
        tmp_files = glob(os.path.join(self.tmp_path, ".xml"))
        for f in natsorted(tmp_files):
            cells.extend(get_cells(f))

        # save list of all cells
        save_cells(os.path.join(self.args["save_path"], "detected_cells.xml"))

        # delete tmp folder
        try:
            shutil.rmtree(self.tmp_path)
        except OSError as e:
            print(
                "Error removing temp file %s : %s"
                % (self.tmp_path, e.strerror)
            )


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


def get_default_config(filename: str) -> dict:
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
        raise error

    return config


def main():
    """
    Main function to execute the segmentation
    """

    default_params = {
        "chunk_size": 500,
        "bkg_subtract": True,
        "save_dir": None,
    }

    seg = Segment(default_params)
    seg.run()
    seg.merge()


if __name__ == "__main__":
    main()
