"""
Scripts that runs the Code Ocean capsule
"""

import json
import logging
import os
import shutil
import logging
from glob import glob
from pathlib import Path
from typing import List, Tuple

from aind_smartspim_segmentation.detect import smartspim_cell_detection
from aind_smartspim_segmentation.utils import utils
from aind_smartspim_segmentation.utils import neuroglancer_utils as ng_utils
from aind_smartspim_segmentation._shared.types import ArrayLike, PathLike
import shutil

def get_data_config(
    data_folder: str,
    results_folder: str,
    processing_manifest_path: str = "processing_manifest*",
    data_description_path: str = "data_description.json",
) -> Tuple:
    """
    Returns the first smartspim dataset found
    in the data folder

    Parameters
    -----------
    data_folder: str
        Path to the folder that contains the data

    processing_manifest_path: str
        Path for the processing manifest

    data_description_path: str
        Path for the data description

    Returns
    -----------
    Tuple[Dict, str]
        Dict: Empty dictionary if the path does not exist,
        dictionary with the data otherwise.

        Str: Empty string if the processing manifest
        was not found
    """

    # Returning first smartspim dataset found
    # Doing this because of Code Ocean, ideally we would have
    # a single dataset in the pipeline
    print(
        glob(f"{data_folder}/{processing_manifest_path}"),
        f"{data_folder}/{processing_manifest_path}",
    )
    processing_data = glob(f"{data_folder}/{processing_manifest_path}")[0]

    derivatives_dict = utils.read_json_as_dict(processing_data)
    data_description_dict = utils.read_json_as_dict(f"{data_folder}/{data_description_path}")

    smartspim_dataset = data_description_dict["name"]

    # copy processing manifest to results folder
    fname = processing_data.split("/")[-1]
    shutil.copyfile(processing_data, f"{results_folder}/{fname}")

    print(f"processing manisfest copied to {results_folder}/{fname}")

    return derivatives_dict, smartspim_dataset

def get_yaml(yaml_path: PathLike):
    """
    Gets the default configuration from a YAML file

    Parameters
    --------------
    filename: str
        Path where the YAML is located

    Returns
    --------------
    dict
        Dictionary with the yaml configuration
    """

    config = None
    try:
        with open(yaml_path, "r") as stream:
            config = yaml.safe_load(stream)
    except Exception as error:
        raise error

    return config

def validate_capsule_inputs(input_elements: List[str]) -> List[str]:
    """
    Validates input elemts for a capsule in
    Code Ocean.

    Parameters
    -----------
    input_elements: List[str]
        Input elements for the capsule. This
        could be sets of files or folders.

    Returns
    -----------
    List[str]
        List of missing files
    """

    missing_inputs = []
    for required_input_element in input_elements:
        required_input_element = Path(required_input_element)

        if not required_input_element.exists():
            missing_inputs.append(str(required_input_element))

    return missing_inputs


def run():
    """
    Run function
    """

    # Absolute paths of common Code Ocean folders
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = []

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(f"We miss the following files in the capsule input: {missing_files}")

    pipeline_config, smartspim_dataset_name = get_data_config(
        data_folder=data_folder, results_folder=results_folder
    )

    segmentation_info = pipeline_config.get("segmentation")

    if segmentation_info is None:
        raise ValueError("Please, provide segmentation channels.")

    channel_to_process = segmentation_info.get("channel")

    # Note: The dispatcher capsule creates a single config with
    # the channels. If the channel key does not exist, it means
    # there are no segmentation channels splitted
    if channel_to_process is not None:

        # get default configs
        smartspim_config = get_yaml(
            os.path.abspath("aind_smartspim_segmentation/params/default_detect_config.yaml")
        )
        smartspim_config['axis_pad'] = int(
            1.6 * max(
                max(smartspim_config['spot_parameters']['sigma_zyx'][1:]),
                smartspim_config['spot_parameters']['sigma_zyx'][0]
            ) * 5
        )

        # add paths to smartspim_config
        smartspim_config["dataset_path"] = os.path.abspath(
            f"{pipeline_config['segmentation']['input_data']}/{channel_to_process}.zarr"
        )

        print("Files in path: ", os.listdir(smartspim_config["dataset_path"]))

        smartspim_config["output_folder"] = f"{results_folder}/cell_{channel_to_process}"
        smartspim_config["metadata_path"] = f"{results_folder}/cell_{channel_to_process}/metadata"
        
        utils.create_folder(dest_dir=str(smartspim_config["metadata_path"]), verbose=True)

        print("Initial cell detection config: ", smartspim_config)

        smartspim_config["name"] = smartspim_dataset_name

        print("Final cell segmentation config: ", smartspim_config)
        
        logger = utils.create_logger(output_log_path=str(smartspim_config["metadata_path"]))
        smartspim_config['logger'] = logger
        
        # run detection
        proposal_df = smartspim_cell_detection(**smartspim_config)
        
        # create nueroglancer link
        smartspim_config["channel"] = channel_to_process
        acquisition = utils.read_json_as_dict(f"{data_folder}/acquisition.json")

        dynamic_range = ng_utils.calculate_dynamic_range(smartspim_config["dataset_path"], 99, 3)
        res = {}
        for axis in pipeline_config['stitching']['resolution']:
            res[axis['axis_name']] = axis['resolution']
    
        ng_config = {
            "base_url": "https://neuroglancer-demo.appspot.com/#!",
            "crossSectionScale": 15,
            "projectionScale": 16384,
            "orientation": acquisition,
            "dimensions" : {
                "z": [res['Z'] * 10**-6, 'm' ],
                "y": [res['Y'] * 10**-6, 'm' ],
                "x": [res['X'] * 10**-6, 'm' ],
                "t": [0.001, 's'],
            },
            "rank": 3,
            "gpuMemoryLimit": 1500000000,
        }
        
        ng_utils.generate_neuroglancer_link(
            proposal_df,
            ng_config,
            smartspim_config,
            dynamic_range,
            logger
        )
        
    else:
        print(f"No segmentation channel, pipeline config: {pipeline_config}")
        utils.save_dict_as_json(
            filename=f"{RESULTS_FOLDER}/segmentation_processing_manifest_empty.json",
            dictionary=pipeline_config,
        )


if __name__ == "__main__":
    run()
