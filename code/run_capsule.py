"""
Main file to execute the smartspim segmentation
in code ocean
"""

import os
import shutil
from glob import glob
from pathlib import Path
from typing import List, Tuple

from aind_smartspim_segmentation import segmentation
from aind_smartspim_segmentation.params import get_yaml
from aind_smartspim_segmentation.utils import utils


def get_data_config(
    data_folder: str,
    results_folder: str,
    processing_manifest_path: str = "segmentation_processing_manifest*",
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

    processing_data = glob(f"{data_folder}/{processing_manifest_path}")[0]
    derivatives_dict = utils.read_json_as_dict(processing_data)
    data_description_dict = utils.read_json_as_dict(f"{data_folder}/{data_description_path}")

    smartspim_dataset = data_description_dict["name"]

    # copy processing manifest to results folder
    fname = processing_data.split("/")[-1]
    shutil.copyfile(processing_data, f"{results_folder}/{fname}")

    print(f"processing manisfest copied to {results_folder}/{fname}")

    return derivatives_dict, smartspim_dataset


def set_up_pipeline_parameters(pipeline_config: dict, default_config: dict):
    """
    Sets up smartspim stitching parameters that come from the
    pipeline configuration

    Parameters
    -----------
    smartspim_dataset: str
        String with the smartspim dataset name

    pipeline_config: dict
        Dictionary that comes with the parameters
        for the pipeline described in the
        processing_manifest.json

    default_config: dict
        Dictionary that has all the default
        parameters to execute this capsule with
        smartspim data

    Returns
    -----------
    Dict
        Dictionary with the combined parameters
    """

    default_config["input_channel"] = f"{pipeline_config['segmentation']['channel']}.zarr"
    default_config["channel"] = pipeline_config["segmentation"]["channel"]
    default_config["input_scale"] = pipeline_config["segmentation"]["input_scale"]
    default_config["chunk_size"] = int(pipeline_config["segmentation"]["chunksize"])
    default_config["cellfinder_params"]["start_plane"] = int(
        pipeline_config["segmentation"]["signal_start"]
    )
    default_config["cellfinder_params"]["end_plane"] = int(
        pipeline_config["segmentation"]["signal_end"]
    )

    return default_config


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
    Main function to execute the smartspim segmentation
    in code ocean
    """

    # Absolute paths of common Code Ocean folders
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")
    scratch_folder = os.path.abspath("../scratch")

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = []

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(f"We miss the following files in the capsule input: {missing_files}")

    pipeline_config, smartspim_dataset_name = get_data_config(
        data_folder=data_folder, results_folder=results_folder
    )

    # get default configs
    default_config = get_yaml(
        os.path.abspath("aind_smartspim_segmentation/params/default_segment_config.yaml")
    )

    # add paths to default_config
    default_config["input_data"] = os.path.abspath(pipeline_config["segmentation"]["input_data"])
    print("Files in path: ", os.listdir(default_config["input_data"]))

    default_config["save_path"] = (
        f"{results_folder}/cell_{pipeline_config['segmentation']['channel']}"
    )
    default_config["metadata_path"] = (
        f"{results_folder}/cell_{pipeline_config['segmentation']['channel']}/metadata"
    )

    print("Initial cell segmentation config: ", default_config)

    # combine configs
    smartspim_config = set_up_pipeline_parameters(
        pipeline_config=pipeline_config, default_config=default_config
    )

    smartspim_config["name"] = smartspim_dataset_name

    print("Final cell segmentation config: ", smartspim_config)

    segmentation.main(
        data_folder=Path(data_folder),
        output_segmented_folder=Path(results_folder),
        intermediate_segmented_folder=Path(scratch_folder),
        smartspim_config=smartspim_config,
    )


if __name__ == "__main__":
    run()
