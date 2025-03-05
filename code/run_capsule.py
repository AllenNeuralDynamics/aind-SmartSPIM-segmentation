"""
Scripts that runs the Code Ocean capsule
"""

import json
import logging
import os
import shutil
from glob import glob
from pathlib import Path
from typing import List, Tuple

from aind_smartspim_segmentation.detect import smartspim_cell_detection
from aind_smartspim_segmentation.utils import utils
from ng_link import NgState
from ng_link.ng_state import get_points_from_xml


def generate_neuroglancer_link(
    image_path: str,
    dataset_name: str,
    channel_name: str,
    detected_cells_path: str,
    output: str,
    voxel_sizes: list,
    logger: logging.Logger,
    neuroglancer_domain: str = "https://aind-neuroglancer-sauujisjxq-uw.a.run.app",
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
        base_url=neuroglancer_domain,
        mount_service="s3",
        bucket_path=bucket_path,
        output_json=os.path.join(output, "visualization"),
        json_name=json_name,
    )

    json_state = neuroglancer_link.state
    json_state["ng_link"] = (
        f"{neuroglancer_domain}#!s3://{bucket_path}/{dataset_name}/image_cell_segmentation/{channel_name}/visualization/neuroglancer_config.json"
    )

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

    # Code Ocean folders
    RESULTS_FOLDER = Path(os.path.abspath("../../../results"))
    # SCRATCH_FOLDER = Path(os.path.abspath("../scratch"))
    DATA_FOLDER = Path(os.path.abspath("../../../data"))

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = []

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(f"We miss the following files in the capsule input: {missing_files}")

    pipeline_config, smartspim_dataset_name = get_data_config(
        data_folder=DATA_FOLDER,
        results_folder=RESULTS_FOLDER,
    )

    input_channel = f"{pipeline_config['pipeline_processing']['segmentation']['channel']}.zarr"

    DATA_PATH = f"{DATA_FOLDER}/{input_channel}"
    SEGMENTATION_PATH = None

    # Output folder
    output_folder = RESULTS_FOLDER.joinpath(
        f"cell_{pipeline_config['pipeline_processing']['segmentation']['channel']}"
    )
    metadata_path = output_folder.joinpath("metadata")

    utils.create_folder(dest_dir=str(metadata_path), verbose=True)

    logger = utils.create_logger(output_log_path=str(metadata_path))

    # Puncta detection parameters

    sigma_zyx = [1.8, 1.0, 1.0]
    background_percentage = 25
    axis_pad = int(1.6 * max(max(sigma_zyx[1:]), sigma_zyx[0]) * 5)
    min_zyx = [3, 3, 3]
    filt_thresh = 20
    raw_thresh = 180
    context_radius = 3
    radius_confidence = 0.05

    # Data loader params
    puncta_params = {
        "dataset_path": DATA_PATH,
        "segmentation_mask_path": SEGMENTATION_PATH,
        "multiscale": "1",
        "prediction_chunksize": (128, 128, 128),
        "target_size_mb": 3048,
        "n_workers": 0,
        "batch_size": 1,
        "axis_pad": axis_pad,
        "output_folder": output_folder,
        "metadata_path": metadata_path,
        "logger": logger,
        "super_chunksize": None,
        "spot_parameters": {
            "sigma_zyx": sigma_zyx,
            "background_percentage": background_percentage,
            "min_zyx": min_zyx,
            "filt_thresh": filt_thresh,
            "raw_thresh": raw_thresh,
            "context_radius": context_radius,
            "radius_confidence": radius_confidence,
        },
    }

    logger.info(
        f"Dataset path: {puncta_params['dataset_path']} - Cell detection params: {puncta_params}"
    )

    smartspim_cell_detection(**puncta_params)

    # generate_neuroglancer_link(
    #     image_path,
    #     smartspim_config["name"],
    #     smartspim_config["channel"],
    #     detected_cells_path,
    #     smartspim_config["save_path"],
    #     smartspim_config["cellfinder_params"]["voxel_sizes"],
    #     logger,
    # )


if __name__ == "__main__":
    run()
