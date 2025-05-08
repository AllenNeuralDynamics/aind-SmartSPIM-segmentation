"""
Main file to execute the smartspim segmentation
in code ocean
"""

import os
import shutil
import logging
from glob import glob
from pathlib import Path
from typing import List, Tuple


from aind_smartspim_segmentation.detect import smartspim_cell_detection
from aind_smartspim_segmentation.params import get_yaml
from aind_smartspim_segmentation.utils import utils
from aind_smartspim_segmentation.utils import neuroglancer_utils as ng_utils
import shutil

def get_data_config(
    data_folder: str,
    results_folder: str,
    processing_manifest_path: str = "processing_manifest*", #"segmentation_processing_manifest*",
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

    #processing_data = glob(f"{data_folder}/{processing_manifest_path}")[0]
    processing_data = glob(f"/data/SmartSPIM_799056_2025-04-29_22-43-05/SPIM/derivatives/{processing_manifest_path}")[0]

    derivatives_dict = utils.read_json_as_dict(processing_data)
    #data_description_dict = utils.read_json_as_dict(f"{data_folder}/{data_description_path}")
    data_description_dict = utils.read_json_as_dict(f"/data/SmartSPIM_799056_2025-04-29_22-43-05_stitched_2025-05-06_08-09-20/{data_description_path}")

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
    #scratch_folder = os.path.abspath("../scratch")

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = []

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(f"We miss the following files in the capsule input: {missing_files}")

    pipeline_config, smartspim_dataset_name = get_data_config(
        data_folder=data_folder, results_folder=results_folder
    )

    #segmentation_info = pipeline_config.get("segmentation")
    segmentation_info = pipeline_config['pipeline_processing'].get("segmentation")

    if segmentation_info is None:
        raise ValueError("Please, provide segmentation channels.")

    #channel_to_process = segmentation_info.get("channel")
    channel_to_process = segmentation_info.get("channels")[0]

    # Note: The dispatcher capsule creates a single config with
    # the channels. If the channel key does not exist, it means
    # there are no segmentation channels splitted
    if channel_to_process is not None:

        # get default configs
        default_config = get_yaml(
            os.path.abspath("aind_smartspim_segmentation/params/default_detect_config.yaml")
        )
        default_config['axis_pad'] = int(
            1.6 * max(
                max(default_config['spot_parameters']['sigma_zyx'][1:]),
                default_config['spot_parameters']['sigma_zyx'][0]
            ) * 5
        )

        # add paths to default_config
        #default_config["dataset_path"] = os.path.abspath(
        #    f"{pipeline_config['segmentation']['input_data']}/{channel_to_process}"
        #)

        default_config["dataset_path"] = '/data/SmartSPIM_799056_2025-04-29_22-43-05_stitched_2025-05-06_08-09-20/image_tile_fusing/OMEZarr/Ex_488_Em_525.zarr'

        #print("Files in path: ", os.listdir(default_config["dataset_path"]))

        default_config["output_folder"] = f"{results_folder}/cell_{channel_to_process}"
        default_config["metadata_path"] = f"{results_folder}/cell_{channel_to_process}/metadata"
        
        utils.create_folder(dest_dir=str(default_config["metadata_path"]), verbose=True)

        print("Initial cell segmentation config: ", default_config)

        # combine configs
        #smartspim_config = set_up_pipeline_parameters(
        #    pipeline_config=pipeline_config, default_config=default_config
        #)

        smartspim_config = default_config
        smartspim_config["name"] = smartspim_dataset_name

        print("Final cell segmentation config: ", smartspim_config)
        #print("Final cell segmentation config: ", default_config)

        #segmentation.main(
        #    intermediate_segmented_folder=Path(scratch_folder),
        #    smartspim_config=smartspim_config,
        #)
        
        logger = utils.create_logger(output_log_path=str(default_config["metadata_path"]))
        smartspim_config['logger'] = logger
        
        # run detection
        proposal_df = smartspim_cell_detection(**default_config)
        
        # create nueroglancer link
        #acquisition = utils.read_json_as_dict(f"{data_folder}/acquisition.json")
        acquisition = utils.read_json_as_dict(f"/data/SmartSPIM_799056_2025-04-29_22-43-05/acquisition.json")
        dynamic_range = ng_utils.calculate_dynamic_range(default_config["dataset_path"], 99, 3)
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
            default_config,
            dynamic_range,
            logger
        )
        
    else:
        print(f"No segmentation channel, pipeline config: {pipeline_config}")
        utils.save_dict_as_json(
            filename=f"{results_folder}/segmentation_processing_manifest_empty.json",
            dictionary=pipeline_config,
        )

        # For post-processing pipeline
        post_process_seg = Path(data_folder).joinpath('image_cell_segmentation')

        if post_process_seg.exists():
            for cell_folder in post_process_seg.glob("cell*"):
                # Proposals added after version 3.0.1
                cell_folder_name = cell_folder.stem
                if cell_folder.joinpath('proposals').exists():
                    cell_folder = cell_folder.joinpath('proposals')

                output_path = results_folder / cell_folder_name
                shutil.copy(
                    str(cell_folder),
                    output_path
                )


if __name__ == "__main__":
    run()
