"""
Module to declare the parameters for the segmentation package
"""

import yaml
from argschema import ArgSchema, InputFile
from argschema.fields import Boolean, Int, List, Str

from .._shared.types import PathLike

class SegParams(ArgSchema):
    """
    Schema format for Segmentation
    """

    config_file = InputFile(
        required=True,
        metadata={"description": "Path to the YAML config file."},
        dump_default="default_segment_config.yaml",
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